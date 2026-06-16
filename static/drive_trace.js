/**
 * drive_trace.js — Drive Trace Mode for TurboTIFFLAS workspace
 *
 * Architecture:
 *  - Reads imageWidth/imageHeight, lastDigitizedDepth, lastDepthConfig,
 *    lastDigitizedCurves, lastCurveTraces from the parent workspace globals.
 *  - Uses getTrackCalibrationsSnapshot(), pixelToTrackValue(),
 *    renderCurveTraceOverlays() from the workspace.
 *  - Renders a zoomed "road" view (800% zoom centred on the marker) on a
 *    private <canvas id="dtRoadCanvas"> inside the modal.
 *  - Steering: ArrowLeft/Right, A/D keys OR mouse/touch horizontal position.
 *  - Optional snap-assist: pulls marker toward the darkest column in a local
 *    neighbourhood of the original image pixels (sampled via an off-screen
 *    canvas).
 *  - On Commit: injects the recorded [xImg, yImg, depthIndex] points into
 *    lastCurveTraces[curveKey] and calls renderCurveTraceOverlays().
 */

(function () {
    'use strict';

    // ── State ────────────────────────────────────────────────────────────────
    const DT = {
        active: false,
        paused: false,
        running: false,

        // Selected curve
        curveKey: null,
        track: null,

        // Car position in *image* pixels
        carX: 0,
        carY: 0,

        // Steering (image pixels/frame, accumulated from keyboard + mouse)
        steerDeltaX: 0,

        // Recorded path: array of [xImg, yImg, depthIndex]
        path: [],

        // Animation
        rafId: null,
        lastTs: null,

        // Keyboard state
        keys: {},

        // Mouse steering: last canvas clientX
        mouseClientX: null,
        canvasCenterX: null,

        // Off-screen canvas holding the original image for assist sampling
        srcCanvas: null,
        srcCtx: null,

        // Road view dimensions (canvas logical pixels)
        ROAD_W: 520,
        ROAD_H: 480,
        ZOOM: 8,           // 800 %
        ROAD_STRIP_H: 3,  // image rows per rendered road row
        placementMode: false,  // true = waiting for user click to place car start
    };

    // ── Helpers ──────────────────────────────────────────────────────────────

    function dtGetSpeed() {
        const el = document.getElementById('dtSpeedSlider');
        return el ? Number(el.value) : 80;  // image px / second
    }

    function dtGetAssist() {
        const el = document.getElementById('dtAssistSlider');
        return el ? Number(el.value) / 100 : 0.3;
    }

    window.dtUpdateSpeedLabel = function () {
        const el = document.getElementById('dtSpeedSlider');
        const lbl = document.getElementById('dtSpeedLabel');
        if (el && lbl) lbl.textContent = el.value + ' px/s';
    };

    window.dtUpdateAssistLabel = function () {
        const el = document.getElementById('dtAssistSlider');
        const lbl = document.getElementById('dtAssistLabel');
        if (el && lbl) lbl.textContent = el.value + '%';
    };

    function dtSetStatus(msg) {
        const el = document.getElementById('dtStatusText');
        if (el) el.textContent = msg;
    }

    function dtUpdateReadout() {
        const rowEl = document.getElementById('dtRowLabel');
        const xEl = document.getElementById('dtXLabel');
        const depEl = document.getElementById('dtDepthLabel');
        const depthArr = (typeof lastDigitizedDepth !== 'undefined') ? lastDigitizedDepth : null;
        const row = Math.round(DT.carY);
        if (rowEl) rowEl.textContent = String(row);
        if (xEl) xEl.textContent = String(Math.round(DT.carX));
        if (depEl && depthArr) {
            // Nearest depth sample
            const cfg = (typeof lastDepthConfig !== 'undefined') ? lastDepthConfig : null;
            if (cfg) {
                const topPx = cfg.topPx || 0;
                const botPx = cfg.bottomPx || 1;
                const topD = cfg.topDepth;
                const botD = cfg.bottomDepth;
                const frac = (DT.carY - topPx) / Math.max(1, botPx - topPx);
                const depth = topD + frac * (botD - topD);
                depEl.textContent = depth.toFixed(1);
            }
        } else if (depEl) {
            depEl.textContent = '—';
        }
    }

    // ── Off-screen source canvas for assist ─────────────────────────────────

    function dtBuildSrcCanvas() {
        const img = document.getElementById('imagePreview');
        if (!img || !img.naturalWidth) return;
        const iw = img.naturalWidth;
        const ih = img.naturalHeight;
        const c = document.createElement('canvas');
        c.width = iw; c.height = ih;
        const ctx = c.getContext('2d');
        ctx.drawImage(img, 0, 0, iw, ih);
        DT.srcCanvas = c;
        DT.srcCtx = ctx;
    }

    /** Return the darkest-column x within [x-radius, x+radius] at image row y */
    function dtFindDarkColumnNear(xImg, yImg, radius) {
        if (!DT.srcCtx) return xImg;
        const iw = DT.srcCanvas.width;
        const ih = DT.srcCanvas.height;
        const x0 = Math.max(0, Math.round(xImg - radius));
        const x1 = Math.min(iw - 1, Math.round(xImg + radius));
        const y0 = Math.max(0, Math.round(yImg - 1));
        const y1 = Math.min(ih - 1, Math.round(yImg + 1));
        const w = x1 - x0 + 1;
        const h = y1 - y0 + 1;
        if (w <= 0 || h <= 0) return xImg;
        const data = DT.srcCtx.getImageData(x0, y0, w, h).data;
        let bestX = xImg;
        let bestDark = 999;
        for (let col = 0; col < w; col++) {
            let sum = 0;
            for (let row = 0; row < h; row++) {
                const i = (row * w + col) * 4;
                sum += (data[i] + data[i + 1] + data[i + 2]) / 3;
            }
            const avg = sum / h;
            if (avg < bestDark) { bestDark = avg; bestX = x0 + col; }
        }
        return bestX;
    }

    // ── Road canvas renderer ─────────────────────────────────────────────────

    function dtDrawRoad() {
        const canvas = document.getElementById('dtRoadCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const W = DT.ROAD_W;
        const H = DT.ROAD_H;

        ctx.clearRect(0, 0, W, H);

        // Dark tarmac background
        ctx.fillStyle = '#0a1018';
        ctx.fillRect(0, 0, W, H);

        if (!DT.srcCtx || !DT.active) return;

        const iw = DT.srcCanvas.width;
        const ih = DT.srcCanvas.height;

        // Determine the source window in image coords
        // We show ROAD_W/ZOOM px wide, centred on carX
        const srcW = W / DT.ZOOM;
        const srcH = H / DT.ZOOM;
        const srcX0 = Math.round(DT.carX - srcW / 2);
        const srcY0 = Math.round(DT.carY - srcH * 0.4); // car sits 40% from top

        // Draw the image strip, clamping OOB regions to a fill
        if (srcX0 >= 0 && srcY0 >= 0 &&
            srcX0 + srcW <= iw && srcY0 + srcH <= ih) {
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(DT.srcCanvas,
                srcX0, srcY0, srcW, srcH,
                0, 0, W, H);
        } else {
            // Partial — fill missing edges then draw what we can
            ctx.fillStyle = '#1e293b';
            ctx.fillRect(0, 0, W, H);
            const cx0 = Math.max(0, srcX0);
            const cy0 = Math.max(0, srcY0);
            const cx1 = Math.min(iw, srcX0 + srcW);
            const cy1 = Math.min(ih, srcY0 + srcH);
            if (cx1 > cx0 && cy1 > cy0) {
                const dx = (cx0 - srcX0) * DT.ZOOM;
                const dy = (cy0 - srcY0) * DT.ZOOM;
                const dw = (cx1 - cx0) * DT.ZOOM;
                const dh = (cy1 - cy0) * DT.ZOOM;
                ctx.imageSmoothingEnabled = false;
                ctx.drawImage(DT.srcCanvas, cx0, cy0, cx1 - cx0, cy1 - cy0,
                    dx, dy, dw, dh);
            }
        }

        // Track boundary lines (if track defined)
        if (DT.track) {
            const lx = (DT.track.leftX - srcX0) * DT.ZOOM;
            const rx = (DT.track.rightX - srcX0) * DT.ZOOM;
            ctx.save();
            ctx.strokeStyle = 'rgba(99,102,241,0.7)';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            if (lx >= 0 && lx <= W) { ctx.beginPath(); ctx.moveTo(lx, 0); ctx.lineTo(lx, H); ctx.stroke(); }
            if (rx >= 0 && rx <= W) { ctx.beginPath(); ctx.moveTo(rx, 0); ctx.lineTo(rx, H); ctx.stroke(); }
            ctx.restore();
        }

        // Recorded path (recent portion)
        if (DT.path.length > 1) {
            ctx.save();
            ctx.strokeStyle = 'rgba(163,230,53,0.85)';
            ctx.lineWidth = 2;
            ctx.lineJoin = 'round';
            ctx.beginPath();
            let started = false;
            for (let i = Math.max(0, DT.path.length - 120); i < DT.path.length; i++) {
                const [px, py] = DT.path[i];
                const cx2 = (px - srcX0) * DT.ZOOM;
                const cy2 = (py - srcY0) * DT.ZOOM;
                if (!started) { ctx.moveTo(cx2, cy2); started = true; }
                else ctx.lineTo(cx2, cy2);
            }
            ctx.stroke();
            ctx.restore();
        }

        // Car marker (triangle + glow)
        const carCanvasX = W / 2;
        const carCanvasY = H * 0.4;
        ctx.save();
        ctx.shadowColor = '#f59e0b';
        ctx.shadowBlur = 18;
        ctx.fillStyle = '#fbbf24';
        ctx.beginPath();
        ctx.moveTo(carCanvasX, carCanvasY - 14);
        ctx.lineTo(carCanvasX - 9, carCanvasY + 10);
        ctx.lineTo(carCanvasX + 9, carCanvasY + 10);
        ctx.closePath();
        ctx.fill();
        ctx.shadowBlur = 0;
        // Outline
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();

        // Scanline progress bar at bottom
        const progFrac = Math.max(0, Math.min(1,
            (DT.carY - (DT.startY || 0)) / Math.max(1, (DT.endY || 1) - (DT.startY || 0))));
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, H - 6, W, 6);
        ctx.fillStyle = '#f59e0b';
        ctx.fillRect(0, H - 6, Math.round(W * progFrac), 6);
    }

    // ── Animation loop ───────────────────────────────────────────────────────

    function dtLoop(ts) {
        if (!DT.running) return;
        if (DT.paused) { DT.rafId = requestAnimationFrame(dtLoop); return; }

        if (!DT.lastTs) DT.lastTs = ts;
        const dt = Math.min((ts - DT.lastTs) / 1000, 0.1); // cap at 100ms
        DT.lastTs = ts;

        const speed = dtGetSpeed(); // image px / second
        const assist = dtGetAssist();

        // ── Advance Y ──
        DT.carY += speed * dt;

        // Clamp to image bounds
        const ih = DT.srcCanvas ? DT.srcCanvas.height : (typeof imageHeight !== 'undefined' ? imageHeight : 9999);
        if (DT.carY >= DT.endY || DT.carY >= ih) {
            DT.carY = Math.min(DT.endY, ih - 1);
            dtFinish();
            return;
        }

        // ── Steer X from keys ──
        const keySpeed = 120; // image px / s
        if (DT.keys['ArrowLeft'] || DT.keys['a'] || DT.keys['A']) DT.steerDeltaX -= keySpeed * dt;
        if (DT.keys['ArrowRight'] || DT.keys['d'] || DT.keys['D']) DT.steerDeltaX += keySpeed * dt;

        // Mouse steering: deviation of mouse from canvas center → steer
        if (DT.mouseClientX !== null && DT.canvasCenterX !== null) {
            const dev = DT.mouseClientX - DT.canvasCenterX; // dom px
            const iw2 = typeof imageWidth !== 'undefined' ? imageWidth : 1000;
            const canvas = document.getElementById('dtRoadCanvas');
            const cw = canvas ? canvas.offsetWidth : DT.ROAD_W;
            // map dom deviation to image pixels
            const imgDev = (dev / (cw / 2)) * (DT.ROAD_W / DT.ZOOM / 2);
            DT.steerDeltaX += imgDev * dt * 2.5;
        }

        // Apply accumulated steer
        DT.carX += DT.steerDeltaX;
        DT.steerDeltaX = 0;

        // ── Snap assist ──
        if (assist > 0 && DT.srcCtx) {
            const radius = 30; // image px search radius
            const darkX = dtFindDarkColumnNear(DT.carX, DT.carY, radius);
            DT.carX += (darkX - DT.carX) * assist * dt * 4;
        }

        // Clamp X to track bounds
        if (DT.track) {
            const lo = Math.min(DT.track.leftX, DT.track.rightX);
            const hi = Math.max(DT.track.leftX, DT.track.rightX);
            DT.carX = Math.max(lo, Math.min(hi, DT.carX));
        }

        // ── Record path point ──
        // Record at ~every 1 image row so we don't over-sample
        const last = DT.path.length ? DT.path[DT.path.length - 1] : null;
        if (!last || Math.abs(DT.carY - last[1]) >= 0.8) {
            // Compute depth index
            let dIdx = -1;
            const depthArr = (typeof lastDigitizedDepth !== 'undefined') ? lastDigitizedDepth : null;
            if (depthArr && depthArr.length > 0) {
                const cfg = (typeof lastDepthConfig !== 'undefined') ? lastDepthConfig : null;
                if (cfg) {
                    const topPx = cfg.topPx || 0;
                    const botPx = cfg.bottomPx || depthArr.length;
                    const frac = (DT.carY - topPx) / Math.max(1, botPx - topPx);
                    dIdx = Math.round(frac * (depthArr.length - 1));
                    dIdx = Math.max(0, Math.min(depthArr.length - 1, dIdx));
                }
            }
            DT.path.push([DT.carX, DT.carY, dIdx]);
        }

        dtUpdateReadout();
        dtDrawRoad();

        DT.rafId = requestAnimationFrame(dtLoop);
    }

    // ── API ──────────────────────────────────────────────────────────────────

    window.openDriveTraceMode = function () {
        const modal = document.getElementById('driveTraceModal');
        if (!modal) return;

        // Populate curve dropdown
        const sel = document.getElementById('dtCurveSelect');
        if (sel) {
            const srcSel = document.getElementById('editCurveSelect');
            sel.innerHTML = '<option value="">— select curve —</option>';
            if (srcSel) {
                Array.from(srcSel.options).forEach(o => {
                    if (!o.value) return;
                    const opt = document.createElement('option');
                    opt.value = o.value;
                    opt.textContent = o.textContent;
                    if (o.selected) opt.selected = true;
                    sel.appendChild(opt);
                });
            }
            if (typeof lastCurveTraces === 'object' && lastCurveTraces) {
                Object.keys(lastCurveTraces).forEach(k => {
                    if (!Array.from(sel.options).find(o => o.value === k)) {
                        const opt = document.createElement('option');
                        opt.value = k; opt.textContent = k;
                        sel.appendChild(opt);
                    }
                });
            }
        }

        modal.style.display = 'flex';
        DT.active = false;
        DT.running = false;
        DT.paused = false;
        DT.placementMode = false;
        DT.path = [];

        const canvas = document.getElementById('dtRoadCanvas');
        if (canvas) {
            canvas.width = DT.ROAD_W;
            canvas.height = DT.ROAD_H;
        }

        document.getElementById('dtHintOverlay').style.display = 'flex';
        document.getElementById('dtStartBtn').style.display = '';
        document.getElementById('dtPauseBtn').style.display = 'none';
        document.getElementById('dtRestartBtn').style.display = 'none';
        document.getElementById('dtCommitBtn').style.display = 'none';

        dtSetStatus('Ready. Select a curve and press Start (or click the road to place the car first).');
        dtBuildSrcCanvas();

        document.addEventListener('keydown', dtKeyDown);
        document.addEventListener('keyup', dtKeyUp);

        if (canvas) {
            canvas.addEventListener('pointermove', dtPointerMove);
            canvas.addEventListener('pointerleave', dtPointerLeave);
            canvas.addEventListener('pointerdown', dtPointerDown);
        }
    };

    window.closeDriveTraceMode = function () {
        dtStop();
        const modal = document.getElementById('driveTraceModal');
        if (modal) modal.style.display = 'none';
        document.removeEventListener('keydown', dtKeyDown);
        document.removeEventListener('keyup', dtKeyUp);
        const canvas = document.getElementById('dtRoadCanvas');
        if (canvas) {
            canvas.removeEventListener('pointermove', dtPointerMove);
            canvas.removeEventListener('pointerleave', dtPointerLeave);
            canvas.removeEventListener('pointerdown', dtPointerDown);
        }
    };

    window.dtStart = function () {
        const sel = document.getElementById('dtCurveSelect');
        if (!sel || !sel.value) {
            dtSetStatus('⚠️ Select a curve first.');
            return;
        }
        DT.curveKey = sel.value;

        const tracks = (typeof getTrackCalibrationsSnapshot === 'function')
            ? getTrackCalibrationsSnapshot() : [];
        DT.track = tracks.find(t =>
            t.id === DT.curveKey ||
            String(t.id || '').toUpperCase() === DT.curveKey.toUpperCase()
        ) || null;

        const cfg = (typeof lastDepthConfig !== 'undefined') ? lastDepthConfig : null;
        DT.startY = cfg ? (cfg.topPx || 0) : 0;
        DT.endY = cfg ? (cfg.bottomPx || (DT.srcCanvas ? DT.srcCanvas.height : 9999)) : 9999;

        // Use placed position if user clicked, otherwise default to track centre
        if (!DT.placementMode) {
            if (DT.track) {
                DT.carX = (DT.track.leftX + DT.track.rightX) / 2;
            } else {
                DT.carX = typeof imageWidth !== 'undefined' ? imageWidth / 2 : 500;
            }
            DT.carY = DT.startY;
        }
        // placementMode = false from here; car pos already set by dtPointerDown

        DT.path = [];
        DT.active = true;
        DT.running = true;
        DT.paused = false;
        DT.lastTs = null;
        DT.steerDeltaX = 0;
        DT.keys = {};

        document.getElementById('dtHintOverlay').style.display = 'none';
        document.getElementById('dtStartBtn').style.display = 'none';
        document.getElementById('dtPauseBtn').style.display = '';
        document.getElementById('dtRestartBtn').style.display = '';
        document.getElementById('dtCommitBtn').style.display = 'none';

        // Set canvas center for mouse steering
        const canvas = document.getElementById('dtRoadCanvas');
        if (canvas) {
            const r = canvas.getBoundingClientRect();
            DT.canvasCenterX = r.left + r.width / 2;
        }

        dtSetStatus('Driving… steer with ← → or move mouse left/right.');
        DT.rafId = requestAnimationFrame(dtLoop);
    };

    window.dtTogglePause = function () {
        DT.paused = !DT.paused;
        const btn = document.getElementById('dtPauseBtn');
        if (btn) btn.textContent = DT.paused ? '▶ Resume' : '⏸ Pause';
        dtSetStatus(DT.paused ? 'Paused. Press Resume to continue.' : 'Driving…');
    };

    window.dtRestart = function () {
        dtStop();
        DT.path = [];
        DT.active = false;
        document.getElementById('dtHintOverlay').style.display = 'flex';
        document.getElementById('dtStartBtn').style.display = '';
        document.getElementById('dtPauseBtn').style.display = 'none';
        document.getElementById('dtRestartBtn').style.display = 'none';
        document.getElementById('dtCommitBtn').style.display = 'none';
        dtSetStatus('Restarted. Press Start to begin again.');
        const canvas = document.getElementById('dtRoadCanvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    };

    function dtFinish() {
        DT.running = false;
        if (DT.rafId) cancelAnimationFrame(DT.rafId);
        document.getElementById('dtPauseBtn').style.display = 'none';
        document.getElementById('dtRestartBtn').style.display = '';
        document.getElementById('dtCommitBtn').style.display = '';
        dtSetStatus(`Done! ${DT.path.length} points recorded. Press ✅ Commit Trace to apply.`);
        dtDrawRoad();
    }

    function dtStop() {
        DT.running = false;
        if (DT.rafId) { cancelAnimationFrame(DT.rafId); DT.rafId = null; }
    }

    window.dtCommit = function () {
        if (!DT.path.length || !DT.curveKey) {
            dtSetStatus('⚠️ No path to commit.');
            return;
        }

        // Build the trace points array in the same format as the rest of the app:
        // [ [xImg, yImg, depthIndex], ... ]
        const pts = DT.path.map(p => [p[0], p[1], p[2]]);

        // Write into lastCurveTraces
        if (typeof lastCurveTraces !== 'undefined') {
            lastCurveTraces[DT.curveKey] = pts;
        } else {
            window.lastCurveTraces = { [DT.curveKey]: pts };
        }

        // Also write back into lastDigitizedCurves if possible
        if (typeof lastDigitizedCurves !== 'undefined' && lastDigitizedCurves &&
            typeof lastDigitizedDepth !== 'undefined' && lastDigitizedDepth &&
            typeof lastDepthConfig !== 'undefined' && lastDepthConfig &&
            DT.track) {

            const nullVal = (typeof lastNullValue !== 'undefined') ? lastNullValue : -999.25;
            const depthArr = lastDigitizedDepth;
            const n = depthArr.length;
            const newValues = new Array(n).fill(nullVal);

            // For each recorded point, write the curve value at the depth index
            for (const [xImg, , dIdx] of pts) {
                if (!Number.isInteger(dIdx) || dIdx < 0 || dIdx >= n) continue;
                const val = (typeof pixelToTrackValue === 'function')
                    ? pixelToTrackValue(xImg, DT.track) : null;
                if (val !== null && Number.isFinite(val)) {
                    newValues[dIdx] = val;
                }
            }

            // Find or create the curve entry
            const key = DT.curveKey;
            if (!lastDigitizedCurves[key]) {
                lastDigitizedCurves[key] = { unit: DT.track.lasUnit || DT.track.displayUnit || '', values: newValues };
            } else {
                lastDigitizedCurves[key].values = newValues;
            }
        }

        // Refresh overlay
        if (typeof renderCurveTraceOverlays === 'function' && typeof lastCurveTraces !== 'undefined') {
            renderCurveTraceOverlays(lastCurveTraces);
        }

        dtSetStatus(`✅ Committed ${pts.length} points for curve "${DT.curveKey}".`);
        document.getElementById('dtCommitBtn').style.display = 'none';

        // Close after short delay
        setTimeout(closeDriveTraceMode, 1200);
    };

    // ── Input handlers ───────────────────────────────────────────────────────

    function dtKeyDown(e) {
        DT.keys[e.key] = true;
        // Prevent arrow keys from scrolling the page while driving
        if (DT.running && !DT.paused &&
            ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
            e.preventDefault();
        }
    }

    function dtKeyUp(e) { DT.keys[e.key] = false; }

    function dtPointerMove(e) {
        DT.mouseClientX = e.clientX;
        const canvas = document.getElementById('dtRoadCanvas');
        if (canvas) {
            const r = canvas.getBoundingClientRect();
            DT.canvasCenterX = r.left + r.width / 2;
        }
    }

    function dtPointerLeave() { DT.mouseClientX = null; }

    /**
     * Click on the road canvas before Start to place the car's starting position.
     * Canvas click coords are converted back to image coords via the current
     * srcCanvas and ZOOM level, then stored in DT.carX / DT.carY.
     */
    function dtPointerDown(e) {
        if (DT.running) return; // only active before start
        const canvas = document.getElementById('dtRoadCanvas');
        if (!canvas || !DT.srcCtx) return;

        const rect = canvas.getBoundingClientRect();
        // Canvas logical coords
        const canvasX = (e.clientX - rect.left) * (DT.ROAD_W / rect.width);
        const canvasY = (e.clientY - rect.top) * (DT.ROAD_H / rect.height);

        // Current viewport in image space (use last known carX/carY or centre)
        const cfg = (typeof lastDepthConfig !== 'undefined') ? lastDepthConfig : null;
        const startY = cfg ? (cfg.topPx || 0) : 0;
        const currentCarX = DT.carX || (DT.srcCanvas ? DT.srcCanvas.width / 2 : 500);
        const currentCarY = DT.carY || startY;

        const srcW = DT.ROAD_W / DT.ZOOM;
        const srcH = DT.ROAD_H / DT.ZOOM;
        const srcX0 = currentCarX - srcW / 2;
        const srcY0 = currentCarY - srcH * 0.4;

        // Map clicked canvas coords → image coords
        const clickImgX = srcX0 + canvasX / DT.ZOOM;
        const clickImgY = srcY0 + canvasY / DT.ZOOM;

        DT.carX = clickImgX;
        DT.carY = clickImgY;
        DT.placementMode = true; // signal dtStart to keep this position

        // Flash the hint overlay off so user sees the placement
        const hint = document.getElementById('dtHintOverlay');
        if (hint) hint.style.display = 'none';

        dtSetStatus(`Car placed at image (${Math.round(clickImgX)}, ${Math.round(clickImgY)}). Press ▶ Start to begin.`);

        // Briefly render the road centred on the new position
        DT.active = true;   // allow dtDrawRoad to render
        dtDrawRoad();
        DT.active = false;
    }

})();

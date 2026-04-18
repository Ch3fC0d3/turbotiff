def digitize():
    """Process digitization request"""
    data = request.json

    # Decode image
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Extract config
    cfg = data['config']
    preview_filters = data.get('preview_filters') or {}
    detected_text = data.get('detected_text') or {}
    depth_cfg = cfg['depth']
    curves = (cfg['curves'] or [])[:6]
    gopt = cfg.get('global_options', {})

    header_metadata = data.get('header_metadata') if isinstance(data, dict) else None

    null_val = float(gopt.get('null', -999.25))
    downsample = int(gopt.get('downsample', 1))
    blur = int(gopt.get('blur', 3))
    min_run = int(gopt.get('min_run', 2))
    smooth_window = int(gopt.get('smooth_window', 5))
    snap_threshold = float(gopt.get('snap_threshold', 1.20)) # Default to 1.20 (20% brighter) as requested/observed

    H, W, _ = img.shape
    top = max(0, int(depth_cfg['top_px']))
    bot = min(H, int(depth_cfg['bottom_px']))
    top_depth = float(depth_cfg['top_depth'])
    bottom_depth = float(depth_cfg['bottom_depth'])
    depth_unit = depth_cfg.get('unit', 'FT')

    nrows = bot - top
    base_depth = compute_depth_vector(nrows, top_depth, bottom_depth)

    # Depth sanity checks
    depth_warnings = compute_depth_warnings({
        'top_px': top,
        'bottom_px': bot,
        'top_depth': top_depth,
        'bottom_depth': bottom_depth,
    }, H)
    
    curve_data = {}
    curve_traces = {}
    curve_warnings = []

    for c in curves:
        # LAS-facing name/unit come from las_mnemonic/las_unit (or name/unit as fallback)
        name = c.get('las_mnemonic') or c.get('name')
        unit = c.get('las_unit') or c.get('unit', '')
        left_px = int(c['left_px'])
        right_px = int(c['right_px'])
        left_value = float(c['left_value'])
        right_value = float(c['right_value'])
        mode = c.get('mode', 'black')
        hot_side = c.get('hot_side')
        pixel_perfect = bool(c.get('pixel_perfect'))
        trace_mode = c.get('trace_mode')
        align_channels = bool(c.get('align_channels'))
        preserve_wiggles = bool(c.get('preserve_wiggles'))
        crest_boost = bool(c.get('crest_boost'))
        if not hot_side and np.isfinite(left_value) and np.isfinite(right_value):
            hot_side = 'right' if right_value >= left_value else 'left'

        # Defensive ROI bounds check: avoid empty slices that crash OpenCV ops.
        # (This can happen if the UI sends left/right reversed, or values are out of range.)
        img_w = int(img.shape[1])
        img_h = int(img.shape[0])
        left_px = max(0, min(img_w - 1, left_px))
        right_px = max(0, min(img_w, right_px))
        if right_px <= left_px:
            curve_warnings.append({
                'curve': name,
                'error': 'Invalid curve bounds (right_px must be > left_px).',
                'left_px': left_px,
                'right_px': right_px,
                'image_width': img_w,
            })
            continue

        top_clamped = max(0, min(img_h - 1, int(top)))
        bot_clamped = max(0, min(img_h, int(bot)))
        if bot_clamped <= top_clamped:
            curve_warnings.append({
                'curve': name,
                'error': 'Invalid depth bounds (bottom_px must be > top_px).',
                'top_px': top_clamped,
                'bottom_px': bot_clamped,
                'image_height': img_h,
            })
            continue

        roi = img[top_clamped:bot_clamped, left_px:right_px]
        if roi is None or roi.size == 0:
            curve_warnings.append({
                'curve': name,
                'error': 'Empty ROI for curve (check left/right and top/bottom).',
                'top_px': top_clamped,
                'bottom_px': bot_clamped,
                'left_px': left_px,
                'right_px': right_px,
            })
            continue

        if align_channels:
            roi = align_rgb_channels(roi)
        if blur > 0:
            bb = blur + 1 if blur % 2 == 0 else blur
            roi = cv2.GaussianBlur(roi, (bb, bb), 0)

        # Define colored modes set (including auto which detects hue automatically)
        colored_modes = {"green", "red", "blue", "auto", "cyan", "magenta", "yellow", "orange", "purple"}

        # NEW: Build a soft probability mask for the curve using color/edges
        # plus vertical-rail suppression. This returns an 8-bit image where
        # higher values mean higher likelihood of curve pixels.
        # Use compute_prob_map for all modes - it has sophisticated edge detection
        # and centerline boost that works well
        mask = compute_prob_map(roi, mode=mode, ui_filters=preview_filters)

        # For black/non-colored modes: definitively zero out vertical rail columns
        # in the final mask so the DP tracer has zero probability there.
        # All the in-compute_prob_map suppression works at float level; this is a
        # hard uint8 zero applied to the mask the tracer actually sees.
        if mode not in colored_modes:
            try:
                h_roi, w_roi = roi.shape[:2]
                _g_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
                _t_roi = cv2.adaptiveThreshold(
                    _g_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV, 21, 4
                )
                # Drop saturated colour pixels so only achromatic dark ink counts
                _hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) if roi.ndim == 3 else None
                if _hsv_roi is not None:
                    _cp_roi = (_hsv_roi[:, :, 1] > 30) & (_hsv_roi[:, :, 2] > 40)
                    _t_roi[_cp_roi] = 0
                
                # 1. Robust Slanted Grid Line Removal (Hough)
                _lines = cv2.HoughLinesP(_t_roi, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
                if _lines is not None:
                    for _line in _lines:
                        _x1, _y1, _x2, _y2 = _line[0]
                        _angle = abs(np.arctan2(_y2 - _y1, _x2 - _x1) * 180.0 / np.pi)
                        if _angle > 70:  # Near vertical
                            cv2.line(mask, (_x1, _y1), (_x2, _y2), 0, 7)

                # 2. Morphological open for strictly vertical fragments
                if h_roi >= 20:
                    _kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
                    _vl = cv2.morphologyEx(_t_roi, cv2.MORPH_OPEN, _kv)
                    
                    # Dilate horizontally to kill the anti-aliased "glow" / blur
                    _kd = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
                    _vl = cv2.dilate(_vl, _kd)
                    
                    # Hard zero the ENTIRE COLUMN if it contains a vertical rail.
                    # If we only zero specific pixels, gaps in the rail allow the free-moving
                    # DP tracer to jump onto the rail fragments during gaps in dashed curves.
                    _rail_cols = np.any(_vl > 0, axis=0)
                    if np.any(_rail_cols):
                        mask[:, _rail_cols] = 0
            except Exception:
                pass

        if mode not in {"green", "red", "blue", "auto", "cyan", "magenta", "yellow", "orange", "purple"}:
            _pm = mask.astype(np.float32) / 255.0
            _pct_nonzero = float(np.mean(_pm > 0.01) * 100)
            _pm_max = float(_pm.max())
            _pm_mean = float(_pm.mean())
            # Debug info removed

        # NEW: Use DP-based smooth path tracing with plausibility checks
        curve_type = c.get('type', 'GR')  # Get curve type for plausibility

        # For explicit color modes, allow more left-right wiggle (lower
        # smoothness penalty) and rely mostly on the DP + local maxima
        # refinement rather than heavy 1D smoothing so the traced path can
        # hug the colored curve as tightly as possible.
        curve_smooth_window = smooth_window
        refine_kwargs = {}
        outlier_threshold = 3.0
        if mode in colored_modes:
            # NO smoothing: window = 1 means no median filter applied
            curve_smooth_window = 1
            # MAXIMUM local window and absolute minimum threshold to snap to any ink
            refine_kwargs = {"dominance_ratio": snap_threshold, "max_shift": 25, "min_prob": 0.005}
            # Disable outlier removal - keep every point for maximum accuracy
            outlier_threshold = 100.0  # Effectively disabled
            dp_smooth_lambda = 0.005
            dp_curv_lambda = 0.002
            max_step_dp = 100
        else:
            # Use user threshold for non-colored modes too (default was 1.1)
            refine_kwargs = {"dominance_ratio": snap_threshold}
            # For black/non-colored modes, we MUST allow large horizontal steps
            # so the tracer can follow highly wobbly dashed curves like SPHI.
            # If we constrain it too much, it will shoot straight through tight
            # corners and leave a trail of disconnected vertical dots.
            # The grid jumping is handled by the strict NaN gap enforcer below
            # and the vertical rail removal above.
            # Black mode MUST have restricted movement to prevent jumping across the grid
            dp_smooth_lambda = 0.001 if curve_type == "GR" else 0.02
            dp_curv_lambda = 0.001 if curve_type == "GR" else 0.005
            max_step_dp = 30 if curve_type == "GR" else 50

        # Optional pixel-perfect skeleton tracer (preserve every bump)
        if ai_tracer.is_available() and trace_mode == "ai_tracer":
            # Use the AI model for tracing
            try:
                # The AI model predicts coordinates relative to the ROI's left edge
                # and already handles scaling to the ROI width.
                xs = ai_tracer.trace(roi)
                confidence = np.ones_like(xs) * 0.95 # Mock high confidence for AI
            except Exception as e:
                print(f"[WARN] AI Tracer failed for {name}: {e}")
                # Fallback to empty if AI fails
                xs = np.full(roi.shape[0], np.nan)
                confidence = np.zeros(roi.shape[0])
        elif pixel_perfect and mode in colored_modes:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if trace_mode == "skeleton_path":
                xs, confidence = trace_curve_skeleton_path(mask)
            else:
                xs, confidence = trace_curve_pixel_perfect(
                    mask,
                    grayscale=gray_roi,
                    bgr=roi,
                    hot_side=hot_side,
                    preserve_wiggles=preserve_wiggles,
                    crest_boost=crest_boost,
                )
            width_px = mask.shape[1]
            # Fill gaps gently to avoid dropping rows
            if xs.size:
                s = pd.Series(xs)
                s = s.interpolate(method='linear', limit_direction='both', limit=max(10, int(xs.size * 0.02)))
                xs = s.to_numpy(dtype=np.float32)
            # Hybrid post-processing: force missed ink peaks onto the curve
            prob = mask.astype(np.float32) / 255.0
            if crest_boost:
                xs = _postprocess_missed_peaks(mask, prob, xs, search_radius=40, min_prob=0.004)
            else:
                xs = _postprocess_missed_peaks(mask, prob, xs, search_radius=30, min_prob=0.008)
        # For colored modes, use the "Fusion" strategy from successful memories:
        # Run both DP and Direct Centerline tracers, then merge per-row based on probability.
        # AND DISABLE EXTRA REFINEMENTS which cause the zig-zag snapping.
        elif mode in colored_modes:
            h_mask, w_mask = mask.shape

            # 1. Run DP Tracer (provides continuity)
            xs_dp, conf_dp = trace_curve_with_dp(
                mask,
                scale_min=left_value,
                scale_max=right_value,
                curve_type=curve_type,
                max_step=max_step_dp,
                smooth_lambda=dp_smooth_lambda,
                curv_lambda=dp_curv_lambda,
                hot_side=hot_side,
            )

            # 2. Local Peak Search Fusion
            # Instead of a global direct tracer (which gets distracted by far-away curves),
            # search locally around the DP path for the true tip of the spike.
            prob_map = mask.astype(np.float32) / 255.0
            xs = np.full(h_mask, np.nan, dtype=np.float32)
            
            search_window = 50
            
            for y in range(h_mask):
                x_dp = xs_dp[y]
                
                if not np.isfinite(x_dp):
                    xs[y] = x_dp
                    continue
                    
                ix_dp = int(round(x_dp))
                if not (0 <= ix_dp < w_mask):
                    xs[y] = x_dp
                    continue
                    
                p_dp = prob_map[y, ix_dp]
                
                # Search for a better peak in the window
                start = max(0, ix_dp - search_window)
                end = min(w_mask, ix_dp + search_window + 1)
                
                # Extract local slice
                local_prob = prob_map[y, start:end]
                if local_prob.size > 0:
                    # Find ALL peaks, not just the first one
                    max_p = local_prob.max()
                    if max_p > 0:
                        # Get all indices that are essentially the max
                        candidates = np.where(local_prob >= max_p * 0.99)[0]

                        # Prefer the strongest local peak that stays close to the DP path.
                        best_cand = candidates[0]
                        best_score = -1e9
                        for c in candidates:
                            x_cand = start + c
                            d = abs(x_cand - x_dp)
                            score = local_prob[c] - 0.15 * d  # strongest penalty to stay on DP path center
                            if score > best_score:
                                best_score = score
                                best_cand = c
                                
                        p_local = float(local_prob[best_cand])

                        # Ridge-centroid snap: if the peak is a short plateau (common on thick ink),
                        # take the weighted centroid of the contiguous plateau region around best_cand.
                        # This avoids consistent 1-2px edge bias from argmax selection.
                        x_local = float(start + best_cand)
                        try:
                            peak_thr = float(max_p) * 0.99
                            left_i = int(best_cand)
                            right_i = int(best_cand)
                            while left_i > 0 and float(local_prob[left_i - 1]) >= peak_thr:
                                left_i -= 1
                            while right_i + 1 < int(local_prob.size) and float(local_prob[right_i + 1]) >= peak_thr:
                                right_i += 1

                            seg = local_prob[left_i:right_i + 1].astype(np.float32)
                            s = float(seg.sum())
                            if s > 1e-8:
                                coords = np.arange(start + left_i, start + right_i + 1, dtype=np.float32)
                                x_local = float((coords * seg).sum() / s)
                        except Exception:
                            x_local = float(start + best_cand)
                        
                        # Fusion Logic:
                        # If local peak is found and is at least 6% as bright as the DP point,
                        # AND it is substantially far away (indicating a missed spike), take it.
                        if p_local > p_dp * 0.06:
                            xs[y] = float(x_local)
                        else:
                            xs[y] = x_dp
                    else:
                        xs[y] = x_dp
                else:
                    xs[y] = x_dp

            # 3. Pure Center-of-Mass refinement (no edge snapping)
            # Use weighted COM per row to center on ink
            h_sr, w_sr = mask.shape
            prob_sr = mask.astype(np.float32) / 255.0
            for y in range(h_sr):
                x_cur = xs[y]
                if not np.isfinite(x_cur):
                    continue
                ix = int(round(x_cur))
                # Search in a local window around current position
                win = 12
                start = max(0, ix - win)
                end = min(w_sr, ix + win + 1)
                row_slice = prob_sr[y, start:end]
                if row_slice.sum() > 1e-6:
                    coords = np.arange(start, end, dtype=np.float32)
                    xs[y] = (coords * row_slice).sum() / row_slice.sum()

            # 4. Refine Peaks (MOVED to after downsampling)
            # We don't run it here to avoid the downsampling smoothing out the sharp tips.
            
            # 8. Minimal cleanup only - NO aggressive snapping to far-away peaks
            s = pd.Series(xs)
            xs = s.interpolate(method='linear', limit_direction='both', limit=max(25, int(xs.size * 0.02))).to_numpy(dtype=np.float32)
            

            # 8b. Clean up artifacts (single-pixel horizontal glitches)
            # The high-sensitivity plateau logic can sometimes trigger on noise.
            # A median filter of size 3 removes single-pixel spikes but keeps real features (width >= 2).
            try:
                 from scipy.signal import medfilt
                 xs_valid_mask = np.isfinite(xs)
                 if np.sum(xs_valid_mask) > 3:
                     # Fill NaNs temporarily
                     xs_filled = xs.copy()
                     xs_filled[~xs_valid_mask] = np.nanmedian(xs)
                     # Apply median filter
                     xs_smooth = medfilt(xs_filled, kernel_size=3)
                     # Restore valid pixels
                     xs[xs_valid_mask] = xs_smooth[xs_valid_mask]
            except ImportError:
                 pass

            # 9. FINAL TIP REFINEMENT (Post-Downsample)
            # Run the peak pusher on the original resolution to catch the absolute edges
            # that might have been smoothed by downsampling.
            # Use small buffer (0.2) to be very sticky to tips.
            # Reduced search radius to 30px to prevent teleporting to distant noise.
            xs = refine_peaks_and_valleys(mask, xs, search_radius=30, min_prob=0.005)

            # Gentle centerline refinement to re-center on ink after outer-edge bias and fusion
            try:
                xs = refine_to_stroke_centerline(mask, xs, threshold_ratio=0.5, window_size=8)
            except Exception:
                pass

            # Global centering correction: subtract median residual vs. ink center-of-mass
            try:
                h_mask, w_mask = mask.shape
                xs_valid = xs[~np.isnan(xs)]
                if xs_valid.size > 0:
                    # Compute per-row center-of-mass of probability
                    probs = mask.astype(np.float32) / 255.0
                    weight_sums = probs.sum(axis=1)
                    com = np.full(h_mask, np.nan, dtype=np.float32)
                    nonzero_rows = weight_sums > 1e-6
                    com[nonzero_rows] = (probs[nonzero_rows] * np.arange(w_mask, dtype=np.float32)).sum(axis=1) / weight_sums[nonzero_rows]
                    deltas = []
                    for y in range(h_mask):
                        if np.isnan(xs[y]) or np.isnan(com[y]):
                            continue
                        deltas.append(xs[y] - com[y])
                    if deltas:
                        median_delta = float(np.median(deltas))
                        xs = xs - median_delta
                    
                    # Do not apply a fixed pixel shift; it prevents true centerline alignment.
            except Exception:
                pass
            
        else:
            # For black/other modes, use enhanced multi-scale tracer
            # This tracer now includes "Grid-Safe Snapping" to handle black grids.
            # It fuses 5 different scales to find the most consistent path and rejects vertical rails.
            xs, confidence = trace_curve_multiscale(
                mask,
                scale_min=left_value,
                scale_max=right_value,
                curve_type=curve_type,
                max_step=max_step_dp,
                smooth_lambda=dp_smooth_lambda,
                hot_side=hot_side,
                bgr_roi=roi,
            )

            # Push trace to hot-side ink edge (tip/crest of each spike)
            # max_dx_pixels=15: grid boundary lines are ~25-30px away, so
            # a 15px hard cap accepts real tip snaps but rejects grid jumps.
            prob_map_bm = mask.astype(np.float32) / 255.0
            xs = ensure_gr_peak_crests(xs, prob_map_bm, hot_side=hot_side, max_dx_pixels=15)

            # Optional final smoothing for non-GR curves (GR needs to stay jagged)
            if curve_type.upper() != "GR":
                 xs = remove_outliers_and_smooth(xs, window=curve_smooth_window, outlier_threshold=outlier_threshold)

        width_px = mask.shape[1]

        # UNIVERSAL GAP FILLING:
        # Aggressive grid removal can leave small gaps where the curve crossed a grid line.
        # We linearly interpolate these gaps to ensure continuity.
        if xs.size > 0:
            s = pd.Series(xs)
            h_mask, w_mask = mask.shape
        # We linearly interpolate these gaps to ensure continuity.
        if xs.size > 0:
            s = pd.Series(xs)
            h_mask, w_mask = mask.shape
            
            # For colored modes, we allow larger gap filling to bridge track lines.
            # For black/non-colored modes (especially dashed curves), we severely
            # restrict gap filling. Interpolating across large gaps creates long,
            # straight, artificial diagonal lines. It is better to return NaN (gaps)
            # than to invent fake data bridging distant points.
            if mode in colored_modes:
                max_gap = max(25, int(h_mask * 0.02))
            else:
                max_gap = max(50, int(h_mask * 0.02)) # Allow ~50px to bridge dashes, big gaps caught by strict enforcement
                
            s = s.interpolate(method='linear', limit_direction='both', limit=max_gap, limit_area=None)
            # Handle edge cases
            if s.isna().any():
                s = s.fillna(method='ffill', limit=max_gap).fillna(method='bfill', limit=max_gap)
            xs = s.to_numpy(dtype=np.float32)

        # For colored modes, apply specific enhancements (peaks, centerline refinement)
        if mode in colored_modes:
            if curve_type.upper() == "GR":
                prob_map = mask.astype(np.float32) / 255.0
                xs = ensure_gr_peak_crests(xs, prob_map, hot_side=hot_side, min_prob=0.01)

            # Final centerline snap for ALL colored modes
            try:
                xs = refine_to_stroke_centerline(mask, xs, threshold_ratio=0.5, window_size=10)
            except Exception:
                pass

            try:
                prob = mask.astype(np.float32) / 255.0
                h_mask, w_mask = prob.shape
                radius = 4
                xs2 = xs.copy()
                for y in range(h_mask):
                    x0 = xs2[y]
                    if not np.isfinite(x0):
                        continue
                    ix = int(round(x0))
                    x_min = max(0, ix - radius)
                    x_max = min(w_mask, ix + radius + 1)
                    row = prob[y, x_min:x_max]
                    if row.size == 0:
                        continue
                    s = float(row.sum())
                    if s <= 1e-8:
                        continue
                    coords = np.arange(x_min, x_max, dtype=np.float32)
                    xs2[y] = float((coords * row).sum() / s)
                xs = xs2
            except Exception:
                pass

            if curve_type.upper() == "GR":
                prob_map = mask.astype(np.float32) / 255.0
                xs = ensure_gr_peak_crests(xs, prob_map, hot_side=hot_side)

            # Optional final local peak snap; kept disabled because it
            # quantizes to integer columns and can reintroduce zig-zags.
            do_final_peak_snap = False
            if do_final_peak_snap:
                # FINAL STEP: refine each point to local probability maximum
                # Tight window to avoid sideways wander
                h_mask, w_mask = mask.shape
                xs_refined_final = np.copy(xs)
                
                local_search_radius = 2  # Very tight window to avoid sideways wander
                
                for y in range(h_mask):
                    if not np.isnan(xs[y]):
                        x_current = int(round(xs[y]))
                        
                        # Define tight search window around current position
                        x_min = max(0, x_current - local_search_radius)
                        x_max = min(w_mask, x_current + local_search_radius + 1)
                        
                        # Find local maximum within this small window
                        row_segment = mask[y, x_min:x_max].astype(np.float32)
                        
                        if len(row_segment) > 0 and row_segment.max() > 0:
                            # Find peak position within window
                            local_peak_idx = np.argmax(row_segment)
                            # Convert back to full image coordinates
                            xs_refined_final[y] = x_min + local_peak_idx
                
                xs = xs_refined_final
        else:
            # For non-colored modes, keep the original vertical-rail rejection logic
            xs_valid = xs[~np.isnan(xs)]
            if xs_valid.size > 0:
                dyn_range = float(np.nanmax(xs_valid) - np.nanmin(xs_valid))
                min_dyn = max(4.0, 0.02 * float(width_px))
                if dyn_range < min_dyn:
                    xs_fallback = pick_curve_x_per_row(mask, min_run=min_run)
                    xs_fallback = smooth_nanmedian(xs_fallback, window=curve_smooth_window)
                    xs = xs_fallback
                    xs_valid = xs[~np.isnan(xs)]

            if xs_valid.size > 0:
                std_x = float(np.nanstd(xs_valid))
                std_threshold = max(1.0, 0.005 * float(width_px))
                # Only reject near-perfectly-vertical traces (rail lock-on).
                # Use a very tight threshold: 0.5% of track width or 1.0px minimum.
                # Slow curves like DTC/RHOB can legitimately have low std.
                if std_x < std_threshold:
                    xs[:] = np.nan

        vals = np.full(xs.shape, np.nan, dtype=np.float32)
        valid = ~np.isnan(xs)
        vals[valid] = left_value + (xs[valid] / max(1, width_px-1)) * (right_value - left_value)

        vals_out = np.where(np.isnan(vals), null_val, vals).astype(np.float32)
        curve_data[name] = {'unit': unit, 'values': vals_out}

        # Build a sparse set of trace points in original image coordinates for UI overlay
        trace_points = []
        if xs.size > 0:
            # Only sample from rows where the DP tracer produced a valid X.
            # This avoids the corner-case where all sampled indices land on
            # NaNs even though some rows are valid, which would yield an
            # empty trace and no cyan dots in the UI.
            valid_rows = np.where(~np.isnan(xs))[0]
            if valid_rows.size > 0:
                # Send EVERY single traced point - no sampling at all.
                # This creates a completely solid line that shows the exact trace.
                for row_idx in valid_rows:
                    x_val = xs[row_idx]
                    x_img = round(left_px + x_val)
                    y_img = int(top + row_idx)
                    trace_points.append([x_img, y_img])

        curve_traces[name] = trace_points
    
    # Resample to fixed 0.5 ft step when using feet
    las_depth = base_depth
    las_curve_data = curve_data
    ai_payload = None
    ai_summary = None
    digitized_depth = None
    digitized_curves = None
    if depth_unit.upper() == "FT" and base_depth.size > 1:
        start = float(base_depth[0])
        stop = float(base_depth[-1])
        step_mag = 0.5

        if stop >= start:
            las_depth = np.arange(start, stop + step_mag * 0.5, step_mag, dtype=np.float32)
        else:
            las_depth = np.arange(start, stop - step_mag * 0.5, -step_mag, dtype=np.float32)

        las_curve_data = {}
        for name, meta in curve_data.items():
            vals = meta["values"].astype(np.float32)
            valid_mask = vals != null_val

            if not np.any(valid_mask):
                new_vals = np.full(las_depth.shape, null_val, dtype=np.float32)
            else:
                depth_valid = base_depth[valid_mask]
                vals_valid = vals[valid_mask]
                order = np.argsort(depth_valid)
                depth_sorted = depth_valid[order]
                vals_sorted = vals_valid[order]
                interp_vals = np.interp(las_depth, depth_sorted, vals_sorted, left=null_val, right=null_val)
                new_vals = interp_vals.astype(np.float32)

            las_curve_data[name] = {"unit": meta.get("unit", ""), "values": new_vals}

    if not curve_data:
        return jsonify({
            'error': 'No valid curves to digitize. Please check curve bounds and depth settings.',
            'curve_warnings': curve_warnings,
            'depth_warnings': depth_warnings,
        }), 400

    # Run simple curve sanity checks (outlier warnings) on the final LAS depth grid
    outlier_warnings = compute_curve_outlier_warnings(curves, las_curve_data, null_val)

    # Prepare digitized vectors for frontend cursor readout (always, even without lasio)
    try:
        digitized_depth = las_depth.tolist()
        digitized_curves = {
            name: {
                "unit": meta.get("unit", ""),
                "values": (meta.get("values").tolist() if meta.get("values") is not None else None),
            }
            for name, meta in las_curve_data.items()
        }
    except Exception:
        digitized_depth = None
        digitized_curves = None

    # Generate LAS file
    las_content = write_las_simple(las_depth, las_curve_data, depth_unit, header_metadata=header_metadata)

    # Validate LAS output if possible
    validation = {
        'passed': True,
        'message': 'LAS validation skipped (lasio not installed).'
    }
    if LASIO_AVAILABLE:
        try:
            lasio.read(StringIO(las_content))
            validation = {
                'passed': True,
                'message': 'LAS parsed successfully with lasio.'
            }
        except Exception as exc:
            validation = {
                'passed': False,
                'message': f'LAS validation failed: {exc}'
            }

        # Build AI analysis payload (OCR + LAS stats + user curve config)
        ai_payload = build_ai_analysis_payload(las_content, detected_text, curves)
        ai_summary = call_hf_curve_analysis(ai_payload) if ai_payload else None

    return jsonify({
        'success': True,
        'las_content': las_content,
        'filename': build_las_filename_from_metadata(header_metadata, default_name='digitized_log.las'),
        'validation': validation,
        'outlier_warnings': outlier_warnings,
        'depth_warnings': depth_warnings,
        'curve_warnings': curve_warnings,
        'curve_traces': curve_traces,
        'ai_payload': ai_payload,
        'ai_summary': ai_summary,
        'digitized_depth': digitized_depth,
        'digitized_curves': digitized_curves,
    })

@app.route('/health')

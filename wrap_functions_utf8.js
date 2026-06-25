        function wrapRawXToTrack(rawX, track) {
            const x = Number(rawX);
            const left = Number(track && track.leftX);
            const right = Number(track && track.rightX);
            const width = right - left;
            if (![x, left, right, width].every(Number.isFinite) || width <= 0) return x;
            const offset = x - left;
            const wrappedOffset = positiveModulo(offset, width);
            // Keep exact upper-boundary samples on the visible right rail. Once
            // there is real overflow, positive modulo resumes from the left rail.
            if (Math.abs(wrappedOffset) < 1e-7 && offset > 0 && Math.abs(offset / width - Math.round(offset / width)) < 1e-7) {
                return right;
            }
            return left + wrappedOffset;
        }

        function unwrapDisplayXNearReference(displayX, referenceRawX, track) {
            const visibleX = Number(displayX);
            const referenceX = Number(referenceRawX);
            const left = Number(track && track.leftX);
            const right = Number(track && track.rightX);
            const width = right - left;
            if (![visibleX, referenceX, left, right, width].every(Number.isFinite) || width <= 0) {
                return visibleX;
            }
            const baseX = wrapRawXToTrack(visibleX, track);
            const nearestCycle = Math.round((referenceX - baseX) / width);
            return baseX + nearestCycle * width;
        }

        function trackValueToPixelX(value, track, options = {}) {
            if (!track) return null;
            const leftX = track.leftX;
            const rightX = track.rightX;
            const scaleMin = track.scaleMin;
            const scaleMax = track.scaleMax;
            const denom = scaleMax - scaleMin;
            if (!Number.isFinite(leftX) || !Number.isFinite(rightX) ||
                !Number.isFinite(scaleMin) || !Number.isFinite(scaleMax) ||
                Math.abs(denom) < 1e-3) {
                return null;
            }
            const valueForDisplay = options && options.rawValue === true
                ? value
                : getVisibleTrackValue(value, track);
            const frac = (valueForDisplay - scaleMin) / denom;
            return leftX + frac * (rightX - leftX);
        }

        function getTracePointRawX(point, track, curveId = null) {
            if (!Array.isArray(point)) return null;
            const storedRawX = point[5] === null || point[5] === undefined || point[5] === ''
                ? null
                : Number(point[5]);
            if (Number.isFinite(storedRawX)) return storedRawX;

            const depthIndex = Number.isInteger(point[2]) ? point[2] : -1;
            if (depthIndex >= 0) {
                const found = findDigitizedCurveEntry(curveId || (track && track.id));
                const values = found && found.entry && Array.isArray(found.entry.values) ? found.entry.values : null;
                if (values && depthIndex < values.length && !isMissingDigitizedValue(values[depthIndex])) {
                    const rawX = trackValueToPixelX(values[depthIndex], track, { rawValue: true });
                    if (Number.isFinite(rawX)) return rawX;
                }
            }
            return Number(point[0]);
        }

        function isTrackWrapActive(track, curveId = null, points = null) {
            if (!track) return false;
            if (track.wrapped) return true;
            if (getWrapMarkersForCurve(curveId || track.id).length) return true;

            const left = Math.min(Number(track.leftX), Number(track.rightX));
            const right = Math.max(Number(track.leftX), Number(track.rightX));
            if (!Number.isFinite(left) || !Number.isFinite(right) || right <= left) return false;
            const tolerance = Math.max(1, (right - left) * 0.002);

            if (Array.isArray(points) && points.some(point => {
                const rawX = getTracePointRawX(point, track, curveId);
                return Number.isFinite(rawX) && (rawX < left - tolerance || rawX > right + tolerance);
            })) {
                return true;
            }

            const found = findDigitizedCurveEntry(curveId || track.id);
            const values = found && found.entry && Array.isArray(found.entry.values)
                ? found.entry.values
                : null;
            return !!(values && values.some(value => {
                if (isMissingDigitizedValue(value)) return false;
                const rawX = trackValueToPixelX(value, track, { rawValue: true });
                return Number.isFinite(rawX) && (rawX < left - tolerance || rawX > right + tolerance);
            }));
        }

        function drawWrappedTrackSegment(ctx, rawX1, y1, rawX2, y2, track, scaleX, scaleY) {
            const left = Number(track && track.leftX);
            const right = Number(track && track.rightX);
            const width = right - left;
            const x1 = Number(rawX1);
            const x2 = Number(rawX2);
            const startY = Number(y1);
            const endY = Number(y2);
            if (![left, right, width, x1, x2, startY, endY].every(Number.isFinite) || width <= 0) {
                ctx.lineTo(Number(rawX2) * scaleX, Number(y2) * scaleY);
                return;
            }

            const dx = x2 - x1;
            if (Math.abs(dx) < 1e-9) {
                ctx.lineTo(wrapRawXToTrack(x2, track) * scaleX, endY * scaleY);
                return;
            }

            const isOnBoundary = rawX => {
                const cycles = (rawX - left) / width;
                return Math.abs(cycles - Math.round(cycles)) < 1e-7;
            };
            // At an exact boundary, the side used depends on travel direction:
            // increasing raw X resumes at the left rail; decreasing resumes right.
            if (isOnBoundary(x1)) {
                ctx.moveTo((dx > 0 ? left : right) * scaleX, startY * scaleY);
            }

            const boundaries = [];
            const eps = 1e-7;
            if (dx > 0) {
                let boundary = left + (Math.floor((x1 - left) / width) + 1) * width;
                while (boundary < x2 - eps) {

def refine_trace_with_local_maxima(mask, xs, max_shift=6, dominance_ratio=1.1, min_prob=0.2):
    """Nudge the DP path toward obvious local maxima in the prob mask.

    For each row, look in a small window around the current DP x and, when
    there is a clearly stronger nearby maximum, move the x coordinate toward
    the probability-weighted centroid of that local peak. This keeps the
    path glued to the same physical curve while following its wiggles more
    tightly.
    """
    if mask is None or xs is None:
        return xs
    if not hasattr(xs, "size") or xs.size == 0:
        return xs

    h, w = mask.shape[:2]
    if h < 1 or w < 1:
        return xs

    prob = mask.astype(np.float32) / 255.0
    xs_ref = xs.copy()

    n_rows = min(h, xs_ref.size)
    for y in range(n_rows):
        x = xs_ref[y]
        if not np.isfinite(x):
            continue

        x_c = int(round(float(x)))
        if x_c < 0 or x_c >= w:
            continue

        row = prob[y]
        x0 = max(0, x_c - max_shift)
        x1 = min(w, x_c + max_shift + 1)
        window = row[x0:x1]
        if window.size == 0:
            continue

        max_p = float(window.max())
        if max_p < min_prob:
            continue

        # Compare the best pixel in the window to the current DP location.
        local_peak_idx = int(np.argmax(window))
        x_peak = x0 + local_peak_idx
        p_peak = float(row[x_peak])
        p_dp = float(row[x_c])
        if p_dp <= 0:
            p_dp = 1e-6

        if p_peak >= dominance_ratio * p_dp:
            # Use a weighted centroid within the local window, restricted to
            # the top part of the peak, so the path follows the center of the
            # curve stroke instead of a single edge pixel.
            xs_local = np.arange(x0, x1, dtype=np.float32)
            weights = window.astype(np.float32)
            peak_mask = weights >= max_p * 0.6

            try:
                # If we have no clearly strong pixels, fall back to any
                # non-zero weights.
                if not np.any(peak_mask):
                    peak_mask = weights > 0.0
                idx_strong = np.flatnonzero(peak_mask)
                if idx_strong.size > 0:
                    # Group consecutive strong pixels into contiguous
                    # segments so we can snap to the center of a physical
                    # stroke rather than an arbitrary mix of nearby blobs.
                    start = idx_strong[0]
                    prev = idx_strong[0]
                    segments = []
                    for idx in idx_strong[1:]:
                        if idx == prev + 1:
                            prev = idx
                        else:
                            segments.append((start, prev))
                            start = idx
                            prev = idx
                    segments.append((start, prev))

                    # Prefer the segment that actually contains the local
                    # peak; otherwise choose the closest segment by center.
                    seg_best = None
                    for s, e in segments:
                        if s <= local_peak_idx <= e:
                            seg_best = (s, e)
                            break
                    if seg_best is None and segments:
                        seg_best = min(
                            segments,
                            key=lambda se: abs((se[0] + se[1]) * 0.5 - local_peak_idx),
                        )

                    if seg_best is not None:
                        s, e = seg_best
                        seg_slice = slice(s, e + 1)
                        seg_weights = weights[seg_slice]
                        seg_xs = xs_local[seg_slice]
                        wsum = float(seg_weights.sum())
                        if wsum > 0.0:
                            x_centroid = float((seg_xs * seg_weights).sum() / wsum)
                        else:
                            x_centroid = float(seg_xs.mean())
                        xs_ref[y] = x_centroid
                        continue

            except Exception:
                # If anything about the segment-based logic misbehaves for a
                # particular row, quietly fall back to the simpler
                # peak-centered weighted centroid used previously.
                pass

            # Fallback: original behavior - centroid of the strong part of
            # the window around the dominant peak.
            if not np.any(peak_mask):
                peak_mask = weights > 0.0
            weights_centroid = weights * peak_mask.astype(np.float32)
            wsum = float(weights_centroid.sum())
            if wsum > 0.0:
                x_centroid = float((xs_local * weights_centroid).sum() / wsum)
                xs_ref[y] = x_centroid

    return xs_ref



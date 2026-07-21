import numpy as np
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap

@jit(nopython=True, cache=True)
def run_viterbi(cost, prob, max_step, smooth_lambda, curv_lambda, wrap_enabled=False):
    """
    Optimized Viterbi algorithm for curve tracing using Numba.
    
    Args:
        cost: (h, w) float32 array of costs
        prob: (h, w) float32 array of probabilities (0-1)
        max_step: int, maximum horizontal jump
        smooth_lambda: float, penalty for 1st derivative (jumps)
        curv_lambda: float, penalty for 2nd derivative (kinks)
        wrap_enabled: allow circular transitions between the track edges
        
    Returns:
        xs: (h,) float32 array of x-coordinates (with NaNs)
        confidence: (h,) float32 array of confidence scores
    """
    h, w = cost.shape
    big = 1e6
    
    # DP tables
    dp = np.full((h, w), big, dtype=np.float32)
    prev = np.full((h, w), -1, dtype=np.int16)
    
    # First row
    dp[0, :] = cost[0, :]
    
    # Forward pass
    for y in range(1, h):
        for x in range(w):
            best_val = big
            best_xp = -1

            candidate_count = 2 * max_step + 1 if wrap_enabled else 0
            x0 = max(0, x - max_step)
            x1 = min(w, x + max_step + 1)
            loop_count = candidate_count if wrap_enabled else x1 - x0
            for candidate_idx in range(loop_count):
                if wrap_enabled:
                    dx = candidate_idx - max_step
                    xp = x - dx
                    if xp < 0:
                        xp += w
                    elif xp >= w:
                        xp -= w
                else:
                    xp = x0 + candidate_idx
                    dx = x - xp
                # 1st derivative penalty
                smooth_penalty = smooth_lambda * (dx * dx)
                
                # 2nd derivative penalty
                if curv_lambda > 0.0 and y >= 2:
                    xpp = prev[y - 1, xp]
                    if xpp >= 0:
                        prev_dx = xp - xpp
                        if wrap_enabled:
                            if 2 * prev_dx > w:
                                prev_dx -= w
                            elif 2 * prev_dx < -w:
                                prev_dx += w
                        k = dx - prev_dx
                        smooth_penalty += curv_lambda * (k * k)
                
                v = dp[y - 1, xp] + cost[y, x] + smooth_penalty
                if v < best_val:
                    best_val = v
                    best_xp = xp
            
            dp[y, x] = best_val
            prev[y, x] = best_xp
            
    # The path must span the full image. Selecting an endpoint from an earlier
    # row favors shorter cumulative paths and leaves the tail under-optimized.
    best_cost = big
    best_y = h - 1
    best_x = 0

    for x in range(w):
        if dp[best_y, x] < best_cost:
            best_cost = dp[best_y, x]
            best_x = x
            
    # Backtrack
    path_x = np.full(h, -1, dtype=np.int32)
    path_x[best_y] = best_x
    
    for y in range(best_y, 0, -1):
        curr_x = path_x[y]
        if curr_x >= 0:
            path_x[y - 1] = prev[y, curr_x]
            
    # Compute confidence and result
    xs = np.full(h, np.nan, dtype=np.float32)
    confidence = np.zeros(h, dtype=np.float32)
    
    for y in range(h):
        x = path_x[y]
        if x < 0 or x >= w:
            confidence[y] = 0.0
            continue
            
        p_best = prob[y, x]
        
        # Compare the selected path pixel against the strongest *other* local
        # candidate. The selected point is not necessarily the row maximum;
        # continuity can intentionally choose a weaker pixel.
        best_other = -1.0
        x0 = max(0, x - max_step)
        x1 = min(w, x + max_step + 1)
        candidate_count = 2 * max_step + 1 if wrap_enabled else x1 - x0
        for candidate_idx in range(candidate_count):
            if wrap_enabled:
                local_dx = candidate_idx - max_step
                xi = x - local_dx
                if xi < 0:
                    xi += w
                elif xi >= w:
                    xi -= w
            else:
                xi = x0 + candidate_idx
            if xi == x:
                continue
            val = prob[y, xi]
            if val > best_other:
                best_other = val

        if best_other < 0.0:
            best_other = 0.0
        conf = p_best - best_other
        if conf < 0.0:
            conf = 0.0
        elif conf > 1.0:
            conf = 1.0
            
        confidence[y] = conf
        
        # Threshold
        if p_best >= 0.01:
            xs[y] = float(x)
            
    return xs, confidence

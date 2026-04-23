# Line Detection Experiment Log

Started: 2026-04-22

Purpose: keep a compact record of black-curve line-detection changes so we do not repeat approaches that made the trace worse.

## Current Baseline

As of `d7d2730`, the app keeps:

- `2b09151 Refine black traces by continuous line support`
- `8069b98 Remove black grid before pixel tracing`
- `d7d2730 Revert "Run staged multi-pass black trace refinement"`

Main active files/functions:

- `web_app.py`
- `compute_prob_map`
- `build_black_prescan_grid_removed`
- `refine_black_trace_to_continuous_line`
- `refine_black_trace_to_dark_run_center`
- `guard_trace_outliers_rolling_median`
- `guard_trace_velocity`

## Experiment History

| Commit | Idea | What Changed | Result | Status | Takeaway |
| --- | --- | --- | --- | --- | --- |
| pre-experiment baseline | DP + black stroke recentering | Used probability map, DP trace, local maxima, dark-run centerline, rolling outlier guards, velocity guard, and small median cleanup. | Worked sometimes, but black traces could lock onto grid/text and draw horizontal shelf artifacts. | Historical baseline | The problem is mostly grid/printed-noise competition, not just smoothing. |
| `ddb0cee` | Suppress short lateral excursions | Added `suppress_short_lateral_excursions` to replace short sideways shelves with a local trend. | Made the trace worse; produced/kept bad shelves and over-corrected real-looking movement. | Reverted by `0e2da4f` | Do not repeat pure post-hoc shelf suppression without image evidence. It can erase valid movement and invent long straight sections. |
| `2b09151` | Look for a continuous line instead of a crest | Replaced the risky darkest-crest snap with `refine_black_trace_to_continuous_line`, scoring nearby pixels by vertical line support and residual curve ink. | Better conceptually; avoids some one-row darkest-pixel traps, but still misses larger grid/text artifacts. | Active | Keep as the current single-pass line-support baseline. If changing it, compare against this commit. |
| `8069b98` | Remove grid before pixel tracing | Added `build_black_prescan_grid_removed`; black/white scans now build the probability map from grid-removed residual pixels instead of raw dark pixels. | Helped reduce raw grid entering the scan, but still not enough on difficult captures. | Active | Good direction. Avoid OR-ing broad raw-dark masks back into the residual mask because that reintroduces grid. |
| `6901221` | Multiple broad-to-tight passes | Added `refine_black_trace_multi_pass`: broad rescue, medium line follow, tight pixel fit. | Looked worse in user screenshot; trace jumped into large false shelves and wrong branches. | Reverted by `d7d2730` | Do not repeat multi-pass broad rescue with weak constraints. More passes amplified mistakes instead of fixing them. |

## Do Not Repeat Without A New Test

- Do not add another broad multi-pass refinement that can move 30+ px unless it is anchored by a reliable crop/test case.
- Do not use darkest-pixel or crest-only snapping for black mode on dense grids; grid and text are often darker than the curve.
- Do not use post-hoc smoothing/shelf suppression as the primary fix; it hides symptoms and can distort real curve movement.
- Do not broadly merge raw black threshold masks with residual masks after grid removal; that can put the removed grid back into the scan.

## Better Next Directions

- Save representative bad crops as fixtures and run a small before/after trace test before pushing changes.
- Make grid removal output visible/debuggable in the UI so we can see whether the curve or the grid is surviving.
- Try connected-component scoring after grid removal: favor components that are vertically continuous, thin, and near the guided trace, while rejecting long horizontal/vertical grid components.
- Try a user-assisted guide pass: let the first trace define a band, remove grid inside that band, then rerun DP only inside the band.
- Add a "trace debug export" that saves `roi`, `prob_map`, `grid_removed`, `residual_mask`, and final `xs` for a selected curve.

## Update Rule

Every line-detection change should add one row above with:

- Commit hash
- Short idea
- What changed
- User-visible result
- Active or reverted status
- What we learned

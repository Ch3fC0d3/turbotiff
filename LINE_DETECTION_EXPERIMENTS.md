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
| `ec3a0da` | Trace debug export | Added black-curve debug artifacts to digitize responses and auto-capture payloads: ROI, overlay, probability map, grid-removed view, residual mask/score, grid score, component preview, and metrics. | Tooling change only; intended to make the next failure inspectable instead of tuning from screenshots. | Active | Use this before the next algorithm change. Check whether failures come from preprocessing, candidate components, or path selection. |

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

## Missing Concepts Captured 2026-04-23

- Orientation-aware scoring: black curve evidence should reward thin, mostly vertical/continuous ridges and penalize horizontal support, even when the horizontal line is darker than the curve.
- Explicit text rejection: labels, numbers, and annotations can create compact blobs, corners, and short dark strokes that beat the curve locally.
- Connected-component pathing: score candidate components or segments before DP instead of letting the solver choose from every noisy pixel.
- Stronger path model: jumps, sudden slope changes, branch switching, and long horizontal dwell should be penalized inside the solver cost, not only cleaned up afterward.
- Ridge or centerline extraction: after residual binarization, thin/skeletonize candidate ink and trace the strongest centerline near the guide.
- Track-aware constraints: use known left/right bounds, guided safe bands, label zones, scale markings, and track geometry to remove false candidates before tracing.
- Failure detection: emit confidence metrics when support is weak, jumps are large, horizontal dwell is high, or too many rows rely on weak evidence.
- Better normalization: deskew, background flattening, local contrast normalization, and line-width normalization can make grid suppression and ridge scoring more reliable.
- Ground-truth fixtures: screenshots alone are not enough; keep a tiny benchmark of bad crops plus expected/manual traces.
- Branch memory: prefer staying on the same ridge/component identity and penalize switching unless support becomes much better.

## Next Experiment Plan

| Priority | Idea | What To Change | Why It May Help | Risk | How To Test | Success Signal |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Trace debug export | Export `roi`, `prob_map`, `grid_removed`, `residual_mask`, `guide_band`, `final_xs`, and overlay image for a selected curve. | Makes failures visible instead of guessing whether preprocessing or path scoring failed. | Low | Run on 3 known bad black-curve examples. | We can see whether the curve survives preprocessing and exactly where the trace jumps. |
| 2 | Guided band rerun | After first trace, build a narrow band around it and rerun black tracing only inside that band. | Limits competition from distant grid/text artifacts. | Low-Med | Compare before/after on the same fixtures. | Fewer branch jumps and fewer long shelves. |
| 3 | Connected-component scoring | After grid removal, score components by vertical continuity, thinness, closeness to guide, and low horizontality; suppress poor components before DP. | Attacks the real issue: grid/text winning as candidates. | Med | Save component overlay and trace result on bad crops. | Trace stays on vertically continuous curve pieces more often. |
| 4 | Horizontality penalty | Penalize long horizontal support in black refinement and/or DP cost. | Shelves are often horizontal junk, not real curve movement. | Low-Med | Use shelf-heavy fixture set. | Noticeable reduction in flat sideways shelves. |
| 5 | Curvature-aware path cost | In DP, penalize sudden slope change and component switching, not just raw displacement. | Prevents wrong-branch jumps before cleanup. | Med | Compare to current velocity/outlier guards alone. | Smoother but still responsive trace with fewer abrupt lane changes. |
| 6 | Text-like blob rejection | Reject compact dark blobs or annotation-like components after grid removal. | Printed numbers and labels may be beating curve locally. | Med | Test on logs with dense labels. | Less attraction to text without harming curve continuity. |
| 7 | Skeleton/centerline trial | For black residuals inside guide band, test thinning/skeleton route and trace strongest centerline. | Thick black strokes may trace better as centerlines than raw dark pixels. | Med-High | Prototype on 2-3 difficult crops only. | Better center placement on thick curves. |
| 8 | Confidence score | Emit warnings when support is weak: too many jumps, low-support rows, long horizontal dwell. | Lets bad traces be flagged instead of silently accepted. | Low | Add metrics to debug export and review failures. | Bad traces become easy to identify automatically. |

## Recommended Order

- Pass 1: visibility first. Implement trace debug export, then guided band rerun.
- Pass 2: better candidate evidence. Implement connected-component scoring, text-like blob rejection, and horizontality penalty.
- Pass 3: stronger path model. Implement curvature-aware path cost and confidence score.
- Pass 4: optional deeper trial. Try skeleton/centerline tracing only after debug tooling and component filtering exist.

## Guardrails For Next Round

- Do not change many trace ideas at once; isolate each experiment.
- Use the same saved bad crops for every experiment.
- Save before/after overlays.
- Record whether shelves decreased.
- Record whether true curve movement was preserved.
- Compare against `d7d2730` plus the current active path.

## Minimum Fixture Set

- Dense grid case.
- Text-heavy case.
- Branch/shelf failure case.
- Almost-works-already case.

## Suggested Detailed Log Row

| Commit | Idea | What Changed | Test Fixtures | Result | Status | Takeaway |
| --- | --- | --- | --- | --- | --- | --- |
| `abcd123` | Guided band rerun | Reran black DP only inside +/-12 px band around first-pass trace. | `black_grid_01`, `black_text_02`, `black_shelf_03` | Reduced distant branch jumps; still failed near labels. | Active | Banding helps, but candidate cleanup is still needed. |

## Update Rule

Every line-detection change should add one row above with:

- Commit hash
- Short idea
- What changed
- User-visible result
- Active or reverted status
- What we learned

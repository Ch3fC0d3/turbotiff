# Curve Carryover Implementation Notes

## Known working reference

- Commit: `9409ba3` (`Restore move carryover crossing`)
- Main implementation file: `templates/workspace.html`
- This commit was the last version before the June 18 rollback request.

## How crossover worked

1. Keep two X coordinates for wrapped trace points:
   - `point[0]` is the visible X inside the track.
   - `point[5]` is the logical/raw X and may continue beyond either rail.
2. `wrapRawXToTrack()` maps raw X back into the visible track using modulo.
3. `getTracePointRawX()` retrieves `point[5]`, falling back to the digitized value or visible X.
4. `isTrackWrapActive()` treats a curve as wrapped when its track flag, wrap markers, raw points, or values indicate overflow.
5. During a Move drag, do not clamp the pointer before it crosses a rail. On the first crossing:
   - Set `track.wrapped = true`.
   - Check the corresponding `wrapped{index}` checkbox.
   - Continue storing the unwrapped X in `point[5]`.
6. Render with `drawWrappedTrackSegment()`. Split the path at each rail with `lineTo()` followed by `moveTo()` on the opposite rail. Never draw a connector across the full track.
7. Keep connector cleanup away from genuine wrap transitions. A real transition must retain adjacent points in different logical cycles so the renderer can split it correctly.

## Important guardrails

- First-crossing activation should apply to the Move/pointer drag only unless Add Point behavior is deliberately redesigned.
- Do not choose carryover sides using visible X alone; compare logical/raw X near the edited point.
- Do not clamp raw X to image or track bounds before converting it to a digitized value.
- Do not replace a wrap transition with a long horizontal segment or relax neighboring points across that transition.
- Preserve the wrapped checkbox state so later renders and edits use the same behavior.

## Recovery reference

To inspect the complete known-good implementation without changing the worktree:

```powershell
git show 9409ba3:templates/workspace.html
git show 9409ba3 -- templates/workspace.html
```


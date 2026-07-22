# Topology Decoder Phase 3

## Status And Scope

Phase 3 adds an opt-in path decoder that models horizontal position, local
slope, and cylindrical wrap index together. It does not replace the production
legacy decoder. The workspace exposes the choice independently as `Legacy DP`
or `Topology DP`, and API metadata records the requested decoder, actual
decoder, fallback reason, topology, configuration, timing, confidence, and
explicit wrap events.

The detector and decoder are separate choices:

```text
image -> classic / neural Phase 2 / hybrid Phase 2 evidence
      -> legacy_dp OR topology_dp
      -> visible X + wrap index + unwrapped X
      -> scale conversion
      -> LAS values and segmented overlay
```

## Decoder State

At row `y`, each state contains:

```text
(visible_x, slope, wrap_index, last_wrap_row, last_wrap_direction, event_count)
```

`visible_x` is always inside the track. The physical horizontal coordinate is:

```text
unwrapped_x = visible_x + wrap_index * track_width
```

The transition cost combines soft observation evidence, step magnitude,
curvature, tangent agreement, wrap cost, optional wrap evidence, and a reverse
wrap penalty. A deterministic beam retains diverse candidates across the track.
Set `beam_width=None` for exact state retention on small test problems. A
candidate-state safeguard rejects unsafe configurations before allocation.

## Bounded And Cylindrical Transitions

| Topology | Allowed transition |
| --- | --- |
| `bounded` | New X must remain inside the track; wrap index remains zero. |
| `cylindrical` | Ordinary steps remain in the current wrap layer. |
| `cylindrical` | Right-to-left wrap is allowed only from the right edge band to the left edge band. |
| `cylindrical` | Left-to-right wrap is allowed only from the left edge band to the right edge band. |

`maximum_wrap_count` limits both event count and wrap-layer magnitude.
`minimum_rows_between_wraps` prevents rapid wrap oscillation. A direction
reversal receives an additional penalty. Border proximity alone does not imply
a wrap; the full path must make the transition worthwhile.

## Evidence

`CurveEvidence` accepts calibrated soft fields without forcing any source to be
binary:

- centerline probability;
- complete-stroke probability;
- smooth distance-to-center evidence;
- local tangent direction;
- grid probability;
- classic detector probability;
- optional per-row wrap probabilities and a validity mask.

Missing fields are omitted from normalization, so absent Phase 2 heads do not
dilute the evidence that is present. Grid is a soft penalty with overlap relief
where center evidence is strong.

## Output Contract

`CurvePathResult` contains:

- `x_by_row`: visible in-track X;
- `unwrapped_x_by_row`: continuous physical X across wraps;
- `wrap_index_by_row`: signed wrap layer;
- `slope_by_row` and `confidence_by_row`;
- explicit wrap events with direction and before/after coordinates;
- visible segments that never connect opposite borders;
- row observation and transition diagnostics;
- runtime, state-count, energy, configuration, and confidence metadata.

The digitization endpoint also returns `curve_trace_segments`. The workspace
uses each segment start as an authoritative pen break. Existing flat
`curve_traces` remain in the response for editing and compatibility.

## Scale Conversion

Scale conversion is deliberately downstream of path decoding. It consumes the
decoder-provided wrap index and never redetects wraps from scaled values.

For a linear track:

```text
value = left + visible_fraction * (right - left)
        + wrap_index * (right - left)
```

For a logarithmic track, each positive wrap multiplies by one full track ratio
and each negative wrap divides by that ratio. This supports positive, negative,
and repeated wraps.

## Fallback Behavior

When `Topology DP` is requested in the web application, any validation or
decode failure is explicit. The response sets `decoder_fallback=true`, records
the exception text in `decoder_fallback_reason`, adds a user-visible warning,
and decodes the same probability evidence with the legacy path. Disabling
Viterbi is also reported as an explicit row-argmax fallback. Topology DP is not
the default.

## Manual Editing

`curve_decoder.editing` provides pure helpers to:

- move one or more visible points;
- add or remove a visual path break;
- add a signed wrap transition;
- remove a wrap transition.

Every edit refreshes unwrapped X, slope, wrap events, and visible segments.
This keeps manual corrections and LAS conversion synchronized.

## Synthetic And Golden Labels

The synthetic generator supports `right_to_left`, `left_to_right`,
`multiple_positive`, `mixed`, `turn_away`, and `border_follow` cases. Wrapped
centerline masks and rendered strokes are split at wrap rows, so labels never
contain an artificial line across the track.

Golden traces store `correct_x_by_row`, `correct_unwrapped_x_by_row`,
`correct_wrap_index_by_row`, topology, and explicit corrected wrap events.
Existing correction captures without wrap labels remain valid bounded cases.

Generate a deterministic wrapped set and convert it to the golden format:

```bash
python -m training.synthetic_log_generator \
  --output-dir evaluation/synthetic_phase3 \
  --count 100 --width 256 --height 512 --seed 3100 \
  --wrap-mode right_to_left

python -m curve_model.golden \
  --synthetic-dir evaluation/synthetic_phase3 \
  --golden-dir evaluation/golden_phase3
```

## Evaluation

Run the five detector/decoder combinations on the same cases:

```bash
python -m curve_decoder.evaluate \
  --golden-dir evaluation/golden_phase3 \
  --output-dir evaluation/phase3_report \
  --phase1-model models/curve_phase1/best.pt \
  --phase2-model models/curve_phase2/best.pt
```

The evaluator writes JSON, CSV, Markdown, row diagnostics, segmented overlays,
and worst-case images. Metrics include unwrapped MAE/P95, wrap precision and
recall with row tolerance, false and missed wraps, wrap-index accuracy,
cross-track connector count, slope error, curvature, runtime, and fallback
counts.

## Promotion Gate And Risks

Phase 3 remains opt-in until a representative manually corrected real-log set
shows improved unwrapped error and wrap recall without increasing false wraps,
grid lock, or runtime beyond the deployment budget. Synthetic smoke results
validate mechanics only.

Known risks are ambiguous border-following ink, very steep transitions beyond
the configured slope bins, long evidence-free spans, incorrect track bounds,
and beam pruning on unusually complex traces. Increase beam size or use exact
mode only for targeted diagnostics; exact state retention is not intended for
full-height production scans.

# Neural Curve Detection Phase 2

## Purpose

Phase 2 adds geometry and connectivity evidence while preserving Classic,
Neural Phase 1, Hybrid Phase 1, and their fallback behavior. It remains an
opt-in prototype. It must not become the default until a representative real
golden set beats the agreed acceptance thresholds.

## Outputs

`CurvePhase2UNet` reuses the Phase 1 encoder and decoder and adds lightweight
heads for:

- `stroke_logits`: complete printed target stroke;
- `centerline_logits`: probable stroke center;
- `distance_field`: normalized smooth ridge, with 1 near the exact center;
- `direction`: normalized `[dx, dy]` tangent vectors;
- `grid_logits`: grid rails, depth lines, and configured track borders.

Direction loss is evaluated only inside the supplied direction tube. Curve and
grid labels are both retained where they cross.

## Synthetic Labels

Generate reproducible version-two data:

```bash
python -m training.synthetic_log_generator \
  --output-dir data/synthetic_logs_v2 \
  --count 10000 --width 512 --height 1024 --seed 1234
```

Each record includes stroke and centerline masks, X by row, distance field,
direction field, valid-direction mask, grid mask, seed, hard-case flag, and all
render/degradation parameters. Hard cases include similar curve/grid colors,
temporary grid following, missing and dashed ink, distractor curves, border
contact, text, low-resolution resampling, and repeated resizing.

## Losses And CAPE

`CurvePhase2Loss` combines stroke/centerline BCE and Dice, Skeleton Recall,
center-weighted Smooth L1 distance regression, masked cosine direction loss,
and weighted grid BCE. Stroke and grid labels have independent validity flags,
so real centerline-only data is not treated as negative stroke/grid truth.

The optional CAPE integration is an isolated differentiable local-path coverage
surrogate. It evaluates centerline support in configured vertical windows and
activates only at `cape.start_epoch`. It is disabled by default and its active
state is logged every epoch and stored in checkpoint configuration. It is not a
claim to reproduce a third-party full CAPE implementation.

## Training And Transfer

```bash
python -m curve_model.phase2_train \
  --phase1-checkpoint models/curve_phase1/best.pt \
  --data-dir data/synthetic_logs_v2 \
  --output-dir models/curve_phase2 \
  --shared-learning-rate 1e-4 \
  --head-learning-rate 5e-4
```

Compatible Phase 1 backbone and stroke/centerline tensors are loaded while the
distance, direction, and grid heads remain newly initialized. Training uses
separate backbone/head learning rates, deterministic splitting, hard-example
weighted sampling, CPU or CUDA, mixed precision on CUDA, resume checkpoints,
per-loss JSON logs, and previews. `--real-data-dir` accepts the golden format;
centerline masks, distance, and direction are derived from corrected X rows.
Missing stroke and grid masks remain explicitly unlabeled. Synthetic/real
sampling ratios and source names are recorded in the checkpoint.

CAPE can be scheduled with:

```bash
python -m curve_model.phase2_train ... --cape --cape-start-epoch 10
```

## Score Fusion And Decoding

`build_phase2_trace_score()` combines calibrated soft centerline, distance,
stroke, and grid evidence. Grid probability is a penalty whose strength is
reduced where center evidence is strong, so crossings remain possible. Optional
skeleton evidence is only a multiplicative bonus; non-skeleton pixels are never
made impossible.

`neural_phase2` sends this score to a Phase 2-specific first-order DP wrapper.
Its transition cost softly rewards agreement with the predicted tangent.
`hybrid_phase2` first adds a soft classic-map contribution. Neither successful
Phase 2 mode runs classic connected-component deletion, binary skeletonization,
black-rail suppression, or black-stroke snapping afterward.

Existing wrapped DP remains available when a curve is already configured as
wrapped, but Phase 2 direction adjustment is then disabled and recorded.
Cylindrical topology and wrap-state decoding are now available as the optional
Phase 3 `topology_dp` path. See `docs/TOPOLOGY_DECODER_PHASE3.md`. The legacy
decoder remains the default.

## Modes And Fallbacks

Configure server-owned checkpoints:

```text
TURBOTIFF_PHASE1_MODEL_PATH=models/curve_phase1/best.pt
TURBOTIFF_PHASE2_MODEL_PATH=models/curve_phase2/best.pt
TURBOTIFF_PHASE2_DEVICE=cpu
```

The workspace selector provides `neural_phase2` and `hybrid_phase2`. A Phase 2
failure tries the corresponding Phase 1 mode, then Classic. Metadata records the
requested mode, actual mode, every fallback reason, checkpoint version,
inference time, fusion configuration, direction-adjustment state, row-level
confidence, and confidence summary.

Phase 2 checkpoints require format version 2, phase 2, all five declared
outputs, and the three added heads. A Phase 1 checkpoint cannot be mistaken for
a Phase 2 checkpoint.

## Evaluation

```bash
python -m curve_model.evaluate \
  --golden-dir evaluation/golden \
  --model models/curve_phase1/best.pt \
  --phase2-model models/curve_phase2/best.pt \
  --output-dir evaluation/phase2
```

The report compares all five modes on the same cases. In addition to Phase 1
error, missing-row, grid-lock, extrema, and timing metrics, it reports major
connectivity gaps, gap lengths and recovery, grid-crossing error, angular
direction error, and thick-stroke center error. JSON, CSV, Markdown, and
worst-error overlays are generated. Aggregate results alone are not sufficient
for promotion; Phase 2 regressions versus Phase 1 must be inspected by case.

## Confidence

Row confidence combines selected center/distance evidence, alternative-path
separation, tangent agreement, grid penalty, neural/classic agreement, and
binary entropy. Results are never silently discarded. The response includes
mean/minimum confidence, low-confidence fraction, and longest low-confidence
run for review prioritization.

## Known Limitations

- No explicit wrap-count or cylindrical state.
- No second-order slope-state Viterbi.
- The optional connectivity term is a lightweight local surrogate.
- Synthetic smoke evaluation verifies plumbing, not production quality.
- Promotion requires a diverse manually corrected real-log golden set.

These outputs and metrics provide the geometric evidence needed for the
slope-aware, wrap-aware Phase 3 decoder without removing earlier modes.

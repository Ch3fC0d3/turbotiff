# Neural Curve Detection Phase 1

## Purpose

Phase 1 adds an isolated learned probability-map prototype without removing
TurboTIFF's classic OpenCV detector or its Viterbi path decoder. The experiment
is successful only if corrected real-log evaluation shows lower centerline X
error and better continuity.

No trained Phase 1 checkpoint is committed. Model paths are server-controlled
through `TURBOTIFF_PHASE1_MODEL_PATH`, and neural failures automatically return
the unmodified classic probability map.

## Synthetic Data

`training.synthetic_log_generator` creates deterministic RGB track crops with:

- smooth, random-walk, sharp-extrema, nearly vertical, and rapid curves;
- black, red, blue, green, and faded gray strokes;
- variable width, dashed ink, faded or missing sections, and border touches;
- linear or logarithmic grids, major/minor rails, text, spots, and stamps;
- brightness variation, blur, noise, speckle, JPEG damage, bleed, rotation,
  shear, and resampling artifacts.

Every sample stores the rendered stroke mask, complete analytical centerline,
one X coordinate per row, seed, and sampled augmentation metadata.

```bash
python -m training.synthetic_log_generator \
  --output-dir data/synthetic_logs \
  --count 10000 --width 512 --height 1024 --seed 1234
```

The generator writes a `manifest.jsonl` plus separate image, mask, centerline,
and metadata directories.

## Model

`curve_model.model.CurvePhase1UNet` is a small RGB U-Net with two independent
one-channel outputs:

- `stroke_logits`: probability of the complete printed stroke;
- `centerline_logits`: probability of the curve centerline.

The default model uses 16 base channels. Training letterboxes samples to a
configurable resolution so batches share a shape without stretching the track.
Production inference does not force a crop into the legacy `128 x 256` shape.
It processes overlapping vertical tiles at native width, pads only to the
network stride, and blends overlaps with feathered weights.

## Losses

`CurveDetectionLoss` combines weighted stroke BCE, stroke Dice, centerline BCE,
centerline Dice, and centerline recall through the predicted stroke. This avoids
optimizing ordinary pixel accuracy on a background-dominated image.

## Training

Install the neural dependencies when they are not already present:

```bash
pip install -r requirements-neural.txt
```

Train locally:

```bash
python -m curve_model.train \
  --data-dir data/synthetic_logs \
  --output-dir models/curve_phase1 \
  --epochs 30 --batch-size 4
```

Training supports deterministic train/validation splits, CPU or CUDA, CUDA
mixed precision, `--resume`, best and latest checkpoints, JSON loss logs,
validation IoU and centerline MAE, and six-panel validation previews. Checkpoint
directories and weights are ignored by Git.

## Inference API

```python
from curve_model import predict_curve_probability

result = predict_curve_probability(track_bgr, "models/curve_phase1/best.pt")
stroke = result["stroke_probability"]
centerline = result["centerline_probability"]
```

Both arrays are `float32 [H, W]` at the original crop size. Loaded models are
cached by path, device, and checkpoint modification time.

## Application Modes

| Mode | Evidence | Decoder | Default |
| --- | --- | --- | --- |
| `classic` | OpenCV color/edge map | Existing DP | Yes |
| `neural_phase1` | Phase 1 stroke + centerline | Existing DP | No |
| `hybrid_phase1` | Phase 1 + classic | Existing DP | No |
| `neural_phase2` | Five-head geometric score | Direction-aware Phase 2 DP | No |
| `hybrid_phase2` | Phase 2 + soft classic contribution | Direction-aware Phase 2 DP | No |

Phase 2 details, checkpoint format, training, and evaluation are documented in
`docs/NEURAL_CURVE_PHASE2.md`.

Each workspace curve has a **Curve Detector** selector:

- `Classic`: the existing `compute_prob_map()` result, unchanged.
- `Neural Phase 1`: `0.70 * centerline + 0.30 * stroke`.
- `Hybrid Phase 1`: normalized neural and classic maps, weighted `0.65 / 0.35`.

All modes feed the existing DP decoder. Weights may be supplied in the curve or
global configuration, while the model path may not. Result
`curve_trace_metadata` records the requested/actual mode, checkpoint, model
version, weights, resolution, duration, and fallback state.

Configure a server:

```text
TURBOTIFF_PHASE1_MODEL_PATH=models/curve_phase1/best.pt
TURBOTIFF_PHASE1_DEVICE=cpu
```

## Golden Evaluation

Convert saved manual-correction captures:

```bash
python -m curve_model.golden \
  --corrections-dir corrections \
  --golden-dir evaluation/golden
```

The golden format stores a track crop, corrected X-by-row array, optional valid
row mask, curve type/color, notes, and source metadata. Real customer data is
ignored by Git.

Compare all three detectors with the same DP decoder:

```bash
python -m curve_model.evaluate \
  --golden-dir evaluation/golden \
  --model models/curve_phase1/best.pt
```

The evaluator writes JSON, CSV, Markdown, and worst-error overlays. Metrics
include mean/median/P90/P95/max X error, accuracy within 1/2/3/5/10 pixels,
missing rows, false constant-X grid-lock runs, peak/valley misses and extras,
probability-map time, decoder time, and total milliseconds per megapixel.

## Limitations And Next Steps

Phase 1 does not predict wrap counts, direction vectors, or signed distance
fields, and it does not replace Viterbi. Synthetic validation is useful for
development but cannot establish superiority. A representative 20-50 case
golden set is required before changing the production default. The full 2D
outputs and tiled inference are intended to support later wrap-aware and richer
geometric models without rewriting the Phase 1 data or evaluation contracts.

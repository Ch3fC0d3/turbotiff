# Real TIFF/LAS training-data curation

`training.real_log_dataset` creates a read-only inventory and a review-gated pilot from an external TIFF/LAS collection. It never edits or copies source logs and never interprets a same-folder association as pixel ground truth.

## Safety gates

Four states remain separate:

1. **Associated pair:** source metadata places the TIFF and LAS under one well identifier.
2. **Audited pair:** TIFF headers and LAS structure, depth, curves, hashes, and units were inspected.
3. **Alignment proposal:** depth control points and track scales can project LAS samples into image pixels.
4. **Training eligible:** a person explicitly approves the alignment and current source hashes still match.

All samples record their split explicitly. Production evaluation must split by well so nearly identical pages from one well cannot leak across training and evaluation.

## Pilot curation and approval

```powershell
python -m training.real_log_dataset `
  --dataset-root D:\Users\gabep\Desktop\TestTiflas\log_pairs_out `
  --output-dir evaluation\real_log_pilot `
  --kgs-count 16 --wvgs-count 4 --hash-files `
  --standalone-tiff D:\Users\gabep\Desktop\TestTiflas\LAVACARIVER_WESTSANDYCREEKUNIT3_MS.tif `
  --standalone-las D:\Users\gabep\Desktop\TestTiflas\LAVACARIVER_WESTSANDYCREEK_UNIT3.las `
  --standalone-well-id LAVACA_RIVER_WEST_SANDY_CREEK_UNIT_3 `
  --alignment evaluation\real_log_pilot\lavaca_alignment_seed.json
```

Approval is explicit and recorded with the reviewer and notes:

```powershell
python -m training.real_log_dataset `
  <same curation arguments> `
  --approve-alignment `
  --reviewer "reviewer name" `
  --review-notes "what was checked"
```

The approved artifact retains TIFF and LAS SHA-256 hashes. Crop generation rejects unapproved alignments and fails if a source hash changes.

## Lavaca result

The mapping uses the printed 1,000-ft line at row 143 and 5,000-ft line at row 4,143. It maps GR, SPDH, M2R2, and M2RX using printed scales. M2RX has distinct 0-20 and 20-200 OHM.M bands; the latter is represented by `M2RX_overrange` rather than wrapping values into the main scale.

The alignment was explicitly approved on 2026-07-20. Raster agreement within three pixels is 97.1% for GR, 100% for SPDH, 100% for M2R2, 69.5% for M2RX on its main scale, and 100% for M2RX on its overrange scale.

## Reproducible GR baseline

`training.real_log_baseline` converts an approved alignment into Phase 1 samples, trains the existing U-Net, and writes hash-linked model provenance:

```powershell
python -m training.real_log_baseline `
  --image D:\Users\gabep\Desktop\TestTiflas\LAVACARIVER_WESTSANDYCREEKUNIT3_MS.tif `
  --las D:\Users\gabep\Desktop\TestTiflas\LAVACARIVER_WESTSANDYCREEK_UNIT3.las `
  --alignment evaluation\real_log_pilot\reviewed_alignment\alignment.json `
  --output-dir evaluation\real_log_pilot\baseline_gr_v1 `
  --track-id GR_main `
  --epochs 50
```

The first run uses 29 non-overlapping GR crops from one well: 15 train crops from 1,000-3,000 ft, 7 validation crops from 3,000-4,000 ft, and 7 test crops from 4,000-5,000 ft. The best model reaches 6.49 px mean absolute error on 896 test rows with no missing predictions. A green-color heuristic reaches 6.13 px but misses 28 rows; the constant baseline reaches 31.49 px and is grid-locked.

This is a pipeline and learnability check, not a production estimate: train, validation, and test are depth blocks from the same well. Cross-well performance requires approving more independent wells and keeping entire wells isolated by split.

## Existing legacy exports

The older `TestTiflas/prepare_training_data.py` and `process_all_pairs.py` assign the entire TIFF height to the LAS depth range, distribute curves evenly across image width, and use observed LAS minima/maxima as track endpoints. Those are configuration heuristics, not verified labels. Prior batch-digitizer outputs are model predictions and must remain pseudo-labels until reviewed.

Before scaling beyond the pilot, verify dataset usage rights, logging-run identity, printed track mnemonics, depth control points, scale direction, logarithmic cycles, and raster/LAS curve agreement.

Historical-pair reuse and its leakage controls are documented in `docs/LEGACY_PAIR_REUSE.md`.

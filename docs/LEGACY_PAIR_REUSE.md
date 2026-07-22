# Reusing historical TIFF/LAS pairs

Historical TurboTIFF test pairs are reusable as training candidates, but their old digitizer predictions and generated configurations are not ground truth.

`training.legacy_pair_ranker` applies the following controls:

- Resolves obsolete image paths by exact well directory and TIFF filename.
- Requires exactly one local LAS file for the resolved TIFF association.
- Uses old configurations only as `automatic_draft` alignment seeds.
- Projects original LAS samples onto the source raster and compares dark-pixel agreement with same-row uniform-X controls.
- Removes long raster grid lines before evidence scoring.
- Hashes selected TIFF and LAS sources.
- Produces GR-only overlay previews for review.
- Prohibits every historically exposed well from becoming a final unbiased test well.
- Never reads old model predictions as labels.

Run the queue builder with:

```powershell
python -m training.legacy_pair_ranker `
  --legacy-configs D:\Users\gabep\Desktop\TestTiflas\temp_train_configs.json `
  --dataset-root D:\Users\gabep\Desktop\TestTiflas\log_pairs_out `
  --output-dir evaluation\legacy_pair_review_queue `
  --required-mnemonic GR `
  --review-count 12
```

The conservative `strong_review_candidate` designation requires at least 50% projected raster hits, a 15 percentage-point lift over controls, and median distance within the scoring radius. It is still not approval. A reviewer must verify logging-run identity, printed depths, track bounds, mnemonic, scale type, and endpoints before `reviewed_approved` can be assigned.

Pairs that fail the evidence gate remain useful source associations. Their alignment must be rebuilt from page-aware depth and track setup rather than inherited from the legacy heuristic.

## First corpus run

The 2026-07-20 run resolved 245 of 500 historical configurations and scored 50 GR candidates. All 12 exported proposals were `weak_legacy_seed`; none passed the strong-evidence gate or became training-eligible. This is a useful rejection result: it prevents the historical configuration heuristic from contaminating the reviewed real-data corpus while retaining the original TIFF/LAS associations for corrected alignment.

A page-aware correction of KGS well `15011232060000` then detected the log body, constrained GR to its printed 0-150 GAPI scale, fitted the blue raster trace, and passed review. Full-resolution agreement is 82.3% within three pixels. Its 35 crops are restricted to diagnostic validation because the well was historically exposed.

The Lavaca-only Phase 1 checkpoint reaches 40.75 px MAE on this independent well, compared with 16.91 px for the blue-raster heuristic. That result establishes the next modeling priority: color/domain augmentation plus additional page-aware approved training wells.

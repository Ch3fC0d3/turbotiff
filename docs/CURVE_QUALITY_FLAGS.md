# Curve quality flags

Flags are multi-valued per sample and remain in project data and provenance companions even when LAS cannot carry them. The Phase 7 vocabulary is:

`ORIGINAL`, `NEURAL_TRACE`, `CLASSIC_FALLBACK`, `MANUAL_CORRECTION`, `RESAMPLED`, `BLENDED`, `INTERPOLATED`, `LOW_CONFIDENCE`, `GRID_LOCK_SUSPECTED`, `SPIKE_SUSPECTED`, `JOIN_BOUNDARY`, `WRAP_EVENT`, `WRAP_ADJUSTED`, `OUT_OF_RANGE`, and `NO_SOURCE_IMAGE`.

Phase 6 currently emits lowercase legacy flags such as `original`, `resampled`, `blended`, and `overlap_conflict`; Phase 7 reads flags case-insensitively where they drive checks. Multiple flags may coexist. Flags describe processing/evidence and do not replace the original curve values or source provenance.

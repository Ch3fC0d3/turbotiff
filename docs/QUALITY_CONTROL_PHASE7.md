# Phase 7 quality control

Phase 7 is an independent gate between `WholeLogResult` assembly and LAS delivery. `evaluate_whole_log_quality()` is deterministic for the same whole-log bytes, metadata, and frozen `QualityControlConfig`; its run ID and finding IDs are content-derived. It does not change curve arrays.

The pipeline checks depth structure and export precision, curve structure/ranges/confidence/provenance, explicit tracing flags, wrap-index topology, page-join review and boundary jumps, metadata, and an internal plus optional `lasio` LAS round trip. Category scores use a geometric mean. A score never overrides a blocker.

Review evidence is obtained lazily with `create_evidence_crop()`. It returns the source-page, source-row, original value, confidence, model/decoder version, wrap offsets, and quality flags for the interval. Raster composition is intentionally left to the UI because Phase 6 provenance does not carry image file paths or alternative paths.

Reports are JSON, CSV, Markdown, and HTML. `write_provenance_companion()` preserves sample-level sources separately from LAS. The implementation does not claim PDF report generation.

Known limitations: deterministic checks are not geological interpretation; image-derived peak support, horizontal-grid following, smoothing comparison, confidence calibration, novelty models, and visual cross-track connector checks require diagnostics not present in the current `WholeLogResult`. LAS writing currently supports unwrapped LAS 2.0 only and rejects other modes. Real-data precision/recall is not claimed until a reviewed frozen corpus is populated.

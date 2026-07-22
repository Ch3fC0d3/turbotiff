# Full-Page Analysis Phase 5

`page_analysis.analyze_well_log_page()` is an optional, deterministic setup
proposal pipeline. It hashes the source, records transforms, proposes page body,
major borders and tracks, and returns schema-versioned structured data. It does
not trace curves or export LAS. Large pages use low-resolution geometry first;
header and curve crops remain addressable in original coordinates.

Automatic drafts are conservative. Missing depth, scale, curve identity, or
wrap topology produces `manual_setup_required`. Manual setup remains unchanged.
Optional OCR/VLM assistance is advisory and must retain image-region evidence.

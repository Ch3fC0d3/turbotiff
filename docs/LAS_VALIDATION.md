# LAS validation

TurboTIFF Phase 7 supports unwrapped LAS 2.0 ASCII. Unsupported LAS versions and ASCII wrapping are rejected explicitly.

Serialization requires a shared depth array. Internal `NaN` becomes the configured `NULL` value (default `-999.25`); infinity and a valid measurement equal to the null sentinel are rejected. The validator checks required sections, declared null, curve count, every ASCII row's column count and numeric values, final newline, STRT/STOP/STEP consistency, and rounded-depth round trip. When `lasio` is installed, its independent result is recorded truthfully in `processing_metadata`.

Draft LAS text contains `TURBOTIFF STATUS: DRAFT UNAPPROVED`. `write_las(..., draft=True)` is required for an unapproved file. Approved output requires an approval record whose QC run and data hash match the current result.

Depth and value precision are recorded with the QC configuration. QC blocks when selected depth precision creates duplicate rows.

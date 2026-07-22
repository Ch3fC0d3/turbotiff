# Multi-Page Assembly Phase 6

`whole_log.assemble_whole_log()` accepts reviewed, immutable page traces and
creates a draft continuous log. It normalizes depth units, proposes global page
order, classifies joins, matches curve identities, reconciles wrap layers,
resamples to one depth grid, and retains source provenance. Page traces are
never modified. Every automatic join begins as `automatic_proposal`; unresolved
joins and high-severity warnings block export readiness.

Automatic assembly is optional and separate from Phase 5 OCR and page tracing.
The current deterministic implementation is intended as a reviewable baseline,
not a replacement for expert assembly on undocumented layouts or logging runs.

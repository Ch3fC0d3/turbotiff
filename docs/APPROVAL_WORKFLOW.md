# Approval workflow

Statuses remain distinct: draft assembly, blocked or needs-review QC, reviewed QC, approved data, export-ready file, and delivered export. QC never creates approval.

1. Run deterministic QC with complete export metadata.
2. Resolve or review findings; any open high/critical finding blocks approval.
3. Review the LAS preview and source evidence.
4. Complete the checklist for depth range/unit, curve identities/units/scales, page order, applicable joins/wraps, critical warnings, LAS metadata, and output preview.
5. Call `approve()` as an `approver`. The immutable record binds the log ID, QC run, export configuration, user, checklist, and whole-log hash.
6. `write_las()` rechecks the approval/QC hashes and validates the exact serialized text.
7. Create a delivery manifest.

Any change to curve data, confidence, flags, provenance, identity, units, or joins changes the whole-log hash. The old approval is then invalid and cannot authorize export. Operator and reviewer roles alone cannot approve.

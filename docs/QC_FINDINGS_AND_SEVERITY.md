# QC findings and severity

Every finding contains a stable ID, category, severity, curve, affected depth interval where applicable, evidence, recommendation, approval-blocking flag, and review state.

- `info`: audit context; normally no action.
- `low`: minor observation.
- `medium`: requires review but does not itself block export.
- `high`: likely data or setup error; blocks approval.
- `critical`: unsafe to approve; blocks approval.

Review states are `open`, `acknowledged`, `corrected`, `accepted_as_real`, `false_positive`, `deferred`, and `not_applicable`. A resolution records reviewer, UTC date, notes, related correction, and replacement QC run. A high/critical item may cease blocking only through an explicit resolving state; acknowledgement and deferral do not clear it. `accepted_as_real` is policy-controlled and remains visible.

Rules identify numerical or trace anomalies, not geological impossibility. Range messages cite the configured curve rule.

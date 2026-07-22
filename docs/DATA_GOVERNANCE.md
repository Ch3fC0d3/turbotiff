# Data Governance

Every correction has one explicit data-use status:

- `training_allowed`: approved records may train models.
- `internal_only`: approved records may train only in the controlled local workflow.
- `evaluation_only`: benchmark use only.
- `client_restricted`: retained for the authorized client workflow, never exported for training.
- `do_not_retain`: no track image asset is written.

Approval and permission are independent; both must permit training. Images are
content-addressed by SHA-256 to avoid duplicate copies. Dataset versions are
immutable directories with canonical content hashes and source record IDs.

Splits group all crops by source TIFF/PDF, log, well, page, or project identity.
Preflight leakage checks compare exact image hashes, source groups, and compact
perceptual hashes against every frozen golden suite. Any match blocks training
until explicitly resolved; it is never silently ignored.

Only necessary provenance should be stored. Client names and free-form source
metadata should be omitted unless operationally required. Dataset and model
registries belong in access-controlled storage and must not be uploaded merely
because a correction was created.

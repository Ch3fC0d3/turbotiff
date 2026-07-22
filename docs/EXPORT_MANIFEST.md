# Export manifests and revisions

Each approved delivery gets a new time-and-randomness-qualified export ID and a JSON manifest. It records log/data/QC/approval IDs, LAS version and SHA-256, companion filenames, creator/time, software commit, model versions, decoder, page-analysis, and assembly versions.

The writer uses an atomic temporary-file replacement and never selects an existing export ID. Preserve delivered LAS and manifest files as an immutable pair. A correction requires a new whole-log hash, QC run, approval, LAS, and export ID; workflow storage may mark the old record superseded without rewriting it.

`python -m quality_control.compare_exports --old OLD.json --new NEW.json --old-las OLD.las --new-las NEW.las` reports manifest metadata changes, approval/hash changes, sample-count changes, changed rows, and maximum numeric change. `validate_manifest()` verifies exact LAS bytes.

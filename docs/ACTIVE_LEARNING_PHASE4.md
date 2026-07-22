# Active Learning Phase 4

Phase 4 is a reviewed-learning workflow, not self-training. `CorrectionStore`
captures the immutable machine prediction, final trace, wrap indexes, validity,
image checksum, inference/preprocessing context, and each edit operation. New
records are pending; a named reviewer must approve them. Rejected and
`needs_more_review` records never export.

Training export also requires `training_allowed` or `internal_only`. It derives
centerline, distance, direction, validity, and wrap labels. It marks stroke and
grid labels unavailable instead of manufacturing negative masks. Phase 4 losses
use explicit availability masks.

Active-learning priority combines uncertainty, detector/path disagreement,
uncertain wraps, topology warnings, and nearest-reviewed-embedding distance.
Review units can point to an interval, wrap, extremum, grid crossing, or gap.
Hard cases remain a distinct sampling source and can be deliberately
oversampled alongside synthetic and corrected-real data.

Commands:

```text
python -m learning.export_corrections --input learning_data/corrections --output datasets --dataset-id real_curves_v1
python -m learning.train_candidate --base-model models/production/current.pt \
  --datasets-root datasets --synthetic-dataset synthetic_v2 \
  --real-dataset real_curves_v1 --hard-dataset hard_cases_v1 \
  --golden-dataset golden_v1 --evaluation-report candidate_vs_production.json \
  --output models/candidates/run_001
```

The command validates every manifest hash and blocks golden leakage before it
trains. It performs availability-aware optimization, records the source mix and
loss for every epoch, requires a frozen candidate-versus-production report,
then registers the new checkpoint as a candidate. Candidate creation never
promotes a model. Frozen real benchmark images are
not currently present in this repository, so real-data accuracy claims require
an externally supplied, authorized golden suite.

# Model Registry

`learning.model_registry.ModelRegistry` stores candidate, production, rejected,
and archived entries in `models/registry.json`. Checkpoints are copied into a
new model-specific location and never overwrite the source checkpoint.

Promotion requires completed evaluation, a clean leakage result, passed
thresholds, a regression report, a named human approver, and a reason. Training
only registers a candidate. Rollback selects a preserved former production
entry without retraining. Every promotion and rollback is appended to history.

```text
python -m learning.promote_model --candidate MODEL_ID --approved-by NAME --reason TEXT --gates gates.json
python -m learning.rollback_model --model OLD_MODEL_ID --approved-by NAME --reason TEXT
```

Production code may call `resolve_production_checkpoint()` and fall back to its
existing configured checkpoint when no registry production pointer exists.
Shadow results always return production as `selected_output`; candidate output
is diagnostic only.

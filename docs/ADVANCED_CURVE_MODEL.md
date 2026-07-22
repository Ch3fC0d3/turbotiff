# Advanced Curve Model Experiment

Phase 4 supplies two prompt-optional families with one output contract:
stroke, centerline, distance, direction, grid, and three-class row wrap logits.
The lightweight family extends the Phase 2 U-Net. The experimental adapter
family adds a high-resolution residual adapter and wider features without a
network download, preserving thin-line skips.

Prompts are an optional fourth image channel. A missing prompt becomes zeros,
so ordinary single-curve inference remains supported. Point, short-line, or
accepted-segment heatmaps can condition multi-curve selection. The topology
decoder remains authoritative for cumulative wrap count and treats wrap logits
as soft transition evidence.

Candidate reports include parameter and trainable-parameter count plus an FP32
size estimate. Benchmark reports must add measured CPU/GPU time, peak memory,
model file size, accuracy by frozen suite, and production-versus-candidate
regressions before promotion. No advanced model has been promoted.

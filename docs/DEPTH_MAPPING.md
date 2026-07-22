# Depth Mapping

Depth OCR stores alternatives with confidence and row location. Sequence
selection favors monotonic candidates; robust fitting rejects large residuals
and retains piecewise-linear `(page_y, depth)` control points. This supports
missing labels and local scanner stretch without assuming constant pixels per
foot. Units and rejected OCR alternatives remain provenance.

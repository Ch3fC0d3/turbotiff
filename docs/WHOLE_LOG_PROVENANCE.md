# Whole-Log Provenance

Each output sample records output depth and every contributing source page,
curve, image row, source depth/value, trace confidence, model/decoder versions,
local wrap index, page wrap offset and final wrap index. Flags distinguish
original, resampled, blended and overlap-conflict samples. Missing coverage has
null value and empty source provenance. Page-level results remain intact.

# Curve Matching Across Pages

Curve identity uses normalized mnemonic aliases, physical unit, track/scale
metadata and overlap shape. Color alone never merges curves. Conflicting units
create separate identities and a critical warning. Different plotted scales
may coexist when both are already converted to compatible physical units.
Original names and styles remain source metadata.

For cyclic curves, candidate page wrap offsets are scored against boundary
physical values. The chosen offset is added to each page-local wrap index; a
low-confidence cyclic offset blocks export.

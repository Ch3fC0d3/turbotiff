# Overlap Alignment

Join relationships are exact continuation, small/large overlap, or small/large
gap using configurable tolerances. Shape comparison uses common-depth
resampling and normalized correlation. Offset and limited linear-stretch models
are supported; excessive stretch is rejected. Agreeing overlap samples may be
confidence-weighted, while disagreement selects the stronger source and adds an
overlap-conflict flag. Large gaps remain null.

"""Human-review warnings from calibrated confidence and decoder diagnostics."""

import numpy as np

def quality_warnings(confidence, wrap_events=(), disagreement=None, novelty=None, low_threshold=.35, minimum_run=8):
    confidence=np.asarray(confidence,dtype=float); warnings=[]; low=confidence<low_threshold; start=None
    for row,value in enumerate(np.r_[low,False]):
        if value and start is None: start=row
        if not value and start is not None:
            if row-start>=minimum_run: warnings.append({"type":"low_confidence","row_start":start,"row_end":row-1})
            start=None
    for event in wrap_events:
        if float(event.get("confidence",1))<.5: warnings.append({"type":"possible_false_wrap","row":event.get("row_after")})
    if disagreement is not None and float(disagreement)>.5: warnings.append({"type":"forward_backward_disagreement"})
    if novelty is not None and float(novelty)>.8: warnings.append({"type":"novel_image_style"})
    return warnings

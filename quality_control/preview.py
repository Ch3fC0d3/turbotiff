from __future__ import annotations
import numpy as np
from .las_validation import serialize_las

def create_las_preview(whole_log,qc,metadata,config,approval=None,row_count=5):
    text=serialize_las(whole_log,metadata,config,approved=approval is not None)
    lines=text.splitlines();data=[line for line in lines[lines.index("~ASCII")+1:] if line.strip()]
    curves=[{"mnemonic":curve.mnemonic,"unit":curve.unit,"description":curve.description,"null_fraction":float(np.mean(~np.isfinite(curve.values)))} for curve in whole_log.curves]
    return {"well_information":dict(metadata),"start":float(whole_log.curves[0].depth[0]),"stop":float(whole_log.curves[0].depth[-1]),"step":float(np.median(np.diff(whole_log.curves[0].depth))),"null":config.null_value,"curves":curves,"first_rows":data[:row_count],"last_rows":data[-row_count:],"qc_status":qc.status,"approval_status":"approved" if approval else "unapproved"}

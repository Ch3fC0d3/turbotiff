from __future__ import annotations
import hashlib,time,cv2,numpy as np
from .models import *
from .orientation import detect_orientation,correction_transform
from .layout import detect_log_body,vertical_separators,group_tracks
from .review import safety_status,to_tracing_requests

def analyze_well_log_page(image,config=None,page_id=None):
    config=config or PageAnalysisConfig(); image=np.asarray(image)
    if image.ndim not in (2,3) or not image.size: raise ValueError("Page image must be non-empty grayscale or BGR")
    started=time.perf_counter(); height,width=image.shape[:2]; digest=hashlib.sha256(image.tobytes()).hexdigest()
    orientation=detect_orientation(image); transform=correction_transform(width,height,orientation["coarse_rotation"],orientation["fine_rotation_degrees"])
    corrected=cv2.warpPerspective(image,transform.original_to_corrected,(width,height),borderValue=255)
    bodies=detect_log_body(corrected); classification="log_data" if bodies and bodies[0].height()>.35*height else ("cover" if np.mean(corrected)<250 else "unknown")
    tracks=[]; separators=[]
    for body in bodies:
        found=vertical_separators(corrected,body,config.minimum_separator_fraction); separators.extend(found); tracks.extend(group_tracks(found,body,config.minimum_track_width))
    critical={"track_bounds":.85 if tracks else 0.,"depth_mapping":0.,"scale_type":0.,"scale_endpoints":0.,"scale_direction":0.,"curve_identity":0.,"wrap_topology":0.}
    result=PageAnalysisResult(page_id or digest[:16],digest,(width,height),(width,height),transform,classification,bodies,[],tracks,[],[],{"orientation":orientation["confidence"],"critical_fields":critical},{"geometry":"phase5-deterministic-v1"},{"orientation":orientation,"separators":separators,"duration_ms":(time.perf_counter()-started)*1000})
    result.processing_metadata["safety_status"]=safety_status(result); return result

__all__=["analyze_well_log_page","PageAnalysisConfig","PageAnalysisResult","CoordinateTransform","BoundingBox","safety_status","to_tracing_requests"]

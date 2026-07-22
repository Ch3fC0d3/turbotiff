from __future__ import annotations
import cv2,numpy as np
from .models import BoundingBox,DetectedTrack,PageAnalysisConfig

def detect_log_body(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image; ink=(gray<245).astype(np.uint8)
    rows=np.flatnonzero(ink.mean(1)>.01); columns=np.flatnonzero(ink.mean(0)>.01)
    if not len(rows) or not len(columns): return []
    return [BoundingBox(float(columns[0]),float(rows[0]),float(columns[-1]+1),float(rows[-1]+1))]

def vertical_separators(image,body,minimum_fraction=.55):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image; crop=gray[int(body.y1):int(body.y2),int(body.x1):int(body.x2)]
    dark=(crop<150).mean(0); candidates=np.flatnonzero(dark>=minimum_fraction); groups=[]
    for x in candidates:
        if not groups or x>groups[-1][-1]+2: groups.append([x])
        else: groups[-1].append(x)
    return [{"x_by_row":float(body.x1+np.mean(group)),"line_type":"border","confidence":float(np.mean(dark[group])),"visible_fraction":float(np.mean(dark[group]))} for group in groups]

def group_tracks(separators,body,minimum_width=40):
    xs=sorted({float(item["x_by_row"]) for item in separators}); tracks=[]
    for left,right in zip(xs,xs[1:]):
        if right-left>=minimum_width:
            tracks.append(DetectedTrack(f"track_{len(tracks)+1}",BoundingBox(left,body.y1,right,body.y2),confidence={"boundaries":min(1.,.5+.5*min(right-left,200)/200)}))
    return tracks

from __future__ import annotations
import cv2, numpy as np
from .models import CoordinateTransform

def detect_orientation(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    edges=cv2.Canny(gray,60,160); lines=cv2.HoughLinesP(edges,1,np.pi/180,max(30,min(gray.shape)//8),minLineLength=min(gray.shape)//4,maxLineGap=20)
    angles=[]
    for line in ([] if lines is None else lines[:,0]):
        x1,y1,x2,y2=line; angles.append(np.degrees(np.arctan2(y2-y1,x2-x1)))
    vertical=sum(abs(abs(a)-90)<15 for a in angles); horizontal=sum(abs(a)<15 or abs(abs(a)-180)<15 for a in angles)
    # Long strip logs are normally taller than wide after coarse correction.
    # Line evidence is retained for confidence/fine skew, while aspect ratio
    # prevents dense horizontal depth grids from defeating the major borders.
    coarse=0 if gray.shape[0]>=gray.shape[1] else 90
    vertical_angles=[a-(90 if a>0 else -90) for a in angles if abs(abs(a)-90)<15]
    fine=float(np.median(vertical_angles)) if vertical_angles else 0.
    confidence=float(max(vertical,horizontal)/max(1,len(angles)))
    return {"coarse_rotation":coarse,"fine_rotation_degrees":fine,"confidence":confidence,"evidence":{"line_count":len(angles)}}

def correction_transform(width,height,coarse_rotation=0,fine_rotation_degrees=0.):
    angle=float(coarse_rotation)+float(fine_rotation_degrees); center=(width/2.,height/2.)
    affine=cv2.getRotationMatrix2D(center,angle,1.); matrix=np.vstack((affine,[0,0,1]))
    return CoordinateTransform.from_matrix(matrix,[{"type":"rotation","degrees":angle,"center":list(center)}])

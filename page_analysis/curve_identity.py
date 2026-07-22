from __future__ import annotations
import cv2,numpy as np
def candidate_curves(image,bounds,maximum=4):
    crop=image[int(bounds.y1):int(bounds.y2),int(bounds.x1):int(bounds.x2)]; hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    saturation=hsv[:,:,1]; candidates=[]
    for index in np.argsort(np.bincount((hsv[:,:,0]//15).ravel(),minlength=12))[::-1]:
        mask=((hsv[:,:,0]//15)==index)&(saturation>55); coverage=float(np.mean(np.any(mask,axis=1)))
        if coverage>.15: candidates.append({"candidate_id":f"curve_{len(candidates)+1}","dominant_hue":int(index*15),"coverage_fraction":coverage,"seed_points":[],"confidence":min(1.,coverage)})
        if len(candidates)>=maximum: break
    return candidates

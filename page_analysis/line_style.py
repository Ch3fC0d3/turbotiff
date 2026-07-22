import numpy as np
def describe_line_sample(crop):
    pixels=np.asarray(crop); dark=np.mean(pixels,axis=2)<220 if pixels.ndim==3 else pixels<220; profile=dark.mean(axis=0); active=profile>.05
    transitions=np.count_nonzero(np.diff(active.astype(np.int8))); style="dashed" if transitions>=4 else "solid"
    color=tuple(int(v) for v in np.median(pixels[dark],axis=0)[::-1]) if pixels.ndim==3 and dark.any() else None
    return {"color_rgb":color,"color_name":None,"thickness_pixels":float(np.max(dark.sum(axis=0))) if dark.any() else None,"dash_pattern":None,"style":style,"confidence":float(min(1.,dark.mean()*10))}

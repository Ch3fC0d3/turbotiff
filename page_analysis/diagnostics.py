import cv2,numpy as np
def render_overlay(image,result):
    output=np.asarray(image).copy()
    for body in result.log_body_regions: cv2.rectangle(output,(round(body.x1),round(body.y1)),(round(body.x2),round(body.y2)),(0,180,0),2)
    for track in result.tracks: cv2.rectangle(output,(round(track.bounds.x1),round(track.bounds.y1)),(round(track.bounds.x2),round(track.bounds.y2)),(0,0,220),2)
    return output

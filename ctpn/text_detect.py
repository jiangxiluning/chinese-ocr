from ctpn.ctpn.model import ctpn
from ctpn.ctpn.detectors import TextDetector
from ctpn.ctpn.other import draw_boxes
import numpy as np
def text_detect(img):
    scores, boxes,img = ctpn(img)
    textdetector  = TextDetector()
    boxes = textdetector.detect(boxes,scores[:, np.newaxis],img.shape[:2])
    text_recs,tmp = draw_boxes(img, boxes, caption='im_name', wait=True,is_display=False)
    return text_recs,tmp,img
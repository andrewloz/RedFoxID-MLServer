# server_utils.py
import numpy as np
from rfdetr.util.coco_classes import COCO_CLASSES

def results_to_proto_boxes(r, pb):
    """
    r: Ultralytics Results (e.g. results[0])
    pb: generated yolo_pb2 module
    normalize: if True, boxes are normalized to [0,1] by (W,H)
    """
    H, W = map(int, r.orig_shape[:2])
    out = pb.ResponsePayload()

    if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
        return out

    names = getattr(r, "names", {}) or {}

    xywh = r.boxes.xywh.detach().cpu().numpy()          
    conf = r.boxes.conf.detach().cpu().numpy()          
    cls  = r.boxes.cls.detach().cpu().numpy().astype(int) 


    for (x, y, w, h), s, c in zip(xywh, conf, cls):
        det = pb.Detection(
            box=pb.BoxXYWH(x=float(x), y=float(y), w=float(w), h=float(h)),
            score=float(s),
            class_id=int(c),
            class_name=names.get(int(c), "")
        )
        out.objects.append(det)

    return out

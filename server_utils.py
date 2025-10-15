import numpy as np
import torch

def results_to_proto_boxes(r, pb):
    """
    r: Ultralytics Results (e.g. results[0])
    pb: generated detect_object_pb2 module
    """
    out = pb.ResponsePayload()

    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out

    # Shape: [N, 6] -> [x, y, w, h, conf, cls]
    t_xywh   = boxes.xywh               # [N,4]
    t_conf   = boxes.conf.unsqueeze(1)  # [N,1]
    t_cls    = boxes.cls.unsqueeze(1)   # [N,1]
    t_all    = torch.cat((t_xywh, t_conf, t_cls), dim=1)
    arr      = t_all.to("cpu", non_blocking=True).numpy()

    names = getattr(r, "names", {}) or {}
    if isinstance(names, dict):
        get_name = names.get
        def class_name_fn(c_int):  # local fn to keep loop tight
            return get_name(c_int, "")
    else:
        def class_name_fn(c_int):
            try:
                return names[c_int]
            except Exception:
                return ""

    Box = pb.BoxXYWH
    Det = pb.Detection

    dets = []
    for x, y, w, h, s, c in arr:
        c_int = int(c)
        dets.append(
            Det(
                box=Box(x=float(x), y=float(y), w=float(w), h=float(h)),
                confidence=float(s),
                class_id=c_int,
                class_name=class_name_fn(c_int),
            )
        )

    out.objects.extend(dets)
    return out

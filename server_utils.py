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


def arrays_to_proto_boxes(boxes_xyxy, scores, class_ids, pb, class_names=None):
    out = pb.ResponsePayload()

    if boxes_xyxy.size == 0:
        return out

    Box = pb.BoxXYWH
    Det = pb.Detection

    for bbox, score, cls_id in zip(boxes_xyxy, scores, class_ids):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w * 0.5
        cy = y1 + h * 0.5

        out.objects.append(
            Det(
                box=Box(x=float(cx), y=float(cy), w=float(w), h=float(h)),
                confidence=float(score),
                class_id=int(cls_id),
                class_name=_class_name(class_names, int(cls_id)),
            )
        )

    return out


def _class_name(class_names, cls_id):
    if class_names is None:
        return ""

    if isinstance(class_names, dict):
        return class_names.get(cls_id, "")

    try:
        return class_names[cls_id]
    except Exception:
        return ""

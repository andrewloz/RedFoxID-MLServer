import openvino as ov
from PIL import Image
import io
import numpy as np   


class OpenvinoInfer:
    def __init__(self):
                # initialise model 
        core = ov.Core()
        self.model = core.compile_model("model/side_plane_model_v26_openvino_model/side_plane_model_v26.xml", "AUTO")
    
    def set_input_shape(self):
        input_port = self.model.input(0)
        input_shape = list(input_port.shape)
         
        elem_type = input_port.element_type
        self.input_name = input_port.any_name

        print(f"input shape: {input_shape}, elemtype: {elem_type}, name: {self.input_name}")
       
        # for now we are only expecting YOLO11 format which typically consists of n,c,h,w our input format is [1,3,640,640]
        if len(input_shape) != 4:
            raise ValueError(f"Unexpected input format {len(input_shape)} got shape {input_shape}")
        N, C, H, W = input_shape

        if C != 3:
            raise ValueError(f"Model expects {C} channels, YOLO usually uses 3.")

        print(N,C,H,W)
     

    def parse_image(self, png_bytes):  
         # Decode PNG bytes to RGB HWC uint8
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")  # (H,W,3) uint8
        arr = np.asarray(img, dtype=np.uint8)
        
        if arr.shape[:2] != (640, 640):
            raise ValueError(f"Expected 640x640, got {arr.shape[:2]}")

        # HWC -> NCHW float32 [0,1]
        x = arr.astype(np.float32) / 255.0                 # (H,W,3) float32
        x = np.transpose(x, (2, 0, 1))[None]               # (1,3,640,640)
        return np.ascontiguousarray(x)                   

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _cxcywh_to_xyxy(self, b):
        # b: (N,4) with [cx, cy, w, h]
        x1 = b[:, 0] - b[:, 2] / 2
        y1 = b[:, 1] - b[:, 3] / 2
        x2 = b[:, 0] + b[:, 2] / 2
        y2 = b[:, 1] + b[:, 3] / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    def _valid_ratio_xyxy(self, boxes, img_w=640, img_h=640):
        # how many boxes look sane: x1<x2, y1<y2 and inside image (with small slack)
        if boxes.size == 0:
            return 0.0
        x1, y1, x2, y2 = boxes.T
        ok = (x2 > x1) & (y2 > y1) & \
             (x1 >= -5) & (y1 >= -5) & (x2 <= img_w+5) & (y2 <= img_h+5)
        return ok.mean()

    def _nms_xyxy(self, boxes, scores, iou_thresh=0.5, topk=300):
        # minimal NMS in NumPy
        if boxes.size == 0:
            return np.empty((0,), dtype=np.int64)
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1][:topk]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_thresh]
        return np.array(keep, dtype=np.int64)

    def decode_yolo_5x8400(self,
        out0, img_wh=(640, 640), conf_thresh=0.25, iou_thresh=0.5, topk_per_stage=300
    ):
        """
        out0: (1, 5, 8400) float32
        returns: boxes_xyxy (M,4), scores (M,)
        """
        assert out0.ndim == 3 and out0.shape[1] == 5, f"Unexpected shape {out0.shape}"
        pred = out0[0].T  # (8400, 5)

        # confidence: use as-is if already in [0,1], else sigmoid
        raw_conf = pred[:, 4]
        if raw_conf.min() >= 0.0 and raw_conf.max() <= 1.0:
            scores = raw_conf
        else:
            scores = self._sigmoid(raw_conf)

        # quick pre-filter to reduce NMS cost
        keep = scores >= conf_thresh
        pred = pred[keep]
        scores = scores[keep]
        if pred.size == 0:
            return np.empty((0,4), np.float32), np.empty((0,), np.float32)

        coords = pred[:, :4]

        # Try two interpretations for the 4 numbers and auto-select the plausible one
        xyxy_as_is = coords.copy()
        xyxy_from_cxcywh = self._cxcywh_to_xyxy(coords)
        r1 = self._valid_ratio_xyxy(xyxy_as_is, *img_wh)
        r2 = self._valid_ratio_xyxy(xyxy_from_cxcywh, *img_wh)
        boxes_xyxy = xyxy_as_is if r1 >= r2 else xyxy_from_cxcywh

        # clip to image bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, img_wh[0])
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, img_wh[1])

        # NMS
        keep_idx = self._nms_xyxy(boxes_xyxy, scores, iou_thresh=iou_thresh, topk=topk_per_stage)
        return boxes_xyxy[keep_idx], scores[keep_idx]

    def predict(
            self,
            source=None,
            **kwargs,
        ):

        print(kwargs['device'])
        self.set_input_shape()
        x = self.parse_image(source)
    
        infer_request = self.model.create_infer_request()
        infer_request.infer({self.input_name: x})
       
        outs = []
        for i, out_port in enumerate(self.model.outputs):
            out_arr = infer_request.get_output_tensor(i).data
            boxes, scores = self.decode_yolo_5x8400(out_arr, img_wh=(640, 640), conf_thresh=0.25, iou_thresh=0.5)
            print("detections:", len(boxes))
            print("first 5:", np.c_[boxes[:5], scores[:5]])

            # print(f"out{i} -> shape: {out_arr.shape}, dtype: {out_arr.dtype}")
            # outs.append(out_arr)

        # print(outs)
   

import openvino as ov
from PIL import Image
import io
import numpy as np


class OpenvinoInfer:
    def __init__(self):
        # initialise model
        # core = ov.Core()
        # self.model = core.compile_model("model/side_plane_model_v26_openvino_model/side_plane_model_v26.xml", "AUTO")
        print("deleteme")

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

    def _cxcywh_to_xyxy(self, b):
        # b: (N,4) with [cx, cy, w, h]
        x1 = b[:, 0] - b[:, 2] / 2
        y1 = b[:, 1] - b[:, 3] / 2
        x2 = b[:, 0] + b[:, 2] / 2
        y2 = b[:, 1] + b[:, 3] / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    def _iou_xyxy(self, box, boxes, area_box, area_boxes):
        # box: (4,), boxes: (M, 4)
        # area_box: scalar, area_boxes: (M,)
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        inter = inter_w * inter_h
        union = area_box + area_boxes - inter + 1e-9
        return inter / union

    def yolo_nms(self, tensor, conf_threshold=0.25, iou_threshold=0.45, top_k=None):
        """
        tensor: (1, 5, N) with [cx, cy, w, h, conf]
        conf: confidence threshold
        iou: IoU threshold for NMS
        top_k: optional max number of detections to consider during NMS

        Returns: boxes (M,4) of format [x1, y1, x2, y2], scores (M,)
        """
        assert tensor.ndim == 3 and tensor.shape[1] == 5, f"Unexpected shape {tensor.shape}"

        p = tensor[0].T  # (N, 5)
        boxes_c = p[:, :4]
        scores = p[:, 4]

        # confidence filter
        mask = scores >= conf_threshold
        if not np.any(mask):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        boxes_xyxy = self._cxcywh_to_xyxy(boxes_c[mask])
        scores = scores[mask]

        # sort by score (desc), optional top_k prefilter
        order = scores.argsort()[::-1]
        if top_k is not None:
            order = order[:top_k]

        # IoU filter
        keep = []
        # Optimization: calculate areas just once for IoU
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            iou = self._iou_xyxy(boxes_xyxy[i], boxes_xyxy[rest], areas[i], areas[rest])
            order = rest[iou <= iou_threshold]

        return boxes_xyxy[keep], scores[keep]

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

        # outs = []
        for i, out_port in enumerate(self.model.outputs):
            out_arr = infer_request.get_output_tensor(i).data
            boxes, scores = self.yolo_nms(out_arr, conf_threshold=0.25, iou_threshold=0.5)
            print("detections:", len(boxes))
            print("first 5:", np.c_[boxes[:5], scores[:5]])

            # print(f"out{i} -> shape: {out_arr.shape}, dtype: {out_arr.dtype}")
            # outs.append(out_arr)

        # print(outs)


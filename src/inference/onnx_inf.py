import onnxruntime as rt
from PIL import Image
import io
import numpy as np

import detect_object_pb2

# IMPORTANT
# currently this onnx inference class is configured to interpret and run RFDETR models
# behaviour using any other type of model is unknown as of now.
class OnnxInfer:
    def __init__(self):
        # initialise model
        self.model = rt.InferenceSession(
            "model/rfdetr.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        # Get input shape
        input_info = self.model.get_inputs()[0]
        self.input_channels, self.input_height, self.input_width = input_info.shape[1:]
        print(self.input_channels, self.input_height, self.input_width)
        self.class_names = ['wheels']


    def preprocess_image(self, png_bytes):
        """Preprocess the input image for inference."""
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB") 
        # Resize the image to the model's input size
        image = image.resize((self.input_width, self.input_height))

        # img[height,width] = 3 channels of rgb [r,g,b]
        # Convert image to numpy array and normalize pixel values
        image = np.array(image).astype(np.float32) / 255.0 # go through each key [Height, Width] and divide each value in [r,g,b] by 255
        # we divide by 255 because the model is trained this way, its trained on rgb values where inputs are 0-1 so we 
        # need to reflect that.
        # HOWEVER, this might be different if using tensorRT or openVINO as the export might bake the RGB calculation into the graph and accept the raw uint8
        # of the original RGB values.

        # Change dimensions from HWC to CHW
        image = np.transpose(image, (2, 0, 1)) 
        # transposes array to the following structure
        # image[0, height, width] = red channel value (divided by 255)
        # image[1, height, width] = green channel value (divided by 255)
        # image[2, height, width] = blue channel value (divided by 255)


        # Add batch dimension
        image = np.expand_dims(image, axis=0) # puts a new dimenion on the axis of 0
        # image[0, 0, height, width] = red channel value (divided by 255)
        # image[0, 1, height, width] = green channel value (divided by 255)
        # image[0, 2, height, width] = blue channel value (divided by 255)

        # now its in the correct format for rfdetr (0, 3, H, W) (batch, channel, height, width)

        return np.ascontiguousarray(image)

    def run_inference(self, image_bytes):
        """Run the model inference and return the raw outputs."""
        # Preprocess the image
        input_image = self.preprocess_image(image_bytes)

        # Get input name from the model
        input_name = self.model.get_inputs()[0].name

        # Run the model
        outputs = self.model.run(None, {input_name: input_image})

        return outputs 

    def post_process(self, outputs, confidence_threshold=0.3, img_w=None, img_h=None, nms_iou_threshold=0.8):
        """
        outputs: [boxes, logits] from RFDETR
            boxes  -> (1, N, 4) in [cx, cy, w, h] normalized 0..1
            logits -> (1, N, C) raw scores per class (not softmaxed)
        confidence_threshold: min class probability to keep a detection
        img_w, img_h: if both given, we'll return pixel coords.
                      if not given, we'll return normalized coords.
        """

        boxes_raw, logits_raw = outputs  # each is a np.ndarray
        boxes = boxes_raw[0]    # (N,4)
        logits = logits_raw[0]  # (N,C)

        # ---- softmax over classes ----
        # subtract max per row for numerical stability
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)  # (N,C)

        # best class + confidence
        best_class_indices = np.argmax(probs, axis=1)       # (N,)
        best_class_confidences = np.max(probs, axis=1)      # (N,)

        # filter by confidence
        keep_mask = best_class_confidences > confidence_threshold
        best_class_indices = best_class_indices[keep_mask]
        best_class_confidences = best_class_confidences[keep_mask]
        boxes = boxes[keep_mask]

        # boxes is [cx, cy, w, h] normalized.
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w  = boxes[:, 2]
        h  = boxes[:, 3]

        x_min = cx - 0.5 * w
        y_min = cy - 0.5 * h
        x_max = cx + 0.5 * w
        y_max = cy + 0.5 * h

        # If img_w/img_h are provided, scale to pixel coords
        if img_w is not None and img_h is not None:
            x_min_pix = x_min * img_w
            y_min_pix = y_min * img_h
            x_max_pix = x_max * img_w
            y_max_pix = y_max * img_h

            bboxes_out = np.stack([x_min_pix, y_min_pix, x_max_pix, y_max_pix], axis=1)
        else:
            # keep normalized [0..1] xyxy
            bboxes_out = np.stack([x_min, y_min, x_max, y_max], axis=1)

        # build final list
        detections = []
        for i in range(len(best_class_indices)):
            detections.append({
                "class": int(best_class_indices[i]),
                "confidence": float(best_class_confidences[i]),
                "bbox": bboxes_out[i].astype(float).tolist()  # [x1,y1,x2,y2]
            })

        return self.nms_detections(detections, iou_threshold=nms_iou_threshold)

    
    def iou_xyxy(self, box_a, box_b):
        """
        box_a, box_b: [x1,y1,x2,y2]
        returns IoU (float 0..1)
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    def nms_detections(self, detections, iou_threshold=0.5):
        """
        detections: list of dicts
            {
                "class": int,
                "confidence": float,
                "bbox": [x1,y1,x2,y2]
            }

        returns: filtered list with NMS applied per class
        """

        # group by class
        by_class = {}
        for det in detections:
            cls = det["class"]
            by_class.setdefault(cls, []).append(det)

        final_kept = []

        for cls, dets in by_class.items():
            # sort by confidence descending
            dets_sorted = sorted(dets, key=lambda d: d["confidence"], reverse=True)

            kept = []
            for det in dets_sorted:
                keep_it = True
                for k in kept:
                    overlap = self.iou_xyxy(det["bbox"], k["bbox"])
                    if overlap > iou_threshold:
                        # too much overlap with a stronger box we already kept
                        keep_it = False
                        break
                if keep_it:
                    kept.append(det)

            final_kept.extend(kept)

        return final_kept


    def detr_detections_to_proto_boxes(self, detections, pb, class_names):
        """
        detections: list of dicts from OnnxInfer.post_process(), after NMS.
            {
                "class": int,
                "confidence": float,
                "bbox": [x1,y1,x2,y2]  # pixel coords
            }

        pb: generated detect_object_pb2 module
        class_names: list like ['wheels', ...]
        """

        out = pb.ResponsePayload()
        Box = pb.BoxXYWH
        Det = pb.Detection

        if not detections:
            return out

        for det in detections:
            cls_id = det["class"]
            conf   = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]

            # convert xyxy -> xywh (center x,y,width,height) like Ultralytics does
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w * 0.5
            cy = y1 + h * 0.5

            out.objects.append(
                Det(
                    box=Box(x=float(cx), y=float(cy), w=float(w), h=float(h)),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=class_names[cls_id] if cls_id < len(class_names) else "",
                )
            )

        return out


    def predict(
        self,
        source=None,
        **kwargs,
    ):
        print(kwargs["device"])
       
        output = self.run_inference(source) 
        detections = self.post_process(output, 0.9, 576, 576, 0.9)
        proto_msg = self.detr_detections_to_proto_boxes(
                detections,
                pb=detect_object_pb2,
                class_names=self.class_names
        )
        
        return proto_msg 

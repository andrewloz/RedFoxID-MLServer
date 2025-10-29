import onnxruntime as rt
from PIL import Image
import io
import numpy as np

import detect_object_pb2

# IMPORTANT
# currently this onnx inference class is configured to interpret and run RFDETR models
# behaviour using any other type of model is unknown as of now.
class RFDETRInfer:
    def __init__(self):
        # initialise model
        self.model = rt.InferenceSession(
            "model/side_plane_model_detr.onnx",
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

    def post_process(self, outputs, confidence_threshold, img_w, img_h):
        """
        So 
        """

        boxes_raw, logits_raw = outputs  # each is a np.ndarray
        boxes = boxes_raw[0]    # (N,4)
        logits = logits_raw[0]  # (N,C)

        wheel_logits = logits[:, 1]
        wheel_probs = self.sigmoid(wheel_logits) # how wheel like is query i
        keep_mask = wheel_probs > confidence_threshold

        boxes_kept = boxes[keep_mask]
        wheel_probs_kept = wheel_probs[keep_mask]

        # boxes is [cx, cy, w, h] normalized.
        cx = boxes_kept[:, 0]
        cy = boxes_kept[:, 1]
        w  = boxes_kept[:, 2]
        h  = boxes_kept[:, 3]

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        x1_pix = x1 * img_w
        y1_pix = y1 * img_h
        x2_pix = x2 * img_w
        y2_pix = y2 * img_h

        # 6. build detections list
        detections = []
        for i in range(len(wheel_probs_kept)):
            detections.append({
                "class": "wheels",
                "confidence": float(wheel_probs_kept[i]),
                "bbox": [
                    float(x1_pix[i]),
                    float(y1_pix[i]),
                    float(x2_pix[i]),
                    float(y2_pix[i]),
                ],  # [x1,y1,x2,y2] in pixels
            })

        return detections

    def sigmoid(self, x):
        z = np.exp(-np.abs(x))
        return np.where(x >= 0, 1 / (1 + z), z / (1 + z))
        
    # def iou_xyxy(self, box_a, box_b):
    #     """
    #     box_a, box_b: [x1,y1,x2,y2]
    #     returns IoU (float 0..1)
    #     """
    #     ax1, ay1, ax2, ay2 = box_a
    #     bx1, by1, bx2, by2 = box_b

    #     inter_x1 = max(ax1, bx1)
    #     inter_y1 = max(ay1, by1)
    #     inter_x2 = min(ax2, bx2)
    #     inter_y2 = min(ay2, by2)

    #     inter_w = max(0.0, inter_x2 - inter_x1)
    #     inter_h = max(0.0, inter_y2 - inter_y1)
    #     inter_area = inter_w * inter_h

    #     area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    #     area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    #     union = area_a + area_b - inter_area
    #     if union <= 0.0:
    #         return 0.0
    #     return inter_area / union

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
                    class_id=1,
                    class_name="wheels",
                )
            )

        return out


    def predict(
        self,
        source=None,
        **kwargs,
    ):
        output = self.run_inference(source) 
        detections = self.post_process(output, 0.9, 576, 576)
        proto_msg = self.detr_detections_to_proto_boxes(
                detections,
                pb=detect_object_pb2,
                class_names=self.class_names
        )

        return proto_msg 

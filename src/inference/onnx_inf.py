from openvino.runtime import Core
from PIL import Image
import io
import numpy as np
import time

import detect_object_pb2

# HEIGHT_WIDTH = 576 # rfdetr medium
HEIGHT_WIDTH = 384 # rfdetr nano


# IMPORTANT
# currently this onnx inference class is configured to interpret and run RFDETR models
# behaviour using any other type of model is unknown as of now.
class RFDETRInfer:
    def __init__(self):
        core = Core()
        # initialise model
        model = core.read_model("model/grid_model_openvino_fp16/inference_model.xml")

        devices = core.get_available_devices()
        print("Available devices:", devices)

        # Compile (this is like optimizing for device)
        # "CPU" always works. "GPU" works if you have Intel GPU + drivers.
        self.compiled_model = core.compile_model(model, "CPU")

        # Get input info
        # OpenVINO models can technically have multiple inputs; we assume 1.
        input_layer = self.compiled_model.input(0)
        shape = list(input_layer.shape)  # e.g. [1,3,HEIGHT_WIDTH,HEIGHT_WIDTH]

        # Sometimes dimension 0 is -1 (dynamic batch). We only care about C,H,W.
        self.input_channels = shape[1]
        self.input_height   = shape[2]
        self.input_width    = shape[3]

        # print(self.input_channels, self.input_height, self.input_width) # this returns 3, 576, 576 this is not right.

        # We'll also hang on to input+output layers for faster lookup in run_inference
        self.input_layer  = input_layer
        self.output_layers = self.compiled_model.outputs

        self.class_names = ['wheels']


    def preprocess_image(self, png_bytes):
        """Preprocess the input image for inference."""
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB") 
        # Resize the image to the model's input size
        image = image.resize((HEIGHT_WIDTH, HEIGHT_WIDTH))

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
        start = time.time()
        input_tensor = self.preprocess_image(image_bytes)
        print(f"image preprocessing: {int(round((time.time() - start) * 1000))}")

        start = time.time()
        # Get input name from the model
        result_tensors = self.compiled_model([input_tensor])
        print(f"inference took: {int(round((time.time() - start) * 1000))}")

        # At this point:
        #   result_tensors[0] is a np.ndarray
        #   result_tensors[1] is a np.ndarray
        #
        # self.output_layers[i] is the *descriptor* that tells you the name of each output.
        # We use that to line them up as [boxes, logits] for post_process.

        # print("DEBUG result types:",
        # type(result_tensors[0]),
        # type(result_tensors[1]))

        # print("DEBUG result shapes:",
        #     result_tensors[0].shape,
        #     result_tensors[1].shape)

        outs_by_name = {}
        for i, out_desc in enumerate(self.output_layers):
            out_name = out_desc.get_any_name() # dets and labels
            outs_by_name[out_name] = result_tensors[i]

        boxes_raw  = outs_by_name["dets"]    # (1,300,4)
        logits_raw = outs_by_name["labels"]  # (1,300,2)

        # Return in the exact shape predict/post_process expects.
        return [boxes_raw, logits_raw]

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

    def _save_annotated_image(self, png_bytes, detections, save_path):
        """
        Draw the detections on the image and write out a PNG file.
        """

        from PIL import ImageDraw, ImageFont

        # open original image and force same size you used for post_process (HEIGHT_WIDTHxHEIGHT_WIDTH)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        img = img.resize((HEIGHT_WIDTH, HEIGHT_WIDTH))

        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class"]
            conf = det["confidence"]

            # box
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=(255, 0, 0),
                width=2,
            )

            # label text
            label = f"{cls_name} {conf:.2f}"

            tb = draw.textbbox((0, 0), label, font=font)
            text_w = tb[2] - tb[0]
            text_h = tb[3] - tb[1]

            bg_tl = (x1, y1 - text_h - 4)
            bg_br = (x1 + text_w + 4, y1)

            draw.rectangle([bg_tl, bg_br], fill=(255, 0, 0))
            draw.text(
                (x1 + 2, y1 - text_h - 2),
                label,
                fill=(255, 255, 255),
                font=font,
            )

        img.save(save_path, format="PNG")


    def predict(
        self,
        source=None,
        **kwargs,
    ):
        # imgsz=(640, 640),
        # # imgsz=(img.shape[0], img.shape[1]),
        # conf=request.confidence_threshold,
        # iou=request.iou_threshold,
        # verbose=bool(int(self.config.get("Verbose", "0"))),
        # save=bool(int(self.config.get("SaveImg", "0"))),
        # project="./output",
        # name=request.image_name, # this stops subdirectories being created, when save is true
        # device=self.config.get("Device", "") # you will want to change this to match your hardware

        print("START PREDICT")

        output = self.run_inference(source)

        # https://github.com/roboflow/rf-detr
        # architecture has specific image size requirements

        start = time.time()
        detections = self.post_process(output, kwargs['conf'], HEIGHT_WIDTH, HEIGHT_WIDTH)
        print(f"post_processing took: {int(round((time.time() - start) * 1000))}")


        start = time.time()
        proto_msg = self.detr_detections_to_proto_boxes(
                detections,
                pb=detect_object_pb2,
                class_names=self.class_names
        )
        print(f"building proto took: {int(round((time.time() - start) * 1000))}")


        if kwargs.get("save"):
            start = time.time()
            base_name = kwargs.get("name", "prediction")
            save_path = f"{kwargs.get('project')}/{base_name}_annotated.png"

            self._save_annotated_image(
                png_bytes=source,
                detections=detections,
                save_path=save_path,
            )
            print(f"saving image took: {int(round((time.time() - start) * 1000))}")


        print("END PREDICT")

        return proto_msg 

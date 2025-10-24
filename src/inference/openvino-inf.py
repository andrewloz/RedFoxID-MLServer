import openvino as ov
from PIL import Image
import io
import numpy as np

if __name__ == "__main__":
    
    with open("input/test.png", "rb") as f:
        png_bytes = f.read() 
        

    # initialise model 
    core = ov.Core()
    model = core.compile_model("model/side_plane_model_v26_openvino_model/side_plane_model_v26.xml", "AUTO")
    input_port = model.input(0)
    input_shape = list(input_port.shape)

    
    elem_type = input_port.element_type
    name = input_port.any_name

    print(f"input shape: {input_shape}, elemtype: {elem_type}, name: {name}")
   
    # for now we are only expecting YOLO11 format which typically consists of n,c,h,w our input format is [1,3,640,640]
    if len(input_shape) != 4:
        raise ValueError(f"Unexpected input format {len(input_shape)} got shape {input_shape}")
    N, C, H, W = input_shape

    if C != 3:
        raise ValueError(f"Model expects {C} channels, YOLO usually uses 3.")
   
    
    # TODO: this needs to be converted to process png img byte array.
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(img)

    # Decode PNG bytes to RGB HWC uint8
    arr = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))  # (H,W,3) uint8
    assert arr.shape[:2] == (640, 640), f"Expected 640x640, got {arr.shape[:2]}"
    # HWC -> CHW -> NCHW, and normalize to [0,1] float32
    x = np.transpose(arr, (2, 0, 1))[None].astype(np.float32) / 255.0   # (1,3,640,640)
    x = np.ascontiguousarray(x)

    print(N,C,H,W)
    infer_request = model.create_infer_request()
    infer_request.infer({name: x})
   
    outs = []
    for i, out_port in enumerate(model.outputs):
        out_arr = infer_request.get_output_tensor(i).data
        print(f"out{i} -> shape: {out_arr.shape}, dtype: {out_arr.dtype}")
        outs.append(out_arr)

    print(outs)



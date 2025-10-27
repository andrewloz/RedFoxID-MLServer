import numpy as np
from src.inference.openvino_inf import OpenvinoInfer

def test_yolo_nms():
    # Create a dummy tensor: (1, 5, N)
    # Format: [cx, cy, w, h, conf]
    tensor = np.zeros((1, 5, 4), dtype=np.float32)
    tensor[0, :, 0] = [100, 100, 50, 50, 0.9]  # High confidence
    tensor[0, :, 1] = [105, 105, 50, 50, 0.8]  # Overlapping, lower confidence
    tensor[0, :, 2] = [300, 300, 50, 50, 0.7]  # Far away
    # 4rth is low confidence, should be filtered out

    ov_inf = OpenvinoInfer()
    boxes, scores = ov_inf.yolo_nms(tensor, conf_threshold=0.5, iou_threshold=0.5)

    # Should keep the first and third box only
    assert boxes.shape == (2, 4)
    assert scores.shape == (2,)
    assert np.all(scores > 0.5)
    print("test_yolo_nms passed.")

if __name__ == "__main__":
    test_yolo_nms()

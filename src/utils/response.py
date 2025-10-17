# what sort of structure or order do we want to recieve data in.
# do we recieve another header that specifies the field name?
# and we just map the message after that content header to a predict dict
# of somekind. 

# look up what a binary frame is, with length prefixes, avoid text, json and base64
import struct

PACKET_HEADER_FORMAT = '<IBBBB'
DETECTIONS_MSG = 2

DETECTIONS_META_FORMAT = '<iiIIBB'
DETECTIONS_META_SIZE = struct.calcsize(DETECTIONS_META_FORMAT)

DETECTION_FORMAT = '<fffffHH'  # 24 bytes
DETECTION_SIZE = struct.calcsize(DETECTION_FORMAT)

class Response:
    def __init__(self):
        #  structure for response package will be
        #  PACKET HEADER - packet_length, msg_type, rsv1, rsv2, rsv3
        #  META - image_width, image_height, inference_time, detection_count, image_name_length, model_name_length
        #  DETECTION_RECORDS[] - x1, y1, x2, y2, confidence, class_id, rsv1
        pass

    def build_detections_packet(width, height, infer_ms, image_name, model_name, detections):
        # detections: iterable of (x_min, y_min, x_max, y_max, confidence, class_id)
        name_b = image_name.encode('utf-8')
        model_b = model_name.encode('utf-8')
        if len(name_b) > 255 or len(model_b) > 255:
            raise ValueError("image_name/model_name must be <= 255 bytes")

        meta = struct.pack(DETECTIONS_META_FORMAT, width, height, infer_ms,
                        len(detections), len(name_b), len(model_b))

        det_bytes = bytearray()
        for x1, y1, x2, y2, conf, cls_id in detections:
            det_bytes += struct.pack(DETECTION_FORMAT, float(x1), float(y1), float(x2), float(y2),
                                    float(conf), int(cls_id) & 0xFFFF, 0)

        body = meta + name_b + model_b + det_bytes
        packet_length = PACKET_HEADER_SIZE + len(body)
        header = struct.pack(PACKET_HEADER_FORMAT, packet_length, DETECTIONS_MSG, 0, 0, 0)
        return header + body
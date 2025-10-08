from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BoxXYWH(_message.Message):
    __slots__ = ("x", "y", "w", "h")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    w: float
    h: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., w: _Optional[float] = ..., h: _Optional[float] = ...) -> None: ...

class Detection(_message.Message):
    __slots__ = ("box", "confidence", "class_id", "class_name")
    BOX_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    CLASS_ID_FIELD_NUMBER: _ClassVar[int]
    CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
    box: BoxXYWH
    confidence: float
    class_id: int
    class_name: str
    def __init__(self, box: _Optional[_Union[BoxXYWH, _Mapping]] = ..., confidence: _Optional[float] = ..., class_id: _Optional[int] = ..., class_name: _Optional[str] = ...) -> None: ...

class ResponsePayload(_message.Message):
    __slots__ = ("objects",)
    OBJECTS_FIELD_NUMBER: _ClassVar[int]
    objects: _containers.RepeatedCompositeFieldContainer[Detection]
    def __init__(self, objects: _Optional[_Iterable[_Union[Detection, _Mapping]]] = ...) -> None: ...

class RequestPayload(_message.Message):
    __slots__ = ("image_rgba_bytes", "image_width", "image_height", "confidence_threshold", "iou_threshold", "image_name")
    IMAGE_RGBA_BYTES_FIELD_NUMBER: _ClassVar[int]
    IMAGE_WIDTH_FIELD_NUMBER: _ClassVar[int]
    IMAGE_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    IOU_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    IMAGE_NAME_FIELD_NUMBER: _ClassVar[int]
    image_rgba_bytes: bytes
    image_width: int
    image_height: int
    confidence_threshold: float
    iou_threshold: float
    image_name: str
    def __init__(self, image_rgba_bytes: _Optional[bytes] = ..., image_width: _Optional[int] = ..., image_height: _Optional[int] = ..., confidence_threshold: _Optional[float] = ..., iou_threshold: _Optional[float] = ..., image_name: _Optional[str] = ...) -> None: ...

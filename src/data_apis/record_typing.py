#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import TypedDict, Annotated, Literal


class Record(TypedDict):
    token: Annotated[str, "Token"]


class Timestamp(TypedDict):
    timestamp: Annotated[int, "Timestamp"]


class ChainElement(TypedDict):
    prev: Annotated[str, "Token of the previous"]
    next: Annotated[str, "Token of the next"]


class SampleRecord(Record, Timestamp, ChainElement):
    scene_token: Annotated[str, "Foreign key linking to scene"]
    data: Annotated[dict, ...]
    anns: Annotated[list[str], ...]


class SampleDataRecord(Record, Timestamp, ChainElement):
    sample_token: Annotated[str, ...]
    ego_pose_token: Annotated[str, ...]
    calibrated_sensor_token: Annotated[str, ...]
    fileformat: Annotated[str, ...]
    is_key_frame: Annotated[bool, bool]
    weight: Annotated[int, ...]
    width: Annotated[int, ...]
    filename: Annotated[str, ...]
    sensor_modality: Annotated[str, ...]
    channel: Annotated[str, ...]


class SampleAnnotationRecord(Record, ChainElement):
    sample_token: Annotated[str, ...]
    instance_token: Annotated[str, ...]
    visibility_token: Annotated[str, ...]
    attribute_tokens: Annotated[list[str], ...]
    translation: Annotated[tuple[float], ...]
    size: Annotated[tuple[float], ...]
    rotation: Annotated[tuple[[float]], ...]
    num_lidar_pts: Annotated[int, ...]
    num_radar_pts: Annotated[int, ...]


class CalibratedSensorRecord(Record):
    sensor_token: Annotated[str, ...]
    translation: Annotated[tuple[float], ...]
    rotation: Annotated[tuple[[float]], ...]
    camera_intrinsic: Annotated[list[list[float]], ..., list()]


class SensorRecord(TypedDict):
    channel: Annotated[str, ...]
    modality: Annotated[str, Literal['camera', 'lidar', 'radar']]


class MapRecord(Record):
    category: Annotated[str, ...]
    filename: Annotated[str, ...]
    log_tokens: Annotated[list[str], ...]


class CategoryRecord(Record):
    name: Annotated[str, ...]
    description: Annotated[str, ...]


class InstanceRecord(Record):
    category_token: Annotated[str, ...]
    nbr_annotations: Annotated[str, ...]
    first_annotation_token: Annotated[str, ...]
    last_annotation_token: Annotated[str, ...]


class EgoPoseRecord(Record, Timestamp):
    translation: Annotated[tuple[float], ...]
    rotation: Annotated[tuple[[float]], ...]

from typing import NamedTuple

import numpy as np


class CameraModel(NamedTuple):
    model_id: int
    model_name: str
    num_params: int


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


class Camera(NamedTuple):
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    fov_x: np.ndarray | float
    fov_y: np.ndarray | float
    depth_params: dict | None
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

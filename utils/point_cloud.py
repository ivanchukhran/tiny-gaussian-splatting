from typing import NamedTuple

import numpy as np


class Point3D(NamedTuple):
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: np.ndarray | float
    image_ids: np.ndarray
    point_2d_indexes: np.ndarray


class BasicPointCloud(NamedTuple):
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray

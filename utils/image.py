from typing import NamedTuple, Tuple

import numpy as np

from .math import qvec2rotmat


class BaseImage(NamedTuple):
    id: str
    qvec: Tuple[float, float, float, float]
    tvec: Tuple[float, float, float]
    camera_id: int
    name: str
    xys: np.ndarray
    point_3d_ids: np.ndarray


class Image(BaseImage):
    def qvec2rotmap(self):
        return qvec2rotmat(self.qvec)

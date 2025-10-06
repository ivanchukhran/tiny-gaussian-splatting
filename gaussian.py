from dataclasses import dataclass
from typing import NamedTuple

import torch

from utils.activation import inverse_sigmoid
from utils.math import covariance_from_scaling_rotation


@dataclass
class GaussianTrainParams:
    position_lr: float
    spatial_lr: float
    feature_lr: float
    opacity_lr: float
    scaling_lr: float
    rotation_lr: float
    position_lr_init: float


class GaussianTrainableParamDict(NamedTuple):
    params: list[torch.Tensor]
    lr: float
    name: str


class GaussianModel:
    _xyz: torch.Tensor
    _features_dc: torch.Tensor
    _features_rest: torch.Tensor
    _opacity: torch.Tensor
    _scaling: torch.Tensor
    _rotation: torch.Tensor
    spatial_lr_scale: float

    def __init__(self):
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._opacity = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self.spatial_lr_scale = 0

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def trainable_params(self, params: GaussianTrainParams):
        w = [
            GaussianTrainableParamDict(
                params=[self._xyz],
                lr=params.position_lr_init * self.spatial_lr_scale,
                name="xyz",
            ),
            GaussianTrainableParamDict(
                params=[self._features_dc], lr=params.feature_lr, name="f_dc"
            ),
            GaussianTrainableParamDict(
                params=[self._features_rest], lr=params.feature_lr / 20.0, name="f_rest"
            ),
            GaussianTrainableParamDict(
                params=[self._opacity], lr=params.opacity_lr, name="opacity"
            ),
            GaussianTrainableParamDict(
                params=[self._scaling], lr=params.scaling_lr, name="scaling"
            ),
            GaussianTrainableParamDict(
                params=[self._rotation], lr=params.rotation_lr, name="rotation"
            ),
        ]

        return w

    @property
    def scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def xyz(self):
        return self._xyz

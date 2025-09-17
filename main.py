import json
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

# from sympy.tensor.array.dense_ndim_array import List,
from sympy.utilities.misc import struct
from typing_extensions import NamedTuple


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


def rotation(r, device: str | torch.device = "cpu"):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def scaling_rotation(s, r, device: str | torch.device = "cpu"):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=device)
    R = rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def covariance_from_scaling_rotation(
    scaling, scaling_modifier, rotation, device: str | torch.device = "cpu"
):
    L = scaling_rotation(scaling_modifier * scaling, rotation, device=device)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_lowerdiag(actual_covariance, device=device)
    return symm


def strip_lowerdiag(l: torch.Tensor, device: str | torch.device = "cpu"):
    uncertainty = torch.zeros((l.shape[0], 6), dtype=torch.float, device=device)

    uncertainty[:, 0] = l[:, 0, 0]
    uncertainty[:, 1] = l[:, 0, 1]
    uncertainty[:, 2] = l[:, 0, 2]
    uncertainty[:, 3] = l[:, 1, 1]
    uncertainty[:, 4] = l[:, 1, 2]
    uncertainty[:, 5] = l[:, 2, 2]
    return uncertainty


# def strip_symmetric(sym: torch.Tensor, device: str | torch.device = 'cpu'):
#     return strip_lowerdiag(sym, device=device)


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


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


def find_latest_iteration(path: str) -> int | None:
    return None


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


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


def read_extrinsics_binary(path: str):
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = binary_image_properties[1:5]
            tvec = binary_image_properties[5:8]
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points_2d = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points_2d,
                format_char_sequence="ddq" * num_points_2d,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point_3d_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point_3d_ids=point_3d_ids,
            )
    return images


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


def read_intrinsics_binary(path: str):
    cameras = {}
    with open(path, "rb") as fid:
        camera_properties = read_next_bytes(
            fid, num_bytes=24, format_char_sequence="iiQQ"
        )
        camera_id = camera_properties[0]
        model_id = camera_properties[1]
        selected_model = CAMERA_MODEL_IDS[model_id]
        model_name = selected_model.model_name
        width = camera_properties[2]
        height = camera_properties[3]
        num_params = selected_model.num_params
        params = read_next_bytes(
            fid, num_bytes=num_params * 8, format_char_sequence="d" * num_params
        )
        cameras[camera_id] = Camera(
            id=camera_id,
            model=model_name,
            width=width,
            height=height,
            params=np.array(params),
        )
    return cameras


def read_colmap_scene_info(
    path: str,
    images_dir: str,
    depth_dir: str,
    eval: bool,
    train_test_exp,
    llffhold=8,
):
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    depth_params = None
    try:
        with open(depth_params_file, "r") as f:
            depth_params = json.load(f)
        all_scales = np.array([depth_params[key]["scale"] for key in depth_params])
        med_scale = (
            np.median(all_scales[all_scales > 0]) if (all_scales > 0).sum() else 0
        )
        for key in depth_params:
            depth_params[key]["med_scale"] = med_scale
    except FileNotFoundError:
        print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
        sys.exit(1)
    except Exception as e:
        print(
            f"An unexpected error occurred when trying to open depth_params.json file: {e}"
        )
        sys.exit(1)

    if eval:
        if llffhold:
            camera_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            camera_names = sorted(camera_names)
            test_camera_names_list = [
                name for idx, name in enumerate(camera_names) if idx % llffhold == 0
            ]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), "r") as file:
                test_camera_names_list = file.read().splitlines()
    else:
        test_camera_names_list = []

    camera_infos_unsorted = read_colmap_cameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        depths_params=depth_params,
        images_folder=os.path.join(path, images_dir),
        depth_folder=os.path.join(path, depth_dir),
        test_camera_names_list=test_camera_names_list,
    )
    camera_infos = sorted(camera_infos_unsorted.copy(), key=lambda x: x.image_name)

    train_camera_infos = [c for c in camera_infos if train_test_exp or not c.is_test]
    test_camera_infos = [c for c in camera_infos if c.is_test]

    nerf_normalization = gen_nerf_pp_norm(train_camera_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    pcd = None

    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except Exception:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        print("Could not fetch ply")

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_camera_infos,
        test_cameras=test_camera_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False,
    )
    return scene_info


@dataclass
class ModelConfig:
    scenes_path: str = "scenes"
    checkpoint_path: str = "checkpoints"


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        gaussians: GaussianModel,
        config: ModelConfig,
        load_iteration: int | None = None,
        shuffle=True,
    ):
        self.gaussians = gaussians
        self.train_cameras = {}
        self.test_cameras = {}

        self.load_iteration = load_iteration

        if load_iteration:
            if load_iteration == -1:
                self.load_iteration = find_latest_iteration(config.checkpoint_path)

        # Just supports sparse Colmap scenes
        scene_infe = read_colmap_scene_info()

    def get_train_cameras(self, scale: float = 1.0):
        return self.train_cameras.get(scale, None)

    def get_test_cameras(self, scale: float = 1.0):
        return self.test_cameras.get(scale, None)


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    device: str | torch.device = "cpu",
):
    pass


def train(scene: Scene):
    viewpoint_stack = scene.get_train_cameras()
    viewpoint_indices = list(range(len(viewpoint_stack)))

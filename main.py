import json
import os
from dataclasses import dataclass

import numpy as np
import torch


def find_latest_iteration(path: str) -> int | None:
    return None


def camera2json(id: int, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.fov_y, camera.height),
        "fx": fov2focal(camera.fov_x, camera.width),
    }
    return camera_entry


@dataclass
class ModelConfig:
    source_path: str
    image_path: str
    depth_path: str
    eval: bool
    train_test_exp: bool
    model_path: str
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
        self.config = config
        self.train_cameras = {}
        self.test_cameras = {}

        self.load_iteration = load_iteration

        if load_iteration:
            if load_iteration == -1:
                self.load_iteration = find_latest_iteration(config.checkpoint_path)

        # Just supports sparse Colmap scenes
        scene_info = read_colmap_scene_info(
            config.source_path,
            config.image_path,
            config.depth_path,
            config.eval,
            config.train_test_exp,
        )

        if not self.load_iteration:
            with (
                open(scene_info.ply_path, "rb") as src_file,
                open(
                    os.path.join(self.config.model_path, "input.ply"), "wb"
                ) as dest_file,
            ):
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera2json(id, cam))
            with open(
                os.path.join(self.config.model_path, "cameras.json"), "w"
            ) as file:
                json.dump(json_cams, file)

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

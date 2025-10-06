import math

import numpy as np
import torch


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


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def world2view(r, t, translate: np.ndarray | None = None, scale: float = 1.0):
    if translate is None:
        translate = np.array([0.0, 0.0, 0.0])
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = r.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    c2w = np.linalg.inv(Rt)
    cam_center = c2w[:3, 3]
    cam_center = (cam_center + translate) * scale
    c2w[:3, 3] = cam_center
    Rt = np.linalg.inv(c2w)
    return np.float32(Rt)


def get_nerf_pp_norm(cam_info):
    def get_center_diagonal(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=1)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        w2c = world2view(cam.R, cam.T)
        c2w = np.linalg.inv(w2c)
        cam_centers.append(c2w[:3, 3:4])

    center, diagonal = get_center_diagonal(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}

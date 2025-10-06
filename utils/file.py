import json
import os
import sys
from io import BufferedReader

import numpy as np
from plyfile import PlyData, PlyElement
from sympy.utilities.misc import struct

from .camera import CAMERA_MODEL_IDS, Camera, CameraInfo
from .image import Image
from .math import focal2fov, get_nerf_pp_norm, qvec2rotmat
from .point_cloud import BasicPointCloud, Point3D
from .scene import SceneInfo


def read_next_bytes(
    readable: BufferedReader, num_bytes, format_char_sequence, endian_character="<"
):
    data = readable.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_extrinsics_binary(path: str):
    images = {}
    with open(path, "rb") as file:
        num_images = read_next_bytes(file, 8, "Q")[0]
        for _ in range(num_images):
            binary_image_properties = read_next_bytes(
                file, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = binary_image_properties[1:5]
            tvec = binary_image_properties[5:8]
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(file, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(file, 1, "c")[0]
            num_points_2d = read_next_bytes(
                file, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                file,
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


def read_colmap_cameras(
    cam_extrinsics,
    cam_intrinsics,
    depths_params,
    images_folder,
    depths_folder,
    test_cam_names_list,
):
    camera_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write(f"Processing camera {idx + 1}/{len(cam_extrinsics)}\r")
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            fov_y = focal2fov(focal_length_x, height)
            fov_x = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            fov_y = focal2fov(focal_length_y, height)
            fov_x = focal2fov(focal_length_x, width)
        else:
            raise ValueError(f"Unsupported camera model: {intr.model}")

        n_remove = len(extr.name.split(".")[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except Exception:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = (
            os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png")
            if depths_folder != ""
            else ""
        )

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            fov_y=fov_y,
            fov_x=fov_x,
            depth_params=depth_params,
            image_path=image_path,
            image_name=image_name,
            depth_path=depth_path,
            width=width,
            height=height,
            is_test=image_name in test_cam_names_list,
        )
        camera_infos.append(cam_info)

    return camera_infos


def read_points_3d_binary(path: str):
    points_3d = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point_2d_indexes = np.array(tuple(map(int, track_elems[1::2])))
            points_3d[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point_2d_indexes=point_2d_indexes,
            )

    return points_3d


def read_points3D_text(path: str):
    points_3d = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and not line.startswith("#"):
                elems = line.split()
                point_3d_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point_2d_indexes = np.array(tuple(map(int, elems[9::2])))
                points_3d[point_3d_id] = Point3D(
                    id=point_3d_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point_2d_indexes=point_2d_indexes,
                )
    return points_3d


def save_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def load_ply(path: str):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


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
        depths_folder=os.path.join(path, depth_dir),
        test_cam_names_list=test_camera_names_list,
    )
    camera_infos = sorted(camera_infos_unsorted.copy(), key=lambda x: x.image_name)

    train_camera_infos = [c for c in camera_infos if train_test_exp or not c.is_test]
    test_camera_infos = [c for c in camera_infos if c.is_test]

    nerf_normalization = get_nerf_pp_norm(train_camera_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    pcd = None

    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points_3d_binary(bin_path)
        except Exception:
            xyz, rgb, _ = read_points3D_text(txt_path)
        save_ply(ply_path, xyz, rgb)

    pcd = load_ply(ply_path)
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_camera_infos,
        test_cameras=test_camera_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False,
    )
    return scene_info

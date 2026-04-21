# adapted from https://github.com/buaacyw/MeshAnythingV2/blob/main/data_process.py
# originally for sampling point clouds from meshes
# We add a function to define the visibility of the point cloud
import argparse
import datetime
import json
import logging
import math
import multiprocessing
import os
import random
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Pool
import objaverse

import numpy as np
import nvdiffrast.torch as dr
from natsort import natsorted

# import objaverse
import open3d as o3d
import torch
import tqdm
import trimesh
from compute_visibility import *
from pytorch3d.renderer import cameras as pt3d_cameras
from torchvision.transforms.functional import to_pil_image

from mesh_to_pc import export_to_watertight

CPU_COUNT = multiprocessing.cpu_count()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def prepare_args():
    parser = argparse.ArgumentParser("PartialMesh.")

    parser.add_argument(
        "--dataset_processor",
        type=str,
        # default="ShapeNet",
        default="PCN",
        choices=["ShapeNet", "Objaverse", "PCN"],
        help="Dataset processor to use.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/ShapeNet/ShapeNetCore",
        # default="data/Objaverse_v1",
        # default='data/huggingface_cache/hub/datasets--allenai--objaverse/blobs',
        help="Base directory containing the dataset.",
    )
    parser.add_argument(
        "--min_face_num",
        type=int,
        default=0,
        help="Minimum number of faces for filtering.",
    )
    parser.add_argument(
        "--max_face_num",
        type=int,
        default=1600,
        help="Maximum number of faces for filtering.",
    )  # 1600 in meshanything v2
    parser.add_argument(
        "--out_dir",
        type=str,
        # default='data/ShapeNet_processed_1600_no_watertight',
        # default="data/Objaverse_processed_1600_no_watertight",
        default="data/PCN_processed",
        help="Output directory to save processed data.",
    )
    # parser.add_argument("--test_length", type=int, required=True, help="Number of samples for the test set.")
    parser.add_argument(
        "--do_visibility",
        type=int,
        default=1,  # 1 for True, 0 for False
        help="Whether to compute visibility.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=4,
        help="Number of views for visibility computation.",
    )
    parser.add_argument(
        "--num_elev",
        type=int,
        default=4,
        help="Number of elevations for visibility computation.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image size for visibility computation.",
    )
    parser.add_argument(
        "--fov_deg",
        type=float,
        default=120.0,
        help="Field of view in degrees for visibility computation.",
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=1,  # 1 for True, 0 for False
        help="Whether to visualize the processed data.",
    )
    parser.add_argument(
        "--do_watertight", action="store_true", help="Whether to use water tight mesh."
    )
    parser.add_argument(
        "--num_sample_points",
        type=int,
        default=16384,
        help="Number of points to sample from the water mesh.",
    )
    args = parser.parse_args()

    return args


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


# a base class for processing mesh datasets
# this class can be extended to implement specific dataset processing logic
class MeshDatasetProcessor:
    def __init__(self, args):
        self.dataset_name = args.dataset_processor
        print(f"Using dataset: {self.dataset_name}")
        self.args = args
        self.base_dir = args.base_dir
        self.min_face_num = args.min_face_num
        self.max_face_num = args.max_face_num
        self.out_dir = args.out_dir
        self.processed_dir = os.path.join(self.out_dir, "processed")
        self.npz_out_dir = os.path.join(self.out_dir, "npz")
        self.pc_out_dir = os.path.join(self.out_dir, "pc")
        self.final_data_save_dir = os.path.join(self.out_dir, "final")
        # self.test_length = args.test_length
        self.do_visibility = args.do_visibility
        self.visualize = args.visualize
        self.do_watertight = args.do_watertight
        self.num_sample_points = args.num_sample_points

        seed_everything(args.seed)

        if self.do_visibility:
            self.image_size = args.image_size
            self.fov_deg = args.fov_deg
            self.num_views = args.num_views
            self.num_elev = args.num_elev
            print("Calculating camera intrinsics...")
            fov_rad = math.radians(self.fov_deg)
            focal_length = (self.image_size / 2) / math.tan(fov_rad / 2)
            cx = cy = self.image_size / 2
            intrinsic = torch.tensor(
                [[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
            intr = intrinsic[None]  # Add batch dimensions
            # generate the visibility camera parameters
            num_views = self.num_views
            num_elev = self.num_elev

            print(f"Generating {num_views} views with {num_elev} elevations each...")
            clip_interval = 360 // num_views
            elev_interval = 120 // (num_elev - 1) if num_elev > 1 else 0

            azim_offset = 45
            azim_list = []
            elev_list = []
            look_at_center_list = []
            look_at_center_default = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
            for i in range(num_views):
                azim = (
                    float(i * clip_interval) + azim_offset + np.random.uniform(-15, 15)
                )
                for j in range(num_elev):
                    # range from -120 / 2 to 120 / 2
                    elev = float(j * elev_interval) - 120.0 / 2 if num_elev > 1 else 0.0
                    azim_list.append(azim)
                    elev_list.append(elev)
                    # randomly offset the look at center between -0.1 and 0.1
                    look_at_center_offset = torch.tensor(
                        [
                            np.random.uniform(-0.1, 0.1),
                            np.random.uniform(-0.1, 0.1),
                            np.random.uniform(-0.1, 0.1),
                        ],
                        dtype=torch.float32,
                    )
                    look_at_center = look_at_center_default + look_at_center_offset
                    look_at_center_list.append(look_at_center)
            azim_list = torch.tensor(azim_list, dtype=torch.float32)
            elev_list = torch.tensor(elev_list, dtype=torch.float32)
            look_at_center_list = torch.stack(look_at_center_list, dim=0)

            # use pytorch3d to generate the camera extrinsics
            Rs, Ts = pt3d_cameras.look_at_view_transform(
                dist=1.0, elev=elev_list, azim=azim_list, at=look_at_center_list
            )
            # stack the camera extrinsics
            extrs = torch.cat(
                [Rs, Ts[..., None]], dim=-1
            )  # shape: (num_views * num_elev, 3, 4)
            self.extrs = extrs[None]  # Add batch and frame dimensions
            N = self.extrs.shape[1]  # Number of views
            # add frame dimension to intrinsics so it is compatible with extrinsics (B, 3, 3) -> (B, F, 3, 3)
            self.intrs = intr[None].repeat(1, N, 1, 1)

            # self.extrs = self.extrs.cuda() if torch.cuda.is_available() else self.extrs
            # self.intrs = self.intrs.cuda() if torch.cuda.is_available() else self.intrs

        os.makedirs(self.out_dir, exist_ok=True)

        self.json_save_path = os.path.join(
            self.out_dir,
            f"filtered_min_{self.min_face_num}_max_{self.max_face_num}.json",
        )

    def filt_objects(self):
        """
        Filter meshes based on face count and save the filtered list to a JSON file.
        This method should be overridden by subclasses to implement specific filtering logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def process_objects(self):
        """
        Process a single mesh data item.
        This method should be overridden by subclasses to implement specific processing logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def process(self):
        """
        Process a single mesh data item.
        This method should be overridden by subclasses to implement specific processing logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class ShapeNetProcessor(MeshDatasetProcessor):
    def __init__(self, args):
        super().__init__(args)

        self.idx_uid_mapping = {}  # mapping from index to uid

    def filt_objects(self):
        """
        Filter ShapeNet objects based on face count and save the filtered list to a JSON file.
        """

        filtered_paths = []
        print(
            f"Filtering ShapeNet objects from {self.base_dir} with face count between {self.min_face_num} and {self.max_face_num}.."
        )
        if os.path.exists(self.json_save_path):
            print(f"File {self.json_save_path} already exists. Skipping filtering.")
            with open(self.json_save_path, "r") as f:
                filtered_paths = json.load(f)
                filtered_paths.sort()
                self.idx_uid_mapping = {
                    i: path for i, path in enumerate(filtered_paths)
                }
            return
        for synset_id in os.listdir(self.base_dir):
            synset_path = os.path.join(self.base_dir, synset_id)
            if not os.path.isdir(synset_path):
                continue
            for model_id in os.listdir(synset_path):
                model_dir = os.path.join(synset_path, model_id, "models")
                obj_path = os.path.join(model_dir, "model_normalized.obj")
                if os.path.exists(obj_path):
                    try:
                        mesh = trimesh.load(obj_path, force="mesh")
                        if (
                            hasattr(mesh, "faces")
                            and self.min_face_num
                            <= mesh.faces.shape[0]
                            <= self.max_face_num
                            and mesh.vertices.shape[0] > 300
                        ):  # NOTE mesh.vertices.shape[0] > 300?
                            filtered_paths.append(obj_path)
                    except Exception as e:
                        logging.warning(f"Skipping {obj_path}: {e}")
        logging.info(f"Filtered {len(filtered_paths)} valid meshes.")
        # Save the filtered paths to a JSON file
        with open(self.json_save_path, "w") as f:
            json.dump(filtered_paths, f)
        filtered_paths.sort()
        self.idx_uid_mapping = {i: path for i, path in enumerate(filtered_paths)}

    def merge_and_scale(self, idx, cur_data, progress, start_time, total_tasks):
        """
        Merge and scale the mesh data, save it to a temporary file, and update progress.
        This method should be overridden by subclasses to implement specific merging logic.
        """
        try:
            idx = str(idx)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                task_start_time = time.time()
                print(f"Processing {idx}...")

                uid = "_".join(
                    cur_data.strip("/").split("/")[-4:-1]
                )  # synset_id/model_id
                # print('uid',uid)
                # write a temp file
                if os.path.exists(os.path.join(self.out_dir, idx)):
                    # check if another file with different uid exists
                    existing_files = os.listdir(os.path.join(self.out_dir, idx))
                    # search for uid.txt file
                    if "uid.txt" in existing_files:
                        with open(os.path.join(self.out_dir, idx, "uid.txt"), "r") as f:
                            existing_uid = (
                                f.read().splitlines()[0].strip()
                            )  # remove newline characters
                        if existing_uid != uid:
                            # abort
                            raise Exception(
                                f"UID mismatch: {existing_uid} != {uid}. Please check the data."
                            )
                        else:
                            if os.path.exists(
                                os.path.join(self.out_dir, idx, "model_normalized.obj")
                            ) and os.path.exists(
                                os.path.join(self.out_dir, idx, uid + ".npz")
                            ):
                                print("Files already exist. Skipping.")
                                return

                os.makedirs(os.path.join(self.out_dir, idx), exist_ok=True)

                if os.path.exists(
                    os.path.join(self.out_dir, idx, uid + ".tmp")
                ) or os.path.exists(os.path.join(self.out_dir, idx, uid + ".npz")):
                    return

                # copy the mesh data to the processed directory
                shutil.copy(
                    cur_data, os.path.join(self.out_dir, idx, "model_normalized.obj")
                )
                mesh = trimesh.load(cur_data, force="mesh")

                npz_to_save = {}
                if (
                    hasattr(mesh, "faces")
                    and mesh.faces.shape[0] >= self.min_face_num
                    and mesh.faces.shape[0] <= self.max_face_num
                ):
                    mesh.merge_vertices()
                    mesh.update_faces(mesh.nondegenerate_faces())
                    mesh.update_faces(mesh.unique_faces())
                    mesh.remove_unreferenced_vertices()

                    # judge = True
                    vertices = np.array(mesh.vertices.copy())
                    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])  # type: ignore
                    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
                    vertices = (
                        vertices / (bounds[1] - bounds[0]).max()
                    )  # -0.5 to 0.5 length 1 # The mesh is scaled here
                    vertices = vertices.clip(-0.5, 0.5)

                    cur_mesh = trimesh.Trimesh(
                        vertices=vertices, faces=mesh.faces.copy()
                    )

                    min_length = (
                        cur_mesh.bounding_box_oriented.edges_unique_length.min()
                    )

                    npz_to_save["vertices"] = mesh.vertices
                    npz_to_save["faces"] = mesh.faces
                    npz_to_save["min_length"] = min_length
                    npz_to_save["uid"] = uid
                    npz_to_save["vertices_num"] = mesh.vertices.shape[0]
                    npz_to_save["faces_num"] = mesh.faces.shape[0]
                if w:
                    for warn in w:
                        logging.warning(f" {uid} : {str(warn.message)}")
                        print("----------------------------------------")
                        print("uid warning:", uid)
                    return
                # save pc_normal
                print(f"Saving {uid} to {self.out_dir}/{idx}...")
                np.savez(os.path.join(self.out_dir, idx, uid + ".npz"), **npz_to_save)
                # save the uid to a txt file
                with open(os.path.join(self.out_dir, idx, "uid.txt"), "w") as f:
                    f.write(uid)

                # os.remove(temp_path)
                task_end_time = time.time()
                task_duration = task_end_time - task_start_time
                progress.value += 1
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / progress.value * total_tasks
                remaining_time = estimated_total_time - elapsed_time
                remaining_td = datetime.timedelta(seconds=int(remaining_time))
                logging.info(
                    f"This task: {task_duration:.2f} s, Already:{elapsed_time}, progress: {progress.value}/{total_tasks}, remaining{remaining_td}"
                )
        except Exception as e:
            logging.error(f"Error in {uid}: {e}")

    def process_objects(self):
        """
        Process a single mesh data item from ShapeNet.
        """
        with open(self.json_save_path, "r") as f:
            all_paths = json.load(f)

        cpu_count = os.cpu_count()
        print(f"CPU count: {cpu_count}")
        total_tasks = len(all_paths)

        manager = Manager()
        progress = manager.Value("i", 0)
        start_time = time.time()

        # test_idx = 0
        # test_data = self.idx_uid_mapping[test_idx]
        # self.merge_and_scale(test_idx, test_data, progress, start_time, total_tasks)

        # test_indices = [0, 921, 1842, 2763, 3684, 4605]  # just for testing

        # for test_idx in test_indices:
        #     test_data = self.idx_uid_mapping[test_idx]
        #     self.merge_and_scale(test_idx, test_data, progress, start_time, total_tasks)

        with Pool(processes=CPU_COUNT) as pool:
            pool.starmap_async(
                self.merge_and_scale,
                [
                    (idx, data, progress, start_time, total_tasks)
                    for idx, data in self.idx_uid_mapping.items()
                ],
            )
            pool.close()
            pool.join()

    def process_npz_file(self, npz_metadata):
        try:
            #  get the root folder
            idx, cur_data = npz_metadata
            uid = "_".join(cur_data.strip("/").split("/")[-4:-1])  # synset_id/model_id

            if os.path.exists(os.path.join(self.out_dir, idx, f"idx_{idx}.npz")):
                print(
                    f"File {os.path.join(self.out_dir, idx, f'idx_{idx}.npz')} already exists. Skipping."
                )
                return

            npz_data = os.path.join(self.out_dir, idx, uid + ".npz")
            with np.load(npz_data) as data:
                data_dict = {key: data[key] for key in data}
                print(
                    "data_dict vertices shape in process_npz_file:",
                    data_dict["vertices"].shape,
                )
                if data_dict and data["faces"].shape[0] >= 20:
                    data_dict["faces_num"] = data["faces"].shape[0]
                    data_dict["vertices_num"] = data["vertices"].shape[0]
                    data_dict["uid"] = uid
                    # save the npz file to the processed directory
                    npz_out_path = os.path.join(self.out_dir, idx, f"idx_{idx}.npz")
                    np.savez(npz_out_path, **data_dict)
                else:
                    pass
        except Exception as e:
            print(f"Error loading {npz_data}: {e}")

    def to_npz_files(self):
        """
        Convert processed mesh files to npz format, separating into train and test sets.
        """

        # get the npz files
        npz_files = [(str(idx), obj) for idx, obj in self.idx_uid_mapping.items()]

        # test_indices = [0, 921, 1842, 2763, 3684, 4605]  # just for testing
        # for test_idx in test_indices:
        #     test_data = npz_files[test_idx]
        #     self.process_npz_file(test_data)

        with ThreadPoolExecutor() as executor:
            futures = []
            for metadata in tqdm.tqdm(npz_files, desc="Processing files"):
                # npz_path = os.path.join(self.processed_dir, filename)
                futures.append(executor.submit(self.process_npz_file, metadata))

            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="Processing results"
            ):
                future.result()  # Wait for each future to complete

    def process_mesh(self, metadata):
        idx, cur_data = metadata

        mesh_data = os.path.join(self.out_dir, idx, f"idx_{idx}.npz")
        mesh_data = np.load(mesh_data, allow_pickle=True)
        vertices = mesh_data["vertices"]

        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max()
        cur_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=mesh_data["faces"],
            force="mesh",
            merge_primitives=True,
        )
        cur_mesh.merge_vertices()
        cur_mesh.update_faces(cur_mesh.nondegenerate_faces())
        cur_mesh.update_faces(cur_mesh.unique_faces())
        cur_mesh.remove_unreferenced_vertices()
        # import pdb; pdb.set_trace()

        if self.do_watertight:
            water_mesh = export_to_watertight(cur_mesh, 7)

            water_p, face_idx = water_mesh.sample(
                self.num_sample_points, return_index=True
            )  # 128 * 128
            water_n = water_mesh.face_normals[face_idx]

            npz_to_save = {}
            pc_normal = np.concatenate([water_p, water_n], axis=-1, dtype=np.float16)
            npz_to_save["pc_normal"] = pc_normal
            npz_to_save["face_idx"] = face_idx

            # also save the water mesh
            water_mesh.export(os.path.join(self.out_dir, idx, f"{idx}_water_mesh.obj"))
            np.savez(os.path.join(self.out_dir, idx, f"{idx}_pc.npz"), **npz_to_save)
            gt_mesh = trimesh.Trimesh(
                vertices=mesh_data["vertices"],
                faces=mesh_data["faces"],
                force="mesh",
                merge_primitives=True,
            )
            gt_mesh.export(os.path.join(self.out_dir, idx, f"{idx}_gt_mesh.obj"))
        else:
            sample_p, face_idx = cur_mesh.sample(
                self.num_sample_points, return_index=True
            )
            sample_n = cur_mesh.face_normals[face_idx]
            npz_to_save = {}
            pc_normal = np.concatenate([sample_p, sample_n], axis=-1, dtype=np.float16)
            npz_to_save["pc_normal"] = pc_normal
            npz_to_save["face_idx"] = face_idx

            np.savez(os.path.join(self.out_dir, idx, f"{idx}_pc.npz"), **npz_to_save)
            gt_mesh = trimesh.Trimesh(
                vertices=mesh_data["vertices"],
                faces=mesh_data["faces"],
                force="mesh",
                merge_primitives=True,
            )
            gt_mesh.export(os.path.join(self.out_dir, idx, f"{idx}_gt_mesh.obj"))

    def compute_visibility(self):
        npz_files = [(str(idx), obj) for idx, obj in self.idx_uid_mapping.items()]

        self.extrs = self.extrs.cuda() if torch.cuda.is_available() else self.extrs
        self.intrs = self.intrs.cuda() if torch.cuda.is_available() else self.intrs

        # test_indices = [0, 921, 1842, 2763, 3684, 4605]  # just for testing
        # test_indices = [10531]  # just for testing
        # for test_idx in test_indices:
        #     test_data = npz_files[test_idx]
        #     self.generate_visibility_samples(test_data)

        for metadata in tqdm.tqdm(npz_files, desc="Processing files"):
            self.generate_visibility_samples(metadata)

    def generate_depthmaps(self, metadata):
        idx, cur_data = metadata
        # load the mesh data
        mesh_data = os.path.join(self.out_dir, idx, f"idx_{idx}.npz")
        mesh_data = np.load(mesh_data, allow_pickle=True)
        vertices = mesh_data["vertices"]

        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max()
        gt_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=mesh_data["faces"],
            force="mesh",
            merge_primitives=True,
        )
        gt_mesh.merge_vertices()
        gt_mesh.update_faces(gt_mesh.nondegenerate_faces())
        gt_mesh.update_faces(gt_mesh.unique_faces())
        gt_mesh.remove_unreferenced_vertices()

        # obj_path = os.path.join(self.out_dir, idx, 'model_normalized.obj')
        # gt_mesh = trimesh.load(obj_path, force='mesh')
        vertices = torch.tensor(gt_mesh.vertices, dtype=torch.float32, device="cuda")
        faces = torch.tensor(gt_mesh.faces, dtype=torch.int32, device="cuda")

        vertices = vertices[None]
        faces = faces[None]

        rast_space_dict = {}
        glctx = dr.RasterizeCudaContext()  # Create the OpenGL context for nvdiffrast
        rast_space_dict["tri_verts"] = faces
        rast_space_dict["glctx"] = glctx

        res = compute_depthmap(
            vertices,
            self.intrs,
            self.extrs,
            rast_space_dict,
            self.image_size,
            self.image_size,
        )
        os.makedirs(os.path.join(self.out_dir, idx, "depth_maps"), exist_ok=True)
        for i, (depth_vis, depth, mask) in enumerate(res):
            depth_map_vis = to_pil_image(depth_vis)
            depth_map_vis.save(
                os.path.join(self.out_dir, idx, "depth_maps", f"depth_map_{i}.png")
            )
            torch.save(
                depth, os.path.join(self.out_dir, idx, "depth_maps", f"depth_{i}.pt")
            )
            torch.save(
                mask, os.path.join(self.out_dir, idx, "depth_maps", f"mask_{i}.pt")
            )

    def generate_visibility_samples(self, metadata):
        idx, cur_data = metadata
        # load the mesh data
        if self.do_watertight:
            water_mesh_sampled = np.load(
                os.path.join(self.out_dir, idx, f"{idx}_pc.npz"), allow_pickle=True
            )
            face_idx = water_mesh_sampled["face_idx"]
            water_p = water_mesh_sampled["pc_normal"][:, :3]
            water_n = water_mesh_sampled["pc_normal"][:, 3:]
        else:
            mesh_sampled = np.load(
                os.path.join(self.out_dir, idx, f"{idx}_pc.npz"), allow_pickle=True
            )
            face_idx = mesh_sampled["face_idx"]
            water_p = mesh_sampled["pc_normal"][:, :3]
            water_n = mesh_sampled["pc_normal"][:, 3:]

        # gt_mesh = trimesh.load(
        #     os.path.join(self.out_dir, idx, f"{idx}_gt_mesh.obj"), force="mesh"
        # )
        mesh_data = os.path.join(self.out_dir, idx, f"idx_{idx}.npz")
        mesh_data = np.load(mesh_data, allow_pickle=True)
        vertices = mesh_data["vertices"]

        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max()
        gt_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=mesh_data["faces"],
            force="mesh",
            merge_primitives=True,
        )
        gt_mesh.merge_vertices()
        gt_mesh.update_faces(gt_mesh.nondegenerate_faces())
        gt_mesh.update_faces(gt_mesh.unique_faces())
        gt_mesh.remove_unreferenced_vertices()

        gt_points_list = []
        gt_normals_list = []
        visible_points_list = []
        visible_normals_list = []
        invisible_points_list = []
        invisible_normals_list = []
        gt_mesh_list = []  # remember to save the original mesh, not the watertight mesh

        # obj_path = os.path.join(self.out_dir, idx, 'model_normalized.obj')
        # gt_mesh = trimesh.load(obj_path, force='mesh')
        if self.do_watertight:
            water_mesh = trimesh.load(
                os.path.join(self.out_dir, idx, f"{idx}_water_mesh.obj"), force="mesh"
            )
            vertices = torch.tensor(
                water_mesh.vertices, dtype=torch.float32, device="cuda"
            )
            faces = torch.tensor(water_mesh.faces, dtype=torch.int32, device="cuda")
        else:
            vertices = torch.tensor(
                gt_mesh.vertices, dtype=torch.float32, device="cuda"
            )
            faces = torch.tensor(gt_mesh.faces, dtype=torch.int32, device="cuda")

        vertices = vertices[None]
        faces = faces[None]

        rast_space_dict = {}
        glctx = dr.RasterizeCudaContext()  # Create the OpenGL context for nvdiffrast
        rast_space_dict["tri_verts"] = faces
        rast_space_dict["glctx"] = glctx

        visible_face_ids_per_view = compute_visibility(
            vertices, self.intrs, self.extrs, rast_space_dict
        )
        # all outer faces:
        merged_faces = torch.unique(torch.cat(visible_face_ids_per_view)).cpu().numpy()
        # all_visible_face_count = merged_faces.shape[0]
        all_visible_face_count = faces.shape[1]

        for i in range(len(visible_face_ids_per_view)):
            visible_face_ids = visible_face_ids_per_view[i].cpu().numpy()

            mask = np.isin(face_idx, visible_face_ids)
            visible_points = water_p[mask]
            visible_normals = water_n[mask]
            # get invisible points as well
            invisible_points = water_p[~mask]
            invisible_normals = water_n[~mask]

            if len(visible_points) < 0.1 * self.num_sample_points:
                print("Too limited visible points for view", i, "in", idx)
                # continue
            # transform the points and normals to egocentric coordinates
            extr_i = self.extrs[0, i].cpu().numpy()  # w_T_c
            # get the inverse of the extrinsic matrix
            extr_i_w = np.eye(4)
            extr_i_w[:3, :3] = extr_i[:3, :3].T
            extr_i_w[:3, 3] = -np.dot(extr_i[:3, :3].T, extr_i[:3, 3])

            # transform visible points and normals
            visible_points = np.dot(visible_points, extr_i_w[:3, :3]) + extr_i[:3, 3]
            visible_normals = np.dot(visible_normals, extr_i_w[:3, :3])
            # transform invisible points and normals
            invisible_points = (
                np.dot(invisible_points, extr_i_w[:3, :3]) + extr_i[:3, 3]
            )
            invisible_normals = np.dot(invisible_normals, extr_i_w[:3, :3])

            # make the points centered around the origin
            center = visible_points.mean(axis=0)
            visible_points -= center
            invisible_points -= center

            visible_points_list.append(visible_points)
            visible_normals_list.append(visible_normals)
            invisible_points_list.append(invisible_points)
            invisible_normals_list.append(invisible_normals)

            # create structured gt: first N visible points, then invisible points
            gt_water_p = np.concatenate([visible_points, invisible_points], axis=0)
            gt_water_n = np.concatenate([visible_normals, invisible_normals], axis=0)
            gt_points_list.append(gt_water_p)
            gt_normals_list.append(gt_water_n)
            # corresponding gt mesh
            # transform the gt mesh vertices to the camera coordinate
            gt_mesh_vertices = (
                np.dot(gt_mesh.vertices, extr_i_w[:3, :3]) + extr_i[:3, 3]
            )

            # this is the dummy center purely for visualization
            # gt mesh is shifted and scale in training, as seen in meshanything_train/loop_set_256.py line 181-219
            # But the pc is shifted to another center
            gt_mesh_vertices = (
                gt_mesh_vertices - center
            )  # make the points centered around the origin
            gt_mesh_faces = gt_mesh.faces.copy()
            # create a trimesh object for the gt mesh
            gt_mesh_i = trimesh.Trimesh(
                vertices=gt_mesh_vertices,
                faces=gt_mesh_faces,
                force="mesh",
                merge_primitives=True,
            )
            gt_mesh_list.append(gt_mesh_i)

        print(f"Done processing {idx}, save to {self.out_dir}/{idx}")
        if self.visualize:
            os.makedirs(
                os.path.join(self.out_dir, idx, "pc_visualization"), exist_ok=True
            )
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(water_p)
            pcd.normals = o3d.utility.Vector3dVector(water_n)
            o3d.io.write_point_cloud(
                os.path.join(self.out_dir, idx, "pc_visualization", f"{idx}_pc.ply"),
                pcd,
            )

            for i in range(len(gt_points_list)):
                gt_points = gt_points_list[i]
                gt_normals = gt_normals_list[i]
                visible_points = visible_points_list[i]
                visible_normals = visible_normals_list[i]
                invisible_points = invisible_points_list[i]
                invisible_normals = invisible_normals_list[i]
                gt_mesh_i = gt_mesh_list[i]

                gt_pc_normal = np.concatenate(
                    [gt_points, gt_normals], axis=-1, dtype=np.float16
                )
                pc_normal = np.concatenate(
                    [visible_points, visible_normals], axis=-1, dtype=np.float16
                )
                invisible_pc_normal = np.concatenate(
                    [invisible_points, invisible_normals], axis=-1, dtype=np.float16
                )
                gt_npz_to_save = {
                    "gt_pc_normal": gt_pc_normal,
                }
                npz_to_save = {
                    "pc_normal": pc_normal,
                }
                invisible_npz_to_save = {
                    "invisible_pc_normal": invisible_pc_normal,
                }
                np.savez(
                    os.path.join(self.out_dir, idx, f"view{i}_gt_pc.npz"),
                    **gt_npz_to_save,
                )
                np.savez(
                    os.path.join(self.out_dir, idx, f"view{i}_pc.npz"), **npz_to_save
                )
                np.savez(
                    os.path.join(self.out_dir, idx, f"view{i}_invisible_pc.npz"),
                    **invisible_npz_to_save,
                )

                if self.visualize:
                    # save the point cloud and normals
                    # os.makedirs(os.path.join(self.out_dir, idx, "pc_visualization"), exist_ok=True)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(visible_points)
                    pcd.normals = o3d.utility.Vector3dVector(visible_normals)
                    o3d.io.write_point_cloud(
                        os.path.join(
                            self.out_dir, idx, "pc_visualization", f"view{i}_pc.ply"
                        ),
                        pcd,
                    )
                    # save the gt point cloud and normals
                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(gt_points)
                    pcd_gt.normals = o3d.utility.Vector3dVector(gt_normals)
                    o3d.io.write_point_cloud(
                        os.path.join(
                            self.out_dir, idx, "pc_visualization", f"view{i}_gt_pc.ply"
                        ),
                        pcd_gt,
                    )
                    # save the invisible point cloud and normals
                    pcd_invisible = o3d.geometry.PointCloud()
                    pcd_invisible.points = o3d.utility.Vector3dVector(invisible_points)
                    pcd_invisible.normals = o3d.utility.Vector3dVector(
                        invisible_normals
                    )
                    o3d.io.write_point_cloud(
                        os.path.join(
                            self.out_dir,
                            idx,
                            "pc_visualization",
                            f"view{i}_invisible_pc.ply",
                        ),
                        pcd_invisible,
                    )

                # save the gt mesh
                gt_mesh_i.export(
                    os.path.join(self.out_dir, idx, f"view{i}_gt_mesh.obj")
                )
        return

    def extract_point_cloud(self):
        print(f"Using {CPU_COUNT} CPU cores")
        npz_files = [(str(idx), obj) for idx, obj in self.idx_uid_mapping.items()]

        # test_indices = [0, 921, 1842, 2763, 3684, 4605]  # just for testing
        # test_indices = [0]  # just for testing
        # for test_idx in test_indices:
        #     test_data = npz_files[test_idx]
        #     self.process_mesh(test_data)

        with ThreadPoolExecutor() as executor:
            futures = []
            for metadata in tqdm.tqdm(npz_files, desc="Processing files"):
                futures.append(executor.submit(self.process_mesh, metadata))

            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="Processing results"
            ):
                future.result()  # Wait for each future to complete

    def merge_npz_files(self):
        for cur_mode in ["train", "test"]:
            npz_file = os.path.join(self.npz_out_dir, f"{cur_mode}.npz")
            data = np.load(npz_file, allow_pickle=True)
            npz_list = data["npz_list"].tolist()
            updated_npz_list = []

            cur_pc_save_dir = os.path.join(self.pc_out_dir, f"{cur_mode}")
            output_npz_file = os.path.join(self.final_data_save_dir, f"{cur_mode}.npz")

            for idx, mesh_data in tqdm.tqdm(
                enumerate(npz_list), total=len(npz_list), desc="Processing files"
            ):
                npz_file = os.path.join(cur_pc_save_dir, f"{idx}.npz")
                if os.path.exists(npz_file):
                    with np.load(npz_file) as npz_data:
                        mesh_data["pc_normal"] = npz_data["pc_normal"]
                        # mesh_data['metrics'] = npz_data['metrics']
                    updated_npz_list.append(mesh_data)

            np.savez(output_npz_file, npz_list=updated_npz_list)
            print(f"Final data saved to {output_npz_file}")
            print(f"Total number of data samples: {len(updated_npz_list)}")

            # Load and check the saved file for any issues
            try:
                loaded_data = np.load(output_npz_file, allow_pickle=True)
                loaded_npz_list = loaded_data["npz_list"].tolist()
                print(f"Loaded {len(loaded_npz_list)} data samples successfully.")
            except Exception as e:
                print(f"Error loading the saved npz file: {e}")

    def filtered_by_visible_faces(self, metadata):
        idx, cur_data = metadata
        water_mesh_sampled = np.load(
            os.path.join(self.out_dir, idx, f"{idx}_pc.npz"), allow_pickle=True
        )
        face_idx = water_mesh_sampled["face_idx"]
        water_mesh = trimesh.load(
            os.path.join(self.out_dir, idx, f"{idx}_water_mesh.obj"), force="mesh"
        )

        vertices = torch.tensor(water_mesh.vertices, dtype=torch.float32, device="cuda")
        faces = torch.tensor(water_mesh.faces, dtype=torch.int32, device="cuda")
        vertices = vertices[None]
        faces = faces[None]

        rast_space_dict = {}
        glctx = dr.RasterizeCudaContext()  # Create the OpenGL context for nvdiffrast
        rast_space_dict["tri_verts"] = faces
        rast_space_dict["glctx"] = glctx

        visible_face_ids_per_view = compute_visibility(
            vertices, self.intrs, self.extrs, rast_space_dict
        )
        # all outer faces:  merged_faces

        MIN_OUTER_FACES_RATIO = 1 / 16 * 1.5
        merged_faces = torch.unique(torch.cat(visible_face_ids_per_view)).cpu().numpy()
        all_visible_face_count = merged_faces.shape[0]
        print(
            f"Total visible faces / all faces: {all_visible_face_count}/{faces.shape[1]}"
        )
        with open(
            os.path.join(
                self.out_dir,
                idx,
                f"valid_views_min_face_ratio{MIN_OUTER_FACES_RATIO}.txt",
            ),
            "w",
        ) as f:
            count = 0
            for i in range(len(visible_face_ids_per_view)):
                visible_face_ids = visible_face_ids_per_view[i].cpu().numpy()
                # print(f"View {i}: {len(visible_face_ids)} visible faces")
                if len(visible_face_ids) >= int(
                    MIN_OUTER_FACES_RATIO * all_visible_face_count
                ):
                    f.write(f"{i}\n")
                    count += 1
            print(f"Found {count} valid views")

    def test_filtered_by_visible_faces(self):
        self.filt_objects()
        npz_files = [(str(idx), obj) for idx, obj in self.idx_uid_mapping.items()]

        self.extrs = self.extrs.cuda() if torch.cuda.is_available() else self.extrs
        self.intrs = self.intrs.cuda() if torch.cuda.is_available() else self.intrs
        for metadata in tqdm.tqdm(npz_files, desc="Processing files"):
            self.filtered_by_visible_faces(metadata)

    def process(self):
        self.filt_objects()
        self.process_objects()
        self.to_npz_files()
        self.extract_point_cloud()
        self.compute_visibility()


class ObjaverseProcessor(MeshDatasetProcessor):
    def __init__(self, args):
        super().__init__(args)
        objaverse.BASE_PATH = args.base_dir  # 'data/Objaverse_v1'
        objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, "hf-objaverse-v1")

        self.base_dir = os.path.join(
            args.base_dir, "hf-objaverse-v1", "glbs"
        )  # Base directory of Objaverse
        self.idx_uid_mapping = {}  # mapping from index to uid
        self.idx_path_mapping = {}  # mapping from index to file path
        self.path_json_save_path = os.path.join(
            self.out_dir,
            f"filtered_min_{self.min_face_num}_max_{self.max_face_num}_paths.json",
        )

    def filt_objects(self):
        """
        Filter Objaverse objects based on face count and save the filtered list to a JSON file.
        """
        filtered_uids = []
        print(
            f"Filtering Objaverse objects from {self.base_dir} with face count between {self.min_face_num} and {self.max_face_num}.."
        )
        if os.path.exists(self.json_save_path):
            print(f"File {self.json_save_path} already exists. Skipping filtering.")
            with open(self.json_save_path, "r") as f:
                filtered_uids = json.load(f)
                filtered_uids.sort()
                self.idx_uid_mapping = {i: uid for i, uid in enumerate(filtered_uids)}
            print(
                f"Loaded {len(filtered_uids)} filtered uids from {self.json_save_path}."
            )
            if os.path.exists(self.path_json_save_path):
                print(
                    f"File {self.path_json_save_path} already exists. Loading existing mapping."
                )
                with open(self.path_json_save_path, "r") as f:
                    self.idx_path_mapping = json.load(f)
                print(
                    f"Loaded {len(self.idx_path_mapping)} paths from {self.path_json_save_path}."
                )
            return

        # Objaverse typically has a flat structure with .obj files
        annotations = objaverse.load_annotations()
        for uid in tqdm.tqdm(list(annotations.keys()), desc="Filtering Objaverse"):
            face_num = annotations[uid]["faceCount"]
            vertex_num = annotations[uid]["vertexCount"]
            if self.min_face_num <= face_num <= self.max_face_num and vertex_num > 300:
                filtered_uids.append(uid)
        logging.info(f"Filtered {len(filtered_uids)} valid meshes.")
        # Save the filtered paths to a JSON file
        with open(self.json_save_path, "w") as f:
            json.dump(filtered_uids, f)
        filtered_uids.sort()
        self.idx_uid_mapping = {i: uid for i, uid in enumerate(filtered_uids)}

    def merge_and_scale(self, idx, cur_data, progress, start_time, total_tasks):
        """
        Merge and scale the mesh data, save it to a temporary file, and update progress.
        """
        try:
            idx = str(idx)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                task_start_time = time.time()
                print(f"Processing {idx}...")

                # For Objaverse, use filename as uid
                # uid = os.path.splitext(os.path.basename(cur_data))[0]
                # uid = cur_data  # uid is directly the identifier in Objaverse
                uid = cur_data.split("/")[-1].split(".")[0]

                # write a temp file
                if os.path.exists(os.path.join(self.out_dir, idx)):
                    # check if another file with different uid exists
                    existing_files = os.listdir(os.path.join(self.out_dir, idx))
                    # search for uid.txt file
                    if "uid.txt" in existing_files:
                        with open(os.path.join(self.out_dir, idx, "uid.txt"), "r") as f:
                            existing_uid = (
                                f.read().splitlines()[0].strip()
                            )  # remove newline characters
                        if existing_uid != uid:
                            # abort
                            raise Exception(
                                f"UID mismatch: {existing_uid} != {uid}. Please check the data."
                            )
                        else:
                            if os.path.exists(
                                os.path.join(self.out_dir, idx, "model_normalized.obj")
                            ):
                                # remove the obj file to save space
                                os.remove(
                                    os.path.join(
                                        self.out_dir, idx, "model_normalized.obj"
                                    )
                                )
                            if os.path.exists(
                                os.path.join(self.out_dir, idx, "model_gt.glb")
                            ) and os.path.exists(
                                os.path.join(self.out_dir, idx, uid + ".npz")
                            ):
                                print("Files already exist. Skipping.")
                                return

                os.makedirs(os.path.join(self.out_dir, idx), exist_ok=True)

                if os.path.exists(
                    os.path.join(self.out_dir, idx, uid + ".tmp")
                ) and os.path.exists(os.path.join(self.out_dir, idx, uid + ".npz")):
                    return

                # copy the mesh data to the processed directory

                shutil.copy(cur_data, os.path.join(self.out_dir, idx, "model_gt.glb"))
                mesh = trimesh.load(cur_data, force="mesh")
                npz_to_save = {}
                if (
                    hasattr(mesh, "faces")
                    and mesh.faces.shape[0] >= self.min_face_num
                    and mesh.faces.shape[0] <= self.max_face_num
                ):
                    mesh.merge_vertices()
                    mesh.update_faces(mesh.nondegenerate_faces())
                    mesh.update_faces(mesh.unique_faces())
                    mesh.remove_unreferenced_vertices()

                    vertices = np.array(mesh.vertices.copy())
                    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])  # type: ignore
                    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
                    vertices = (
                        vertices / (bounds[1] - bounds[0]).max()
                    )  # -0.5 to 0.5 length 1 # The mesh is scaled here
                    vertices = vertices.clip(-0.5, 0.5)

                    cur_mesh = trimesh.Trimesh(
                        vertices=vertices, faces=mesh.faces.copy()
                    )

                    min_length = (
                        cur_mesh.bounding_box_oriented.edges_unique_length.min()
                    )

                    npz_to_save["vertices"] = mesh.vertices
                    npz_to_save["faces"] = mesh.faces
                    npz_to_save["min_length"] = min_length
                    npz_to_save["uid"] = uid
                    npz_to_save["vertices_num"] = mesh.vertices.shape[0]
                    npz_to_save["faces_num"] = mesh.faces.shape[0]
                if w:
                    for warn in w:
                        logging.warning(f" {uid} : {str(warn.message)}")
                        print("uid warning:", uid)
                    return
                # save pc_normal
                print(f"Saving {uid} to {self.out_dir}/{idx}...")
                np.savez(os.path.join(self.out_dir, idx, uid + ".npz"), **npz_to_save)
                # save the uid to a txt file
                with open(os.path.join(self.out_dir, idx, "uid.txt"), "w") as f:
                    # clear the file first
                    f.truncate(0)
                    f.write(uid)

                task_end_time = time.time()
                task_duration = task_end_time - task_start_time
                progress.value += 1
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / progress.value * total_tasks
                remaining_time = estimated_total_time - elapsed_time
                remaining_td = datetime.timedelta(seconds=int(remaining_time))
                logging.info(
                    f"This task: {task_duration:.2f} s, Already:{elapsed_time}, progress: {progress.value}/{total_tasks}, remaining{remaining_td}"
                )
        except Exception as e:
            logging.error(f"Error in {uid}: {e}")

    def process_objects(self):
        """
        Process a single mesh data item from Objaverse.
        """
        if os.path.exists(self.path_json_save_path):
            print(
                f"File {self.path_json_save_path} already exists. Loading existing mapping."
            )
            with open(self.path_json_save_path, "r") as f:
                self.idx_path_mapping = json.load(f)
            print(
                f"Loaded {len(self.idx_path_mapping)} paths from {self.path_json_save_path}."
            )
        else:
            key_list = list(self.idx_uid_mapping.keys())
            value_list = list(self.idx_uid_mapping.values())
            for cur_cat in tqdm.tqdm(
                (sorted(os.listdir(os.path.join(self.base_dir)))), desc="Categories"
            ):
                cur_cat_dir = os.path.join(self.base_dir, cur_cat)
                cur_files = sorted(os.listdir(cur_cat_dir))
                cur_files = [
                    os.path.join(cur_cat_dir, x)
                    for x in cur_files
                    if "stl" in x
                    or "obj" in x
                    or "ply" in x
                    or "glb" in x
                    or "gltf" in x
                ]
                cur_files = [
                    x
                    for x in cur_files
                    if x.split("/")[-1].split(".")[0] in self.idx_uid_mapping.values()
                ]
                # map index to file path
                for path in cur_files:
                    uid = path.split("/")[-1].split(".")[0]
                    idx = key_list[value_list.index(uid)]
                    self.idx_path_mapping[idx] = path
            # also save the mapping to a json file
            # sort the mapping by key
            self.idx_path_mapping = dict(sorted(self.idx_path_mapping.items()))
            with open(self.path_json_save_path, "w") as f:
                json.dump(self.idx_path_mapping, f)
            print(f"Saved mapping to {self.path_json_save_path}")

        print(len(self.idx_path_mapping), "files to process.")
        cpu_count = os.cpu_count()
        print(f"CPU count: {cpu_count}")
        total_tasks = len(self.idx_path_mapping)

        manager = Manager()
        progress = manager.Value("i", 0)
        start_time = time.time()

        # test_indices = ["70898"]  # just for testing
        # for test_idx in test_indices:
        #     test_data = (test_idx, self.idx_path_mapping[test_idx])
        #     self.merge_and_scale(
        #         test_idx,
        #         self.idx_path_mapping[test_idx],
        #         progress,
        #         start_time,
        #         total_tasks,
        #     )
        with Pool(processes=CPU_COUNT) as pool:
            pool.starmap_async(
                self.merge_and_scale,
                [
                    (idx, data, progress, start_time, total_tasks)
                    for idx, data in self.idx_path_mapping.items()
                ],
            )
            pool.close()
            pool.join()

    def process_npz_file(self, npz_metadata):
        try:
            #  get the root folder
            idx, cur_data = npz_metadata
            # uid = os.path.splitext(os.path.basename(cur_data))[0]
            uid = cur_data.split("/")[-1].split(".")[0]
            # if os.path.exists(os.path.join(self.out_dir, idx, f"idx_{idx}.npz")):
            #     print(
            #         f"File {os.path.join(self.out_dir, idx, f'idx_{idx}.npz')} already exists. Skipping."
            #     )
            #     return

            npz_data = os.path.join(self.out_dir, idx, uid + ".npz")

            with np.load(npz_data) as data:
                data_dict = {key: data[key] for key in data}
                print(
                    "data_dict vertices shape in process_npz_file:",
                    data_dict["vertices"].shape,
                )

                if data_dict and data["faces"].shape[0] >= 20:
                    data_dict["faces_num"] = data["faces"].shape[0]
                    data_dict["vertices_num"] = data["vertices"].shape[0]
                    data_dict["uid"] = uid
                    # save the npz file to the processed directory
                    npz_out_path = os.path.join(self.out_dir, idx, f"idx_{idx}.npz")
                    np.savez(npz_out_path, **data_dict)
                else:
                    pass
        except Exception as e:
            print(f"Error loading {npz_data}: {e}")

    def to_npz_files(self):
        """
        Convert processed mesh files to npz format, separating into train and test sets.
        """
        # get the npz files
        npz_files = [(str(idx), obj) for idx, obj in self.idx_path_mapping.items()]
        # test_indices = [70898]  # just for testing
        # for test_idx in test_indices:
        #     test_data = npz_files[test_idx]
        #     self.process_npz_file(test_data)
        with ThreadPoolExecutor() as executor:
            futures = []
            for metadata in tqdm.tqdm(npz_files, desc="Processing files"):
                futures.append(executor.submit(self.process_npz_file, metadata))

            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="Processing results"
            ):
                future.result()  # Wait for each future to complete

    def process_mesh(self, metadata):
        idx, cur_data = metadata
        print(f"Processing mesh {idx}...")
        mesh_data = os.path.join(self.out_dir, idx, f"idx_{idx}.npz")
        if not os.path.exists(mesh_data):
            print(
                f"File {mesh_data} does not exist. (This is attributed to invalid glb files.) Skipping."
            )
            return
        self.valid_mesh_data_num += 1
        mesh_data = np.load(mesh_data, allow_pickle=True)
        vertices = mesh_data["vertices"]

        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max()
        cur_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=mesh_data["faces"],
            force="mesh",
            merge_primitives=True,
        )

        if self.do_watertight:
            water_mesh = export_to_watertight(cur_mesh, 7)

            water_p, face_idx = water_mesh.sample(
                self.num_sample_points, return_index=True
            )  # 128 * 128
            water_n = water_mesh.face_normals[face_idx]

            npz_to_save = {}
            pc_normal = np.concatenate([water_p, water_n], axis=-1, dtype=np.float16)
            npz_to_save["pc_normal"] = pc_normal
            npz_to_save["face_idx"] = face_idx

            # also save the water mesh
            water_mesh.export(os.path.join(self.out_dir, idx, f"{idx}_water_mesh.obj"))
            np.savez(os.path.join(self.out_dir, idx, f"{idx}_pc.npz"), **npz_to_save)
            gt_mesh = trimesh.Trimesh(
                vertices=mesh_data["vertices"],
                faces=mesh_data["faces"],
                force="mesh",
                merge_primitives=True,
            )
            gt_mesh.export(os.path.join(self.out_dir, idx, f"{idx}_gt_mesh.obj"))
        else:
            sample_p, face_idx = cur_mesh.sample(
                self.num_sample_points, return_index=True
            )
            sample_n = cur_mesh.face_normals[face_idx]
            npz_to_save = {}
            pc_normal = np.concatenate([sample_p, sample_n], axis=-1, dtype=np.float16)
            npz_to_save["pc_normal"] = pc_normal
            npz_to_save["face_idx"] = face_idx

            np.savez(os.path.join(self.out_dir, idx, f"{idx}_pc.npz"), **npz_to_save)
            gt_mesh = trimesh.Trimesh(
                vertices=mesh_data["vertices"],
                faces=mesh_data["faces"],
                force="mesh",
                merge_primitives=True,
            )
            gt_mesh.export(os.path.join(self.out_dir, idx, f"{idx}_gt_mesh.obj"))

    def compute_visibility(self):
        npz_files = [(str(idx), obj) for idx, obj in self.idx_uid_mapping.items()]

        self.extrs = self.extrs.cuda() if torch.cuda.is_available() else self.extrs
        self.intrs = self.intrs.cuda() if torch.cuda.is_available() else self.intrs

        for metadata in tqdm.tqdm(npz_files, desc="Processing files"):
            self.generate_visibility_samples(metadata)

    def generate_visibility_samples(self, metadata):
        idx, cur_data = metadata
        # load the mesh data
        if not os.path.exists(os.path.join(self.out_dir, idx, f"{idx}_pc.npz")):
            print(
                f"File {os.path.join(self.out_dir, idx, f'{idx}_pc.npz')} does not exist. Skipping."
            )
            return
        # if file already exists, skip
        do_processing = False
        for i in range(len(self.extrs[0])):
            if (
                not os.path.exists(os.path.join(self.out_dir, idx, f"view{i}_pc.npz"))
                or not os.path.exists(
                    os.path.join(self.out_dir, idx, f"view{i}_gt_pc.npz")
                )
                or not os.path.exists(
                    os.path.join(self.out_dir, idx, f"view{i}_invisible_pc.npz")
                )
            ):
                do_processing = True
                break
        if not do_processing:
            print(f"Visibility files already exist for {idx}. Skipping.")
            return
        if self.do_watertight:
            water_mesh_sampled = np.load(
                os.path.join(self.out_dir, idx, f"{idx}_pc.npz"), allow_pickle=True
            )
            face_idx = water_mesh_sampled["face_idx"]
            water_p = water_mesh_sampled["pc_normal"][:, :3]
            water_n = water_mesh_sampled["pc_normal"][:, 3:]
        else:
            mesh_sampled = np.load(
                os.path.join(self.out_dir, idx, f"{idx}_pc.npz"), allow_pickle=True
            )
            face_idx = mesh_sampled["face_idx"]
            water_p = mesh_sampled["pc_normal"][:, :3]
            water_n = mesh_sampled["pc_normal"][:, 3:]

        # gt_mesh = trimesh.load(
        #     os.path.join(self.out_dir, idx, f"{idx}_gt_mesh.obj"), force="mesh"
        # )
        mesh_data = os.path.join(self.out_dir, idx, f"idx_{idx}.npz")
        mesh_data = np.load(mesh_data, allow_pickle=True)
        vertices = mesh_data["vertices"]

        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max()
        gt_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=mesh_data["faces"],
            force="mesh",
            merge_primitives=True,
        )
        gt_mesh.merge_vertices()
        gt_mesh.update_faces(gt_mesh.nondegenerate_faces())
        gt_mesh.update_faces(gt_mesh.unique_faces())
        gt_mesh.remove_unreferenced_vertices()

        gt_points_list = []
        gt_normals_list = []
        visible_points_list = []
        visible_normals_list = []
        invisible_points_list = []
        invisible_normals_list = []
        gt_mesh_list = []  # remember to save the original mesh, not the watertight mesh

        if self.do_watertight:
            water_mesh = trimesh.load(
                os.path.join(self.out_dir, idx, f"{idx}_water_mesh.obj"), force="mesh"
            )
            vertices = torch.tensor(
                water_mesh.vertices, dtype=torch.float32, device="cuda"
            )
            faces = torch.tensor(water_mesh.faces, dtype=torch.int32, device="cuda")
        else:
            vertices = torch.tensor(
                gt_mesh.vertices, dtype=torch.float32, device="cuda"
            )
            faces = torch.tensor(gt_mesh.faces, dtype=torch.int32, device="cuda")

        vertices = vertices[None]
        faces = faces[None]

        rast_space_dict = {}
        glctx = dr.RasterizeCudaContext()  # Create the OpenGL context for nvdiffrast
        rast_space_dict["tri_verts"] = faces
        rast_space_dict["glctx"] = glctx

        visible_face_ids_per_view = compute_visibility(
            vertices, self.intrs, self.extrs, rast_space_dict
        )
        # all outer faces:
        merged_faces = torch.unique(torch.cat(visible_face_ids_per_view)).cpu().numpy()
        all_visible_face_count = merged_faces.shape[0]
        for i in range(len(visible_face_ids_per_view)):
            visible_face_ids = visible_face_ids_per_view[i].cpu().numpy()

            mask = np.isin(face_idx, visible_face_ids)
            visible_points = water_p[mask]
            visible_normals = water_n[mask]
            # get invisible points as well
            invisible_points = water_p[~mask]
            invisible_normals = water_n[~mask]

            if len(visible_points) < 0.1 * self.num_sample_points:
                print("Too limited visible points for view", i, "in", idx)
                # continue
            # transform the points and normals to egocentric coordinates
            extr_i = self.extrs[0, i].cpu().numpy()  # w_T_c
            # get the inverse of the extrinsic matrix
            extr_i_w = np.eye(4)
            extr_i_w[:3, :3] = extr_i[:3, :3].T
            extr_i_w[:3, 3] = -np.dot(extr_i[:3, :3].T, extr_i[:3, 3])

            # transform visible points and normals
            visible_points = np.dot(visible_points, extr_i_w[:3, :3]) + extr_i[:3, 3]
            visible_normals = np.dot(visible_normals, extr_i_w[:3, :3])
            # transform invisible points and normals
            invisible_points = (
                np.dot(invisible_points, extr_i_w[:3, :3]) + extr_i[:3, 3]
            )
            invisible_normals = np.dot(invisible_normals, extr_i_w[:3, :3])

            # make the points centered around the origin
            center = visible_points.mean(axis=0)
            visible_points -= center
            invisible_points -= center

            visible_points_list.append(visible_points)
            visible_normals_list.append(visible_normals)
            invisible_points_list.append(invisible_points)
            invisible_normals_list.append(invisible_normals)

            # create structured gt: first N visible points, then invisible points
            gt_water_p = np.concatenate([visible_points, invisible_points], axis=0)
            gt_water_n = np.concatenate([visible_normals, invisible_normals], axis=0)
            gt_points_list.append(gt_water_p)
            gt_normals_list.append(gt_water_n)
            # corresponding gt mesh
            # transform the gt mesh vertices to the camera coordinate
            gt_mesh_vertices = (
                np.dot(gt_mesh.vertices, extr_i_w[:3, :3]) + extr_i[:3, 3]
            )

            # this is the dummy center purely for visualization
            # gt mesh is shifted and scale in training, as seen in meshanything_train/loop_set_256.py line 181-219
            # But the pc is shifted to another center
            gt_mesh_vertices = (
                gt_mesh_vertices - center
            )  # make the points centered around the origin
            gt_mesh_faces = gt_mesh.faces.copy()
            # create a trimesh object for the gt mesh
            gt_mesh_i = trimesh.Trimesh(
                vertices=gt_mesh_vertices,
                faces=gt_mesh_faces,
                force="mesh",
                merge_primitives=True,
            )
            gt_mesh_list.append(gt_mesh_i)

        if self.visualize:
            os.makedirs(
                os.path.join(self.out_dir, idx, "pc_visualization"), exist_ok=True
            )
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(water_p)
            pcd.normals = o3d.utility.Vector3dVector(water_n)
            o3d.io.write_point_cloud(
                os.path.join(self.out_dir, idx, "pc_visualization", f"{idx}_pc.ply"),
                pcd,
            )

            for i in range(len(gt_points_list)):
                gt_points = gt_points_list[i]
                gt_normals = gt_normals_list[i]
                visible_points = visible_points_list[i]
                visible_normals = visible_normals_list[i]
                invisible_points = invisible_points_list[i]
                invisible_normals = invisible_normals_list[i]
                gt_mesh_i = gt_mesh_list[i]

                gt_pc_normal = np.concatenate(
                    [gt_points, gt_normals], axis=-1, dtype=np.float16
                )
                pc_normal = np.concatenate(
                    [visible_points, visible_normals], axis=-1, dtype=np.float16
                )
                invisible_pc_normal = np.concatenate(
                    [invisible_points, invisible_normals], axis=-1, dtype=np.float16
                )
                gt_npz_to_save = {
                    "gt_pc_normal": gt_pc_normal,
                }
                npz_to_save = {
                    "pc_normal": pc_normal,
                }
                invisible_npz_to_save = {
                    "invisible_pc_normal": invisible_pc_normal,
                }
                np.savez(
                    os.path.join(self.out_dir, idx, f"view{i}_gt_pc.npz"),
                    **gt_npz_to_save,
                )
                np.savez(
                    os.path.join(self.out_dir, idx, f"view{i}_pc.npz"), **npz_to_save
                )
                np.savez(
                    os.path.join(self.out_dir, idx, f"view{i}_invisible_pc.npz"),
                    **invisible_npz_to_save,
                )

                if self.visualize:
                    # save the point cloud and normals
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(visible_points)
                    pcd.normals = o3d.utility.Vector3dVector(visible_normals)
                    o3d.io.write_point_cloud(
                        os.path.join(
                            self.out_dir, idx, "pc_visualization", f"view{i}_pc.ply"
                        ),
                        pcd,
                    )
                    # save the gt point cloud and normals
                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(gt_points)
                    pcd_gt.normals = o3d.utility.Vector3dVector(gt_normals)
                    o3d.io.write_point_cloud(
                        os.path.join(
                            self.out_dir, idx, "pc_visualization", f"view{i}_gt_pc.ply"
                        ),
                        pcd_gt,
                    )
                    # save the invisible point cloud and normals
                    pcd_invisible = o3d.geometry.PointCloud()
                    pcd_invisible.points = o3d.utility.Vector3dVector(invisible_points)
                    pcd_invisible.normals = o3d.utility.Vector3dVector(
                        invisible_normals
                    )
                    o3d.io.write_point_cloud(
                        os.path.join(
                            self.out_dir,
                            idx,
                            "pc_visualization",
                            f"view{i}_invisible_pc.ply",
                        ),
                        pcd_invisible,
                    )

                # save the gt mesh
                gt_mesh_i.export(
                    os.path.join(self.out_dir, idx, f"view{i}_gt_mesh.obj")
                )
        return

    def extract_point_cloud(self):
        print(f"Using {CPU_COUNT} CPU cores")
        npz_files = [(str(idx), obj) for idx, obj in self.idx_uid_mapping.items()]
        self.valid_mesh_data_num = 0
        with ThreadPoolExecutor() as executor:
            futures = []
            for metadata in tqdm.tqdm(npz_files, desc="Processing files"):
                futures.append(executor.submit(self.process_mesh, metadata))

            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="Processing results"
            ):
                future.result()  # Wait for each future to complete
        print(f"Total valid mesh data: {self.valid_mesh_data_num} / {len(npz_files)}")

    def merge_npz_files(self):
        for cur_mode in ["train", "test"]:
            npz_file = os.path.join(self.npz_out_dir, f"{cur_mode}.npz")
            data = np.load(npz_file, allow_pickle=True)
            npz_list = data["npz_list"].tolist()
            updated_npz_list = []

            cur_pc_save_dir = os.path.join(self.pc_out_dir, f"{cur_mode}")
            output_npz_file = os.path.join(self.final_data_save_dir, f"{cur_mode}.npz")

            for idx, mesh_data in tqdm.tqdm(
                enumerate(npz_list), total=len(npz_list), desc="Processing files"
            ):
                npz_file = os.path.join(cur_pc_save_dir, f"{idx}.npz")
                if os.path.exists(npz_file):
                    with np.load(npz_file) as npz_data:
                        mesh_data["pc_normal"] = npz_data["pc_normal"]
                    updated_npz_list.append(mesh_data)

            np.savez(output_npz_file, npz_list=updated_npz_list)
            print(f"Final data saved to {output_npz_file}")
            print(f"Total number of data samples: {len(updated_npz_list)}")

            # Load and check the saved file for any issues
            try:
                loaded_data = np.load(output_npz_file, allow_pickle=True)
                loaded_npz_list = loaded_data["npz_list"].tolist()
                print(f"Loaded {len(loaded_npz_list)} data samples successfully.")
            except Exception as e:
                print(f"Error loading the saved npz file: {e}")

    def filtered_by_visible_faces(self, metadata):
        idx, cur_data = metadata
        water_mesh_sampled = np.load(
            os.path.join(self.out_dir, idx, f"{idx}_pc.npz"), allow_pickle=True
        )
        face_idx = water_mesh_sampled["face_idx"]
        water_mesh = trimesh.load(
            os.path.join(self.out_dir, idx, f"{idx}_water_mesh.obj"), force="mesh"
        )

        vertices = torch.tensor(water_mesh.vertices, dtype=torch.float32, device="cuda")
        faces = torch.tensor(water_mesh.faces, dtype=torch.int32, device="cuda")
        vertices = vertices[None]
        faces = faces[None]

        rast_space_dict = {}
        glctx = dr.RasterizeCudaContext()  # Create the OpenGL context for nvdiffrast
        rast_space_dict["tri_verts"] = faces
        rast_space_dict["glctx"] = glctx

        visible_face_ids_per_view = compute_visibility(
            vertices, self.intrs, self.extrs, rast_space_dict
        )
        # all outer faces:  merged_faces

        MIN_OUTER_FACES_RATIO = 1 / 16 * 1.5
        merged_faces = torch.unique(torch.cat(visible_face_ids_per_view)).cpu().numpy()
        all_visible_face_count = merged_faces.shape[0]
        print(
            f"Total visible faces / all faces: {all_visible_face_count}/{faces.shape[1]}"
        )
        with open(
            os.path.join(
                self.out_dir,
                idx,
                f"valid_views_min_face_ratio{MIN_OUTER_FACES_RATIO}.txt",
            ),
            "w",
        ) as f:
            count = 0
            for i in range(len(visible_face_ids_per_view)):
                visible_face_ids = visible_face_ids_per_view[i].cpu().numpy()
                if len(visible_face_ids) >= int(
                    MIN_OUTER_FACES_RATIO * all_visible_face_count
                ):
                    f.write(f"{i}\n")
                    count += 1
            print(f"Found {count} valid views")

    def test_filtered_by_visible_faces(self):
        self.filt_objects()
        npz_files = [(str(idx), obj) for idx, obj in self.idx_uid_mapping.items()]

        self.extrs = self.extrs.cuda() if torch.cuda.is_available() else self.extrs
        self.intrs = self.intrs.cuda() if torch.cuda.is_available() else self.intrs
        for metadata in tqdm.tqdm(npz_files, desc="Processing files"):
            self.filtered_by_visible_faces(metadata)

    def process(self):
        self.filt_objects()
        self.process_objects()
        self.to_npz_files()
        self.extract_point_cloud()
        self.compute_visibility()


class PCNProcessor(MeshDatasetProcessor):
    def __init__(self, args):
        args.do_visibility = True
        super().__init__(args)
        self.taxonomy_ids = [
            "02691156",
            "02933112",
            "02958343",
            "03001627",
            "03636649",
            "04256520",
            "04379243",
            "04530566",
        ]
        self.PCN_folder = 'data/PCN/ShapeNetCompletion'
        self.json_save_path = os.path.join(
            self.out_dir,
            f'idx_uid_mapping.json'
        )

        self.path_json_save_path = os.path.join(
            self.out_dir, 
            f'idx_path_mapping.json'
        )
        
        self.idx_uid_mapping = {}
        self.idx_uid_path_mapping = {}

    def filt_objects(self):
        if os.path.exists(self.json_save_path):
            print(
                f"File {self.json_save_path} already exists. Loading existing mapping."
            )
            with open(self.json_save_path, "r") as f:
                self.idx_uid_mapping = json.load(f)
            print(
                f"Loaded {len(self.idx_uid_mapping)} uids from {self.json_save_path}."
            )
            
            if os.path.exists(self.path_json_save_path):
                print(
                    f"File {self.path_json_save_path} already exists. Loading existing path mapping."
                )
                with open(self.path_json_save_path, "r") as f:
                    self.idx_uid_path_mapping = json.load(f)
                print(
                    f"Loaded {len(self.idx_uid_path_mapping)} paths from {self.path_json_save_path}."
                )
            return
        
        os.makedirs(os.path.join(self.out_dir, 'splits'), exist_ok=True)
        # create train, test, and val split
        train_split_path = os.path.join(self.out_dir, 'splits', 'train_split.txt')
        test_split_path = os.path.join(self.out_dir, 'splits', 'test_split.txt')
        val_split_path = os.path.join(self.out_dir, 'splits', 'val_split.txt')
        # if not exists, create the files

        f_train = open(train_split_path, 'w')
        f_test = open(test_split_path, 'w')
        f_val = open(val_split_path, 'w')

        folder_keys = ['train', 'test', 'val']

        splits_dict = {
            'train': f_train,
            'test': f_test,
            'val': f_val
        }
        
        idx = 0
        for taxonomy_id in self.taxonomy_ids:
            category_source_dir = os.path.join(self.base_dir, taxonomy_id)
            category_pcn_test_dir = os.path.join(self.PCN_folder, 'test', 'complete', taxonomy_id)
            category_pcn_train_dir = os.path.join(self.PCN_folder, 'train', 'complete', taxonomy_id)
            category_pcn_val_dir = os.path.join(self.PCN_folder, 'val', 'complete', taxonomy_id)

            category_pcn_dict = {
                'train': category_pcn_train_dir,
                'test': category_pcn_test_dir,
                'val': category_pcn_val_dir
            }

            if not os.path.exists(category_source_dir):
                logging.error(
                    f"Category directory {category_source_dir} does not exist. Please check the base directory path."
                )
                assert False
            
            for key in folder_keys:
                split_file = splits_dict[key]

                pcn_category_dir = category_pcn_dict[key]

                pcd_files = os.listdir(pcn_category_dir)
                pcd_files = natsorted(pcd_files)
                
                for pcd in pcd_files:
                    # write the idx to the split file
                    split_file.write(f"{idx}\n")
                    uid = pcd.strip().split('.')[0]
                    self.idx_uid_mapping[str(idx)] = f'{taxonomy_id}_{uid}'
                    self.idx_uid_path_mapping[str(idx)] = os.path.join(category_source_dir, f'{uid}', 'models', 'model_normalized.obj')
                    idx += 1
                    
                    # locate the corresponding source mesh file
        f_train.close()
        f_test.close()
        f_val.close()
        with open(self.json_save_path, "w") as f:
            json.dump(self.idx_uid_mapping, f)
        print(f"Saved idx-uid mapping to {self.json_save_path}")
        with open(self.path_json_save_path, "w") as f:
            json.dump(self.idx_uid_path_mapping, f)
        print(f"Saved idx-path mapping to {self.path_json_save_path}")
        return
    
    def merge_and_scale(self, idx, cur_data, progress, start_time, total_tasks):
        """
        Merge and scale the mesh data, save it to a temporary file, and update progress.
        This method should be overridden by subclasses to implement specific merging logic.
        """
        try:
            idx = str(idx)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                task_start_time = time.time()
                print(f"Processing {idx}...")

                uid = "_".join(
                    cur_data.strip("/").split("/")[-4:-1]
                )  # synset_id/model_id
                # print('uid',uid)
                # write a temp file
                
                if os.path.exists(os.path.join(self.out_dir, idx)):
                    # check if another file with different uid exists
                    existing_files = os.listdir(os.path.join(self.out_dir, idx))
                    # search for uid.txt file
                    if "uid.txt" in existing_files:
                        with open(os.path.join(self.out_dir, idx, "uid.txt"), "r") as f:
                            existing_uid = (
                                f.read().splitlines()[0].strip()
                            )  # remove newline characters
                        if existing_uid != uid:
                            # abort
                            raise Exception(
                                f"UID mismatch: {existing_uid} != {uid}. Please check the data."
                            )
                        else:
                            if os.path.exists(
                                os.path.join(self.out_dir, idx, "model_normalized.obj")
                            ) and os.path.exists(
                                os.path.join(self.out_dir, idx, uid + ".npz")
                            ):
                                print("Files already exist. Skipping.")
                                return

                os.makedirs(os.path.join(self.out_dir, idx), exist_ok=True)

                if os.path.exists(
                    os.path.join(self.out_dir, idx, uid + ".tmp")
                ) or os.path.exists(os.path.join(self.out_dir, idx, uid + ".npz")):
                    return

                # copy the mesh data to the processed directory
                shutil.copy(
                    cur_data, os.path.join(self.out_dir, idx, "model_normalized.obj")
                )
                mesh = trimesh.load(cur_data, force="mesh")

                npz_to_save = {}
                if (
                    hasattr(mesh, "faces")
                    # and mesh.faces.shape[0] >= self.min_face_num
                    # and mesh.faces.shape[0] <= self.max_face_num
                ):
                    mesh.merge_vertices()
                    mesh.update_faces(mesh.nondegenerate_faces())
                    mesh.update_faces(mesh.unique_faces())
                    mesh.remove_unreferenced_vertices()

                    # judge = True
                    vertices = np.array(mesh.vertices.copy())
                    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])  # type: ignore
                    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
                    vertices = (
                        vertices / (bounds[1] - bounds[0]).max()
                    )  # -0.5 to 0.5 length 1 # The mesh is scaled here
                    vertices = vertices.clip(-0.5, 0.5)

                    cur_mesh = trimesh.Trimesh(
                        vertices=vertices, faces=mesh.faces.copy()
                    )

                    min_length = (
                        cur_mesh.bounding_box_oriented.edges_unique_length.min()
                    )

                    npz_to_save["vertices"] = mesh.vertices
                    npz_to_save["faces"] = mesh.faces
                    npz_to_save["min_length"] = min_length
                    npz_to_save["uid"] = uid
                    npz_to_save["vertices_num"] = mesh.vertices.shape[0]
                    npz_to_save["faces_num"] = mesh.faces.shape[0]
                if w:
                    for warn in w:
                        logging.warning(f" {uid} : {str(warn.message)}")
                        print("----------------------------------------")
                        print("uid warning:", uid)
                    return
                # save pc_normal
                print(f"Saving {uid} to {self.out_dir}/{idx}...")
                np.savez(os.path.join(self.out_dir, idx, uid + ".npz"), **npz_to_save)
                # save the uid to a txt file
                with open(os.path.join(self.out_dir, idx, "uid.txt"), "w") as f:
                    f.write(uid)

                # os.remove(temp_path)
                task_end_time = time.time()
                task_duration = task_end_time - task_start_time
                progress.value += 1
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / progress.value * total_tasks
                remaining_time = estimated_total_time - elapsed_time
                remaining_td = datetime.timedelta(seconds=int(remaining_time))
                logging.info(
                    f"This task: {task_duration:.2f} s, Already:{elapsed_time}, progress: {progress.value}/{total_tasks}, remaining{remaining_td}"
                )
        except Exception as e:
            logging.error(f"Error in {uid}: {e}")
        
    def process_objects(self):
        """
        Process a single mesh data item from PCN.
        """
        with open(self.path_json_save_path, "r") as f:
            all_paths = json.load(f)

        cpu_count = os.cpu_count()
        print(f"CPU count: {cpu_count}")
        total_tasks = len(all_paths)

        manager = Manager()
        progress = manager.Value("i", 0)
        start_time = time.time()

        # test_idx = 0
        # test_data = self.idx_uid_path_mapping[str(test_idx)]
        # self.merge_and_scale(test_idx, test_data, progress, start_time, total_tasks)

        with Pool(processes=CPU_COUNT) as pool:
            pool.starmap_async(
                self.merge_and_scale,
                [
                    (idx, data, progress, start_time, total_tasks)
                    for idx, data in self.idx_uid_path_mapping.items()
                ],
            )
            pool.close()
            pool.join()
    
    def process_npz_file(self, npz_metadata):
        try:
            #  get the root folder
            idx, cur_data = npz_metadata
            uid = "_".join(cur_data.strip("/").split("/")[-4:-1])  # synset_id/model_id
            if os.path.exists(os.path.join(self.out_dir, idx, f"idx_{idx}.npz")):
                print(
                    f"File {os.path.join(self.out_dir, idx, f'idx_{idx}.npz')} already exists. Skipping."
                )
                return

            npz_data = os.path.join(self.out_dir, idx, uid + ".npz")
            
            with np.load(npz_data) as data:
                data_dict = {key: data[key] for key in data}
                print(
                    "data_dict vertices shape in process_npz_file:",
                    data_dict["vertices"].shape,
                )
                
                if data_dict and data["faces"].shape[0] >= 20:
                    data_dict["faces_num"] = data["faces"].shape[0]
                    data_dict["vertices_num"] = data["vertices"].shape[0]
                    data_dict["uid"] = uid
                    # save the npz file to the processed directory
                    npz_out_path = os.path.join(self.out_dir, idx, f"idx_{idx}.npz")
                    np.savez(npz_out_path, **data_dict)
                else:
                    pass
        except Exception as e:
            print(f"Error loading {npz_data}: {e}")

    def to_npz_files(self):
        """
        Convert processed mesh files to npz format, separating into train and test sets.
        """

        # get the npz files
        npz_files = [(str(idx), obj) for idx, obj in self.idx_uid_path_mapping.items()]

        # test_indices = [0]  # just for testing
        # for test_idx in test_indices:
        #     test_data = npz_files[test_idx]
        #     self.process_npz_file(test_data)

        with ThreadPoolExecutor() as executor:
            futures = []
            for metadata in tqdm.tqdm(npz_files, desc="Processing files"):
                # npz_path = os.path.join(self.processed_dir, filename)
                futures.append(executor.submit(self.process_npz_file, metadata))

            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="Processing results"
            ):
                future.result()  # Wait for each future to complete

    def process_mesh(self, metadata):
        idx, cur_data = metadata

        mesh_data = os.path.join(self.out_dir, idx, f"idx_{idx}.npz")
        if not os.path.exists(mesh_data):
            print(
                f"File {mesh_data} does not exist. Skipping."
            )
            return
        mesh_data = np.load(mesh_data, allow_pickle=True)
        vertices = mesh_data["vertices"]

        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max()
        cur_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=mesh_data["faces"],
            force="mesh",
            merge_primitives=True,
        )
        cur_mesh.merge_vertices()
        cur_mesh.update_faces(cur_mesh.nondegenerate_faces())
        cur_mesh.update_faces(cur_mesh.unique_faces())
        cur_mesh.remove_unreferenced_vertices()
        # import pdb; pdb.set_trace()

        if self.do_watertight:
            water_mesh = export_to_watertight(cur_mesh, 7)

            water_p, face_idx = water_mesh.sample(
                self.num_sample_points, return_index=True
            )  # 128 * 128
            water_n = water_mesh.face_normals[face_idx]

            npz_to_save = {}
            pc_normal = np.concatenate([water_p, water_n], axis=-1, dtype=np.float16)
            npz_to_save["pc_normal"] = pc_normal
            npz_to_save["face_idx"] = face_idx

            # also save the water mesh
            water_mesh.export(os.path.join(self.out_dir, idx, f"{idx}_water_mesh.obj"))
            np.savez(os.path.join(self.out_dir, idx, f"{idx}_pc.npz"), **npz_to_save)
            gt_mesh = trimesh.Trimesh(
                vertices=mesh_data["vertices"],
                faces=mesh_data["faces"],
                force="mesh",
                merge_primitives=True,
            )
            gt_mesh.export(os.path.join(self.out_dir, idx, f"{idx}_gt_mesh.obj"))
        else:
            sample_p, face_idx = cur_mesh.sample(
                self.num_sample_points, return_index=True
            )
            sample_n = cur_mesh.face_normals[face_idx]
            npz_to_save = {}
            pc_normal = np.concatenate([sample_p, sample_n], axis=-1, dtype=np.float16)
            npz_to_save["pc_normal"] = pc_normal
            npz_to_save["face_idx"] = face_idx

            np.savez(os.path.join(self.out_dir, idx, f"{idx}_pc.npz"), **npz_to_save)
            gt_mesh = trimesh.Trimesh(
                vertices=mesh_data["vertices"],
                faces=mesh_data["faces"],
                force="mesh",
                merge_primitives=True,
            )
            gt_mesh.export(os.path.join(self.out_dir, idx, f"{idx}_gt_mesh.obj"))

    def extract_point_cloud(self):
        print(f"Using {CPU_COUNT} CPU cores")
        npz_files = [(str(idx), obj) for idx, obj in self.idx_uid_path_mapping.items()]
        # test_indices = [0]  # just for testing
        # for test_idx in test_indices:
        #     test_data = npz_files[test_idx]
        #     self.process_mesh(test_data)
        with ThreadPoolExecutor() as executor:
            futures = []
            for metadata in tqdm.tqdm(npz_files, desc="Processing files"):
                futures.append(executor.submit(self.process_mesh, metadata))

            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="Processing results"
            ):
                future.result()  # Wait for each future to complete

    def generate_visibility_samples(self, metadata):
        idx, cur_data = metadata
        # load the mesh data
        if not os.path.exists(os.path.join(self.out_dir, idx, f"{idx}_pc.npz")):
            print(
                f"File {os.path.join(self.out_dir, idx, f'{idx}_pc.npz')} does not exist. Skipping."
            )
            return
        if self.do_watertight:
            water_mesh_sampled = np.load(
                os.path.join(self.out_dir, idx, f"{idx}_pc.npz"), allow_pickle=True
            )
            face_idx = water_mesh_sampled["face_idx"]
            water_p = water_mesh_sampled["pc_normal"][:, :3]
            water_n = water_mesh_sampled["pc_normal"][:, 3:]
        else:
            mesh_sampled = np.load(
                os.path.join(self.out_dir, idx, f"{idx}_pc.npz"), allow_pickle=True
            )
            face_idx = mesh_sampled["face_idx"]
            water_p = mesh_sampled["pc_normal"][:, :3]
            water_n = mesh_sampled["pc_normal"][:, 3:]

        # gt_mesh = trimesh.load(
        #     os.path.join(self.out_dir, idx, f"{idx}_gt_mesh.obj"), force="mesh"
        # )
        mesh_data = os.path.join(self.out_dir, idx, f"idx_{idx}.npz")
        mesh_data = np.load(mesh_data, allow_pickle=True)
        vertices = mesh_data["vertices"]

        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max()
        gt_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=mesh_data["faces"],
            force="mesh",
            merge_primitives=True,
        )
        gt_mesh.merge_vertices()
        gt_mesh.update_faces(gt_mesh.nondegenerate_faces())
        gt_mesh.update_faces(gt_mesh.unique_faces())
        gt_mesh.remove_unreferenced_vertices()

        gt_points_list = []
        gt_normals_list = []
        visible_points_list = []
        visible_normals_list = []
        invisible_points_list = []
        invisible_normals_list = []
        gt_mesh_list = []  # remember to save the original mesh, not the watertight mesh

        # obj_path = os.path.join(self.out_dir, idx, 'model_normalized.obj')
        # gt_mesh = trimesh.load(obj_path, force='mesh')
        if self.do_watertight:
            water_mesh = trimesh.load(
                os.path.join(self.out_dir, idx, f"{idx}_water_mesh.obj"), force="mesh"
            )
            vertices = torch.tensor(
                water_mesh.vertices, dtype=torch.float32, device="cuda"
            )
            faces = torch.tensor(water_mesh.faces, dtype=torch.int32, device="cuda")
        else:
            vertices = torch.tensor(
                gt_mesh.vertices, dtype=torch.float32, device="cuda"
            )
            faces = torch.tensor(gt_mesh.faces, dtype=torch.int32, device="cuda")

        vertices = vertices[None]
        faces = faces[None]

        rast_space_dict = {}
        glctx = dr.RasterizeCudaContext()  # Create the OpenGL context for nvdiffrast
        rast_space_dict["tri_verts"] = faces
        rast_space_dict["glctx"] = glctx

        visible_face_ids_per_view = compute_visibility(
            vertices, self.intrs, self.extrs, rast_space_dict
        )
        # all outer faces:
        merged_faces = torch.unique(torch.cat(visible_face_ids_per_view)).cpu().numpy()
        # all_visible_face_count = merged_faces.shape[0]
        all_visible_face_count = faces.shape[1]
        
        for i in range(len(visible_face_ids_per_view)):
            visible_face_ids = visible_face_ids_per_view[i].cpu().numpy()

            mask = np.isin(face_idx, visible_face_ids)
            visible_points = water_p[mask]
            visible_normals = water_n[mask]
            # get invisible points as well
            invisible_points = water_p[~mask]
            invisible_normals = water_n[~mask]

            if len(visible_points) < 0.1 * self.num_sample_points:
                print("Too limited visible points for view", i, "in", idx)
                # continue
            # transform the points and normals to egocentric coordinates
            extr_i = self.extrs[0, i].cpu().numpy()  # w_T_c
            # get the inverse of the extrinsic matrix
            extr_i_w = np.eye(4)
            extr_i_w[:3, :3] = extr_i[:3, :3].T
            extr_i_w[:3, 3] = -np.dot(extr_i[:3, :3].T, extr_i[:3, 3])

            # transform visible points and normals
            visible_points = np.dot(visible_points, extr_i_w[:3, :3]) + extr_i[:3, 3]
            visible_normals = np.dot(visible_normals, extr_i_w[:3, :3])
            # transform invisible points and normals
            invisible_points = (
                np.dot(invisible_points, extr_i_w[:3, :3]) + extr_i[:3, 3]
            )
            invisible_normals = np.dot(invisible_normals, extr_i_w[:3, :3])

            # make the points centered around the origin
            center = visible_points.mean(axis=0)
            visible_points -= center
            invisible_points -= center

            visible_points_list.append(visible_points)
            visible_normals_list.append(visible_normals)
            invisible_points_list.append(invisible_points)
            invisible_normals_list.append(invisible_normals)

            # create structured gt: first N visible points, then invisible points
            gt_water_p = np.concatenate([visible_points, invisible_points], axis=0)
            gt_water_n = np.concatenate([visible_normals, invisible_normals], axis=0)
            gt_points_list.append(gt_water_p)
            gt_normals_list.append(gt_water_n)
            # corresponding gt mesh
            # transform the gt mesh vertices to the camera coordinate
            gt_mesh_vertices = (
                np.dot(gt_mesh.vertices, extr_i_w[:3, :3]) + extr_i[:3, 3]
            )

            # this is the dummy center purely for visualization
            # gt mesh is shifted and scale in training, as seen in meshanything_train/loop_set_256.py line 181-219
            # But the pc is shifted to another center
            gt_mesh_vertices = (
                gt_mesh_vertices - center
            )  # make the points centered around the origin
            gt_mesh_faces = gt_mesh.faces.copy()
            # create a trimesh object for the gt mesh
            gt_mesh_i = trimesh.Trimesh(
                vertices=gt_mesh_vertices,
                faces=gt_mesh_faces,
                force="mesh",
                merge_primitives=True,
            )
            gt_mesh_list.append(gt_mesh_i)
        
        print(f"Done processing {idx}, save to {self.out_dir}/{idx}")
        
        if self.visualize:
            os.makedirs(
                os.path.join(self.out_dir, idx, "pc_visualization"), exist_ok=True
            )
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(water_p)
            pcd.normals = o3d.utility.Vector3dVector(water_n)
            o3d.io.write_point_cloud(
                os.path.join(self.out_dir, idx, "pc_visualization", f"{idx}_pc.ply"),
                pcd,
            )

            for i in range(len(gt_points_list)):
                gt_points = gt_points_list[i]
                gt_normals = gt_normals_list[i]
                visible_points = visible_points_list[i]
                visible_normals = visible_normals_list[i]
                invisible_points = invisible_points_list[i]
                invisible_normals = invisible_normals_list[i]
                gt_mesh_i = gt_mesh_list[i]

                gt_pc_normal = np.concatenate(
                    [gt_points, gt_normals], axis=-1, dtype=np.float16
                )
                pc_normal = np.concatenate(
                    [visible_points, visible_normals], axis=-1, dtype=np.float16
                )
                invisible_pc_normal = np.concatenate(
                    [invisible_points, invisible_normals], axis=-1, dtype=np.float16
                )
                gt_npz_to_save = {
                    "gt_pc_normal": gt_pc_normal,
                }
                npz_to_save = {
                    "pc_normal": pc_normal,
                }
                invisible_npz_to_save = {
                    "invisible_pc_normal": invisible_pc_normal,
                }
                np.savez(
                    os.path.join(self.out_dir, idx, f"view{i}_gt_pc.npz"),
                    **gt_npz_to_save,
                )
                np.savez(
                    os.path.join(self.out_dir, idx, f"view{i}_pc.npz"), **npz_to_save
                )
                np.savez(
                    os.path.join(self.out_dir, idx, f"view{i}_invisible_pc.npz"),
                    **invisible_npz_to_save,
                )

                if self.visualize:
                    # save the point cloud and normals
                    # os.makedirs(os.path.join(self.out_dir, idx, "pc_visualization"), exist_ok=True)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(visible_points)
                    pcd.normals = o3d.utility.Vector3dVector(visible_normals)
                    o3d.io.write_point_cloud(
                        os.path.join(
                            self.out_dir, idx, "pc_visualization", f"view{i}_pc.ply"
                        ),
                        pcd,
                    )
                    # save the gt point cloud and normals
                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(gt_points)
                    pcd_gt.normals = o3d.utility.Vector3dVector(gt_normals)
                    o3d.io.write_point_cloud(
                        os.path.join(
                            self.out_dir, idx, "pc_visualization", f"view{i}_gt_pc.ply"
                        ),
                        pcd_gt,
                    )
                    # save the invisible point cloud and normals
                    pcd_invisible = o3d.geometry.PointCloud()
                    pcd_invisible.points = o3d.utility.Vector3dVector(invisible_points)
                    pcd_invisible.normals = o3d.utility.Vector3dVector(
                        invisible_normals
                    )
                    o3d.io.write_point_cloud(
                        os.path.join(
                            self.out_dir,
                            idx,
                            "pc_visualization",
                            f"view{i}_invisible_pc.ply",
                        ),
                        pcd_invisible,
                    )

                # save the gt mesh
                gt_mesh_i.export(
                    os.path.join(self.out_dir, idx, f"view{i}_gt_mesh.obj")
                )
        return

    def compute_visibility(self):
        npz_files = [(str(idx), obj) for idx, obj in self.idx_uid_path_mapping.items()]

        self.extrs = self.extrs.cuda() if torch.cuda.is_available() else self.extrs
        self.intrs = self.intrs.cuda() if torch.cuda.is_available() else self.intrs

        # test_indices = [0]  # just for testing
        # for test_idx in test_indices:
        #     test_data = npz_files[test_idx]
        #     self.generate_visibility_samples(test_data)

        for metadata in tqdm.tqdm(npz_files, desc="Processing files"):
            self.generate_visibility_samples(metadata)
    
    def process(self):
        self.filt_objects()
        self.process_objects()
        self.to_npz_files()
        self.extract_point_cloud()
        self.compute_visibility()


if __name__ == "__main__":
    """
    Example usage:
    python data_processor.py
    This generates a series of files:
        uid.txt: contains the uid of the mesh
        {uid}.npz: raw mesh files directly read from ShapeNet
        idx_{idx}.npz: processed mesh files with vertices, faces, min_length, uid, vertices_num, faces_num
        {idx}_pc.npz: point cloud with normals sampled from the water mesh
        {idx}_water_mesh.obj: watertight mesh created from the original mesh
        view{i}_pc.npz: point cloud with normals sampled from the water mesh for each view after visibility computation
        view{i}_gt_pc.npz: ground truth point cloud with normals for each view after visibility computation
        view{i}_gt_mesh.obj: ground truth mesh for each view after visibility computation
        pc_visualization: a folder containing the point cloud and normals for visualization purposes
    Configure the parameters in the prepare_args() function as needed for convenience.
    """
    args = prepare_args()

    processor_class = eval(f"{args.dataset_processor}Processor")
    processor = processor_class(args)
    print(f"begin processing dataset {args.dataset_processor}...")

    processor.process()
    # processor.test_filtered_by_visible_faces()

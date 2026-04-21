import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import trimesh
from shapeatlas_utils.utils import mesh_sort
import argparse
import ipdb
from shapeatlas_utils.data_processor import *
from shapeatlas_utils.compute_visibility import *
import pytorch3d as pt3d
from pytorch3d.renderer import cameras as pt3d_cameras
import open3d as o3d
import time

if __name__ == "__main__":
    print('Testing compute_visibility...')
    time_start = time.time()
    file_name = 'data/shapenet_test/train.npz'
    npz_list = np.load(file_name, allow_pickle=True)
    npz_list = npz_list['npz_list'].tolist()

    test_idx = 1145
    mesh_data = npz_list[test_idx]

    vertices = mesh_data['vertices']
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    end_time = time.time()
    print(f'Loaded mesh data in {end_time - time_start:.2f} seconds')
    print("Trimesh")
    time_start = time.time()
    cur_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh_data['faces'], force='mesh', merge_primitives=True)
    end_time = time.time()
    print(f'Created trimesh in {end_time - time_start:.2f} seconds')
    print(f'Processing mesh {test_idx} with {len(cur_mesh.vertices)} vertices and {len(cur_mesh.faces)} faces')

    time_start = time.time()
    water_mesh = export_to_watertight(cur_mesh, 7)
    end_time = time.time()
    print(f'Exported to watertight mesh in {end_time - time_start:.2f} seconds')
    water_p, face_idx = water_mesh.sample(20000, return_index=True)
    water_n = water_mesh.face_normals[face_idx]

    vertices = torch.tensor(water_mesh.vertices, dtype=torch.float32, device='cuda')
    faces = torch.tensor(water_mesh.faces, dtype=torch.int32, device='cuda')

    vertices = vertices[None]  # Add batch dimensions
    faces = faces[None] # Add batch dimension

    rast_space_dict = {}
    glctx = dr.RasterizeCudaContext()  # Create the OpenGL context for nvdiffrast
    rast_space_dict['tri_verts'] = faces
    rast_space_dict['glctx'] = glctx

    image_size = 512
    fov_deg =60
    print('Calculating camera intrinsics...')
    fov_rad = math.radians(fov_deg)
    focal_length = (image_size / 2) / math.tan(fov_rad / 2)
    cx = cy = image_size / 2
    intrinsic = torch.tensor([
        [focal_length, 0.0, cx],
        [0.0, focal_length, cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=vertices.device)
    intr = intrinsic[None]  # Add batch dimensions


    num_views = 4
    num_elev = 4
    
    print(f'Generating {num_views} views with {num_elev} elevations each...')
    clip_interval = 360 // num_views
    elev_interval = 150 // (num_elev - 1) if num_elev > 1 else 0
    
    azim_list = []
    elev_list = []
    look_at_center_list = []
    look_at_center_default = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=vertices.device)

    # set random seed for reproducibility
    np.random.seed(0)

    for i in range(num_views):
        azim = float(i * clip_interval)
        for j in range(num_elev):
            # range from -150 / 2 to 150 / 2
            elev = float(j * elev_interval) - 150.0 / 2 if num_elev > 1 else 0.0
            azim_list.append(azim)
            elev_list.append(elev)
            # randomly offset the look at center between -0.1 and 0.1
            look_at_center_offset = torch.tensor([
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.1, 0.1)
            ], dtype=torch.float32, device=vertices.device)
            look_at_center = look_at_center_default + look_at_center_offset
            look_at_center_list.append(look_at_center)
    azim_list = torch.tensor(azim_list, dtype=torch.float32, device=vertices.device)
    elev_list = torch.tensor(elev_list, dtype=torch.float32, device=vertices.device)
    look_at_center_list = torch.stack(look_at_center_list, dim=0)

    # use pytorch3d to generate the camera extrinsics
    Rs, Ts = pt3d_cameras.look_at_view_transform(
        dist=1.0,
        elev=elev_list,
        azim=azim_list,
        at=look_at_center_list,
        device=vertices.device
    )

    # stack the camera extrinsics
    extrs = torch.cat([Rs, Ts[..., None]], dim=-1)  # shape: (num_views * num_elev, 3, 4)
    extr = extrs[None]  # Add batch and frame dimensions
    N = extr.shape[1]  # Number of views
    # add frame dimension to intrinsics so it is compatible with extrinsics (B, 3, 3) -> (B, F, 3, 3)
    intr = intr[None].repeat(1, N, 1, 1)

    print('Computing visibility for each view...')
    visible_face_ids_per_view = compute_visibility(vertices, intr, extr, rast_space_dict)
    # for each view, filter the visible points and normals
    visible_points_list = []
    visible_normals_list = []
    for i in range(len(visible_face_ids_per_view)):
        visible_face_ids = visible_face_ids_per_view[i].cpu().numpy()

        mask = np.isin(face_idx, visible_face_ids)
        visible_points = water_p[mask]
        visible_normals = water_n[mask]
        # transform the points and normals to egocentric coordinates
        extr_i = extr[0, i].cpu().numpy() # w_T_c
        # get the inverse of the extrinsic matrix
        extr_i_w = np.eye(4)
        extr_i_w[:3, :3] = extr_i[:3, :3].T
        extr_i_w[:3, 3] = -np.dot(extr_i[:3, :3].T, extr_i[:3, 3])
        visible_points = np.dot(visible_points, extr_i_w[:3, :3]) + extr_i[:3, 3]
        visible_normals = np.dot(visible_normals, extr_i_w[:3, :3])
        # make the points centered around the origin
        center = visible_points.mean(axis=0)
        visible_points -= center

        visible_points_list.append(visible_points)
        visible_normals_list.append(visible_normals)
    
    os.makedirs('output/ego_view', exist_ok=True)
    print('Saving visible points and normals...')
    # save the visible points and normals as a ply file
    for i, (visible_points, visible_normals) in enumerate(zip(visible_points_list, visible_normals_list)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(visible_points)
        pcd.normals = o3d.utility.Vector3dVector(visible_normals)
        o3d.io.write_point_cloud(f'output/ego_view/visible_points_view_{i}.ply', pcd)


# adapted from https://github.com/GaussianCube/GaussianCube_Construction
# originally used for Gaussian points
# adapted for point clouds for mesh generation: points + normals
import argparse
import concurrent.futures
import multiprocessing
import os
import pdb
import time
from multiprocessing import Pool
from typing import NamedTuple

import ipdb
import numpy as np
import open3d as o3d
import ot
import torch
from lapjv import lapjv
from fastlapjv import fastlapjv
import lap
from plyfile import PlyData
from pytorch3d.ops import sample_farthest_points
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image

N_SQRT = 128
N = N_SQRT * N_SQRT

transform = T.Compose([
    T.Resize((N_SQRT, N_SQRT)),
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225]) 
])
num_segments = 4
# segment_size = 8192
segment_size = N // num_segments

class GaussianPointCloud(NamedTuple):
    xyz : torch.tensor
    features_dc : torch.tensor
    features_rest : torch.tensor
    opacity : torch.tensor
    scaling : torch.tensor
    rotation : torch.tensor

class PointCloud(NamedTuple):
    xyz : torch.tensor
    normal : torch.tensor

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def inverse_scaling(x):
    return torch.log(x)


def process_single_wrapper(line, source_root, save_root, max_sh_degree, bound, visualize_mapping):  
    # Call the original processing function with the provided arguments  
    process_single(os.path.join(source_root, line), save_root, max_sh_degree, bound, visualize_mapping)

def process_pcd_wrapper_new(line, source_root, save_root, sphere_points=None, visualize_mapping=False):
    
    process_pcd2sphere_new(os.path.join(source_root, line), save_root, sphere_points=sphere_points, visualize_mapping=visualize_mapping)

def process_single_pcd_wrapper(line, source_root, save_root, sphere_points=None, visualize_mapping=False):
    # Call the original processing function with the provided arguments  
    process_single_pcd2sphere(os.path.join(source_root, line), save_root, sphere_points=sphere_points, visualize_mapping=visualize_mapping)

def init_volume_grid(bound=0.45, num_pts_each_axis=32):
    # Define the range and number of points  
    start = -bound
    stop = bound
    num_points = num_pts_each_axis  # Adjust the number of points to your preference  
    
    # Create a linear space for each axis  
    x = np.linspace(start, stop, num_points)  
    y = np.linspace(start, stop, num_points)  
    z = np.linspace(start, stop, num_points)  
    
    # Create a 3D grid of points using meshgrid  
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  
    
    # Stack the grid points in a single array of shape (N, 3)  
    xyz = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T  
    
    return xyz

def init_unit_sphere_grid(N=N):
    # fibonacci_sphere 
    indices = np.arange(0, N)
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle ~2.399963
    y = 1 - (indices / (N - 1)) * 2  # y goes from 1 to -1
    radius = np.sqrt(1 - y * y)
    
    theta = phi * indices
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    xyz = np.stack((x, z, y), axis=1)
    # note np.lexsort sort by the last key first in descending order
    sorted_indices = np.lexsort((-xyz[:, 0], -xyz[:, 1], -xyz[:, 2]))  # sort by z, y, x -> in xyz[x, y, z]  = [x, y, z]
    # keys = (
    #     xyz[:, 0],
    #     np.floor(xyz[:,0]).astype(int),
    #     np.floor(xyz[:,1]).astype(int),
    #     np.floor(xyz[:,2]).astype(int)
    # )
    # sorted_indices = np.lexsort(keys)
    xyz = xyz[sorted_indices]  # !!!! keep the order 
    return torch.tensor(xyz, dtype=torch.float, device="cpu")

def equirectangular_projection(sphere_points):
    # for each point in sphere_points, compute the longitude and latitude
    x, y, z = sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2]
    lon = torch.atan2(y, x)
    lat = torch.asin(z)

    # convert to degrees
    lon = torch.rad2deg(lon) # 
    lat = torch.rad2deg(lat) # south to north
    # return the shifted longitude and latitude to [-180, 180] and [-90, 90]
    lon = (lon + 180) 
    lat = (lat + 90) 
    # return torch.stack((lon, lat), dim=1)  # shape: (N, 2), 1 to 1 mapping from sphere points
    return torch.stack((lat, lon), dim=1)  # shape: (N, 2), 1 to 1 mapping from sphere points


def get_grid_points(N_sqrt=N_SQRT):
    x = np.arange(N_sqrt)
    y = np.arange(N_sqrt)
    col_indices, row_indices = np.meshgrid(x, y)
    # Stack to get (row, col) pairs and reshape to (N, 2)
    grid_points = np.stack((row_indices, col_indices), axis=-1).reshape(-1, 2) # shape: (N_sqrt*N_sqrt, 2)
    grid_sorted_indices = np.lexsort((grid_points[:, 0], grid_points[:, 1]))  # sort by row and column indices
    grid_points = grid_points[grid_sorted_indices]  
    
    # rescale grid points to [-0.5, 0.5] for color mapping
    grid_x = (grid_points[:, 0] - grid_points[:, 0].min()) / (grid_points[:, 0].max() - grid_points[:, 0].min()) - 0.5
    grid_y = (grid_points[:, 1] - grid_points[:, 1].min()) / (grid_points[:, 1].max() - grid_points[:, 1].min()) - 0.5
    grid_points = np.column_stack((grid_x, grid_y, np.ones_like(grid_x)))  # Add dummy zeros for z-coordinate
    return grid_points  # shape: (N_sqrt*N_sqrt, 3), 1 to 1 mapping from grid points to sphere points

def simple_ot_map_eq2grid(latlon_points, N_sqrt=N_SQRT, scaling_factor=1, sort=False):
    # latlon_points: (N, 2) tensor of latitude and longitude points
    # grid_points: (N, 2) tensor of grid points indices (N_sqrt, N_sqrt)
    assert latlon_points.shape[0] == N_SQRT * N_SQRT, "latlon_points should have shape (N_sqrt*N_sqrt, 2)"
    # generate grid points
    x = np.arange(N_sqrt)
    y = np.arange(N_sqrt)
    col_indices, row_indices = np.meshgrid(x, y)
    # Stack to get (row, col) pairs and reshape to (N, 2)
    # Be careful with the ordering. Wrong order causes wrong mapping due to the segmentation below
    grid_points = np.stack((row_indices, col_indices), axis=-1).reshape(-1, 2).astype(np.float32) # shape: (N_sqrt*N_sqrt, 2)
    grid_sorted_indices = np.lexsort((grid_points[:, 1], grid_points[:, 0]))  # sort by row and column indices
    grid_points = grid_points[grid_sorted_indices]

    latlon_points = latlon_points.cpu().numpy()
    if sort:
        latlon_sorted_indices = np.lexsort((latlon_points[:, 1], latlon_points[:, 0])) 
        reversed_latlon_sorted_indices = np.argsort(latlon_sorted_indices)  # to map back to original order
        sorted_latlon_points = latlon_points[latlon_sorted_indices] 
        cost, corrs_1_to_2, corrs_2_to_1, index = compute_lap(sorted_latlon_points, grid_points, scaling_factor=scaling_factor, method='lapjv')
    else:
        cost, corrs_1_to_2, corrs_2_to_1, index = compute_lap(latlon_points, grid_points, scaling_factor=scaling_factor, method='lapjv')
    if sort:
        return corrs_2_to_1, corrs_1_to_2, latlon_sorted_indices, reversed_latlon_sorted_indices
    return corrs_2_to_1, corrs_1_to_2, None, None

def ot_map_eq2grid(latlon_points, N_sqrt=N_SQRT, scaling_factor=1):
    # latlon_points: (N, 2) tensor of latitude and longitude points
    # grid_points: (N, 2) tensor of grid points indices (N_sqrt, N_sqrt)
    assert latlon_points.shape[0] == N_SQRT * N_SQRT, "latlon_points should have shape (N_sqrt*N_sqrt, 2)"
    # generate grid points
    x = np.arange(N_sqrt)
    y = np.arange(N_sqrt)
    col_indices, row_indices = np.meshgrid(x, y)
    # Stack to get (row, col) pairs and reshape to (N, 2)
    # Be careful with the ordering. Wrong order causes wrong mapping due to the segmentation below
    grid_points = np.stack((row_indices, col_indices), axis=-1).reshape(-1, 2).astype(np.float32) # shape: (N_sqrt*N_sqrt, 2)
    grid_sorted_indices = np.lexsort((grid_points[:, 1], grid_points[:, 0]))  # sort by row and column indices
    grid_points = grid_points[grid_sorted_indices]
    
    latlon_points = latlon_points.cpu().numpy()
    latlon_sorted_indices = np.lexsort((latlon_points[:, 1], latlon_points[:, 0])) 
    reversed_latlon_sorted_indices = np.argsort(latlon_sorted_indices)  # to map back to original order
    sorted_latlon_points = latlon_points[latlon_sorted_indices] 
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  
        futures = {executor.submit(compute_lap,  
                                sorted_latlon_points[i*segment_size:(i+1)*segment_size],  
                                grid_points[i*segment_size:(i+1)*segment_size],  
                                scaling_factor=scaling_factor, index=i, method='lapjv'): i for i in range(num_segments)}  
    
        results = [None] * num_segments  # Prepare a list to hold results in order  
        for future in concurrent.futures.as_completed(futures):  
            cost, x, y, index = future.result()  
            results[index] = (cost, x, y)  # Store the results in the corresponding index 
    corrs_2_to_1 = np.concatenate([y + i*segment_size for i, (cost, x, y) in enumerate(results)], axis=0)
    corrs_1_to_2 = np.concatenate([x + i*segment_size for i, (cost, x, y) in enumerate(results)], axis=0)
    # one to one mapping from sorted latlon_points to grid_points

    # latlon[latlon_sorted_indices][corrs_2_to_1] gives the grid points corresponding to the latlon points
    # grid_points[corrs_1_to_2][reversed_latlon_sorted_indices] gives the latlon points corresponding to the grid points
    return corrs_2_to_1, corrs_1_to_2, latlon_sorted_indices, reversed_latlon_sorted_indices


def compute_lap(p1_segment, p2_segment, scaling_factor=1000, index=0, method="fastlapjv"):  
    cost_matrix = ot.dist(p1_segment, p2_segment, metric='sqeuclidean')  
    # Scale to integers for faster computation
    scaled_cost_matrix = np.rint(cost_matrix * scaling_factor).astype(int)   
    if method == "fastlapjv":
        x, y, cost = fastlapjv(scaled_cost_matrix, k_value=50) # fast lapjv
    elif method == "scipy":
        x, y = linear_sum_assignment(scaled_cost_matrix)
        cost = scaled_cost_matrix[x, y].sum()
    elif method == "lapjv":
        x, y, cost = lapjv(scaled_cost_matrix) # original lapjv
    elif method == "lap":
        cost, x, y = lap.lapjv(scaled_cost_matrix, extend_cost=False, cost_limit=scaling_factor) # from lap
    else: 
        raise ValueError("Unknown method: {}".format(method))
        
    # p1_segment[x]: reorder p1 to match p2
    # p2_segment[y]: reorder p2 to match p1
    return cost, x, y, index

def load_npz(path):
    data = np.load(path)
    if 'pc_normal' in data.keys():
        pc_normal = data["pc_normal"]
    elif 'gt_pc_normal' in data.keys():
        pc_normal = data["gt_pc_normal"]
    elif 'invisible_pc_normal' in data.keys():
        pc_normal = data["invisible_pc_normal"]
    else:
        raise ValueError("The npz file does not contain 'pc_normal' or 'gt_pc_normal' key.")
    # pc_normal = data["pc_normal"]
    points = pc_normal[:, :3]
    normals = pc_normal[:, 3:]
    return PointCloud(torch.tensor(points, dtype=torch.float), torch.tensor(normals, dtype=torch.float))

def load_ply(path, max_sh_degree=3):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3))
    features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # (P,F*SH_coeffs)
    features_extra = features_extra

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = torch.tensor(xyz, dtype=torch.float, device="cpu")
    features_dc = torch.tensor(features_dc, dtype=torch.float, device="cpu").contiguous()
    features_rest = torch.tensor(features_extra, dtype=torch.float, device="cpu").contiguous()
    opacity = torch.tensor(opacities, dtype=torch.float, device="cpu")
    scaling = torch.tensor(scales, dtype=torch.float, device="cpu")
    rotation = torch.tensor(rots, dtype=torch.float, device="cpu")
    print("xyz shape: {} \t features_dc shape: {} \t features_rest shape: {} \t opacity shape: {} \t scaling shape: {} \t rotation shape: {}".format(xyz.shape, features_dc.shape, features_rest.shape, opacity.shape, scaling.shape, rotation.shape))
    return GaussianPointCloud(xyz, features_dc, features_rest, opacity, scaling, rotation)

def padding_and_prune_input(point_cloud, num_pts=N):
    xyz = point_cloud.xyz
    if xyz.shape[0] > num_pts:
        # use pytorch3d to sample farthest points
        _, sample_indices = sample_farthest_points(xyz.unsqueeze(0), K=num_pts)
        new_xyz = xyz[sample_indices[0]]
        new_normal = point_cloud.normal[sample_indices[0]]
        new_point_cloud = PointCloud(new_xyz, new_normal)
        return new_point_cloud

    elif xyz.shape[0] == num_pts:
        return point_cloud
    else:
        padding_num = num_pts - xyz.shape[0]
        padding_xyz = torch.zeros((padding_num, 3), dtype=torch.float)
        # using length 1 normal for padding
        base = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float)
        padding_normal = base.unsqueeze(0).repeat(padding_num, 1)

        new_xyz = torch.cat([xyz, padding_xyz], dim=0)
        new_normal = torch.cat([point_cloud.normal, padding_normal], dim=0)

        new_point_cloud = PointCloud(new_xyz, new_normal)
        return new_point_cloud

def padding_input(point_cloud, num_pts=32768, bound=0.45):
    xyz = point_cloud.xyz
    if xyz.shape[0] > num_pts:
        raise ValueError("The number of points in the input point cloud is larger than the maximum number of points allowed.")
    elif xyz.shape[0] == num_pts:
        return point_cloud
    else:
        padding_num = num_pts - xyz.shape[0]
        padding_xyz = torch.tensor([bound, bound, bound], dtype=torch.float).unsqueeze(0).repeat(padding_num, 1)
        padding_features_dc = torch.zeros((padding_num, point_cloud.features_dc.shape[1]), dtype=torch.float)
        padding_features_rest = torch.zeros((padding_num, point_cloud.features_rest.shape[1]), dtype=torch.float)
        padding_opacity = inverse_sigmoid(torch.ones((padding_num, point_cloud.opacity.shape[1]), dtype=torch.float) * 1e-6)
        padding_scaling = inverse_scaling(torch.ones((padding_num, point_cloud.scaling.shape[1]), dtype=torch.float) * 1e-6)
        padding_rotation = torch.nn.functional.normalize(torch.ones((padding_num, point_cloud.rotation.shape[1]), dtype=torch.float))
        new_xyz = torch.cat([xyz, padding_xyz], dim=0)
        new_features_dc = torch.cat([point_cloud.features_dc, padding_features_dc], dim=0)
        new_features_rest = torch.cat([point_cloud.features_rest, padding_features_rest], dim=0)
        new_opacity = torch.cat([point_cloud.opacity, padding_opacity], dim=0)
        new_scaling = torch.cat([point_cloud.scaling, padding_scaling], dim=0)
        new_rotation = torch.cat([point_cloud.rotation, padding_rotation], dim=0)
        new_point_cloud = GaussianPointCloud(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        return new_point_cloud



def lag_segment_matching(p1, p2):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  
        futures = {executor.submit(compute_lap,  
                                p1[i*segment_size:(i+1)*segment_size],  
                                p2[i*segment_size:(i+1)*segment_size],  
                                scaling_factor=1000, index=i): i for i in range(num_segments)}  
    
        results = [None] * num_segments  # Prepare a list to hold results in order  
        for future in concurrent.futures.as_completed(futures):  
            cost, x, y, index = future.result()  
            results[index] = (cost, x, y)  # Store the results in the corresponding index 
    corrs_2_to_1 = np.concatenate([y + i*segment_size for i, (cost, x, y) in enumerate(results)], axis=0)
    corrs_1_to_2 = np.concatenate([x + i*segment_size for i, (cost, x, y) in enumerate(results)], axis=0)
    return corrs_2_to_1, corrs_1_to_2

def activate_volume(volume, max_sh_degree=0):
    scaling_activation = torch.exp
    opacity_activation = torch.sigmoid
    rotation_activation = torch.nn.functional.normalize

    sh_dim = 3 * ((max_sh_degree + 1) ** 2 - 1)
    H, W, D, C = volume.shape
    volume = volume.reshape(-1, volume.shape[-1])
    volume[..., 6+sh_dim:7+sh_dim] = opacity_activation(volume[..., 6+sh_dim:7+sh_dim])
    volume[..., 7+sh_dim:10+sh_dim] = scaling_activation(volume[..., 7+sh_dim:10+sh_dim])
    volume[..., 10+sh_dim:] = rotation_activation(volume[..., 10+sh_dim:])
    volume = volume.reshape(H, W, D, -1)
    return volume


def process_pcd2sphere_new(source_root, save_root, sphere_points=None, visualize_mapping=False, do_visibility=True, num_view=16):
    '''
    map point cloud to unit sphere using OT
    '''
    # retrieve the index of the files from source_root
    data_index = os.path.basename(source_root).split('.')[0]
    os.makedirs(os.path.join(save_root, data_index, "spherical_ot"), exist_ok=True)
    if visualize_mapping:
        os.makedirs(os.path.join(save_root, data_index, "spherical_ot_visualization"), exist_ok=True)
    
    # if os.path.exists(os.path.join(save_root, data_index, "spherical_ot", f'{data_index}_pc.pt')):
    #     print("File {} already exists. Skip.".format(os.path.join(save_root, data_index, "spherical_ot", f'{data_index}_pc.pt')))
    #     return
    if sphere_points is None:
        sphere_points = init_unit_sphere_grid(N)
    
    data_list = []
    data_list.append(f'{data_index}_pc.npz')
    if do_visibility:
        for i in range(num_view):
            data_list.append(f'view{i}_pc.npz')
            # data_list.append(f'view{i}_gt_pc.npz')
            data_list.append(f'view{i}_invisible_pc.npz')
            if i > 4: # for quick debug. remove later
                break
    sphere_points = sphere_points.cpu().numpy()

    for data in data_list:
        data_path = os.path.join(source_root, data)
        source_pcd = load_npz(data_path)
        data_name = data.split('.')[0]
        pcd_padded = padding_and_prune_input(source_pcd, num_pts=N)
        xyz = pcd_padded.xyz.cpu().numpy()
        sorted_indices = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
        xyz = xyz[sorted_indices]
        start = time.time()
        corrs_2_to_1, corrs_1_to_2 = lag_segment_matching(xyz, sphere_points)
        print("Time taken: {}".format(time.time() - start))
        
        ####### Point Cloud Visualization ##########
        if visualize_mapping:
            min_old, max_old = -1, 1  
            min_new, max_new = 0, 1  
            p2 = sphere_points.copy()
            p2 = (p2 - min_old) / (max_old - min_old) * (max_new - min_new) + min_new  # rescale to 0 to 1 for color mapping
            colors = p2[corrs_1_to_2]

            mapping_of_p1 = o3d.geometry.PointCloud()  
            mapping_of_p1.points = o3d.utility.Vector3dVector(xyz)  
            mapping_of_p1.colors = o3d.utility.Vector3dVector(colors)  # Assuming 'colors' is a NumPy array with the same length as p1 
            mapping_of_p2 = o3d.geometry.PointCloud()
            mapping_of_p2.points = o3d.utility.Vector3dVector(sphere_points[corrs_1_to_2])
            mapping_of_p2.colors = o3d.utility.Vector3dVector(colors)  # Assuming 'colors' is a NumPy array with the same length as p1

            o3d.io.write_point_cloud(os.path.join(save_root, data_index, "spherical_ot_visualization", f'{data_name}.ply'), mapping_of_p1)
            o3d.io.write_point_cloud(os.path.join(save_root, data_index, "spherical_ot_visualization", f'{data_name}_sphere.ply'), mapping_of_p2)
        ###############################################
        # corrs_2_to_1: transform the first to the second
        xyz_offset = torch.from_numpy(xyz[corrs_2_to_1] - sphere_points)
        new_pcd = PointCloud(xyz_offset, pcd_padded.normal[sorted_indices][corrs_2_to_1])
        new_pcd_dict = {"xyz": new_pcd.xyz, "normal": new_pcd.normal}
        # note there might be distribution shift for incomplete atlas
        torch.save(new_pcd_dict, os.path.join(save_root, data_index, "spherical_ot", f'{data_name}.pt'))
        # also save the original point cloud and mapping
        # for future retrieve mapping for reconstruction
        original_pcd_dict = {"xyz": pcd_padded.xyz, "normal": pcd_padded.normal, "sorted_indices": sorted_indices, "corrs_2_to_1": corrs_2_to_1, "corrs_1_to_2": corrs_1_to_2}
        torch.save(original_pcd_dict, os.path.join(save_root, data_index, "spherical_ot", f'{data_name}_original_and_mapping.pt'))

def process_single_pcd2sphere(source_root, save_root, sphere_points=None, visualize_mapping=False):
    '''
    map point cloud to unit sphere using OT
    '''
    # retrieve the npz file name from source_root
    npz_file_index = os.path.basename(source_root).split('.')[0]
    os.makedirs(os.path.join(save_root, npz_file_index, "spherical_ot"), exist_ok=True)

    if sphere_points is None:
        sphere_points = init_unit_sphere_grid(N)
    
    sphere_points = sphere_points.cpu().numpy()
    source_pcd = load_npz(source_root)
    pcd_padded = padding_and_prune_input(source_pcd, num_pts=N)
    xyz = pcd_padded.xyz.cpu().numpy()
    sorted_indices = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    xyz = xyz[sorted_indices]
    start = time.time()
    corrs_2_to_1, corrs_1_to_2 = lag_segment_matching(xyz, sphere_points)
    print("Time taken: {}".format(time.time() - start))
    
    ####### Point Cloud Visualization ##########
    if visualize_mapping:
        min_old, max_old = -1, 1  
        min_new, max_new = 0, 1  
        p2 = sphere_points.copy()
        p2 = (p2 - min_old) / (max_old - min_old) * (max_new - min_new) + min_new  # rescale to 0 to 1 for color mapping
        colors = p2[corrs_1_to_2]

        mapping_of_p1 = o3d.geometry.PointCloud()  
        mapping_of_p1.points = o3d.utility.Vector3dVector(xyz)  
        mapping_of_p1.colors = o3d.utility.Vector3dVector(colors)  # Assuming 'colors' is a NumPy array with the same length as p1 
        mapping_of_p2 = o3d.geometry.PointCloud()
        mapping_of_p2.points = o3d.utility.Vector3dVector(sphere_points[corrs_1_to_2])
        mapping_of_p2.colors = o3d.utility.Vector3dVector(colors)  # Assuming 'colors' is a NumPy array with the same length as p1
        
        os.makedirs(os.path.join(save_root,npz_file_index, "spherical_ot_visualization"), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(save_root, npz_file_index, "spherical_ot_visualization", os.path.basename(os.path.normpath(source_root)).split('.')[0]+".ply"), mapping_of_p1)
        o3d.io.write_point_cloud(os.path.join(save_root, npz_file_index, "spherical_ot_visualization", os.path.basename(os.path.normpath(source_root)).split('.')[0]+"_sphere.ply"), mapping_of_p2)
    ###############################################
    # corrs_2_to_1: transform the first to the second
    xyz_offset = torch.from_numpy(xyz[corrs_2_to_1] - sphere_points)
    new_pcd = PointCloud(xyz_offset, pcd_padded.normal[sorted_indices][corrs_2_to_1])
    new_pcd_dict = {"xyz": new_pcd.xyz, "normal": new_pcd.normal}
    torch.save(new_pcd_dict, os.path.join(save_root, npz_file_index, "spherical_ot", os.path.basename(os.path.normpath(source_root)).split('.')[0] +".pt"))
    # also save the original point cloud and mapping
    # for future retrieve mapping for reconstruction
    original_pcd_dict = {"xyz": pcd_padded.xyz, "normal": pcd_padded.normal, "sorted_indices": sorted_indices, "corrs_2_to_1": corrs_2_to_1, "corrs_1_to_2": corrs_1_to_2}
    torch.save(original_pcd_dict, os.path.join(save_root, npz_file_index, "spherical_ot", os.path.basename(os.path.normpath(source_root)).split('.')[0] +"_original_and_mapping.pt"))
    return new_pcd

def process_single(source_root, save_root, max_sh_degree=0, bound=0.45, visualize_mapping=False):
    os.makedirs(os.path.join(save_root, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "volume"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "volume_act"), exist_ok=True)
    source_name = os.path.basename(os.path.normpath(source_root))
    source_file = os.path.join(source_root, "point_cloud/iteration_30000/point_cloud.ply")

    if not os.path.exists(source_file):
        print("File {} does not exist. Skip.".format(source_file))
        return
    if os.path.exists(os.path.join(save_root, "volume", source_name+".pt")):
        print("File {} already exists. Skip.".format(source_name))
        return
    
    generated_gaussian = load_ply(source_file, max_sh_degree)
    std_volume = init_volume_grid(bound=bound, num_pts_each_axis=32)
    generated_gaussian_padded = padding_input(generated_gaussian, bound=bound)
    xyz = generated_gaussian_padded.xyz.cpu().numpy()

    sorted_indices = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))  
    xyz = xyz[sorted_indices]
    
    start = time.time()
    corrs_2_to_1, corrs_1_to_2 = lag_segment_matching(xyz, std_volume)
    print("Time taken: {}".format(time.time() - start))

    ########## Point Cloud Visualization ##########
    if visualize_mapping:
        min_old, max_old = -bound, bound  
        min_new, max_new = 0, 1  
        p2 = std_volume.copy()
        p2 = (p2 - min_old) / (max_old - min_old) * (max_new - min_new) + min_new 
        colors = p2[corrs_1_to_2]

        mapping_of_p1 = o3d.geometry.PointCloud()  
        mapping_of_p1.points = o3d.utility.Vector3dVector(xyz)  
        mapping_of_p1.colors = o3d.utility.Vector3dVector(colors)  # Assuming 'colors' is a NumPy array with the same length as p1 
        os.makedirs(os.path.join(save_root, "point_cloud"), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(save_root, "point_cloud", source_name+".ply"), mapping_of_p1) 
    ###############################################
    
    xyz_offset = torch.from_numpy(xyz[corrs_2_to_1] - std_volume)
    new_gaussian = GaussianPointCloud(xyz_offset, generated_gaussian_padded.features_dc[sorted_indices][corrs_2_to_1], generated_gaussian_padded.features_rest[sorted_indices][corrs_2_to_1], generated_gaussian_padded.opacity[sorted_indices][corrs_2_to_1], generated_gaussian_padded.scaling[sorted_indices][corrs_2_to_1], generated_gaussian_padded.rotation[sorted_indices][corrs_2_to_1])

    new_gaussian_dict = {"xyz": new_gaussian.xyz, "features_dc": new_gaussian.features_dc, "features_rest": new_gaussian.features_rest, "opacity": new_gaussian.opacity, "scaling": new_gaussian.scaling, "rotation": new_gaussian.rotation}
    torch.save(new_gaussian_dict, os.path.join(save_root, "raw", source_name+".pt"))
    
    if new_gaussian.features_rest.shape[-1] == 0:
        volume = torch.cat([new_gaussian.xyz, new_gaussian.features_dc, new_gaussian.opacity, new_gaussian.scaling, new_gaussian.rotation], dim=-1).reshape(32, 32, 32, -1)
        act_volume = activate_volume(volume.clone(), max_sh_degree)
    else:
        volume = torch.cat([new_gaussian.xyz, new_gaussian.features_dc, new_gaussian.features_rest, new_gaussian.opacity, new_gaussian.scaling, new_gaussian.rotation], dim=-1).reshape(32, 32, 32, -1)
        act_volume = activate_volume(volume.clone(), max_sh_degree)
    torch.save(volume, os.path.join(save_root, "volume", source_name+".pt"))
    torch.save(act_volume, os.path.join(save_root, "volume_act", source_name+".pt"))

    return new_gaussian

def sphere2grid_new(source_root, save_root, lines, visualize_mapping=False, visualization_order='xyz', do_visibility=True, num_view=16):
    sphere_points = init_unit_sphere_grid()
    latlon = equirectangular_projection(sphere_points)
    corrs_2_to_1, corrs_1_to_2, latlon_sorted_indices, reversed_latlon_sorted_indices = ot_map_eq2grid(latlon_points=latlon)
    # os.makedirs(os.path.join(save_root), exist_ok=True)
    for line in lines:
        print("Processing data: {}".format(line))

        data_list = []
        data_list.append(f'{line}_pc.pt')
        if do_visibility:
            for i in range(num_view):
                data_list.append(f'view{i}_pc.pt')
                data_list.append(f'view{i}_gt_pc.pt')
                data_list.append(f'view{i}_invisible_pc.pt')
                if i > 4: # for quick debug. remove later
                    break
        # load the point cloud
        # npz_file_index = line.split('.')[0]
        # source_file = os.path.join(source_root, f'{npz_file_index}', 'spherical_ot', f'{npz_file_index}.pt')
        for data in data_list:
            data_path = os.path.join(source_root, line, 'spherical_ot', data)
            data_name = data.split('.')[0]
            pcd = torch.load(data_path)
            # ordered the same as sphere_points or latlon
            xyz = pcd["xyz"]
            normal = pcd["normal"]
            if visualize_mapping:
                if visualization_order == 'xyz':
                    os.makedirs(os.path.join(save_root, line, 'atlas_pcd'), exist_ok=True)
                    min_old, max_old = -1, 1  
                    min_new, max_new = 0, 1  
                    p2 = sphere_points.cpu().numpy()
                    p2 = (p2 - min_old) / (max_old - min_old) * (max_new - min_new) + min_new  # rescale to 0 to 1 for color mapping
                    colors = p2
                    grid_points = get_grid_points(N_SQRT)
                    mapping_of_xyz = o3d.geometry.PointCloud()  
                    mapping_of_xyz.points = o3d.utility.Vector3dVector(xyz + sphere_points)
                    mapping_of_xyz.colors = o3d.utility.Vector3dVector(colors) 
                    
                    mapping_of_sph = o3d.geometry.PointCloud()
                    mapping_of_sph.points = o3d.utility.Vector3dVector(sphere_points)
                    mapping_of_sph.colors = o3d.utility.Vector3dVector(colors)  
                    
                    mapping_of_grid = o3d.geometry.PointCloud()
                    mapping_of_grid.points = o3d.utility.Vector3dVector(grid_points[corrs_1_to_2][reversed_latlon_sorted_indices])
                    mapping_of_grid.colors = o3d.utility.Vector3dVector(colors) 

                    o3d.io.write_point_cloud(os.path.join(save_root, line, 'atlas_pcd', f"{data_name}.ply"), mapping_of_xyz)
                    o3d.io.write_point_cloud(os.path.join(save_root, line, 'atlas_pcd', f"{data_name}_sphere.ply"), mapping_of_sph)
                    o3d.io.write_point_cloud(os.path.join(save_root, line, 'atlas_pcd', f"{data_name}_grid.ply"), mapping_of_grid)
                elif visualization_order == 'grid':
                    os.makedirs(os.path.join(save_root, line, 'atlas_pcd_grid_order'), exist_ok=True)
                    min_old, max_old = -1, 1  
                    min_new, max_new = 0, 1  
                    p2 = sphere_points.cpu().numpy()
                    p2 = (p2 - min_old) / (max_old - min_old) * (max_new - min_new) + min_new  # rescale to 0 to 1 for color mapping
                    colors = p2[latlon_sorted_indices][corrs_2_to_1]
                    grid_points = get_grid_points(N_SQRT)
                    mapping_of_xyz = o3d.geometry.PointCloud()  
                    mapping_of_xyz.points = o3d.utility.Vector3dVector((xyz + sphere_points)[latlon_sorted_indices][corrs_2_to_1])
                    mapping_of_xyz.colors = o3d.utility.Vector3dVector(colors) 
                    
                    mapping_of_sph = o3d.geometry.PointCloud()
                    mapping_of_sph.points = o3d.utility.Vector3dVector(sphere_points[latlon_sorted_indices][corrs_2_to_1])
                    mapping_of_sph.colors = o3d.utility.Vector3dVector(colors)  
                    
                    mapping_of_grid = o3d.geometry.PointCloud()
                    mapping_of_grid.points = o3d.utility.Vector3dVector(grid_points)
                    mapping_of_grid.colors = o3d.utility.Vector3dVector(colors) 

                    o3d.io.write_point_cloud(os.path.join(save_root, line, 'atlas_pcd_grid_order', f"{data_name}.ply"), mapping_of_xyz)
                    o3d.io.write_point_cloud(os.path.join(save_root, line, 'atlas_pcd_grid_order', f"{data_name}_sphere.ply"), mapping_of_sph)
                    o3d.io.write_point_cloud(os.path.join(save_root, line, 'atlas_pcd_grid_order', f"{data_name}_grid.ply"), mapping_of_grid)
                # arrange the order to grid order
            
            xyz = xyz[latlon_sorted_indices][corrs_2_to_1]
            normal = normal[latlon_sorted_indices][corrs_2_to_1]
            # round grid value from [-0.5, 0.5] to [0, 127]
            # One more round of ordering to ensure a correct mapping to image
            xyz = xyz.reshape(N_SQRT, N_SQRT, 3)
            normal = normal.reshape(N_SQRT, N_SQRT, 3)
            # save to file as image
            os.makedirs(os.path.join(save_root, line, 'atlas'), exist_ok=True)
            # save the raw files
            torch.save({"xyz": xyz, "normal": normal}, os.path.join(save_root, line, 'atlas', f'{data_name}_xyz_normal.pt'))
            # check, rescale and save the xyz and normal images for visualization
            # xyz[:,:,0] = (xyz[:,:,0] - xyz[:,:,0].min()) / (xyz[:,:,0].max() - xyz[:,:,0].min())
            # xyz[:,:,1] = (xyz[:,:,1] - xyz[:,:,1].min()) / (xyz[:,:,1].max() - xyz[:,:,1].min())
            # xyz[:,:,2] = (xyz[:,:,2] - xyz[:,:,2].min()) / (xyz[:,:,2].max() - xyz[:,:,2].min())
            # normal[:,:,0] = (normal[:,:,0] - normal[:,:,0].min()) / (normal[:,:,0].max() - normal[:,:,0].min())
            # normal[:,:,1] = (normal[:,:,1] - normal[:,:,1].min()) / (normal[:,:,1].max() - normal[:,:,1].min())
            # normal[:,:,2] = (normal[:,:,2] - normal[:,:,2].min()) / (normal[:,:,2].max() - normal[:,:,2].min())
            xyz_vis = (xyz + 1) * 0.5
            normal_vis = (normal + 1) * 0.5
            xyz_vis = torch.clamp(xyz_vis, 0, 1)
            normal_vis = torch.clamp(normal_vis, 0, 1)
            xyz_image = to_pil_image(xyz_vis.permute(2, 0, 1))
            normal_image = to_pil_image(normal_vis.permute(2, 0, 1))
            xyz_image.save(os.path.join(save_root, line, 'atlas', f'{data_name}_xyz.png'))
            normal_image.save(os.path.join(save_root, line, 'atlas', f'{data_name}_normal.png'))


def shpere2grid(source_root, save_root, lines, visualize_mapping=False, visualization_order='xyz'):
    sphere_points = init_unit_sphere_grid()
    latlon = equirectangular_projection(sphere_points)
    corrs_2_to_1, corrs_1_to_2, latlon_sorted_indices, reversed_latlon_sorted_indices = ot_map_eq2grid(latlon_points=latlon)
    os.makedirs(os.path.join(save_root), exist_ok=True)
    for line in lines:
        print("Processing file: {}".format(line))
        # load the point cloud
        npz_file_index = line.split('.')[0]
        source_file = os.path.join(source_root, f'{npz_file_index}', 'spherical_ot', f'{npz_file_index}.pt')
        pcd = torch.load(source_file)
        # ordered the same as sphere_points or latlon
        xyz = pcd["xyz"]
        normal = pcd["normal"]
        if visualize_mapping:
            if visualization_order == 'xyz':
                os.makedirs(os.path.join(save_root, os.path.basename(source_file).split('.')[0], 'atlas_pcd'), exist_ok=True)
                min_old, max_old = -1, 1  
                min_new, max_new = 0, 1  
                p2 = sphere_points.cpu().numpy()
                p2 = (p2 - min_old) / (max_old - min_old) * (max_new - min_new) + min_new  # rescale to 0 to 1 for color mapping
                colors = p2
                grid_points = get_grid_points(N_SQRT)
                mapping_of_xyz = o3d.geometry.PointCloud()  
                mapping_of_xyz.points = o3d.utility.Vector3dVector(xyz + sphere_points)
                mapping_of_xyz.colors = o3d.utility.Vector3dVector(colors) 
                
                mapping_of_sph = o3d.geometry.PointCloud()
                mapping_of_sph.points = o3d.utility.Vector3dVector(sphere_points)
                mapping_of_sph.colors = o3d.utility.Vector3dVector(colors)  
                
                mapping_of_grid = o3d.geometry.PointCloud()
                mapping_of_grid.points = o3d.utility.Vector3dVector(grid_points[corrs_1_to_2][reversed_latlon_sorted_indices])
                mapping_of_grid.colors = o3d.utility.Vector3dVector(colors) 

                o3d.io.write_point_cloud(os.path.join(save_root, os.path.basename(source_file).split('.')[0], 'atlas_pcd', line.split('.')[0]+".ply"), mapping_of_xyz)
                o3d.io.write_point_cloud(os.path.join(save_root, os.path.basename(source_file).split('.')[0], 'atlas_pcd', line.split('.')[0]+"_sphere.ply"), mapping_of_sph)
                o3d.io.write_point_cloud(os.path.join(save_root, os.path.basename(source_file).split('.')[0], 'atlas_pcd', line.split('.')[0]+"_grid.ply"), mapping_of_grid)
            elif visualization_order == 'grid':
                os.makedirs(os.path.join(save_root, os.path.basename(source_file).split('.')[0], 'atlas_pcd_grid_order'), exist_ok=True)
                min_old, max_old = -1, 1  
                min_new, max_new = 0, 1  
                p2 = sphere_points.cpu().numpy()
                p2 = (p2 - min_old) / (max_old - min_old) * (max_new - min_new) + min_new  # rescale to 0 to 1 for color mapping
                colors = p2[latlon_sorted_indices][corrs_2_to_1]
                grid_points = get_grid_points(N_SQRT)
                mapping_of_xyz = o3d.geometry.PointCloud()  
                mapping_of_xyz.points = o3d.utility.Vector3dVector((xyz + sphere_points)[latlon_sorted_indices][corrs_2_to_1])
                mapping_of_xyz.colors = o3d.utility.Vector3dVector(colors) 
                
                mapping_of_sph = o3d.geometry.PointCloud()
                mapping_of_sph.points = o3d.utility.Vector3dVector(sphere_points[latlon_sorted_indices][corrs_2_to_1])
                mapping_of_sph.colors = o3d.utility.Vector3dVector(colors)  
                
                mapping_of_grid = o3d.geometry.PointCloud()
                mapping_of_grid.points = o3d.utility.Vector3dVector(grid_points)
                mapping_of_grid.colors = o3d.utility.Vector3dVector(colors) 

                o3d.io.write_point_cloud(os.path.join(save_root, os.path.basename(source_file).split('.')[0], 'atlas_pcd_grid_order', line.split('.')[0]+".ply"), mapping_of_xyz)
                o3d.io.write_point_cloud(os.path.join(save_root, os.path.basename(source_file).split('.')[0], 'atlas_pcd_grid_order', line.split('.')[0]+"_sphere.ply"), mapping_of_sph)
                o3d.io.write_point_cloud(os.path.join(save_root, os.path.basename(source_file).split('.')[0], 'atlas_pcd_grid_order', line.split('.')[0]+"_grid.ply"), mapping_of_grid)
            # arrange the order to grid order
        
        xyz = xyz[latlon_sorted_indices][corrs_2_to_1]
        normal = normal[latlon_sorted_indices][corrs_2_to_1]
        # round grid value from [-0.5, 0.5] to [0, 127]
        # One more round of ordering to ensure a correct mapping to image
        xyz = xyz.reshape(N_SQRT, N_SQRT, 3)
        normal = normal.reshape(N_SQRT, N_SQRT, 3)
        # save to file as image
        os.makedirs(os.path.join(save_root, os.path.basename(source_file).split('.')[0], 'atlas'), exist_ok=True)
        # save the raw files 
        torch.save({"xyz": xyz, "normal": normal}, os.path.join(save_root, os.path.basename(source_file).split('.')[0], 'atlas', 'xyz_normal.pt'))
        # check, rescale and save the xyz and normal images for visualization
        # xyz[:,:,0] = (xyz[:,:,0] - xyz[:,:,0].min()) / (xyz[:,:,0].max() - xyz[:,:,0].min())
        # xyz[:,:,1] = (xyz[:,:,1] - xyz[:,:,1].min()) / (xyz[:,:,1].max() - xyz[:,:,1].min())
        # xyz[:,:,2] = (xyz[:,:,2] - xyz[:,:,2].min()) / (xyz[:,:,2].max() - xyz[:,:,2].min())
        # normal[:,:,0] = (normal[:,:,0] - normal[:,:,0].min()) / (normal[:,:,0].max() - normal[:,:,0].min())
        # normal[:,:,1] = (normal[:,:,1] - normal[:,:,1].min()) / (normal[:,:,1].max() - normal[:,:,1].min())
        # normal[:,:,2] = (normal[:,:,2] - normal[:,:,2].min()) / (normal[:,:,2].max() - normal[:,:,2].min())
        xyz_vis = (xyz + 1) * 0.5
        normal_vis = (normal + 1) * 0.5
        xyz_vis = torch.clamp(xyz_vis, 0, 1)
        normal_vis = torch.clamp(normal_vis, 0, 1)
        xyz_image = to_pil_image(xyz_vis.permute(2, 0, 1))
        normal_image = to_pil_image(normal_vis.permute(2, 0, 1))
        xyz_image.save(os.path.join(save_root, os.path.basename(source_file).split('.')[0], 'atlas', 'xyz.png'))
        normal_image.save(os.path.join(save_root, os.path.basename(source_file).split('.')[0], 'atlas', 'normal.png'))

def grid2pcd(source_folder, mapping_folder, saving_folder, line, reversed_latlon_sorted_indices, grid_corrs_1_to_2):
    # this is in grid order
    id = line.split('.')[0]

    # pcd_file = os.path.join(source_folder, f"{id}", "generated_xyz_normal.pt")
    pcd_file = os.path.join(source_folder, f"{id}", 'atlas', 'xyz_normal.pt')
    pcd = torch.load(pcd_file)
    xyz = pcd["xyz"].reshape(-1, 3)
    normal = pcd["normal"].reshape(-1, 3)

    mapping_file = os.path.join(mapping_folder, f"{id}", "spherical_ot", f"{id}_original_and_mapping.pt")
    corrs_1_to_2 = torch.load(mapping_file)["corrs_1_to_2"]
    # sphere_points order
    xyz = xyz[grid_corrs_1_to_2][reversed_latlon_sorted_indices]
    normal = normal[grid_corrs_1_to_2][reversed_latlon_sorted_indices]
    sphere_points = init_unit_sphere_grid(N)
    xyz = xyz + sphere_points
    xyz = xyz[corrs_1_to_2]
    normal = normal[corrs_1_to_2]

    # to numpy
    xyz = xyz.cpu().numpy()
    normal = normal.cpu().numpy()

    # save to npz file
    os.makedirs(os.path.join(saving_folder, f"{id}"), exist_ok=True)
    pc_normal = np.concatenate((xyz, normal), axis=-1, dtype=np.float16)
    npz_to_save = {}
    npz_to_save["pc_normal"] = pc_normal
    npz_file = os.path.join(saving_folder, f"{id}", "gen_pc_normal.npz")
    np.savez(npz_file, **npz_to_save)

    # save to ply file with o3d
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(xyz)
    pc_o3d.normals = o3d.utility.Vector3dVector(normal)
    o3d.io.write_point_cloud(os.path.join(saving_folder, f"{id}", "gen_pc_normal.ply"), pc_o3d)
    return

def GaussianOT_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=str, default="./output_dc_fitting/")
    parser.add_argument("--save_root", type=str, default="./output_gaussiancube/")
    parser.add_argument("--max_sh_degree", type=int, default=0)
    parser.add_argument("--txt_file", type=str, default="./example_data/shapenet_car.txt")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1)
    parser.add_argument("--bound", type=float, default=0.45)
    parser.add_argument("--visualize_mapping", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    with open(args.txt_file, "r") as f:
        lines = f.read().splitlines()[args.start_idx:args.end_idx]
  
    # Number of worker processes to use  
    num_workers = args.num_workers
  
    # Create a pool of workers and map the processing function to the data  
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:  
        # Submit all the tasks and wait for them to complete  
        futures = [executor.submit(process_single_wrapper, line, args.source_root, args.save_root, args.max_sh_degree, args.bound, args.visualize_mapping) for line in lines]  
        for future in futures:  
            future.result()  # You can add error handling here if needed    

def sphere2grid_main():
    '''load the xyz and normal from the point cloud, and map them to the grid points using the OT mapping'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--txt_file", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=5526)
    parser.add_argument("--visualize_mapping", action="store_true")
    parser.add_argument("--visualization_order", type=str, default='xyz', choices=['xyz', 'grid'], help="Order of visualization: 'xyz' for xyz order, 'grid' for grid order")
    args = parser.parse_args()

    with open(args.txt_file, "r") as f:
        lines = f.read().splitlines()[args.start_idx:args.end_idx]
    
    shpere2grid(source_root=args.source_root, save_root=args.save_root, lines=lines, visualize_mapping=args.visualize_mapping, visualization_order=args.visualization_order)


def pcd_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--txt_file", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=5526)
    parser.add_argument("--visualize_mapping", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()
    
    with open(args.txt_file, "r") as f:
        lines = f.read().splitlines()[args.start_idx:args.end_idx]
  
    # Number of worker processes to use  
    num_workers = args.num_workers
    
    sphere_points = init_unit_sphere_grid(N)
    # Create a pool of workers and map the processing function to the data  
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:  
        # Submit all the tasks and wait for them to complete  
        futures = [executor.submit(process_single_pcd_wrapper, line, args.source_root, args.save_root, sphere_points, args.visualize_mapping) for line in lines]  
        for future in futures:  
            future.result()  # You can add error handling here if needed 

def sphere2grid_main_new():
    '''
    load the xyz and normal from the point cloud, and map them to the grid points using the OT mapping
    Example usage:
    python ot_structuralization.py --visualize_mapping
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=str, default="data/ShapeNet_processed")
    parser.add_argument("--visualize_mapping", action="store_true")
    parser.add_argument("--visualization_order", type=str, default='xyz', choices=['xyz', 'grid'], help="Order of visualization: 'xyz' for xyz order, 'grid' for grid order")
    args = parser.parse_args()

    lines = [d for d in os.listdir(args.source_root) if os.path.isdir(os.path.join(args.source_root, d))]
    lines =  sorted(lines, key=int)
    sphere2grid_new(source_root=args.source_root, save_root=args.source_root, lines=lines, visualize_mapping=args.visualize_mapping, visualization_order=args.visualization_order)

def split_data(source_root, num_splits=5):
    '''
    Split the data into num_splits parts, each part contains 1/num_splits of the data
    '''
    lines = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    lines =  sorted(lines, key=int)
    num_lines = len(lines)
    split_size = num_lines // num_splits

    split_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shape_net_split')
    os.makedirs(split_dir, exist_ok=True)
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_splits - 1 else num_lines
        split_lines = lines[start_idx:end_idx]
        with open(os.path.join(split_dir, f'split_{i}.txt'), 'w') as f:
            for line in split_lines:
                f.write(f"{line}\n")

def pcd_main_new():
    '''
    Example useage:
    python ot_structuralization.py --visualize_mapping --split 1 --num_workers 12
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=str, default="data/ShapeNet_processed")
    parser.add_argument("--visualize_mapping", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--full_parallel", action="store_true", help="If set, will use all available CPU cores")
    parser.add_argument("--split", type=str, default=None, help="If set, will split the data into multiple parts and process each part separately")
    parser.add_argument("--split_base_dir", type=str, default="shapeatlas_utils/shape_net_split", help="Base directory for split files")
    args = parser.parse_args()
    
    if args.full_parallel:
        CPU_COUNT = multiprocessing.cpu_count()
        num_workers = CPU_COUNT // 5 # since each ot use 4 threads and 1 threads to call. 
    # Number of worker processes to use
    else:
        num_workers = args.num_workers
    # list all the files under source_root
    if args.split is None:
        lines = [d for d in os.listdir(args.source_root) if os.path.isdir(os.path.join(args.source_root, d))]
        lines =  sorted(lines, key=int) 
    else:
        split_file = os.path.join(args.split_base_dir, f'split_{args.split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file {split_file} does not exist.")
        with open(split_file, 'r') as f:
            lines = f.read().splitlines()
        lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines
        lines = sorted(lines, key=int)  # Ensure the lines are sorted numerically

    # lines = ['0']
    sphere_points = init_unit_sphere_grid(N)
    # Create a pool of workers and map the processing function to the data  
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:  
        # Submit all the tasks and wait for them to complete  
        futures = [executor.submit(process_pcd_wrapper_new, line, args.source_root, args.source_root, sphere_points, args.visualize_mapping) for line in lines]  
        for future in futures:  
            future.result()  # You can add error handling here if needed 


if __name__ == "__main__":
    # pcd_main_new()
    sphere2grid_main_new()
    # split_data('data/ShapeNet_processed', num_splits=6)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import trimesh
from shapeatlas_utils.utils import mesh_sort
import argparse
import random
from shapeatlas_utils.ot_structuralization import *
from shapeatlas_utils.uneven_ot_structuralization import *
from PIL import Image
import torch
import open3d as o3d
import ipdb

# def info(type, value, tb):
#     ipdb.post_mortem(tb)

# sys.excepthook = info

def test_spherical_points():
    pts = init_unit_sphere_grid()
    # convert to open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # save to file
    o3d.io.write_point_cloud('output/spherical_points.ply', pcd)

def test_padding_and_prune():
    source_pcd_path = '/scr/yaohe/data/ShapeNet/noisy_and_cropped/crop_20_noise_0-01/0.npz'

    source_pcd = load_npz(source_pcd_path)
    print(f"Source point cloud shape: {source_pcd.xyz.shape}")
    pcd_padded_and_pruned = padding_and_prune_input(source_pcd, num_pts=N)
    print(f"Padded and pruned point cloud shape: {pcd_padded_and_pruned.xyz.shape}")
    # convert to open3d point cloud
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(source_pcd.xyz)
    o3d_pcd.normals = o3d.utility.Vector3dVector(source_pcd.normal)
    # save to file
    o3d.io.write_point_cloud('output/source_pcd.ply', o3d_pcd)
    o3d_padded_and_pruned = o3d.geometry.PointCloud()
    o3d_padded_and_pruned.points = o3d.utility.Vector3dVector(pcd_padded_and_pruned.xyz)
    o3d_padded_and_pruned.normals = o3d.utility.Vector3dVector(pcd_padded_and_pruned.normal)
    # save padded and pruned point cloud
    o3d.io.write_point_cloud('output/padded_and_pruned_pcd.ply', o3d_padded_and_pruned)

def test_ot_single():
    source_pcd_path = '/scr/yaohe/data/ShapeNet/MeshAnythingProcessed_pc/train/0.npz'
    save_root = 'output/ot_test'
    spherical_pts = init_unit_sphere_grid()
    process_single_pcd2sphere(source_pcd_path,save_root,sphere_points=spherical_pts,visuzalize_mapping=True)

def test_ot_map2grid(order="latlon"):
    save_root = 'output/ot_test'
    os.makedirs(os.path.join(save_root, "spherical_2_grid"), exist_ok=True)
    sphere_points = init_unit_sphere_grid()
    latlon = equirectangular_projection(sphere_points)
    corrs_2_to_1, corrs_1_to_2, latlon_sorted_indices, reversed_latlon_sorted_indices = ot_map_eq2grid(latlon_points=latlon)
    # rescale latlon to [0, 1] for visualization
    # lon = latlon[:, 0]
    # lat = latlon[:, 1]
    # lon = (lon - lon.min()) / (lon.max() - lon.min()) * 2 - 1# rescale to [-1, 1] for color mapping
    # lat = (lat - lat.min()) / (lat.max() - lat.min()) - 0.5 # rescale to [-0.5, 0.5] for color mapping
    lat = latlon[:, 0] #z
    lon = latlon[:, 1] #xy
    lon = (lon - lon.min()) / (lon.max() - lon.min()) * 2 - 1 # rescale to [-1, 1] for color mapping
    lat = (lat - lat.min()) / (lat.max() - lat.min()) * 2 - 1 # - 0.5 # rescale to [-0.5, 0.5] for color mapping
    # stack lat and lon
    # rescaled_latlon = np.stack([lon, lat], axis=1)
    rescaled_latlon = np.stack([lat, lon], axis=1)
    
    rescaled_latlon = np.column_stack((rescaled_latlon, np.zeros_like(lon)))  # Add dummy zeros for z-coordinate
    
    if order == "latlon":
        min_old, max_old = -1, 1  
        min_new, max_new = 0, 1  
        p1 = sphere_points.cpu().numpy().copy()
        p1 = (p1 - min_old) / (max_old - min_old) * (max_new - min_new) + min_new  # rescale to 0 to 1 for color mapping
        # use one field for color mapping
        # ipdb.set_trace()
        color = np.zeros_like(p1)
        # color = p1.copy()
        # color = np.zeros_like(p1)
        # rescale z-coordinate to [0, 255] for color mapping 
        # color[:, 0] = (p1[:, 0] - p1[:, 0].min()) / (p1[:, 0].max() - p1[:, 0].min())
        # circular color mapping, order the color according to latitude
        color[:, 0] = (rescaled_latlon[:, 0] - rescaled_latlon[:, 0].min()) / (rescaled_latlon[:, 0].max() - rescaled_latlon[:, 0].min())
        # color[:, 1] = (rescaled_latlon[:, 1] - rescaled_latlon[:, 1].min()) / (rescaled_latlon[:, 1].max() - rescaled_latlon[:, 1].min())
        mapping_of_latlon = o3d.geometry.PointCloud()
        # select first half of the points for visualization
        mapping_of_latlon.points = o3d.utility.Vector3dVector(rescaled_latlon[0:int(N/2)])
        mapping_of_latlon.colors = o3d.utility.Vector3dVector(color[0:int(N/2)])  # Assuming 'colors' is a NumPy array with the same length as p1

        mapping_of_p1 = o3d.geometry.PointCloud()
        mapping_of_p1.points = o3d.utility.Vector3dVector(p1[0:int(N/2)])
        mapping_of_p1.colors = o3d.utility.Vector3dVector(color[0:int(N/2)])  # Assuming 'colors' is a NumPy array with the same length as p1

        N_sqrt = 128
        x = np.arange(N_sqrt)
        y = np.arange(N_sqrt)
        col_indices, row_indices = np.meshgrid(x, y)
        # Stack to get (row, col) pairs and reshape to (N, 2)
        grid_points = np.stack((row_indices, col_indices), axis=-1).reshape(-1, 2) # shape: (N_sqrt*N_sqrt, 2)
        grid_sorted_indices = np.lexsort((grid_points[:, 1], grid_points[:, 0]))  # sort by row and column indices
        grid_points = grid_points[grid_sorted_indices]  
        # rescale grid points to [-0.5, 0.5] for color mapping
        grid_x = (grid_points[:, 0] - grid_points[:, 0].min()) / (grid_points[:, 0].max() - grid_points[:, 0].min()) - 0.5
        grid_y = (grid_points[:, 1] - grid_points[:, 1].min()) / (grid_points[:, 1].max() - grid_points[:, 1].min()) - 0.5
        grid_points = np.column_stack((grid_x, grid_y, np.ones_like(grid_x)))  # Add dummy zeros for z-coordinate

        mapping_of_p2 = o3d.geometry.PointCloud()
        mapping_of_p2.points = o3d.utility.Vector3dVector(grid_points[corrs_1_to_2][reversed_latlon_sorted_indices])
        mapping_of_p2.colors = o3d.utility.Vector3dVector(color)  # Assuming 'colors' is a NumPy array with the same length as p1
        os.makedirs(os.path.join(save_root,"spherical_ot_visualization"), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(save_root, "spherical_ot_visualization", "latlon.ply"), mapping_of_latlon)
        o3d.io.write_point_cloud(os.path.join(save_root, "spherical_ot_visualization", "sphere.ply"), mapping_of_p1)
        o3d.io.write_point_cloud(os.path.join(save_root, "spherical_ot_visualization", "grid.ply"), mapping_of_p2)
    elif order == "grid":
        min_old, max_old = -1, 1  
        min_new, max_new = 0, 1  
        p1 = sphere_points.cpu().numpy().copy()
        p1 = (p1 - min_old) / (max_old - min_old) * (max_new - min_new) + min_new  # rescale to 0 to 1 for color mapping

        N_sqrt = 128
        x = np.arange(N_sqrt)
        y = np.arange(N_sqrt)
        col_indices, row_indices = np.meshgrid(x, y)
        # Stack to get (row, col) pairs and reshape to (N, 2)
        grid_points = np.stack((row_indices, col_indices), axis=-1).reshape(-1, 2) # shape: (N_sqrt*N_sqrt, 2)
        grid_sorted_indices = np.lexsort((grid_points[:, 1], grid_points[:, 0]))  # sort by row and column indices
        grid_points = grid_points[grid_sorted_indices]  
        
        # rescale grid points to [-0.5, 0.5] for color mapping
        grid_x = (grid_points[:, 0] - grid_points[:, 0].min()) / (grid_points[:, 0].max() - grid_points[:, 0].min()) - 0.5
        grid_y = (grid_points[:, 1] - grid_points[:, 1].min()) / (grid_points[:, 1].max() - grid_points[:, 1].min()) - 0.5
        grid_points = np.column_stack((grid_x, grid_y, np.ones_like(grid_x)))  # Add dummy zeros for z-coordinate

        # use one field for color mapping
        color = np.zeros_like(p1)
        # color[:, 0] = (grid_points[:, 0] - grid_points[:, 0].min()) / (grid_points[:, 0].max() - grid_points[:, 0].min())
        color[:, 1] = (grid_points[:, 1] - grid_points[:, 1].min()) / (grid_points[:, 1].max() - grid_points[:, 1].min())
        mapping_of_latlon = o3d.geometry.PointCloud()
        mapping_of_latlon.points = o3d.utility.Vector3dVector(rescaled_latlon[latlon_sorted_indices][corrs_2_to_1])
        mapping_of_latlon.colors = o3d.utility.Vector3dVector(color)  
        
        mapping_of_p1 = o3d.geometry.PointCloud()
        mapping_of_p1.points = o3d.utility.Vector3dVector(p1[latlon_sorted_indices][corrs_2_to_1])
        mapping_of_p1.colors = o3d.utility.Vector3dVector(color)  # Assuming 'colors' is a NumPy array with the same length as p1

        mapping_of_p2 = o3d.geometry.PointCloud()
        mapping_of_p2.points = o3d.utility.Vector3dVector(grid_points)
        mapping_of_p2.colors = o3d.utility.Vector3dVector(color)  # Assuming 'colors' is a NumPy array with the same length as p1
        os.makedirs(os.path.join(save_root,"spherical_ot_visualization_grid_order"), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(save_root, "spherical_ot_visualization_grid_order", "latlon.ply"), mapping_of_latlon)
        o3d.io.write_point_cloud(os.path.join(save_root, "spherical_ot_visualization_grid_order", "sphere.ply"), mapping_of_p1)
        o3d.io.write_point_cloud(os.path.join(save_root, "spherical_ot_visualization_grid_order", "grid.ply"), mapping_of_p2)


def generate_data_profile():
    # generate a text file list all the data name
    # data_root = '/scr/yaohe/data/ShapeNet/ShapeNet_all_pcd'\
    random.seed(0)
    np.random.seed(0)

    data_root = '/scr/yaohe/data/ShapeNet/MeshAnythingProcessed_pc/train'
    data_list = os.listdir(data_root)
    data_list = [data for data in data_list if data.endswith('.npz')]
    data_list.sort(key=lambda x: int(x.split('.')[0]))  # Sort by the numeric part of the filename
    # with open(os.path.join(data_root, '../train_data_list.txt'), 'w') as f:
    #     for data in data_list:
    #         f.write(data + '\n')
    test_water_indices = np.random.choice(len(data_list), size=552, replace=False)
    test_data = [data_list[i] for i in test_water_indices]
    train_data = [data_list[i] for i in range(len(data_list)) if i not in test_water_indices]

    print(f"Train water: {len(train_data)}, Test water: {len(test_data)}")
    # save the name to test_split.txt and train_split.txt
    with open(os.path.join(data_root, '../test_split.txt'), 'w') as f:
        for data in test_data:
            f.write(data + '\n')
    with open(os.path.join(data_root, '../train_split.txt'), 'w') as f:
        for data in train_data:
            f.write(data + '\n')


def test_pcd_reconstruction():
    source_folder = '/scr/yaohe/data/ShapeNet/pcd2shpere/train'
    mapping_folder = '/scr/yaohe/data/ShapeNet/pcd2shpere/train'
    saving_folder = 'output/pcd_reconstruction'
    os.makedirs(saving_folder, exist_ok=True)
    line = '0'
    sphere_points = init_unit_sphere_grid()
    latlon = equirectangular_projection(sphere_points)
    # corrs_2_to_1, corrs_1_to_2, latlon_sorted_indices, reversed_latlon_sorted_indices = ot_map_eq2grid(latlon_points=latlon)

    # now load the corrs_2_to_1, corrs_1_to_2, latlon_sorted_indices, reversed_latlon_sorted_indices from the npz file
    ot_mapping_file = os.path.join(saving_folder, 'ot_mapping.npz')
    ot_mapping_data = np.load(ot_mapping_file)
    corrs_2_to_1 = ot_mapping_data['corrs_2_to_1']
    corrs_1_to_2 = ot_mapping_data['corrs_1_to_2']
    latlon_sorted_indices = ot_mapping_data['latlon_sorted_indices']
    reversed_latlon_sorted_indices = ot_mapping_data['reversed_latlon_sorted_indices']
    
    grid2pcd(
        source_folder,
        mapping_folder,
        saving_folder,
        line,
        reversed_latlon_sorted_indices,
        corrs_1_to_2
    )

    # also save corrs_2_to_1, corrs_1_to_2, latlon_sorted_indices, reversed_latlon_sorted_indices to a npz file
    # np.savez(os.path.join(saving_folder, 'ot_mapping.npz'),
    #          corrs_2_to_1=corrs_2_to_1,
    #          corrs_1_to_2=corrs_1_to_2,
    #          latlon_sorted_indices=latlon_sorted_indices,
    #          reversed_latlon_sorted_indices=reversed_latlon_sorted_indices)

def visualize_training_log():
    normal_npy_path = '/scr/yaohe/data/partial_mesh/exp_output/train_unet_2025-06-02_11-05-50/eval/debug_0/15/gen_normal.npy'
    xyz_npy_path = '/scr/yaohe/data/partial_mesh/exp_output/train_unet_2025-06-02_11-05-50/eval/debug_0/15/gen_xyz.npy'
    saving_folder = 'output/pcd_reconstruction'

    normal = np.load(normal_npy_path)
    xyz = np.load(xyz_npy_path)
    ot_mapping_file = os.path.join(saving_folder, 'ot_mapping.npz')
    ot_mapping_data = np.load(ot_mapping_file)
    corrs_2_to_1 = ot_mapping_data['corrs_2_to_1']
    corrs_1_to_2 = ot_mapping_data['corrs_1_to_2']
    latlon_sorted_indices = ot_mapping_data['latlon_sorted_indices']
    reversed_latlon_sorted_indices = ot_mapping_data['reversed_latlon_sorted_indices']

    sphere_points = init_unit_sphere_grid()

    xyz = xyz.reshape(-1, 3)
    normal = normal.reshape(-1, 3)
    xyz = xyz[corrs_1_to_2][reversed_latlon_sorted_indices]
    xyz = xyz + sphere_points.cpu().numpy()
    normal = normal[corrs_1_to_2][reversed_latlon_sorted_indices]
    # convert to open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.normals = o3d.utility.Vector3dVector(normal)
    # save to file
    o3d.io.write_point_cloud(os.path.join(saving_folder, 'reconstructed_pcd.ply'), pcd)


def reconstruct_pcd_from_atlas():
    gen_atlas_folder = '/scr/yaohe/data/partial_mesh/exp_output/train_unet_2025-06-02_11-05-50/eval/debug_0'
    test_split = '/scr/yaohe/data/ShapeNet/MeshAnythingProcessed_pc/test_split.txt'
    missing_split = '/scr/yaohe/data/ShapeNet/evaluation/cropped_20_noise_0-01_npy_inferred_test/missing_files.txt'
    save_root = '/scr/yaohe/data/ShapeNet/evaluation/gen_pcd'

    ot_mapping_file = 'tests/output/pcd_reconstruction/ot_mapping.npz'
    ot_mapping_data = np.load(ot_mapping_file)
    corrs_2_to_1 = ot_mapping_data['corrs_2_to_1']
    corrs_1_to_2 = ot_mapping_data['corrs_1_to_2']
    latlon_sorted_indices = ot_mapping_data['latlon_sorted_indices']
    reversed_latlon_sorted_indices = ot_mapping_data['reversed_latlon_sorted_indices']

    sphere_points = init_unit_sphere_grid()

    os.makedirs(os.path.join(save_root, 'gen_pcd'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'gen_npy'), exist_ok=True)
    with open(test_split, 'r') as f:
        test_data = f.readlines()
    with open(missing_split, 'r') as f:
        missing_data = f.readlines()
    test_data = [data.strip().split('.')[0] for data in test_data]
    missing_data = [data.strip().split('.')[0] for data in missing_data]
    test_data = [data for data in test_data if data not in missing_data]
    print(f"Test data length: {len(test_data)}")
    for data in test_data:
        # load the gen_normal.npy and gen_xyz.npy
        gen_normal_path = os.path.join(gen_atlas_folder, data, 'gen_normal.npy')
        gen_xyz_path = os.path.join(gen_atlas_folder, data, 'gen_xyz.npy')
        assert os.path.exists(gen_normal_path), f"File {gen_normal_path} does not exist."
        assert os.path.exists(gen_xyz_path), f"File {gen_xyz_path} does not exist."

        # hacking, failed
        # input_normal_path = os.path.join(gen_atlas_folder, data, 'normal_input.npy')
        # input_xyz_path = os.path.join(gen_atlas_folder, data, 'xyz_input.npy')
        # input_normal = np.load(input_normal_path)
        # input_xyz = np.load(input_xyz_path)
        # input_normal = input_normal.reshape(-1, 3)
        # input_xyz = input_xyz.reshape(-1, 3)
        # input_xyz = input_xyz[corrs_1_to_2][reversed_latlon_sorted_indices] + sphere_points.cpu().numpy()
        # input_normal = input_normal[corrs_1_to_2][reversed_latlon_sorted_indices]

        # # create a mask indicating the zero points in the input
        # xyz_mask = np.all(input_xyz == 0, axis=1)
        # normal_mask = np.all(input_normal == 0, axis=1)
        # gen_normal = np.load(gen_normal_path)
        # gen_xyz = np.load(gen_xyz_path)
        # # # reconstruct the point cloud
        # gen_normal = gen_normal.reshape(-1, 3)
        # gen_xyz = gen_xyz.reshape(-1, 3)
        
        # gen_xyz = gen_xyz[corrs_1_to_2][reversed_latlon_sorted_indices]
        # gen_xyz = gen_xyz + sphere_points.cpu().numpy()
        # gen_normal = gen_normal[corrs_1_to_2][reversed_latlon_sorted_indices]

        # apply the mask to the generated point cloud, replace the zero points in the input with the generated points
        # input_xyz[xyz_mask] = gen_xyz[xyz_mask]
        # input_normal[normal_mask] = gen_normal[normal_mask]
        # gen_xyz = input_xyz
        # gen_normal = input_normal

        gen_normal = np.load(gen_normal_path)
        gen_xyz = np.load(gen_xyz_path)
        gen_normal = gen_normal.reshape(-1, 3)
        gen_xyz = gen_xyz.reshape(-1, 3)
        gen_xyz = gen_xyz[corrs_1_to_2][reversed_latlon_sorted_indices]
        gen_xyz = gen_xyz + sphere_points.cpu().numpy()
        gen_normal = gen_normal[corrs_1_to_2][reversed_latlon_sorted_indices]
        # normalize the normal vectors
        gen_normal = gen_normal / np.linalg.norm(gen_normal, axis=1, keepdims=True)

        # convert to npy (N, 6), first 3 columns are xyz, last 3 columns are normal
        gen_pcd = np.concatenate([gen_xyz, gen_normal], axis=1)
        # save to npy file
        gen_pcd_path = os.path.join(save_root, 'gen_npy', f"{data}.npy")
        np.save(gen_pcd_path, gen_pcd)
        # also save to pcd for visualization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gen_xyz)
        pcd.normals = o3d.utility.Vector3dVector(gen_normal)
        # save to file
        o3d.io.write_point_cloud(os.path.join(save_root, 'gen_pcd', f"{data}.ply"), pcd)

def split_folder():
    data_root = '/scr/yaohe/data/ShapeNet/evaluation/gen_pcd/gen_npy'
    split_num = 4
    data_list = os.listdir(data_root)
    data_list = [data for data in data_list if data.endswith('.npy')]
    data_list.sort(key=lambda x: int(x.split('.')[0]))  # Sort by the numeric part of the filename
    # 113 1001 1013 1022 1034 1050 1051 1098 1114 1116 1121 1139 1145 1161 1170 1189 1197 1203
    done_files = ['113.npy', '1001.npy', '1013.npy', '1022.npy', '1034.npy', '1050.npy', '1051.npy', '1098.npy',
                  '1114.npy', '1116.npy', '1121.npy', '1139.npy', '1145.npy', '1161.npy', '1170.npy',
                  '1189.npy', '1197.npy', '1203.npy']
    data_list = [data for data in data_list if data not in done_files]
    print(f"Total data: {len(data_list)}")
    # split the data_list into split_num parts
    split_size = len(data_list) // split_num
    for i in range(split_num):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < split_num - 1 else len(data_list)
        split_data = data_list[start_idx:end_idx]
        sub_split_folder = os.path.join(f'{data_root}_split_{i}')
        os.makedirs(sub_split_folder, exist_ok=True)
        for data in split_data:
            # copy the file to the sub_split_folder
            src_path = os.path.join(data_root, data)
            dst_path = os.path.join(sub_split_folder, data)
            os.system(f'cp {src_path} {dst_path}')

def merge_split():
    data_root = '/scr/yaohe/data/ShapeNet/evaluation/gen_pcd/'
    splits = ['gen_npy_inferred/03_23-17-43',
              'gen_npy_inferred_split_0/04_00-09-25',
              'gen_npy_inferred_split_1/04_00-09-25',
              'gen_npy_inferred_split_2/04_00-09-25',
              'gen_npy_inferred_split_3/04_02-06-56']
    print(f"Total splits: {len(splits)}")
    save_folder = '/scr/yaohe/data/ShapeNet/evaluation/gen_obj'
    os.makedirs(save_folder, exist_ok=True)
    for split in splits:
        split_folder = os.path.join(data_root, split)
        data_list = os.listdir(split_folder)
        data_list = [data for data in data_list if data.endswith('.obj')]
        for data in data_list:
            id = data.split('.')[0]
            os.makedirs(os.path.join(save_folder, id), exist_ok=True)
            src_path = os.path.join(split_folder, data)
            dst_path = os.path.join(save_folder, id, data)
            os.system(f'cp {src_path} {dst_path}')


def uneven_lag_matching_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=str, default="data/ShapeNet_processed_tmp")
    parser.add_argument("--visualize_mapping", action="store_true")
    parser.add_argument("--save_root", type=str, default="tests/output/uneven_ot_test")
    parser.add_argument("--do_visibility", action="store_true", help="If set, will also do atlas for visible pt")
    parser.add_argument("--num_view", type=int, default=16, help="Number of views to generate for visibility")
    parser.add_argument("--views_per_batch", type=int, default=2, help="Number of views to generate per batch")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--full_parallel", action="store_true", help="If set, will use all available CPU cores")
    parser.add_argument("--split", type=str, default=None, help="If set, will split the data into multiple parts and process each part separately")
    parser.add_argument("--split_base_dir", type=str, default="shapeatlas_utils/shape_net_split", help="Base directory for split files")
    args = parser.parse_args()
    lines = ['1']
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)
    source_root = os.path.join(args.source_root, lines[0])
    data_index = os.path.basename(source_root).split('.')[0]
    sphere_points = init_unit_sphere_grid(N)
    # sphere_points = sphere_points.cpu().numpy()
    os.makedirs(os.path.join(save_root, data_index, "spherical_ot"), exist_ok=True)
    if args.visualize_mapping:
        os.makedirs(os.path.join(save_root, data_index, "spherical_ot_visualization"), exist_ok=True)
    # view_id = 0
    # process_pcd_uneven_wrapper(lines[0], args.source_root, args.save_root, sphere_points, args.visualize_mapping)
    # process_single_view_uneven(view_id, source_root, save_root, data_index, sphere_points, use_multiprocessing=True)
    batch = [4, 5, 6, 7]
    process_view_batch_wrapper(batch, source_root, save_root, lines[0], sphere_points, True, True)
    
def uneven_single_sphere2grid_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=str, default="tests/output/uneven_ot_test")
    parser.add_argument("--visualize_mapping", action="store_true")
    parser.add_argument("--save_root", type=str, default="tests/output/uneven_ot_test")
    parser.add_argument("--do_visibility", action="store_true", help="If set, will also do atlas for visible pt")
    parser.add_argument("--num_view", type=int, default=16, help="Number of views to generate for visibility")
    parser.add_argument("--views_per_batch", type=int, default=4, help="Number of views to generate per batch")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--full_parallel", action="store_true", help="If set, will use all available CPU cores")
    parser.add_argument("--split", type=str, default=None, help="If set, will split the data into multiple parts and process each part separately")
    parser.add_argument("--split_base_dir", type=str, default="shapeatlas_utils/shape_net_split", help="Base directory for split files")
    args = parser.parse_args()
    
    sort = True
    sphere_points = init_unit_sphere_grid(N)
    latlon = equirectangular_projection(sphere_points)
    if os.path.exists(os.path.join(args.save_root, "spherical_ot", "correspondences_sort.npz")) and sort:
        # load
        print('found correspondences, load it ... ')
        correspondences = np.load(os.path.join(args.save_root, "spherical_ot", "correspondences_sort.npz"))
        corrs_2_to_1 = correspondences["corrs_2_to_1"]
        corrs_1_to_2 = correspondences["corrs_1_to_2"]
        latlon_sorted_indices = correspondences["latlon_sorted_indices"]
        reversed_latlon_sorted_indices = correspondences["reversed_latlon_sorted_indices"]
    elif os.path.exists(os.path.join(args.save_root, "spherical_ot", "correspondences_no_sort.npz")) and not sort:
        correspondences = np.load(os.path.join(args.save_root, "spherical_ot", "correspondences_no_sort.npz"))
        corrs_2_to_1 = correspondences["corrs_2_to_1"]
        corrs_1_to_2 = correspondences["corrs_1_to_2"]
    else:
        if sort:
            os.makedirs(os.path.join(args.save_root, "spherical_ot"), exist_ok=True)
            corrs_2_to_1, corrs_1_to_2, latlon_sorted_indices, reversed_latlon_sorted_indices = simple_ot_map_eq2grid(latlon, sort=sort)
            # save the correspondences
            np.savez_compressed(os.path.join(args.save_root, "spherical_ot", "correspondences_sort.npz"),
                                corrs_2_to_1=corrs_2_to_1,
                                corrs_1_to_2=corrs_1_to_2,
                                latlon_sorted_indices=latlon_sorted_indices,
                                reversed_latlon_sorted_indices=reversed_latlon_sorted_indices)
        else:
            os.makedirs(os.path.join(args.save_root, "spherical_ot"), exist_ok=True)
            corrs_2_to_1, corrs_1_to_2, latlon_sorted_indices, reversed_latlon_sorted_indices = simple_ot_map_eq2grid(latlon, sort=sort)
            # save the correspondences
            np.savez_compressed(os.path.join(args.save_root, "spherical_ot", "correspondences_no_sort.npz"),
                                corrs_2_to_1=corrs_2_to_1,
                                corrs_1_to_2=corrs_1_to_2)

    # load spherical
    id = '0'
    view = '0'
    root_data_folder = os.path.join(args.source_root, id, "spherical_ot")
    full_pt_file = os.path.join(root_data_folder, f"view{view}_full.pt")
    view_visible_pt_file = os.path.join(root_data_folder, f"view{view}_visible.pt")
    view_invisible_pt_file = os.path.join(root_data_folder, f"view{view}_invisible.pt")

    full_pcd = torch.load(full_pt_file)
    view_visible_pcd = torch.load(view_visible_pt_file)
    view_invisible_pcd = torch.load(view_invisible_pt_file)

    full_xyz = full_pcd['xyz']
    full_normal = full_pcd['normal']
    view_visible_xyz = view_visible_pcd['xyz']
    view_visible_normal = view_visible_pcd['normal']
    view_visible_sample_indices = view_visible_pcd['sample_indices']
    view_invisible_xyz = view_invisible_pcd['xyz']
    view_invisible_normal = view_invisible_pcd['normal']
    view_invisible_sample_indices = view_invisible_pcd['sample_indices']
    
    if sort:
        full_xyz = full_xyz[latlon_sorted_indices][corrs_2_to_1]
        full_normal = full_normal[latlon_sorted_indices][corrs_2_to_1]

        view_visible_xyz = view_visible_xyz[latlon_sorted_indices][corrs_2_to_1]
        view_visible_normal = view_visible_normal[latlon_sorted_indices][corrs_2_to_1]
        view_visible_sample_mask = torch.zeros(sphere_points.shape[0], dtype=torch.bool)
        view_visible_sample_mask[view_visible_sample_indices] = True
        view_visible_sample_mask = view_visible_sample_mask[latlon_sorted_indices][corrs_2_to_1]

        view_invisible_xyz = view_invisible_xyz[latlon_sorted_indices][corrs_2_to_1]
        view_invisible_normal = view_invisible_normal[latlon_sorted_indices][corrs_2_to_1]
        view_invisible_sample_mask = torch.zeros(sphere_points.shape[0], dtype=torch.bool)
        view_invisible_sample_mask[view_invisible_sample_indices] = True
        view_invisible_sample_mask = view_invisible_sample_mask[latlon_sorted_indices][corrs_2_to_1]
    else:
        full_xyz = full_xyz[corrs_2_to_1]
        full_normal = full_normal[corrs_2_to_1]

        view_visible_xyz = view_visible_xyz[corrs_2_to_1]
        view_visible_normal = view_visible_normal[corrs_2_to_1]
        view_visible_sample_mask = torch.zeros(sphere_points.shape[0], dtype=torch.bool)
        view_visible_sample_mask[view_visible_sample_indices] = True
        view_visible_sample_mask = view_visible_sample_mask[corrs_2_to_1]

        view_invisible_xyz = view_invisible_xyz[corrs_2_to_1]
        view_invisible_normal = view_invisible_normal[corrs_2_to_1]
        view_invisible_sample_mask = torch.zeros(sphere_points.shape[0], dtype=torch.bool)
        view_invisible_sample_mask[view_invisible_sample_indices] = True
        view_invisible_sample_mask = view_invisible_sample_mask[corrs_2_to_1]

    full_xyz = full_xyz.reshape(N_SQRT, N_SQRT, 3)
    full_normal = full_normal.reshape(N_SQRT, N_SQRT, 3)
    view_visible_xyz = view_visible_xyz.reshape(N_SQRT, N_SQRT, 3)
    view_visible_normal = view_visible_normal.reshape(N_SQRT, N_SQRT, 3)
    view_visible_sample_mask = view_visible_sample_mask.reshape(N_SQRT, N_SQRT)

    view_invisible_xyz = view_invisible_xyz.reshape(N_SQRT, N_SQRT, 3)
    view_invisible_normal = view_invisible_normal.reshape(N_SQRT, N_SQRT, 3)
    view_invisible_sample_mask = view_invisible_sample_mask.reshape(N_SQRT, N_SQRT)

    if sort:
        save_path = os.path.join(args.save_root, id, "atlas", "sort")
    else:
        save_path = os.path.join(args.save_root, id, "atlas", "unsort")
    os.makedirs(save_path, exist_ok=True)
    torch.save({
        'xyz': full_xyz,
        'normal': full_normal
    }, os.path.join(save_path, f"view{view}_full.pt"))
    torch.save({
        'xyz': view_visible_xyz,
        'normal': view_visible_normal,
        'mask': view_visible_sample_mask
    }, os.path.join(save_path, f"view{view}_visible.pt"))
    torch.save({
        'xyz': view_invisible_xyz,
        'normal': view_invisible_normal,
        'mask': view_invisible_sample_mask
    }, os.path.join(save_path, f"view{view}_invisible.pt"))


    full_xyz_vis = (full_xyz + 1) * 0.5
    full_normal_vis = (full_normal + 1) * 0.5
    full_xyz_vis = torch.clamp(full_xyz_vis, 0, 1)
    full_normal_vis = torch.clamp(full_normal_vis, 0, 1)

    view_visible_xyz_vis = (view_visible_xyz + 1) * 0.5
    view_visible_normal_vis = (view_visible_normal + 1) * 0.5
    view_visible_xyz_vis = torch.clamp(view_visible_xyz_vis, 0, 1)
    view_visible_normal_vis = torch.clamp(view_visible_normal_vis, 0, 1)
    view_visible_mask = view_visible_sample_mask.to(torch.uint8) * 255
    
    view_invisible_xyz_vis = (view_invisible_xyz + 1) * 0.5
    view_invisible_normal_vis = (view_invisible_normal + 1) * 0.5
    view_invisible_xyz_vis = torch.clamp(view_invisible_xyz_vis, 0, 1)
    view_invisible_normal_vis = torch.clamp(view_invisible_normal_vis, 0, 1)
    view_invisible_mask = view_invisible_sample_mask.to(torch.uint8) * 255

    full_xyz_image = to_pil_image(full_xyz_vis.permute(2,0,1))
    full_normal_image = to_pil_image(full_normal_vis.permute(2,0,1))

    view_visible_xyz_image = to_pil_image(view_visible_xyz_vis.permute(2,0,1))
    view_visible_normal_image = to_pil_image(view_visible_normal_vis.permute(2,0,1))
    view_visible_mask = to_pil_image(view_visible_mask)

    view_invisible_xyz_image = to_pil_image(view_invisible_xyz_vis.permute(2,0,1))
    view_invisible_normal_image = to_pil_image(view_invisible_normal_vis.permute(2,0,1))
    view_invisible_mask = to_pil_image(view_invisible_mask)

    # save
    full_xyz_image.save(os.path.join(save_path, f"view{view}_full.png"))
    full_normal_image.save(os.path.join(save_path, f"view{view}_full_normal.png"))

    view_visible_xyz_image.save(os.path.join(save_path, f"view{view}_visible.png"))
    view_visible_normal_image.save(os.path.join(save_path, f"view{view}_visible_normal.png"))
    view_visible_mask.save(os.path.join(save_path, f"view{view}_visible_mask.png"))

    view_invisible_xyz_image.save(os.path.join(save_path, f"view{view}_invisible.png"))
    view_invisible_normal_image.save(os.path.join(save_path, f"view{view}_invisible_normal.png"))
    view_invisible_mask.save(os.path.join(save_path, f"view{view}_invisible_mask.png"))


def uneven_grid2pcd_test():
    # this is in grid order
    parser = argparse.ArgumentParser()
    # parser.add_argument("--source_root", type=str, default="tests/output/uneven_ot_test")
    parser.add_argument("--source_root", type=str, default="data/ShapeNet_processed_mini_test")
    parser.add_argument("--visualize_mapping", action="store_true")
    parser.add_argument("--save_root", type=str, default="tests/output/uneven_ot_test")
    parser.add_argument("--do_visibility", action="store_true", help="If set, will also do atlas for visible pt")
    parser.add_argument("--num_view", type=int, default=16, help="Number of views to generate for visibility")
    parser.add_argument("--views_per_batch", type=int, default=4, help="Number of views to generate per batch")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--full_parallel", action="store_true", help="If set, will use all available CPU cores")
    parser.add_argument("--split", type=str, default=None, help="If set, will split the data into multiple parts and process each part separately")
    parser.add_argument("--split_base_dir", type=str, default="shapeatlas_utils/shape_net_split", help="Base directory for split files")
    args = parser.parse_args()

    sphere_points = init_unit_sphere_grid(N)
    latlon = equirectangular_projection(sphere_points)
    sort = False
    if os.path.exists(os.path.join(args.save_root, "spherical_ot", "correspondences_no_sort.npz")):
        # load
        print('found correspondences, load it ... ')
        correspondences = np.load(os.path.join(args.save_root, "spherical_ot", "correspondences_no_sort.npz"))
        corrs_2_to_1 = correspondences["corrs_2_to_1"]
        corrs_1_to_2 = correspondences["corrs_1_to_2"]
    else:
        os.makedirs(os.path.join(args.save_root, "spherical_ot"), exist_ok=True)
        corrs_2_to_1, corrs_1_to_2, latlon_sorted_indices, reversed_latlon_sorted_indices = simple_ot_map_eq2grid(latlon)
        # save the correspondences
        np.savez_compressed(os.path.join(args.save_root, "spherical_ot", "correspondences_no_sort.npz"),
                            corrs_2_to_1=corrs_2_to_1,
                            corrs_1_to_2=corrs_1_to_2,)
    id = '0'
    view = '10'
    source_folder = os.path.join(args.source_root, f'{id}')
    saving_folder = os.path.join(args.save_root, f'{id}', 'reconstructed')
    # pcd_file = os.path.join(source_folder, f"{id}", "generated_xyz_normal.pt")
    prefix = f'view{view}'

    visible_pt = os.path.join(source_folder, 'atlas', f"{prefix}_visible.pt")
    invisible_pt = os.path.join(source_folder, 'atlas', f"{prefix}_invisible.pt")
    full_pt = os.path.join(source_folder, 'atlas', f"{prefix}_full.pt")

    visible_pcd = torch.load(visible_pt)

    visible_xyz = visible_pcd["xyz"].reshape(-1, 3)
    visible_normal = visible_pcd["normal"].reshape(-1, 3)
    
    # visible_xyz = visible_xyz[corrs_1_to_2][reversed_latlon_sorted_indices] + sphere_points
    # visible_normal = visible_normal[corrs_1_to_2][reversed_latlon_sorted_indices]
    visible_xyz = visible_xyz[corrs_1_to_2] + sphere_points
    visible_normal = visible_normal[corrs_1_to_2]

    # to numpy
    visible_xyz = visible_xyz.cpu().numpy()
    visible_normal = visible_normal.cpu().numpy()

    # save to npz file
    os.makedirs(saving_folder, exist_ok=True)

    # save to ply file with o3d
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(visible_xyz)
    pc_o3d.normals = o3d.utility.Vector3dVector(visible_normal)
    o3d.io.write_point_cloud(os.path.join(saving_folder, "gen_pc_normal_visible.ply"), pc_o3d)

    invisible_pcd = torch.load(invisible_pt)

    invisible_xyz = invisible_pcd["xyz"].reshape(-1, 3)
    invisible_normal = invisible_pcd["normal"].reshape(-1, 3)

    # invisible_xyz = invisible_xyz[corrs_1_to_2][reversed_latlon_sorted_indices] + sphere_points
    # invisible_normal = invisible_normal[corrs_1_to_2][reversed_latlon_sorted_indices]
    invisible_xyz = invisible_xyz[corrs_1_to_2] + sphere_points
    invisible_normal = invisible_normal[corrs_1_to_2]

    # to numpy
    invisible_xyz = invisible_xyz.cpu().numpy()
    invisible_normal = invisible_normal.cpu().numpy()

    # save to npz file
    os.makedirs(saving_folder, exist_ok=True)

    # save to ply file with o3d
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(invisible_xyz)
    pc_o3d.normals = o3d.utility.Vector3dVector(invisible_normal)
    o3d.io.write_point_cloud(os.path.join(saving_folder, "gen_pc_normal_invisible.ply"), pc_o3d)


    full_pcd = torch.load(full_pt)

    full_xyz = full_pcd["xyz"].reshape(-1, 3)
    full_normal = full_pcd["normal"].reshape(-1, 3)

    # full_xyz = full_xyz[corrs_1_to_2][reversed_latlon_sorted_indices] + sphere_points
    # full_normal = full_normal[corrs_1_to_2][reversed_latlon_sorted_indices]
    full_xyz = full_xyz[corrs_1_to_2] + sphere_points
    full_normal = full_normal[corrs_1_to_2]
    # to numpy
    full_xyz = full_xyz.cpu().numpy()
    full_normal = full_normal.cpu().numpy()

    # save to npz file
    os.makedirs(saving_folder, exist_ok=True)

    # save to ply file with o3d
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(full_xyz)
    pc_o3d.normals = o3d.utility.Vector3dVector(full_normal)
    o3d.io.write_point_cloud(os.path.join(saving_folder, "gen_pc_normal_full.ply"), pc_o3d)

    return




def check_gen_pcd():
    # this is in grid order
    parser = argparse.ArgumentParser()
    # parser.add_argument("--source_root", type=str, default="tests/output/uneven_ot_test")
    parser.add_argument("--source_root", type=str, default="output/exp_output/train_unet_2025-08-16_22-16-54/eval/debug_0")
    parser.add_argument("--visualize_mapping", action="store_true")
    parser.add_argument("--save_root", type=str, default="output/exp_output/train_unet_2025-08-16_22-16-54/eval/debug_0")
    parser.add_argument("--do_visibility", action="store_true", help="If set, will also do atlas for visible pt")
    parser.add_argument("--num_view", type=int, default=16, help="Number of views to generate for visibility")
    parser.add_argument("--views_per_batch", type=int, default=4, help="Number of views to generate per batch")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--full_parallel", action="store_true", help="If set, will use all available CPU cores")
    parser.add_argument("--split", type=str, default=None, help="If set, will split the data into multiple parts and process each part separately")
    parser.add_argument("--split_base_dir", type=str, default="shapeatlas_utils/shape_net_split", help="Base directory for split files")
    args = parser.parse_args()

    sphere_points = init_unit_sphere_grid(N)
    latlon = equirectangular_projection(sphere_points)
    sort = False
    if os.path.exists(os.path.join('shapeatlas_utils/sph2grid_correspondences/correspondences_no_sort.npz')):
        # load
        print('found correspondences, load it ... ')
        correspondences = np.load(os.path.join('shapeatlas_utils/sph2grid_correspondences/correspondences_no_sort.npz'))
        corrs_2_to_1 = correspondences["corrs_2_to_1"]
        corrs_1_to_2 = correspondences["corrs_1_to_2"]
    else:
        os.makedirs(os.path.join(args.save_root, "spherical_ot"), exist_ok=True)
        print("Not found, estimate it")
        corrs_2_to_1, corrs_1_to_2, latlon_sorted_indices, reversed_latlon_sorted_indices = simple_ot_map_eq2grid(latlon)
        # save the correspondences
        np.savez_compressed(os.path.join('shapeatlas_utils/sph2grid_correspondences/correspondences_no_sort.npz'),
                            corrs_2_to_1=corrs_2_to_1,
                            corrs_1_to_2=corrs_1_to_2,)

    test_split = os.path.join('shapeatlas_utils/shapenet_splits/train_test_split/test_split.txt')

    with open(test_split, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line]  # remove empty lines
    sphere_points = sphere_points.cpu().numpy()
    for id in lines:
        print(f"processing {id}")
        views = range(0, 16)
        
        # to string
        views = [f'{view}' for view in views]
        for view in views:
            print(f'processing view {view}')
            if not os.path.exists(os.path.join(args.save_root, f'{id}', f'{view}')):  # Check if the view folder exists
                print(f"View folder for {view} does not exist, skipping...")
                continue
            source_folder = os.path.join(args.source_root, f'{id}', f'{view}')
            saving_folder = os.path.join(args.save_root, f'{id}', f'{view}', 'reconstructed')
            # save to npz file
            os.makedirs(saving_folder, exist_ok=True)
            # pcd_file = os.path.join(source_folder, f"{id}", "generated_xyz_normal.pt")
            prefix = f'view{view}'

            gen_xyz = os.path.join(source_folder, f"gen_xyz.npy")
            gen_normal = os.path.join(source_folder, f"gen_normal.npy")
            
            gt_xyz = os.path.join(source_folder, f"xyz_gt.npy")
            gt_normal = os.path.join(source_folder, f"normal_gt.npy")
            
            xyz_input = os.path.join(source_folder, f"xyz_input.npy")
            normal_input = os.path.join(source_folder, f"normal_input.npy")

            
            gen_xyz = np.load(gen_xyz)[0].reshape(-1, 3)
            gen_normal = np.load(gen_normal)[0].reshape(-1, 3)
            gt_xyz = np.load(gt_xyz).reshape(-1, 3)
            gt_normal = np.load(gt_normal).reshape(-1, 3)
            xyz_input = np.load(xyz_input).reshape(-1, 3)
            normal_input = np.load(normal_input).reshape(-1, 3)

            gen_xyz = gen_xyz[corrs_1_to_2] + sphere_points
            gen_normal = gen_normal[corrs_1_to_2]

            gt_xyz = gt_xyz[corrs_1_to_2] + sphere_points
            gt_normal = gt_normal[corrs_1_to_2]

            input_xyz = xyz_input[corrs_1_to_2] + sphere_points
            input_normal = normal_input[corrs_1_to_2]

            # save to ply file with o3d
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(gen_xyz)
            pc_o3d.normals = o3d.utility.Vector3dVector(gen_normal)
            o3d.io.write_point_cloud(os.path.join(saving_folder, "gen_pc_normal_visible.ply"), pc_o3d)

            # save ground truth point cloud
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(gt_xyz)
            pc_o3d.normals = o3d.utility.Vector3dVector(gt_normal)
            o3d.io.write_point_cloud(os.path.join(saving_folder, "gt_pc_normal_visible.ply"), pc_o3d)

            # save input point cloud
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(input_xyz)
            pc_o3d.normals = o3d.utility.Vector3dVector(input_normal)
            o3d.io.write_point_cloud(os.path.join(saving_folder, "input_pc_normal_visible.ply"), pc_o3d)

    return

if __name__ == "__main__":
    # test_spherical_points()
    # test_padding_and_prune()
    # test_ot_map2grid(order="grid")
    # test_ot_map2grid(order="latlon")
    # generate_data_profile()
    # pcd_main()
    # sphere2grid_main()
    # test_pcd_reconstruction()
    # visualize_training_log()
    # reconstruct_pcd_from_atlas()
    # split_folder()
    # merge_split()
    # uneven_lag_matching_test()
    # uneven_single_sphere2grid_test()
    # uneven_grid2pcd_test()
    check_gen_pcd()


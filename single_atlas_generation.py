from tracemalloc import start

from flask import json
from shapeatlas_utils.uneven_ot_structuralization import uneven_lag_segment_matching
from shapeatlas_utils.ot_structuralization import *
import torch
import os
import numpy as np
import time 
import json
import shutil

def spherical_ot(pcd, sph, save_root, name):
    start = time.time()
    visible_xyz = np.asarray(pcd.points)
    visible_normals = np.asarray(pcd.normals)
    visible_sorted_indices = np.lexsort(
        (-visible_xyz[:, 0], -visible_xyz[:, 1], -visible_xyz[:, 2])
    )
    sph = sph.cpu().numpy()
    visible_xyz = visible_xyz[visible_sorted_indices]
    visible_normals = visible_normals[visible_sorted_indices]
    (visible_corrs_2_to_1,
        visible_corrs_1_to_2,
        visible_sample_indices,
        visible_corrs_2_to_1_full) = uneven_lag_segment_matching(visible_xyz, sph)
    end = time.time()
    sph_ot_duration = end - start

    xyz_offset_visible = torch.from_numpy(
        visible_xyz[visible_corrs_2_to_1_full] - sph
    )
    normal_visible = torch.from_numpy(
        visible_normals[visible_corrs_2_to_1_full]
    )
    new_pcd_visible = PointCloud(
        xyz_offset_visible,
        normal_visible
    )
    
    new_pcd_visible_dict = {
        "xyz": new_pcd_visible.xyz,
        "normal": new_pcd_visible.normal,
        "sample_indices": visible_sample_indices,
    }

    os.makedirs(os.path.join(save_root, f'{name}', "spherical_ot"), exist_ok=True)
    torch.save(
        new_pcd_visible_dict,
        os.path.join(
            save_root, f'{name}', "spherical_ot", f"{name}.pt"
        ),
    )
    return sph_ot_duration

def plain_ot(sph, corrs_path, visible_pt_file, save_root_folder, name):
    
    latlon = equirectangular_projection(sph)
    if os.path.exists(corrs_path):
        # load
        print("found correspondences, load it ... ")
        correspondences = np.load(corrs_path)
        corrs_2_to_1 = correspondences["corrs_2_to_1"]
        corrs_1_to_2 = correspondences["corrs_1_to_2"]
    else:
        os.makedirs(os.path.join(save_root_folder, "correspondences"), exist_ok=True)
        (
            corrs_2_to_1,
            corrs_1_to_2,
            latlon_sorted_indices,
            reversed_latlon_sorted_indices,
        ) = simple_ot_map_eq2grid(latlon)
        # save the correspondences
        np.savez_compressed(
            os.path.join(save_root_folder, "correspondences", "correspondences_no_sort.npz"),
            corrs_2_to_1=corrs_2_to_1,
            corrs_1_to_2=corrs_1_to_2,
        )
    
    view_visible_pcd = torch.load(visible_pt_file)
    view_visible_xyz = view_visible_pcd["xyz"]
    view_visible_normal = view_visible_pcd["normal"]
    view_visible_sample_indices = view_visible_pcd["sample_indices"]

    start = time.time()
    view_visible_xyz = view_visible_xyz[corrs_2_to_1]
    view_visible_normal = view_visible_normal[corrs_2_to_1]
    
    view_visible_sample_mask = torch.zeros(
        sph.shape[0], dtype=torch.bool
    )
    view_visible_sample_mask[view_visible_sample_indices] = True
    view_visible_sample_mask = view_visible_sample_mask[corrs_2_to_1]

    view_visible_xyz = view_visible_xyz.reshape(N_SQRT, N_SQRT, 3)
    view_visible_normal = view_visible_normal.reshape(N_SQRT, N_SQRT, 3)
    view_visible_sample_mask = view_visible_sample_mask.reshape(N_SQRT, N_SQRT)
    end = time.time()
    plain_ot_duration = end - start
    os.makedirs(os.path.join(save_root_folder, f'{name}', 'plain_ot'), exist_ok=True)
    # import pdb; pdb.set_trace()
    torch.save(
        {
            "xyz": view_visible_xyz,
            "normal": view_visible_normal,
            "mask": view_visible_sample_mask,
        },
        os.path.join(save_root_folder, f'{name}', 'plain_ot', f"{name}.pt"),
    )
    
    view_visible_xyz_vis = (view_visible_xyz + 1) * 0.5
    view_visible_normal_vis = (view_visible_normal + 1) * 0.5
    view_visible_xyz_vis = torch.clamp(view_visible_xyz_vis, 0, 1)
    view_visible_normal_vis = torch.clamp(view_visible_normal_vis, 0, 1)
    view_visible_mask = view_visible_sample_mask.to(torch.uint8) * 255

    view_visible_xyz_image = to_pil_image(view_visible_xyz_vis.permute(2, 0, 1))
    view_visible_normal_image = to_pil_image(
        view_visible_normal_vis.permute(2, 0, 1)
    )
    view_visible_mask = to_pil_image(view_visible_mask)

    view_visible_xyz_image.save(
        os.path.join(save_root_folder, f'{name}', 'plain_ot', f"{name}_xyz.png")
    )
    view_visible_normal_image.save(
        os.path.join(save_root_folder, f'{name}', 'plain_ot', f"{name}_normal.png")
    )
    view_visible_mask.save(
        os.path.join(save_root_folder, f'{name}', 'plain_ot', f"{name}_mask.png")
    )
    return plain_ot_duration


def stress_test():
    sphere_points = init_unit_sphere_grid(N)
    corrs_path = "shapeatlas_utils/sph2grid_correspondences/correspondences_no_sort.npz"

    save_root = 'output/ours/stress_test_full_atlas/'

    source_root = 'output/ours/stress_test_input_full'

    sph_ot_time = {}
    plain_ot_time = {}

    object_list = os.listdir(source_root)
    object_list = [f for f in object_list if f.endswith('.ply') and 'gt' not in f]
    save_path = os.path.join(save_root)
    os.makedirs(save_path, exist_ok=True)
    for obj_name in object_list:
        obj_path = os.path.join(source_root, obj_name)
        pcd = o3d.io.read_point_cloud(obj_path)
        name = obj_name.replace('.ply', '')
        duration = spherical_ot(pcd, sphere_points, save_path, name)
        sph_ot_time[name] = (len(pcd.points),duration)
        pt_path = os.path.join(save_path, f'{name}', 'spherical_ot', f'{name}.pt')
        plain_ot_duration = plain_ot(sphere_points, corrs_path, pt_path, save_path, name)
        plain_ot_time[name] = (len(pcd.points), plain_ot_duration)

        print("Spherical OT time:", sph_ot_time)
        print("Plain OT time:", plain_ot_time)

        # save the timing information
        with open(os.path.join(save_path, 'spherical_ot_time.json'), 'w') as f:
            json.dump(sph_ot_time, f)
        with open(os.path.join(save_path, 'plain_ot_time.json'), 'w') as f:
            json.dump(plain_ot_time, f)

if __name__ == "__main__":
    # compc()
    stress_test()
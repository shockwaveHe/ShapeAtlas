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
from plyfile import PlyData
from scipy.optimize import linear_sum_assignment

# from pytorch3d.ops import sample_farthest_points
from scipy.spatial import cKDTree
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image

from shapeatlas_utils.ot_structuralization import *

# structure
# p1[corrs_2_to_1]: map to sphere_points order, corrs_2_to_1 size < N
# sphere_points[sampled_indices][x]: map to p1
# corrs_2_to_1, corrs_1_to_2, sample_indices, x = uneven_lag_segment_matching(p1, sphere)
# xyz[corrs_2_to_1] - sphere_points


MIN_POINTS_PER_VIEW = int(1000)
# MIN_POINTS_PER_VIEW = int(0.4 * 128 * 128)


# new implementation for uneven point cloud
def process_pcd_uneven_wrapper(
    line,
    source_root,
    save_root,
    sphere_points=None,
    visualize_mapping=False,
    do_visibility=True,
    num_view=16,
    use_multiprocessing=True,
    views_per_batch=4,
):
    print("views_per_batch:", views_per_batch)
    process_pcd2sphere_uneven(
        os.path.join(source_root, line),
        save_root,
        sphere_points=sphere_points,
        visualize_mapping=visualize_mapping,
        do_visibility=do_visibility,
        num_view=num_view,
        use_multiprocessing=use_multiprocessing,
        views_per_batch=views_per_batch,
    )


def process_pcd2sphere_uneven(
    source_root,
    save_root,
    sphere_points=None,
    visualize_mapping=False,
    do_visibility=True,
    num_view=16,
    use_multiprocessing=True,
    views_per_batch=4,
):
    """
    Map point cloud to unit sphere using OT with multiprocessing optimization.
    Now includes view-level parallelization to process multiple views simultaneously.

    Args:
        use_multiprocessing: If True, use parallel processing for the three expensive computations.
                           If False, use sequential processing (original behavior).
        views_per_batch: Number of views to process in parallel (default: 4)
    """
    # retrieve the index of the files from source_root
    data_index = os.path.basename(source_root).split(".")[0]
    os.makedirs(os.path.join(save_root, data_index, "spherical_ot"), exist_ok=True)
    if visualize_mapping:
        os.makedirs(
            os.path.join(save_root, data_index, "spherical_ot_visualization"),
            exist_ok=True,
        )

    if sphere_points is None:
        sphere_points = init_unit_sphere_grid(N)

    sphere_points = sphere_points.cpu().numpy()

    if not do_visibility:
        print("Visibility processing disabled, skipping view processing")
        return

    # Process all 16 views with batched parallelization
    print(f"Processing {num_view} views with batch size {views_per_batch}")

    # Create batches of view IDs
    view_batches = []
    for i in range(0, num_view, views_per_batch):
        batch = list(range(i, min(i + views_per_batch, num_view)))
        view_batches.append(batch)

    print(f"Created {len(view_batches)} batches: {view_batches}")

    start_time = time.time()
    all_results = []

    # Process view batches in parallel
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(len(view_batches), 4)
    ) as executor:
        future_to_batch = {
            executor.submit(
                process_view_batch_wrapper,
                batch,
                source_root,
                save_root,
                data_index,
                sphere_points,
                use_multiprocessing,
                visualize_mapping,
            ): batch_idx
            for batch_idx, batch in enumerate(view_batches)
        }
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                print(
                    f"Completed batch {batch_idx}/{len(view_batches)} with {len(batch_results)} views"
                )
            except Exception as e:
                print(f"Batch {batch_idx} failed with error: {e}")

    total_time = time.time() - start_time
    print(
        f"All {len(all_results)} views processed successfully in {total_time:.2f} seconds"
    )
    print(f"Average time per view: {total_time / len(view_batches):.2f} seconds")

    return all_results


def uneven_lag_segment_matching(p1, p2, dynamics=False, method="fastlapjv"):
    cost, x, y, y_remaining, sample_indices, remaining_indices = compute_lap_unequal(
        p1, p2, scaling_factor=1000, dynamics=dynamics, method=method
    )

    if y_remaining is None:
        # If no remaining indices, return directly
        corrs_2_to_1 = y
        corrs_1_to_2 = x
        return corrs_2_to_1, corrs_1_to_2, None, None

    corrs_2_to_1 = y
    corrs_1_to_2 = x

    corrs_2_to_1_full = np.zeros((p2.shape[0],), dtype=int)

    corrs_2_to_1_full[sample_indices] = y
    corrs_2_to_1_full[remaining_indices] = y_remaining
    # merge the y and y_remaining to get the full mapping, ordered by the sample_indices and remaining_indices
    # xyz[corrs_2_to_1] matches to p2[sample_indices]
    # xyz[corrs_2_to_1_full] matches to p2
    return corrs_2_to_1, corrs_1_to_2, sample_indices, corrs_2_to_1_full


def compute_lap_dynamic(p1, p2, num_segments=4):
    #!!! This will corrupt the OT distribution. Don't use. Remove later
    num_pts = p1.shape[0]
    dynamic_segment_size = num_pts // num_segments
    indices = {}
    for i in range(num_segments):
        start_idx = i * dynamic_segment_size
        end_idx = (i + 1) * dynamic_segment_size if i < num_segments - 1 else num_pts
        indices[i] = (start_idx, end_idx)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_segments) as executor:
        futures = {
            executor.submit(
                compute_lap,
                p1[start_idx:end_idx],
                p2[start_idx:end_idx],
                scaling_factor=1000,
                index=i,
            ): i
            for i, (start_idx, end_idx) in indices.items()
        }
        results = [None] * num_segments  # Prepare a list to hold results in order
        for future in concurrent.futures.as_completed(futures):
            cost, x, y, index = future.result()
            results[index] = (cost, x, y)
    final_x = []
    final_y = []
    costs = []
    for i, (cost, x, y) in enumerate(results):
        start_idx, end_idx = indices[i]
        final_y.append(y + start_idx)
        final_x.append(x + start_idx)
        costs.append(cost)

    final_xs = np.concatenate(final_x, axis=0)
    final_ys = np.concatenate(final_y, axis=0)
    return final_xs, final_ys, costs


def compute_lap_unequal(
    p1_segment, p2_segment, scaling_factor=1000, dynamics=True, method="fastlapjv"
):
    n_target_points = p1_segment.shape[0]

    if n_target_points < MIN_POINTS_PER_VIEW:
        print(f"Warning: Not enough points this view, skip.")
        return None, None, None, None, None, None
    if p2_segment.shape[0] == n_target_points:
        # cost_matrix = ot.dist(p1_segment, p2_segment, metric='sqeuclidean')
        # scaled_cost_matrix = np.rint(cost_matrix * scaling_factor).astype(int)
        # x, y, cost = lapjv(scaled_cost_matrix)
        start_time = time.time()
        if dynamics:
            x, y, cost = compute_lap_dynamic(p1_segment, p2_segment, num_segments=4)
        else:
            cost, x, y, index = compute_lap(
                p1_segment, p2_segment, scaling_factor=scaling_factor, method=method
            )
        print(
            f"Time taken for {method}: {time.time() - start_time:.2f} seconds with {n_target_points} points"
        )
        return (
            cost,
            x,
            y,
            None,
            None,
            None,
        )  # No remaining indices or sample indices needed

    # FPS sample on p2_segment to match the number of points in p1_segment
    # p2_segment_tensor = torch.tensor(p2_segment).unsqueeze(0)
    # _, sample_indices = sample_farthest_points(p2_segment_tensor, K=n_target_points)
    # get indices of p2_segment
    # p2_segment_indices = np.arange(p2_segment.shape[0])
    # p2_segment_indices = np.concatenate([p2_segment_indices[:, None], np.zeros((p2_segment_indices.shape[0], 2), dtype=p2_segment_indices.dtype)], axis=1)
    # p2_segment_tensor = torch.tensor(p2_segment_indices).unsqueeze(0)

    # _, sample_indices = sample_farthest_points(p2_segment_tensor, K=n_target_points)

    # uniformly sample p2_segment to match the number of points in p1_segment
    step = p2_segment.shape[0] / n_target_points
    sample_indices = np.arange(0, p2_segment.shape[0], step)
    sample_indices = np.round(sample_indices).astype(int)
    sample_indices = sample_indices[:n_target_points]

    # sort sample_indices
    # sample_indices = sample_indices[0].cpu().numpy()
    sample_indices.sort()
    remaining_indices = np.setdiff1d(np.arange(p2_segment.shape[0]), sample_indices)
    remaining_indices.sort()
    subset_p2_segment = p2_segment[sample_indices]
    remaining_p2_segment = p2_segment[remaining_indices]
    """
    # save this for visualization
    save_root = os.path.join("tests/output/uneven_ot_test")
    subset_pt_pcd = o3d.geometry.PointCloud()
    subset_pt_pcd.points = o3d.utility.Vector3dVector(subset_p2_segment)
    o3d.io.write_point_cloud(os.path.join(save_root, "subset_p2_segment.ply"), subset_pt_pcd)
    remaining_pt_pcd = o3d.geometry.PointCloud()
    remaining_pt_pcd.points = o3d.utility.Vector3dVector(remaining_p2_segment)
    o3d.io.write_point_cloud(os.path.join(save_root, "remaining_p2_segment.ply"), remaining_pt_pcd)
    """
    start_time = time.time()
    # cost_matrix = ot.dist(p1_segment, subset_p2_segment, metric='sqeuclidean')
    # scaled_cost_matrix = np.rint(cost_matrix * scaling_factor).astype(int)
    # x, y, cost = lapjv(scaled_cost_matrix)

    # p1_segment[x]: reorder p1 to match p2
    # p2_segment[y]: reorder p2 to match p1
    if dynamics:
        x, y, cost = compute_lap_dynamic(p1_segment, subset_p2_segment, num_segments=4)
    else:
        cost, x, y, index = compute_lap(
            p1_segment, subset_p2_segment, scaling_factor=scaling_factor
        )
    print(
        f"Time taken for {method}: {time.time() - start_time:.2f} seconds with {n_target_points} points"
    )
    if cost is None:
        return None, None, None, None, None, None
    tree = cKDTree(subset_p2_segment)
    # For the remaining points in p2_segment, find the nearest points in p1_segment
    knn_indices = tree.query(remaining_p2_segment, k=1)[1]

    y_remaining = y[knn_indices]

    # y[i]: index in p1
    # x[i]: index in p2
    # knn_indices[i]: index in subset_p2_segment for the remaining points in p2
    return cost, x, y, y_remaining, sample_indices, remaining_indices


def parallel_uneven_lag_matching(
    visible_xyz,
    invisible_xyz,
    sphere_points,
    data_index,
    view,
    use_multiprocessing=True,
):
    """
    Optimized function to run three uneven_lag_segment_matching computations in parallel.

    Args:
        visible_xyz: numpy array for visible point cloud
        invisible_xyz: numpy array for invisible point cloud
        sphere_points: numpy array for sphere points
        use_multiprocessing: whether to use parallel processing (default: True)
        timeout: timeout in seconds for each computation (default: 600 = 10 minutes)

    Returns:
        tuple: (visible_results, invisible_results, full_results)
               where each result is (corrs_2_to_1, corrs_1_to_2)
    """
    # account time
    start_time = time.time()
    if not use_multiprocessing:
        # Sequential fallback
        print(f"Data {data_index} view {view} Sequential processing started")
        visible_results = uneven_lag_segment_matching(visible_xyz, sphere_points)
        invisible_results = uneven_lag_segment_matching(invisible_xyz, sphere_points)
        combined_xyz = np.concatenate((visible_xyz, invisible_xyz), axis=0)
        combined_xyz_sorted_indices = np.lexsort(
            (-combined_xyz[:, 0], -combined_xyz[:, 1], -combined_xyz[:, 2])
        )
        combined_xyz = combined_xyz[combined_xyz_sorted_indices]
        full_results = uneven_lag_segment_matching(combined_xyz, sphere_points)
        print(
            f"Sequential processing completed in {time.time() - start_time:.2f} seconds"
        )
        return visible_results, invisible_results, full_results

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
            combined_xyz = np.concatenate((visible_xyz, invisible_xyz), axis=0)
            combined_xyz_sorted_indices = np.lexsort(
                (-combined_xyz[:, 0], -combined_xyz[:, 1], -combined_xyz[:, 2])
            )
            combined_xyz = combined_xyz[combined_xyz_sorted_indices]

            # Submit all three computations simultaneously
            future_visible = executor.submit(
                uneven_lag_segment_matching, visible_xyz, sphere_points
            )
            future_invisible = executor.submit(
                uneven_lag_segment_matching, invisible_xyz, sphere_points
            )
            future_full = executor.submit(
                uneven_lag_segment_matching, combined_xyz, sphere_points
            )

            # Collect results with timeout
            visible_results = future_visible.result()
            invisible_results = future_invisible.result()
            full_results = future_full.result()

            print(
                f"Data {data_index} view {view} Parallel processing completed successfully in {time.time() - start_time:.2f} seconds"
            )
            return visible_results, invisible_results, full_results

    except Exception as e:
        print(
            f"Parallel processing failed ({e}), Suspecting bad data causing ot failure."
        )
        # # Sequential fallback
        # visible_results = uneven_lag_segment_matching(visible_xyz, sphere_points)
        # invisible_results = uneven_lag_segment_matching(invisible_xyz, sphere_points)
        # combined_xyz = np.concatenate((visible_xyz, invisible_xyz), axis=0)
        # combined_xyz_sorted_indices = np.lexsort((-combined_xyz[:, 0], -combined_xyz[:, 1], -combined_xyz[:, 2]))
        # combined_xyz = combined_xyz[combined_xyz_sorted_indices]
        # full_results = uneven_lag_segment_matching(combined_xyz, sphere_points)
        # print(f"Sequential processing completed in {time.time() - start_time:.2f} seconds")
        # return visible_results, invisible_results, full_results
        return (
            (None, None, None, None),
            (None, None, None, None),
            (None, None, None, None),
        )


def process_single_view_uneven(
    view_id,
    source_root,
    save_root,
    data_index,
    sphere_points,
    use_multiprocessing=True,
    visualize_mapping=True,
):
    """
    Process a single view with visible/invisible point clouds using optimal transport.

    Args:
        view_id: The view number to process
        source_root: Root directory containing the data
        save_root: Root directory for saving results
        data_index: Data index for organizing outputs
        sphere_points: Pre-computed sphere points
        use_multiprocessing: Whether to use parallel processing for OT computations
        visualize_mapping: Whether to save visualization files

    Returns:
        dict: Results containing correspondences for visible, invisible, and full point clouds
    """
    # skip if the output already exists
    if (
        os.path.exists(
            os.path.join(
                save_root, data_index, "spherical_ot", f"view{view_id}_full.pt"
            )
        )
        and os.path.exists(
            os.path.join(
                save_root, data_index, "spherical_ot", f"view{view_id}_visible.pt"
            )
        )
        and os.path.exists(
            os.path.join(
                save_root, data_index, "spherical_ot", f"view{view_id}_invisible.pt"
            )
        )
    ):
        print(f"Data {data_index} view {view_id} already processed, skipping.")
        return None
    visible_data = f"view{view_id}_pc.npz"
    invisible_data = f"view{view_id}_invisible_pc.npz"
    visible_data_path = os.path.join(source_root, visible_data)
    invisible_data_path = os.path.join(source_root, invisible_data)

    if not os.path.exists(visible_data_path) or not os.path.exists(invisible_data_path):
        print(f"Warning: Missing data files for data {data_index} view {view_id}")
        return None
    print(f"Loading data for data {data_index} view {view_id}")
    visible_pcd = load_npz(visible_data_path)
    invisible_pcd = load_npz(invisible_data_path)
    print("finish loading")

    visible_xyz = visible_pcd.xyz.cpu().numpy()
    invisible_xyz = invisible_pcd.xyz.cpu().numpy()
    visible_sorted_indices = np.lexsort(
        (-visible_xyz[:, 0], -visible_xyz[:, 1], -visible_xyz[:, 2])
    )
    invisible_sorted_indices = np.lexsort(
        (-invisible_xyz[:, 0], -invisible_xyz[:, 1], -invisible_xyz[:, 2])
    )
    visible_xyz = visible_xyz[visible_sorted_indices]
    invisible_xyz = invisible_xyz[invisible_sorted_indices]

    # OPTIMIZED: Use parallel processing for the three expensive computations
    start_time = time.time()
    print("starting OT matching...")
    (
        (
            visible_corrs_2_to_1,
            visible_corrs_1_to_2,
            visible_sample_indices,
            visible_corrs_2_to_1_full,
        ),
        (
            invisible_corrs_2_to_1,
            invisible_corrs_1_to_2,
            invisible_sample_indices,
            invisible_corrs_2_to_1_full,
        ),
        (full_corrs_2_to_1, full_corrs_1_to_2, _, _),
    ) = parallel_uneven_lag_matching(
        visible_xyz,
        invisible_xyz,
        sphere_points,
        data_index,
        view_id,
        use_multiprocessing=use_multiprocessing,
    )

    processing_time = time.time() - start_time
    print(
        f"Data {data_index} View {view_id} processing completed in {processing_time:.2f} seconds"
    )
    if visible_corrs_2_to_1 is None:
        print(
            f"Warning: No valid correspondences found for data {data_index} view {view_id}"
        )
        return None
    if invisible_corrs_2_to_1 is None:
        print(
            f"Warning: No valid correspondences found for data {data_index} view {view_id}"
        )
        return None
    if full_corrs_2_to_1 is None:
        print(
            f"Warning: No valid correspondences found for data {data_index} view {view_id}"
        )
        return None
    # Save results (this would be expanded based on your saving requirements)
    results = {
        "view_id": view_id,
        "visible_corrs": (visible_corrs_2_to_1, visible_corrs_1_to_2),
        "invisible_corrs": (invisible_corrs_2_to_1, invisible_corrs_1_to_2),
        "full_corrs": (full_corrs_2_to_1, full_corrs_1_to_2),
        "processing_time": processing_time,
    }

    full_xyz = np.concatenate((visible_xyz, invisible_xyz), axis=0)
    full_xyz_sorted_indices = np.lexsort(
        (-full_xyz[:, 0], -full_xyz[:, 1], -full_xyz[:, 2])
    )
    full_xyz = full_xyz[full_xyz_sorted_indices]

    visible_xyz_full = visible_xyz[visible_corrs_2_to_1_full]
    invisible_xyz_full = invisible_xyz[invisible_corrs_2_to_1_full]

    if visualize_mapping:
        os.makedirs(
            os.path.join(save_root, data_index, "spherical_ot_visualization"),
            exist_ok=True,
        )
        os.makedirs(os.path.join(save_root, data_index, "spherical_ot"), exist_ok=True)
        min_old, max_old = -1, 1
        min_new, max_new = 0, 1
        """
        # This part use sphere_points for coloring. But we want to use p1 for coloring
        p2 = sphere_points.copy()
        p2 = (p2 - min_old) / (max_old - min_old) * (max_new - min_new) + min_new  # rescale to 0 to 1 for color mapping
        
        visible_colors = p2[visible_sample_indices][visible_corrs_1_to_2]

        invisible_colors = p2[invisible_sample_indices][invisible_corrs_1_to_2]

        full_colors = p2[full_corrs_1_to_2]


        # Create point clouds for visualization
        mapping_of_visible = o3d.geometry.PointCloud()
        mapping_of_visible.points = o3d.utility.Vector3dVector(visible_xyz)
        mapping_of_visible.colors = o3d.utility.Vector3dVector(visible_colors)

        mapping_of_p2_visible = o3d.geometry.PointCloud()
        mapping_of_p2_visible.points = o3d.utility.Vector3dVector(sphere_points[visible_sample_indices][visible_corrs_1_to_2])
        mapping_of_p2_visible.colors = o3d.utility.Vector3dVector(visible_colors)

        o3d.io.write_point_cloud(os.path.join(save_root, data_index, "spherical_ot_visualization", f'view{view_id}_visible.ply'), mapping_of_visible)
        o3d.io.write_point_cloud(os.path.join(save_root, data_index, "spherical_ot_visualization", f'view{view_id}_visible_sphere.ply'), mapping_of_p2_visible)

        # Invisible point cloud visualization
        mapping_of_invisible = o3d.geometry.PointCloud()
        mapping_of_invisible.points = o3d.utility.Vector3dVector(invisible_xyz)
        mapping_of_invisible.colors = o3d.utility.Vector3dVector(invisible_colors)

        mapping_of_p2_invisible = o3d.geometry.PointCloud()
        mapping_of_p2_invisible.points = o3d.utility.Vector3dVector(sphere_points[invisible_sample_indices][invisible_corrs_1_to_2])
        mapping_of_p2_invisible.colors = o3d.utility.Vector3dVector(invisible_colors)

        o3d.io.write_point_cloud(os.path.join(save_root, data_index, "spherical_ot_visualization", f'view{view_id}_invisible.ply'), mapping_of_invisible)
        o3d.io.write_point_cloud(os.path.join(save_root, data_index, "spherical_ot_visualization", f'view{view_id}_invisible_sphere.ply'), mapping_of_p2_invisible)

        # Full point cloud visualization
        mapping_of_full = o3d.geometry.PointCloud()
        
        mapping_of_full.points = o3d.utility.Vector3dVector(full_xyz)
        mapping_of_full.colors = o3d.utility.Vector3dVector(full_colors)

        mapping_of_p2_full = o3d.geometry.PointCloud()
        mapping_of_p2_full.points = o3d.utility.Vector3dVector(sphere_points[full_corrs_1_to_2])
        mapping_of_p2_full.colors = o3d.utility.Vector3dVector(full_colors)

        o3d.io.write_point_cloud(os.path.join(save_root, data_index, "spherical_ot_visualization", f'view{view_id}_full.ply'), mapping_of_full)
        o3d.io.write_point_cloud(os.path.join(save_root, data_index, "spherical_ot_visualization", f'view{view_id}_full_sphere.ply'), mapping_of_p2_full)
        """
        # for visible points
        visible_colors = visible_xyz.copy()
        visible_colors = (visible_colors - min_old) / (max_old - min_old) * (
            max_new - min_new
        ) + min_new  # rescale to 0 to 1 for color mapping

        mapping_of_visible = o3d.geometry.PointCloud()
        mapping_of_visible.points = o3d.utility.Vector3dVector(visible_xyz)
        mapping_of_visible.colors = o3d.utility.Vector3dVector(visible_colors)

        mapping_of_spherical_visible = o3d.geometry.PointCloud()
        mapping_of_spherical_visible.points = o3d.utility.Vector3dVector(
            sphere_points[visible_sample_indices]
        )
        mapping_of_spherical_visible.colors = o3d.utility.Vector3dVector(
            visible_colors[visible_corrs_2_to_1]
        )

        mapping_of_spherical_visible_full = o3d.geometry.PointCloud()
        mapping_of_spherical_visible_full.points = o3d.utility.Vector3dVector(
            sphere_points
        )
        mapping_of_spherical_visible_full.colors = o3d.utility.Vector3dVector(
            visible_colors[visible_corrs_2_to_1_full]
        )

        o3d.io.write_point_cloud(
            os.path.join(
                save_root,
                data_index,
                "spherical_ot_visualization",
                f"view{view_id}_visible.ply",
            ),
            mapping_of_visible,
        )
        o3d.io.write_point_cloud(
            os.path.join(
                save_root,
                data_index,
                "spherical_ot_visualization",
                f"view{view_id}_spherical_visible.ply",
            ),
            mapping_of_spherical_visible,
        )
        o3d.io.write_point_cloud(
            os.path.join(
                save_root,
                data_index,
                "spherical_ot_visualization",
                f"view{view_id}_spherical_visible_full.ply",
            ),
            mapping_of_spherical_visible_full,
        )

        # for invisible points
        invisible_colors = invisible_xyz.copy()
        invisible_colors = (invisible_colors - min_old) / (max_old - min_old) * (
            max_new - min_new
        ) + min_new  # rescale to 0 to 1 for color mapping

        mapping_of_invisible = o3d.geometry.PointCloud()
        mapping_of_invisible.points = o3d.utility.Vector3dVector(invisible_xyz)
        mapping_of_invisible.colors = o3d.utility.Vector3dVector(invisible_colors)

        mapping_of_spherical_invisible = o3d.geometry.PointCloud()
        mapping_of_spherical_invisible.points = o3d.utility.Vector3dVector(
            sphere_points[invisible_sample_indices]
        )
        mapping_of_spherical_invisible.colors = o3d.utility.Vector3dVector(
            invisible_colors[invisible_corrs_2_to_1]
        )

        mapping_of_spherical_invisible_full = o3d.geometry.PointCloud()
        mapping_of_spherical_invisible_full.points = o3d.utility.Vector3dVector(
            sphere_points
        )
        mapping_of_spherical_invisible_full.colors = o3d.utility.Vector3dVector(
            invisible_colors[invisible_corrs_2_to_1_full]
        )

        o3d.io.write_point_cloud(
            os.path.join(
                save_root,
                data_index,
                "spherical_ot_visualization",
                f"view{view_id}_invisible.ply",
            ),
            mapping_of_invisible,
        )
        o3d.io.write_point_cloud(
            os.path.join(
                save_root,
                data_index,
                "spherical_ot_visualization",
                f"view{view_id}_spherical_invisible.ply",
            ),
            mapping_of_spherical_invisible,
        )
        o3d.io.write_point_cloud(
            os.path.join(
                save_root,
                data_index,
                "spherical_ot_visualization",
                f"view{view_id}_spherical_invisible_full.ply",
            ),
            mapping_of_spherical_invisible_full,
        )

        # for the whole pointcloud
        full_colors = full_xyz.copy()
        full_colors = (full_colors - min_old) / (max_old - min_old) * (
            max_new - min_new
        ) + min_new

        mapping_of_full = o3d.geometry.PointCloud()
        mapping_of_full.points = o3d.utility.Vector3dVector(full_xyz)
        mapping_of_full.colors = o3d.utility.Vector3dVector(full_colors)

        mapping_of_p2_full = o3d.geometry.PointCloud()
        mapping_of_p2_full.points = o3d.utility.Vector3dVector(sphere_points)
        mapping_of_p2_full.colors = o3d.utility.Vector3dVector(
            full_colors[full_corrs_2_to_1]
        )

        o3d.io.write_point_cloud(
            os.path.join(
                save_root,
                data_index,
                "spherical_ot_visualization",
                f"view{view_id}_full.ply",
            ),
            mapping_of_full,
        )
        o3d.io.write_point_cloud(
            os.path.join(
                save_root,
                data_index,
                "spherical_ot_visualization",
                f"view{view_id}_full_sphere.ply",
            ),
            mapping_of_p2_full,
        )

    xyz_offset_visible = torch.from_numpy(
        visible_xyz[visible_corrs_2_to_1_full] - sphere_points
    )
    xyz_offset_invisible = torch.from_numpy(
        invisible_xyz[invisible_corrs_2_to_1_full] - sphere_points
    )
    xyz_offset_full = torch.from_numpy(full_xyz[full_corrs_2_to_1] - sphere_points)

    new_pcd_visible = PointCloud(
        xyz_offset_visible,
        visible_pcd.normal[visible_sorted_indices][visible_corrs_2_to_1_full],
    )
    new_pcd_invisible = PointCloud(
        xyz_offset_invisible,
        invisible_pcd.normal[invisible_sorted_indices][invisible_corrs_2_to_1_full],
    )
    new_pcd_full = PointCloud(
        xyz_offset_full,
        torch.cat(
            (
                visible_pcd.normal[visible_sorted_indices],
                invisible_pcd.normal[invisible_sorted_indices],
            ),
            dim=0,
        )[full_xyz_sorted_indices][full_corrs_2_to_1],
    )

    new_pcd_visible_dict = {
        "xyz": new_pcd_visible.xyz,
        "normal": new_pcd_visible.normal,
        "sample_indices": visible_sample_indices,
    }
    new_pcd_invisible_dict = {
        "xyz": new_pcd_invisible.xyz,
        "normal": new_pcd_invisible.normal,
        "sample_indices": invisible_sample_indices,
    }
    new_pcd_full_dict = {"xyz": new_pcd_full.xyz, "normal": new_pcd_full.normal}

    # Save the processed point clouds
    torch.save(
        new_pcd_visible_dict,
        os.path.join(
            save_root, data_index, "spherical_ot", f"view{view_id}_visible.pt"
        ),
    )
    torch.save(
        new_pcd_invisible_dict,
        os.path.join(
            save_root, data_index, "spherical_ot", f"view{view_id}_invisible.pt"
        ),
    )
    torch.save(
        new_pcd_full_dict,
        os.path.join(save_root, data_index, "spherical_ot", f"view{view_id}_full.pt"),
    )
    # Also save the original point cloud and mapping
    original_pcd_dict = {
        "visible": {
            "xyz": visible_pcd.xyz,
            "normal": visible_pcd.normal,
            "sorted_indices": visible_sorted_indices,
            "corrs_2_to_1": visible_corrs_2_to_1,
            "corrs_1_to_2": visible_corrs_1_to_2,
            "sample_indices": visible_sample_indices,
            "corrs_2_to_1_full": visible_corrs_2_to_1_full,
        },
        "invisible": {
            "xyz": invisible_pcd.xyz,
            "normal": invisible_pcd.normal,
            "sorted_indices": invisible_sorted_indices,
            "corrs_2_to_1": invisible_corrs_2_to_1,
            "corrs_1_to_2": invisible_corrs_1_to_2,
            "sample_indices": invisible_sample_indices,
            "corrs_2_to_1_full": invisible_corrs_2_to_1_full,
        },
        "full": {
            "xyz": full_xyz,
            "normal": torch.cat(
                (
                    visible_pcd.normal[visible_sorted_indices],
                    invisible_pcd.normal[invisible_sorted_indices],
                ),
                dim=0,
            )[full_xyz_sorted_indices],
            "corrs_2_to_1": full_corrs_2_to_1,
            "corrs_1_to_2": full_corrs_1_to_2,
        },
    }
    torch.save(
        original_pcd_dict,
        os.path.join(
            save_root,
            data_index,
            "spherical_ot",
            f"view{view_id}_original_and_mapping.pt",
        ),
    )
    return results


def process_view_batch_wrapper(
    view_ids,
    source_root,
    save_root,
    data_index,
    sphere_points,
    use_multiprocessing,
    visualize_mapping,
):
    """Wrapper function for multiprocessing view batches"""
    batch_results = []
    for view_id in view_ids:
        result = process_single_view_uneven(
            view_id,
            source_root,
            save_root,
            data_index,
            sphere_points,
            use_multiprocessing,
            visualize_mapping,
        )
        # if result is not None:
        batch_results.append(result)
    return batch_results


def uneven_pcd_main_new():
    """
    Example useage: (remember to get enough CPU and memory)
    python uneven_ot_structuralization.py --visualize_mapping --do_visibility --split 1 --num_workers 4 --ignore_idx_less_than 0
    python uneven_ot_structuralization.py --visualize_mapping --do_visibility --num_workers 4
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        type=str,
        # default="data/Objaverse_processed_1600_no_watertight",
        default="data/PCN_processed",
    )
    parser.add_argument("--visualize_mapping", action="store_true")
    parser.add_argument(
        "--do_visibility",
        action="store_true",
        help="If set, will also do atlas for visible pt",
    )
    parser.add_argument(
        "--num_view",
        type=int,
        default=16,
        help="Number of views to generate for visibility",
    )
    parser.add_argument(
        "--views_per_batch",
        type=int,
        default=4,
        help="Number of views to generate per batch",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--full_parallel",
        action="store_true",
        help="If set, will use all available CPU cores",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="If set, will split the data into multiple parts and process each part separately",
    )
    parser.add_argument(
        "--split_base_dir",
        type=str,
        # default="shapeatlas_utils/objaverse_splits/pc2sph_split_7000",
        default="shapeatlas_utils/PCN_splits/pc2sph_5500",
        help="Base directory for split files",
    )

    parser.add_argument(
        "--ignore_idx_less_than",
        type=int,
        default=0,
        help="If set, will ignore points with index less than this value",
    )

    args = parser.parse_args()

    if args.full_parallel:
        CPU_COUNT = multiprocessing.cpu_count()
        num_workers = (
            CPU_COUNT // 5
        )  # since each ot use 4 threads and 1 threads to call.
    # Number of worker processes to use
    else:
        num_workers = args.num_workers
    # list all the files under source_root
    if args.split is None:
        lines = [
            d
            for d in os.listdir(args.source_root)
            if os.path.isdir(os.path.join(args.source_root, d))
        ]
        lines = sorted(lines, key=int)
    else:
        if args.split is None:
            lines = ["0"]
            pass
        else:
            split_file = os.path.join(args.split_base_dir, f"split_{args.split}.txt")
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file {split_file} does not exist.")
            with open(split_file, "r") as f:
                lines = f.read().splitlines()
            lines = [
                line.strip() for line in lines if line.strip()
            ]  # Remove empty lines
            lines = sorted(lines, key=int)  # Ensure the lines are sorted numerically

    # lines = ['0']
    sphere_points = init_unit_sphere_grid(N)
    # Create a pool of workers and map the processing function to the data
    print("views_per_batch", args.views_per_batch)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all the tasks and wait for them to complete
        futures = [
            executor.submit(
                process_pcd_uneven_wrapper,
                line,
                args.source_root,
                args.source_root,
                sphere_points=sphere_points,
                visualize_mapping=args.visualize_mapping,
                do_visibility=args.do_visibility,
                num_view=args.num_view,
                use_multiprocessing=True,
                views_per_batch=args.views_per_batch,
            )
            for line in lines if int(line) >= args.ignore_idx_less_than
        ]
        for future in futures:
            future.result()  # You can add error handling here if needed


def uneven_sphere2grid(
    source_root, save_root, lines, correspondence_path="", num_view=16
):
    sphere_points = init_unit_sphere_grid(N)
    latlon = equirectangular_projection(sphere_points)
    if os.path.exists(correspondence_path):
        # load
        print("found correspondences, load it ... ")
        correspondences = np.load(correspondence_path)
        corrs_2_to_1 = correspondences["corrs_2_to_1"]
        corrs_1_to_2 = correspondences["corrs_1_to_2"]
        # latlon_sorted_indices = correspondences["latlon_sorted_indices"]
        # reversed_latlon_sorted_indices = correspondences["reversed_latlon_sorted_indices"]
    else:
        os.makedirs(os.path.join(save_root, "correspondences"), exist_ok=True)
        (
            corrs_2_to_1,
            corrs_1_to_2,
            latlon_sorted_indices,
            reversed_latlon_sorted_indices,
        ) = simple_ot_map_eq2grid(latlon)
        # save the correspondences
        np.savez_compressed(
            os.path.join(save_root, "correspondences", "correspondences_no_sort.npz"),
            corrs_2_to_1=corrs_2_to_1,
            corrs_1_to_2=corrs_1_to_2,
        )
        # latlon_sorted_indices=latlon_sorted_indices,
        # reversed_latlon_sorted_indices=reversed_latlon_sorted_indices)

    for line in lines:
        print("Processing data: {}".format(line))
        root_data_folder = os.path.join(source_root, line, "spherical_ot")
        save_root_folder = os.path.join(save_root, line, "atlas")
        os.makedirs(save_root_folder, exist_ok=True)
        for view in range(num_view):
            full_pt_file = os.path.join(root_data_folder, f"view{view}_full.pt")
            view_visible_pt_file = os.path.join(
                root_data_folder, f"view{view}_visible.pt"
            )
            view_invisible_pt_file = os.path.join(
                root_data_folder, f"view{view}_invisible.pt"
            )

            if (
                not os.path.exists(full_pt_file)
                or not os.path.exists(view_visible_pt_file)
                or not os.path.exists(view_invisible_pt_file)
            ):
                print(f"Warning: Missing data files for data {line} view {view}")
                print("Possibily due to bad data for valid OT correspondence. Skip ...")
                continue

            # skip if already done
            if (
                os.path.exists(os.path.join(save_root_folder, f"view{view}_full.pt"))
                and os.path.exists(
                    os.path.join(save_root_folder, f"view{view}_visible.pt")
                )
                and os.path.exists(
                    os.path.join(save_root_folder, f"view{view}_invisible.pt")
                )
                and os.path.exists(
                    os.path.join(save_root_folder, f"view{view}_full.png")
                )
                and os.path.exists(
                    os.path.join(save_root_folder, f"view{view}_visible.png")
                )
                and os.path.exists(
                    os.path.join(save_root_folder, f"view{view}_invisible.png")
                )
            ):
                print(f"Data {line} view {view} already processed, skip ...")
                continue
            full_pcd = torch.load(full_pt_file)
            view_visible_pcd = torch.load(view_visible_pt_file)
            view_invisible_pcd = torch.load(view_invisible_pt_file)

            full_xyz = full_pcd["xyz"]
            full_normal = full_pcd["normal"]
            view_visible_xyz = view_visible_pcd["xyz"]
            view_visible_normal = view_visible_pcd["normal"]
            view_visible_sample_indices = view_visible_pcd["sample_indices"]
            view_invisible_xyz = view_invisible_pcd["xyz"]
            view_invisible_normal = view_invisible_pcd["normal"]
            view_invisible_sample_indices = view_invisible_pcd["sample_indices"]

            full_xyz = full_xyz[corrs_2_to_1]
            full_normal = full_normal[corrs_2_to_1]

            view_visible_xyz = view_visible_xyz[corrs_2_to_1]
            view_visible_normal = view_visible_normal[corrs_2_to_1]
            view_visible_sample_mask = torch.zeros(
                sphere_points.shape[0], dtype=torch.bool
            )
            view_visible_sample_mask[view_visible_sample_indices] = True
            view_visible_sample_mask = view_visible_sample_mask[corrs_2_to_1]

            view_invisible_xyz = view_invisible_xyz[corrs_2_to_1]
            view_invisible_normal = view_invisible_normal[corrs_2_to_1]
            view_invisible_sample_mask = torch.zeros(
                sphere_points.shape[0], dtype=torch.bool
            )
            view_invisible_sample_mask[view_invisible_sample_indices] = True
            view_invisible_sample_mask = view_invisible_sample_mask[corrs_2_to_1]

            full_xyz = full_xyz.reshape(N_SQRT, N_SQRT, 3)
            full_normal = full_normal.reshape(N_SQRT, N_SQRT, 3)
            view_visible_xyz = view_visible_xyz.reshape(N_SQRT, N_SQRT, 3)
            view_visible_normal = view_visible_normal.reshape(N_SQRT, N_SQRT, 3)
            view_visible_sample_mask = view_visible_sample_mask.reshape(N_SQRT, N_SQRT)

            view_invisible_xyz = view_invisible_xyz.reshape(N_SQRT, N_SQRT, 3)
            view_invisible_normal = view_invisible_normal.reshape(N_SQRT, N_SQRT, 3)
            view_invisible_sample_mask = view_invisible_sample_mask.reshape(
                N_SQRT, N_SQRT
            )

            torch.save(
                {"xyz": full_xyz, "normal": full_normal},
                os.path.join(save_root_folder, f"view{view}_full.pt"),
            )
            torch.save(
                {
                    "xyz": view_visible_xyz,
                    "normal": view_visible_normal,
                    "mask": view_visible_sample_mask,
                },
                os.path.join(save_root_folder, f"view{view}_visible.pt"),
            )
            torch.save(
                {
                    "xyz": view_invisible_xyz,
                    "normal": view_invisible_normal,
                    "mask": view_invisible_sample_mask,
                },
                os.path.join(save_root_folder, f"view{view}_invisible.pt"),
            )

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

            full_xyz_image = to_pil_image(full_xyz_vis.permute(2, 0, 1))
            full_normal_image = to_pil_image(full_normal_vis.permute(2, 0, 1))

            view_visible_xyz_image = to_pil_image(view_visible_xyz_vis.permute(2, 0, 1))
            view_visible_normal_image = to_pil_image(
                view_visible_normal_vis.permute(2, 0, 1)
            )
            view_visible_mask = to_pil_image(view_visible_mask)

            view_invisible_xyz_image = to_pil_image(
                view_invisible_xyz_vis.permute(2, 0, 1)
            )
            view_invisible_normal_image = to_pil_image(
                view_invisible_normal_vis.permute(2, 0, 1)
            )
            view_invisible_mask = to_pil_image(view_invisible_mask)

            # save
            full_xyz_image.save(os.path.join(save_root_folder, f"view{view}_full.png"))
            full_normal_image.save(
                os.path.join(save_root_folder, f"view{view}_full_normal.png")
            )

            view_visible_xyz_image.save(
                os.path.join(save_root_folder, f"view{view}_visible.png")
            )
            view_visible_normal_image.save(
                os.path.join(save_root_folder, f"view{view}_visible_normal.png")
            )
            view_visible_mask.save(
                os.path.join(save_root_folder, f"view{view}_visible_mask.png")
            )

            view_invisible_xyz_image.save(
                os.path.join(save_root_folder, f"view{view}_invisible.png")
            )
            view_invisible_normal_image.save(
                os.path.join(save_root_folder, f"view{view}_invisible_normal.png")
            )
            view_invisible_mask.save(
                os.path.join(save_root_folder, f"view{view}_invisible_mask.png")
            )


def uneven_sphere2grid_main():
    """
    load the xyz and normal from the point cloud, and map them to the grid points using the OT mapping
    Example usage:
    python uneven_ot_structuralization.py --split 0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        type=str,
        default="data/PCN_processed",
    )
    parser.add_argument(
        "--correspondance_path",
        type=str,
        default="shapeatlas_utils/sph2grid_correspondences/correspondences_no_sort.npz",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="If set, will split the data into multiple parts and process each part separately",
    )
    parser.add_argument(
        "--split_base_dir",
        type=str,
        default="shapeatlas_utils/PCN_splits/sph2grid_16000",
        help="Base directory for split files",
    )
    args = parser.parse_args()
    print('Processing data from ', args.source_root)
    # list all the files under source_root
    if args.split is None:
        lines = [
            d
            for d in os.listdir(args.source_root)
            if os.path.isdir(os.path.join(args.source_root, d))
        ]
        lines = sorted(lines, key=int)
    else:
        if args.split is None:
            lines = ["0"]
            pass
        else:
            split_file = os.path.join(args.split_base_dir, f"split_{args.split}.txt")
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file {split_file} does not exist.")
            with open(split_file, "r") as f:
                lines = f.read().splitlines()
            lines = [
                line.strip() for line in lines if line.strip()
            ]  # Remove empty lines
            lines = sorted(lines, key=int)  # Ensure the lines are sorted numerically

    # lines = ['0', '3684']
    print(f"Processing {len(lines)} data")
    uneven_sphere2grid(
        source_root=args.source_root,
        save_root=args.source_root,
        lines=lines,
        correspondence_path=args.correspondance_path,
    )


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        type=str,
        choices=["sphere", "atlas"],
        help="'sphere': map point clouds to sphere (step 1), 'atlas': map sphere to 2D atlas grid (step 2)",
    )
    # Parse only the first arg; the rest are forwarded to the sub-function's parser
    step_arg, remaining = parser.parse_known_args()
    # Reset sys.argv so the sub-function's argparse sees only its own args
    sys.argv = [sys.argv[0]] + remaining

    if step_arg.step == "sphere":
        uneven_pcd_main_new()
    else:
        uneven_sphere2grid_main()

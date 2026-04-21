
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import concurrent.futures
from shapeatlas_utils.ot_structuralization import load_npz
from shapeatlas_utils.uneven_ot_structuralization import *
import time
source_root = 'data/ShapeNet_processed_tmp'


def load_data(data_idx, view_id):
    visible_data = f'view{view_id}_pc.npz'
    invisible_data = f'view{view_id}_invisible_pc.npz'
    print(f"Loading data for view {view_id}")
    visible_data_path = os.path.join(source_root, f'{data_idx}', visible_data)
    invisible_data_path = os.path.join(source_root, f'{data_idx}', invisible_data)
    visible_pcd = load_npz(visible_data_path)
    invisible_pcd = load_npz(invisible_data_path)


# data_idx = 570
# with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
#     for view_id in range(16):
#         executor.submit(load_data, data_idx, view_id)

def process_single_view_uneven(view_ids, source_root, save_root, data_index, sphere_points, use_multiprocessing=True, visualize_mapping=True):
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
    for view_id in view_ids:
        visible_data = f'view{view_id}_pc.npz'
        invisible_data = f'view{view_id}_invisible_pc.npz'
        visible_data_path = os.path.join(source_root, f'{data_index}', visible_data)
        invisible_data_path = os.path.join(source_root, f'{data_index}', invisible_data)

        if not os.path.exists(visible_data_path) or not os.path.exists(invisible_data_path):
            print(f"Warning: Missing data files for data {data_index} view {view_id}")
            import pdb; pdb.set_trace()
            return None
        print(f"Loading data for data {data_index} view {view_id}")
        visible_pcd = load_npz(visible_data_path)
        invisible_pcd = load_npz(invisible_data_path)
        print('finish loading')

        visible_xyz = visible_pcd.xyz.cpu().numpy()
        invisible_xyz = invisible_pcd.xyz.cpu().numpy()
        visible_sorted_indices = np.lexsort((-visible_xyz[:, 0], -visible_xyz[:, 1], -visible_xyz[:, 2]))
        invisible_sorted_indices = np.lexsort((-invisible_xyz[:, 0], -invisible_xyz[:, 1], -invisible_xyz[:, 2]))
        visible_xyz = visible_xyz[visible_sorted_indices]
        invisible_xyz = invisible_xyz[invisible_sorted_indices]

        # OPTIMIZED: Use parallel processing for the three expensive computations
        start_time = time.time()
        print('starting OT matching...')
        print(f"Data {data_index} view {view_id} Sequential processing started")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                combined_xyz = np.concatenate((visible_xyz, invisible_xyz), axis=0)
                combined_xyz_sorted_indices = np.lexsort((-combined_xyz[:, 0], -combined_xyz[:, 1], -combined_xyz[:, 2]))
                combined_xyz = combined_xyz[combined_xyz_sorted_indices]
                future_visible = executor.submit(uneven_lag_segment_matching, visible_xyz, sphere_points)
                future_invisible = executor.submit(uneven_lag_segment_matching, invisible_xyz, sphere_points)
                future_full = executor.submit(uneven_lag_segment_matching, combined_xyz, sphere_points)

                visible_results = future_visible.result()
                invisible_results = future_invisible.result()
                full_results = future_full.result()
                # print(f"Sequential processing completed in {time.time() - start_time:.2f} seconds")
                processing_time = time.time() - start_time
                print(f"Data {data_index} View {view_id} processing completed in {processing_time:.2f} seconds")
        except Exception as e:
            print('bad data')
            print(e)
sphere_points = init_unit_sphere_grid(N).cpu().numpy()

view_ids = [5]
process_single_view_uneven(view_ids, source_root, "", 570, sphere_points, use_multiprocessing=True, visualize_mapping=True)
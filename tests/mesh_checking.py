import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import trimesh
from shapeatlas_utils.utils import mesh_sort
import argparse
import ipdb
from shapeatlas_utils.data_processor import *
def main1():
    parser = argparse.ArgumentParser()

    parser.add_argument("--shapenet_base_dir", type=str, required=True)
    parser.add_argument("--data", default='0.npz', type=str)
    parser.add_argument("--input_type", default='mesh', type=str)
    parser.add_argument("--do_pc_count", default=False, type=bool)
    args = parser.parse_args()
    data_path = os.path.join(args.shapenet_base_dir, args.data)


    if args.do_pc_count:
        assert args.input_type == 'pc_normal'
        files = os.listdir(args.shapenet_base_dir)
        npy_files = [f for f in files if f.endswith('.npz')]
        npy_files.sort()
        average_pc_count = 0
        for file in npy_files:
            data_path = os.path.join(args.shapenet_base_dir, file)
            data = np.load(data_path)
            pc_count = data['pc_normal'].shape[0]
            average_pc_count += pc_count
        average_pc_count /= len(npy_files)
        print(f"Average point cloud count: {average_pc_count}")
    if args.input_type == 'pc_normal':
        data_path = os.path.join(args.shapenet_base_dir, args.data)
        data = np.load(data_path)
    else:
        cur_mesh = trimesh.load(data_path)

        vertices = cur_mesh.vertices
        faces = cur_mesh.faces
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max()
        vertices = vertices.clip(-0.5, 0.5)
        vertices, faces = mesh_sort(vertices, faces)


def main2():
    npz_file = '/scr/yaohe/data/ShapeNet_all_pcd_npz/train.npz'

    npz_list = np.load(npz_file, allow_pickle=True)
    npz_list = npz_list['npz_list'].tolist()

    idx = 0
    mesh_data = npz_list[idx]

    process_mesh(mesh_data,idx, 'test')

if __name__ == "__main__":
    main2()
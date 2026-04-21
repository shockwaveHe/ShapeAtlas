import concurrent.futures
import json
import logging
import os
import random
import threading
from pathlib import Path

import cv2
import imageio
import ipdb
import networkx as nx
import numpy as np
import torch
import trimesh
import yaml
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from torch.utils.data import Dataset
from tqdm import tqdm


class AtlasDataset(Dataset):
    def __init__(self, opt, dataset_name='', phase='train'):
        self.opt = opt
        self.phase = phase
        self.dataset_name = dataset_name

        if self.phase == 'train':
            self.split = opt[self.dataset_name]['train_split']
        elif self.phase == 'test':
            self.split = opt[self.dataset_name]['test_split']

        self.dataset_folder = self.opt[self.dataset_name]['dataset_dir']
        self.num_views = self.opt.get('num_views', 16)
        self.mode = self.opt['mode'] # finetune, train
        self.w_mesh_supervision = self.opt.get('w_mesh_supervision', False)
        if self.mode == 'finetune':
            print(" [Mode] Finetune UNet. Only load gt, view and id.")
        self.data = []
        self._data_lock = threading.Lock()
        self.load_data()
    
    def load_single_data(self,line):
            line = line.strip()
            sample_root = os.path.join(self.dataset_folder, line, 'atlas')
            views = range(self.num_views)
            for i in views:
                if os.path.exists(os.path.join(sample_root, f'view{i}_full.pt')) and \
                        os.path.exists(os.path.join(sample_root, f'view{i}_visible.pt')):
                    data_sample = {
                        'gt': os.path.join(sample_root, f'view{i}_full.pt'), # full pt
                        'input': os.path.join(sample_root, f'view{i}_visible.pt'), # visible pt as input
                        'id': line.split('.')[0],
                        'view': i,
                        'gt_normal_image': os.path.join(sample_root, f'view{i}_full_normal.png'),
                        'gt_xyz_image': os.path.join(sample_root, f'view{i}_full.png'),
                        'input_normal_image': os.path.join(sample_root, f'view{i}_visible_normal.png'),
                        'input_xyz_image': os.path.join(sample_root, f'view{i}_visible.png'),
                        'input_mask': os.path.join(sample_root, f'view{i}_visible_mask.png'),
                        'gt_mesh': os.path.join(self.dataset_folder, line, f'view{i}_gt_mesh.obj'),
                        'taxonomy': '',
                        'uid': ''
                    }
                    with self._data_lock:
                        self.data.append(data_sample)

    def load_single_view_data(self, line):
        # only for evaluation
        input_line = line
        line = input_line.strip().split(',')[0]
        sample_root = os.path.join(self.dataset_folder, line, 'atlas')
        views = [input_line.strip().split(',')[1]]
        for i in views:
            if os.path.exists(os.path.join(sample_root, f'view{i}_full.pt')) and \
                    os.path.exists(os.path.join(sample_root, f'view{i}_visible.pt')):
                if self.dataset_name == 'PCN' or self.dataset_name == 'ShapeNet':
                    all_files = os.listdir(os.path.join(self.dataset_folder, line))
                    meta_file = [f for f in all_files if 'models.npz' in f][0]
                    taxonomy = meta_file.split('_')[0]
                    uid = meta_file.split('_')[1]
                elif self.dataset_name == 'Objaverse':
                    # read uid.txt to get taxonomy and uid
                    uid_file = os.path.join(self.dataset_folder, line, 'uid.txt')
                    with open(uid_file, 'r') as f:
                        uid = f.readline().strip()
                data_sample = {
                    'gt': os.path.join(sample_root, f'view{i}_full.pt'), # full pt
                    'input': os.path.join(sample_root, f'view{i}_visible.pt'), # visible pt as input
                    'id': line.split('.')[0],
                    'view': i,
                    'gt_normal_image': os.path.join(sample_root, f'view{i}_full_normal.png'),
                    'gt_xyz_image': os.path.join(sample_root, f'view{i}_full.png'),
                    'input_normal_image': os.path.join(sample_root, f'view{i}_visible_normal.png'),
                    'input_xyz_image': os.path.join(sample_root, f'view{i}_visible.png'),
                    'input_mask': os.path.join(sample_root, f'view{i}_visible_mask.png'),
                    'gt_mesh': os.path.join(self.dataset_folder, line, f'view{i}_gt_mesh.obj')
                }
                if self.dataset_name == 'PCN' or self.dataset_name == 'ShapeNet':
                    data_sample['taxonomy'] = taxonomy
                    data_sample['uid'] = uid
                elif self.dataset_name == 'Objaverse':
                    data_sample['uid'] = uid
                    data_sample['taxonomy'] = ''
                else:
                    data_sample['taxonomy'] = ''
                    data_sample['uid'] = ''
                # with self._data_lock:
                self.data.append(data_sample)
            else:
                print(f'Warning: missing data for {line} view {i}')
                    
    def load_data(self):
        num_workers = os.cpu_count() // 2
        with open(self.split, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc='Loading data'):
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    if 'view' in self.split:
                        future = executor.submit(self.load_single_view_data, line)
                    else:
                        future = executor.submit(self.load_single_data, line)
                    future.result()  # Wait for the thread to complete
                # if len(self.data) > 20:
                #     break
        print(f'Loaded {len(self.data)} samples for {self.dataset_name} dataset.')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        gt_file = item['gt']
        input_file = item['input']

        gt_atlas = torch.load(gt_file)
        input_atlas = torch.load(input_file)
        
        gt_xyz = gt_atlas['xyz']
        gt_normal = gt_atlas['normal']
        # concatenate xyz and normal
        gt_xyz_normal = torch.cat((gt_xyz, gt_normal), dim=-1)
        input_xyz = input_atlas['xyz']
        input_normal = input_atlas['normal']
        input_mask = input_atlas['mask']
        gt_mesh_file = item['gt_mesh']
        gt_mesh = load_objs_as_meshes([gt_mesh_file])
        
        # concatenate xyz and normal
        input_xyz_normal = torch.cat((input_xyz, input_normal), dim=-1)

        verts = gt_mesh.verts_list()[0]
        faces = gt_mesh.faces_list()[0]
        verts = torch.tensor(verts, dtype=torch.float32)
        faces = torch.tensor(faces, dtype=torch.long)
        return {
            'gt': gt_xyz_normal,
            'input': input_xyz_normal,
            'id': self.dataset_name + '_' + str(item['id']),
            'input_mask': input_mask,
            'view': item['view'],
            'mesh_verts': verts,
            'mesh_faces': faces,
            'idx': idx,
            'taxonomy': item['taxonomy'],
            'uid': item['uid']
        }


    def get_mesh_data(self, idx):
        pass

    def tokenize(self, mesh):
        naive_v_length = mesh.faces.shape[0] * 9

        graph = mesh.vertex_adjacency_graph

        unvisited_faces = mesh.faces.copy()
        dis_vertices = np.asarray((mesh.vertices.copy() + 0.5) * self.num_tokens)

        sequence = []
        while unvisited_faces.shape[0] > 0:
            # find the face with the smallest index
            if len(sequence) == 0 or sequence[-1] == -1:
                cur_face = unvisited_faces[0]
                unvisited_faces = unvisited_faces[1:]
                sequence.extend(cur_face.tolist())
            else:
                cur_cache = sequence[-2:]
                commons = sorted(list(nx.common_neighbors(graph, cur_cache[0], cur_cache[1])))
                next_token = None
                for common in commons:
                    common_face = sorted(np.array(cur_cache + [common]))
                    # find index of common face
                    equals = np.where((unvisited_faces == common_face).all(axis=1))[0]
                    assert len(equals) == 1 or len(equals) == 0
                    if len(equals) == 1:
                        next_token = common
                        next_face_index = equals[0]
                        break
                if next_token is not None:
                    unvisited_faces = np.delete(unvisited_faces, next_face_index, axis=0)
                    sequence.append(int(next_token))
                else:
                    sequence.append(-1)

        final_sequence = []
        id_sequence = []
        split_flag = 3
        for token_id in sequence:
            if token_id == -1:
                final_sequence.append(self.num_tokens)
                id_sequence.append(3)
                split_flag = 3
            else:
                final_sequence.extend(dis_vertices[token_id].tolist())
                if split_flag == 0:
                    id_sequence.extend([7,8,9])
                else:
                    split_flag -= 1
                    id_sequence.extend([4,5,6])

        assert len(final_sequence) == len(id_sequence)
        cur_ratio = len(final_sequence) / naive_v_length
        if cur_ratio >= self.max_seq_ratio:
            # print(f"token sequence too long: {cur_ratio}")
            return None, None
        else:
            return final_sequence, id_sequence

    def sort_vertices_and_faces(self, vertices_, faces_):
        assert (vertices_ <= 0.5).all() and (vertices_ >= -0.5).all() # [-0.5, 0.5]
        vertices = (vertices_+0.5) * self.num_tokens # [0, num_tokens]
        vertices -= 0.5 # for evenly distributed, [-0.5, num_tokens -0.5] will be round to 0 or num_tokens (-1)
        vertices_quantized_ = np.clip(vertices.round(), 0, self.num_tokens-1).astype(int)  # [0, num_tokens -1]
        origin_face_num = len(faces_)

        cur_mesh = trimesh.Trimesh(vertices=vertices_quantized_, faces=faces_)

        cur_mesh.merge_vertices()
        cur_mesh.update_faces(cur_mesh.nondegenerate_faces())
        cur_mesh.update_faces(cur_mesh.unique_faces())
        cur_mesh.remove_unreferenced_vertices()

        if len(cur_mesh.faces) < self.min_triangles/3*2 or len(cur_mesh.faces) < origin_face_num*0.2:
            return None, None

        sort_inds = np.lexsort(cur_mesh.vertices.T)
        vertices = cur_mesh.vertices[sort_inds]
        faces = [np.argsort(sort_inds)[f] for f in cur_mesh.faces]

        faces = [sorted(sub_arr) for sub_arr in faces]

        def sort_faces(face):
            return face[0], face[1], face[2]

        faces = sorted(faces, key=sort_faces)

        vertices = vertices / self.num_tokens - 0.5  # [0, num_tokens -1] to [-0.5, 0.5)  for computing

        return vertices, faces
    
class ShapeNetAtlasDataset(AtlasDataset):
    def __init__(self, opt, phase='train'):
        super().__init__(opt, 'ShapeNet', phase)

class ObjaverseAtlasDataset(AtlasDataset):
    def __init__(self, opt, phase='train'):
        super().__init__(opt, 'Objaverse', phase)
        
class PCNAtlasDataset(AtlasDataset):
    def __init__(self, opt, phase='train'):
        super().__init__(opt, 'PCN', phase)
        

class ComPCAtlasDataset(Dataset):
    def __init__(self, opt, dataset_name):
        self.opt = opt
        self.dataset_name = dataset_name
        self.dataset_folder = self.opt[self.dataset_name]['dataset_dir']
        self.data = []

        self.load_data()
    
    def load_data(self):
        subdirs = os.listdir(self.dataset_folder)
        for subdir in subdirs:
            if os.path.isdir(os.path.join(self.dataset_folder, subdir)):
                object_name = subdir
                subdir_path = os.path.join(self.dataset_folder, subdir)
                input_file = os.path.join(subdir_path, 'plain_ot', f'{object_name}.pt')
                data_sample = {
                    'input': input_file,
                    'gt': os.path.join(subdir_path, 'gt_centered.ply'),
                    'name': object_name
                }
                self.data.append(data_sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_file = item['input']

        input_atlas = torch.load(input_file)
        
        input_xyz = input_atlas['xyz']
        input_normal = input_atlas['normal']
        input_mask = input_atlas['mask']
        
        # concatenate xyz and normal
        input_xyz_normal = torch.cat((input_xyz, input_normal), dim=-1)

        return {
            # 'gt': item['gt'],
            'input': input_xyz_normal,
            'input_mask': input_mask,
            'name': item['name'],
            'dataset_name': self.dataset_name
        }


class StressTestAtlasDataset(Dataset):
    def __init__(self, opt, dataset_name):
        self.opt = opt
        self.dataset_name = dataset_name
        self.dataset_folder = self.opt[self.dataset_name]['dataset_dir']
        self.data = []

        self.load_data()
    
    def load_data(self):
        subdirs = os.listdir(self.dataset_folder)
        for subdir in subdirs:
            if os.path.isdir(os.path.join(self.dataset_folder, subdir)):
                object_name = subdir
                subdir_path = os.path.join(self.dataset_folder, subdir)
                input_file = os.path.join(subdir_path, 'plain_ot', f'{object_name}.pt')
                data_sample = {
                    'input': input_file,
                    'gt': os.path.join(subdir_path, 'gt_centered.ply'),
                    'name': object_name
                }
                self.data.append(data_sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_file = item['input']

        input_atlas = torch.load(input_file)
        
        input_xyz = input_atlas['xyz']
        input_normal = input_atlas['normal']
        input_mask = input_atlas['mask']
        
        # concatenate xyz and normal
        input_xyz_normal = torch.cat((input_xyz, input_normal), dim=-1)

        return {
            # 'gt': item['gt'],
            'input': input_xyz_normal,
            'input_mask': input_mask,
            'name': item['name'],
            'dataset_name': self.dataset_name
        }


def test_shapenet_dataset():
    cfg_file = 'config/shapenet_eval_config.yaml'
    with open(cfg_file, 'r') as f:
        opt = yaml.safe_load(f)
    shapenet_dataset = ShapeNetAtlasDataset(opt, phase=opt['phase'])
    print(len(shapenet_dataset))

    rand_idx = random.randint(0, len(shapenet_dataset) - 1)
    sample = shapenet_dataset[rand_idx]
    # sample = shapenet_dataset.get_item(rand_idx)
    print(sample['gt'].shape)
    print(sample['input'].shape)

    verts = sample['mesh_verts']
    faces = sample['mesh_faces']
    
    meshes = Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0))

    print(meshes)


def test_compc_dataset():
    cfg_file = 'config/eval_condition_unet_compc.yaml'
    with open(cfg_file, 'r') as f:
        opt = yaml.safe_load(f)
    dataset_name = opt['datasets'][0]
    compc_dataset = ComPCAtlasDataset(opt, dataset_name)
    print(len(compc_dataset))

    rand_idx = random.randint(0, len(compc_dataset) - 1)
    sample = compc_dataset[rand_idx]
    print(sample['input'].shape)
    print(sample['input_mask'].shape)
    print(sample['name'])

def test_stresstest_dataset():
    cfg_file = 'config/eval_condition_unet_stress.yaml'
    with open(cfg_file, 'r') as f:
        opt = yaml.safe_load(f)
    dataset_name = opt['datasets'][0]
    stress_dataset = StressTestAtlasDataset(opt, dataset_name)
    print(len(stress_dataset))

    rand_idx = random.randint(0, len(stress_dataset) - 1)
    sample = stress_dataset[rand_idx]
    print(sample['input'].shape)
    print(sample['input_mask'].shape)
    print(sample['name'])
    
    for sample in stress_dataset:
        print(sample['name'])
if __name__ == "__main__":
    test_stresstest_dataset()
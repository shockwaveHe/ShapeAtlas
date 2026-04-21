import numpy as np
import trimesh
import open3d as o3d
import os 

def mesh_sort(vertices_, faces_):
    assert (vertices_ <= 0.5).all() and (vertices_ >= -0.5).all()  # [-0.5, 0.5]
    vertices = (vertices_ + 0.5) * 128  # [0, num_tokens]
    vertices -= 0.5  # for evenly distributed, [-0.5, num_tokens -0.5] will be round to 0 or num_tokens (-1)
    vertices_quantized_ = np.clip(vertices.round(), 0, 128 - 1).astype(int)  # [0, num_tokens -1]

    cur_mesh = trimesh.Trimesh(vertices=vertices_quantized_, faces=faces_)

    cur_mesh.merge_vertices()
    cur_mesh.update_faces(cur_mesh.nondegenerate_faces())
    cur_mesh.update_faces(cur_mesh.unique_faces())
    cur_mesh.remove_unreferenced_vertices()

    sort_inds = np.lexsort(cur_mesh.vertices.T)
    vertices = cur_mesh.vertices[sort_inds]
    faces = [np.argsort(sort_inds)[f] for f in cur_mesh.faces]

    faces = [sorted(sub_arr) for sub_arr in faces]

    def sort_faces(face):
        return face[0], face[1], face[2]

    faces = sorted(faces, key=sort_faces)

    vertices = vertices / 128 - 0.5  # [0, num_tokens -1] to [-0.5, 0.5)  for computing

    return vertices, faces

def npz2ply(npz_path, ply_path):
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    data = np.load(npz_path)
    points = data['points']  # Nx3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(ply_path, pcd)

def pcd2ply(pcd_path, ply_path):
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    pcd = o3d.io.read_point_cloud(pcd_path)
    o3d.io.write_point_cloud(ply_path, pcd)

def ply2pcd(ply_path, pcd_path):
    os.makedirs(os.path.dirname(pcd_path), exist_ok=True)
    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.io.write_point_cloud(pcd_path, pcd)

def npy2ply(npy_path, ply_path):
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    points = np.load(npy_path)  # Nx3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(ply_path, pcd)
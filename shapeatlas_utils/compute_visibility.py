import math
import os
import pdb

import numpy as np
import nvdiffrast.torch as dr
import open3d as o3d
import torch
import trimesh


def generate_virtual_camera(
    vertices: torch.Tensor, image_size: int = 512, fov_deg: float = 60
):
    """
    Generate OpenCV-style camera intrinsics and extrinsics for rendering a 3D object.

    Args:
        vertices (torch.Tensor): [N, 3] tensor of 3D vertices.
        image_size (int): Image width and height in pixels (square image).
        fov_deg (float): Field of view in degrees.

    Returns:
        intrinsic (torch.Tensor): [3, 3] camera intrinsic matrix.
        extrinsic (torch.Tensor): [3, 4] camera extrinsic matrix [R | t].
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3, "Expected input shape [N, 3]"

    # Step 1: Compute object center
    center = vertices.mean(dim=0)  # [3]

    # Step 2: Define camera position (1 meter in front)
    cam_z_offset = 1.0
    camera_position = center + torch.tensor(
        [0.0, 0.0, cam_z_offset], device=vertices.device
    )

    # Step 3: Look-at rotation matrix
    def look_at(
        cam_pos, target, up=torch.tensor([0.0, 1.0, 0.0], device=vertices.device)
    ):
        forward = target - cam_pos
        forward = forward / forward.norm()
        right = torch.cross(up, forward)
        right = right / right.norm()
        true_up = torch.cross(forward, right)
        rot = torch.stack([right, true_up, forward], dim=1)  # [3,3]
        trans = -rot.T @ cam_pos  # [3]
        extrinsic = torch.eye(4, device=vertices.device)
        extrinsic[:3, :3] = rot.T
        extrinsic[:3, 3] = trans
        return extrinsic[:3, :]  # [3, 4]

    extrinsic = look_at(camera_position, center)

    # Step 4: Intrinsic matrix
    fov_rad = math.radians(fov_deg)
    focal_length = (image_size / 2) / math.tan(fov_rad / 2)
    cx = cy = image_size / 2

    intrinsic = torch.tensor(
        [[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=vertices.device,
    )

    return intrinsic, extrinsic


def world_to_clip_space(vertices, extr, P_clip, translate):
    """
    Convert world space vertices to clip space for a batch of inputs with multiple views.

    Args:
        vertices: A tensor of shape [batch_size, num_vertices, 3], representing world space vertices.
        extr: A tensor of shape [batch_size, num_views, 3, 4], representing the extrinsic matrices for each batch and view.
        P_clip: A tensor of shape [batch_size, num_views, 4, 4], representing the perspective projection matrices for each batch and view.
        translate: A tensor of shape [batch_size, num_views, 4, 4], representing the translation matrices for each batch and view.

    Returns:
        vertices_clip: A tensor of shape [batch_size, num_views, num_vertices, 4], containing the clip space vertices for each batch and view.
    """

    device = vertices.device
    # vertices.shape [batch_size, num_vertices, 3]
    batch_size, num_vertices = vertices.shape[0], vertices.shape[1]
    num_views = extr.shape[1]

    # Add a homogeneous coordinate (1) to the vertices
    ones = torch.ones((batch_size, num_vertices, 1), dtype=torch.float32, device=device)

    vertices_h = torch.cat(
        [vertices, ones], dim=-1
    )  # Shape: [batch_size, num_vertices, 4]

    # Append the [0, 0, 0, 1] row to the extrinsic matrix for each batch and view (to make it 4x4)
    ones_row = (
        torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device)
        .view(1, 1, 1, 4)
        .repeat(batch_size, num_views, 1, 1)
    )
    extr_4x4 = torch.cat(
        [extr, ones_row], dim=2
    )  # Shape: [batch_size, num_views, 4, 4]

    # Correction matrix for OpenGL (batch processing)
    correction_matrix = (
        torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_views, 1, 1)
    )  # Shape: [batch_size, num_views, 4, 4]

    # Apply correction to extrinsic matrices
    extr_corrected = torch.matmul(
        correction_matrix, extr_4x4
    )  # Shape: [batch_size, num_views, 4, 4]

    # Expand vertices to handle multiple views
    # vertices_h = vertices_h.unsqueeze(1).repeat(1, num_views, 1, 1)  # Shape: [batch_size, num_views, num_vertices, 4]

    # Transform vertices to camera space using the extrinsic matrix (for all views)
    vertices_cam = torch.matmul(
        vertices_h, extr_corrected.transpose(2, 3)
    )  # Shape: [batch_size, num_views, num_vertices, 4]

    # Combine the projection matrix and the translation matrix
    mtx = torch.matmul(P_clip, translate)  # Shape: [batch_size, num_views, 4, 4]

    # Transform vertices to clip space (for all views)
    vertices_clip = torch.matmul(
        vertices_cam, mtx.transpose(2, 3)
    )  # Shape: [batch_size, num_views, num_vertices, 4]

    # Perform perspective division
    vertices_clip = (
        vertices_clip / vertices_clip[:, :, :, 3:4]
    )  # Divide by w (homogeneous coordinate)

    return vertices_clip[:, :, :, :4]  # Return the x, y, z, w coordinates


def perspective_projection_opencv_to_opengl(fx, fy, cx, cy, near, far, width, height):
    device = fx.device
    batch_size, num_views = near.shape[0], near.shape[1]

    # Convert near and far to negative (as in OpenGL convention)
    near = -near
    far = -far

    # Compute field of view in radians for each batch element and view
    fovy_rad = 2 * torch.atan(height / (2 * fy))  # Shape: [batch_size, num_views]
    fovx_rad = 2 * torch.atan(width / (2 * fx))  # Shape: [batch_size, num_views]

    # Aspect ratio
    aspect = width / height  # Shape: [batch_size, num_views] #

    # Compute cotangent of half the vertical field of view #
    cot_half_fov = 1 / torch.tan(fovy_rad / 2)  # Shape: [batch_size, num_views]

    # Create the perspective projection matrix P_clip for each batch and view
    P_clip = torch.zeros(
        (batch_size, num_views, 4, 4), dtype=torch.float32, device=device
    )
    # NVDiffrast author's official link to the perspective projection
    # search "the perspective projection"
    # https://www3.ntu.edu.sg/home/ehchua/programming/opengl/CG_BasicsTheory.html
    #
    P_clip[:, :, 0, 0] = cot_half_fov / aspect
    P_clip[:, :, 1, 1] = cot_half_fov
    P_clip[:, :, 2, 2] = (far) / (near - far)  # depth scaling
    P_clip[:, :, 2, 3] = (far * near) / (near - far)  # depth translation
    P_clip[:, :, 3, 2] = -1  # Fixed for perspective projection
    # Compute the principal point translation in OpenGL coordinates
    c_y_flipped = height - cy  # Flip y-axis, Shape: [batch_size, num_views]
    p_x = cx - width / 2  # X translation, Shape: [batch_size, num_views]
    p_y = c_y_flipped - height / 2  # Y translation, Shape: [batch_size, num_views]

    # Scaling factor (adjust if necessary) ##
    scale = 0.001  # 2.0 (don't use 2 anymore) # Scaling applied uniformly

    # Create the translation matrix for each batch and view
    translate = (
        torch.eye(4, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_views, 1, 1)
    )  # Shape: [batch_size, num_views, 4, 4]

    # Apply translation to the x and y coordinates #
    translate[:, :, 0, 3] = -scale * p_x  # Apply translation in x direction
    translate[:, :, 1, 3] = scale * 2.0 * p_y  # Apply translation in y direction

    return P_clip, translate


def compute_near_far(vertices, extr):
    """#
    Compute near and far values for a batch of vertices and multiple views (extrinsic matrices).

    Args:
        vertices: A tensor of shape [batch, num_vertices, 3], representing the vertex positions.
        extr: A tensor of shape [batch, num_views, 3, 4], representing the extrinsic matrices for each batch and view.

    Returns:
        near: A tensor of shape [batch, num_views], representing the near values for each batch and view.
        far: A tensor of shape [batch, num_views], representing the far values for each batch and view.
    """
    batch_size = vertices.shape[0]
    num_vertices = vertices.shape[1]
    num_views = extr.shape[1]

    # Add a homogeneous coordinate (1) to the vertices
    # ones = torch.ones((batch_size, num_vertices, 1), dtype=torch.float32, device=vertices.device)
    ones = torch.ones(
        (batch_size, num_vertices, 1), dtype=torch.float32, device=vertices.device
    )
    vertices_h = torch.cat([vertices, ones], dim=-1)  # Shape: [batch, num_vertices, 4]

    # Append the [0, 0, 0, 1] row to the extrinsic matrix for each batch and view
    ones_row = (
        torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=vertices.device)
        .view(1, 1, 1, 4)
        .repeat(batch_size, num_views, 1, 1)
    )

    extr_4x4 = torch.cat([extr, ones_row], dim=2)  # Shape: [batch, num_views, 4, 4]

    # Create a correction matrix for OpenGL transformation and repeat it for each view
    correction_matrix = (
        torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
            device=vertices.device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, num_views, 1, 1)
    )  # Shape: [batch, num_views, 4, 4]

    # Apply correction to the extrinsic matrices
    extr_opengl = torch.matmul(
        correction_matrix, extr_4x4
    )  # Shape: [batch, num_views, 4, 4]

    # Transform vertices into camera space using the extrinsic matrix
    # We need to reshape vertices_h to match the shape required for batched matrix multiplication
    # vertices_h = vertices_h.unsqueeze(1).repeat(1, num_views, 1, 1)  # Shape: [batch, num_views, num_vertices, 4]
    vertices_cam = torch.matmul(
        vertices_h, extr_opengl.transpose(2, 3)
    )  # Shape: [batch, num_views, num_vertices, 4]
    # Extract the z-values (depth) in camera space for each batch and view
    z_vals = vertices_cam[:, :, :, 2]  # Shape: [batch, num_views, num_vertices]

    # Compute near and far for each batch and view
    near = torch.max(z_vals, dim=2)[
        0
    ]  # Near plane is the maximum z-value (farthest from the camera)
    far = torch.min(z_vals, dim=2)[
        0
    ]  # Far plane is the minimum z-value (closest to the camera)

    return near, far


def load_obj(file_path):
    # from barycentric_sample_density.py
    vertices = []
    faces = []
    uvs = []

    with open(file_path, "r") as obj_file:
        for line in obj_file:
            tokens = line.split()
            if not tokens:
                continue

            if tokens[0] == "v":
                # Vertex coordinates
                x, y, z = map(float, tokens[1:4])
                vertices.append((x, y, z))
            elif tokens[0] == "vt":
                # UV texture coordinates
                u, v = map(float, tokens[1:3])
                uvs.append((u, v))
            elif tokens[0] == "f":
                # Face information (vertex indices and UV indices)
                face = []
                for token in tokens[1:]:
                    vertex_info = token.split("/")
                    vertex_index = int(vertex_info[0]) - 1  # OBJ uses 1-based indexing
                    uv_index = int(vertex_info[1]) - 1 if len(vertex_info) > 1 else None
                    face.append((vertex_index, uv_index))
                faces.append(face)

    return vertices, faces, uvs


def find_3d_vertices_for_uv(faces):
    uv_to_vertices = {}

    vertices = []
    uvs = []

    for face in faces:
        for idx in range(3):
            vertex_index, uv_index = face[idx]

            if uv_index in uv_to_vertices:
                uv_to_vertices[uv_index].add(vertex_index)
            else:
                uv_to_vertices[uv_index] = set()
                uv_to_vertices[uv_index].add(vertex_index)

    return uv_to_vertices


def load_uv_space_rasterization_data(obj_path):
    smplx_uv_fp = os.path.join(obj_path)
    canonical_vertices, canonical_faces, canonical_uvs = load_obj(smplx_uv_fp)

    canonical_vertices = np.array(canonical_vertices, dtype=np.float32)  # (10475, 3)
    canonical_faces = np.array(canonical_faces, dtype=np.int32)  # (20908, 3, 2)
    canonical_uvs = np.array(canonical_uvs, dtype=np.float32)  # (11313, 2)

    uv_pts_mapping = find_3d_vertices_for_uv(canonical_faces)
    # Ensure that the UV indices are processed in order (0 -> 1 -> 2, etc.)
    sorted_uv_indices = sorted(uv_pts_mapping.keys())
    # Extract the vertex indices corresponding to the sorted UV indices

    uv_to_vertex_indices = [
        uv_pts_mapping[uv_idx].pop() for uv_idx in sorted_uv_indices
    ]
    uv_to_vertex_indices = torch.tensor(uv_to_vertex_indices, dtype=torch.long).to(
        device="cuda"
    )

    canonical_vertices = torch.from_numpy(canonical_vertices).to(
        dtype=torch.float32, device="cuda"
    )
    canonical_faces = torch.from_numpy(canonical_faces).to(
        dtype=torch.int32, device="cuda"
    )
    canonical_uvs = torch.from_numpy(canonical_uvs).to(
        dtype=torch.float32, device="cuda"
    )

    rast_uv_space = {
        "canonical_vertices": canonical_vertices,
        "canonical_faces": canonical_faces,
        "canonical_uvs": canonical_uvs,
        "uv_to_vertex_indices": uv_to_vertex_indices,
    }

    return rast_uv_space  #


def compute_visibility(
    vertices, intr, extr, rast_uv_space_dict, image_width=512, image_height=512
):
    glctx = rast_uv_space_dict["glctx"]
    tri_verts = rast_uv_space_dict["tri_verts"]
    # B = batch size, N = number of frames, V = number of vertices, _ = 3D coordinates
    B, V, _ = vertices.shape

    # visibility map

    # intr = data[view_mode]['intr']  # [B=4, N=3, 3, 3]
    # extr = data[view_mode]['extr']  # [B=4, N=3, 3, 4]

    B, N, _, _ = intr.shape

    # intr = intr.reshape(-1,3,3)
    # extr = extr.reshape(-1,3,4)
    # near [batch_size, num_view], far [batch_size, num_view]
    near, far = compute_near_far(vertices, extr)
    f_x = intr[:, :, 0, 0]  # Shape: [batch_size, num_views]
    f_y = intr[:, :, 1, 1]  # Shape: [batch_size, num_views]
    c_x = intr[:, :, 0, 2]  # Shape: [batch_size, num_views]
    c_y = intr[:, :, 1, 2]  # Shape: [batch_size, num_views]

    P_clip, translate = perspective_projection_opencv_to_opengl(
        f_x, f_y, c_x, c_y, near, far, image_width, image_height
    )

    pos_clip = world_to_clip_space(vertices, extr, P_clip, translate)

    # pos_clip = pos_clip.unsqueeze(0)  # [1, 10475, 4]

    # tri = torch.from_numpy(self.faces[..., 0]).to(dtype=torch.int32, device='cuda')
    # Reshape pos_clip from [batch_size, num_views, num_vertices, 4] to [batch_size * num_views, num_vertices, 4]
    batch_size, num_views, num_vertices, _ = pos_clip.shape
    pos_clip_flattened = pos_clip.view(batch_size * num_views, num_vertices, 4)

    #  resolution for the visibility computation should be high.
    # resolution = [self.image_width,self.image_height]
    # Rasterize with the flattened pos_clip
    # rast_image_space [2, H=512, W=512, 4]

    # originally, the resolution was [self.image_width, self.image_height]
    # but we don't have to render a high-resolution image to compute the visibility
    rast_image_space, _ = dr.rasterize(
        glctx,
        pos_clip_flattened.contiguous(),
        tri_verts.squeeze(0),
        resolution=[512, 512],
    )

    batch_size_views = rast_image_space.shape[0]  # batch_size * num_views
    num_vertices = vertices.shape[1]  # Number of vertices

    B, N = (
        batch_size * num_views,
        num_vertices,
    )  # B = batch_size * num_views, N = num_vertices

    # 1. Extract face (triangle) IDs from rasterization output
    #    rast_image_space[..., 3] gives the face ID at each pixel
    face_id = rast_image_space[..., 3].long() - 1  # Shape: [B, height, width]
    # valid face id starts from 1 so extract 1 (0 is background id)

    """
    # for the visualization 
    import imageio
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # Step 1: Extract per-pixel face ID from rasterization result
    face_id_map = rast_image_space[..., 3][0]  # [H, W], float32
    face_id_map = face_id_map.long().cpu() - 1  # convert to int, subtract 1 (background becomes -1)

    # Step 2: Replace background (-1) with 0 for safe colormap indexing
    face_id_map[face_id_map < 0] = 0

    # Step 3: Normalize face IDs to [0, 1] range
    num_faces = 632  # or tri_verts.shape[0]
    face_id_norm = face_id_map.float() / num_faces  # [H, W], float in [0, ~1]

    # Step 4: Apply matplotlib colormap (e.g., 'jet')
    colormap = cm.get_cmap('jet')  # can use 'viridis', 'turbo', 'plasma', etc.
    face_id_colored = colormap(face_id_norm.numpy())  # [H, W, 4] RGBA

    # Step 5: Convert to RGB uint8 image
    face_id_rgb = (face_id_colored[..., :3] * 255).astype( np.uint8)  # Drop alpha channel

    # Step 5: Save image
    imageio.imwrite("/scr/youngjo2/data/face_id_visualization.png", face_id_rgb)
    """

    # 2. Initialize face visibility map (binary): Shape [batch_size * num_views, num_faces]
    num_faces = tri_verts.squeeze(0).shape[0]
    face_visibility_map = torch.zeros((B, num_faces), device=vertices.device)

    # 3. Scatter face visibility across the rasterized image
    #    Each pixel has a face ID; we mark that face as visible
    valid_faces_mask = (
        face_id >= 0
    )  # Mask out background (invalid) pixels, face_id -1 is background
    # Use scatter_add_ to update face visibility where valid face IDs exist
    # We convert face IDs to absolute indices for each batch
    face_visibility_map.scatter_add_(
        1, face_id.clamp(min=0).view(B, -1), valid_faces_mask.view(B, -1).float()
    )
    # face_visibility_map.shape [batch_size * num_views = 4*3 = 12, num_faces = 20908]
    # Convert face_visibility_map to binary (1 if visible, 0 if not)
    face_visibility_map = face_visibility_map > 0

    face_visibility_map = face_visibility_map.float()

    visible_face_ids_per_view = [
        torch.nonzero(vis_row, as_tuple=False).squeeze(1)
        for vis_row in face_visibility_map
    ]

    """
    # 4. Flatten tri_verts to get face-to-vertex mapping
    packed_faces = tri_verts.view(-1)  # Shape: [num_faces * 3] # [20908 * 3 = 62724]

    # 5. Initialize vertex visibility map (binary): Shape [batch_size * num_views, num_vertices] [4 * 3, 10475]
    vertex_visibility_map = torch.zeros((B, N),device=vertices.device)

    # 6. Map visible faces to their corresponding vertices
    # Instead of extracting dynamic indices, we use fixed-size face-to-vertex mapping
    # We use the face_visibility_map to gather the visible faces and their corresponding vertices
    face_visibility_map_expanded = face_visibility_map.unsqueeze(
        -1).expand(B, num_faces, 3)  # Shape: [B, num_faces, 3]

    # This expands the mapping so that each face's three vertices can be indexed
    face_visibility_map_expanded = face_visibility_map_expanded.reshape(B, -1)  # Shape: [B, num_faces * 3]

    # Gather the vertex indices corresponding to the visible faces
    visible_verts_idx = packed_faces.unsqueeze(0).repeat(B,
                                                            1)  # Shape: [B, num_faces * 3]

    # Mask out invisible faces
    visible_verts_idx = visible_verts_idx * face_visibility_map_expanded.long()

    # 7. Scatter vertex visibility based on the visible vertices
    # Shape: [batch_size * num_views, num_vertices]
    vertex_visibility_map.scatter_add_(1, visible_verts_idx,
                                        torch.ones_like(
                                            visible_verts_idx,
                                            device=vertices.device).float())

    vertex_visibility_map = (vertex_visibility_map > 0).float()
    vertex_visibility_map = vertex_visibility_map[
        ..., None]  # Shape: [batch_size * num_views, num_vertices, 1] # visibility per vertex

    return vertex_visibility_map
    """
    return visible_face_ids_per_view


def compute_depthmap(
    vertices, intr, extr, rast_uv_space_dict, image_width=512, image_height=512
):
    glctx = rast_uv_space_dict["glctx"]
    tri_verts = rast_uv_space_dict["tri_verts"]

    B, V, _ = vertices.shape
    B, N, _, _ = intr.shape

    near, far = compute_near_far(vertices, extr)
    f_x, f_y = intr[:, :, 0, 0], intr[:, :, 1, 1]
    c_x, c_y = intr[:, :, 0, 2], intr[:, :, 1, 2]

    P_clip, translate = perspective_projection_opencv_to_opengl(
        f_x, f_y, c_x, c_y, near, far, image_width, image_height
    )

    pos_clip = world_to_clip_space(vertices, extr, P_clip, translate)

    batch_size, num_views, num_vertices, _ = pos_clip.shape
    pos_clip_flattened = pos_clip.view(batch_size * num_views, num_vertices, 4)

    # Rasterize
    rast, _ = dr.rasterize(
        glctx,
        pos_clip_flattened.contiguous(),
        tri_verts.squeeze(0),
        resolution=[image_width, image_height],
    )
    Rs = extr[:, :, :3, :3]
    Rs_T = Rs.transpose(2, 3)
    Ts = extr[:, :, :3, 3:]
    Ts_t = -torch.matmul(Rs_T, Ts)

    Rs_T = Rs_T.squeeze(0)
    Ts_t = Ts_t.squeeze(0)
    import pdb; pdb.set_trace()
    vertices_cam = torch.matmul(Rs_T.unsqueeze(1), vertices.unsqueeze(-1)) + Ts_t.unsqueeze(1)

    depth_values = vertices_cam[:, :, 2]  # [B, N, V]
    
    out, _ = dr.interpolate(depth_values.contiguous(), rast, tri_verts.squeeze(0))
    # import pdb; pdb.set_trace()
    # covert out to depth map
    
    # depth_image = out.permute(0,3,1,2)
    triangle_index = rast[..., 3]
    forground_mask = triangle_index >= 0
    forground_mask = forground_mask.unsqueeze(-1)
    depth_image = out * forground_mask.float()
    return depth_image


def main(obj_path, intr_path, extr_path, image_height, image_width):  #
    """
    Compute visibility from a loaded mesh.
    """
    rast_uv_space_dict = load_uv_space_rasterization_data(obj_path)
    glctx = dr.RasterizeCudaContext()  # Create the OpenGL context for nvdiffrast

    # Rasterize UV space
    print("Pre-heating the NVDiffrast with UV space rasterization")
    vertices = rast_uv_space_dict["canonical_vertices"]
    uvs = rast_uv_space_dict["canonical_uvs"]
    faces = rast_uv_space_dict["canonical_faces"]

    with torch.no_grad():
        pos = uvs
        pos = 2 * pos - 1

        final_pos = torch.stack(
            [
                pos[..., 0],
                pos[..., 1],
                torch.zeros_like(pos[..., 0]),
                torch.ones_like(pos[..., 0]),
            ],
            dim=-1,
        )
        pos_uv = final_pos.reshape((1, -1, 4)).contiguous()
        tri_uv = faces[..., 1].contiguous()
        tri_verts = faces[..., 0].contiguous()  #
        # [1, 1024, 1024, 4]
        rast_uv_space, _ = dr.rasterize(glctx, pos_uv, tri_uv, resolution=[512, 512])
        face_id_raw = rast_uv_space[..., 3:]
    face_id = face_id_raw[0]

    # uv_to_vertex_indices = rast_uv_space_dict['uv_to_vertex_indices']

    rast_uv_space_dict["tri_verts"] = tri_verts
    rast_uv_space_dict["tri_uv"] = tri_uv
    rast_uv_space_dict["glctx"] = glctx
    # TODO Please load intrinsic and extrinsic matrix and convert it to tensor
    intr, extr = generate_virtual_camera(vertices)

    # vertices.shape [N,3]
    # intr.shape [3,3]
    # extr.shape [3,3]

    # reshape the vertices from [N,3] to [B=1,#Frame=1,N,3]
    vertices = vertices[None, None]
    # reshape intr from [3,3] to [B=1,#Frame=1,3,3]
    intr = intr[None, None]
    # reshape extr from [3,3] to [B=1,#Frame=,3,3]
    extr = extr[None, None]

    visible_face_ids_per_view = compute_visibility(
        vertices, intr, extr, rast_uv_space_dict
    )


def shapenet_visibility(trimesh, intr, extr):
    """
    Compute visibility for a given trimesh object.
    This function is a placeholder and should be implemented based on the specific requirements.
    """
    # Convert trimesh to vertices and faces
    vertices = torch.tensor(trimesh.vertices, dtype=torch.float32).cuda()
    faces = torch.tensor(trimesh.faces, dtype=torch.int32).cuda()

    vertices = vertices[None, None]  # Add batch and frame dimensions
    faces = faces[None, None]  # Add batch and frame dimensions
    # Generate virtual camera intrinsics and extrinsics
    # intr, extr = generate_virtual_camera(vertices)
    rast_space_dict = {}
    glctx = dr.RasterizeCudaContext()  # Create the OpenGL context for nvdiffrast
    rast_space_dict["tri_verts"] = faces
    rast_space_dict["glctx"] = glctx

    visibile_faces = compute_visibility(vertices, intr, extr, rast_space_dict)

    return visibile_faces


if __name__ == "__main__":
    intr_path = ""
    extr_path = ""
    image_height = 512
    image_width = 512
    obj_path = "tests/output/ego_view/model_normalized.obj"
    main(obj_path, intr_path, extr_path, image_height, image_width)

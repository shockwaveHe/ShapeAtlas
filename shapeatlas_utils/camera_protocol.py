import numpy as np
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os
from camera_visual import *
import torch
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple

def ExtractCameraCenter(Ts):
    return Ts[:,:3,3]

def ExtractViewVector(Ts):
    '''
    Return a vector that points to the view center from the camera center
    '''
    Rs = Ts[:,:3, :3]
    c_view_vectors = np.array([0, 0, 1])
    w_view_vectors = np.dot(Rs, c_view_vectors)
    return w_view_vectors 

def getSphereRepresentation(Ts):
    centers = ExtractCameraCenter(Ts)
    view_vectors = ExtractViewVector(Ts)
    
    nlines = centers.shape[0]
    # calculate view center: formulate as a linear system Ax=b
    # 1. build b
    bmat = np.reshape(centers, (-1, 1))
    # 2. build A
    A = np.zeros((3 * nlines, nlines + 3))
    for i in range(nlines):
        for j in range(3):
            A[3 * i + j, i] = -view_vectors[i][j]
            A[3 * i + j, nlines + j] = 1
    # calculate distance
    x = np.linalg.lstsq(A, bmat, rcond=None)
    view_center = x[0][-3:]
    view_center[0] = 0
    view_center[2] = 0
    dist_c2obj = np.max([np.linalg.norm(centers[i] - view_center.reshape(3,)) for i in range(nlines)])
    dist_c2origin = np.max([np.linalg.norm(centers[i]) for i in range(nlines)])
    elev = np.max([np.arctan2(centers[i][1], np.linalg.norm(np.array([centers[i][0], .0, centers[i][2]]))) for i in range(nlines)])
    return dist_c2obj, dist_c2origin, elev, view_center, x


def get_camera_extrinsics(elevation, azimuth, target=[0, 0, 0], dist=1):
    '''
    standard camera conversion from spherical representation
    following pytorch3d
    '''
    # looks like this definition is different from pytorch3d
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)
    # Convert elevation and azimuth angles to Cartesian coordinates on a unit sphere Yao: modified to be not a unit sphere
    x = dist * np.cos(elevation) * np.sin(azimuth)
    y = dist * np.sin(elevation)
    z = dist * np.cos(elevation) * np.cos(azimuth)
    
    # Calculate camera position, target, and up vectors
    camera_pos = np.array([x, y, z]) # world frame
    # target = np.array([0, 0, 0])
    target = np.array(target)
    up = np.array([0, 1, 0])
    
    # Construct view matrix
    forward = target - camera_pos
    forward /= np.linalg.norm(forward) # z
    right = np.cross(up, forward) 
    right /= np.linalg.norm(right) # x
    new_up = np.cross(forward, right) # y
    new_up /= np.linalg.norm(new_up) # y
    cam2world = np.eye(4)
    cam2world[:3, 0] = right
    cam2world[:3, 1] = new_up
    cam2world[:3, 2] = forward
    cam2world[:3, 3] = camera_pos
    return cam2world

def init_camera_extrinsics(num_views, num_elev, dist = 1.0, look_at_center = [0, 0, 0]):
    '''
    The camera protocal follows the one from 4d_dress but in the magicman conversion
    '''
    clip_interval = 360 // num_views
    elev_interval = 120 // (num_elev - 1) if num_elev > 1 else 0
    azim_list = []
    elev_list = []
    camera_list = []
    for i in range(num_views):
        azim = float(i*clip_interval)
        for j in range(num_elev):
            # range from -60 to 60
            elev = float(j*elev_interval) - 60.0 if num_elev > 1 else 0.0
            # get camera
            azim_list.append(azim)
            elev_list.append(elev)
    print("azim_list:", azim_list)
    print("elev_list:", elev_list)
    for azim, elev in zip(azim_list, elev_list):
        camera = get_camera_extrinsics(elev, azim, look_at_center, dist)
        camera_list.append(camera)
    cameras = np.stack(camera_list, axis=0) # (f, 4, 4)
    return azim_list, elev_list, cameras


# code from pytorch3d

def look_at_rotation(
    camera_position, at=((0, 0, 0),), up=((0, 1, 0),), device: str = "cpu"
) -> torch.Tensor:
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.

    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

    Args:
        camera_position: position of the camera in world coordinates
        at: position of the object in world coordinates
        up: vector specifying the up direction in the world coordinate frame.

    The inputs camera_position, at and up can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)

    The vectors are broadcast against each other so they all have shape (N, 3).

    Returns:
        R: (N, 3, 3) batched rotation matrices
    """
    # Format input and broadcast
    # broadcasted_args = convert_to_tensors_and_broadcast(
    #     camera_position, at, up, device=device
    # )
    # camera_position, at, up = broadcasted_args
    up_tensor = torch.tensor(up, device=device, dtype=camera_position.dtype)
    up_tensor = up_tensor.expand(camera_position.shape[0], -1)
    up_tensor.reshape(camera_position.shape)
    for t, n in zip([camera_position, at, up_tensor], ["camera_position", "at", "up"]):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
    z_axis = F.normalize(at - camera_position, eps=1e-5)

    x_axis = F.normalize(torch.cross(up_tensor, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)

def camera_position_from_spherical_angles(
    distance, elevation, azimuth, degrees: bool = True, device: str = "cpu"
) -> torch.Tensor:
    """
    Calculate the location of the camera based on the distance away from
    the target point, the elevation and azimuth angles.

    Args:
        distance: distance of the camera from the object.
        elevation, azimuth: angles.
            The inputs distance, elevation and azimuth can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N) or (1)
        degrees: bool, whether the angles are specified in degrees or radians.
        device: str or torch.device, device for new tensors to be placed on.

    The vectors are broadcast against each other so they all have shape (N, 1).

    Returns:
        camera_position: (N, 3) xyz location of the camera.
    """
    # broadcasted_args = convert_to_tensors_and_broadcast(
    #     distance, elevation, azimuth, device=device
    # )
    # dist, elev, azim = broadcasted_args
    dist, elev, azim = distance, elevation, azimuth
    if degrees:
        elev = math.pi / 180.0 * elev
        azim = math.pi / 180.0 * azim
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    camera_position = torch.stack([x, y, z], dim=1)
    if camera_position.dim() == 0:
        camera_position = camera_position.view(1, -1)  # add batch dim.
    return camera_position.view(-1, 3)

def look_at_view_transform(
    dist=1.0,
    elev=0.0,
    azim=0.0,
    degrees: bool = True,
    eye: Optional[Sequence] = None,
    at=((0, 0, 0),),  # (1, 3)
    up=((0, 1, 0),),  # (1, 3)
    device="cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function returns a rotation and translation matrix
    to apply the 'Look At' transformation from world -> view coordinates [0].

    Args:
        dist: distance of the camera from the object
        elev: angle in degres or radians. This is the angle between the
            vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
        azim: angle in degrees or radians. The vector from the object to
            the camera is projected onto a horizontal plane y = 0.
            azim is the angle between the projected vector and a
            reference vector at (1, 0, 0) on the reference plane (the horizontal plane).
        dist, elem and azim can be of shape (1), (N).
        degrees: boolean flag to indicate if the elevation and azimuth
            angles are specified in degrees or radians.
        eye: the position of the camera(s) in world coordinates. If eye is not
            None, it will overide the camera position derived from dist, elev, azim.
        up: the direction of the x axis in the world coordinate system.
        at: the position of the object(s) in world coordinates.
        eye, up and at can be of shape (1, 3) or (N, 3).

    Returns:
        2-element tuple containing

        - **R**: the rotation to apply to the points to align with the camera.
        - **T**: the translation to apply to the points to align with the camera.

    References:
    [0] https://www.scratchapixel.com
    """

    if eye is not None:
        # broadcasted_args = convert_to_tensors_and_broadcast(eye, at, up, device=device)
        # eye, at, up = broadcasted_args
        C = eye
    else:
        # broadcasted_args = convert_to_tensors_and_broadcast(
        #     dist, elev, azim, at, up, device=device
        # )
        # dist, elev, azim, at, up = broadcasted_args
        C = camera_position_from_spherical_angles(
            dist, elev, azim, degrees=degrees, device=device
        )
    
    R = look_at_rotation(C, at, up, device=device)
    T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]
    return R, T

################################################################

def estimate_sphere_exp_main():
    # load the numpy files
    root_dir = '/home/yao/Camera_visualization/4d_dress/parm/'

    sequences = '00122'

    # takes = [2,4,5,7,8]
    takes = [2]
    # takes = [2, 3, 4, 5, 6, 8, 9]
    # exts = ['0', '1']
    exts = ['0']
    cameras = []

    extrinsics = []
    for take in takes:
        for i in range(4):
            path = os.path.join(root_dir, f'{sequences}_Inner_Take{take}_{str(i).zfill(3)}')
            for ext in exts:
                ext_file = f'{ext}_extrinsic.npy'
                abs_path = os.path.join(path, ext_file)
                print(abs_path)
                data = np.load(abs_path)
                rot = data[:3,:3]
                trans = data[:3,3]
                inv_transform = np.eye(4)
                inv_transform[:3,:3] = rot.T
                inv_transform[:3,3] = -np.dot(rot.T,trans)
                extrinsics.append(inv_transform)

    exts = np.array(extrinsics)
    # import pdb; pdb.set_trace()
    dist_c2obj, dist_c2origin, elev, view_center, x = getSphereRepresentation(exts) # least square fitting
    print("the sphere distance to object is:", dist_c2obj) # 2.8920653553258524
    print("the sphere distance to origin is:", dist_c2origin) # 3.3395688240055774
    print("the elevation is:", np.degrees(elev)) # 32.91761924270414 degree
    print("The viewing center is:", view_center) #The viewing center is: [[0.        ][1.15449313][0.        ]]


if __name__ == "__main__":
    visualized_cameras = []
    azim_list, elev_list, cameras = init_camera_extrinsics(4, 4)
    for cam in cameras:
        visualized_cameras.append(camera(cam[:3,3], cam[:3,:3], 'purple'))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ar1 = [0, 1]
    ar2 = [0, 0]

    ax.set_xticks(np.arange(-2, 2, 1))
    ax.set_yticks(np.arange(-2, 2, 1))
    ax.set_zticks(np.arange(-2, 2, 1))
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    plt.plot(ar1, ar2, ar2, color="blue")
    plt.plot(ar2, ar1, ar2, color="green")
    plt.plot(ar2, ar2, ar1, color="red")
    ax.scatter([0], [0], [0], marker="o", color="black", s=20)
    ax.scatter([1], [0], [0], marker=">", color="blue", s=20)
    ax.scatter([0], [1], [0], marker=">", color="green", s=20)
    ax.scatter([0], [0], [1], marker=">", color="red", s=20)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for cam in visualized_cameras:
        cam.plot_camera(ax)
        cam.plot_axis(ax)

    plt.show()
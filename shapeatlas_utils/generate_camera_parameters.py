import numpy as np
import torch
import random
import math
from pytorch3d.renderer import cameras as pt3d_cameras
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)

def camera_setup():

    seed_everything(42)
    image_size = 512
    fov_deg = 120
    num_views = 4
    num_elev = 4
    print("Calculating camera intrinsics...")
    fov_rad = math.radians(fov_deg)
    focal_length = (image_size / 2) / math.tan(fov_rad / 2)
    cx = cy = image_size / 2
    intrinsic = torch.tensor(
        [[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    intr = intrinsic[None]  # Add batch dimensions
    # generate the visibility camera parameters
    num_views = num_views
    num_elev = num_elev

    print(f"Generating {num_views} views with {num_elev} elevations each...")
    clip_interval = 360 // num_views
    elev_interval = 120 // (num_elev - 1) if num_elev > 1 else 0

    azim_offset = 45
    azim_list = []
    elev_list = []
    look_at_center_list = []
    look_at_center_default = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    for i in range(num_views):
        azim = (
            float(i * clip_interval) + azim_offset + np.random.uniform(-15, 15)
        )
        for j in range(num_elev):
            # range from -120 / 2 to 120 / 2
            elev = float(j * elev_interval) - 120.0 / 2 if num_elev > 1 else 0.0
            azim_list.append(azim)
            elev_list.append(elev)
            # randomly offset the look at center between -0.1 and 0.1
            look_at_center_offset = torch.tensor(
                [
                    np.random.uniform(-0.1, 0.1),
                    np.random.uniform(-0.1, 0.1),
                    np.random.uniform(-0.1, 0.1),
                ],
                dtype=torch.float32,
            )
            look_at_center = look_at_center_default + look_at_center_offset
            look_at_center_list.append(look_at_center)
    azim_list = torch.tensor(azim_list, dtype=torch.float32)
    elev_list = torch.tensor(elev_list, dtype=torch.float32)
    look_at_center_list = torch.stack(look_at_center_list, dim=0)

    # use pytorch3d to generate the camera extrinsics
    Rs, Ts = pt3d_cameras.look_at_view_transform(
        dist=1.0, elev=elev_list, azim=azim_list, at=look_at_center_list
    )
    # stack the camera extrinsics
    extrs = torch.cat(
        [Rs, Ts[..., None]], dim=-1
    )  # shape: (num_views * num_elev, 3, 4)
    extrs = extrs[None]  # Add batch and frame dimensions
    N = extrs.shape[1]  # Number of views
    # add frame dimension to intrinsics so it is compatible with extrinsics (B, 3, 3) -> (B, F, 3, 3)
    intrs = intr[None].repeat(1, N, 1, 1)

    return intrs, extrs

if __name__ == "__main__":
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'camera_parameters')
    intrs, extrs = camera_setup()
    # save intrs and extrs to torch files
    torch.save(intrs, f"{save_dir}/camera_intrinsics.pt")
    torch.save(extrs, f"{save_dir}/camera_extrinsics.pt")

    # save intrs and extrs to numpy files
    np.save(f"{save_dir}/camera_intrinsics.npy", intrs.cpu().numpy())
    np.save(f"{save_dir}/camera_extrinsics.npy", extrs.cpu().numpy())
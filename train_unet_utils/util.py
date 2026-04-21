import gc
import os
import os.path as osp
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import tqdm
from accelerate.logging import get_logger
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, ImageOps
from pytorch3d.loss import chamfer_distance
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from models.mutual_self_attention_2d import ReferenceAttentionControl

logger = get_logger(__name__, log_level="INFO")


N_SQRT = 128
N = N_SQRT * N_SQRT


def init_unit_sphere_grid(N=N):
    # fibonacci_sphere
    indices = np.arange(0, N)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle ~2.399963
    y = 1 - (indices / (N - 1)) * 2  # y goes from 1 to -1
    radius = np.sqrt(1 - y * y)

    theta = phi * indices
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    xyz = np.stack((x, z, y), axis=1)
    # note np.lexsort sort by the last key first in descending order
    sorted_indices = np.lexsort(
        (-xyz[:, 0], -xyz[:, 1], -xyz[:, 2])
    )  # sort by z, y, x -> in xyz[x, y, z]  = [x, y, z]

    xyz = xyz[sorted_indices]
    return torch.tensor(xyz, dtype=torch.float, device="cpu")


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


def load_weights(
    gaussian_unet,
    reference_unet,
    ckpt_dir,
    ckpt_step="latest",
):
    if ckpt_step == "latest":
        ckpt_files = sorted(
            os.listdir(ckpt_dir), key=lambda x: int(x.split("-")[-1].split(".")[0])
        )
        latest_pth_name = (Path(ckpt_dir) / ckpt_files[-1]).stem
        ckpt_step = int(latest_pth_name.split("-")[-1])

    gaussian_unet.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, f"gaussian_unet-{ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )

    reference_unet.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, f"reference_unet-{ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )

    logger.info(f"Resume models from {ckpt_dir}, step={ckpt_step}")


def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def padding_pil(img_pil, img_size):
    # resize a PIL image and zero padding the short edge
    W, H = img_pil.size
    resize_ratio = img_size / max(W, H)
    new_W, new_H = int(W * resize_ratio), int(H * resize_ratio)
    img_pil = img_pil.resize((new_W, new_H))

    left = (img_size - new_W) // 2
    right = img_size - new_W - left
    top = (img_size - new_H) // 2
    bottom = img_size - new_H - top

    padding_border = (left, top, right, bottom)
    img_pil = ImageOps.expand(img_pil, border=padding_border, fill=0)

    return img_pil


def concat_pil(img_pil_lst):
    # horizontally concat PIL images
    # NOTE(ZSH): assume all images are of same size
    W, H = img_pil_lst[0].size
    num_img = len(img_pil_lst)
    new_width = num_img * W
    new_image = Image.new("RGB", (new_width, H), color=0)
    for img_idx, img in enumerate(img_pil_lst):
        new_image.paste(img, (W * img_idx, 0))

    return new_image


def prepare_latents(
    batch_size,
    num_channels_latents,
    width,
    height,
    dtype,
    device,
    generator,
    init_noise_sigma,
    vae_scale_factor=8,
    latents=None,
):
    shape = (
        batch_size,
        num_channels_latents,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * init_noise_sigma
    return latents


def prepare_extra_step_kwargs(generator, eta, scheduler):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    import inspect

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(
        inspect.signature(scheduler.step).parameters.keys()
    )
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def decode_latents(vae, latents):
    from einops import rearrange

    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    # latents = latents.to(torch.float16)
    # video = self.vae.decode(latents).sample
    video = []
    for frame_idx in tqdm(range(latents.shape[0])):
        video.append(vae.decode(latents[frame_idx : frame_idx + 1]).sample)
    video = torch.cat(video)
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    video = video.cpu().float().numpy()
    return video


@torch.no_grad()
def log_validation(
    batch,
    val_image_prompt_embeds,
    # image_encoder,
    model,
    scheduler,
    accelerator,
    global_step,
    save_dir,
    # guidance_types,
    num_inference_steps=20,
    guidance_scale=3.5,
    seed=42,
    dtype=torch.float32,
    num_multi_view_supervision=1,
    save_gt=True,
    **kwargs,
):
    """
    Args:
        batch: Training batch with input and reference data.
        image_encoder: CLIP encoder for image embeddings.
        gaussian_unet: Generates gaussian parameter map.
        guidance_encoders: Dictionary of guidance encoders (e.g., normals, depth).
        scheduler: Diffusion scheduler.
        accelerator: Accelerator object for device management.
        global_step: Current training step.
        save_dir: Directory to save validation images.
        guidance_types: List of guidance types (e.g., "normal", "depth").
        num_inference_steps: Number of denoising steps.
        guidance_scale: Scale for classifier-free guidance.
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info("Running validation ...")
    torch.cuda.empty_cache()
    unwrap_model = accelerator.unwrap_model(model)
    reference_unet = unwrap_model.reference_unet
    gaussian_unet = unwrap_model.gaussian_unet
    val_image_prompt_embeds = val_image_prompt_embeds.to(dtype=dtype)

    # image_encoder = accelerator.unwrap_model(image_encoder).to(dtype)

    output_type = "tensor"
    return_dict = True
    callback = None
    callback_steps = 1

    eta = 0.0

    # unwrap_model = accelerator.unwrap_model(gaussian_unet)

    # gaussian_unet = unwrap_model.gaussian_unet

    generator = torch.manual_seed(seed)

    device = accelerator.device

    sphere_points = kwargs.get("sphere_points", None)
    corrs_1_to_2 = kwargs.get("corrs_1_to_2", None)

    # ---- 1) FORCE MODELS TO DESIRED DTYPE/DEVICE (before any wrappers) ----
    # gaussian_unet = gaussian_unet.to(device=device, dtype=weight_dtype)
    # reference_unet = reference_unet.to(device=device, dtype=weight_dtype)
    # image_encoder = image_encoder.to(device=device, dtype=weight_dtype)

    batch_size = batch["input"].size(0)
    # GT images
    complete_map = batch["gt"]  # [-1,1]
    if save_gt:
        ids = batch["id"]
        views = batch["view"]
        for b in range(batch_size):
            id = ids[b]
            view = views[b]
            os.makedirs(os.path.join(save_dir, f"{id}", f"{view}"), exist_ok=True)
            gt_xyz = complete_map[:, :, :, 0:3][b]  # [B, 3, H, W]
            gt_normal = complete_map[:, :, :, 3:6][b]  # [B, 3, H, W]

            # save to image
            gt_xyz_vis = (gt_xyz + 1) * 0.5
            gt_normal_vis = (gt_normal + 1) * 0.5
            gt_xyz_vis = gt_xyz_vis.clamp(0, 1)
            gt_normal_vis = gt_normal_vis.clamp(0, 1)
            gt_xyz_vis = to_pil_image(
                gt_xyz_vis.permute(2, 0, 1)
            )  # [3, H, W] -> [H, W, 3]
            gt_normal_vis = to_pil_image(gt_normal_vis.permute(2, 0, 1))
            gt_xyz_vis.save(os.path.join(save_dir, f"{id}", f"{view}", "gt_xyz.png"))
            gt_normal_vis.save(
                os.path.join(save_dir, f"{id}", f"{view}", "gt_normal.png")
            )
            # save to npy
            gt_xyz_ply = gt_xyz.cpu().numpy()
            gt_normal_ply = gt_normal.cpu().numpy()
            np.save(
                os.path.join(save_dir, f"{id}", f"{view}", "xyz_gt.npy"),
                gt_xyz_ply,
            )
            np.save(
                os.path.join(save_dir, f"{id}", f"{view}", "normal_gt.npy"),
                gt_normal_ply,
            )

            # visualize the input point cloud
            if sphere_points is not None and corrs_1_to_2 is not None:
                gt_xyz = gt_xyz.reshape(-1, 3)
                gt_normal = gt_normal.reshape(-1, 3)
                gt_xyz = gt_xyz[corrs_1_to_2] + sphere_points
                gt_normal = gt_normal[corrs_1_to_2]

                pc_o3d = o3d.geometry.PointCloud()
                pc_o3d.points = o3d.utility.Vector3dVector(gt_xyz.cpu().numpy())
                pc_o3d.normals = o3d.utility.Vector3dVector(gt_normal.cpu().numpy())
                o3d.io.write_point_cloud(
                    os.path.join(save_dir, f"{id}", f"{view}", "gt_pc.ply"), pc_o3d
                )

    complete_map = complete_map.permute(0, 3, 1, 2)  # [B, C, H, W]
    dummy_map = -torch.ones_like(complete_map[:, 0:2, :, :], dtype=complete_map.dtype)
    # concatenate the dummy map to the complete map
    complete_map = torch.cat((complete_map, dummy_map), dim=1)  # [B, 8, H, W]

    # incomplete map shape [num_batch, 8, 128, 128]
    incomplete_map = batch["input"].to(
        device
    )  # .to(device,torch.float16) #[:,0]  # .reshape(-1, 3, height, width)

    # visualize input
    if sphere_points is not None and corrs_1_to_2 is not None:
        ids = batch["id"]
        views = batch["view"]
        for b in range(batch_size):
            id = ids[b]
            view = views[b]
            os.makedirs(os.path.join(save_dir, f"{id}", f"{view}"), exist_ok=True)
            input_xyz = incomplete_map[:, :, :, 0:3][b]
            input_normal = incomplete_map[:, :, :, 3:6][b]

            # only save ply
            input_xyz = input_xyz.reshape(-1, 3)
            input_normal = input_normal.reshape(-1, 3)
            input_xyz = input_xyz[corrs_1_to_2] + sphere_points
            input_normal = input_normal[corrs_1_to_2]
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(input_xyz.cpu().numpy())
            pc_o3d.normals = o3d.utility.Vector3dVector(input_normal.cpu().numpy())
            o3d.io.write_point_cloud(
                os.path.join(save_dir, f"{id}", f"{view}", "input_pc.ply"), pc_o3d
            )

    incomplete_map = incomplete_map.permute(0, 3, 1, 2)  # [B, C, H, W]
    # also concatenate the incomplete map with the dummy map
    mask = batch["input_mask"].to(dtype=complete_map.dtype, device=device)
    mask = mask.unsqueeze(1)
    incomplete_map = torch.cat((incomplete_map, mask), dim=1)  # [B, 1, H, W]
    one_channel_dummy = -torch.ones_like(
        incomplete_map[:, 0:1, :, :], dtype=incomplete_map.dtype
    )
    incomplete_map = torch.cat((incomplete_map, one_channel_dummy), dim=1)
    # [num_batch, 8, height, width]
    num_batch, num_channels, image_height, image_width = complete_map.shape

    # Initialize random noise latents
    latents = (
        torch.randn(
            (batch_size, num_channels, image_height, image_width),
            device=device,
        )
        * scheduler.init_noise_sigma
    )  # Use initial noise scaling as per scheduler

    # Prepare reference latents
    device = accelerator.device

    # do_classifier_free_guidance = guidance_scale > 1.0
    do_classifier_free_guidance = False

    # Prepare timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # clip_img = torch.zeros(
    #     (num_batch, 3, 224, 224), dtype=image_encoder.dtype, device=accelerator.device
    # )

    # # clip_image_embeds [b=1, 768]
    # clip_image_embeds = image_encoder(
    #     clip_img.to(device, dtype=image_encoder.dtype)
    # ).image_embeds
    # image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d=768)
    image_prompt_embeds = val_image_prompt_embeds
    uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)

    if do_classifier_free_guidance:
        image_prompt_embeds = torch.cat(
            [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
        )

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=do_classifier_free_guidance,
        mode="write",
        batch_size=batch_size,
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        gaussian_unet,
        do_classifier_free_guidance=do_classifier_free_guidance,
        mode="read",
        batch_size=batch_size,
        fusion_blocks="full",
    )

    num_images_per_prompt = 1

    # num_channels_latents = gaussian_unet.in_channels
    latents = prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels,
        image_width,
        image_height,
        image_prompt_embeds.dtype,
        device,
        generator,
        scheduler.init_noise_sigma,
        vae_scale_factor=1,
    )  # .to(torch.float16)

    latents_dtype = latents.dtype

    # Prepare extra step kwargs.
    extra_step_kwargs = prepare_extra_step_kwargs(generator, eta, scheduler)
    incomplete_atlas_latents = incomplete_map
    # denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
    with tqdm(total=num_inference_steps, desc="Inference Progress") as progress_bar:
        for i, t in enumerate(timesteps):
            # 1. Forward reference image
            if i == 0:
                reference_unet(
                    incomplete_atlas_latents.repeat(
                        (2 if do_classifier_free_guidance else 1), 1, 1, 1
                    ),
                    # [b=2, 4*num_frames, 64, 64]
                    torch.zeros_like(t),
                    encoder_hidden_states=image_prompt_embeds,
                    return_dict=False,
                )

                # 2. Update reference unet feature into denosing net
                reference_control_reader.update(reference_control_writer)

            # 3.1 expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # maybe adjust the incomplete map tensor shape according to the latent_model_input
            # latent_model_input = latent_model_input + incomplete_map

            noise_pred = gaussian_unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_prompt_embeds,
                # guidance_fea=guidance_fea,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0
            ):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(scheduler, "order", 1)
                    callback(step_idx, t, latents)

        reference_control_reader.clear()
        reference_control_writer.clear()

    latents = latents.permute(0, 2, 3, 1)  # [b, h, w, c]
    xyz = latents[:, :, :, 0:3]  # [b, h, w, 3]
    normal = latents[:, :, :, 3:6]  # [b, h, w, 3]

    # save the xyz and normal maps into ply file
    xyz_ply = xyz.cpu().numpy()
    normal_ply = normal.cpu().numpy()

    for b in range(xyz_ply.shape[0]):
        id = ids[b]
        view = views[b]

        xyz_vis = (xyz[b] + 1) * 0.5
        normal_vis = (normal[b] + 1) * 0.5

        xyz_vis = xyz_vis.clamp(0, 1)  # [h, w, 3]
        normal_vis = normal_vis.clamp(0, 1)  # [h, w, 3]

        xyz_vis = to_pil_image(xyz_vis.permute(2, 0, 1))  # [3, h, w] -> [h, w, 3]
        normal_vis = to_pil_image(normal_vis.permute(2, 0, 1))
        xyz_vis.save(os.path.join(save_dir, f"{id}", f"{view}", "gen_xyz.png"))
        normal_vis.save(os.path.join(save_dir, f"{id}", f"{view}", "gen_normal.png"))

        if sphere_points is not None and corrs_1_to_2 is not None:
            gen_xyz = xyz[b].reshape(-1, 3)
            gen_normal = normal[b].reshape(-1, 3)
            gen_xyz = gen_xyz[corrs_1_to_2] + sphere_points
            gen_normal = gen_normal[corrs_1_to_2]

            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(gen_xyz.cpu().numpy())
            pc_o3d.normals = o3d.utility.Vector3dVector(gen_normal.cpu().numpy())
            o3d.io.write_point_cloud(
                os.path.join(save_dir, f"{id}", f"{view}", "gen_pc.ply"), pc_o3d
            )

    # save the xyz and normal maps into ply file
    np.save(os.path.join(save_dir, "gen_xyz.npy"), xyz_ply)
    np.save(os.path.join(save_dir, "gen_normal.npy"), normal_ply)
    # Post-processing
    print("vae input latents {}".format(latents.dtype))

    # images = decode_latents(vae, latents.to(torch.float16))  # (b, c, 1, h, w)
    # pred_gaussian_map = latents.cpu().float().numpy()

    # image_encoder = image_encoder.to(dtype=torch.float16)

    del gaussian_unet
    del latents
    del noise_pred
    del batch
    del image_prompt_embeds
    # Trigger garbage collection and CUDA cache clear

    gc.collect()
    torch.cuda.empty_cache()
    # del pipeline

    logger.info("validation Finished")


def infoCD(x, y, x_normals=None, y_normals=None, eps=1e-7):
    """Compute the InfoNCE-based Chamfer Distance between two point clouds.
    From https://github.com/ark1234/NeurIPS2023-InfoCD

    Args:
        x: Point cloud of shape (B, N, D).
        y: Point cloud of shape (B, M, D).
        eps: Small value to avoid log(0)."""
    (d1, d2), _ = chamfer_distance(
        x,
        y,
        batch_reduction=None,
        point_reduction=None,
        x_normals=x_normals,
        y_normals=y_normals,
    )
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)

    d1 = torch.sqrt(d1)
    d2 = torch.sqrt(d2)

    weights1 = torch.exp(-0.5 * d1)
    weights2 = torch.exp(-0.5 * d2)

    distances1 = -torch.log(
        weights1 / (torch.sum(weights1 + 1e-7, dim=-1).unsqueeze(-1)) ** 1e-7
    )
    distances2 = -torch.log(
        weights2 / (torch.sum(weights2 + 1e-7, dim=-1).unsqueeze(-1)) ** 1e-7
    )
    
    return (torch.sum(distances1) + torch.sum(distances2)) / 2

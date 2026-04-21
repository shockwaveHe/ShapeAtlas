import argparse
import logging
import os
import os.path as osp
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
import pdb
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, Union

import imageio
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

from models.mutual_self_attention_2d import ReferenceAttentionControl
from models.unet_2d_condition import UNet2DConditionModel
from shapeatlas_utils.dataloader import ShapeNetAtlasDataset as AtlasDataset


def setup_savedir(cfg):
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if cfg.exp_name is None:
        savedir = f"results/exp-{time_str}"
    else:
        savedir = f"results/{cfg.exp_name}-{time_str}"

    os.makedirs(savedir, exist_ok=True)

    return savedir


def fetch_data(data, device):
    for key in data.keys():
        if key in ["input", "gt"]:
            data[key] = data[key].to(device, torch.float32)

    return data


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


def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # save_dir = setup_savedir(cfg)
    save_dir = cfg.test_out_path
    logging.info(f"Running inference ...")

    # setup pretrained models
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    weight_dtype = torch.float32

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    gaussian_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )

    # replace conv_in and conv_out with new 8-channel layers
    gaussian_unet.conv_in = nn.Conv2d(8, 320, kernel_size=3, padding=1)
    gaussian_unet.conv_out = nn.Conv2d(320, 8, kernel_size=3, padding=1)
    
    reference_unet.conv_in = nn.Conv2d(8, 320, kernel_size=3, padding=1)
    reference_unet.conv_out = nn.Conv2d(320, 8, kernel_size=3, padding=1)

    if cfg.ckpt_step == "latest":
        ckpt_files = sorted(
            os.listdir(cfg.ckpt_dir), key=lambda x: int(x.split("-")[-1].split(".")[0])
        )
        latest_pth_name = (Path(cfg.ckpt_dir) / ckpt_files[-1]).stem
        stage1_ckpt_step = int(latest_pth_name.split("-")[-1])

    gaussian_unet.load_state_dict(
        torch.load(
            os.path.join(cfg.ckpt_dir, f"gaussian_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=True,  # False,
    )

    reference_unet.load_state_dict(
        torch.load(
            os.path.join(cfg.ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=True,  # False,
    )

    gaussian_unet = gaussian_unet.to(device="cuda", dtype=weight_dtype)
    reference_unet = reference_unet.to(device="cuda", dtype=weight_dtype)
    print("Model weights loaded successfully!")

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            gaussian_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    test_dataset = AtlasDataset(cfg, phase="test")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    num_total_samples = len(test_dataset)

    device = next(gaussian_unet.parameters()).device

    eta = 0.0
    seed = 42
    generator = torch.manual_seed(seed)
    output_type = "tensor"
    return_dict = True
    callback = None
    callback_steps = 1

    num_inference_steps = 20
    guidance_scale = 2
    seed = 42
    num_batch = 1

    do_classifier_free_guidance = guidance_scale > 1.0

    # Prepare timesteps
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps

    import time

    total_running_time = 0.0

    # for idx in tqdm(range(num_total_samples)):
    for idx, batch in enumerate(test_dataloader):
        with torch.no_grad():
            print(f"Processing sample {idx + 1}/{num_total_samples} ...")
            batch = fetch_data(batch, device)
            # batch['input'].shape [1, 128, 128, 6]
            # batch['gt'].shape [1, 128, 128, 6]

            complete_map = batch["gt"]  # [-1,1]

            id = batch["id"][0]
            view = batch["view"][0]
            os.makedirs(os.path.join(save_dir, f"{id}", f"{view}"), exist_ok=True)
            save_gt = True
            if save_gt:
                gt_xyz = complete_map[:, :, :, 0:3][0]  # [B, 3, H, W]
                gt_normal = complete_map[:, :, :, 3:6][0]  # [B, 3, H, W]
                # save to image
                gt_xyz_vis = (gt_xyz + 1) * 0.5
                gt_normal_vis = (gt_normal + 1) * 0.5
                gt_xyz_vis = gt_xyz_vis.clamp(0, 1)
                gt_normal_vis = gt_normal_vis.clamp(0, 1)
                gt_xyz_vis = to_pil_image(
                    gt_xyz_vis.permute(2, 0, 1)
                )  # [3, H, W] -> [H, W, 3]
                gt_normal_vis = to_pil_image(gt_normal_vis.permute(2, 0, 1))
                gt_xyz_vis.save(
                    os.path.join(save_dir, f"{id}", f"{view}", f"gt_xyz.png")
                )
                gt_normal_vis.save(
                    os.path.join(save_dir, f"{id}", f"{view}", f"gt_normal.png")
                )
                # save to npy
                gt_xyz_ply = gt_xyz.cpu().numpy()
                gt_normal_ply = gt_normal.cpu().numpy()
                np.save(
                    os.path.join(save_dir, f"{id}", f"{view}", f"xyz_gt.npy"),
                    gt_xyz_ply,
                )
                np.save(
                    os.path.join(save_dir, f"{id}", f"{view}", f"normal_gt.npy"),
                    gt_normal_ply,
                )

            complete_map = complete_map.permute(0, 3, 1, 2)  # [B, C, H, W]
            dummy_map = -torch.ones_like(
                complete_map[:, 0:2, :, :], dtype=complete_map.dtype
            )
            # concatenate the dummy map to the complete map
            complete_map = torch.cat((complete_map, dummy_map), dim=1)  # [B, 8, H, W]

            # incomplete map shape [num_batch, 8, 128, 128]
            incomplete_map = batch["input"]

            save_input = True
            if save_input:
                input_xyz = incomplete_map[:, :, :, 0:3][0]
                input_normal = incomplete_map[:, :, :, 3:6][0]
                # save to image
                input_xyz_vis = (input_xyz + 1) * 0.5
                input_normal_vis = (input_normal + 1) * 0.5
                input_xyz_vis = input_xyz_vis.clamp(0, 1)
                input_normal_vis = input_normal_vis.clamp(0, 1)
                input_xyz_vis = to_pil_image(
                    input_xyz_vis.permute(2, 0, 1)
                )  # [3, H, W] -> [H, W, 3]
                input_normal_vis = to_pil_image(input_normal_vis.permute(2, 0, 1))
                input_xyz_vis.save(
                    os.path.join(save_dir, f"{id}", f"{view}", f"input_xyz.png")
                )
                input_normal_vis.save(
                    os.path.join(save_dir, f"{id}", f"{view}", f"input_normal.png")
                )
                input_mask = batch["input_mask"][0].to(torch.uint8) * 255
                input_mask = to_pil_image(input_mask)
                input_mask.save(
                    os.path.join(save_dir, f"{id}", f"{view}", f"input_mask.png")
                )
                # save to npy
                input_xyz_ply = input_xyz.cpu().numpy()
                input_normal_ply = input_normal.cpu().numpy()
                np.save(
                    os.path.join(save_dir, f"{id}", f"{view}", f"xyz_input.npy"),
                    input_xyz_ply,
                )
                np.save(
                    os.path.join(save_dir, f"{id}", f"{view}", f"normal_input.npy"),
                    input_normal_ply,
                )

            incomplete_map = incomplete_map.permute(0, 3, 1, 2)  # [B, C, H, W]
            # also concatenate the incomplete map with the dummy map
            incomplete_map = torch.cat(
                (incomplete_map, dummy_map), dim=1
            )  # [B, 8, H, W]

            num_batch, num_channels, image_height, image_width = incomplete_map.shape

            clip_img = torch.zeros(
                (num_batch, 3, 224, 224),  #
                dtype=image_enc.dtype,
                device=device,
            )

            # clip_img = torch.stack(clip_image_list, dim=0).to(dtype=image_enc.dtype, device=image_enc.device)
            # clip_image_embeds [b=1, 768]

            clip_image_embeds = image_enc(
                clip_img.to(device, dtype=image_enc.dtype)
            ).image_embeds
            image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d=768)
            uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)

            if do_classifier_free_guidance:  # True
                image_prompt_embeds = torch.cat(
                    [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
                )
                # image_prompt_embeds.shape [1, 1, 768] -> [2, 1, 768]
            reference_control_writer = ReferenceAttentionControl(
                reference_unet,
                do_classifier_free_guidance=False,
                mode="write",  #
                fusion_blocks="full",
            )
            reference_control_reader = ReferenceAttentionControl(
                gaussian_unet,
                do_classifier_free_guidance=False,
                mode="read",
                fusion_blocks="full",
            )

            num_images_per_prompt = 1

            latents = prepare_latents(
                num_batch * num_images_per_prompt,
                num_channels,
                image_width,
                image_height,
                clip_image_embeds.dtype,
                device,
                generator,
                noise_scheduler.init_noise_sigma,
                vae_scale_factor=1,
            )
            """
            latents = prepare_latents(
                num_batch * num_images_per_prompt,
                num_channels_latents,
                image_width,
                image_height,
                video_length,
                clip_image_embeds.dtype,
                device,
                generator,
                noise_scheduler.init_noise_sigma
            )
            """

            # Prepare extra step kwargs.
            extra_step_kwargs = prepare_extra_step_kwargs(
                generator, eta, noise_scheduler
            )

            # denoising loop
            num_warmup_steps = (
                len(timesteps) - num_inference_steps * noise_scheduler.order
            )
            with tqdm(
                total=num_inference_steps, desc="Inference Progress"
            ) as progress_bar:
                for i, t in enumerate(timesteps):
                    # 1. Forward reference image
                    if i == 0:
                        reference_unet(
                            incomplete_map.repeat(
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
                        torch.cat([latents] * 2)
                        if do_classifier_free_guidance
                        else latents
                    )
                    latent_model_input = noise_scheduler.scale_model_input(
                        latent_model_input, t
                    )
                    # import pdb; pdb.set_trace()
                    latent_model_input = latent_model_input + incomplete_map
                    # latent_model_input = torch.cat([incomplete_map] * 2)
                    noise_pred = gaussian_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_prompt_embeds,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = noise_scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % noise_scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(noise_scheduler, "order", 1)
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
                xyz_vis = (xyz[b] + 1) * 0.5
                normal_vis = (normal[b] + 1) * 0.5

                xyz_vis = xyz_vis.clamp(0, 1)  # [h, w, 3]
                normal_vis = normal_vis.clamp(0, 1)  # [h, w, 3]

                xyz_vis = to_pil_image(
                    xyz_vis.permute(2, 0, 1)
                )  # [3, h, w] -> [h, w, 3]
                normal_vis = to_pil_image(normal_vis.permute(2, 0, 1))
                xyz_vis.save(
                    os.path.join(save_dir, f"{id}", f"{view}", f"gen_xyz_{b:04d}.png")
                )
                normal_vis.save(
                    os.path.join(
                        save_dir, f"{id}", f"{view}", f"gen_normal_{b:04d}.png"
                    )
                )
            # save the xyz and normal maps into ply file
            np.save(os.path.join(save_dir, f"{id}", f"{view}", f"gen_xyz.npy"), xyz_ply)
            np.save(
                os.path.join(save_dir, f"{id}", f"{view}", f"gen_normal.npy"),
                normal_ply,
            )


if __name__ == "__main__":
    """
    USAGE
    CUDA_VISIBLE_DEVICES=0 python eval_condition_unet.py
    """

    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/eval_condition_unet.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")

    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamp = "0"
    exp_name = "debug"

    config.test_out_path = os.path.join(
        os.path.dirname(config.ckpt_dir), "eval", f"{exp_name}_{timestamp}"
    )

    # Create the directory if it doesn't exist
    os.makedirs(config.test_out_path, exist_ok=True)

    main(config)

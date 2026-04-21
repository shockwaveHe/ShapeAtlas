import argparse
import logging
import math
import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import os.path as osp
import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union

import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import BaseOutput, check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

import wandb
from models.unet_2d_condition import UNet2DConditionModel

# from lib.datasets.data_loader import AtlasDataset
from partial_mesh_utils.dataloader import ShapeNetAtlasDataset as AtlasDataset
from train_unet_utils.tb_tracker import TbTracker
from train_unet_utils.util import compute_snr, delete_additional_ckpt, seed_everything

warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


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


@dataclass
class MultiGuidance2VideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


@torch.no_grad()
def log_validation(
    batch,
    image_encoder,
    gaussian_unet,
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

    logger.info("Running validation ...")
    torch.cuda.empty_cache()

    output_type = "tensor"
    return_dict = True
    callback = None
    callback_steps = 1

    eta = 0.0

    unwrap_model = accelerator.unwrap_model(gaussian_unet)

    # gaussian_unet = unwrap_model.gaussian_unet

    generator = torch.manual_seed(seed)

    image_encoder = image_encoder  # .to(dtype=torch.float16)

    device = accelerator.device
    batch_size = batch["input"].size(0)
    # GT images
    complete_map = batch["gt"]  # [-1,1]
    complete_map = complete_map.permute(0, 3, 1, 2)  # [B, C, H, W]
    dummy_map = -torch.ones_like(complete_map[:, 0:2, :, :], dtype=complete_map.dtype)
    # concatenate the dummy map to the complete map
    complete_map = torch.cat((complete_map, dummy_map), dim=1)  # [B, 8, H, W]

    # incomplete map shape [num_batch, 8, 128, 128]
    incomplete_map = batch["input"].to(
        device
    )  # .to(device,torch.float16) #[:,0]  # .reshape(-1, 3, height, width)
    incomplete_map = incomplete_map.permute(0, 3, 1, 2)  # [B, C, H, W]
    # also concatenate the incomplete map with the dummy map
    incomplete_map = torch.cat((incomplete_map, dummy_map), dim=1)  # [B, 8, H, W]

    # [num_batch, 8, height, width]
    num_batch, num_channels, image_height, image_width = incomplete_map.shape

    # Initialize random noise latents
    latents = (
        torch.randn(
            (batch_size, num_channels, image_height, image_width),
            device=device,
        )
        * scheduler.init_noise_sigma
    )  # Use initial noise scaling as per scheduler

    # Prepare reference latents

    vae_height = image_height
    vae_width = image_width

    device = accelerator.device

    do_classifier_free_guidance = guidance_scale > 1.0

    # Prepare timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # batch_size = 1
    # vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    clip_img = torch.zeros(
        (num_batch, 3, 224, 224), dtype=image_encoder.dtype, device=accelerator.device
    )

    # clip_image_embeds [b=1, 768]
    clip_image_embeds = image_encoder(
        clip_img.to(device, dtype=image_encoder.dtype)
    ).image_embeds
    image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d=768)
    uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)

    if do_classifier_free_guidance:
        image_prompt_embeds = torch.cat(
            [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
        )

    num_images_per_prompt = 1

    # num_channels_latents = gaussian_unet.in_channels
    latents = prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels,
        image_width,
        image_height,
        clip_image_embeds.dtype,
        device,
        generator,
        scheduler.init_noise_sigma,
        vae_scale_factor=1,
    )  # .to(torch.float16)

    latents_dtype = latents.dtype

    # Prepare extra step kwargs.
    extra_step_kwargs = prepare_extra_step_kwargs(generator, eta, scheduler)

    # denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
    with tqdm(total=num_inference_steps, desc="Inference Progress") as progress_bar:
        for i, t in enumerate(timesteps):
            # 3.1 expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # maybe adjust the incomplete map tensor shape according to the latent_model_input
            latent_model_input = latent_model_input + incomplete_map

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

    latents = latents.permute(0, 2, 3, 1)  # [b, h, w, c]
    xyz = latents[:, :, :, 0:3]  # [b, h, w, 3]
    normal = latents[:, :, :, 3:6]  # [b, h, w, 3]

    # save the xyz and normal maps into ply file
    xyz_ply = xyz.cpu().numpy()
    normal_ply = normal.cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)

    for b in range(xyz_ply.shape[0]):
        xyz_vis = (xyz[b] + 1) * 0.5
        normal_vis = (normal[b] + 1) * 0.5

        xyz_vis = xyz_vis.clamp(0, 1)  # [h, w, 3]
        normal_vis = normal_vis.clamp(0, 1)  # [h, w, 3]

        xyz_vis = to_pil_image(xyz_vis.permute(2, 0, 1))  # [3, h, w] -> [h, w, 3]
        normal_vis = to_pil_image(normal_vis.permute(2, 0, 1))
        xyz_vis.save(os.path.join(save_dir, f"xyz_{b:04d}.png"))
        normal_vis.save(os.path.join(save_dir, f"normal_{b:04d}.png"))
    # save the xyz and normal maps into ply file
    np.save(os.path.join(save_dir, "xyz.npy"), xyz_ply)
    np.save(os.path.join(save_dir, "normal.npy"), normal_ply)
    # Post-processing
    print("vae input latents {}".format(latents.dtype))

    # images = decode_latents(vae, latents.to(torch.float16))  # (b, c, 1, h, w)
    pred_gaussian_map = latents.cpu().float().numpy()

    image_encoder = image_encoder.to(dtype=torch.float16)

    del gaussian_unet
    del latents
    del noise_pred
    del batch
    del image_prompt_embeds
    del clip_image_embeds

    # Trigger garbage collection and CUDA cache clear
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    # del pipeline

    logger.info("validation Finished")


def load_weights(
    gaussian_unet,
    ckpt_dir,
    ckpt_step="latest",
):
    if ckpt_step == "latest":
        ckpt_files = sorted(
            os.listdir(ckpt_dir), key=lambda x: int(x.split("-")[-1].split(".")[0])
        )
        latest_pth_name = (Path(ckpt_dir) / ckpt_files[-1]).stem
        stage1_ckpt_step = int(latest_pth_name.split("-")[-1])

    gaussian_unet.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, f"gaussian_unet-{ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )

    logger.info(f"Loaded stage2 models from {ckpt_dir}, step={ckpt_step}")


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    print(cfg.solver.mixed_precision)  # fp16

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(cfg.output_dir, f"{cfg.exp_name}_{timestamp}")

    wandb.init(project="partial_mesh",
               sync_tensorboard=True,
               name=f"{cfg.exp_name}_{timestamp}",
               config=OmegaConf.to_container(cfg))

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "sanity_check"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "saved_models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "validation"), exist_ok=True)
    tb_tracker = TbTracker(f"{cfg.exp_name}_{timestamp}/events", cfg.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        device_placement=True,  #
        mixed_precision=cfg.solver.mixed_precision,
        split_batches=True,
        log_with=tb_tracker,
        project_dir=f"{save_dir}",
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path
    ).to(dtype=weight_dtype)  ##.to(dtype=weight_dtype, device="cuda")

    # load pretrained weights from Stable Diffusion
    gaussian_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )
    # extract original conv_in / conv_out weights
    conv_in_weight = gaussian_unet.conv_in.weight.data.clone()  # [320, 4, 3, 3]
    conv_in_bias = gaussian_unet.conv_in.bias.data.clone()  # [320]
    conv_out_weight = gaussian_unet.conv_out.weight.data.clone()  # [4, 320, 3, 3]
    conv_out_bias = gaussian_unet.conv_out.bias.data.clone()  # [4]

    # tile weights to match 8 input/output channels
    conv_in_weight_tiled = conv_in_weight.repeat(1, 2, 1, 1) / 2.0  # [320, 8, 3, 3]
    conv_out_weight_tiled = conv_out_weight.repeat(2, 1, 1, 1) / 2.0  # [8, 320, 3, 3]
    conv_out_bias_tiled = conv_out_bias.repeat(2) / 2.0  # [8]

    # replace conv_in and conv_out with new 8-channel layers
    gaussian_unet.conv_in = nn.Conv2d(8, 320, kernel_size=3, padding=1)
    gaussian_unet.conv_out = nn.Conv2d(320, 8, kernel_size=3, padding=1)

    # assign the tiled weights
    with torch.no_grad():
        gaussian_unet.conv_in.weight.copy_(conv_in_weight_tiled)
        gaussian_unet.conv_in.bias.copy_(conv_in_bias)
        gaussian_unet.conv_out.weight.copy_(conv_out_weight_tiled)
        gaussian_unet.conv_out.bias.copy_(conv_out_bias_tiled)

    if cfg.resume_from_checkpoint:
        load_weights(
            gaussian_unet,
            cfg.resume_from_checkpoint,
            "latest",
        )

    gaussian_unet.requires_grad_(True)
    image_enc.requires_grad_(False)

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            gaussian_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:  # False
        gaussian_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(
        filter(lambda p: p.requires_grad, gaussian_unet.parameters())
    )

    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        step_rules=cfg.solver.step_rule,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    # Modify here!
    # train_dataset = AtlasDataset(data_root='/scr/yaohe/data',phase='train')
    # val_dataset = AtlasDataset(data_root='/scr/yaohe/data',phase='val')
    train_dataset = AtlasDataset(cfg, phase="train")
    val_dataset = AtlasDataset(cfg, phase="test")

    # to debug
    # train_set.get_item_debug(0)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.train_bs,
        shuffle=True,
        drop_last=True,
        num_workers=4,  # 4
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.validation.val_bs,
        shuffle=True,
        drop_last=True,
        num_workers=4,  # 4
    )
    val_iterator = iter(val_dataloader)

    (
        gaussian_unet,
        image_enc,
        optimizer,
        train_dataloader,
        val_dataloader,
        val_iterator,
        lr_scheduler,
    ) = accelerator.prepare(
        gaussian_unet,
        image_enc,
        optimizer,
        train_dataloader,
        val_dataloader,
        val_iterator,
        lr_scheduler,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    logger.info("Start training ...")
    logger.info(f"Num Samples: {len(train_dataset)}")
    logger.info(f"Train Batchsize: {cfg.data.train_bs}")
    logger.info(f"Num Epochs: {num_train_epochs}")
    logger.info(f"Total Steps: {cfg.solver.max_train_steps}")

    global_step, first_epoch = 0, 0

    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # Training Loop
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            print(f"step: {step}")

            with accelerator.accumulate(gaussian_unet):
                ### encode GT
                complete_map = batch["gt"].to(weight_dtype)  # [0,1]
                complete_map = complete_map.permute(0, 3, 1, 2)  # [B, C, H, W]

                # shape [num_batch, 8 = 3 (position) + 3 (normal) + 2 (dummy), 128, 128]
                dummy_map = -torch.ones_like(
                    complete_map[:, 0:2, :, :], dtype=complete_map.dtype
                )
                # concatenate the dummy map to the complete map
                complete_map = torch.cat((complete_map, dummy_map), dim=1)
                # value [-1,1]
                num_batch, num_channel, image_height, image_width = complete_map.shape

                noise = torch.randn_like(complete_map)
                if cfg.noise_offset > 0.0:  # 0.05
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], 1, 1, 1),
                        device=accelerator.device,  # noise.device,
                    )

                bsz = complete_map.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,  # 1000
                    (bsz,),
                    device=accelerator.device,  # latents.device,
                )
                timesteps = timesteps.long()

                uncond_fwd = random.random() < cfg.uncond_ratio  # 0.1

                with torch.no_grad():
                    vae_height = image_height
                    vae_width = image_width

                    # [b=1, 3, 224, 224]
                    clip_img = torch.zeros(
                        (num_batch, 3, 224, 224),
                        dtype=image_enc.dtype,
                        device=accelerator.device,
                    )
                    clip_image_embeds = image_enc(
                        clip_img.to(device=accelerator.device, dtype=image_enc.dtype)
                    ).image_embeds.to(image_enc.dtype)

                    image_prompt_embeds = clip_image_embeds.unsqueeze(
                        1
                    )  # (bs, 1, d=768)
                    image_prompt_embeds = image_prompt_embeds.to(image_enc.dtype)

                # add noise to GT
                noisy_latents = train_noise_scheduler.add_noise(
                    complete_map, noise, timesteps
                )

                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":  # this
                    target = train_noise_scheduler.get_velocity(
                        complete_map, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                """
                model_pred = model(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    image_prompt_embeds,
                    tgt_guid_imgs,
                    uncond_fwd,
                )
                """

                incomplete_map = batch["input"]
                # permute the incomplete map to match the input shape
                incomplete_map = incomplete_map.permute(0, 3, 1, 2)  # [B, C, H, W]
                # also concatenate the incomplete map with the dummy map
                incomplete_map = torch.cat((incomplete_map, dummy_map), dim=1)

                noisy_latents = incomplete_map + noisy_latents

                model_pred = gaussian_unet(
                    noisy_latents, timesteps, clip_image_embeds[:, None]
                ).sample
                # target = target.to(dtype=model_pred.dtype)

                if cfg.snr_gamma == 0:  # cfg.snr_gamma = 5.0
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()

                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                step_loss_log = loss.detach().item()

                # import gc
                # gc.collect()
                # accelerator.free_memory()
                # torch.cuda.empty_cache()

            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                tb_tracker.add_scalar(
                    tag="train loss", scalar_value=train_loss, global_step=global_step
                )
                train_loss = 0.0
                # 　save checkpoints
                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            save_dir, "checkpoints", f"checkpoint-{global_step}"
                        )
                        delete_additional_ckpt(save_dir, 6)
                        accelerator.save_state(save_path)

                        # unwrap_model = accelerator.unwrap_model(gaussian_unet)

                        save_checkpoint(
                            gaussian_unet,
                            f"{save_dir}/saved_models",
                            "gaussian_unet",
                            global_step,
                            total_limit=None,
                        )

                # log validation

                if (
                    global_step % cfg.validation.validation_steps == 0
                    or global_step == 1
                ):
                    if accelerator.is_main_process:
                        try:
                            val_batch = next(val_iterator)
                        except:
                            val_iterator = iter(val_dataloader)
                            val_batch = next(val_iterator)

                        for key in val_batch.keys():
                            if key in ["input", "gt"]:
                                val_batch[key] = val_batch[key].to(
                                    accelerator.device, torch.float16
                                )

                        with torch.no_grad():
                            log_validation(
                                batch=val_batch,
                                image_encoder=image_enc,
                                gaussian_unet=gaussian_unet,
                                scheduler=val_noise_scheduler,
                                accelerator=accelerator,
                                global_step=global_step,
                                save_dir=f"{save_dir}/validation/checkpoint-{global_step}",
                                num_inference_steps=20,
                                guidance_scale=1.0,
                            )

            logs = {
                "step_loss": step_loss_log,  # loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "stage": 1,
            }

            wandb.log(logs)

            progress_bar.set_postfix(**logs)  #

            if global_step >= cfg.solver.max_train_steps:
                break
            print(torch.cuda.max_memory_allocated() / 1024**2, "MB used this step")
    accelerator.wait_for_everyone()
    accelerator.end_training()


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


if __name__ == "__main__":  #
    """
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --main_process_port 29500 train_unet.py
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29500 train_unet.py
    CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29500 train_unet.py
    
    # for debugging (pdb)
    CUDA_VISIBLE_DEVICES=0 python train_unet.py 

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/train_unet.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")

    main(config)

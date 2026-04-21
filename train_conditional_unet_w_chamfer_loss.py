import argparse
import gc
import logging
import math
import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import os.path as osp
import random
import warnings
from datetime import datetime

import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

import wandb

# from MeshAnything.models.meshanything_v2 import MeshAnythingV2
from models.atlas_model import AtlasModel
from models.mutual_self_attention_2d import ReferenceAttentionControl
from models.unet_2d_condition import UNet2DConditionModel

# from lib.datasets.data_loader import AtlasDataset
from shapeatlas_utils.dataloader import ObjaverseAtlasDataset, ShapeNetAtlasDataset, PCNAtlasDataset
from train_unet_utils.tb_tracker import TbTracker
from train_unet_utils.util import (
    N,
    compute_snr,
    delete_additional_ckpt,
    infoCD,
    init_unit_sphere_grid,
    load_weights,
    log_validation,
    logger,
    save_checkpoint,
    seed_everything,
)

warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")


# logger = get_logger(__name__, log_level="INFO")
def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    print(cfg.solver.mixed_precision)  # fp16

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(cfg.output_dir, f"{cfg.exp_name}_{timestamp}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "sanity_check"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "saved_models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "validation"), exist_ok=True)
    # Tensorboard tracker
    tb_tracker = TbTracker(f"{cfg.exp_name}_{timestamp}/events", cfg.output_dir)

    # accelerator setup
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        device_placement=True,  #
        mixed_precision=cfg.solver.mixed_precision,
        split_batches=True,
        log_with=tb_tracker,
        project_dir=f"{save_dir}",
        kwargs_handlers=[kwargs],
    )

    # logger setup
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
    if accelerator.is_main_process:
        wandb.init(
            project=f"partial_mesh",
            sync_tensorboard=True,
            name=f"{cfg.exp_name}_{timestamp}_{cfg.wandb_tag}",
            config=OmegaConf.to_container(cfg),
        )

    # general config
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

    ################################################################################
    # training and model setup
    ################################################################################

    # 1. Diffusion Unet model
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
        # ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )
    gaussian_unet.load_state_dict(
        torch.load(
            osp.join(cfg.ckpt_dir, "denoising_unet.pth"),
            map_location="cuda",
        ),
        strict=False,
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

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
        # ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )  # .to(device="cuda")

    reference_unet.load_state_dict(
        torch.load(
            osp.join(cfg.ckpt_dir, "reference_unet.pth"),
            map_location="cuda",
        ),
        strict=False,
    )

    # extract original conv_in / conv_out weights
    conv_in_weight = reference_unet.conv_in.weight.data.clone()  # [320, 4, 3, 3]
    conv_in_bias = reference_unet.conv_in.bias.data.clone()  # [320]
    conv_out_weight = reference_unet.conv_out.weight.data.clone()  # [4, 320, 3, 3]
    conv_out_bias = reference_unet.conv_out.bias.data.clone()  # [4]

    # tile weights to match 8 input/output channels
    conv_in_weight_tiled = conv_in_weight.repeat(1, 2, 1, 1) / 2.0  # [320, 8, 3, 3]
    conv_out_weight_tiled = conv_out_weight.repeat(2, 1, 1, 1) / 2.0  # [8, 320, 3, 3]
    conv_out_bias_tiled = conv_out_bias.repeat(2) / 2.0  # [8]

    # replace conv_in and conv_out with new 8-channel layers
    reference_unet.conv_in = nn.Conv2d(8, 320, kernel_size=3, padding=1)
    reference_unet.conv_out = nn.Conv2d(320, 8, kernel_size=3, padding=1)

    # assign the tiled weights
    with torch.no_grad():
        reference_unet.conv_in.weight.copy_(conv_in_weight_tiled)
        reference_unet.conv_in.bias.copy_(conv_in_bias)
        reference_unet.conv_out.weight.copy_(conv_out_weight_tiled)
        reference_unet.conv_out.bias.copy_(conv_out_bias_tiled)

    # gaussian_unet = gaussian_unet.to(device="cuda", dtype=weight_dtype)
    # reference_unet = reference_unet.to(device="cuda", dtype=weight_dtype)

    if cfg.resume_from_checkpoint:
        load_weights(
            gaussian_unet, reference_unet, cfg.resume_from_checkpoint, "latest"
        )
        gaussian_unet = gaussian_unet.to(device="cuda")  # , dtype=weight_dtype)
        reference_unet = reference_unet.to(device="cuda")  # , dtype=weight_dtype)

    # gaussian_unet.requires_grad_(True)
    # freeze this
    if cfg.freeze_gaussian_unet:
        gaussian_unet.requires_grad_(False)
    else:
        gaussian_unet.requires_grad_(True)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(True)
    # for name, param in reference_unet.named_parameters():
    #     if "up_blocks.3" in name:
    #         param.requires_grad_(False)
    #     else:
    #         param.requires_grad_(True)

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

    model = AtlasModel(
        reference_unet,
        gaussian_unet,
        reference_control_writer,
        reference_control_reader,
        # guidance_encoder_group,
    )
    # _ = model.to(device="cuda", dtype=weight_dtype)
    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            if cfg.solver.use_flash_attention:
                print("Using flash attention for unet!")
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

                gaussian_unet.enable_xformers_memory_efficient_attention(
                    attention_op=MemoryEfficientAttentionFlashAttentionOp
                )
                reference_unet.enable_xformers_memory_efficient_attention(
                    attention_op=MemoryEfficientAttentionFlashAttentionOp
                )
            else:
                gaussian_unet.enable_xformers_memory_efficient_attention()
                reference_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:  # False
        gaussian_unet.enable_gradient_checkpointing()
        reference_unet.enable_gradient_checkpointing()

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
    # 2. Mesh generation Transformer model
    # meshanything = MeshAnythingV2.from_pretrained("Yiwen-ntu/meshanythingv2")

    # Freeze the mesh transformer during training but only provide supervision
    # meshanything.requires_grad_(False)

    # trainable_params = list(filter(lambda p: p.requires_grad, gaussian_unet.parameters()))
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
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
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )
    ################################################################################
    # training and model setup end
    ################################################################################

    train_dataset_list = []
    test_dataset_list = []
    for dataset_name in cfg.datasets:
        if dataset_name == "ShapeNet":
            train_dataset = ShapeNetAtlasDataset(cfg, phase="train")
            test_dataset = ShapeNetAtlasDataset(cfg, phase="test")
        elif dataset_name == "Objaverse":
            train_dataset = ObjaverseAtlasDataset(cfg, phase="train")
            test_dataset = ObjaverseAtlasDataset(cfg, phase="test")
        elif dataset_name == "PCN":
            train_dataset = PCNAtlasDataset(cfg, phase="train")
            test_dataset = PCNAtlasDataset(cfg, phase="test")
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        train_dataset_list.append(train_dataset)
        test_dataset_list.append(test_dataset)
    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(test_dataset_list)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.train_bs,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    print(f"Number of training examples: {len(train_dataset)}")

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.validation.val_bs,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    val_iterator = iter(val_dataloader)
    (
        model,
        image_enc,
        optimizer,
        train_dataloader,
        val_dataloader,
        val_iterator,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        image_enc,
        optimizer,
        train_dataloader,
        val_dataloader,
        val_iterator,
        lr_scheduler,
    )

    # if cfg.accelerator_resume_from_checkpoint:
    #     # need to load optimizer and scheduler states
    #     accelerator.load_state(cfg.accelerator_resume_from_checkpoint)

    ##############################################################################
    # OT reconstruction setup
    alphas_cumprod = train_noise_scheduler.alphas_cumprod.to(device=accelerator.device)

    sphere_points = init_unit_sphere_grid(N).to(device=accelerator.device)

    correspondences = np.load(cfg.correspondence_file)
    # corrs_2_to_1 = correspondences["corrs_2_to_1"]
    corrs_1_to_2 = correspondences["corrs_1_to_2"]
    corrs_1_to_2 = torch.tensor(corrs_1_to_2, device=accelerator.device)
    ##############################################################################
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
    with torch.no_grad():
        # [b=1, 3, 224, 224]
        clip_img = torch.zeros(
            (cfg.data.train_bs, 3, 224, 224),
            dtype=image_enc.dtype,
            device=accelerator.device,
        )
        clip_image_embeds = image_enc(
            clip_img.to(device=accelerator.device, dtype=image_enc.dtype)
        ).image_embeds.to(image_enc.dtype)

        image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d=768)
        image_prompt_embeds = image_prompt_embeds.to(image_enc.dtype)

        val_clip_img = torch.zeros(
            (cfg.validation.val_bs, 3, 224, 224),
            dtype=image_enc.dtype,
            device=accelerator.device,
        )
        val_clip_image_embeds = image_enc(
            val_clip_img.to(device=accelerator.device, dtype=image_enc.dtype)
        ).image_embeds.to(image_enc.dtype)
        val_image_prompt_embeds = val_clip_image_embeds.unsqueeze(1)  # (bs, 1, d=768)
        val_image_prompt_embeds = val_image_prompt_embeds.to(image_enc.dtype)

    image_enc = accelerator.free_memory(image_enc)
    del image_enc
    gc.collect()
    torch.cuda.empty_cache()
    # Training Loop
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            print(f"step: {step}")
            
            with accelerator.accumulate(model):
                ### encode GT
                complete_map = batch["gt"]  # .to(weight_dtype)  # [0,1]
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

                # with torch.no_grad():
                #     vae_height = image_height
                #     vae_width = image_width

                #     # [b=1, 3, 224, 224]
                #     clip_img = torch.zeros(
                #         (num_batch, 3, 224, 224),
                #         dtype=image_enc.dtype,
                #         device=accelerator.device,
                #     )
                #     clip_image_embeds = image_enc(
                #         clip_img.to(device=accelerator.device, dtype=image_enc.dtype)
                #     ).image_embeds.to(image_enc.dtype)

                #     image_prompt_embeds = clip_image_embeds.unsqueeze(
                #         1
                #     )  # (bs, 1, d=768)
                #     image_prompt_embeds = image_prompt_embeds.to(image_enc.dtype)

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

                incomplete_map = batch["input"].to(dtype=complete_map.dtype)

                # permute the incomplete map to match the input shape
                incomplete_map = incomplete_map.permute(0, 3, 1, 2)  # [B, C, H, W]
                # also concatenate the incomplete map with the dummy map

                mask = batch["input_mask"].to(dtype=complete_map.dtype)
                mask = mask.unsqueeze(1)
                incomplete_map = torch.cat((incomplete_map, mask), dim=1)
                one_channel_dummy = -torch.ones_like(
                    incomplete_map[:, 0:1, :, :], dtype=incomplete_map.dtype
                )
                incomplete_map = torch.cat(
                    (incomplete_map, one_channel_dummy), dim=1
                )
                # incomplete_map = torch.cat((incomplete_map, dummy_map), dim=1)

                # noisy_latents = incomplete_map + noisy_latents
                # model_pred = gaussian_unet(noisy_latents, timesteps,clip_image_embeds[:, None]).sample

                model_pred = model(
                    noisy_latents,
                    timesteps,
                    incomplete_map,  # ref_image_latents,
                    image_prompt_embeds,
                    # tgt_guid_imgs,
                    uncond_fwd,
                )
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
                
                total_loss = loss
                # add reconstruction loss
                # use velocity to reconstruct the clean atlas
                if cfg.pointcloud_reconstruction_loss_weight > 0.0 or cfg.point_mehs_distance_loss_weight > 0.0:
                    # 1. get timestep-related alpha_bar and 1-alpha_bar
                    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                    while len(sqrt_alpha_prod.shape) < len(complete_map.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

                    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                    while len(sqrt_one_minus_alpha_prod.shape) < len(
                        complete_map.shape
                    ):
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(
                            -1
                        )
                    if train_noise_scheduler.config.prediction_type == "epsilon":
                        raise ValueError("do not support epsilon prediction!")
                    elif train_noise_scheduler.config.prediction_type == "v_prediction":
                        # 2. get the predicted clean atlas from model prediction (velocity)
                        predicted_sample = (
                            sqrt_alpha_prod * noisy_latents
                            - sqrt_one_minus_alpha_prod * model_pred
                        )
                        predicted_sample = predicted_sample.permute(
                            0, 2, 3, 1
                        )  # [B, H, W, C]
                if cfg.pointcloud_reconstruction_loss_weight > 0.0:
                    if not cfg.use_info_loss:
                        if not cfg.use_normal:
                            chamfer_distance_loss, _ = chamfer_distance(
                                predicted_sample[:, :, :, 0:3].reshape(bsz, -1, 3)[
                                    :, corrs_1_to_2
                                ]
                                + sphere_points,
                                complete_map[:, 0:3, :, :]
                                .permute(0, 2, 3, 1)
                                .reshape(bsz, -1, 3)[:, corrs_1_to_2]
                                + sphere_points,
                                batch_reduction=None,
                                point_reduction="mean",
                            )
                        else:
                            chamfer_distance_loss, _ = chamfer_distance(
                                predicted_sample[:, :, :, 0:3].reshape(bsz, -1, 3)[
                                    :, corrs_1_to_2
                                ]
                                + sphere_points.repeat(1, 2),
                                complete_map[:, 0:3, :, :]
                                .permute(0, 2, 3, 1)
                                .reshape(bsz, -1, 3)[:, corrs_1_to_2]
                                + sphere_points,
                                batch_reduction=None,
                                point_reduction="mean",
                                x_normals=predicted_sample[:, :, :, 3:6].reshape(
                                    bsz, -1, 3
                                )[:, corrs_1_to_2],
                                y_normals=complete_map[:, 3:6, :, :]
                                .permute(0, 2, 3, 1)
                                .reshape(bsz, -1, 3)[:, corrs_1_to_2],
                            )
                        chamfer_distance_loss = (
                            (1.0 / (timesteps.float() + 1))
                            * chamfer_distance_loss  # add 1 to avoid div 0
                        ).mean()
                    else:
                        if not cfg.use_normal:
                            chamfer_distance_loss = infoCD(
                                predicted_sample[:, :, :, 0:3].reshape(bsz, -1, 3)[
                                    :, corrs_1_to_2
                                ]
                                + sphere_points,
                                complete_map[:, 0:3, :, :]
                                .permute(0, 2, 3, 1)
                                .reshape(bsz, -1, 3)[:, corrs_1_to_2]
                                + sphere_points,
                            )
                        else:
                            chamfer_distance_loss = infoCD(
                                predicted_sample[:, :, :, 0:3].reshape(bsz, -1, 3)[
                                    :, corrs_1_to_2
                                ]
                                + sphere_points,
                                complete_map[:, 0:3, :, :]
                                .permute(0, 2, 3, 1)
                                .reshape(bsz, -1, 3)[:, corrs_1_to_2]
                                + sphere_points,
                                x_normals=predicted_sample[:, :, :, 3:6].reshape(
                                    bsz, -1, 3
                                )[:, corrs_1_to_2],
                                y_normals=complete_map[:, 3:6, :, :]
                                .permute(0, 2, 3, 1)
                                .reshape(bsz, -1, 3)[:, corrs_1_to_2],
                            )
                        chamfer_distance_loss = (
                            (1.0 / (timesteps.float() + 1))
                            * chamfer_distance_loss  # add 1 to avoid div 0
                        ).mean()

                    total_loss = (
                        total_loss
                        + cfg.pointcloud_reconstruction_loss_weight
                        * chamfer_distance_loss
                    )
                if cfg.point_mesh_distance_loss_weight > 0.0:
                    verts = batch['mesh_verts']
                    faces = batch['mesh_faces']
                    
                    # may be buggy according to pytorch3d doc on batch > 1
                    meshes = Meshes(verts=verts, faces=faces)
                    pcl = Pointclouds(
                        points=predicted_sample[:, :, :, 0:3].reshape(bsz, -1, 3)[
                            :, corrs_1_to_2
                        ]
                        + sphere_points,
                        normals=predicted_sample[:, :, :, 3:6].reshape(bsz, -1, 3)[
                            :, corrs_1_to_2
                        ],
                    )
                    point_mesh_distance_loss = point_mesh_face_distance(
                        meshes.to(device=accelerator.device),
                        pcl.to(device=accelerator.device),
                    )
                    
                    point_mesh_distance_loss = (
                        (1.0 / (timesteps.float() + 1))
                        * point_mesh_distance_loss  # add 1 to avoid div 0
                    ).mean()
                    total_loss = (
                        total_loss
                        + cfg.point_mesh_distance_loss_weight
                        * point_mesh_distance_loss
                    )
                # else:
                #     total_loss = loss

                avg_loss = accelerator.gather(
                    total_loss.repeat(cfg.data.train_bs)
                ).mean()

                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps
                # Backpropagate
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                loss_dict = {
                    "step_train_loss_log": train_loss,
                }
                if cfg.pointcloud_reconstruction_loss_weight > 0.0:
                    step_chamfer_loss_log = chamfer_distance_loss.detach().item()
                    loss_dict["diffusion_loss"] = loss.detach().item()
                    loss_dict["chamfer_distance_loss_weight"] = (
                        cfg.pointcloud_reconstruction_loss_weight
                    )
                    loss_dict["chamfer_distance_loss"] = step_chamfer_loss_log
                
                if cfg.point_mesh_distance_loss_weight > 0.0:
                    step_point_mesh_distance_loss_log = point_mesh_distance_loss.detach().item()
                    loss_dict["point_mesh_distance_loss_weight"] = (
                        cfg.point_mesh_distance_loss_weight
                    )
                    loss_dict["point_mesh_distance_loss"] = step_point_mesh_distance_loss_log
                # import gc
                # gc.collect()
                # accelerator.free_memory()
                # torch.cuda.empty_cache()

            # Logging
            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                tb_tracker.add_scalar(
                    tag="train loss", scalar_value=train_loss, global_step=global_step
                )
                train_loss = 0.0
                # 　save checkpoints
                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        unwrap_model = accelerator.unwrap_model(model)
                        save_path = os.path.join(
                            save_dir, "checkpoints", f"checkpoint-{global_step}"
                        )
                        delete_additional_ckpt(save_dir, 6)
                        accelerator.save_state(save_path)

                        # unwrap_model = accelerator.unwrap_model(gaussian_unet)

                        save_checkpoint(
                            unwrap_model.gaussian_unet,
                            f"{save_dir}/saved_models",
                            "gaussian_unet",
                            global_step,
                            total_limit=None,
                        )

                        save_checkpoint(
                            unwrap_model.reference_unet,
                            f"{save_dir}/saved_models",
                            "reference_unet",
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
                                val_batch[key] = val_batch[key].to(accelerator.device)

                        with torch.no_grad():
                            log_validation(
                                batch=val_batch,
                                val_image_prompt_embeds=val_image_prompt_embeds,
                                model=model,
                                scheduler=val_noise_scheduler,
                                accelerator=accelerator,
                                global_step=global_step,
                                save_dir=f"{save_dir}/validation/checkpoint-{global_step}",
                                num_inference_steps=20,
                                guidance_scale=1.0,
                                sphere_points=sphere_points,
                                corrs_1_to_2=corrs_1_to_2,
                            )

            # logs = {
            #     "step_loss": step_loss_log,  # loss.detach().item(),
            #     "lr": lr_scheduler.get_last_lr()[0],
            #     "stage": 1,
            # }
            logs = loss_dict
            logs.update(
                {
                    "lr": lr_scheduler.get_last_lr()[0],
                    "stage": 1,
                }
            )

            # save memory
            del (
                batch,
                model_pred,
                incomplete_map,
                complete_map,
                noisy_latents,
                target,
                # image_prompt_embeds,
            )
            if accelerator.is_main_process:
                wandb.log(logs)

            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
            # if accelerator.is_main_process:
            #     print(torch.cuda.max_memory_allocated() / 1024**2, "MB used this step")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":  #
    """
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --debug --num_processes 8 --main_process_port 29500 train_conditional_unet_w_chamfer_loss.py
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --debug --num_processes 2 --main_process_port 29500 train_conditional_unet_w_chamfer_loss.py
    CUDA_VISIBLE_DEVICES=0 accelerate launch --debug --num_processes 1 --main_process_port 29500 train_conditional_unet_w_chamfer_loss.py

    # for debugging (pdb)
    CUDA_VISIBLE_DEVICES=0 python train_conditional_unet_w_chamfer_loss.py

    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config/train_condition_unet_w_chamfer_loss.yaml",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
    )
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    config.wandb_tag = args.tag
    main(config)

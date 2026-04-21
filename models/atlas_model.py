import torch
import torch.nn as nn
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_2d_blocks import *
from typing import Optional, Tuple, Union
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)

class AtlasModel(nn.Module):  #
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        gaussian_unet: Union[UNet2DConditionModel],
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.gaussian_unet = gaussian_unet

        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        """
        self.guidance_types = []
        self.guidance_input_channels = []

        for guidance_type, guidance_module in guidance_encoder_group.items():
            setattr(self, f"guidance_encoder_{guidance_type}", guidance_module)
            self.guidance_types.append(guidance_type)
            self.guidance_input_channels.append(guidance_module.guidance_input_channels)
        """

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        # multi_guidance_cond,
        uncond_fwd: bool = False,
    ):
        # multi_guidance_cond [b=1, 3, 1, 512, 512] (yj)
        # self.guidance_input_channels [3, 3, 3, 3] (original champ)

        """
        guidance_cond_group = torch.split(multi_guidance_cond, self.guidance_input_channels, dim=1)


        guidance_fea_lst = []
        for guidance_idx, guidance_cond in enumerate(guidance_cond_group):
            guidance_encoder = getattr(
                self, f"guidance_encoder_{self.guidance_types[guidance_idx]}"
            )
            # guidance_cond [b=1, 3, 1, 768, 768]
            # guidance_fea [b=1, 320, 1, 96, 96] (stage1)
            # guidance_fea [b=1, 320, 24, 64, 64] (stage2)


            # guidance_cond [2, 256, 1, 512, 512]
            # guidance_fea [2, 320, 1, 64, 64]
            guidance_fea = guidance_encoder(guidance_cond)

            guidance_fea_lst += [guidance_fea]

        # torch.stack(guidance_fea_lst, dim=0) [1, 1, 320, 1, 64, 64]
        # guidance_fea [1, 320, 1, 64, 64] (stage 1)
        # guidance_fea [1, 320, 24, 64, 64] (stage 2)
        guidance_fea = torch.stack(guidance_fea_lst, dim=0).sum(0)
        """
        if not uncond_fwd:  # False #
            ref_timesteps = torch.zeros_like(timesteps)

            # original s3 training with batch size 1 video length 15
            # ref_image_latents [15, 4, 64, 64]
            # clip_image_embeds [1, 1, 768]

            if ref_image_latents.shape[0] != clip_image_embeds.shape[0]:
                # clip_image_embeds_tmp = clip_image_embeds.expand(ref_image_latents.shape[0],-1,-1)
                clip_image_embeds_tmp = clip_image_embeds.repeat(
                    ref_image_latents.shape[0] // clip_image_embeds.shape[0], 1, 1
                )
            else:
                clip_image_embeds_tmp = clip_image_embeds
            self.reference_unet(
                ref_image_latents,  # [1, 4, 96, ,96], incomplete map
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds_tmp,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.gaussian_unet(
            noisy_latents,
            timesteps,
            # guidance_fea=guidance_fea,
            encoder_hidden_states=clip_image_embeds[:, None],
        ).sample

        return model_pred

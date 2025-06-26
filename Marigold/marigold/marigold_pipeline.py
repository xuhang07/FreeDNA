# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-05-24
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
from typing import Dict, Optional, Union
from scipy.spatial.distance import pdist
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
    DDIMInverseScheduler,
)
from diffusers.utils import BaseOutput

from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.attention import Attention as CrossAttention, FeedForward, AdaLayerNorm

from src.util.multi_res_noise import multi_res_noise_like
from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)
import torch.nn as nn
import torch.nn.functional as F
def kitti_benchmark_crop(input_img):
    """
    Crop images to KITTI benchmark size
    Args:
        `input_img` (torch.Tensor): Input image to be cropped.

    Returns:
        torch.Tensor:Cropped image.
    """
    KB_CROP_HEIGHT = 352
    KB_CROP_WIDTH = 1216

    height, width = input_img.shape[-2:]
    top_margin = int(height - KB_CROP_HEIGHT)
    left_margin = int((width - KB_CROP_WIDTH) / 2)
    if 2 == len(input_img.shape):
        out = input_img[
            top_margin : top_margin + KB_CROP_HEIGHT,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            top_margin : top_margin + KB_CROP_HEIGHT,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    elif 4 == len(input_img.shape):
        out = input_img[
            :,:,
            top_margin : top_margin + KB_CROP_HEIGHT,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    return out


class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]
    uncertainty: Union[None, np.ndarray]


class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        scale_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are scale-invariant. This value must be set in
            the model config. When used together with the `shift_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        shift_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are shift-invariant. This value must be set in
            the model config. When used together with the `scale_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        latent : Optional[list] = None,
        is_IN: Optional[bool] = False,
        t_change:int=1000,
        train_noise_variance: Optional[np.ndarray] = None,
        adapter: Optional[nn.Module] = None,
        mask: Optional[np.ndarray] = None,
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            scale_invariant (`str`, *optional*, defaults to `True`):
                Flag of scale-invariant prediction, if True, scale will be adjusted from the raw prediction.
            shift_invariant (`str`, *optional*, defaults to `True`):
                Flag of shift-invariant prediction, if True, shift will be adjusted from the raw prediction, if False, near plane will be fixed at 0m.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1
        if mask is not None:
            match_input_res = True
        else:
            match_input_res = False

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )
        if mask is not None:
            mask = torch.tensor(mask).bool().to(self.device).unsqueeze(1)
            invalid_mask = ~mask
            valid_mask_down = ~torch.max_pool2d(
                            invalid_mask.float(), 8, 8
                        ).bool()
            valid_mask_down = valid_mask_down.repeat((1, 4, 1, 1))
        else:
            valid_mask_down = None
        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        # assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict depth maps (batched)
        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            depth_pred_raw, img_list= self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
                latent = latent,
                is_IN = is_IN,
                t_change=t_change,
                train_noise_variance = train_noise_variance,
                adapter = adapter,
                mask = valid_mask_down,
            )
            depth_pred_ls.append(depth_pred_raw.detach())
        depth_preds = torch.concat(depth_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depth(
                depth_preds,
                scale_invariant=self.scale_invariant,
                shift_invariant=self.shift_invariant,
                max_res=50,
                **(ensemble_kwargs or {}),
            )
        else:
            depth_pred = depth_preds
            pred_uncert = None
       #  print(depth_pred.size())
        # Resize back to original resolution
        # pred_uncert = None
        match_input_res = True
        if match_input_res:
            depth_pred = resize(
                depth_preds,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )

        # Convert to numpy
        depth_pred = depth_pred.squeeze()
        depth_pred = depth_pred.cpu().numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        # Clip output range
       #  depth_pred = depth_pred.clip(0, 1)

        # Colorize
        color_map = 'Spectral'
        if color_map is not None:
            depth_colored = colorize_depth_maps(
                depth_pred, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)
        else:
            depth_colored_img = None

        return MarigoldDepthOutput(
            depth_np=depth_pred,
            depth_colored=depth_colored_img,
            uncertainty=pred_uncert,
        ),img_list

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        latent: Optional[list] = None,
        is_IN: Optional[bool] = False,
        t_change: int = 1000,
        train_noise_variance: Optional[np.ndarray] = None,
        adapter: Optional[nn.Module] = None,
        mask: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        IN = torch.nn.InstanceNorm2d(3)
        # shift = torch.tensor([-0.23353664,-0.22471737,-0.37908713]).view(1,3,1,1)
        # scale = torch.tensor([0.29039687,0.28956587,0.29296139]).view(1,3,1,1)
        # rgb_in = IN(rgb_in)*scale+shift
        rgb_in = rgb_in.to(device)
        # adapter = None
        if adapter is not None:
            rgb_in = adapter(rgb_in)
        if(is_IN):
            rgb_in_IN = IN(rgb_in)
            
            rgb_in_IN = torch.clamp(rgb_in_IN,-1,1)
        else:
            rgb_in_IN = rgb_in
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in).repeat(1,1,1,1)
        rgb_IN_latent = self.encode_rgb(rgb_in_IN)
        # Initial depth map (noise)
        # torch.manual_seed(42)
        depth_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]
        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]
        # Denoising loop
        H,W = depth_latent.shape[2:]
        def norm(noise):
            return (noise-noise.mean(dim=(1,2,3),keepdim=True))/noise.std(dim=(1,2,3),keepdim=True)
        
        # noise_1,noise_2,noise_3,noise_4 = depth_latent[:,:,0:H//2,0:W//2],depth_latent[:,:,0:H//2,W//2:],depth_latent[:,:,H//2:H,0:W//2],depth_latent[:,:,H//2:H,W//2:]
        # noise_norm_1,noise_norm_2,noise_norm_3,noise_norm_4 = norm(noise_1),norm(noise_2),norm(noise_3),norm(noise_4)
        # dists_1 = pdist(noise_norm_1.contiguous().view(noise_norm_1.shape[0],-1).cpu().numpy(),metric='euclidean').mean()
        # dists_2 = pdist(noise_norm_2.contiguous().view(noise_norm_2.shape[0],-1).cpu().numpy(),metric='euclidean').mean()
        # dists_3 = pdist(noise_norm_3.contiguous().view(noise_norm_3.shape[0],-1).cpu().numpy(),metric='euclidean').mean()
        # dists_4 = pdist(noise_norm_4.contiguous().view(noise_norm_4.shape[0],-1).cpu().numpy(),metric='euclidean').mean()

        # if(dists_1<dists_2 and dists_1<dists_3 and dists_1<dists_4):
        #     scale_noise = noise_1.std()/depth_latent.std()
        # elif(dists_2<dists_1 and dists_2<dists_3 and dists_2<dists_4):
        #     scale_noise = noise_2.std()/depth_latent.std()
        # elif(dists_3<dists_1 and dists_3<dists_2 and dists_3<dists_4):
        #     scale_noise = noise_3.std()/depth_latent.std()
        # elif(dists_4<dists_1 and dists_4<dists_2 and dists_4<dists_3):
        #     scale_noise = noise_4.std()/depth_latent.std()
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)
        # rgb_latent = torch.cat([rgb_latent,rgb_IN_latent],dim=1)
        # scale = torch.tensor([1,1])
        scale = 1
        scale_sum=0
        scale_noise = 1
        delta_N_0 = None
        delta_N_1 = None
        variance_list = []
        for i, t in iterable:
            # if(t<t_change):
            #     unet_input = torch.cat(
            #         [rgb_latent, depth_latent], dim=1
            #     )  # this order is important
            # else:
            #     unet_input = torch.cat(
            #         [rgb_IN_latent, depth_latent], dim=1
            #     )
            
            unet_input = torch.cat(
                    [rgb_latent, depth_latent], dim=1
                )
            # print(unet_input.size())
            # predict the noise residual
            v_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]
            # compute the previous noisy sample x_t -> x_t-1
            prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.final_alpha_cumprod
            beta_prod_t = 1 - alpha_prod_t
            # pred_original_sample = (alpha_prod_t**0.5) * depth_latent - (beta_prod_t**0.5) * v_pred
            pred_epsilon = (alpha_prod_t**0.5) * v_pred + (beta_prod_t**0.5) * depth_latent
            pred_original_sample = (depth_latent-beta_prod_t**0.5*pred_epsilon)/alpha_prod_t**0.5
            # scale = 1

            if(train_noise_variance is not None and i<50):
                delta_N_1 = np.sqrt(np.mean(pred_epsilon.cpu().numpy()**2)/(train_noise_variance[i])).item()
                if delta_N_0 is not None:
                    scale = delta_N_1-delta_N_0+1-scale_sum
                    # scale/=scale_noise
                else:
                    scale = 1
                print(1./scale)
                scale_sum+=scale-1
                delta_N_0 = delta_N_1
            if mask is not None and i<48:
                noise_1,noise_2,noise_3,noise_4 = pred_epsilon[:,:,0:H//2,0:W//2],pred_epsilon[:,:,0:H//2,W//2:],pred_epsilon[:,:,H//2:H,0:W//2],pred_epsilon[:,:,H//2:H,W//2:]
                noise_norm_1,noise_norm_2,noise_norm_3,noise_norm_4 = norm(noise_1),norm(noise_2),norm(noise_3),norm(noise_4)
                dists_1 = pdist(noise_norm_1.contiguous().view(noise_norm_1.shape[0],-1).cpu().numpy(),metric='euclidean').mean()
                dists_2 = pdist(noise_norm_2.contiguous().view(noise_norm_2.shape[0],-1).cpu().numpy(),metric='euclidean').mean()
                dists_3 = pdist(noise_norm_3.contiguous().view(noise_norm_3.shape[0],-1).cpu().numpy(),metric='euclidean').mean()
                dists_4 = pdist(noise_norm_4.contiguous().view(noise_norm_4.shape[0],-1).cpu().numpy(),metric='euclidean').mean()

                if(dists_1<dists_2 and dists_1<dists_3 and dists_1<dists_4):
                    scale_noise = noise_1.std()/depth_latent.std()
                elif(dists_2<dists_1 and dists_2<dists_3 and dists_2<dists_4):
                    scale_noise = noise_2.std()/depth_latent.std()
                elif(dists_3<dists_1 and dists_3<dists_2 and dists_3<dists_4):
                    scale_noise = noise_3.std()/depth_latent.std()
                elif(dists_4<dists_1 and dists_4<dists_2 and dists_4<dists_3):
                    scale_noise = noise_4.std()/depth_latent.std()
                delta_N_1 = np.sqrt(np.mean(pred_original_sample.cpu().numpy()**2)/np.mean(pred_original_sample[mask[i].unsqueeze(0).repeat(pred_original_sample.shape[0],1,1,1)==1].cpu().numpy()**2)).item()
                if delta_N_0 is not None:
                    scale = delta_N_1-delta_N_0+1-scale_sum
                    scale/=scale_noise
                else:
                    scale = 1
                print(1./scale,scale_noise)
                scale_sum+=(scale-1)
                delta_N_0 = delta_N_1
                # if i ==20:
                #     break

                # scale=0.994
            pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
            prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample*1.02+ pred_sample_direction
            depth_latent = prev_sample
            if i % 10 == 0:
                variance_list.append(pred_original_sample)
            # depth_latent = self.scheduler.step(
            #     v_pred, t, depth_latent, generator=generator
            # ).prev_sample
            # depth_latent = (depth_latent/depth_latent.std()-depth_latent.mean())*ideal_latent.std()+ideal_latent.mean()
            # print((depth_latent-ideal_latent).mean())
        
        # depth = depth_latent
        depth = self.decode_depth(depth_latent)
        for i in range(len(variance_list)):
            variance_list[i] = ((torch.clip(self.decode_depth(variance_list[i]),-1.0,1.0)+1.0)/2.0).cpu().numpy()
        # depth = self.decode_depth(pred_original_sample)
        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0

        return depth,variance_list

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        # depth_mean = stacked
        return depth_mean
    @torch.no_grad()
    def inversion(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        depth_image: np.ndarray,
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        save_latent: bool = False,
        ensemble_kwargs: Dict = None,
    ) -> list:
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"
        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        depth_image = torch.tensor(depth_image).unsqueeze(0).unsqueeze(0)
        depth_image = resize_max_res(
                depth_image,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )
        
        def scale_shift_depth(depth_raw):
                    valid_mask = (depth_raw>=1e-5) &(depth_raw<=80.0)
                    q2 = torch.quantile(depth_raw[valid_mask],0.02)
                    q98 = torch.quantile(depth_raw[valid_mask],0.98)
                    depth_norm = (depth_raw-q2)/(q98-q2)*2-1
                    depth_norm = depth_norm.clip(-1,1)
                    return depth_norm,valid_mask

        depth_image,valid_mask = scale_shift_depth(depth_image)
        depth_image = depth_image.repeat((1,3,1,1))
        # label_image = torch.nn.functional.interpolate(label_image, size=rgb_norm.shape[2:], mode='bilinear', align_corners=False)
        depth_image = depth_image.to(self.device)

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        ensemble_size=1
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict depth maps (batched)
        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            latent_list = self.single_inversion(
                rgb_in=batched_img,
                latents = depth_image,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
                save_latent = save_latent,
                mask = valid_mask
            )
        torch.cuda.empty_cache()  # clear vram cache for ensembling
        if (save_latent):
            return latent_list
        
    @torch.no_grad()
    def single_inversion(
        self,
        rgb_in: torch.Tensor,
        latents: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        save_latent: bool = False,
        mask: torch.Tensor = None,
    ) -> list:
        device = self.device
        rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)
        latents = self.encode_rgb(latents)
        mask = mask.bool().to(self.device)
        invalid_mask = ~mask
        valid_mask_down = ~torch.max_pool2d(
                        invalid_mask.float(), 8, 8
                    ).bool()
        valid_mask_down = valid_mask_down.repeat((1, 4, 1, 1))
        # Initial depth map (noise)
        # depth_latent = torch.randn(
        #     rgb_latent.shape,
        #     device=device,
        #     dtype=self.dtype,
        #     generator=generator,
        # )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]

        # Denoising loop

            
        self.scheduler.set_timesteps(num_inference_steps,device=device)
        timesteps = reversed(self.scheduler.timesteps)
        # inverscheduler = DDIMInverseScheduler.from_pretrained("/home/xuhang/code/Marigold/weights/marigold-depth-v1-0/scheduler/scheduler_config.json")
        # inverscheduler.set_timesteps(num_inference_steps,device=device)
        latent_list = []
        latent_list.append(latents)
        # for module in self.unet.modules():
        #     if(isinstance(module,CrossAttention)):
        #         print(f"Module type: {type(module)}")
        #         print("-" * 50)
        # exit()
        for i in tqdm(range(num_inference_steps)):
            t = timesteps[i]
            # print(rgb_latent.shape,latents.shape)
            unet_input = torch.cat(
                [rgb_latent, latents], dim=1
            )  # this order is important
            v_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample
            current_t = t#t
            next_t = min(999, t.item() + (1000//num_inference_steps))# min(999, t.item() + (1000//num_inference_steps)) # t+1
            alpha_t = self.scheduler.alphas_cumprod[current_t]
            alpha_t_next = self.scheduler.alphas_cumprod[next_t]
            noise_pred = alpha_t.sqrt()*v_pred+(1-alpha_t).sqrt()*latents
            latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
            # latents = inverscheduler.step(
            #     noise_pred, t, latents
            # ).prev_sample

            # current_t = t
            # alpha_t = self.scheduler.alphas_cumprod[current_t]
            # depth_latent = torch.randn(
            # rgb_latent.shape,
            # device=device,
            # dtype=self.dtype,
            # generator=generator,
            # )  # [B, 4, h, w]
            # latent = latents*alpha_t.sqrt()+depth_latent*(1-alpha_t).sqrt()

            if(save_latent):
                latent_list.append(latents[valid_mask_down])
        if (save_latent):
            return latent_list
        else:
            return None




class ExposureBiasPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        scale_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are scale-invariant. This value must be set in
            the model config. When used together with the `shift_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        shift_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are shift-invariant. This value must be set in
            the model config. When used together with the `scale_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        depth_image: Union[Image.Image, torch.Tensor, np.ndarray],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        mask: np.ndarray = None,
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            scale_invariant (`str`, *optional*, defaults to `True`):
                Flag of scale-invariant prediction, if True, scale will be adjusted from the raw prediction.
            shift_invariant (`str`, *optional*, defaults to `True`):
                Flag of shift-invariant prediction, if True, shift will be adjusted from the raw prediction, if False, near plane will be fixed at 0m.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        # Model-specific optimal default values leading to fast and reasonable results.
        
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        ensemble_size=1
        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"
        
        # rgb = kitti_benchmark_crop(rgb)
        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        depth_image = torch.tensor(depth_image).unsqueeze(0).unsqueeze(0)
        # depth_image = kitti_benchmark_crop(depth_image)
        depth_image = resize_max_res(
                depth_image,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )
        
        def scale_shift_depth(depth_raw):
                    valid_mask = (depth_raw>=1e-5) &(depth_raw<=80.0)
                    q2 = torch.quantile(depth_raw[valid_mask],0.02)
                    q98 = torch.quantile(depth_raw[valid_mask],0.98)
                    depth_norm = (depth_raw-q2)/(q98-q2)*2-1
                    depth_norm = depth_norm.clip(-1,1)
                    return depth_norm,valid_mask

        depth_image,valid_mask = scale_shift_depth(depth_image)
        depth_image = depth_image.repeat((1,3,1,1))
        mask = (rgb.float().mean(dim=(0,1))<torch.quantile(rgb.float().mean(dim=(0,1)),0.75)) & (rgb.float().mean(dim=(0,1))>torch.quantile(rgb.float().mean(dim=(0,1)),0.25))
        # mask = ~mask
        mask = mask*valid_mask
        # print(valid_mask.sum()/valid_mask.numel())
        
        # depth_image = torch.tensor(depth_image).unsqueeze(0)
        

        depth_image = depth_image.to(self.device)
        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict depth maps (batched)
        noise_variance_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            noise_variance_list = self.single_infer(
                rgb_in=batched_img,
                depth_in = depth_image, 
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
                mask=mask,
            )
        # depth_preds = torch.concat(depth_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        return noise_variance_list

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        depth_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        latent: Optional[list] = None,
        mask: np.ndarray=None,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_in = rgb_in.to(device).repeat(4,1,1,1)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]
        IN = torch.nn.InstanceNorm2d(3)
        rgb_in_IN = IN(rgb_in)
        rgb_in_IN = torch.clamp(rgb_in_IN,-1,1)
        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)
        rgb_latent_IN = self.encode_rgb(rgb_in_IN)

        # rgb_latent = torch.cat([rgb_latent,rgb_latent_IN],dim=0)

        depth_gt_latent = self.encode_rgb(depth_in)
        # depth_gt_latent = depth_in
        mask = torch.tensor(mask).bool().to(self.device)
        invalid_mask = ~mask
        valid_mask_down = ~torch.max_pool2d(
                        invalid_mask.float(), 8, 8
                    ).bool()
        valid_mask_down = valid_mask_down.repeat((4, 4, 1, 1))
        # print(valid_mask_down.sum()/valid_mask_down.numel())
        # Initial depth map (noise)
        depth_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]
        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]
        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)
        noise_variance_list = []
        v_0_list = []
        v_1_list = []
        for i, t in iterable:
            # depth_latent = torch.randn(
            # rgb_latent.shape,
            # device=device,
            # dtype=self.dtype,
            # generator=generator,
            # )
            alpha_t = self.scheduler.alphas_cumprod[t]
            # depth_latent_gt = alpha_t.sqrt()*depth_gt_latent+(1-alpha_t).sqrt()*depth_latent
            # depth_latent_gt = self.scheduler.add_noise(depth_gt_latent,depth_latent,t)
            # if t<0:
            #     unet_input = torch.cat(
            #         [rgb_latent, depth_latent], dim=1
            #     ).float()  # this order is important
            # else:
            #     unet_input = torch.cat(
            #         [rgb_latent_IN,depth_latent],dim=1
            #     ).float()
            unet_input = torch.cat(
                [rgb_latent,depth_latent],dim=1
            ).float()
            v_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]
            prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.final_alpha_cumprod
            beta_prod_t = 1 - alpha_prod_t
            # pred_original_sample = (alpha_prod_t**0.5) * depth_latent - (beta_prod_t**0.5) * v_pred
            pred_epsilon = (alpha_prod_t**0.5) * v_pred + (beta_prod_t**0.5) * depth_latent
            pred_original_sample = (depth_latent-beta_prod_t**0.5*pred_epsilon)/alpha_prod_t**0.5
            pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
            prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample*0.994+ pred_sample_direction
            depth_latent = prev_sample
            threshold = np.clip(i/50.,0.1,0.9).item()
            pred_x = self.decode_depth(pred_original_sample)
            mask = pred_x.var(0)<torch.quantile(pred_x.var(0),threshold)
            # v_0 = noise_pred[valid_mask_down==0]
            # v_1 = noise_pred[valid_mask_down==1]
            # compute the previous noisy sample x_t -> x_t-1
            # noise_variance_list.append(np.mean(x0_pred.cpu().numpy()**2,axis=(1,2,3)))
            # v_0_list.append(np.mean((v_0.cpu().numpy())**2))
            # v_1_list.append(np.mean((v_1.cpu().numpy())**2))
            
            noise_variance_list.append(mask.cpu().numpy())
        return noise_variance_list
            

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean
    





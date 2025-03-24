"""
===========================================================
PSC Gaussian Diffusion Model for Sketch Colorization
===========================================================

This script contains the implementation of the PSC Gaussian Diffusion model used for sketch colorization. It includes the following components:
1. L2_loss: Compute the L2 loss (mean squared error) between the input and target tensors.
2. extract: Extract values from a tensor at specified indices and reshape to match a given shape.
3. PSCGaussianDiffusion: Implements the Gaussian diffusion model with PSC (Patch-based Spatial Consistency) for image denoising.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from utils.image.warp_psc import warp_reference_with_psc

def L2_loss(input, target):
    """
    Compute the L2 loss (mean squared error) between the input and target tensors.

    Args:
        input (torch.Tensor): The input tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        torch.Tensor: The computed L2 loss.
    """
    return F.mse_loss(input, target, reduction="mean")

def extract(a, t, x_shape):
    """
    Extract values from a tensor `a` at indices specified by `t` and reshape to match `x_shape`.

    Args:
        a (torch.Tensor): The tensor to extract values from.
        t (torch.Tensor): The indices to extract.
        x_shape (tuple): The shape to reshape the extracted values to.

    Returns:
        torch.Tensor: The extracted and reshaped values.
    """
    batch_size = t.shape[0]
    out = a.to(t.device).gather(0, t).float()
    out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out

class PSCGaussianDiffusion(nn.Module):
    """
    This class implements a Gaussian diffusion model with PSC (Patch-based Spatial Consistency) for image denoising.
    It uses a denoising UNet, a reference UNet, and a PSC model for feature extraction and transformation.

    Args:
        time_step (int): Number of time steps for the diffusion process.
        betas (dict): Dictionary containing the start and end values for the beta schedule.
        denoising_unet (nn.Module): The denoising UNet model.
        psc_model (nn.Module): The PSC model for patch-based spatial consistency.

    Methods:
        q_sample(x_0, t, noise): Sample from the diffusion process.
        p_sample(x_t, t, ref_feats, x_cond, eta): Sample from the reverse diffusion process.
        inference(x_t, ref_feats, x_cond, eta): Perform inference using the reverse diffusion process.
        forward(image_of_the_character, t, sketch_of_the_character, flow_warped, ref_feats): Forward pass of the diffusion model.
    """

    def __init__(self, time_step, betas, denoising_unet, psc_model):
        super().__init__()
        self.denoise_fn = denoising_unet
        self.psc_model = psc_model
        self.time_steps = time_step

        # Beta schedule
        scale = 1000 / self.time_steps
        betas = torch.linspace(
            scale * betas['linear_start'],
            scale * betas['linear_end'],
            self.time_steps,
            dtype=torch.float32
        )
        alphas = 1. - betas
        gammas = torch.cumprod(alphas, axis=0)
        gammas_prev = F.pad(gammas[:-1], (1,0), value=1.)

        # Register buffers
        self.register_buffer('gammas', gammas)
        self.register_buffer('sqrt_reciprocal_gammas', torch.sqrt(1. / gammas))
        self.register_buffer('sqrt_reciprocal_gammas_m1', torch.sqrt(1. / gammas - 1))

        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(gammas_prev) / (1. - gammas))
        self.register_buffer('posterior_mean_coef2', (1. - gammas_prev) * torch.sqrt(alphas) / (1. - gammas))

        self.loss_fn = partial(F.mse_loss, reduction="sum")

    def q_sample(self, x_0, t, noise=None):
        """
        Sample from the diffusion process. q(x_t | x_0)

        Args:
            x_0 (torch.Tensor): The original image tensor.
            t (torch.Tensor): The time step tensor.
            noise (torch.Tensor, optional): The noise tensor. Default is None.

        Returns:
            torch.Tensor: The noisy image tensor.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        gammas_t = extract(self.gammas, t, x_0.shape).to(x_0.device)
        return torch.sqrt(gammas_t) * x_0 + torch.sqrt(1 - gammas_t) * noise

    @torch.no_grad()
    def p_sample(self, x_t, t, ref_feats, x_cond=None, eta=1):
        """
        Sample from the reverse diffusion process.

        Args:
            x_t (torch.Tensor): The noisy image tensor.
            t (torch.Tensor): The time step tensor.
            ref_feats (torch.Tensor): The reference features tensor.
            x_cond (torch.Tensor, optional): The conditioning tensor. Default is None.
            eta (float, optional): The noise scale factor. Default is 1.

        Returns:
            torch.Tensor: The denoised image tensor.
        """
        if x_cond is not None:
            # Concatenate x_t and x_cond along the channel dimension
            x_in = torch.cat([x_t, x_cond], dim=1)
            predicted_noise = self.denoise_fn(x_in, t, ref_feats=ref_feats)
        else:
            predicted_noise = self.denoise_fn(x_t, t, ref_feats=ref_feats)
        predicted_x_0 = extract(self.sqrt_reciprocal_gammas, t, x_t.shape) * x_t - extract(self.sqrt_reciprocal_gammas_m1, t, x_t.shape) * predicted_noise
        predicted_x_0 = torch.clamp(predicted_x_0, min=-1., max=1.)
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * predicted_x_0 + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_log_variance = extract(self.posterior_log_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero_mask = eta * ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        return posterior_mean + nonzero_mask * (0.5 * posterior_log_variance).exp() * noise
    
    @torch.no_grad()
    def inference(self, x_t, ref_feats, x_cond=None, eta=1):
        """
        Perform inference using the reverse diffusion process.

        Args:
            x_t (torch.Tensor): The noisy image tensor.
            ref_feats (torch.Tensor): The reference features tensor.
            x_cond (torch.Tensor, optional): The conditioning tensor. Default is None.
            eta (float, optional): The noise scale factor. Default is 1.

        Returns:
            list of torch.Tensor: The list of denoised image tensors at each time step.
        """
        batch_size = 1
        device = next(self.parameters()).device
        ret = []
        for i in reversed(range(0, self.time_steps)):
            x_t = self.p_sample(x_t=x_t, t=torch.full((batch_size, ), i, device=device, dtype=torch.long), ref_feats=ref_feats, x_cond=x_cond, eta=eta)
            ret.append(x_t.cpu())
        return ret

    def forward(self, image_of_the_character, t, sketch_of_the_character, flow_warped, ref_feats):
        """
        Forward pass of the diffusion model.

        Args:
            image_of_the_character (torch.Tensor): The original reference image tensor.
            t (torch.Tensor): The time step tensor.
            sketch_of_the_character (torch.Tensor): The sketch image tensor.
            flow_warped (torch.Tensor): The warped reference image tensor.
            ref_feats (torch.Tensor): The reference features tensor.

        Returns:
            torch.Tensor: The computed diffusion loss.
        """
        # noise
        noise = torch.randn_like(image_of_the_character)

        # q_sample => x_noisy
        x_noisy = self.q_sample(image_of_the_character, t, noise)

        # build a 6-channel condition from (sketch, warped_ref)
        x_cond = torch.cat([sketch_of_the_character, flow_warped], dim=1) # Conditioning on the ref and the deformation flow

        # cat x_noisy [B,3,H,W] with x_cond [B,6,H,W] => (B,9,H,W)
        x_in = torch.cat([x_noisy, x_cond], dim=1)

        # pass to denoising net
        noise_tilde = self.denoise_fn(x_in, t, ref_feats=ref_feats)

        # standard diffusion loss on noise
        diff_loss = self.loss_fn(noise, noise_tilde)

        return diff_loss

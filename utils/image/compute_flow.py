"""
===========================================================
Optical Flow Computation Utilities
===========================================================

This script contains utility functions for computing and visualizing optical flow between images. It includes the following functions:
1. flow_to_rgb: Convert optical flow to an RGB image.
2. compute_flow: Compute the optical flow between a sketch and a reference image using a given model.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.colors as mcolors

def flow_to_rgb(flow):
    """
    Convert optical flow to an RGB image.

    Args:
        flow (torch.Tensor): Optical flow tensor of shape (B, H, W, 2) or (H, W, 2).

    Returns:
        np.ndarray: RGB image representing the optical flow.
    """
    # Move to CPU + NumPy
    flow_np = flow.detach().cpu().numpy()  
    
    # If flow has a batch dimension (B=1), squeeze it out
    if flow_np.ndim == 4:
        # shape is (B, H, W, 2) -> (H, W, 2)
        flow_np = flow_np[0]
    
    fx, fy = flow_np[..., 0], flow_np[..., 1]
    mag = np.sqrt(fx**2 + fy**2)
    ang = np.arctan2(fy, fx) + np.pi
    
    import cv2
    hsv = np.zeros((flow_np.shape[0], flow_np.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * (180.0 / np.pi / 2.0)  # Hue
    hsv[..., 1] = 255                         # Saturation
    if np.max(mag) < 1e-5:                    # Avoid /0
        hsv[..., 2] = 0
    else:
        hsv[..., 2] = np.clip(mag * 255 / np.max(mag), 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def compute_flow(model, sketch_of_the_character_tenseur, other_image_of_the_character_tenseur, image_size=256):
    """
    Compute the optical flow between a sketch and a reference image using a given model.

    Args:
        model (torch.nn.Module): Model used to compute the flow.
        sketch_of_the_character_tenseur (torch.Tensor): Sketch image tensor.
        other_image_of_the_character_tenseur (torch.Tensor): Reference image tensor.
        image_size (int): Size of the output image.

    Returns:
        np.ndarray: RGB image representing the optical flow.
    """
    with torch.no_grad():
        photo  = other_image_of_the_character_tenseur.cuda(non_blocking=True)
        sketch = sketch_of_the_character_tenseur.cuda(non_blocking=True)
        if photo.dim() == 3:
            photo = photo.unsqueeze(0)
        if sketch.dim() == 3:
            sketch = sketch.unsqueeze(0)
        _, photo_res = model.encoder_q(photo,   cond=0, return_map=True)
        _, sketch_res= model.encoder_q(sketch, cond=1, return_map=True)
        fwd_flow, bwd_flow = model.forward_stn(photo_res, sketch_res)
        fwd_flow = F.interpolate(
            fwd_flow.permute(0,3,1,2),
            size=(image_size, image_size),
            mode="bilinear", align_corners=True
        ).permute(0,2,3,1)
        photo_warped = F.grid_sample(
            photo, fwd_flow, 
            mode='bilinear', padding_mode='border', align_corners=True
        )
        return flow_to_rgb(fwd_flow)

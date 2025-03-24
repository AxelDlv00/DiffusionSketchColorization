"""
===========================================================
Warping Utilities Using PSCNet
===========================================================

This script contains utility functions for warping reference images to align with sketches using PSCNet. It includes the following function:
1. warp_reference_with_psc: Warp a reference image to align with a sketch using PSCNet.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import torch
import torch.nn.functional as F

def warp_reference_with_psc(sketch, reference, psc_model):
    """
    Given a sketch and reference image (both tensors of shape (B, 3, H, W)),
    use PSCNet to compute a forward flow and warp the reference image to align with the sketch.

    Args:
        sketch (torch.Tensor): Sketch image tensor of shape (B, 3, H, W).
        reference (torch.Tensor): Reference image tensor of shape (B, 3, H, W).
        psc_model (torch.nn.Module): PSCNet model used to compute the flow.

    Returns:
        torch.Tensor: Warped reference image tensor of shape (B, 3, H, W).
    """
    with torch.no_grad():
        # PSCNet expects 3-channel inputs and uses a 'cond' flag
        _, ref_map = psc_model.encoder_q(reference, cond=0, return_map=True)
        _, sketch_map = psc_model.encoder_q(sketch, cond=1, return_map=True)
        fwd_flow, _ = psc_model.forward_stn(ref_map, sketch_map)
        # Resize flow to match the sketch/reference dimensions
        fwd_flow = F.interpolate(
            fwd_flow.permute(0, 3, 1, 2),
            size=sketch.shape[-2:],
            mode="bilinear",
            align_corners=True
        ).permute(0, 2, 3, 1)
        warped_reference = F.grid_sample(
            reference, fwd_flow,
            mode='bilinear', padding_mode='border', align_corners=True
        )
    return warped_reference


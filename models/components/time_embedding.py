"""
===========================================================
Time Embedding Utility
===========================================================

This script contains a utility function for generating sinusoidal time embeddings. It includes the following function:
1. time_embedding: Generate a sinusoidal time embedding for a given time step.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import torch
import math

def time_embedding(time_step, dimension, max_period=1000):
    """
    Generate a sinusoidal time embedding for a given time step.

    Args:
        time_step (torch.Tensor): Tensor of shape [N], one per batch element.
        dimension (int): The dimension of the output embedding.
        max_period (int): Maximum period for the sinusoidal embedding.

    Returns:
        torch.Tensor: Tensor of shape [N, dimension] containing the time embeddings.
    """
    half = dimension // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=time_step.device)
    args = time_step[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dimension % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
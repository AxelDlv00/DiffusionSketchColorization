"""
===========================================================
Reference UNet for Sketch Colorization
===========================================================

This script contains the implementation of the Reference UNet model used for encoding reference images in the context of anime diffusion models. It includes the following class:
1. ReferenceUNet: Implements a U-Net-like architecture with residual blocks and attention mechanisms.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.residual_block import ResidualBlock
from models.attention import RefUNetAttentionBlock
from utils.image import progressive_patch
from torchvision import transforms as T

class ReferenceUNet(nn.Module):
    """
    This class implements a U-Net-like architecture with residual blocks and attention mechanisms.
    It is used for encoding reference images in the context of anime diffusion models.

    Args:
    - in_channels (int): Number of input channels (typically 3 for RGB).
    - base_ch (int): Number of base channels.
    - num_res_blocks (int): Number of residual blocks per level.
    - num_attn_blocks (int): Number of attention blocks in the bottleneck.
    - cbam (bool): Whether to use CBAM (Convolutional Block Attention Module). Default is False.
    - device (str): Device to run the model on. Default is 'cuda'.

    Methods:
    - set_to_shuffle(to_shuffle): Set whether to apply patch shuffle.
    - forward(ref_img, patch_size): Forward pass of the network.

    Example:
    >>> model = ReferenceUNet(in_channels=3, base_ch=64, num_res_blocks=3, num_attn_blocks=3)
    >>> output = model(input_tensor)
    """

    def __init__(self, in_channels=3, base_ch=64, num_res_blocks=3, num_attn_blocks=3, cbam=False, device=torch.device('cuda')):
        super().__init__()

        # Level 1: Initial encoding with several residual blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            *[ResidualBlock(base_ch, base_ch, cbam=cbam) for _ in range(num_res_blocks)]
        )
        
        # Level 2: Downsampling and encoding
        self.down1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1),  # Downsample by 2
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            *[ResidualBlock(base_ch * 2, base_ch * 2, cbam=cbam) for _ in range(num_res_blocks)]
        )
        
        # Level 3: Downsampling and encoding
        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1),  # Downsample by 2
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            *[ResidualBlock(base_ch * 4, base_ch * 4, cbam=cbam) for _ in range(num_res_blocks)]
        )
        
        # Skip connection: project x2 to add to x3
        self.skip_conv = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=1) # TO REMOVE ? 
        
        # Attention block(s) in the bottleneck
        self.attn_blocks = nn.Sequential(
            *[RefUNetAttentionBlock(base_ch * 4, num_heads=4, dropout=0.1) for _ in range(num_attn_blocks)]
        )
        
        # Final refinement layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True)
        )

        self.device = device

    def forward(self, ref_patch):
        """
        Forward pass of the network.

        Args:
        - ref_img (PIL Image or Tensor): Reference image.
        - patch_size (int): Size of the patches for patch shuffle.

        Returns:
        - torch.Tensor: Encoded features.
        """
        x = ref_patch.to(self.device)
    
        # Encoding levels
        x1 = self.enc1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # Skip connection
        skip = self.skip_conv(x2)
        skip = F.avg_pool2d(skip, kernel_size=2)
        x3 = x3 + skip

        # Attention blocks
        x3 = self.attn_blocks(x3)

        # Final refinement
        feats = self.final_conv(x3)
        return feats

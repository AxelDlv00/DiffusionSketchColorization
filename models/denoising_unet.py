"""
===========================================================
Denoising UNet for Sketch Colorization
===========================================================

This script contains the implementation of the Denoising UNet model used for sketch colorization. It includes the following classes:
1. SampleBlock: Implements a sampling block for upsampling or downsampling.
2. DenoisingUNet: Implements the Denoising UNet with residual blocks and cross-attention mechanisms.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.xt_module import Module, Sequential
from models.residual_block import ResidualBlock
from models.attention import CrossAttentionBlock


class SampleBlock(Module):
    """
    This class implements a sampling block for upsampling or downsampling.
    It is used in the context of the Denoising UNet.

    Args:
    - sampling_type (str, optional): Type of sampling ('up' for upsampling, 'down' for downsampling). Default is None.

    Methods:
    - forward(x, t): Forward pass of the sampling block.

    Example:
    >>> sample_block = SampleBlock(sampling_type="up")
    >>> output = sample_block(input_tensor)
    """

    def __init__(self, sampling_type=None):
        super().__init__()

        # Member variables
        if sampling_type == "up":
            self.sampling = nn.Upsample(scale_factor=2, mode="nearest")
        elif sampling_type == "down":
            self.sampling = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.sampling = nn.Identity()

    def forward(self, x, t=None):
        return self.sampling(x)


class DenoisingUNet(nn.Module):
    """
    This class implements a Denoising UNet with residual blocks and cross-attention mechanisms.
    It is used for denoising images in the context of anime diffusion models.

    Args:
    - channel_in (int): Number of input channels.
    - channel_out (int, optional): Number of output channels. Default is None.
    - channel_base (int): Number of base channels.
    - channel_features (int): Dimension of the reference features for cross-attention.
    - n_res_blocks (int): Number of residual blocks per level.
    - dropout (float): Dropout rate.
    - channel_mult (tuple): Multipliers for the number of channels at each level.
    - attention_head (int): Number of attention heads.
    - cbam (bool): Whether to use CBAM (Convolutional Block Attention Module). Default is True.

    Methods:
    - forward(x, t, ref_feats): Forward pass of the network.

    Example:
    >>> model = DenoisingUNet(channel_in=3, channel_out=3, channel_base=64, channel_features=64)
    >>> output = model(input_tensor, t_tensor, ref_feats_tensor)
    """

    def __init__(self, channel_in, channel_out=None, channel_base=64, channel_features=64,
                 n_res_blocks=2, dropout=0, channel_mult=(1, 2, 4, 8), attention_head=4, cbam=True):
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out or channel_in
        self.channel_base = channel_base
        self.channel_features = channel_features  # Expected dimension of ref_feats in the cross-attention

        # Time embedding layer
        time_embedding_channel = channel_base * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(self.channel_base, time_embedding_channel),
            nn.SiLU(),
            nn.Linear(time_embedding_channel, time_embedding_channel)
        )

        # Input layer
        self.input = nn.Sequential(
            nn.Conv2d(self.channel_in, self.channel_base, kernel_size=1)
        )

        # Encoder blocks
        channel_sequence = [channel_base]
        ch = self.channel_base
        self.encoder_block = nn.ModuleList()
        for l, mult in enumerate(channel_mult):
            for _ in range(n_res_blocks):
                self.encoder_block.append(
                    ResidualBlock(ch, mult * self.channel_base, time_channel=time_embedding_channel, dropout=dropout)
                )
                ch = mult * self.channel_base
                channel_sequence.append(ch)
            if l != len(channel_mult) - 1:
                self.encoder_block.append(SampleBlock(sampling_type="down"))
                channel_sequence.append(ch)
        
        # Projection layer for reference features
        self.ref_proj = nn.Conv2d(self.channel_base * 4, self.channel_features, kernel_size=1) # projects a vector of 256 to 64
        ref_ch = self.channel_features  # Here, 64

        # Bottleneck blocks
        self.bottom_block0 = ResidualBlock(ch, ch, time_channel=time_embedding_channel, dropout=dropout)
        self.cross_attn_block = CrossAttentionBlock(dim=ch, cross_dim=ref_ch, num_heads=attention_head)
        self.bottom_block2 = ResidualBlock(ch, ch, time_channel=time_embedding_channel, dropout=dropout)
        
        # Decoder blocks
        self.decoder_block = nn.ModuleList()
        for l, mult in reversed(list(enumerate(channel_mult))):
            for _ in range(n_res_blocks):
                self.decoder_block.append(
                    ResidualBlock(ch + channel_sequence.pop(), mult * self.channel_base,
                                  time_channel=time_embedding_channel, dropout=dropout, cbam=cbam)
                )
                ch = mult * self.channel_base
            if l > 0:
                self.decoder_block.append(
                    Sequential(
                        ResidualBlock(ch + channel_sequence.pop(), mult * self.channel_base, time_channel=time_embedding_channel, dropout=dropout),
                        SampleBlock(sampling_type="up")
                    )
                )
                ch = mult * self.channel_base
        
        # Output layer
        self.output = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, self.channel_out, kernel_size=1)
        )
        
    def forward(self, x, t=None, ref_feats=None):
        """
        Forward pass of the network.

        Args:
        - x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].
        - t (torch.Tensor, optional): Time embedding tensor. Default is None.
        - ref_feats (torch.Tensor, optional): Reference features for cross-attention. Default is None.

        Returns:
        - torch.Tensor: Output tensor after applying the denoising UNet.
        """
        # Compute time embedding
        t_emb = self.time_embedding(torch.randn(x.shape[0], self.channel_base).to(x.device)) if t is not None else None
        h = self.input(x)
        ht = [h]
        for module in self.encoder_block:
            h = module(h, t_emb)
            ht.append(h)
        h = self.bottom_block0(h, t_emb)
        if ref_feats is not None:
            # Project reference features to match the number of channels
            ref_feats = self.ref_proj(ref_feats)
            if ref_feats.shape[-2:] != h.shape[-2:]:
                ref_feats = F.interpolate(ref_feats, size=h.shape[-2:], mode="nearest")
            h = self.cross_attn_block(h, ref_feats)
            h = self.bottom_block2(h, t_emb)   # To remove ?
        for module in self.decoder_block:
            h = torch.cat([h, ht.pop()], dim=1)
            h = module(h, t_emb)
        return self.output(h)

"""
===========================================================
Residual Block for Sketch Colorization
===========================================================

This script contains the implementation of the Residual Block used in the context of anime diffusion models. It includes the following class:
1. ResidualBlock: Implements a residual block with optional attention mechanisms.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import torch.nn as nn
from models.attention import CBAM

class ResidualBlock(nn.Module):
    """
    This class implements a residual block with optional attention mechanisms.
    It is used in the context of anime diffusion models.

    Args:
    - channel_in (int): Number of input channels.
    - channel_out (int): Number of output channels.
    - time_channel (int, optional): Dimension of the time embeddings. Default is None.
    - dropout (float, optional): Dropout rate. Default is 0.
    - cbam (bool, optional): Whether to use CBAM (Convolutional Block Attention Module). Default is False.

    The residual block consists of:
    - Two convolutional layers with GroupNorm and SiLU activations.
    - An optional time embedding layer.
    - An optional CBAM attention mechanism.
    - A skip connection to add the input to the output.
    """

    def __init__(
        self,
        channel_in,
        channel_out,
        time_channel=None,
        dropout=0,
        cbam=False
    ):
        super().__init__()

        # Member variables
        self.channel_base = channel_in
        self.channel_out = channel_out

        # Time embedding layer (optional)
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channel, self.channel_out)
        ) if time_channel is not None else None

        # First convolutional layer with GroupNorm and SiLU activation
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, self.channel_base),
            nn.SiLU(),
            nn.Conv2d(self.channel_base, self.channel_out, kernel_size=3, padding=1)
        )

        # Second convolutional layer with GroupNorm, SiLU activation, and Dropout
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, self.channel_out),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.channel_out, self.channel_out, kernel_size=3, padding=1)
        )

        # Attention mechanism (optional)
        self.attention = nn.Sequential(
            CBAM(self.channel_out)
        ) if cbam else nn.Identity()

        # Skip connection
        if self.channel_base == self.channel_out:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(self.channel_base, self.channel_out, kernel_size=1)

    def forward(self, x, t=None):
        """
        Forward pass of the residual block.

        Args:
        - x (torch.Tensor): Input tensor of shape [batch_size, x_channels, height, width].
        - t (torch.Tensor, optional): Time embedding tensor of shape [batch_size, t_embedding]. Default is None.

        Returns:
        - torch.Tensor: Output tensor after applying the residual block.
        """
        # Apply the first convolutional layer
        h = self.conv1(x)

        # Add time embedding if provided
        if t is not None and self.time_emb is not None:
            h += self.time_emb(t)[:, :, None, None]

        # Apply the attention mechanism
        h = self.attention(h)

        # Apply the second convolutional layer
        h = self.conv2(h)

        # Add the skip connection
        return self.skip_connection(x) + h
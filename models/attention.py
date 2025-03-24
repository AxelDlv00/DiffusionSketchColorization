"""
===========================================================
Attention Mechanisms for Neural Networks
===========================================================

This script contains various attention mechanisms used in neural networks. It includes the following classes:
1. ChannelAttention: Implements the Channel Attention mechanism.
2. SpatialAttention: Implements the Spatial Attention mechanism.
3. CBAM: Combines both Channel and Spatial Attention mechanisms.
4. SimpleSelfAttention: Implements a simple self-attention mechanism.
5. RefUNetAttentionBlock: Implements a comprehensive attention block similar to modern transformer blocks.
6. CrossAttentionBlock: Implements a cross-attention block for integrating features from another source.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    This class implements the Channel Attention mechanism.
    It focuses on the importance of each channel in the feature map.

    Args:
    - in_channel (int): Number of input channels.
    - ratio (int): Reduction ratio for the intermediate layer (default is 16).

    Methods:
    - forward(x): Forward pass of the channel attention mechanism.

    Example:
    >>> ca = ChannelAttention(in_channel=64)
    >>> output = ca(input_tensor)
    """

    def __init__(self, in_channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // ratio, False),  # Fully connected layer with reduction ratio
            nn.ReLU(),  # ReLU activation
            nn.Linear(in_channel // ratio, in_channel, False)  # Fully connected layer to restore original channel size
        )
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation to get attention weights

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view(b, c)  # Apply max pooling and reshape
        avg_pool_out = self.avg_pool(x).view(b, c)  # Apply average pooling and reshape

        max_fc_out = self.fc(max_pool_out)  # Pass max pooled output through fully connected layers
        avg_fc_out = self.fc(avg_pool_out)  # Pass average pooled output through fully connected layers

        out = max_fc_out + avg_fc_out  # Combine the outputs
        out = self.sigmoid(out).view(b, c, 1, 1)  # Apply sigmoid and reshape to match input dimensions

        return x * out  # Multiply input by attention weights


class SpatialAttention(nn.Module):
    """
    This class implements the Spatial Attention mechanism.
    It focuses on the importance of each spatial location in the feature map.

    Args:
    - kernel_size (int): Size of the convolutional kernel (default is 7).

    Methods:
    - forward(x): Forward pass of the spatial attention mechanism.

    Example:
    >>> sa = SpatialAttention(kernel_size=7)
    >>> output = sa(input_tensor)
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)  # Convolutional layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation to get attention weights

    def forward(self, x):
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling along the channel dimension
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)  # Mean pooling along the channel dimension
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)  # Concatenate max and mean pooled outputs

        out = self.conv(pool_out)  # Apply convolution
        out = self.sigmoid(out)  # Apply sigmoid to get attention weights

        return x * out  # Multiply input by attention weights


class CBAM(nn.Module):
    """
    This class implements the Convolutional Block Attention Module (CBAM).
    It combines both Channel Attention and Spatial Attention mechanisms.

    Args:
    - in_channel (int): Number of input channels.
    - ratio (int): Reduction ratio for the Channel Attention (default is 16).
    - kernel_size (int): Size of the convolutional kernel for the Spatial Attention (default is 7).

    Methods:
    - forward(x): Forward pass of the CBAM.

    Example:
    >>> cbam = CBAM(in_channel=64)
    >>> output = cbam(input_tensor)
    """

    def __init__(self, in_channel, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channel, ratio)  # Initialize Channel Attention
        self.spatial_attention = SpatialAttention(kernel_size)  # Initialize Spatial Attention

    def forward(self, x):
        x = self.channel_attention(x)  # Apply Channel Attention
        x = self.spatial_attention(x)  # Apply Spatial Attention

        return x  # Return the output with attention applied


class SimpleSelfAttention(nn.Module):
    """
    This class implements a simple self-attention mechanism.
    It focuses on capturing long-range dependencies in the feature map.

    Args:
    - in_channels (int): Number of input channels.

    Methods:
    - forward(x): Forward pass of the self-attention mechanism.

    Example:
    >>> sa = SimpleSelfAttention(in_channels=64)
    >>> output = sa(input_tensor)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, N, C//8)
        proj_key   = self.key(x).view(B, -1, H * W)                       # (B, C//8, N)
        energy     = torch.bmm(proj_query, proj_key)                      # (B, N, N)
        attention  = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(B, -1, H * W)                       # (B, C, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))             # (B, C, N)
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out


class RefUNetAttentionBlock(nn.Module):
    """
    This class implements a more comprehensive attention block that integrates not only an attention layer,
    but also normalization, residual connections, and a feed-forward MLP, making it more similar to modern transformer blocks.
    It helps capture global dependencies and refine the representation compared to simple self-attention.

    Args:
    - dim (int): Dimension of the input (and output) channels.
    - num_heads (int): Number of heads for multi-head attention.
    - dropout (float, optional): Dropout rate to apply in attention and MLP. Default is 0.0.

    Methods:
    - forward(x): Forward pass of the attention block.

    Example:
    >>> attn_block = RefUNetAttentionBlock(dim=64, num_heads=8)
    >>> output = attn_block(input_tensor)
    """

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Normalization before self-attention
        self.norm1 = nn.LayerNorm(dim)
        # Multi-head attention module from PyTorch (expects a sequence)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        
        # Normalization before the MLP
        self.norm2 = nn.LayerNorm(dim)
        # Transformer-like feed-forward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Forward pass of the attention block.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
        - torch.Tensor: Output tensor after applying the attention block.
        """
        B, C, H, W = x.size()
        # Flatten the spatial dimension to get a sequence: (seq_len, batch, dim)
        x_seq = x.view(B, C, H * W).permute(2, 0, 1)
        
        # Self-attention
        x_norm = self.norm1(x_seq)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x_res = x_seq + attn_out  # Residual connection
        
        # MLP (feed-forward)
        x_norm2 = self.norm2(x_res)
        mlp_out = self.mlp(x_norm2)
        x_res2 = x_res + mlp_out  # Another residual connection
        
        # Reshape back to (batch, dim, H, W)
        out = x_res2.permute(1, 2, 0).view(B, C, H, W)
        return out


class CrossAttentionBlock(nn.Module):
    """
    This class implements a cross-attention block that allows the model to attend to features from another source.
    It integrates features from a reference image into the main image features.

    Args:
    - dim (int): Dimension of the input (and output) channels.
    - cross_dim (int, optional): Dimension of the cross-attention channels. Default is None.
    - num_heads (int): Number of heads for multi-head attention.
    - dropout (float, optional): Dropout rate to apply in attention and MLP. Default is 0.0.

    Methods:
    - forward(x, cross): Forward pass of the cross-attention block.

    Example:
    >>> cross_attn_block = CrossAttentionBlock(dim=64, cross_dim=128, num_heads=4)
    >>> output = cross_attn_block(input_tensor, cross_tensor)
    """

    def __init__(self, dim, cross_dim=None, num_heads=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.cross_dim = cross_dim if cross_dim is not None else dim
        self.num_heads = num_heads
        self.norm_query = nn.LayerNorm(dim)
        # Projection to adapt reference features if necessary
        if self.cross_dim != dim:
            self.cross_proj = nn.Linear(self.cross_dim, dim)
        else:
            self.cross_proj = nn.Identity()
        self.norm_cross = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, cross):
        """
        Forward pass of the cross-attention block.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, C, H, W).
        - cross (torch.Tensor): Cross-attention tensor of shape (B, cross_channels, H, W).

        Returns:
        - torch.Tensor: Output tensor after applying the cross-attention block.
        """
        B, C, H, W = x.shape
        # Ensure cross has the same spatial resolution as x
        if cross.shape[-2:] != (H, W):
            cross = F.interpolate(cross, size=(H, W), mode="nearest").contiguous()
        # cross is assumed to be of shape (B, cross_channels, H, W)
        # Transform it into a sequence (B, H*W, cross_channels)
        cross_seq = cross.view(B, cross.shape[1], H * W).permute(0, 2, 1)
        # Apply the projection (defined in __init__) to get the expected dimension
        cross_seq = self.cross_proj(cross_seq)
        # Transform x into a sequence (B, H*W, C)
        x_seq = x.view(B, C, H * W).permute(0, 2, 1)
        x_norm = self.norm_query(x_seq)
        cross_norm = self.norm_cross(cross_seq)
        attn_out, _ = self.attn(query=x_norm, key=cross_norm, value=cross_norm)
        x_res = x_seq + attn_out
        x_norm2 = self.norm_out(x_res)
        mlp_out = self.mlp(x_norm2)
        x_final = x_res + mlp_out
        # Reshape back to (B, C, H, W)
        return x_final.permute(0, 2, 1).view(B, C, H, W)
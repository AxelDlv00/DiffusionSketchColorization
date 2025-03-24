"""
===========================================================
Extended Module for Time-Dependent Layers
===========================================================

This script defines an extended module class for handling time-dependent layers in neural networks. It includes the following classes:
1. Module: Abstract base class for time-dependent layers.
2. Sequential: Sequential container for stacking time-dependent layers.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

from abc import abstractmethod
import torch.nn as nn

class Module(nn.Module):
    """
    Abstract base class for time-dependent layers.

    Args:
        x (torch.Tensor): Input tensor.
        t (torch.Tensor): Time step tensor.

    Returns:
        torch.Tensor: Output tensor after applying the layer.
    """
    @abstractmethod
    def forward(self, x, t):
        raise NotImplemented

class Sequential(nn.Sequential, Module):
    """
    Sequential container for stacking time-dependent layers.

    Args:
        x (torch.Tensor): Input tensor.
        t (torch.Tensor): Time step tensor.

    Returns:
        torch.Tensor: Output tensor after applying all layers in sequence.
    """
    @abstractmethod
    def forward(self, x, t):
        for layer in self:
            x = layer(x, t) if isinstance(layer, Module) else layer(x)
        return x
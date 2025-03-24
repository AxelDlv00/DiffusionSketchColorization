"""
===========================================================
Progressive Patch Shuffling Utilities
===========================================================

This script contains utility functions for shuffling image patches recursively. It includes the following functions:
1. recursive_patch_shuffle: Shuffle patches of an image recursively and return a PIL image.
2. recursive_patch_shuffle_Tensor: Shuffle patches of an image tensor recursively and return a tensor.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import torch
import torchvision.transforms as T
import numpy as np

def recursive_patch_shuffle(img, patch_size=16, depth=2):
    """
    Shuffle patches of an image recursively and return a PIL image.

    Args:
        img (PIL.Image): Input image.
        patch_size (int): Size of the patches.
        depth (int): Number of recursive shuffling levels.

    Returns:
        PIL.Image: Image with shuffled patches.
    """
    img_tensor = T.ToTensor()(img)
    _, H, W = img_tensor.shape

    H_pad = (patch_size - H % patch_size) % patch_size
    W_pad = (patch_size - W % patch_size) % patch_size

    # Padding to ensure full patch divisions
    img_tensor = torch.nn.functional.pad(img_tensor, (0, W_pad, 0, H_pad))
    
    def shuffle_patches(tensor, level):
        if level == 0:
            return tensor
        
        C, H, W = tensor.shape
        
        if H < patch_size or W < patch_size:
            return tensor  # Stop if patches are too small

        patches = tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, C, patch_size, patch_size)
        
        perm = torch.randperm(len(patches))
        shuffled_patches = patches[perm]
        
        shuffled_img = shuffled_patches.view(H // patch_size, W // patch_size, C, patch_size, patch_size)
        shuffled_img = shuffled_img.permute(2, 0, 3, 1, 4).reshape(C, H, W)
        
        return shuffle_patches(shuffled_img, level - 1)
    
    shuffled_tensor = shuffle_patches(img_tensor, depth)
    return T.ToPILImage()(shuffled_tensor)

def recursive_patch_shuffle_Tensor(tenseur, patch_size=16, depth=2):
    """
    Shuffle patches of an image tensor recursively and return a tensor.

    Args:
        tenseur (torch.Tensor): Input image tensor.
        patch_size (int): Size of the patches.
        depth (int): Number of recursive shuffling levels.

    Returns:
        torch.Tensor: Tensor with shuffled patches.
    """
    img_tensor = tenseur
    _, H, W = img_tensor.shape

    H_pad = (patch_size - H % patch_size) % patch_size
    W_pad = (patch_size - W % patch_size) % patch_size

    # Padding to ensure full patch divisions
    img_tensor = torch.nn.functional.pad(img_tensor, (0, W_pad, 0, H_pad))
    
    def shuffle_patches(tensor, level):
        if level == 0:
            return tensor
        
        C, H, W = tensor.shape
        
        if H < patch_size or W < patch_size:
            return tensor  # Stop if patches are too small

        patches = tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, C, patch_size, patch_size)
        
        perm = torch.randperm(len(patches))
        shuffled_patches = patches[perm]
        
        shuffled_img = shuffled_patches.view(H // patch_size, W // patch_size, C, patch_size, patch_size)
        shuffled_img = shuffled_img.permute(2, 0, 3, 1, 4).reshape(C, H, W)
        
        return shuffle_patches(shuffled_img, level - 1)
    
    shuffled_tensor = shuffle_patches(img_tensor, depth)
    return shuffled_tensor
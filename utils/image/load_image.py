"""
===========================================================
Image Loading Utility
===========================================================

This script contains a utility function for loading and transforming images. It includes the following function:
1. load_image: Load an image from a given path and apply transformations.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import torchvision.transforms as transforms
from PIL import Image

# === Function: Load Image ===
def load_image(image_path, size=(256, 256)):
    """
    Load an image from a given path and apply transformations.

    Args:
        image_path (str): Path to the image file.
        size (tuple): Desired size of the output image (width, height).

    Returns:
        torch.Tensor: Transformed image tensor of shape (1, 3, H, W).
    """
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # (1, 3, H, W)
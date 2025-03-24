"""
===========================================================
Data Augmentation Utilities for Sketch Colorization
===========================================================

This script contains utility functions for augmenting data by applying random deformations and transformations to images. It includes the following functions:
1. generate_random_deformation_flow: Creates a random smooth deformation field.
2. rotate_and_crop: Applies rotation and zoom-in to avoid black borders.
3. apply_random_transformations: Applies the same deformation flow and random rotation to both reference and sketch images.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import cv2
import os
import numpy as np
import random
import tqdm
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor

# === Function: Generate Random Deformation Flow ===
def generate_random_deformation_flow(h, w, strength=10, noise_level=3, blur_sigma=5):
    """
    Creates a random smooth deformation field.

    Args:
        h (int): Height of the image.
        w (int): Width of the image.
        strength (int): Strength of the deformation.
        noise_level (int): Level of Gaussian noise to add.
        blur_sigma (int): Sigma value for Gaussian blur.

    Returns:
        tuple: Maps for x and y coordinates after deformation.
    """
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    flow_x = strength * np.sin(2 * np.pi * y / h + random.uniform(0, np.pi * 2))
    flow_y = strength * np.cos(2 * np.pi * x / w + random.uniform(0, np.pi * 2))

    # Add random Gaussian noise
    noise_x = np.random.normal(0, noise_level, (h, w))
    noise_y = np.random.normal(0, noise_level, (h, w))

    # Combine smooth flow with noise
    flow_x += noise_x
    flow_y += noise_y

    # Apply Gaussian blur for smoothness
    flow_x = gaussian_filter(flow_x, sigma=blur_sigma)
    flow_y = gaussian_filter(flow_y, sigma=blur_sigma)

    # Clip displacement values
    flow_x = np.clip(flow_x, -strength, strength)
    flow_y = np.clip(flow_y, -strength, strength)

    # Compute new pixel positions
    map_x = np.clip(x + flow_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + flow_y, 0, h - 1).astype(np.float32)

    return map_x, map_y

# === Function: Apply Rotation with Zoom ===
def rotate_and_crop(image, angle, zoom=1.1):
    """
    Applies rotation and zoom-in to avoid black borders.

    Args:
        image (np.ndarray): Input image.
        angle (float): Rotation angle in degrees.
        zoom (float): Zoom factor.

    Returns:
        np.ndarray: Rotated and zoomed image.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute transformation matrix with scaling
    M = cv2.getRotationMatrix2D(center, angle, zoom)

    # Apply affine transformation (rotation + zoom)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return rotated

# === Function: Apply Deformation & Rotation ===
def apply_random_transformations(image, sketch, strength=20, noise_level=10, rotation_range=45, zoom_factor=1.1):
    """
    Applies the same deformation flow and random rotation to both reference and sketch images.

    Args:
        image (np.ndarray): Reference image.
        sketch (np.ndarray): Sketch image.
        strength (int): Strength of the deformation.
        noise_level (int): Level of Gaussian noise to add.
        rotation_range (int): Range of rotation angles.
        zoom_factor (float): Zoom factor.

    Returns:
        tuple: Transformed reference and sketch images.
    """
    h, w, _ = image.shape
    map_x, map_y = generate_random_deformation_flow(h, w, strength, noise_level)
    
    # Apply deformation to both images
    warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    warped_sketch = cv2.remap(sketch, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Apply random rotation (same for both)
    angle = random.uniform(-rotation_range, rotation_range)
    warped_image = rotate_and_crop(warped_image, angle, zoom_factor)
    warped_sketch = rotate_and_crop(warped_sketch, angle, zoom_factor)

    return warped_image, warped_sketch

"""
===========================================================
Image Processing Utilities for Sketch Colorization
===========================================================

This script contains utility functions for processing colored images into sketches. It includes the following functions:
1. threshold: Apply a threshold to an image.
2. extract_regions_by_class: Extract pixel regions for each class in a class matrix.
3. enhance_lines_preserving_details: Enhance visible lines in an image while preserving details.
4. get_sketch: Convert a colored image to a sketch.
5. pencil_sketch: Generate a pencil sketch with adjustable parameters.
6. plot_aside: Plot the original image and its sketch side by side.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import numpy as np  # grey images are stored in memory as 2D arrays, color images as 3D arrays
import cv2  # opencv computer vision library
import matplotlib.pyplot as plt  # for plotting images

def threshold(img, value):
    """
    Apply a threshold to an image.

    Args:
        img (np.ndarray): Image to threshold.
        value (int): Threshold value.

    Returns:
        np.ndarray: Thresholded image.
    """
    return np.where(img <= value, 0, 255).astype(np.uint8)

def extract_regions_by_class(classes_matrix, class_list):
    """
    Extract pixel regions for each class in the class matrix.

    Args:
        classes_matrix (np.ndarray): Matrix of class labels.
        class_list (list): List of class numbers to extract.

    Returns:
        dict: Dictionary of masks for each class.
    """
    regions = {}
    for class_num in class_list:
        mask = (classes_matrix == class_num)
        regions[class_num] = mask
    return regions

def enhance_lines_preserving_details(image, alpha=1, beta=1):
    """
    Enhance visible lines in an image while preserving the original details.

    Args:
        image (numpy.ndarray): Input image (BGR format).
        alpha (float): Contrast control (1.0-3.0).
        beta (int): Brightness control (0-100).

    Returns:
        numpy.ndarray: Image with enhanced lines.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a slight Gaussian blur to reduce noise without oversimplifying details
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use the original image and adjust contrast and brightness
    enhanced = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)

    # Convert back to 3-channel image for visualization
    result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return result

def get_sketch(image):
    """
    Convert a colored image to a sketch.

    Args:
        image (numpy.ndarray): Input colored image (BGR format).

    Returns:
        numpy.ndarray: Sketch image.
    """
    # Convert to grayscale
    image_denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 25, 3, 21)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    invertedblur = 255 - blurred
    lines = cv2.divide(gray, invertedblur, scale=256.0)
    lines = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
    return threshold(enhance_lines_preserving_details(lines), 254)

def pencil_sketch(image, blur_ksize=31, contrast=1, brightness=5, darkness_percentile=100):
    """
    Generates a pencil sketch with adjustable blur, contrast, brightness, and thresholding.

    Args:
        image (np.ndarray): Input grayscale image.
        blur_ksize (int): Kernel size for Gaussian blur.
        contrast (float): Contrast adjustment factor.
        brightness (int): Brightness adjustment factor.
        darkness_percentile (float): Percentage of darkest pixels to keep (0-100).
        
    Returns:
        np.ndarray: Thresholded pencil sketch.
    """
    # Invert colors
    inverted = cv2.bitwise_not(image)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted, (blur_ksize, blur_ksize), 0)
    inverted_blurred = cv2.bitwise_not(blurred)
    
    # Create the pencil sketch
    sketch = cv2.divide(image, inverted_blurred, scale=256.0)
    
    # Adjust contrast and brightness
    sketch = cv2.convertScaleAbs(sketch, alpha=contrast, beta=brightness)
    
    # Compute the intensity threshold using percentile
    threshold_value = np.percentile(sketch, darkness_percentile)
    
    # Apply threshold: Keep only pixels darker than the threshold
    sketch[sketch > threshold_value] = 255  # Convert bright pixels to white
    
    return sketch

def plot_aside(img, blur_ksize=31, contrast=1, brightness=5, darkness_percentile=100):
    """
    Plot the original image and its sketch side by side.

    Args:
        img (np.ndarray): Input colored image (BGR format).
        blur_ksize (int): Kernel size for Gaussian blur.
        contrast (float): Contrast adjustment factor.
        brightness (int): Brightness adjustment factor.
        darkness_percentile (float): Percentage of darkest pixels to keep (0-100).
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sketch = pencil_sketch(grey, blur_ksize, contrast, brightness, darkness_percentile)
    # Plot the original image and its sketch
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(sketch, cmap="gray")
    axes[1].set_title(f"Sketch, blur={blur_ksize}, contrast={contrast}, brightness={brightness}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

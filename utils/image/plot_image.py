"""
===========================================================
Image Plotting Utilities
===========================================================

This script contains utility functions for plotting images using Matplotlib. It includes the following functions:
1. show_img: Display a single image on a given axis.
2. plot_img: Plot multiple images in a single figure.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import matplotlib.pyplot as plt
import numpy as np

# === Function: Show Image ===
def show_img(ax, tensor, title=""):
    """
    Display a single image on a given axis.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib axis to display the image on.
        tensor (torch.Tensor): Image tensor to display.
        title (str): Title of the image.
    """
    t = tensor.clamp(0, 1).cpu()
    if t.dim() == 4:
        t = t.squeeze(0)
    if t.shape[0] == 1:
        np_img = t.squeeze(0).numpy()
        ax.imshow(np_img, cmap='gray')
    else:
        np_img = t.permute(1, 2, 0).numpy()
        ax.imshow(np_img)
    ax.set_title(title)
    ax.axis('off')

# === Function: Plot Multiple Images ===
def plot_img(imgs, titles=None, figsize=(12, 12)):
    """
    Plot multiple images in a single figure.

    Args:
        imgs (list of torch.Tensor): List of image tensors to plot.
        titles (list of str, optional): List of titles for each image.
        figsize (tuple): Size of the figure.
    """
    fig, axes = plt.subplots(1, len(imgs), figsize=figsize)
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        title = titles[i] if titles else ""
        show_img(ax, img, title)
    plt.tight_layout()
    plt.show()
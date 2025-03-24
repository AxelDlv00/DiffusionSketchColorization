"""
===========================================================
PSCNet Model Visualization Utilities
===========================================================

This script contains utility functions for visualizing the outputs of the PSCNet model. It includes the following functions:
1. visualize_samples_loader: Visualize samples from the test DataLoader by generating colorized outputs using the provided models.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import torch
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from utils.image.warp_psc import warp_reference_with_psc

@torch.no_grad()
def visualize_samples_loader(
    test_loader,
    diffusion_model,  # PSCGaussianDiffusion
    ref_unet,
    psc_model,
    device="cuda",
    num_samples=5
):
    """
    Visualize samples from the test DataLoader by generating colorized outputs
    using PSCGaussianDiffusion.

    We assume each batch from test_loader is:
      (sketch, image, other_image, patched_ref, flow_warp)
    and each is shape [1, 3, H, W] if batch_size=1.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # We only want to visualize up to `num_samples` batches
    # Convert the loader to an iterator
    test_loader_iter = iter(test_loader)

    for _ in range(num_samples):
        try:
            batch = next(test_loader_iter)
        except StopIteration:
            break

        # batch is a tuple of length 5 (each shape [1, 3, H, W] if batch_size=1)
        (
            sketch_of_the_character,
            image_of_the_character,
            other_image_of_the_character,
            patched_ref,
            flow_warp
        ) = [x.to(device) for x in batch]

        # Shapes now (assuming batch_size=1):
        #  sketch_of_the_character: [1, 3, H, W]
        #  image_of_the_character : [1, 3, H, W]
        #  other_image_of_the_character: [1, 3, H, W]
        #  patched_ref: [1, 3, H, W]
        #  flow_warp : [1, 3, H, W]

        # 1) Get reference feats from 'patched_ref'
        ref_feats = ref_unet(patched_ref)  # up to your model details

        # 2) Create random noise x_T of shape [1, 3, H, W]
        x_t = torch.randn_like(other_image_of_the_character)

        # 3) Build the 6-channel conditioning: [1, 6, H, W]
        #    (sketch + flow_warp)
        x_cond = torch.cat([sketch_of_the_character, flow_warp], dim=1)

        # 4) Run the full reverse diffusion
        output_list = diffusion_model.inference(
            x_t=x_t,
            ref_feats=ref_feats,
            x_cond=x_cond,
            eta=1.0
        )
        final_output = output_list[-1]  # The final image

        # Convert each tensor to a NumPy image for plotting
        sketch_np = sketch_of_the_character[0].cpu().permute(1, 2, 0).numpy()  # [H,W,3]
        flow_np   = flow_warp[0].cpu().permute(1, 2, 0).numpy()                # [H,W,3]
        output_np = final_output[0].cpu().permute(1,2,0).numpy()               # [H,W,3]
        gt_np     = image_of_the_character[0].cpu().permute(1, 2, 0).numpy()   # [H,W,3]
        other_np  = other_image_of_the_character[0].cpu().permute(1, 2, 0).numpy()

        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(sketch_np)
        axes[0, 0].set_title("Sketch")

        axes[0, 1].imshow(other_np)
        axes[0, 1].set_title("Other Ref (Unwarped)")

        axes[0, 2].imshow(flow_np)
        axes[0, 2].set_title("Flow Warp")

        axes[1, 0].imshow(gt_np)
        axes[1, 0].set_title("Ground Truth Image")

        axes[1, 1].imshow(output_np)
        axes[1, 1].set_title("Diffusion Output")

        axes[1, 2].axis('off')
        plt.tight_layout()
        plt.show()

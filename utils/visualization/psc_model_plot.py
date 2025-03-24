"""
===========================================================
PSCNet Model Visualization Utilities
===========================================================

This script contains utility functions for visualizing the outputs of the PSCNet model. It includes the following functions:
1. visualize_samples: Visualize samples from the test dataset by generating colorized outputs using the provided models.
2. visualize_samples_loader: Visualize samples from the test DataLoader by generating colorized outputs using the provided models.

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import torch
from utils.image.warp_psc import warp_reference_with_psc
import matplotlib.pyplot as plt
import random

def visualize_samples(test_dataset, ref_unet, denoise_unet, psc_model, device="cuda", num_samples=5):
    """
    Visualize samples from the test dataset by generating colorized outputs using the provided models.

    Args:
        test_dataset (Dataset): The dataset containing test samples.
        ref_unet (nn.Module): The ReferenceUNet model used for extracting features from the reference image.
        denoise_unet (nn.Module): The DenoisingUNet model used for generating the final colorized output.
        psc_model (nn.Module): The PSCNet model used for warping the reference image.
        device (str): The device to run the models on ("cuda" for GPU, "cpu" for CPU).
        num_samples (int): The number of samples to visualize.

    Returns:
        None
    """
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    dataset_size = len(test_dataset)
    num_samples = min(num_samples, dataset_size)
    # Shuffle indices to randomly select samples
    random_indices = random.sample(range(dataset_size), num_samples)
    
    for i in random_indices:
        # Get a sketch sample from the dataset
        sketch, _, _ = test_dataset[i]
        # Randomly choose a mismatched reference sample
        j = random.randint(0, dataset_size - 1)
        _, mismatched_reference, mismatched_patched_ref = test_dataset[j]
        # Add batch dimension and move to the selected device
        sketch = sketch.unsqueeze(0).to(device)
        mismatched_reference = mismatched_reference.unsqueeze(0).to(device)
        mismatched_patched_ref = mismatched_patched_ref.unsqueeze(0).to(device)
        # Create a noisy sketch: convert sketch to grayscale and add Gaussian noise
        sketch_gray = sketch.mean(dim=1, keepdim=True)
        noisy_sketch = torch.clamp(sketch_gray + 0.1 * torch.randn_like(sketch_gray), 0., 1.)
        # Warp the mismatched reference using PSCNet (using the sketch and mismatched reference)
        warped_ref = warp_reference_with_psc(sketch, mismatched_reference, psc_model)
        # Concatenate the noisy sketch and the warped mismatched reference
        denoise_input = torch.cat([noisy_sketch, warped_ref], dim=1)
        # Extract features from the mismatched patched reference using ReferenceUNet
        ref_features = ref_unet(mismatched_patched_ref)
        # Generate the output using the DenoisingUNet
        output = denoise_unet(denoise_input, ref_features)
        # Convert tensors to numpy for visualization
        noisy_np = noisy_sketch[0].detach().cpu().squeeze().numpy()
        warped_np = warped_ref[0].detach().cpu().permute(1, 2, 0).numpy()
        output_np = output[0].detach().cpu().permute(1, 2, 0).numpy()
        mismatched_ref_np = mismatched_reference[0].detach().cpu().permute(1, 2, 0).numpy()
        # Plot the mismatched visualization:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(noisy_np, cmap='gray')
        axes[0].set_title("Noisy Sketch")
        axes[0].axis("off")

        axes[1].imshow(mismatched_reference[0].cpu().permute(1, 2, 0))
        axes[1].set_title("Mismatched Reference")
        axes[1].axis("off")
        
        axes[2].imshow(warped_np)
        axes[2].set_title("Warped Mismatched Ref")
        axes[2].axis("off")
        
        axes[3].imshow(output_np)
        axes[3].set_title("Output Color")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()


def visualize_samples_loader(test_loader, ref_unet, denoise_unet, psc_model,
                             device="cuda", num_samples=5):
    """
    Visualize samples from the test DataLoader by generating colorized outputs
    using the provided models.

    Args:
        test_loader (DataLoader): The DataLoader containing test samples.
        ref_unet (nn.Module): The ReferenceUNet model used for extracting features from the reference image.
        denoise_unet (nn.Module): The DenoisingUNet model used for generating the final colorized output.
        psc_model (nn.Module): The PSCNet model used for warping the reference image.
        device (str): The device to run the models on ("cuda" for GPU, "cpu" for CPU).
        num_samples (int): The number of samples to visualize.

    Returns:
        None
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Flatten out a few samples from the loader
    # Each batch is (sketch, reference, sketch_transformed, reference_transformed), each shape [B, 3, H, W]
    # zip(*batch) => yields B tuples, each with 4 items of shape [3, H, W]
    test_samples = []
    for batch in test_loader:
        test_samples.extend(zip(*batch))  
        if len(test_samples) >= num_samples:
            break

    dataset_size = len(test_samples)
    if dataset_size == 0:
        print("No samples found in test_loader.")
        return

    num_samples = min(num_samples, dataset_size)
    random_indices = random.sample(range(dataset_size), num_samples)

    for idx in random_indices:
        # sample: (sketch, reference, sketch_transformed, reference_transformed)
        sketch, reference, sketch_transformed, reference_transformed = test_samples[idx]

        # pick a different sample for mismatch
        j = idx
        while j == idx:
            j = random.randint(0, dataset_size - 1)
        mismatched_sketch, mismatched_reference, mismatched_patched_ref, _ = test_samples[j]

        # Add batch dim => shape [1, 3, H, W]
        sketch = sketch.unsqueeze(0).to(device)
        mismatched_reference = mismatched_reference.unsqueeze(0).to(device)
        mismatched_patched_ref = mismatched_patched_ref.unsqueeze(0).to(device)

        # (Optional) add mild noise
        noisy_sketch = torch.clamp(sketch + 0.1*torch.randn_like(sketch), 0., 1.) 
        # shape => [1, 3, H, W]

        # Warp the mismatched reference using PSC
        # shape => [1, 3, H, W]
        warped_ref = warp_reference_with_psc(sketch, mismatched_reference, psc_model)

        # Now we want 9 channels => cat(1) => [1, 9, H, W]
        # Because each is 3 channels
        print("sketch shape:", sketch.shape)                  # Expect [1, 3, H, W]
        print("noisy_sketch shape:", noisy_sketch.shape)      # Expect [1, 3, H, W]
        print("warped_ref shape:", warped_ref.shape)          # Expect [1, 3, H, W]
        print("mismatched_patched_ref shape:", mismatched_patched_ref.shape)  # Expect [1, 3, H, W]
        denoise_input = torch.cat([noisy_sketch, warped_ref, mismatched_patched_ref], dim=1)
        print(f"denoise_input shape: {denoise_input.shape}")  # Should be [1, 9, H, W]

        # Extract reference features
        ref_features = ref_unet(mismatched_patched_ref)

        # Inference
        output = denoise_unet(denoise_input, ref_features)  # shape => [1, 3, H, W]

        # Convert to numpy
        noisy_np = noisy_sketch[0].detach().cpu().permute(1, 2, 0).numpy()
        warped_np = warped_ref[0].detach().cpu().permute(1, 2, 0).numpy()
        output_np = output[0].detach().cpu().permute(1, 2, 0).numpy()
        mismatch_ref_np = mismatched_reference[0].detach().cpu().permute(1, 2, 0).numpy()

        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(16,4))
        axes[0].imshow(noisy_np)
        axes[0].set_title("Noisy Sketch (3ch)")
        axes[0].axis("off")

        axes[1].imshow(mismatch_ref_np)
        axes[1].set_title("Mismatched Ref (3ch)")
        axes[1].axis("off")

        axes[2].imshow(warped_np)
        axes[2].set_title("Warped Ref -> Sketch")
        axes[2].axis("off")

        axes[3].imshow(output_np)
        axes[3].set_title("Output (3ch)")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()
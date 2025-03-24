"""
===========================================================
Custom Anime Dataset for Sketch Colorization
===========================================================

This script defines the CustomAnimeDataset class used for loading and processing the anime sketch and reference images for the sketch colorization model. It includes the following steps:
1. Initializing the dataset with paths and parameters
2. Loading and pairing sketch and reference images
3. Applying transformations and generating triplets

Author: Axel Delaval and Adama Koïta
Year: 2025
===========================================================
"""

import os
import itertools
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import random
import torchvision.transforms as T
from utils.image.progressive_patch import recursive_patch_shuffle_Tensor
from utils.image.compute_flow import compute_flow

class CustomAnimeDataset(Dataset):
    def __init__(self, psc, patch_size, sketch_dir, reference_dir, subset_percentage=1.0, seed=42, number_of_ref_per_sketch=5):
        """
        Initialize the CustomAnimeDataset.

        Args:
            psc (object): PSC model for computing flow.
            patch_size (int): Size of the patches for shuffling.
            sketch_dir (str): Directory containing sketch images.
            reference_dir (str): Directory containing reference images.
            subset_percentage (float): Percentage of the dataset to use.
            seed (int): Random seed for reproducibility.
            number_of_ref_per_sketch (int): Number of reference images per sketch.
        """
        self.sketch_dir = sketch_dir
        self.reference_dir = reference_dir
        self.psc = psc
        self.patch_size = patch_size
        self.characters_list = sorted(os.listdir(sketch_dir))
        self.triplets = []
        random.seed(seed)

        for character in self.characters_list:
            sketch_character_path = os.path.join(sketch_dir, character)
            reference_character_path = os.path.join(reference_dir, character)
            if not (os.path.isdir(sketch_character_path) and os.path.isdir(reference_character_path)):
                continue

            # Gather all images for that character
            image_filenames = sorted(os.listdir(sketch_character_path))
            number_of_ref_per_sketch_temp = min(number_of_ref_per_sketch, len(image_filenames) - 1)
            
            if number_of_ref_per_sketch_temp < 1:
                continue # Need at least 2 to form a pair/triplet

            for sketch_filename in image_filenames:
                sketch_img_path = os.path.join(sketch_character_path, sketch_filename)
                reference_img_path = os.path.join(reference_character_path, sketch_filename)
                if not (os.path.exists(sketch_img_path) and os.path.exists(reference_img_path)):
                    continue # Skip if the pair is not complete

                # All possible "others"
                other_references = [f for f in image_filenames if f != sketch_filename]
                if not other_references:
                    continue

                # Randomly take number_of_ref_per_sketch_temp different references
                other_references = random.sample(other_references, number_of_ref_per_sketch_temp) 
                for other_ref_filename in other_references:
                    other_reference_img_path = os.path.join(reference_character_path, other_ref_filename)
                    if os.path.exists(other_reference_img_path):
                        self.triplets.append((sketch_img_path, reference_img_path, other_reference_img_path))

        # Subset if needed
        subset_size = int(len(self.triplets) * subset_percentage)
        self.triplets = random.sample(self.triplets, subset_size)
        print(f"Final number of triplets after applying subset={subset_percentage}: {len(self.triplets)}")

    def __len__(self):
        """Return the total number of triplets."""
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        Get a triplet of images (sketch, reference, other reference) and their transformations.

        Args:
            idx (int): Index of the triplet.

        Returns:
            tuple: Transformed sketch, reference, other reference, patched reference, and flow tensors.
        """
        sketch_path, reference_path, other_reference_path = self.triplets[idx]

        # Load images in RGB
        sketch_img = Image.open(sketch_path).convert("RGB")
        reference_img = Image.open(reference_path).convert("RGB")
        other_reference_img = Image.open(other_reference_path).convert("RGB")

        transform = T.Compose([
            T.Resize((256, 256)), 
            T.ToTensor()
        ])
        sketch_tensor = transform(sketch_img)
        reference_tensor = transform(reference_img)
        other_reference_tensor = transform(other_reference_img)
        patched_ref_tensor = recursive_patch_shuffle_Tensor(other_reference_tensor, patch_size=self.patch_size, depth=2)
        flow_rgb = compute_flow(self.psc, sketch_tensor, other_reference_tensor, image_size=256)
        flow_pil = Image.fromarray(flow_rgb)  
        flow_tensor = transform(flow_pil)     

        return (sketch_tensor, reference_tensor, other_reference_tensor, patched_ref_tensor, flow_tensor)
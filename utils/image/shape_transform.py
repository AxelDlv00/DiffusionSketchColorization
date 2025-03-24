import os
import random
import math
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor

class ShapeTransform_:
    """
    Class to apply an affine transformation (rotation, scales, shear)
    identically on one or more PIL.Image objects.

    Args:
    - angle_range (tuple): Range for the rotation angle.
    - scale_x_range (tuple): Range for scaling in x.
    - scale_y_range (tuple): Range for scaling in y.
    - shear_x_range (tuple): Range for shearing in x.
    - shear_y_range (tuple): Range for shearing in y.

    Methods:
    - _generate_params(): Generates random transformation parameters.
    - _apply_transform(image, params): Applies the transformation to an image.
    - __call__(*images): Applies the same transformation to the provided images.

    Example:
    >>> transform = ShapeTransform_()
    >>> transformed_image = transform(image)
    """

    def __init__(self, 
                 angle_range=(-180, 180),
                 scale_x_range=(0.8, 1.2),
                 scale_y_range=(0.8, 1.2),
                 shear_x_range=(-20, 20),
                 shear_y_range=(-20, 20)):
        self.angle_range = angle_range
        self.scale_x_range = scale_x_range
        self.scale_y_range = scale_y_range
        self.shear_x_range = shear_x_range
        self.shear_y_range = shear_y_range

    def _generate_params(self):
        """
        Generates random transformation parameters.

        Returns:
        - dict: Dictionary containing the transformation parameters.
        """
        angle = random.uniform(*self.angle_range)
        scale_x = random.uniform(*self.scale_x_range)
        scale_y = random.uniform(*self.scale_y_range)
        shear_x = random.uniform(*self.shear_x_range)
        shear_y = random.uniform(*self.shear_y_range)
        return {
            'angle': angle,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'shear_x': shear_x,
            'shear_y': shear_y
        }

    def _apply_transform(self, image, params):
        """
        Applies the transformation to an image.

        Args:
        - image (PIL.Image): Image to transform.
        - params (dict): Transformation parameters.

        Returns:
        - PIL.Image: Transformed image.
        """
        w, h = image.size
        cx, cy = w / 2, h / 2

        # Convert to radians
        theta = math.radians(params['angle'])
        shear_x_rad = math.radians(params['shear_x'])
        shear_y_rad = math.radians(params['shear_y'])

        # Non-uniform scaling matrix
        S = np.array([
            [params['scale_x'], 0, 0],
            [0, params['scale_y'], 0],
            [0, 0, 1]
        ])

        # Shear matrix
        Sh = np.array([
            [1, math.tan(shear_x_rad), 0],
            [math.tan(shear_y_rad), 1, 0],
            [0, 0, 1]
        ])

        # Rotation matrix
        R = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ])

        # Combined affine matrix
        M = R @ Sh @ S

        # Translation to apply the transformation around the center
        T1 = np.array([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1]
        ])
        T2 = np.array([
            [1, 0, cx],
            [0, 1, cy],
            [0, 0, 1]
        ])
        M_final = T2 @ M @ T1

        # PIL.Image.transform expects the inverse matrix
        M_inv = np.linalg.inv(M_final)
        coeffs = M_inv[:2, :].flatten().tolist()

        # Transformation with bilinear interpolation and white background
        return image.transform((w, h), Image.AFFINE, data=coeffs,
                               resample=Image.BILINEAR, fillcolor=(255, 255, 255))

    def __call__(self, *images):
        """
        Applies the same transformation to the provided images.
        If a single argument is passed, returns the transformed image.
        If multiple images are passed, returns a tuple containing
        each of the transformed images.

        Args:
        - *images: One or more PIL.Image images to transform.

        Returns:
        - PIL.Image or tuple of PIL.Image: Transformed image(s).
        """
        params = self._generate_params()
        results = [self._apply_transform(image, params) for image in images]
        return results[0] if len(results) == 1 else tuple(results)


# Example of usage in a Dataset
class AnimeDataset_(Dataset):
    """
    Dataset loading corresponding images (sketch and reference)
    and applying both a standard transformation (e.g., ToTensor)
    and an identical shape transformation via ShapeTransform.

    The outputs are:
      sketch, reference, sketch_shape_transformed, ref_shape_transformed

    Args:
    - sketch_dir (str): Directory containing the sketch images.
    - reference_dir (str): Directory containing the reference images.
    - transform_standard (callable, optional): Standard transformation to apply (e.g., ToTensor).
    - shape_transform (callable, optional): Instance of ShapeTransform for shape transformations.
    - subset_percentage (float, optional): Percentage of the dataset to use. Default is 1.0.
    - seed (int, optional): Seed for random. Default is 42.

    Methods:
    - __len__(): Returns the size of the dataset.
    - __getitem__(idx): Returns a sample from the dataset.

    Example:
    >>> dataset = AnimeDataset_(sketch_dir="path/to/sketch", reference_dir="path/to/reference")
    >>> sketch, reference, sketch_shape_transformed, ref_shape_transformed = dataset[0]
    """

    def __init__(self, sketch_dir, reference_dir, transform_standard=None, shape_transform=None,
                 subset_percentage=1.0, seed=42):
        self.sketch_dir = sketch_dir
        self.reference_dir = reference_dir
        self.transform_standard = transform_standard  # Standard transformation (e.g., ToTensor)
        self.shape_transform = shape_transform        # Instance of ShapeTransform

        all_images = [f for f in os.listdir(sketch_dir) if f.lower().endswith(('.jpg', '.png'))]
        random.seed(seed)
        random.shuffle(all_images)
        num_images = int(len(all_images) * subset_percentage)
        self.image_names = all_images[:num_images]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        sketch_path = os.path.join(self.sketch_dir, img_name)
        reference_path = os.path.join(self.reference_dir, img_name)

        # Load images in RGB
        sketch_img = Image.open(sketch_path).convert("RGB")
        reference_img = Image.open(reference_path).convert("RGB")

        # Apply the standard transformation (e.g., ToTensor)
        if self.transform_standard:
            sketch = self.transform_standard(sketch_img)
            reference = self.transform_standard(reference_img)
        else:
            sketch = ToTensor()(sketch_img)
            reference = ToTensor()(reference_img)

        # Apply the identical shape transformation to both images if defined
        if self.shape_transform:
            sketch_shape_transformed, ref_shape_transformed = self.shape_transform(sketch_img, reference_img)
            # Optionally, you can apply the standard transformation to the transformed images
            if self.transform_standard:
                sketch_shape_transformed = self.transform_standard(sketch_shape_transformed)
                ref_shape_transformed = self.transform_standard(ref_shape_transformed)
            else:
                sketch_shape_transformed = ToTensor()(sketch_shape_transformed)
                ref_shape_transformed = ToTensor()(ref_shape_transformed)
        else:
            sketch_shape_transformed = sketch
            ref_shape_transformed = reference

        return sketch, reference, sketch_shape_transformed, ref_shape_transformed


import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from typing import Optional, Tuple, List


def is_image_file(filename: str) -> bool:
    """Check if file is a supported image format"""
    return any(filename.endswith(ext) for ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff'])


class ImagePreprocessingPipeline:
    """Standard image preprocessing pipeline for deep learning (compatible with academic papers)"""

    def __init__(self,
                 input_size: Tuple[int, int] = (256, 256),
                 normalize_range: Tuple[float, float] = (0.0, 1.0),
                 mean: Optional[Tuple[float, float, float]] = None,
                 std: Optional[Tuple[float, float, float]] = None,
                 is_grayscale: bool = False,
                 use_data_augmentation: bool = False):
        """
        Args:
            input_size: Target image size (H, W)
            normalize_range: Output pixel value range (min, max)
            mean: Mean values for standardization (if None, skip)
            std: Std values for standardization (if None, skip)
            is_grayscale: Convert images to grayscale if True
            use_data_augmentation: Enable data augmentation for training
        """
        self.input_size = input_size
        self.normalize_range = normalize_range
        self.mean = mean
        self.std = std
        self.is_grayscale = is_grayscale
        self.use_data_aug = use_data_augmentation

    def get_transforms(self, is_training: bool = True) -> transforms.Compose:
        """Get composed transforms for training/validation"""
        transform_list = []

        # Basic preprocessing
        if self.is_grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        # Resize (maintain aspect ratio for validation, force resize for training)
        if is_training:
            transform_list.append(transforms.Resize(self.input_size, transforms.InterpolationMode.LANCZOS))
        else:
            transform_list.append(transforms.Resize(self.input_size, transforms.InterpolationMode.LANCZOS))

        # Data augmentation (only for training)
        if is_training and self.use_data_aug:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=(-15, 15), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(self.input_size, padding=16, padding_mode='reflect'),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
            ])

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Normalize to target range
        if self.normalize_range != (0.0, 1.0):
            scale = self.normalize_range[1] - self.normalize_range[0]
            shift = self.normalize_range[0]
            transform_list.append(transforms.Lambda(lambda x: x * scale + shift))

        # Standardization (mean-std normalization)
        if self.mean is not None and self.std is not None:
            transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))

        return transforms.Compose(transform_list)


class StandardImageDataset(Dataset):
    """Standard dataset class with integrated preprocessing"""

    def __init__(self,
                 data_dir: str,
                 preprocess_pipeline: ImagePreprocessingPipeline,
                 is_training: bool = True):
        """
        Args:
            data_dir: Directory containing images
            preprocess_pipeline: Instance of ImagePreprocessingPipeline
            is_training: Whether to use training transforms
        """
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if is_image_file(f)]
        self.image_paths.sort()  # Ensure consistent ordering
        self.transform = preprocess_pipeline.get_transforms(is_training=is_training)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load image (RGB mode by default)
        img = Image.open(self.image_paths[idx]).convert('RGB')
        # Apply preprocessing
        img_tensor = self.transform(img)
        # Return tensor and filename (for evaluation)
        return img_tensor, os.path.basename(self.image_paths[idx])


# Example usage (compliant with paper code standards)
if __name__ == "__main__":
    # Configuration (can be adjusted based on dataset characteristics)
    config = {
        'input_size': (256, 256),
        'normalize_range': (0.0, 1.0),
        'mean': (0.485, 0.456, 0.406),  # ImageNet stats (for transfer learning)
        'std': (0.229, 0.224, 0.225),
        'is_grayscale': False,
        'use_data_augmentation': True,
        'train_data_dir': './dataset/train',
        'val_data_dir': './dataset/val',
        'batch_size': 16,
        'num_workers': 4
    }

    # Initialize preprocessing pipeline
    preprocess = ImagePreprocessingPipeline(
        input_size=config['input_size'],
        normalize_range=config['normalize_range'],
        mean=config['mean'],
        std=config['std'],
        is_grayscale=config['is_grayscale'],
        use_data_augmentation=config['use_data_augmentation']
    )

    # Create datasets
    train_dataset = StandardImageDataset(
        data_dir=config['train_data_dir'],
        preprocess_pipeline=preprocess,
        is_training=True
    )
    val_dataset = StandardImageDataset(
        data_dir=config['val_data_dir'],
        preprocess_pipeline=preprocess,
        is_training=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )

    # Verify pipeline (optional for paper supplementary)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    for imgs, fnames in train_loader:
        print(f"Batch shape: {imgs.shape} (dtype: {imgs.dtype})")
        print(f"Pixel value range: [{imgs.min():.3f}, {imgs.max():.3f}]")
        break
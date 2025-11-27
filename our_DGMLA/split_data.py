import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImagePairDataset(Dataset):
    """Dataset for paired reference and source images"""

    def __init__(self, ref_path, src_path, transform=None):
        self.ref_path = ref_path
        self.src_path = src_path
        self.transform = transform

        # Get sorted image paths to ensure pair consistency
        self.ref_files = sorted([os.path.join(ref_path, f) for f in os.listdir(ref_path)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.src_files = sorted([os.path.join(src_path, f) for f in os.listdir(src_path)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.ref_files)

    def __getitem__(self, idx):
        # Load and convert to RGB
        ref_img = Image.open(self.ref_files[idx]).convert('RGB')
        src_img = Image.open(self.src_files[idx]).convert('RGB')

        # Apply transforms if provided
        if self.transform:
            ref_img = self.transform(ref_img)
            src_img = self.transform(src_img)

        return ref_img, src_img


def dataset_loader(ref_path, src_path, transform=None):
    """
    Load paired image dataset.
    Args:
        ref_path: Path to reference (clean) images
        src_path: Path to source (original) images
        transform: Image transforms (optional)
    Returns:
        ImagePairDataset instance
    """
    return ImagePairDataset(ref_path, src_path, transform=transform)
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from collections import OrderedDict


def is_image_file(filename):
    """Check if file is an image by extension"""
    return any(filename.endswith(ext) for ext in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'bmp'])


class CustomDataset(Dataset):
    """Paired image dataset for training/validation"""

    def __init__(self, clean_images_path, ori_images_path, mode='train', transform=None):
        self.clean_path = clean_images_path
        self.ori_path = ori_images_path
        self.mode = mode
        self.transform = transform if transform is not None else transforms.ToTensor()

        # Load and sort image paths (ensure pair consistency)
        self.clean_files = sorted(
            [os.path.join(clean_images_path, f) for f in os.listdir(clean_images_path) if is_image_file(f)])
        self.ori_files = sorted(
            [os.path.join(ori_images_path, f) for f in os.listdir(ori_images_path) if is_image_file(f)])

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # Load image paths
        clean_path = self.clean_files[idx]
        ori_path = self.ori_files[idx]

        # Load as grayscale for discriminator training/validation
        if self.mode in ['d_train', 'd_val']:
            clean_img = Image.open(clean_path).convert('L')
        else:
            clean_img = Image.open(clean_path)
        ori_img = Image.open(ori_path)

        # Apply transforms
        if self.transform:
            clean_img = self.transform(clean_img)
            ori_img = self.transform(ori_img)

        return clean_img, ori_img


class test_loader(Dataset):
    """Dataset loader for test images"""

    def __init__(self, ori_images_path):
        super(test_loader, self).__init__()
        # Load sorted test image paths
        image_names = sorted(os.listdir(ori_images_path))
        self.image_paths = [os.path.join(ori_images_path, x) for x in image_names if is_image_file(x)]
        self.total = len(self.image_paths)

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        filename = os.path.basename(img_path)

        # Load and normalize image
        img = Image.open(img_path)
        img = (np.asarray(img) / 255.0).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img, filename


def load_checkpoint(model, weights_path):
    """Load model weights, handle DataParallel's 'module.' prefix"""
    checkpoint = torch.load(weights_path)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        # Remove 'module.' prefix for non-DataParallel inference
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
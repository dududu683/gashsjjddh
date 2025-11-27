import os
from PIL import Image
import random


def resize_images_to_target_size(src_dir, dest_dir, file_list, target_resolution=(256, 256)):
    """Resize images to target resolution and save to destination directory.

    Args:
        src_dir (str): Path to source directory containing original images
        dest_dir (str): Path to destination directory for resized images
        file_list (list): List of image filenames to process
        target_resolution (tuple): Desired output resolution (width, height)
    """
    # Create destination directory if it doesn't exist
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    # Process each image in the list
    for fname in file_list:
        src_path = os.path.join(src_dir, fname)
        dest_path = os.path.join(dest_dir, fname)

        # Open, resize with high-quality filter, and save
        with Image.open(src_path) as img:
            resized_img = img.resize(target_resolution, Image.LANCZOS)
            resized_img.save(dest_path)


def prepare_lsui_dataset():

    # Base paths configuration
    root_path = "dataset/UIEB"
    raw_input_path = os.path.join(root_path, "input")
    ground_truth_path = os.path.join(root_path, "GT")

    # Output paths for processed data
    train_raw_save_path = os.path.join(root_path, "train_256/input")
    test_raw_save_path = os.path.join(root_path, "test_256/input")
    train_gt_save_path = os.path.join(root_path, "train_256/val")
    test_gt_save_path = os.path.join(root_path, "test_256/val")

    # Get all valid image filenames (filter non-image files implicitly)
    image_filenames = [f for f in os.listdir(raw_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.seed(42)  # Fixed seed for reproducible split
    random.shuffle(image_filenames)

    # Dataset split configuration (1500 train, 200 test)
    train_split_size = 1500
    train_files = image_filenames[:train_split_size]
    test_files = image_filenames[train_split_size:train_split_size + 200]

    # Resize and save all subsets
    print("Processing training set...")
    resize_images_to_target_size(raw_input_path, train_raw_save_path, train_files)
    resize_images_to_target_size(ground_truth_path, train_gt_save_path, train_files)

    print("Processing test set...")
    resize_images_to_target_size(raw_input_path, test_raw_save_path, test_files)
    resize_images_to_target_size(ground_truth_path, test_gt_save_path, test_files)

    print("Dataset preparation completed!")
    print(f"Train set size: {len(train_files)} images")
    print(f"Test set size: {len(test_files)} images")
    print(f"Target resolution: 256x256")


if __name__ == "__main__":
    prepare_lsui_dataset()
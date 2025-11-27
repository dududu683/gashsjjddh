import torch
import torch.nn as nn
import torchvision
import torch.optim

import os
import argparse
import utils
from tqdm import tqdm

from networks import GraphEnhancedNetwork

import time


def test(config):
    # Initialize enhancement network
    enhan_net = GraphEnhancedNetwork.GraphEnhancedNet(config.block_size).cuda()
    # Load pre-trained checkpoint
    utils.load_checkpoint(enhan_net, os.path.join(config.checkpoint_path, config.net_name, 'model_epoch_60.pth'))

    print("GPU ID:", config.cudaid)
    # Set CUDA device order and visible devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid
    device_ids = [i for i in range(torch.cuda.device_count())]

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        enhan_net = nn.DataParallel(enhan_net, device_ids=device_ids)

    print(f"Test dataset path: {os.path.join(config.ori_images_path, config.dataset_name)}")
    # Load test dataset
    test_dataset = utils.test_loader(config.ori_images_path)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=True
    )

    # Create result directory
    result_dir = os.path.join(config.result_path, config.net_name, config.dataset_name)
    os.makedirs(result_dir, exist_ok=True)  # Avoid directory creation errors

    # Set network to evaluation mode
    enhan_net.eval()

    with torch.no_grad():
        for i, (img_ori, filenames) in enumerate(tqdm(test_loader)):
            # Clean GPU cache to save memory
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img_ori = img_ori.cuda()

            # Forward pass (only one input parameter required)
            enhan_image = enhan_net(img_ori)

            # Save enhanced images
            for j in range(len(enhan_image)):
                save_path = os.path.join(result_dir, os.path.basename(filenames[j]))
                torchvision.utils.save_image(enhan_image[j], save_path)


if __name__ == '__main__':
    """
    Test script for underwater image enhancement network
    Parameter explanation:
        --ori_images_path: Path to original underwater test images
    """

    # Clean initial GPU cache
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="Underwater Image Enhancement Test Script")

    # Network parameter
    parser.add_argument('--block_size', type=int, default=16, help="Downsampling block size")

    # Input parameters
    parser.add_argument('--net_name', type=str, default="", help="Name of the trained network")
    parser.add_argument('--dataset_name', type=str, default="data", help="Name of the test dataset")
    parser.add_argument('--ori_images_path', type=str, default="testdata/raw",
                        help="Path to original test images")

    # Test parameters
    parser.add_argument('--batch_size', type=int, default=1, help="Test batch size")
    parser.add_argument('--num_workers', type=int, default=6, help="Number of data loading workers")
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint",
                        help="Path to pre-trained checkpoint")
    parser.add_argument('--result_path', type=str, default="results/60e/",
                        help="Path to save enhanced results")
    parser.add_argument('--cudaid', type=str, default="0", help="Select CUDA device ID (0-7)")

    config = parser.parse_args()

    # Create parent directories if they don't exist
    if not os.path.exists(os.path.join(config.result_path, config.net_name)):
        os.mkdir(os.path.join(config.result_path, config.net_name))
    if not os.path.exists(os.path.join(config.result_path, config.net_name, config.dataset_name)):
        os.mkdir(os.path.join(config.result_path, config.net_name, config.dataset_name))

    # Record test start time
    start_time = time.time()
    test(config)
    # Print total test time
    print(f"Total test time: {time.time() - start_time:.2f} seconds")
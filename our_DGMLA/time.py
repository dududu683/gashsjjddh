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
    utils.load_checkpoint(enhan_net, os.path.join(config.checkpoint_path, config.net_name, 'model_epoch_60.pth'))

    print("GPU ID:", config.cudaid)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid
    device_ids = [i for i in range(torch.cuda.device_count())]

    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        enhan_net = nn.DataParallel(enhan_net, device_ids=device_ids)

    print(f"Test dataset path: {os.path.join(config.ori_images_path, config.dataset_name)}")
    test_dataset = utils.test_loader(config.ori_images_path)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=True
    )

    result_dir = os.path.join(config.result_path, config.net_name, config.dataset_name)

    enhan_net.eval()

    # Metrics for time and memory statistics
    total_time = 0.0
    total_images = 0
    memory_list = []
    start_total = time.time()  # Record total start time

    with torch.no_grad():
        for i, (img_ori, filenames) in enumerate(tqdm(test_loader)):
            # Reset memory stats and clear cache
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            start_time = time.time()

            img_ori = img_ori.cuda()

            # Model inference
            enhan_image = enhan_net(img_ori)

            end_time = time.time()

            # Accumulate processing time
            batch_time = end_time - start_time
            total_time += batch_time
            total_images += img_ori.size(0)  # Support variable batch size

            # Get current batch memory usage (MB)
            current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            memory_list.append(current_memory)

            # Save enhanced results
            for j in range(len(enhan_image)):
                save_path = os.path.join(result_dir, os.path.basename(filenames[j]))
                torchvision.utils.save_image(enhan_image[j], save_path)

    # Calculate and print statistics
    avg_time_per_image = total_time / total_images
    avg_memory = sum(memory_list) / len(memory_list)
    max_memory = max(memory_list)

    print(f"\nTotal images processed: {total_images}")
    print(f"Average time per image: {avg_time_per_image:.4f} seconds")
    print(f"Average memory usage: {avg_memory:.2f} MB")
    print(f"Maximum memory usage: {max_memory:.2f} MB")
    print(f"Total execution time: {time.time() - start_total:.2f} seconds")


if __name__ == '__main__':
    """
    Test script for underwater image enhancement network
    Parameter:
        --ori_images_path: Path to original underwater test images
    """

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--block_size', type=int, default=16)

    # Input parameters
    parser.add_argument('--net_name', type=str, default="")
    parser.add_argument('--dataset_name', type=str, default="data")
    parser.add_argument('--ori_images_path', type=str, default="testdata/raw")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint")
    parser.add_argument('--result_path', type=str, default="results/1/")
    parser.add_argument('--cudaid', type=str, default="0", help="Select CUDA device ID (0-7)")

    config = parser.parse_args()

    # Create result directories
    result_net_dir = os.path.join(config.result_path, config.net_name)
    if not os.path.exists(result_net_dir):
        os.makedirs(result_net_dir, exist_ok=True)

    result_dataset_dir = os.path.join(result_net_dir, config.dataset_name)
    if not os.path.exists(result_dataset_dir):
        os.makedirs(result_dataset_dir, exist_ok=True)

    start_time = time.time()
    test(config)
    print(f"Final execution time: {time.time() - start_time:.2f} seconds")
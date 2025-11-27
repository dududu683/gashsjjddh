import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import time
import utils
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from loss import Charbonnier_Loss, SSIM_Loss, torchPSNR
from networks import GraphEnhancedNetwork
import torchvision.transforms as transforms
import torch.nn.functional as F


def train(config):
    start_time = time.time()  # Record total training start time

    # Initialize model
    GraphEnhancedNet = GraphEnhancedNetwork.GraphEnhancedNet(config.block_size).cuda()

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        device_ids = [i for i in range(torch.cuda.device_count())]
        GraphEnhancedNet = nn.DataParallel(GraphEnhancedNet, device_ids=device_ids)

    # Load datasets
    dataset_load_start = time.time()
    train_dataset = utils.CustomDataset(config.enhan_images_path, config.ori_images_path, mode="train",
                                        transform=transforms.ToTensor())
    val_dataset = utils.CustomDataset(config.val_enhan_images_path, config.val_ori_images_path, mode="val",
                                      transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.train_batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.val_batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             pin_memory=True)
    print(f"Dataset loading time: {time.time() - dataset_load_start:.2f} seconds")

    # Optimizer and loss functions
    optimizer = optim.Adam(GraphEnhancedNet.parameters(), lr=config.lr)

    # Learning rate scheduler
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            config.num_epochs - warmup_epochs,
                                                            eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer,
                                       multiplier=1,
                                       total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)

    criterion_char = Charbonnier_Loss()
    criterion_ssim = SSIM_Loss()

    # Training records
    best_psnr = 0
    best_epoch = 0
    checkpoint_dir = os.path.join(config.checkpoint_path, config.net_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Gradient loss function
    def gradient_loss(output, target):
        grad_x_output = torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])
        grad_y_output = torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :])
        grad_x_target = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        grad_y_target = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        return F.l1_loss(grad_x_output, grad_x_target) + F.l1_loss(grad_y_output, grad_y_target)

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()
        train_loss = []
        val_psnr = []
        print("*" * 30 + f"Epoch {epoch} Training" + "*" * 30 + '\n')

        # Training phase
        GraphEnhancedNet.train()
        for img_clean, img_ori in tqdm(train_loader):
            img_clean = img_clean.cuda()
            img_ori = img_ori.cuda()

            # Dynamic graph construction is completed inside the model
            enhanced_image = GraphEnhancedNet(img_ori)  # No longer pass adjacency matrix

            # Loss calculation
            char_loss = criterion_char(img_clean, enhanced_image)
            ssim_loss = criterion_ssim(img_clean, enhanced_image)
            l2_loss = F.mse_loss(img_clean, enhanced_image)
            ssim_loss = 1.0 - ssim_loss  # Convert to loss form
            grad_loss = gradient_loss(enhanced_image, img_clean)
            sum_loss = 1.0 * char_loss + 1.0 * l2_loss + 0.3 * ssim_loss + 0.1 * grad_loss

            # Backpropagation
            optimizer.zero_grad()
            sum_loss.backward(retain_graph=True)  # [Key modification] Add retain_graph=True
            torch.nn.utils.clip_grad_norm_(GraphEnhancedNet.parameters(), config.grad_clip_norm)
            optimizer.step()

            # Record loss and free memory
            train_loss.append(sum_loss.item())
            del enhanced_image, char_loss, ssim_loss, l2_loss, grad_loss, sum_loss
            torch.cuda.empty_cache()

        # Record training loss
        mean_loss = np.mean(train_loss)
        with open(os.path.join(checkpoint_dir, "loss.log"), "a+") as f:
            f.write(f"Epoch {epoch} average loss: {mean_loss:.6f}\n")

        # Validation phase
        GraphEnhancedNet.eval()
        val_start_time = time.time()
        with torch.no_grad():
            for img_clean, img_ori in val_loader:
                img_clean = img_clean.cuda()
                img_ori = img_ori.cuda()

                # Dynamic graph construction is completed inside the model
                enhanced_image = GraphEnhancedNet(img_ori)  # No longer pass adjacency matrix

                psnr = torchPSNR(enhanced_image, img_clean)
                val_psnr.append(psnr.item())

        mean_val_psnr = np.mean(val_psnr)
        print(f"Validation phase time: {time.time() - val_start_time:.2f} seconds")

        # Update best model
        if mean_val_psnr > best_psnr:
            best_psnr = mean_val_psnr
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'state_dict': GraphEnhancedNet.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(checkpoint_dir, "model_best.pth"))

        # Update learning rate
        scheduler.step()

        # Print log
        print("------------------------------------------------------------------")
        print(f"Epoch: {epoch}\tTime: {time.time() - epoch_start_time:.2f}s\t"
              f"Loss: {mean_loss:.6f}\tLearning Rate: {scheduler.get_lr()[0]:.6f}")
        print("------------------------------------------------------------------")
        print(f"[Epoch {epoch} PSNR: {mean_val_psnr:.4f} --- "
              f"Best Epoch {best_epoch} Best PSNR {best_psnr:.4f}]")

        with open(os.path.join(checkpoint_dir, "val_PSNR.log"), "a+") as f:
            f.write(f"[Epoch {epoch} PSNR: {mean_val_psnr:.4f} --- "
                    f"Best Epoch {best_epoch} Best PSNR {best_psnr:.4f}]\n")

        # Save latest model
        torch.save({
            'epoch': epoch,
            'state_dict': GraphEnhancedNet.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join(checkpoint_dir, "model_latest.pth"))

        # Save model periodically
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': GraphEnhancedNet.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth"))

    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    print(f"Training completed! Best PSNR: {best_psnr:.4f} (Epoch {best_epoch})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Block size parameter
    parser.add_argument('--block_size', type=int, default=16)

    # Input parameters
    parser.add_argument('--net_name', type=str, default="")
    parser.add_argument('--enhan_images_path', type=str, default="dataset/train/target/")
    parser.add_argument('--ori_images_path', type=str, default="dataset/train/raw/")
    parser.add_argument('--val_enhan_images_path', type=str, default="dataset/val/target/")
    parser.add_argument('--val_ori_images_path', type=str, default="dataset/val/raw/")

    # Training parameters
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint/")
    parser.add_argument('--cudaid', type=str, default="0", help="Select CUDA device ID (0-7)")

    config = parser.parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid

    # Start training
    train(config)
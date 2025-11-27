import os
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_avg_psnr_ssim(ref_dir, res_dir):
    # Get all image filenames from reference directory
    images = [f for f in os.listdir(ref_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_psnr = 0
    total_ssim = 0
    num_images = len(images)

    for image_name in images:
        # Read reference and result images
        img_ref = cv2.imread(os.path.join(ref_dir, image_name))
        img_res = cv2.imread(os.path.join(res_dir, image_name))

        # Calculate PSNR
        psnr_score = psnr(img_ref, img_res)

        # Calculate SSIM (channel_axis=-1 for color images)
        ssim_score = ssim(img_ref, img_res, channel_axis=-1)

        # Print metrics for each image
        print(f"{image_name}: PSNR = {psnr_score:.4f} dB, SSIM = {ssim_score:.4f}")

        # Accumulate metrics
        total_psnr += psnr_score
        total_ssim += ssim_score

    # Calculate average metrics
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images

    # Print average results
    print(f"\nAverage PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    return avg_psnr, avg_ssim

# Specify paths to reference and result directories
ref_dir = './data/test/val/'
res_dir = './results/UIEB/data/'

# Execute calculation
calculate_avg_psnr_ssim(ref_dir, res_dir)
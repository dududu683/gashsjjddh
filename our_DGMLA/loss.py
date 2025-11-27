import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.autograd import Variable
from math import exp



# Gradient loss for edge preservation
class Gradient_Loss(nn.Module):
    def __init__(self):
        super(Gradient_Loss, self).__init__()
        self.kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device='cuda').view(1, 1, 3, 3)
        self.kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device='cuda').view(1, 1, 3, 3)

    def forward(self, x, y):
        c = x.shape[1]
        kernel_x = self.kernel_x.expand(c, 1, 3, 3)
        kernel_y = self.kernel_y.expand(c, 1, 3, 3)

        grad_x_x = F.conv2d(x, kernel_x, groups=c, padding=1)
        grad_y_x = F.conv2d(x, kernel_y, groups=c, padding=1)
        grad_x_y = F.conv2d(y, kernel_x, groups=c, padding=1)
        grad_y_y = F.conv2d(y, kernel_y, groups=c, padding=1)

        loss_x = F.l1_loss(grad_x_x, grad_x_y)
        loss_y = F.l1_loss(grad_y_x, grad_y_y)
        return loss_x + loss_y


# Laplacian loss for detail preservation
class Laplacian_Loss(nn.Module):
    def __init__(self):
        super(Laplacian_Loss, self).__init__()
        self.kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device='cuda').view(1, 1, 3, 3)

    def forward(self, x, y):
        c = x.shape[1]
        kernel = self.kernel.expand(c, 1, 3, 3)

        lap_x = F.conv2d(x, kernel, groups=c, padding=1)
        lap_y = F.conv2d(y, kernel, groups=c, padding=1)

        return F.mse_loss(lap_x, lap_y)


class VGG_Loss(nn.Module):
    def __init__(self, n_layers=5):
        super(VGG_Loss, self).__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        vgg = torchvision.models.vgg19(pretrained=True).features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.cuda())
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().cuda()

    def forward(self, x, y):
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            x = layer(x)
            with torch.no_grad():
                y = layer(y)
            loss += weight * self.criterion(x, y)

        return loss


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# Create 2D Gaussian window from 1D kernel
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def torchFSIM(tar_img, prd_img):
    fsim_value = (2 * tar_img * prd_img + 0.01) / (tar_img ** 2 + prd_img ** 2 + 0.01)
    return fsim_value.mean()


def torchSSIM(tar_img, prd_img):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mean_tar = tar_img.mean()
    mean_prd = prd_img.mean()
    cov = ((tar_img - mean_tar) * (prd_img - mean_prd)).mean()
    var_tar = ((tar_img - mean_tar) ** 2).mean()
    var_prd = ((prd_img - mean_prd) ** 2).mean()
    ssim_value = (2 * mean_tar * mean_prd + C1) * (2 * cov + C2) / (mean_tar ** 2 + mean_prd ** 2 + C1) / (
                var_tar + var_prd + C2)
    return ssim_value


def torchMSE(tar_img, prd_img):
    imdff = prd_img - tar_img
    mse_value = (imdff ** 2).mean()
    return mse_value


# Huber loss (smooth L1 loss)
class Huber_Loss(nn.Module):
    def __init__(self, delta=1.0):
        super(Huber_Loss, self).__init__()
        self.delta = delta

    def forward(self, x, y):
        diff = torch.abs(x - y)
        loss = torch.where(diff < self.delta, 0.5 * diff ** 2, self.delta * (diff - 0.5 * self.delta))
        return torch.mean(loss)


# Total variation loss for image smoothness
class TV_Loss(nn.Module):
    def forward(self, x):
        tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return tv_h + tv_w



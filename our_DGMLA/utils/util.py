import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
import numpy as np


class ChannelAttentionModule(nn.Module):
    """Channel-wise attention module with squeeze-and-excitation mechanism"""
    def __init__(self, num_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention_mlp = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_channels // 16, num_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_mask = self.avg_pool(x)
        attention_mask = self.attention_mlp(attention_mask)
        return x * attention_mask


class ResidualEnhanceBlock(nn.Module):
    """Residual block with channel attention integration"""
    def __init__(self, num_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=0),
            ChannelAttentionModule(num_channels)
        )

    def forward(self, x):
        residual = self.conv_block(x)
        return x + residual


class UpsampleBlock(nn.Module):
    """Bilinear upsampling block (scale factor = 2)"""
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True
        )

    def forward(self, x):
        return self.upsample(x)


class ConvolutionBlock(nn.Module):
    """Double convolution block with reflection padding and LeakyReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_layers(x)


class LatentCodeEstimator(nn.Module):
    """Estimate latent codes (u, s) with mean and std from feature maps"""
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.global_mean_proj = nn.Conv2d(128, 2 * latent_dim, kernel_size=1, padding=0)
        self.global_std_proj = nn.Conv2d(128, 2 * latent_dim, kernel_size=1, padding=0)

    def forward(self, x):
        # Compute global mean encoding
        mean_feat = x.mean(dim=[2, 3], keepdim=True)
        mean_proj = self.global_mean_proj(mean_feat)
        mean_proj = mean_proj.view(-1, 2 * self.latent_dim)
        u_mu = mean_proj[:, :self.latent_dim]
        u_std = torch.exp(mean_proj[:, self.latent_dim:])
        u_dist = Independent(Normal(loc=u_mu, scale=u_std), 1)

        # Compute global std encoding
        std_feat = x.std(dim=[2, 3], keepdim=True)
        std_proj = self.global_std_proj(std_feat)
        std_proj = std_proj.view(-1, 2 * self.latent_dim)
        s_mu = std_proj[:, :self.latent_dim]
        s_std = torch.exp(std_proj[:, self.latent_dim:])
        s_dist = Independent(Normal(loc=s_mu, scale=s_std), 1)

        return u_dist, s_dist, u_mu, s_mu, u_std, s_std
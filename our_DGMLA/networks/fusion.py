import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATConv


# ECA Attention Module
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


# Local Feature Aggregation Module
class LocalFeatureAggregation(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(LocalFeatureAggregation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


# Enhanced GCN with GAT integration
class EnhancedGCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c, heads=4, dropout=0.4):
        super(EnhancedGCN, self).__init__()
        self.gat1 = GATConv(in_c, hid_c, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hid_c * heads, out_c, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return x


# Basic Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + residual


# Progressive Feature Fusion Module
class ProgressiveFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProgressiveFusion, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.relu(x)


# Multi-Scale Context Aggregator (PPM-based)
class MultiScaleContextAggregator(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(MultiScaleContextAggregator, self).__init__()
        self.pool_sizes = pool_sizes
        self.pool_branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((size, size)),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1),
                nn.ReLU(inplace=True)
            ) for size in pool_sizes
        ])
        self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2:]
        features = [x]
        for branch in self.pool_branches:
            feat = branch(x)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)
            features.append(feat)
        out = torch.cat(features, dim=1)
        out = self.final_conv(out)
        return out


# Convolution layer wrapper
def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# Basic convolution block with optional BN and ReLU
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# Channel pooling (max + avg)
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


# Spatial Attention Layer
class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


# Channel Attention Layer
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Spatial-Channel Attention (SCA) Module
class SCA(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8,
                 bias=False, bn=False, act=nn.PReLU(), res_scale=1):
        super(SCA, self).__init__()

        # Feature extraction branch
        modules_body = [
            conv(n_feat, n_feat * 2, kernel_size, bias=bias),
            act,
            conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        ]
        self.body = nn.Sequential(*modules_body)

        # Spatial attention
        self.SA = nn.Sequential(
            spatial_attn_layer(kernel_size=5),
            nn.InstanceNorm2d(n_feat)
        )

        # Channel attention
        self.CA = nn.Sequential(
            ca_layer(n_feat, reduction, bias=bias),
            nn.Dropout2d(0.05)
        )

        # Dynamic weight fusion
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(n_feat)
        )

        # Residual enhancement
        self.gamma = nn.Parameter(torch.tensor([0.1]))
        self.smooth = conv(n_feat, n_feat, 5)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        res = self.body(x)

        # Dual attention processing
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)

        # Dynamic weighted fusion
        fused = torch.cat([
            self.alpha * sa_branch,
            (1 - self.alpha) * ca_branch
        ], dim=1)

        # Feature integration
        res = self.conv1x1(fused)

        # Residual connection with smoothing
        res = self.gamma * self.smooth(res)
        output = identity + res

        return output
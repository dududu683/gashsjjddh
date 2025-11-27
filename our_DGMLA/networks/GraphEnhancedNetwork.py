import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from networks.LAAT import LAAT
from networks.fusion import MultiScaleContextAggregator
import time
import math

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("thop not installed, FLOPs calculation skipped")


class EnhancedRB(nn.Module):
    def __init__(self, channels):
        super(EnhancedRB, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(4, channels)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.ca(out) * out
        out += identity
        return F.relu(out)


class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c, dropout=0.4):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_c, hid_c)
        self.conv2 = GCNConv(hid_c, out_c)
        self.dropout = dropout
        self.norm = nn.LayerNorm(hid_c)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.norm(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


class FeatureEnhancer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, in_channels)
        self.activation = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        identity = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + identity
        return self.activation(out)


class GraphEnhancedNet(nn.Module):
    def __init__(self, down_size=16, in_c=3, out_c=3, hid_size=1024, bc=64, k_neighbors=8, sim_threshold=0.5):
        super(GraphEnhancedNet, self).__init__()
        self.k_neighbors = k_neighbors
        self.base_sim_threshold = sim_threshold
        self.down_size = down_size

        assert bc % 4 == 0, "Base channels must be divisible by 4"
        assert hid_size % 16 == 0, "Hidden size must be divisible by 16"

        # Encoder
        self.conv_ini = nn.Sequential(
            nn.Conv2d(in_c, bc, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(bc, bc, 3, 1, 1, bias=True),
            nn.GroupNorm(4, bc)
        )

        self.res_blocks = nn.Sequential(
            EnhancedRB(bc),
            EnhancedRB(bc),
            EnhancedRB(bc),
            EnhancedRB(bc)
        )

        self.feature_enhancer = FeatureEnhancer(bc)
        self.down_conv = nn.Conv2d(bc, hid_size, kernel_size=down_size, stride=down_size, padding=0)

        # Graph convolution
        self.GCN = GCN(hid_size, 1500, hid_size)
        self.post_gcn_norm = nn.LayerNorm(hid_size)

        # Attention module
        self.norm_before_laat = nn.LayerNorm(hid_size)
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.laat = LAAT(
            input_dim=hid_size,
            num_heads=16,
            q_lora_rank=256,
            kv_lora_rank=128,
            qk_static_dim=128,
            qk_rotary_dim=64,
            v_dim=128,
            max_seq_length=(256 // down_size) ** 2,
            max_batch_size=16,
            attention_mode='efficient'
        )

        self.laat_proj = nn.Sequential(
            nn.Linear(hid_size, hid_size * 2),
            nn.GELU(),
            nn.Linear(hid_size * 2, hid_size)
        )

        # Decoder
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(hid_size, hid_size // 2, 4, 2, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hid_size // 2, hid_size // 2, 3, 1, 1),
            nn.GroupNorm(4, hid_size // 2)
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(hid_size // 2, hid_size // 4, 4, 2, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hid_size // 4, hid_size // 4, 3, 1, 1),
            nn.GroupNorm(4, hid_size // 4)
        )
        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose2d(hid_size // 4, hid_size // 8, 4, 2, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hid_size // 8, hid_size // 8, 3, 1, 1),
            nn.GroupNorm(4, hid_size // 8)
        )
        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose2d(hid_size // 8, hid_size // 16, 4, 2, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hid_size // 16, hid_size // 16, 3, 1, 1),
            nn.GroupNorm(4, hid_size // 16)
        )

        # Skip connection
        self.skip_conv = nn.Sequential(
            nn.Conv2d(bc, hid_size // 16, 1),
            nn.GroupNorm(4, hid_size // 16)
        )

        # Multi-scale aggregation
        self.mca = MultiScaleContextAggregator(hid_size // 16, pool_sizes=[1, 2, 3, 6])

        # Final output
        self.final_conv = nn.Sequential(
            EnhancedRB(hid_size // 16),
            nn.Conv2d(hid_size // 16, hid_size // 32, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hid_size // 32, out_c, 3, 1, 1, bias=True)
        )


    def forward(self, x):
        # Encoding
        e1 = self.conv_ini(x)
        e1 = self.res_blocks(e1)
        e1 = self.feature_enhancer(e1)

        # Downsampling
        d1 = self.down_conv(e1)

        # Build graph
        edge_index, edge_weight = self._build_dynamic_graph(d1)

        # Graph convolution
        B, C, H, W = d1.shape
        N = H * W
        gcn_input = d1.permute(0, 2, 3, 1).reshape(-1, C)
        e2 = self.GCN(gcn_input, edge_index, edge_weight)
        e2 = e2.view(B, H, W, C).permute(0, 3, 1, 2)
        e2 = self.post_gcn_norm(e2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + d1

        # Attention processing
        laat_input = e2.permute(0, 2, 3, 1)
        laat_input = self.norm_before_laat(laat_input)
        laat_input = laat_input.reshape(B, -1, self.laat.dim)

        laat_output = self.laat(laat_input, start_pos=0)
        laat_output = self.laat_proj(laat_output)
        fused = e2 + laat_output.view(e2.shape) * self.scale

        # Decoding
        e3 = self.up_conv1(fused)
        e3 = self.up_conv2(e3)
        e3 = self.up_conv3(e3)
        e3 = self.up_conv4(e3)

        # Skip connection
        skip = self.skip_conv(e1)
        e3 = F.interpolate(e3, size=skip.shape[2:], mode='bilinear', align_corners=False)
        e3 = e3 + skip

        # Final processing
        e3 = self.mca(e3)
        out = self.final_conv(e3)
        return torch.sigmoid(out)


def calculate_graph_flops(model, x):
    with torch.no_grad():
        e1 = model.conv_ini(x)
        e1 = model.res_blocks(e1)
        e1 = model.feature_enhancer(e1)
        d1 = model.down_conv(e1)

        B, C, H, W = d1.shape
        N = H * W

        # Cosine similarity FLOPs
        sim_matrix_flops = N * N * C
        # Top-k FLOPs (approximate)
        k = min(model.k_neighbors + 1, N)
        topk_flops = N * N * math.log2(k) if k > 1 else 0

        return sim_matrix_flops + topk_flops


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize model
    model = GraphEnhancedNet(
        down_size=16,
        hid_size=1024,
        bc=64,
        k_neighbors=8,
        sim_threshold=0.5
    ).to(device)

    # Test input
    batch_size = 2
    input_size = 256
    x = torch.randn(batch_size, 3, input_size, input_size, requires_grad=True, device=device)
    print(f"Input shape: {x.shape}")

    # Calculate parameters
    total_params = sum([param.nelement() for param in model.parameters()])
    trainable_params = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print(f"\nTotal params: {total_params / 1e6:.2f} M")
    print(f"Trainable params: {trainable_params / 1e6:.2f} M")

    # Calculate FLOPs
    if THOP_AVAILABLE:
        try:
            print("\nCalculating FLOPs...")
            flops, params = profile(model, inputs=(x,), verbose=False)
            flops_formatted, _ = clever_format([flops, 0], "%.3f")
            print(f"Core modules FLOPs: {flops_formatted}")

            graph_flops = calculate_graph_flops(model, x)
            graph_flops_formatted, _ = clever_format([graph_flops, 0], "%.3f")
            print(f"Graph construction FLOPs: {graph_flops_formatted}")

            total_flops = flops + graph_flops
            total_flops_formatted, _ = clever_format([total_flops, 0], "%.3f")
            print(f"Total FLOPs: {total_flops_formatted}")
        except Exception as e:
            print(f"FLOPs calculation failed: {e}")
    else:
        print("\nthop not installed, FLOPs calculation skipped")

    # Inference test
    torch.cuda.empty_cache()
    print("\nTesting inference performance...")

    # Forward pass
    start_time = time.time()
    output = model(x)
    forward_time = time.time() - start_time
    print(f"Output shape: {output.shape}")
    print(f"Forward time: {forward_time:.4f} s")
    print(f"Forward FPS: {batch_size / forward_time:.2f}")

    # Backward pass
    loss = F.mse_loss(output, torch.randn_like(output))
    start_time = time.time()
    loss.backward()
    backward_time = time.time() - start_time
    print(f"Backward time: {backward_time:.4f} s")
    print(f"Total (forward+backward) time: {forward_time + backward_time:.4f} s")

    # GPU memory usage
    if torch.cuda.is_available():
        allocated_mem = torch.cuda.memory_allocated() / 1e6
        max_allocated_mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"\nAllocated GPU memory: {allocated_mem:.2f} MB")
        print(f"Peak GPU memory: {max_allocated_mem:.2f} MB")
        torch.cuda.empty_cache()
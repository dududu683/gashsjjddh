import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNormalization(nn.Module):
    def __init__(self, hidden_dim, epsilon=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.eps = epsilon

    def forward(self, x):
        x_float = x.float()
        squared_x = x_float.pow(2)
        var = squared_x.mean(dim=-1, keepdim=True)  # Compute variance
        normalized_x = x_float * torch.rsqrt(var + self.eps)  # RMS normalization
        return self.scale * normalized_x.float()


def half_rotate(x):
    half_dim = x.size(-1) // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)  # Rotate half dimensions


def rotary_pos_emb_apply(q, k, cos_mat, sin_mat, unsqueeze_dim=2):
    cos_expanded = cos_mat.unsqueeze(unsqueeze_dim)
    sin_expanded = sin_mat.unsqueeze(unsqueeze_dim)

    q_rot = q * cos_expanded + half_rotate(q) * sin_expanded  # Apply rotary to query
    k_rot = k * cos_expanded + half_rotate(k) * sin_expanded  # Apply rotary to key
    return q_rot, k_rot


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, emb_dim, max_sequence_len=1024):
        super().__init__()
        self.embedding_dim = emb_dim
        self.max_len = max_sequence_len

        frequency_steps = torch.arange(0, emb_dim, 2, dtype=torch.float32)
        self.inverse_freq = 1.0 / (10000 ** (frequency_steps / emb_dim))  # Frequency scale

        timesteps = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(1)
        freq_bands = timesteps @ self.inverse_freq.unsqueeze(0)
        self.register_buffer("cos_vals", torch.cat((freq_bands.cos(), freq_bands.cos()), dim=-1), persistent=False)
        self.register_buffer("sin_vals", torch.cat((freq_bands.sin(), freq_bands.sin()), dim=-1), persistent=False)

    def forward(self, q, k):
        seq_len_q = q.shape[1]
        cos_slice = self.cos_vals[:seq_len_q].unsqueeze(0)  # Get cos for current seq
        sin_slice = self.sin_vals[:seq_len_q].unsqueeze(0)  # Get sin for current seq
        return rotary_pos_emb_apply(q, k, cos_slice, sin_slice)




if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(1234)

    # Config
    batch_size = 8
    sequence_length = 64
    hidden_dimension = 2048

    # Generate input
    input_features = torch.randn(batch_size, sequence_length, hidden_dimension)

    # Model params
    model_params = {
        'input_dim': hidden_dimension,
        'num_heads': 8,
        'q_lora_rank': 64,
        'kv_lora_rank': 32,
        'qk_static_dim': 128,
        'qk_rotary_dim': 32,
        'v_dim': 128,
        'max_seq_length': 256,
        'max_batch_size': 16
    }

    # Init model
    attention_module = LAAT(**model_params)

    # Forward passes
    start_positions = [0, 32, 64]
    for idx, start_pos in enumerate(start_positions):
        output = attention_module(input_features, start_pos=start_pos)

        # Print info
        print(f"Forward Pass {idx + 1}:")
        print(f"  Input Shape: {input_features.shape}")
        print(f"  Start Position: {start_pos}")
        print(f"  Output Shape: {output.shape}")
        print(f"  Output Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")

        # Cache info
        cache_shape = attention_module.kv_feature_cache.shape
        print(f"  KV Cache Shape: {cache_shape}")
        print(f"  Cache Usage: {min(start_pos + sequence_length, cache_shape[1])}/{cache_shape[1]}")
        print("-" * 50)

    # Test with mask
    attention_mask = torch.ones(batch_size, sequence_length, sequence_length)
    attention_mask[:, :, 40:] = -1e9  # Mask positions
    masked_output = attention_module(input_features, start_pos=0, mask=attention_mask)
    print("Test with Attention Mask:")
    print(f"  Masked Output Shape: {masked_output.shape}")
    print(f"  Masked Output Mean: {masked_output.mean().item():.4f}")

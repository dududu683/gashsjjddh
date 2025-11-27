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
        var = squared_x.mean(dim=-1, keepdim=True)
        normalized_x = x_float * torch.rsqrt(var + self.eps)
        return self.scale * normalized_x.float()


def half_rotate(x):
    half_dim = x.size(-1) // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def rotary_pos_emb_apply(q, k, cos_mat, sin_mat, unsqueeze_dim=2):
    cos_expanded = cos_mat.unsqueeze(unsqueeze_dim)
    sin_expanded = sin_mat.unsqueeze(unsqueeze_dim)

    q_rot = q * cos_expanded + half_rotate(q) * sin_expanded
    k_rot = k * cos_expanded + half_rotate(k) * sin_expanded
    return q_rot, k_rot


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, emb_dim, max_sequence_len=1024):
        super().__init__()
        self.embedding_dim = emb_dim
        self.max_len = max_sequence_len

        frequency_steps = torch.arange(0, emb_dim, 2, dtype=torch.float32)
        self.inverse_freq = 1.0 / (10000 ** (frequency_steps / emb_dim))

        timesteps = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(1)
        freq_bands = timesteps @ self.inverse_freq.unsqueeze(0)
        self.register_buffer("cos_vals", torch.cat((freq_bands.cos(), freq_bands.cos()), dim=-1), persistent=False)
        self.register_buffer("sin_vals", torch.cat((freq_bands.sin(), freq_bands.sin()), dim=-1), persistent=False)

    def forward(self, q, k):
        seq_len_q = q.shape[1]
        cos_slice = self.cos_vals[:seq_len_q].unsqueeze(0)
        sin_slice = self.sin_vals[:seq_len_q].unsqueeze(0)
        return rotary_pos_emb_apply(q, k, cos_slice, sin_slice)


class LAAT(nn.Module):
    def __init__(self,
                 input_dim,
                 num_heads,
                 q_lora_rank,
                 kv_lora_rank,
                 qk_static_dim,
                 qk_rotary_dim,
                 v_dim,
                 max_seq_length,
                 max_batch_size,
                 attention_mode):
        super().__init__()
        self.dim = input_dim
        self.num_heads = num_heads
        self.q_rank = q_lora_rank
        self.kv_rank = kv_lora_rank
        self.qk_static = qk_static_dim
        self.qk_rotary = qk_rotary_dim
        self.qk_total_dim = self.qk_static + self.qk_rotary
        self.v_head_dim = v_dim
        self.mode = attention_mode
        self.max_seq_len = max_seq_length
        self.max_bs = max_batch_size

        # Query projection layers
        self.q_down_proj = nn.Linear(self.dim, self.q_rank)
        self.q_norm = RMSNormalization(self.q_rank)
        self.q_up_proj = nn.Linear(self.q_rank, self.num_heads * self.qk_total_dim)

        # KV projection layers
        self.kv_down_proj = nn.Linear(self.dim, self.kv_rank + self.qk_rotary)
        self.kv_norm = RMSNormalization(self.kv_rank)
        self.kv_up_proj = nn.Linear(self.kv_rank, self.num_heads * (self.qk_static + self.v_head_dim))

        # Output projection
        self.output_proj = nn.Linear(self.num_heads * self.v_head_dim, self.dim)

        # Rotary position embedding
        self.rotary_embedding = RotaryPositionEmbedding(self.qk_rotary)

        # Cache initialization
        if self.mode == 'low':
            self.register_buffer(
                'key_cache',
                torch.zeros(self.max_bs, self.max_seq_len, self.num_heads, self.qk_total_dim),
                persistent=False
            )
            self.register_buffer(
                'value_cache',
                torch.zeros(self.max_bs, self.max_seq_len, self.num_heads, self.v_head_dim),
                persistent=False
            )
        else:
            self.register_buffer(
                'kv_feature_cache',
                torch.zeros(self.max_bs, self.max_seq_len, self.kv_rank),
                persistent=False
            )
            self.register_buffer(
                'rot_emb_cache',
                torch.zeros(self.max_bs, self.max_seq_len, self.qk_rotary),
                persistent=False
            )

    def forward(self, x, start_pos: int, mask=None):
        bs, seq_len, _ = x.shape
        end_pos = start_pos + seq_len

        # Query processing
        q_proj = self.q_down_proj(x)
        q_normed = self.q_norm(q_proj)
        q_expanded = self.q_up_proj(q_normed)
        q_reshaped = q_expanded.reshape(bs, seq_len, self.num_heads, self.qk_total_dim)
        q_static, q_rot = torch.split(
            q_reshaped,
            [self.qk_static, self.qk_rotary],
            dim=-1
        )

        # KV processing
        kv_combined = self.kv_down_proj(x)
        kv_features, k_rot_raw = torch.split(
            kv_combined,
            [self.kv_rank, self.qk_rotary],
            dim=-1
        )

        # Apply rotary position embedding
        k_rot_expanded = k_rot_raw.unsqueeze(dim=2)
        q_rot_processed, k_rot_processed = self.rotary_embedding(q_rot, k_rot_expanded)

        if self.mode == 'low':
            # Combine query components
            q_final = torch.cat([q_static, q_rot_processed], dim=-1)

            # KV upsampling and splitting
            kv_normed = self.kv_norm(kv_features)
            kv_expanded = self.kv_up_proj(kv_normed)
            kv_reshaped = kv_expanded.reshape(bs, seq_len, self.num_heads, self.qk_static + self.v_head_dim)
            k_static, v_final = torch.split(kv_reshaped, [self.qk_static, self.v_head_dim], dim=-1)

            # Combine key components
            k_final = torch.cat([k_static, k_rot_processed.expand(bs, seq_len, self.num_heads, self.qk_rotary)], dim=-1)

            # Update cache
            self.key_cache[:bs, start_pos:end_pos, ...] = k_final
            self.value_cache[:bs, start_pos:end_pos, ...] = v_final

            # Compute attention scores
            attn_scores = torch.matmul(
                q_final.transpose(1, 2),
                self.key_cache[:bs, :end_pos, ...].transpose(1, 2).transpose(2, 3)
            ) / math.sqrt(self.qk_total_dim)
            attn_scores = attn_scores.transpose(1, 2)

        else:
            # Flatten rotary embedding
            k_rot_flat = k_rot_processed.squeeze(dim=2)
            kv_normed = self.kv_norm(kv_features)

            # Update cache
            self.kv_feature_cache[:bs, start_pos:end_pos, :] = kv_normed
            self.rot_emb_cache[:bs, start_pos:end_pos, :] = k_rot_flat

            # Project query static part
            q_static_proj = torch.einsum("bshd,hdc->bshc", q_static,
                                         self.kv_up_proj.weight[:self.num_heads * self.qk_static, :].reshape(
                                             self.num_heads, self.qk_static, self.kv_rank))

            # Compute attention scores
            attn_static = torch.einsum("bshc,btc->bsht", q_static_proj, self.kv_feature_cache[:bs, :end_pos, :])
            attn_rot = torch.einsum("bshr,btr->bsht", q_rot_processed, self.rot_emb_cache[:bs, :end_pos, :])
            attn_scores = (attn_static + attn_rot) / math.sqrt(self.qk_total_dim)

        # Apply attention mask if provided
        if mask is not None:
            attn_scores = attn_scores + mask.unsqueeze(2)

        # Attention normalization
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute context vector
        if self.mode == 'low':
            context_vec = torch.einsum("bsht,bthd->bshd", attn_weights, self.value_cache[:bs, :end_pos, ...])
        else:
            context_proj = torch.einsum("bsht,btc->bshc", attn_weights, self.kv_feature_cache[:bs, :end_pos, :])
            context_vec = torch.einsum("bshc,hdc->bshd", context_proj,
                                       self.kv_up_proj.weight[-self.num_heads * self.v_head_dim:, :].reshape(
                                           self.num_heads, self.v_head_dim, self.kv_rank))

        # Final projection
        context_flat = context_vec.contiguous().view(bs, seq_len, -1)
        output = self.output_proj(context_flat)

        return output


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(1234)

    # Configuration for underwater image enhancement task
    batch_size = 8
    sequence_length = 64
    hidden_dimension = 2048

    # Generate random input tensor (simulating feature maps from underwater images)
    input_features = torch.randn(batch_size, sequence_length, hidden_dimension)

    # Model hyperparameters (adjusted for efficiency in image processing)
    model_params = {
        'input_dim': hidden_dimension,
        'num_heads': 8,
        'q_lora_rank': 64,
        'kv_lora_rank': 32,
        'qk_static_dim': 128,
        'qk_rotary_dim': 32,
        'v_dim': 128,
        'max_seq_length': 256,
        'max_batch_size': 16,
        'attention_mode': 'efficient'
    }

    # Initialize attention module
    attention_module = LAAT(**model_params)

    # Forward pass with different start positions (simulating sequential processing)
    start_positions = [0, 32, 64]
    for idx, start_pos in enumerate(start_positions):
        output = attention_module(input_features, start_pos=start_pos)

        # Print detailed information for each forward pass
        print(f"Forward Pass {idx + 1}:")
        print(f"  Input Shape: {input_features.shape}")
        print(f"  Start Position: {start_pos}")
        print(f"  Output Shape: {output.shape}")
        print(f"  Output Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")

        # Print cache information if in efficient mode
        if model_params['attention_mode'] != 'low':
            cache_shape = attention_module.kv_feature_cache.shape
            print(f"  KV Feature Cache Shape: {cache_shape}")
            print(f"  Cache Usage: {min(start_pos + sequence_length, cache_shape[1])}/{cache_shape[1]}")
        print("-" * 50)

    # Test with attention mask (simulating padding in variable-length feature sequences)
    attention_mask = torch.ones(batch_size, sequence_length, sequence_length)
    attention_mask[:, :, 40:] = -1e9  # Mask last 24 positions
    masked_output = attention_module(input_features, start_pos=0, mask=attention_mask)
    print("Test with Attention Mask:")
    print(f"  Masked Output Shape: {masked_output.shape}")
    print(f"  Masked Output Mean: {masked_output.mean().item():.4f}")
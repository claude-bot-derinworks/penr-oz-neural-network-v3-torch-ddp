import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as nnf


class CausalSelfAttention(nn.Module):
    def __init__(self, num_heads: int, dropout: float=0.0,
                 num_kv_heads: int=None, rope_theta: float=None,
                 head_dim: int=None, q_norm: nn.Module=None,
                 k_norm: nn.Module=None, v_norm: nn.Module=None,
                 kv_shared_layer_idx: int=None,
                 sliding_window: int=None, attn_scale: float=None,
                 rotary_dim: int=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.v_norm = v_norm
        self.kv_shared_layer_idx = kv_shared_layer_idx
        self.sliding_window = sliding_window
        self.attn_scale = attn_scale
        self.rotary_dim = rotary_dim
        self._kv_cache = None
        self._layer_idx = 0
        if rope_theta is not None and head_dim is not None:
            rope_dim = rotary_dim if rotary_dim is not None else head_dim
            inv_freq = 1.0 / (rope_theta ** (
                torch.arange(0, rope_dim, 2, dtype=torch.float32) / head_dim
            ))
            if rotary_dim is not None and rotary_dim < head_dim:
                inv_freq = torch.cat([inv_freq,
                                      torch.zeros(head_dim // 2 - rope_dim // 2,
                                                   dtype=torch.float32)])
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def set_kv_cache(self, kv_cache, layer_idx: int):
        """Attach a KVCache for incremental decoding.

        :param kv_cache: KVCache instance (or None to disable).
        :param layer_idx: Layer index within the cache.
        """
        self._kv_cache = kv_cache
        self._layer_idx = layer_idx

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _rope_cos_sin(self, seq_len: int, rope_dim: int, offset: int,
                      device, dtype) -> tuple[Tensor, Tensor]:
        if hasattr(self, "inv_freq"):
            inv_freq = self.inv_freq
        else:
            inv_freq = 1.0 / (self.rope_theta ** (
                torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim
            ))
        t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0).to(dtype)
        sin = emb.sin().unsqueeze(0).unsqueeze(0).to(dtype)
        return cos, sin

    def _apply_rope(self, q: Tensor, k: Tensor, rope_dim: int, offset: int = 0) -> tuple[Tensor, Tensor]:
        """Apply rotary position embeddings to query and key tensors."""
        cos, sin = self._rope_cos_sin(q.shape[2], rope_dim, offset, q.device, q.dtype)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    def forward(self, query_key_value: Tensor) -> Tensor:
        batch_size, block_size, total_dim = query_key_value.size()
        head_dim = total_dim // (self.num_heads + 2 * self.num_kv_heads)
        q_dim = self.num_heads * head_dim
        kv_dim = self.num_kv_heads * head_dim

        q, k_raw, v_raw = query_key_value.split([q_dim, kv_dim, kv_dim], dim=2)
        q = q.view(batch_size, block_size, self.num_heads, head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)

        rope_dim = self.rotary_dim if self.rotary_dim is not None else head_dim

        if self.kv_shared_layer_idx is not None and self._kv_cache is not None:
            # KV-shared layer: reuse reference layer's cached K/V
            # (already after k_norm, RoPE, and GQA expansion).  Apply RoPE to
            # own Q only; the offset matches the reference layer's position.
            ref_seq_len = self._kv_cache.seq_len(self.kv_shared_layer_idx)
            offset = ref_seq_len - block_size
            if self.rope_theta is not None:
                cos, sin = self._rope_cos_sin(block_size, rope_dim, offset, q.device, q.dtype)
                q = q * cos + self._rotate_half(q) * sin
            k, v = self._kv_cache.get(self.kv_shared_layer_idx)
            is_causal = (block_size > 1)
        else:
            k = k_raw.view(batch_size, block_size, self.num_kv_heads, head_dim).transpose(1, 2)
            v = v_raw.view(batch_size, block_size, self.num_kv_heads, head_dim).transpose(1, 2)
            if self.k_norm is not None:
                k = self.k_norm(k)
            if self.v_norm is not None:
                v = self.v_norm(v)
            if self.rope_theta is not None:
                offset = self._kv_cache.seq_len(self._layer_idx) if self._kv_cache is not None else 0
                cos, sin = self._rope_cos_sin(block_size, rope_dim, offset, q.device, q.dtype)
                q = q * cos + self._rotate_half(q) * sin
                k = k * cos + self._rotate_half(k) * sin
            # Expand KV heads for grouped-query attention
            if self.num_kv_heads < self.num_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k = k[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1).reshape(
                    batch_size, self.num_heads, -1, head_dim)
                v = v[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1).reshape(
                    batch_size, self.num_heads, -1, head_dim)
            # If KV cache is attached, append and retrieve full k/v
            if self._kv_cache is not None:
                k, v = self._kv_cache.append(self._layer_idx, k, v)
                is_causal = (block_size > 1)
            else:
                is_causal = True

        # apply scaled dot product attention formula
        dropout = self.dropout if self.training else 0.0
        attn_mask = None
        if self.sliding_window is not None and is_causal:
            kv_len = k.shape[2]
            q_abs = torch.arange(kv_len - block_size, kv_len, device=q.device)
            k_abs = torch.arange(kv_len, device=q.device)
            diff = q_abs.unsqueeze(1) - k_abs.unsqueeze(0)
            attn_mask = (diff >= 0) & (diff < self.sliding_window)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            is_causal = False
        scale = self.attn_scale if self.attn_scale is not None else (1.0 / (head_dim ** 0.5))
        output = nnf.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout, is_causal=is_causal, scale=scale)
        # combine head outputs -> (batch size, block size, q_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, block_size, q_dim)
        return output


class PositionEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._position_offset = 0

    @property
    def position_offset(self) -> int:
        return self._position_offset

    @position_offset.setter
    def position_offset(self, value: int):
        self._position_offset = value

    def forward(self, input_data: Tensor) -> Tensor:
        _, num_positions = input_data.shape
        positions = torch.arange(
            self._position_offset, self._position_offset + num_positions,
            dtype=torch.long, device=input_data.device
        )
        forwarded = super().forward(positions)
        return forwarded


class Summation(nn.Sequential):
    def forward(self, input_data: Tensor) -> Tensor:
        forwarded = self[0].forward(input_data)
        for layer in self[1:]:
            # please note: torch autograd fails with += in-place op, so use a = a + b instead
            forwarded = forwarded + layer(input_data)
        return forwarded


class ResidualConnection(nn.Sequential):
    def forward(self, forwarded: Tensor) -> Tensor:
        for layer in self:
            # please note: torch autograd fails with += in-place op, so use a = a + b instead
            forwarded = forwarded + layer(forwarded)
        return forwarded


class SoftmaxOnLast(nn.Softmax):
    def __init__(self, dim: int = -1, softcap: float = None):
        super().__init__(dim=dim)
        self.softcap = softcap

    def forward(self, logits: Tensor) -> Tensor:
        last_logits = logits[:, -1, :]
        if self.softcap is not None:
            last_logits = self.softcap * torch.tanh(last_logits / self.softcap)
        probs = super().forward(last_logits)
        return probs


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6,
                 with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * x_float.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        if self.with_scale:
            return normed.to(dtype) * self.weight
        return normed.to(dtype)


class GatedMLP(nn.Module):
    """Gated MLP with configurable activation (used in Gemma/LLaMA models)."""
    def __init__(self, in_features: int, intermediate_size: int,
                 bias: bool = False, activation: str = "gelu_pytorch_tanh"):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(in_features, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, in_features, bias=bias)
        if activation in ("silu", "swish"):
            self.act = nn.SiLU()
        elif activation == "gelu_pytorch_tanh":
            self.act = nn.GELU(approximate="tanh")
        else:
            self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class ScaledEmbedding(nn.Embedding):
    """Embedding with output scaled by a fixed factor."""
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 scale: float = 1.0, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.scale = scale

    def forward(self, input_data: Tensor) -> Tensor:
        return super().forward(input_data) * self.scale


class TransformerBlock(nn.Module):
    """Transformer decoder block with optional post-normalization.

    When *post_norm_on_residual* is ``False`` (Gemma sandwich-norm pattern),
    post-norms wrap only the branch output **before** it is added to the
    residual::

        h = x + post_attn_norm(attn_block(x))

    When ``True``, post-norms are applied **after** the residual addition::

        h = post_attn_norm(x + attn_block(x))
    """
    def __init__(self, attn_block: nn.Module, mlp_block: nn.Module,
                 post_attn_norm: nn.Module = None, post_mlp_norm: nn.Module = None,
                 post_norm_on_residual: bool = True,
                 has_layer_scalar: bool = False,
                 ple_gate: nn.Module = None, ple_proj: nn.Module = None,
                 ple_norm: nn.Module = None):
        super().__init__()
        self.attn_block = attn_block
        self.mlp_block = mlp_block
        self.post_attn_norm = post_attn_norm
        self.post_mlp_norm = post_mlp_norm
        self.post_norm_on_residual = post_norm_on_residual
        self.ple_gate = ple_gate
        self.ple_proj = ple_proj
        self.ple_norm = ple_norm
        self._ple_input = None
        if has_layer_scalar:
            self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        attn_out = self.attn_block(x)
        if self.post_attn_norm is not None and not self.post_norm_on_residual:
            attn_out = self.post_attn_norm(attn_out)
        h = x + attn_out
        if self.post_attn_norm is not None and self.post_norm_on_residual:
            h = self.post_attn_norm(h)

        mlp_out = self.mlp_block(h)
        if self.post_mlp_norm is not None and not self.post_norm_on_residual:
            mlp_out = self.post_mlp_norm(mlp_out)
        out = h + mlp_out
        if self.post_mlp_norm is not None and self.post_norm_on_residual:
            out = self.post_mlp_norm(out)

        if self.ple_gate is not None and self._ple_input is not None:
            residual = out
            out = nnf.gelu(self.ple_gate(out), approximate='tanh')
            out = out * self._ple_input
            out = self.ple_proj(out)
            out = self.ple_norm(out)
            out = residual + out
            self._ple_input = None

        if hasattr(self, 'layer_scalar'):
            out = out * self.layer_scalar

        return out


class PerLayerEmbedding(nn.Module):
    """Computes per-layer embeddings (PLE) from token IDs and hidden states.

    Produces a conditioning signal for each transformer layer by combining
    a token-identity lookup with a context-aware projection of the embedding
    output.  The combined signal is split into per-layer chunks and
    distributed to TransformerBlock modules before they execute.
    """
    def __init__(self, embed_per_layer: nn.Module, projection: nn.Module,
                 norm: nn.Module, n_layers: int, ple_dim: int,
                 hidden_size: int):
        super().__init__()
        self.embed_per_layer = embed_per_layer
        self.projection = projection
        self.norm = norm
        self.n_layers = n_layers
        self.ple_dim = ple_dim
        self.hidden_scale = hidden_size ** -0.5
        self._input_ids = None
        self._transformer_blocks = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self._input_ids is None or self._transformer_blocks is None:
            return hidden_states

        shape = hidden_states.shape[:-1] + (self.n_layers, self.ple_dim)
        ple_token = self.embed_per_layer(self._input_ids).view(shape)
        ple_context = self.projection(hidden_states).view(shape) * self.hidden_scale
        ple_context = self.norm(ple_context)
        ple_combined = (ple_token + ple_context) * (2.0 ** -0.5)

        for i, block in enumerate(self._transformer_blocks):
            block._ple_input = ple_combined[..., i, :]

        self._input_ids = None
        return hidden_states

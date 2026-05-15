import unittest
from parameterized import parameterized
import torch
from torch import Tensor
import torch.nn as nn
import neural_net_layers as nnl


class TestNeuralNetLayers(unittest.TestCase):

    @parameterized.expand([
        (nnl.CausalSelfAttention, dict(num_heads=2)),
        (nnl.CausalSelfAttention, dict(num_heads=2, dropout=0.2)),
        (nnl.CausalSelfAttention, dict(num_heads=4, num_kv_heads=2)),
        (nnl.CausalSelfAttention, dict(num_heads=4, num_kv_heads=2, rope_theta=10000.0)),
        (nnl.CausalSelfAttention, dict(num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4)),
        (nnl.CausalSelfAttention, dict(num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4,
                                       q_norm=nnl.RMSNorm(4), k_norm=nnl.RMSNorm(4))),
        (nnl.CausalSelfAttention, dict(num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4,
                                       kv_shared_layer_idx=0)),
        (nnl.CausalSelfAttention, dict(num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4,
                                       sliding_window=4)),
        (nnl.CausalSelfAttention, dict(num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4,
                                       attn_scale=1.0)),
        (nnl.CausalSelfAttention, dict(num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=8,
                                       rotary_dim=2)),
        (nnl.PositionEmbedding, dict(num_embeddings=27, embedding_dim=4)),
        (nnl.Summation, [nn.Embedding(27, 4),
                         nnl.PositionEmbedding(8, 4)]),
        (nnl.ResidualConnection, [nn.LayerNorm(4), nn.Linear(4, 8)]),
        (nnl.SoftmaxOnLast, dict(dim=-1)),
        (nnl.SoftmaxOnLast, dict(dim=-1, softcap=30.0)),
        (nnl.RMSNorm, dict(normalized_shape=4)),
        (nnl.GatedMLP, dict(in_features=4, intermediate_size=8)),
        (nnl.ScaledEmbedding, dict(num_embeddings=27, embedding_dim=4, scale=2.0)),
        (nnl.TransformerBlock, dict(
            attn_block=nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
            mlp_block=nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)))),
        (nnl.TransformerBlock, dict(
            attn_block=nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
            mlp_block=nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
            post_attn_norm=nnl.RMSNorm(4), post_mlp_norm=nnl.RMSNorm(4),
            post_norm_on_residual=False)),
        (nnl.TransformerBlock, dict(
            attn_block=nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
            mlp_block=nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
            has_layer_scalar=True)),
        (nnl.PerLayerEmbedding, dict(
            embed_per_layer=nnl.ScaledEmbedding(16, 8, scale=2.0),
            projection=nn.Linear(4, 8, bias=False),
            norm=nnl.RMSNorm(4),
            n_layers=2, ple_dim=4, hidden_size=4)),
    ])
    def test_layer_init(self, layer_class: type, layer_args: dict | list):
        layer = layer_class(**layer_args) if isinstance(layer_args, dict) else layer_class(*layer_args)

        self.assertIsInstance(layer, nn.Module)

    @parameterized.expand([
        (nnl.CausalSelfAttention(2),
         torch.randn(5, 8, 12), (5, 8, 4)),
        (nnl.CausalSelfAttention(3, 0.2),
         torch.randn(5, 5, 45), (5, 5, 15)),
        # GQA: 4 query heads, 2 kv heads, head_dim=4 -> qkv_dim = 4*4 + 2*2*4 = 32
        (nnl.CausalSelfAttention(num_heads=4, num_kv_heads=2),
         torch.randn(2, 6, 32), (2, 6, 16)),
        # GQA + RoPE: same dims with rope_theta
        (nnl.CausalSelfAttention(num_heads=4, num_kv_heads=2, rope_theta=10000.0),
         torch.randn(2, 6, 32), (2, 6, 16)),
        # GQA + RoPE with precomputed inv_freq buffer
        (nnl.CausalSelfAttention(num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4),
         torch.randn(2, 6, 32), (2, 6, 16)),
        # GQA + RoPE + q_norm/k_norm (Gemma 2+ pattern)
        (nnl.CausalSelfAttention(num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4,
                                 q_norm=nnl.RMSNorm(4), k_norm=nnl.RMSNorm(4)),
         torch.randn(2, 6, 32), (2, 6, 16)),
        (nnl.PositionEmbedding(27, 4),
         torch.randint(0, 27, (5, 8)), (8, 4)),
        (nnl.Summation(nn.Embedding(27, 4),
                       nnl.PositionEmbedding(8, 4)),
         torch.randint(0, 27, (5, 8)), (5, 8, 4)),
        (nn.Sequential(nn.LayerNorm(4, bias=False),
                       nn.Linear(4, 12, False),
                       nnl.CausalSelfAttention(4, 0.2),
                       nn.Linear(4, 4, False),
                       nn.Dropout(0.2)),
         torch.randn(5, 8, 4), (5, 8, 4)),
        # RMSNorm forward
        (nnl.RMSNorm(4),
         torch.randn(5, 8, 4), (5, 8, 4)),
        # GatedMLP forward
        (nnl.GatedMLP(4, 8),
         torch.randn(5, 8, 4), (5, 8, 4)),
        # ScaledEmbedding forward
        (nnl.ScaledEmbedding(27, 4, scale=2.0),
         torch.randint(0, 27, (5, 8)), (5, 8, 4)),
        # TransformerBlock forward (no post-norms)
        (nnl.TransformerBlock(
            attn_block=nn.Sequential(
                nnl.RMSNorm(4),
                nn.Linear(4, 12, False),
                nnl.CausalSelfAttention(4),
                nn.Linear(4, 4, False)),
            mlp_block=nn.Sequential(
                nnl.RMSNorm(4),
                nnl.GatedMLP(4, 8))),
         torch.randn(2, 6, 4), (2, 6, 4)),
        # TransformerBlock forward (with post-norms, Gemma 3 pattern)
        (nnl.TransformerBlock(
            attn_block=nn.Sequential(
                nnl.RMSNorm(4),
                nn.Linear(4, 12, False),
                nnl.CausalSelfAttention(4),
                nn.Linear(4, 4, False)),
            mlp_block=nn.Sequential(
                nnl.RMSNorm(4),
                nnl.GatedMLP(4, 8)),
            post_attn_norm=nnl.RMSNorm(4),
            post_mlp_norm=nnl.RMSNorm(4)),
         torch.randn(2, 6, 4), (2, 6, 4)),
        # TransformerBlock forward (with post-norms, Gemma 2 pattern)
        (nnl.TransformerBlock(
            attn_block=nn.Sequential(
                nnl.RMSNorm(4),
                nn.Linear(4, 12, False),
                nnl.CausalSelfAttention(4),
                nn.Linear(4, 4, False)),
            mlp_block=nn.Sequential(
                nnl.RMSNorm(4),
                nnl.GatedMLP(4, 8)),
            post_attn_norm=nnl.RMSNorm(4),
            post_mlp_norm=nnl.RMSNorm(4),
            post_norm_on_residual=False),
         torch.randn(2, 6, 4), (2, 6, 4)),
        # Full GPT-2 style model
        (nn.Sequential(
            nnl.Summation(nn.Embedding(27, 4),
                          nnl.PositionEmbedding(8, 4)),
            nn.Dropout(0.2),
           *[nnl.ResidualConnection(
               nn.Sequential(
                   nn.LayerNorm(4, bias=False),
                   nn.Linear(4, 12, False),
                   nnl.CausalSelfAttention(4, 0.2),
                   nn.Linear(4, 4, False),
                   nn.Dropout(0.2)
               ),
               nn.Sequential(
                   nn.LayerNorm(4, bias=False),
                   nn.Linear(4, 16, False),
                   nn.GELU(),
                   nn.Linear(16, 4, False),
                  nn.Dropout(0.2)))
               for _ in range(2)],
            nn.LayerNorm(4, bias=False),
            nn.Linear(4, 27, bias=False),
            nnl.SoftmaxOnLast(dim=-1)),
         torch.randint(0, 27, (5, 8)), (5, 27)),
    ])
    def test_forward(self, layer: nn.Module, input_data: Tensor, expected_out_shape: tuple):
        output: Tensor = layer(input_data)

        self.assertIsNotNone(output)
        self.assertEqual(expected_out_shape, tuple(output.shape))

    def test_qk_norm_modifies_attention_output(self):
        """When q_norm/k_norm are provided, attention output differs from unnormalized."""
        num_heads, num_kv_heads, head_dim = 4, 2, 4
        qkv_dim = num_heads * head_dim + 2 * num_kv_heads * head_dim
        batch, seq = 1, 3

        attn_no_norm = nnl.CausalSelfAttention(
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            rope_theta=10000.0, head_dim=head_dim)
        attn_with_norm = nnl.CausalSelfAttention(
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            rope_theta=10000.0, head_dim=head_dim,
            q_norm=nnl.RMSNorm(head_dim), k_norm=nnl.RMSNorm(head_dim))

        # Use non-unit weights so norms have a visible effect
        attn_with_norm.q_norm.weight.data.fill_(2.0)
        attn_with_norm.k_norm.weight.data.fill_(0.5)

        torch.manual_seed(42)
        qkv = torch.randn(batch, seq, qkv_dim)

        out_no_norm = attn_no_norm(qkv)
        out_with_norm = attn_with_norm(qkv)

        self.assertEqual(out_no_norm.shape, out_with_norm.shape)
        self.assertFalse(torch.allclose(out_no_norm, out_with_norm, atol=1e-6),
                         "q_norm/k_norm should change the attention output")

    def test_qk_norm_registers_as_submodules(self):
        """q_norm and k_norm appear in state_dict when provided."""
        attn = nnl.CausalSelfAttention(
            num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4,
            q_norm=nnl.RMSNorm(4), k_norm=nnl.RMSNorm(4))
        sd_keys = set(attn.state_dict().keys())
        self.assertIn("q_norm.weight", sd_keys)
        self.assertIn("k_norm.weight", sd_keys)

    def test_no_qk_norm_no_extra_state_keys(self):
        """Without q_norm/k_norm, state_dict has no norm keys."""
        attn = nnl.CausalSelfAttention(
            num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4)
        sd_keys = set(attn.state_dict().keys())
        self.assertNotIn("q_norm.weight", sd_keys)
        self.assertNotIn("k_norm.weight", sd_keys)

    def test_kv_shared_layer_uses_reference_cache(self):
        """KV-shared layer reads K/V from reference layer's cache (post k_norm,
        post-RoPE, post-GQA-expansion). The shared layer applies its own q_norm
        and RoPE to Q only — never writing K/V to its own cache."""
        from kv_cache import KVCache

        num_heads, num_kv_heads, head_dim = 4, 2, 4
        qkv_dim = num_heads * head_dim + 2 * num_kv_heads * head_dim
        batch, seq = 1, 3

        # Reference layer: non-shared
        attn_ref = nnl.CausalSelfAttention(
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            rope_theta=10000.0, head_dim=head_dim)
        # Shared layer: references layer 0
        attn_shared = nnl.CausalSelfAttention(
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            rope_theta=10000.0, head_dim=head_dim,
            kv_shared_layer_idx=0)

        cache = KVCache(num_layers=2)
        attn_ref.set_kv_cache(cache, 0)
        attn_shared.set_kv_cache(cache, 1)

        torch.manual_seed(42)
        qkv_ref = torch.randn(batch, seq, qkv_dim)
        qkv_shared = torch.randn(batch, seq, qkv_dim)

        attn_ref(qkv_ref)
        attn_shared(qkv_shared)

        # Reference layer wrote to its own cache; shared layer did NOT write
        self.assertEqual(cache.seq_len(0), seq)
        self.assertEqual(cache.seq_len(1), 0,
                         "Shared layer should not write to its own cache slot")

    def test_kv_shared_layer_offset_during_decode(self):
        """During incremental decode the shared layer derives Q's RoPE offset
        from the reference layer's cache length (after the reference appends)."""
        from kv_cache import KVCache

        num_heads, num_kv_heads, head_dim = 4, 2, 4
        qkv_dim = num_heads * head_dim + 2 * num_kv_heads * head_dim
        batch, seq = 1, 3

        attn_ref = nnl.CausalSelfAttention(
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            rope_theta=10000.0, head_dim=head_dim)
        attn_shared = nnl.CausalSelfAttention(
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            rope_theta=10000.0, head_dim=head_dim,
            kv_shared_layer_idx=0)

        cache = KVCache(num_layers=2)
        attn_ref.set_kv_cache(cache, 0)
        attn_shared.set_kv_cache(cache, 1)

        # Prefill
        qkv = torch.randn(batch, seq, qkv_dim)
        attn_ref(qkv)
        attn_shared(qkv)

        # Decode one token; should not raise even though shared layer's own
        # cache stayed empty.
        qkv_one = torch.randn(batch, 1, qkv_dim)
        attn_ref(qkv_one)
        out_shared = attn_shared(qkv_one)

        self.assertEqual(out_shared.shape, (batch, 1, num_heads * head_dim))
        self.assertEqual(cache.seq_len(0), seq + 1)
        self.assertEqual(cache.seq_len(1), 0)

    def test_rope_offset_uses_own_layer_idx_with_kv_cache(self):
        """Each attention layer must read its own KV cache seq_len for RoPE offset,
        not layer 0's.  When layers run sequentially during a forward pass,
        layer 0 appends to its cache before layer 1 executes.  If layer 1
        reads layer 0's (already-updated) cache length, the RoPE positions
        are wrong for every layer after the first."""
        from kv_cache import KVCache

        num_heads, num_kv_heads, head_dim = 4, 2, 4
        qkv_dim = num_heads * head_dim + 2 * num_kv_heads * head_dim
        batch, seq = 1, 3

        attn0 = nnl.CausalSelfAttention(num_heads=num_heads, num_kv_heads=num_kv_heads,
                                          rope_theta=10000.0, head_dim=head_dim)
        attn1 = nnl.CausalSelfAttention(num_heads=num_heads, num_kv_heads=num_kv_heads,
                                          rope_theta=10000.0, head_dim=head_dim)

        cache = KVCache(num_layers=2)
        attn0.set_kv_cache(cache, layer_idx=0)
        attn1.set_kv_cache(cache, layer_idx=1)

        qkv = torch.randn(batch, seq, qkv_dim)

        # Prefill: both layers process the full sequence
        attn0(qkv)  # layer 0 cache now has `seq` entries
        attn1(qkv)  # layer 1 should use its own cache (was empty) for offset

        # After prefill both caches must have the same length
        self.assertEqual(cache.seq_len(0), seq)
        self.assertEqual(cache.seq_len(1), seq)

        # Incremental decode: single new token
        qkv_one = torch.randn(batch, 1, qkv_dim)
        out0 = attn0(qkv_one)  # layer 0 cache → seq+1
        out1 = attn1(qkv_one)  # layer 1 must read own cache (seq), not layer 0 (seq+1)

        self.assertEqual(cache.seq_len(0), seq + 1)
        self.assertEqual(cache.seq_len(1), seq + 1)

        # Both layers have identical config (same inv_freq buffer, no trainable
        # weights) and received the same inputs with the same correct RoPE
        # offsets (0 during prefill, seq during decode).  Their outputs must
        # match.  If the bug were present, layer 1 would have used layer 0's
        # already-updated cache length (seq+1 instead of seq) for its RoPE
        # offset, producing a different result.
        self.assertTrue(torch.allclose(out0, out1, atol=1e-6),
                        "Layer 0 and Layer 1 outputs should match when using correct per-layer offsets")


    def test_softmax_on_last_softcap_bounds_logits(self):
        """Softcapping applies tanh(logits/cap)*cap before softmax, bounding extreme values."""
        cap = 30.0
        layer_capped = nnl.SoftmaxOnLast(dim=-1, softcap=cap)
        layer_plain = nnl.SoftmaxOnLast(dim=-1)

        # Two large logits that differ by 100 — without capping the top token
        # dominates; with capping both saturate to ~cap and share probability
        logits = torch.zeros(1, 1, 4)
        logits[0, 0, 0] = 200.0
        logits[0, 0, 1] = 100.0

        probs_capped = layer_capped(logits)
        probs_plain = layer_plain(logits)

        # Without capping, the gap of 100 makes token 0 dominate
        self.assertGreater(probs_plain[0, 0].item(), 0.99)
        # With capping, both saturate near cap — distribution flattens
        self.assertLess(probs_capped[0, 0].item(), 0.6)
        # All probabilities should still sum to 1
        self.assertAlmostEqual(probs_capped.sum().item(), 1.0, places=5)

    def test_softmax_on_last_no_softcap_unchanged(self):
        """Without softcap, SoftmaxOnLast behaves identically to the original."""
        layer = nnl.SoftmaxOnLast(dim=-1)
        logits = torch.randn(2, 4, 8)
        expected = torch.softmax(logits[:, -1, :], dim=-1)
        result = layer(logits)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_sliding_window_restricts_attention(self):
        """Sliding window attention ignores tokens beyond the window."""
        num_heads, head_dim, window = 2, 4, 2
        qkv_dim = num_heads * head_dim * 3
        seq = 6

        attn_full = nnl.CausalSelfAttention(num_heads=num_heads, head_dim=head_dim)
        attn_sliding = nnl.CausalSelfAttention(
            num_heads=num_heads, head_dim=head_dim, sliding_window=window)

        torch.manual_seed(42)
        qkv = torch.randn(1, seq, qkv_dim)

        out_full = attn_full(qkv)
        out_sliding = attn_sliding(qkv)

        # First token sees only itself — window doesn't matter, outputs match
        self.assertTrue(torch.allclose(out_full[:, 0, :], out_sliding[:, 0, :], atol=1e-5))
        # Later tokens differ because sliding window masks out distant keys
        self.assertFalse(torch.allclose(out_full[:, -1, :], out_sliding[:, -1, :], atol=1e-5))

    def test_attn_scale_overrides_default(self):
        """Custom attn_scale=1.0 differs from default 1/sqrt(head_dim) scaling."""
        num_heads, head_dim = 2, 16
        qkv_dim = num_heads * head_dim * 3
        seq = 4

        attn_default = nnl.CausalSelfAttention(num_heads=num_heads, head_dim=head_dim)
        attn_unit = nnl.CausalSelfAttention(
            num_heads=num_heads, head_dim=head_dim, attn_scale=1.0)

        torch.manual_seed(99)
        qkv = torch.randn(1, seq, qkv_dim)
        out_default = attn_default(qkv)
        out_unit = attn_unit(qkv)

        self.assertFalse(torch.allclose(out_default, out_unit, atol=1e-5))

    def test_partial_rotary_differs_from_full(self):
        """Partial RoPE (rotary_dim < head_dim) produces different output than full RoPE."""
        num_heads, head_dim = 2, 8
        qkv_dim = num_heads * head_dim * 3
        seq = 4

        attn_full_rope = nnl.CausalSelfAttention(
            num_heads=num_heads, head_dim=head_dim, rope_theta=10000.0)
        attn_partial = nnl.CausalSelfAttention(
            num_heads=num_heads, head_dim=head_dim, rope_theta=10000.0,
            rotary_dim=2)

        torch.manual_seed(77)
        qkv = torch.randn(1, seq, qkv_dim)
        out_full = attn_full_rope(qkv)
        out_partial = attn_partial(qkv)

        self.assertFalse(torch.allclose(out_full, out_partial, atol=1e-5))

    def test_transformer_block_layer_scalar(self):
        """TransformerBlock with has_layer_scalar registers buffer and applies it."""
        block = nnl.TransformerBlock(
            attn_block=nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
            mlp_block=nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
            has_layer_scalar=True)
        self.assertIn("layer_scalar", dict(block.named_buffers()))
        self.assertEqual(block.layer_scalar.item(), 1.0)
        block.layer_scalar.fill_(0.5)
        x = torch.randn(1, 3, 4)
        block_no_scalar = nnl.TransformerBlock(
            attn_block=block.attn_block, mlp_block=block.mlp_block)
        out_no_scalar = block_no_scalar(x)
        out_scaled = block(x)
        self.assertTrue(torch.allclose(out_scaled, out_no_scalar * 0.5, atol=1e-5))

    def test_per_layer_embedding_distributes_to_blocks(self):
        """PerLayerEmbedding computes PLE and distributes chunks to blocks."""
        n_layers, ple_dim, hidden_size, vocab_size = 2, 4, 8, 16
        ple = nnl.PerLayerEmbedding(
            embed_per_layer=nnl.ScaledEmbedding(vocab_size, n_layers * ple_dim, scale=float(ple_dim ** 0.5)),
            projection=nn.Linear(hidden_size, n_layers * ple_dim, bias=False),
            norm=nnl.RMSNorm(ple_dim),
            n_layers=n_layers, ple_dim=ple_dim, hidden_size=hidden_size)
        blocks = [
            nnl.TransformerBlock(
                attn_block=nn.Sequential(nn.Linear(hidden_size, hidden_size)),
                mlp_block=nn.Sequential(nn.Linear(hidden_size, hidden_size)),
                ple_gate=nn.Linear(hidden_size, ple_dim, bias=False),
                ple_proj=nn.Linear(ple_dim, hidden_size, bias=False),
                ple_norm=nnl.RMSNorm(hidden_size))
            for _ in range(n_layers)
        ]
        ple._transformer_blocks = blocks
        ple._input_ids = torch.randint(0, vocab_size, (1, 3))
        hidden = torch.randn(1, 3, hidden_size)
        out = ple(hidden)
        self.assertTrue(torch.equal(out, hidden))
        for block in blocks:
            self.assertIsNotNone(block._ple_input)
            self.assertEqual(block._ple_input.shape, (1, 3, ple_dim))

    def test_transformer_block_ple_modifies_output(self):
        """TransformerBlock with PLE produces different output than without."""
        hidden_size, ple_dim = 8, 4
        attn = nn.Sequential(nn.Linear(hidden_size, hidden_size))
        mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size))
        block_ple = nnl.TransformerBlock(
            attn_block=attn, mlp_block=mlp,
            ple_gate=nn.Linear(hidden_size, ple_dim, bias=False),
            ple_proj=nn.Linear(ple_dim, hidden_size, bias=False),
            ple_norm=nnl.RMSNorm(hidden_size))
        block_no_ple = nnl.TransformerBlock(attn_block=attn, mlp_block=mlp)
        x = torch.randn(1, 3, hidden_size)
        block_ple._ple_input = torch.randn(1, 3, ple_dim)
        out_ple = block_ple(x)
        out_no_ple = block_no_ple(x)
        self.assertFalse(torch.allclose(out_ple, out_no_ple, atol=1e-5))


if __name__ == '__main__':
    unittest.main()

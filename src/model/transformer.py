"""Decoder-only Transformer built from scratch with PyTorch.

Architecture (GPT-2 style pre-norm):

    Input token IDs
      → Token Embedding + Learned Positional Embedding
      → N × TransformerBlock:
            LayerNorm → CausalSelfAttention → Residual
            LayerNorm → MLP → Residual
      → LayerNorm
      → Linear projection → vocab-sized logits

Every hyperparameter comes from ModelConfig so the same code
works for tiny (5M param) to medium (100M+ param) models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal (autoregressive) masking."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Single linear for Q, K, V (more efficient than three separate ones)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.attn_drop = nn.Dropout(config.dropout_rate)
        self.proj_drop = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch 2 fused attention (FlashAttention / memory-efficient kernel)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        # Combine heads back to (B, T, C)
        out = out.transpose(1, 2).reshape(B, T, C)

        out = self.proj_drop(self.proj(out))
        return out


class MLP(nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.drop = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Attn → Add, LN → MLP → Add."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    """Full decoder-only transformer for next-token prediction.

    Input:  (batch, seq_len) integer token IDs
    Output: (batch, seq_len, vocab_size) logits
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout_rate)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        B, T = x.shape
        device = x.device

        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

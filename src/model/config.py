"""Model configuration for the decoder-only transformer.

All architecture hyperparameters live here as a single dataclass.
Defaults give a ~5M parameter model trainable on a Colab T4 GPU.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 4096          # must match tokenizer vocab size
    context_length: int = 128       # maximum sequence length
    d_model: int = 256              # embedding / hidden dimension
    n_heads: int = 4                # number of attention heads
    n_layers: int = 4               # number of transformer blocks
    d_ff: int = 1024                # feed-forward inner dimension (typically 4 * d_model)
    dropout_rate: float = 0.1       # dropout probability

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )

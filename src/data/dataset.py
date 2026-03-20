"""Token dataset for next-token prediction training.

Loads pre-tokenized .npy part files (produced by CorpusEncoder),
concatenates them into a single flat token array, and serves
random batches of (input, target) pairs for causal language modeling.

Each batch:
    inputs:  (batch_size, context_length)   — token IDs
    targets: (batch_size, context_length)   — same sequence shifted right by 1
"""

from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    """Serves random (input, target) batches from a flat token array.

    Usage:
        dataset = TokenDataset.from_parts("dataset/tokens_parts", context_length=128)
        train, val = dataset.split(val_fraction=0.05)
        x, y = train.get_batch(batch_size=64, device="cuda")
    """

    def __init__(self, tokens, context_length):
        self.tokens = torch.tensor(np.asarray(tokens), dtype=torch.long)
        self.context_length = context_length
        self.n_sequences = len(self.tokens) - context_length
        self.n_tokens = len(self.tokens)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_parts(cls, parts_dir, context_length, max_parts=None):
        """Load and concatenate .npy part files from a directory.

        Args:
            parts_dir:      Path to directory with part_XXXXX.npy files.
            context_length: Sequence length for training windows.
            max_parts:      Load only the first N parts (useful for quick tests).
        """
        parts_dir = Path(parts_dir)
        part_files = sorted(parts_dir.glob("part_*.npy"))
        if max_parts:
            part_files = part_files[:max_parts]

        print(f"Loading {len(part_files)} token parts from {parts_dir} ...")
        tokens = np.concatenate([np.load(f) for f in part_files])
        print(f"Total tokens: {len(tokens):,}")
        return cls(tokens, context_length)

    @classmethod
    def from_file(cls, path, context_length):
        """Load from a single merged tokens.npy file."""
        tokens = np.load(path)
        print(f"Loaded {len(tokens):,} tokens from {path}")
        return cls(tokens, context_length)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(self, val_fraction=0.05):
        """Split into train / validation datasets by token position.

        Returns:
            (train_dataset, val_dataset)
        """
        n = len(self.tokens)
        split_idx = int(n * (1 - val_fraction))
        train = TokenDataset(self.tokens[:split_idx], self.context_length)
        val = TokenDataset(self.tokens[split_idx:], self.context_length)
        print(f"Train: {train.n_tokens:,} tokens ({train.n_sequences:,} sequences)")
        print(f"Val:   {val.n_tokens:,} tokens ({val.n_sequences:,} sequences)")
        return train, val

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def get_batch(self, batch_size, device="cpu"):
        """Sample a random batch of (input, target) pairs.

        Args:
            batch_size: Number of sequences in the batch.
            device:     Target device ("cpu" or "cuda").

        Returns:
            inputs:  tensor of shape (batch_size, context_length), dtype long
            targets: tensor of shape (batch_size, context_length), dtype long
        """
        # Random start positions
        starts = torch.randint(0, self.n_sequences, (batch_size,))

        # Fancy-index full windows of length (context_length + 1)
        offsets = torch.arange(self.context_length + 1)
        indices = starts.unsqueeze(1) + offsets.unsqueeze(0)   # (batch, ctx+1)
        sequences = self.tokens[indices]

        inputs = sequences[:, :-1].to(device)
        targets = sequences[:, 1:].to(device)
        return inputs, targets

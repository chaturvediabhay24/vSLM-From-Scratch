"""Fine-tuning dataset for chatbot Q&A format.

Loads Q&A pairs from CSV files, formats them as:
    Question: {question} Answer: {answer}<EOS>

Returns (inputs, targets, loss_mask) batches where loss_mask is 1
only for the answer tokens — so the model learns to generate answers,
not memorize question prefixes.
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch


class FineTuneDataset:
    """Serves (input, target, loss_mask) batches from Q&A examples.

    Usage:
        dataset = FineTuneDataset.from_csv_dir(
            "dataset/personal-details", tokenizer, context_length=128,
        )
        train, val = dataset.split()
        x, y, mask = train.get_batch(batch_size=32, device="cuda")
    """

    def __init__(self, examples, context_length, pad_token_id=0):
        """
        Args:
            examples:       List of dicts with keys:
                            - "input_ids": full tokenized sequence (list[int])
                            - "answer_start": index where answer tokens begin
            context_length: Max sequence length (sequences are padded/truncated).
            pad_token_id:   Token ID used for padding (default 0).
        """
        self.examples = examples
        self.context_length = context_length
        self.pad_token_id = pad_token_id

    @classmethod
    def from_csv_dir(cls, csv_dir, tokenizer, context_length):
        """Load all Q&A CSVs from a directory and build examples.

        Each CSV must have columns: question, answer.
        """
        csv_dir = Path(csv_dir)
        csv_files = sorted(csv_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {csv_dir}")

        eos_token = "<EOS>"
        pairs = []
        for csv_file in csv_files:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    q = row["question"].strip()
                    a = row["answer"].strip()
                    if q and a:
                        pairs.append((q, a))

        print(f"Loaded {len(pairs)} Q&A pairs from {len(csv_files)} CSV files")

        examples = []
        skipped = 0
        for q, a in pairs:
            full_text = f"Question: {q} Answer: {a}{eos_token}"
            prefix_text = f"Question: {q} Answer: "

            full_ids = tokenizer.encode(full_text)
            prefix_ids = tokenizer.encode(prefix_text)
            answer_start = len(prefix_ids)

            # Need at least 2 tokens (1 input + 1 target) to train
            if len(full_ids) < 2:
                skipped += 1
                continue

            # Truncate if longer than context_length + 1
            if len(full_ids) > context_length + 1:
                full_ids = full_ids[: context_length + 1]

            examples.append({
                "input_ids": full_ids,
                "answer_start": answer_start,
            })

        if skipped:
            print(f"Skipped {skipped} pairs (too short)")
        print(f"Created {len(examples)} fine-tuning examples")

        return cls(examples, context_length)

    @classmethod
    def from_pairs(cls, pairs, tokenizer, context_length):
        """Build dataset from a list of (question, answer) tuples directly."""
        eos_token = "<EOS>"
        examples = []
        for q, a in pairs:
            full_text = f"Question: {q} Answer: {a}{eos_token}"
            prefix_text = f"Question: {q} Answer: "

            full_ids = tokenizer.encode(full_text)
            prefix_ids = tokenizer.encode(prefix_text)
            answer_start = len(prefix_ids)

            if len(full_ids) < 2:
                continue
            if len(full_ids) > context_length + 1:
                full_ids = full_ids[: context_length + 1]

            examples.append({
                "input_ids": full_ids,
                "answer_start": answer_start,
            })

        return cls(examples, context_length)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(self, val_fraction=0.1):
        """Randomly split into train / validation datasets.

        Returns:
            (train_dataset, val_dataset)
        """
        examples = list(self.examples)
        random.shuffle(examples)
        split_idx = int(len(examples) * (1 - val_fraction))
        train = FineTuneDataset(examples[:split_idx], self.context_length, self.pad_token_id)
        val = FineTuneDataset(examples[split_idx:], self.context_length, self.pad_token_id)
        print(f"Train: {len(train.examples)} examples")
        print(f"Val:   {len(val.examples)} examples")
        return train, val

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def get_batch(self, batch_size, device="cpu"):
        """Sample a random batch of (input, target, loss_mask).

        Args:
            batch_size: Number of examples in the batch.
            device:     Target device ("cpu" or "cuda").

        Returns:
            inputs:    (batch_size, context_length) — token IDs
            targets:   (batch_size, context_length) — shifted right by 1
            loss_mask: (batch_size, context_length) — 1 for answer positions, 0 elsewhere
        """
        indices = random.choices(range(len(self.examples)), k=batch_size)

        inputs_batch = []
        targets_batch = []
        mask_batch = []

        for idx in indices:
            ex = self.examples[idx]
            ids = ex["input_ids"]
            ans_start = ex["answer_start"]

            # input = ids[:-1], target = ids[1:]
            inp = ids[:-1]
            tgt = ids[1:]

            seq_len = len(inp)
            pad_len = self.context_length - seq_len

            # Pad sequences
            if pad_len > 0:
                inp = inp + [self.pad_token_id] * pad_len
                tgt = tgt + [self.pad_token_id] * pad_len

            # Truncate (shouldn't happen due to earlier truncation, but safety)
            inp = inp[: self.context_length]
            tgt = tgt[: self.context_length]

            # Loss mask: 1 for answer positions (ans_start-1 because of shift),
            # 0 for question prefix and padding
            # In the target sequence, answer tokens start at position (ans_start - 1)
            mask = [0] * self.context_length
            ans_target_start = max(0, ans_start - 1)
            for i in range(ans_target_start, min(seq_len, self.context_length)):
                mask[i] = 1

            inputs_batch.append(inp)
            targets_batch.append(tgt)
            mask_batch.append(mask)

        inputs = torch.tensor(inputs_batch, dtype=torch.long, device=device)
        targets = torch.tensor(targets_batch, dtype=torch.long, device=device)
        loss_mask = torch.tensor(mask_batch, dtype=torch.float32, device=device)

        return inputs, targets, loss_mask

    def __len__(self):
        return len(self.examples)

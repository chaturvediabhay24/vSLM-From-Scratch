"""Training loop for the transformer language model.

Uses PyTorch for the full lifecycle: init → train → evaluate → generate.

Usage:
    trainer = Trainer(model, model_config, train_config)
    history = trainer.train(train_data, val_data)
    print(trainer.generate(tokenizer, "Once upon a time"))
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------------

@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    max_steps: int = 50_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 100
    eval_every: int = 2000
    eval_batches: int = 20          # batches to average for validation loss
    checkpoint_every: int = 10_000
    checkpoint_dir: str = "checkpoints"
    seed: int = 42


# ------------------------------------------------------------------
# LR schedule helper
# ------------------------------------------------------------------

def _lr_lambda(step, warmup_steps, max_steps, min_lr_ratio=0.1):
    """Linear warmup then cosine decay to min_lr_ratio * peak_lr."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class Trainer:
    """Trains a PyTorch transformer model.

    Handles optimizer setup, training loop with gradient clipping,
    logging, checkpointing, and autoregressive text generation.
    """

    def __init__(self, model, model_config, train_config=None):
        self.model = model
        self.model_config = model_config
        self.train_config = train_config or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, dataset):
        """Average loss over several random validation batches."""
        self.model.eval()
        cfg = self.train_config
        total = 0.0
        for _ in range(cfg.eval_batches):
            inputs, targets = dataset.get_batch(cfg.batch_size, self.device)
            logits = self.model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
            )
            total += loss.item()
        self.model.train()
        return total / cfg.eval_batches

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, train_data, val_data=None):
        """Run the full training loop.

        Args:
            train_data: TokenDataset for training.
            val_data:   TokenDataset for validation (optional).

        Returns:
            dict of logged metrics (step, train_loss, val_loss).
        """
        cfg = self.train_config
        torch.manual_seed(cfg.seed)

        model = self.model.to(self.device)
        print(f"Model parameters: {model.count_parameters():,}")
        print(f"Device: {self.device}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _lr_lambda(step, cfg.warmup_steps, cfg.max_steps),
        )

        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        history = {"step": [], "train_loss": [], "val_loss": []}
        model.train()
        t0 = time.time()

        for step in range(1, cfg.max_steps + 1):
            inputs, targets = train_data.get_batch(cfg.batch_size, self.device)

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            # Logging
            if step % cfg.log_every == 0:
                elapsed = time.time() - t0
                ms_per_step = elapsed / step * 1000
                lr = scheduler.get_last_lr()[0]
                print(
                    f"step {step:>6d}/{cfg.max_steps} | "
                    f"loss {loss.item():.4f} | "
                    f"lr {lr:.2e} | "
                    f"{ms_per_step:.1f} ms/step | "
                    f"{elapsed / 60:.1f} min elapsed"
                )
                history["step"].append(step)
                history["train_loss"].append(loss.item())

            # Validation
            if val_data and step % cfg.eval_every == 0:
                val_loss = self.evaluate(val_data)
                print(f"  → val loss: {val_loss:.4f}")
                history["val_loss"].append(val_loss)

            # Checkpoint
            if step % cfg.checkpoint_every == 0:
                path = ckpt_dir / f"step_{step}.pt"
                self.save_checkpoint(path, optimizer, scheduler, step)
                print(f"  → checkpoint saved: {path}")

        # Final save
        self.save_checkpoint(ckpt_dir / "final.pt", optimizer, scheduler, cfg.max_steps)
        total_time = time.time() - t0
        print(f"\nTraining complete in {total_time / 60:.1f} min")

        return history

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, tokenizer, prompt="Once upon a time",
                 max_tokens=200, temperature=1.0, top_k=0):
        """Autoregressively generate text from a prompt.

        Args:
            tokenizer:   BPETokenizer instance (for encode/decode).
            prompt:      Starting text string.
            max_tokens:  Maximum number of new tokens to generate.
            temperature: Sampling temperature (lower = more greedy).
            top_k:       If > 0, only sample from the top-k logits.

        Returns:
            Generated text string (including the prompt).
        """
        model = self.model
        model.eval()
        device = self.device

        tokens = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.long, device=device,
        )
        eos_id = tokenizer.vocab_inverse.get("<EOS>", -1)

        for _ in range(max_tokens):
            # Crop to context window
            context = tokens[-self.model_config.context_length:]
            logits = model(context.unsqueeze(0))
            next_logits = logits[0, -1] / temperature

            # Optional top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[-1]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token])

            if next_token.item() == eos_id:
                break

        model.train()
        return tokenizer.decode(tokens.tolist())

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path, optimizer=None, scheduler=None, step=0):
        """Save model + optimizer state."""
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "step": step,
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(ckpt, path)

    def load_checkpoint(self, path, optimizer=None, scheduler=None):
        """Load model + optimizer state. Returns the step number."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt.get("step", 0)

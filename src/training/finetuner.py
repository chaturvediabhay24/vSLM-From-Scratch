"""Fine-tuning loop for chatbot Q&A format.

Loads a pre-trained checkpoint and fine-tunes on Q&A pairs with
masked loss — only answer tokens contribute to the gradient.

Usage:
    finetuner = FineTuner(model, model_config, finetune_config)
    history = finetuner.finetune(
        train_data, val_data,
        pretrained_checkpoint="dataset/model-checkpoints/step_10000.pt",
        tokenizer=tokenizer,
    )
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class FineTuneConfig:
    batch_size: int = 32
    learning_rate: float = 5e-5          # much lower than pre-training
    warmup_steps: int = 100
    max_steps: int = 3000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 500
    eval_batches: int = 10
    checkpoint_every: int = 1000
    checkpoint_dir: str = "checkpoints-finetune"
    seed: int = 42
    use_amp: bool = True
    compile_model: bool = True
    sample_prompts: tuple = (
        "Question: Who is Abhay? Answer:",
        "Question: What is Abhay's current role? Answer:",
        "Question: hi Answer:",
    )
    sample_max_tokens: int = 80
    sample_temperature: float = 0.7
    sample_top_k: int = 40


def _ft_lr_lambda(step, warmup_steps, max_steps, min_lr_ratio=0.1):
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


class FineTuner:
    """Fine-tunes a pre-trained transformer on Q&A data with masked loss."""

    def __init__(self, model, model_config, finetune_config=None):
        self.model = model
        self.model_config = model_config
        self.config = finetune_config or FineTuneConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load pre-trained weights
    # ------------------------------------------------------------------

    def load_pretrained(self, checkpoint_path):
        """Load model weights from a pre-training checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        step = ckpt.get("step", 0)
        print(f"Loaded pre-trained checkpoint: {checkpoint_path} (step {step})")
        return step

    # ------------------------------------------------------------------
    # Masked loss
    # ------------------------------------------------------------------

    @staticmethod
    def masked_cross_entropy(logits, targets, loss_mask):
        """Cross-entropy loss computed only where loss_mask == 1.

        Args:
            logits:    (B, T, vocab_size)
            targets:   (B, T)
            loss_mask: (B, T) — 1 for answer positions, 0 elsewhere
        """
        B, T, V = logits.shape
        # Flatten
        loss_per_token = F.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T),
            reduction="none",
        ).reshape(B, T)

        # Apply mask and average over non-zero positions
        masked_loss = (loss_per_token * loss_mask).sum()
        n_tokens = loss_mask.sum().clamp(min=1)
        return masked_loss / n_tokens

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, dataset):
        """Average masked loss over validation batches."""
        self.model.eval()
        cfg = self.config
        use_amp = cfg.use_amp and self.device.type == "cuda"
        total = 0.0
        for _ in range(cfg.eval_batches):
            inputs, targets, loss_mask = dataset.get_batch(cfg.batch_size, self.device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = self.model(inputs)
                loss = self.masked_cross_entropy(logits, targets, loss_mask)
            total += loss.item()
        self.model.train()
        return total / cfg.eval_batches

    # ------------------------------------------------------------------
    # Main fine-tuning loop
    # ------------------------------------------------------------------

    def finetune(self, train_data, val_data=None, pretrained_checkpoint=None,
                 tokenizer=None):
        """Run fine-tuning.

        Args:
            train_data:             FineTuneDataset for training.
            val_data:               FineTuneDataset for validation (optional).
            pretrained_checkpoint:  Path to .pt checkpoint to load before training.
            tokenizer:              BPETokenizer for sample generation.

        Returns:
            dict of logged metrics.
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)

        # Load pre-trained weights
        if pretrained_checkpoint is not None:
            self.load_pretrained(pretrained_checkpoint)

        model = self.model.to(self.device)
        print(f"Model parameters: {model.count_parameters():,}")
        print(f"Device: {self.device}")
        print(f"Training examples: {len(train_data)}")

        if cfg.compile_model and hasattr(torch, "compile"):
            print("Compiling model with torch.compile ...")
            model = torch.compile(model)

        use_amp = cfg.use_amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        if use_amp:
            print("Using AMP (float16)")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _ft_lr_lambda(step, cfg.warmup_steps, cfg.max_steps),
        )

        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        history = {"step": [], "train_loss": [], "val_loss": []}
        model.train()
        t0 = time.time()

        for step in range(1, cfg.max_steps + 1):
            inputs, targets, loss_mask = train_data.get_batch(cfg.batch_size, self.device)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(inputs)
                loss = self.masked_cross_entropy(logits, targets, loss_mask)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Logging
            if step % cfg.log_every == 0:
                elapsed = time.time() - t0
                ms_per_step = elapsed / step * 1000
                lr = scheduler.get_last_lr()[0]
                print(
                    f"step {step:>5d}/{cfg.max_steps} | "
                    f"loss {loss.item():.4f} | "
                    f"lr {lr:.2e} | "
                    f"{ms_per_step:.1f} ms/step | "
                    f"{elapsed / 60:.1f} min"
                )
                history["step"].append(step)
                history["train_loss"].append(loss.item())

            # Validation + sample generation
            if val_data and step % cfg.eval_every == 0:
                val_loss = self.evaluate(val_data)
                print(f"  → val loss: {val_loss:.4f}")
                history["val_loss"].append(val_loss)

                if tokenizer is not None:
                    for prompt in cfg.sample_prompts:
                        sample = self.generate(
                            tokenizer, prompt=prompt,
                            max_tokens=cfg.sample_max_tokens,
                            temperature=cfg.sample_temperature,
                            top_k=cfg.sample_top_k,
                        )
                        print(f"  → {sample}")

            # Checkpoint
            if step % cfg.checkpoint_every == 0:
                path = ckpt_dir / f"finetune_step_{step}.pt"
                self.save_checkpoint(path, optimizer, scheduler, step)
                print(f"  → checkpoint saved: {path}")

        # Final save
        final_path = ckpt_dir / "finetune_final.pt"
        self.save_checkpoint(final_path, optimizer, scheduler, cfg.max_steps)
        total_time = time.time() - t0
        print(f"\nFine-tuning complete in {total_time / 60:.1f} min")
        print(f"Final checkpoint: {final_path}")

        return history

    # ------------------------------------------------------------------
    # Text generation (reused from Trainer pattern)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, tokenizer, prompt="Question: Who is Abhay? Answer:",
                 max_tokens=200, temperature=1.0, top_k=0,
                 repetition_penalty=1.2):
        """Autoregressively generate text from a prompt."""
        model = self.model
        model.eval()

        tokens = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.long, device=self.device,
        )
        eos_id = tokenizer.vocab_inverse.get("<EOS>", -1)

        for _ in range(max_tokens):
            context = tokens[-self.model_config.context_length:]
            logits = model(context.unsqueeze(0))
            next_logits = logits[0, -1]

            if repetition_penalty != 1.0:
                for token_id in set(tokens.tolist()):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= repetition_penalty
                    else:
                        next_logits[token_id] *= repetition_penalty

            next_logits = next_logits / temperature

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
        ckpt = {"model_state_dict": self.model.state_dict(), "step": step}
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(ckpt, path)

    def load_checkpoint(self, path, optimizer=None, scheduler=None):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt.get("step", 0)

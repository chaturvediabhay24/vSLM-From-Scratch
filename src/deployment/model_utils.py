"""Model loading, generation, attention extraction, and perplexity utilities.

All inference runs on CPU for deployment simplicity.
Torch imports are lazy so the tokenizer section works without PyTorch installed.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

CHECKPOINT_PATH = PROJECT_ROOT / "dataset" / "model-checkpoints" / "step_10000.pt"
TOKENIZER_PATH = PROJECT_ROOT / "dataset" / "tokenizer.json"


@st.cache_resource
def load_model_and_tokenizer():
    """Load trained model + tokenizer once and cache across sessions.

    Returns:
        (model, tokenizer, config, device, training_step) on success.
        (None, None, None, None, error_string) on failure.
    """
    try:
        import torch
        from src.model.config import ModelConfig
        from src.model.transformer import Transformer
        from src.tokenization.tokenizer import BPETokenizer
    except ImportError as e:
        return None, None, None, None, f"Missing dependency: {e}"

    device = torch.device("cpu")

    if not CHECKPOINT_PATH.exists():
        return None, None, None, device, f"Checkpoint not found: {CHECKPOINT_PATH}"

    if not TOKENIZER_PATH.exists():
        return None, None, None, device, f"Tokenizer not found: {TOKENIZER_PATH}"

    # Load checkpoint first to infer config from saved weights
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    # Infer context_length from positional embedding shape
    pos_emb_shape = state["pos_emb.weight"].shape  # (context_length, d_model)
    config = ModelConfig(context_length=pos_emb_shape[0])

    model = Transformer(config)
    model.load_state_dict(state)
    model.eval()

    tokenizer = BPETokenizer.load(str(TOKENIZER_PATH))
    training_step = ckpt.get("step", 0)

    return model, tokenizer, config, device, training_step


# ------------------------------------------------------------------
# Streaming generation with top-p and probability tracking
# ------------------------------------------------------------------

def generate_streaming(
    model,
    tokenizer,
    config,
    device,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.0,
    repetition_penalty: float = 1.2,
    track_probabilities: bool = False,
    seed: int | None = None,
):
    """Generator that yields one token at a time for streaming display.

    Yields dicts:
        {
            "token_id": int,
            "token_text": str,
            "top_k_probs": [(token_id, token_text, prob), ...] or None,
            "full_text_so_far": str,
        }
    """
    import torch
    import torch.nn.functional as F

    if seed is not None:
        torch.manual_seed(seed)

    tokens = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device=device,
    )
    eos_id = tokenizer.vocab_inverse.get("<EOS>", -1)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context window
            context = tokens[-config.context_length :]
            logits = model(context.unsqueeze(0))
            next_logits = logits[0, -1].clone()

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(tokens.tolist()):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= repetition_penalty
                    else:
                        next_logits[token_id] *= repetition_penalty

            # Temperature scaling
            next_logits = next_logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[-1]] = float("-inf")

            # Top-p (nucleus) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Mask tokens whose cumulative probability exceeds the threshold
                sorted_mask = cumulative_probs > top_p
                # Keep at least one token
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                indices_to_remove = sorted_indices[sorted_mask]
                next_logits[indices_to_remove] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # Track top-k probability distribution for visualization
            top_probs_data = None
            if track_probabilities:
                top_vals, top_ids = torch.topk(probs, min(10, probs.size(-1)))
                top_probs_data = []
                for tid, p in zip(top_ids, top_vals):
                    try:
                        text = tokenizer.decode([tid.item()])
                    except Exception:
                        text = f"[{tid.item()}]"
                    top_probs_data.append((tid.item(), text, p.item()))

            tokens = torch.cat([tokens, next_token])

            try:
                new_token_text = tokenizer.decode([next_token.item()])
            except Exception:
                new_token_text = ""

            yield {
                "token_id": next_token.item(),
                "token_text": new_token_text,
                "top_k_probs": top_probs_data,
                "full_text_so_far": tokenizer.decode(tokens.tolist()),
            }

            if next_token.item() == eos_id:
                break


def generate_full(
    model, tokenizer, config, device, prompt, **kwargs,
):
    """Non-streaming generation. Returns (full_text, list_of_step_dicts)."""
    steps = []
    full_text = prompt
    for step in generate_streaming(model, tokenizer, config, device, prompt, **kwargs):
        steps.append(step)
        full_text = step["full_text_so_far"]
    return full_text, steps


# ------------------------------------------------------------------
# Attention weight extraction
# ------------------------------------------------------------------

def extract_attention_weights(model, token_ids, device):
    """Run a forward pass with manual attention to extract weights.

    Mirrors CausalSelfAttention.forward() but computes Q @ K^T explicitly
    instead of using scaled_dot_product_attention.

    Returns:
        list of tensors, one per layer — each (n_heads, seq_len, seq_len)
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        x = torch.tensor([token_ids], dtype=torch.long, device=device)
        B, T = x.shape

        # Embeddings
        tok_emb = model.tok_emb(x)
        pos_emb = model.pos_emb(torch.arange(T, device=device))
        hidden = tok_emb + pos_emb  # no dropout in eval mode

        all_attention_weights = []

        for block in model.blocks:
            # Pre-norm
            normed = block.ln1(hidden)

            # Manual QKV (mirrors CausalSelfAttention)
            B_curr, T_curr, C = normed.shape
            attn = block.attn
            qkv = attn.qkv(normed)
            q, k, v = qkv.split(C, dim=-1)

            q = q.reshape(B_curr, T_curr, attn.n_heads, attn.head_dim).transpose(1, 2)
            k = k.reshape(B_curr, T_curr, attn.n_heads, attn.head_dim).transpose(1, 2)
            v = v.reshape(B_curr, T_curr, attn.n_heads, attn.head_dim).transpose(1, 2)

            # Explicit attention scores
            scale = attn.head_dim ** -0.5
            attn_scores = (q @ k.transpose(-2, -1)) * scale

            # Causal mask
            causal_mask = torch.triu(
                torch.ones(T_curr, T_curr, device=device, dtype=torch.bool), diagonal=1,
            )
            attn_scores.masked_fill_(causal_mask, float("-inf"))

            attn_weights = F.softmax(attn_scores, dim=-1)  # (B, n_heads, T, T)
            all_attention_weights.append(attn_weights[0].cpu())

            # Propagate hidden state using the fused kernel for correctness
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).reshape(B_curr, T_curr, C)
            out = attn.proj(out)
            hidden = hidden + out

            # MLP sub-block
            hidden = hidden + block.mlp(block.ln2(hidden))

    return all_attention_weights


# ------------------------------------------------------------------
# Perplexity
# ------------------------------------------------------------------

def compute_perplexity(model, tokenizer, text, config, device):
    """Compute perplexity of text under the model.

    Returns perplexity (float) or None if text is too short.
    """
    import torch
    import torch.nn.functional as F

    token_ids = tokenizer.encode(text)
    if len(token_ids) < 2:
        return None

    # Truncate to context length
    token_ids = token_ids[: config.context_length + 1]

    with torch.no_grad():
        x = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return torch.exp(loss).item()

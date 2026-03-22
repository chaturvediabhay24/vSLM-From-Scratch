"""FastAPI server — portfolio landing page + LLM demo API.

Run from project root:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(title="Abhay Chaturvedi — Portfolio & LLM Demo")

app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "portfolio"), name="static")

# ── Lazy singletons ───────────────────────────────────────────────
_tokenizer = None
_model = None
_config = None
_device = None
_training_step = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from src.tokenization.tokenizer import BPETokenizer

        _tokenizer = BPETokenizer.load(
            str(PROJECT_ROOT / "dataset" / "tokenizer.json")
        )
    return _tokenizer


def get_model():
    global _model, _config, _device, _training_step
    if _model is not None:
        return _model, _config, _device, _training_step
    import torch
    from src.model.config import ModelConfig
    from src.model.transformer import Transformer

    device = torch.device("cpu")
    ckpt_path = PROJECT_ROOT / "dataset" / "model-checkpoints" / "step_10000.pt"
    if not ckpt_path.exists():
        return None, None, device, 0

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    cfg = ModelConfig(context_length=state["pos_emb.weight"].shape[0])
    model = Transformer(cfg)
    model.load_state_dict(state)
    model.eval()

    _model, _config, _device, _training_step = model, cfg, device, ckpt.get("step", 0)
    return _model, _config, _device, _training_step


# ── Request schemas ───────────────────────────────────────────────


class TextRequest(BaseModel):
    text: str


class EncodeStepsRequest(BaseModel):
    word: str


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.0
    repetition_penalty: float = 1.2


# ── Pages ─────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    return HTMLResponse(
        (PROJECT_ROOT / "portfolio" / "index.html").read_text("utf-8")
    )


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    return HTMLResponse(
        (PROJECT_ROOT / "portfolio" / "demo.html").read_text("utf-8")
    )


# ── Tokenizer API ─────────────────────────────────────────────────


def _token_color(tid: int) -> str:
    return f"hsl({(tid * 137.508) % 360:.0f}, 70%, 82%)"


@app.post("/api/tokenize")
async def api_tokenize(req: TextRequest):
    tok = get_tokenizer()
    from src.deployment.tokenizer_utils import (
        compute_compression_ratio,
        token_id_to_display,
    )

    ids = tok.encode(req.text)
    special_ids = {tok.vocab_inverse[st] for st in tok.special_tokens if st in tok.vocab_inverse}
    tokens = [
        {
            "id": tid,
            "text": token_id_to_display(tok, tid),
            "special": tid in special_ids,
            "color": _token_color(tid),
        }
        for tid in ids
    ]
    stats = compute_compression_ratio(tok, req.text)
    decoded = tok.decode(ids)
    return {
        "tokens": tokens,
        "token_ids": ids,
        "decoded": decoded,
        "roundtrip_ok": decoded == req.text,
        "stats": stats,
    }


@app.post("/api/encode-steps")
async def api_encode_steps(req: EncodeStepsRequest):
    tok = get_tokenizer()
    from src.deployment.tokenizer_utils import encode_with_steps

    steps = encode_with_steps(tok, req.word)
    for s in steps:
        s["colors"] = [_token_color(tid) for tid in s["token_ids"]]
        s["merged_pair"] = list(s["merged_pair"]) if s["merged_pair"] else None
    return {"steps": steps}


@app.get("/api/merges")
async def api_merges(q: str = "", limit: int = 100, offset: int = 0):
    tok = get_tokenizer()
    from src.deployment.tokenizer_utils import token_id_to_display

    rows = []
    for rank, (a, b) in enumerate(tok.merges):
        ta, tb = tok.vocab[a], tok.vocab[b]
        merged = (ta + tb) if isinstance(ta, bytes) and isinstance(tb, bytes) else (str(ta) + str(tb))
        mid = tok.vocab_inverse.get(merged, -1)
        mt = token_id_to_display(tok, mid) if mid != -1 else "?"
        if q and q.lower() not in mt.lower():
            continue
        rows.append(
            {
                "rank": rank,
                "a": token_id_to_display(tok, a),
                "a_id": a,
                "b": token_id_to_display(tok, b),
                "b_id": b,
                "result": mt,
                "result_id": mid,
            }
        )
    total = len(rows)
    return {"merges": rows[offset : offset + limit], "total": total}


@app.get("/api/stats")
async def api_stats():
    tok = get_tokenizer()
    from src.deployment.tokenizer_utils import compute_stats

    return compute_stats(tok)


@app.post("/api/compression")
async def api_compression(req: TextRequest):
    tok = get_tokenizer()
    from src.deployment.tokenizer_utils import (
        char_level_tokenize,
        compute_compression_ratio,
        word_level_tokenize,
    )

    return {
        "bpe": compute_compression_ratio(tok, req.text),
        "char_tokens": len(char_level_tokenize(req.text)),
        "word_tokens": len(word_level_tokenize(req.text)),
    }


# ── Model API ─────────────────────────────────────────────────────


@app.get("/api/model-info")
async def api_model_info():
    try:
        model, cfg, device, step = get_model()
        tok = get_tokenizer()
        if model is None:
            return {"available": False}
        return {
            "available": True,
            "config": {
                "d_model": cfg.d_model,
                "n_heads": cfg.n_heads,
                "n_layers": cfg.n_layers,
                "d_ff": cfg.d_ff,
                "context_length": cfg.context_length,
                "dropout_rate": cfg.dropout_rate,
                "vocab_size": cfg.vocab_size,
            },
            "param_count": sum(p.numel() for p in model.parameters()),
            "training_step": step,
            "special_tokens": len(tok.special_tokens),
            "merges": len(tok.merges),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    model, config, device, _ = get_model()
    if model is None:
        return {"error": "Model checkpoint not found"}
    tok = get_tokenizer()

    def stream():
        import torch
        import torch.nn.functional as F

        tokens = torch.tensor(tok.encode(req.prompt), dtype=torch.long, device=device)
        eos_id = tok.vocab_inverse.get("<EOS>", -1)

        with torch.no_grad():
            for _ in range(req.max_new_tokens):
                ctx = tokens[-config.context_length :]
                logits = model(ctx.unsqueeze(0))[0, -1].clone()

                if req.repetition_penalty != 1.0:
                    for tid in set(tokens.tolist()):
                        if logits[tid] > 0:
                            logits[tid] /= req.repetition_penalty
                        else:
                            logits[tid] *= req.repetition_penalty

                logits /= max(req.temperature, 1e-8)

                if req.top_k > 0:
                    v, _ = torch.topk(logits, min(req.top_k, logits.size(-1)))
                    logits[logits < v[-1]] = float("-inf")

                if 0.0 < req.top_p < 1.0:
                    sl, si = torch.sort(logits, descending=True)
                    cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                    mask = cp > req.top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    logits[si[mask]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, nxt])

                try:
                    txt = tok.decode([nxt.item()])
                except Exception:
                    txt = ""

                yield f"data: {json.dumps({'t': txt, 'd': False})}\n\n"
                if nxt.item() == eos_id:
                    break

        yield f"data: {json.dumps({'t': '', 'd': True})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/attention")
async def api_attention(req: TextRequest):
    model, config, device, _ = get_model()
    if model is None:
        return {"error": "Model checkpoint not found"}
    tok = get_tokenizer()
    from src.deployment.tokenizer_utils import token_id_to_display

    import torch
    import torch.nn.functional as F

    ids = tok.encode(req.text)[:60]
    labels = [token_id_to_display(tok, t) for t in ids]

    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.long, device=device)
        T = x.shape[1]
        tok_emb = model.tok_emb(x)
        pos_emb = model.pos_emb(torch.arange(T, device=device))
        h = tok_emb + pos_emb
        layers = []
        for block in model.blocks:
            normed = block.ln1(h)
            B, Tc, C = normed.shape
            attn = block.attn
            qkv = attn.qkv(normed)
            q, k, v = qkv.split(C, dim=-1)
            q = q.reshape(B, Tc, attn.n_heads, attn.head_dim).transpose(1, 2)
            k = k.reshape(B, Tc, attn.n_heads, attn.head_dim).transpose(1, 2)
            v = v.reshape(B, Tc, attn.n_heads, attn.head_dim).transpose(1, 2)
            sc = (q @ k.transpose(-2, -1)) * (attn.head_dim**-0.5)
            cm = torch.triu(torch.ones(Tc, Tc, device=device, dtype=torch.bool), 1)
            sc.masked_fill_(cm, float("-inf"))
            w = F.softmax(sc, dim=-1)[0]
            layers.append([w[hi].tolist() for hi in range(w.shape[0])])
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = attn.proj(out.transpose(1, 2).reshape(B, Tc, C))
            h = h + out
            h = h + block.mlp(block.ln2(h))

    return {
        "labels": labels,
        "layers": layers,
        "n_layers": len(layers),
        "n_heads": len(layers[0]) if layers else 0,
    }


@app.post("/api/perplexity")
async def api_perplexity(req: TextRequest):
    model, config, device, _ = get_model()
    if model is None:
        return {"error": "Model checkpoint not found"}
    tok = get_tokenizer()

    import torch
    import torch.nn.functional as F

    ids = tok.encode(req.text)
    if len(ids) < 2:
        return {"perplexity": None, "token_count": len(ids)}
    ids = ids[: config.context_length + 1]
    with torch.no_grad():
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    return {"perplexity": round(torch.exp(loss).item(), 2), "token_count": len(ids)}

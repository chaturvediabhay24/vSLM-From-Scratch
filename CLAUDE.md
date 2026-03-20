# LLM From Scratch

## Project Goal
Train a small language model from scratch that takes a **configurable context length** as input and **predicts the next token** (autoregressive, one token at a time).

## Dataset
- **TinyStories** dataset located at `dataset/tiny-stories/train.csv`
- Single column: `text` — short children's stories
- Stories are concatenated into a single corpus separated by `<EOS>` tokens during preprocessing

## Project Structure
```
├── dataset/
│   ├── tiny-stories/
│   │   └── train.csv
│   ├── tokens_parts/              # Pre-tokenized .npy chunks (uint16)
│   └── tokenizer.json             # Trained BPE tokenizer
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocessing.py      # PreProcessing class: loads CSV, builds corpus
│   ├── tokenization/
│   │   ├── __init__.py
│   │   ├── tokenizer.py          # BPETokenizer class: byte-level BPE from scratch
│   │   └── corpus_encoder.py     # CorpusEncoder: parallel chunked tokenization
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py             # ModelConfig dataclass (all hyperparameters)
│   │   └── transformer.py        # Decoder-only transformer (PyTorch)
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py            # TokenDataset: loads .npy parts, serves batches
│   └── training/
│       ├── __init__.py
│       └── trainer.py            # Trainer class + TrainingConfig
├── notebooks/
│   ├── preprocess.ipynb           # Notebook for running preprocessing
│   ├── tokenizer.ipynb            # Notebook for training & testing BPE tokenizer
│   └── training.ipynb             # Notebook for model training (run on Colab GPU)
├── requirements.txt               # Dependencies for Colab (torch, numpy, etc.)
├── venv/                          # Python 3.12 virtual environment
└── CLAUDE.md
```

## Architecture Decisions
- **Context length**: Configurable parameter (default 128, supports 64–512+)
- **Task**: Next-token prediction (causal language modeling)
- **Tokenization**: Byte-level BPE (vocab_size=4096)
- **Framework**: PyTorch
- **Model**: Decoder-only transformer, GPT-2 style pre-norm (~5M params at default config)
  - Token + learned positional embeddings
  - N × TransformerBlock (LayerNorm → CausalSelfAttention → Residual, LayerNorm → MLP → Residual)
  - Final LayerNorm → Dense projection to vocab logits
- **Default config**: d_model=256, n_heads=4, n_layers=4, d_ff=1024, dropout=0.1

## Preprocessing Pipeline
- `PreProcessing` class in `src/preprocessing/preprocessing.py`
- Loads CSV, joins all stories with `<EOS>` delimiter into a single corpus string
- Usage: `PreProcessing(filepath).preprocess()` returns the full corpus

## Tokenization Pipeline
- `BPETokenizer` class in `src/tokenization/tokenizer.py`
- Byte-level BPE: base vocab of 256 bytes + `<EOS>` special token, then learned merges
- Trains on a 100K story sample (not the full 20.5M row corpus) for speed
- GPT-2 style pre-tokenization (regex word splitting); special tokens are atomic
- Usage: `BPETokenizer(vocab_size=4096).train(corpus)` → `.encode(text)` / `.decode(ids)`
- Save/load via JSON: `tokenizer.save(path)` / `BPETokenizer.load(path)`
- Trained tokenizer saved to `dataset/tokenizer.json`

## Model & Training Pipeline
- `ModelConfig` dataclass in `src/model/config.py` — all architecture hyperparameters
- `Transformer` in `src/model/transformer.py` — full model (CausalSelfAttention, MLP, TransformerBlock)
- `TokenDataset` in `src/data/dataset.py` — loads `tokens_parts/*.npy`, creates train/val split, serves batches
- `Trainer` + `TrainingConfig` in `src/training/trainer.py` — training loop, eval, generation, checkpointing
- Usage: see `notebooks/training.ipynb` (designed for Colab with GPU)
- Checkpoints saved as `.pt` files via `torch.save` (model + optimizer + scheduler state)

## Pre-tokenized Data
- `CorpusEncoder` in `src/tokenization/corpus_encoder.py` encodes full CSV in parallel chunks
- Output: `dataset/tokens_parts/part_XXXXX.npy` (uint16 arrays, ~1M tokens each)
- `TokenDataset.from_parts()` loads and concatenates all parts for training

## Development
- Python 3.12, virtualenv at `./venv`
- Notebooks use `sys.path.insert(0, "..")` to import from `src/`
- Activate env: `source venv/bin/activate`
- For Colab: PyTorch with CUDA is pre-installed, no extra installs needed

## Conventions
- Keep model code in `src/` as Python modules
- Use `notebooks/` for experimentation and visualization
- Class-based design for pipeline stages (preprocessing, tokenization, model, training)

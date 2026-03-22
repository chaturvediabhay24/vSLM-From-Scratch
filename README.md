# LLM From Scratch

Training a small language model from scratch — from raw text to next-token prediction.

**Dataset:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (~2.1M short children's stories)

## Pipeline

| Stage | Status | Description |
|-------|--------|-------------|
| Preprocessing | Done | Load CSV, concatenate stories with `<EOS>` delimiter |
| Tokenization | Done | Byte-level BPE, vocab size 4096, trained on 100K story sample — [Colab notebook](https://colab.research.google.com/drive/1Riq4f9PHAPItrwEiaKZQV_N97Hyt-rkk?usp=sharing) |
| Corpus Encoding | Done | Tokenize full 2.1M stories into token IDs (parallel, resumable) |
| Model | Done | Decoder-only transformer (~5M params), GPT-2 style pre-norm |
| Training | Done | Next-token prediction (causal LM) — [Colab notebook](https://colab.research.google.com/drive/1z9Y99AVLcChQC8aGIN7Fg5uW3g-PxbA6) |
| Deployment | Done | Interactive Streamlit UI showcasing the tokenizer |

## Demo

Run the interactive tokenizer showcase locally:

```bash
source venv/bin/activate
pip install -r requirements.txt
streamlit run src/deployment/app.py
```

Opens at `http://localhost:8501` with five tabs:

| Tab | What it does |
|-----|-------------|
| **About** | Model architecture specs, pipeline overview, design decision explanations |
| **Playground** | Type text and see colored tokens with IDs on hover, encode/decode roundtrip verification, live compression stats |
| **BPE Deep Dive** | Step-by-step merge walkthrough with slider, full vocab/merges explorer table, side-by-side comparison (BPE vs character-level vs word-level) |
| **Live Training** | Train a mini BPE tokenizer on custom text and watch merges happen in real time; edge case showcase (emojis, code, multilingual, numbers) |
| **Stats Dashboard** | Vocab metrics, token length distribution chart, token type breakdown, compression ratio calculator |

## Project Structure

```
src/
  preprocessing/
    preprocessing.py       # PreProcessing: loads CSV, builds corpus
  tokenization/
    tokenizer.py           # BPETokenizer: byte-level BPE from scratch
    corpus_encoder.py      # CorpusEncoder: parallel, resumable full-corpus encoding
  model/
    config.py              # ModelConfig: all architecture hyperparameters
    transformer.py         # Decoder-only transformer (PyTorch)
  data/
    dataset.py             # TokenDataset: loads .npy parts, serves batches
  training/
    trainer.py             # Trainer: training loop, eval, generation, checkpointing
  deployment/
    app.py                 # Streamlit app entry point
    styles.py              # Token coloring, HTML/CSS helpers
    tokenizer_utils.py     # Visualization utilities (encode with steps, stats)
    tabs/
      about.py             # Landing page tab
      playground.py        # Live tokenizer playground tab
      bpe_deep_dive.py     # Merge walkthrough + vocab explorer tab
      live_training.py     # Mini training demo + edge cases tab
      stats_dashboard.py   # Stats and charts tab
notebooks/
  preprocess.ipynb         # Preprocessing exploration
  tokenizer.ipynb          # Tokenizer training & corpus encoding
  training.ipynb           # Model training (Colab GPU)
dataset/
  tiny-stories/train.csv   # Raw stories
  tokenizer.json           # Trained BPE tokenizer
  tokens_parts/            # Pre-tokenized .npy chunks (424 parts)
```

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Use the notebooks in `notebooks/` for step-by-step exploration, or run the Streamlit demo to interact with the tokenizer.

## Architecture

- **Tokenizer:** Byte-level BPE built from scratch (no libraries) — 256 byte tokens + `<EOS>` special token + 3,839 learned merges = 4,096 vocab
- **Model:** Decoder-only transformer, d_model=256, 4 heads, 4 layers, ~5M parameters
- **Training:** Next-token prediction with AdamW, cosine LR schedule, mixed precision (AMP)

## Design Choices

- **Byte-level BPE** built from scratch (no libraries) — base vocab of 256 bytes + learned merges
- **GPT-2 style** pre-tokenization (regex word splitting) — prevents cross-word merges
- **Parallel corpus encoding** using multiprocessing (8 workers) with resume support via part files
- **Pre-norm transformer** (GPT-2 style) — LayerNorm before attention and MLP for stable training
- **Weight tying** — input and output embeddings share parameters
- **Context length:** 128 tokens (configurable 64–512+)
- **Python 3.12**

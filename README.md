# LLM From Scratch

Training a small language model from scratch — from raw text to next-token prediction.

**Dataset:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (~2.1M short children's stories)

## Pipeline

| Stage | Status | Description |
|-------|--------|-------------|
| Preprocessing | Done | Load CSV, concatenate stories with `<EOS>` delimiter |
| Tokenization | Done | Byte-level BPE, vocab size 4096, trained on 100K story sample |
| Corpus Encoding | Done | Tokenize full 2.1M stories into token IDs (parallel, resumable) |
| Model | TBD | Decoder-only transformer, context length 100 |
| Training | TBD | Next-token prediction (causal LM) |

## Project Structure

```
src/
  preprocessing/
    preprocessing.py     # PreProcessing: loads CSV, builds corpus
  tokenization/
    tokenizer.py         # BPETokenizer: byte-level BPE from scratch
    corpus_encoder.py    # CorpusEncoder: parallel, resumable full-corpus encoding
notebooks/
  preprocess.ipynb       # Preprocessing exploration
  tokenizer.ipynb        # Tokenizer training & corpus encoding
dataset/
  tiny-stories/train.csv # Raw stories
  tokenizer.json         # Trained BPE tokenizer
  tokens.npy             # Full corpus as uint16 token IDs
```

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install numpy pandas jupyterlab
```

Use the notebooks in `notebooks/` for step-by-step exploration.

## Design Choices

- **Byte-level BPE** built from scratch (no libraries) — base vocab of 256 bytes + learned merges
- **GPT-2 style** pre-tokenization (regex word splitting)
- **Parallel corpus encoding** using multiprocessing (8 workers) with resume support via part files
- **Context length:** 100 tokens
- **Python 3.12**

# LLM From Scratch

## Project Goal
Train a small language model from scratch that takes **100 tokens as input** and **predicts the next token** (autoregressive, one token at a time).

## Dataset
- **TinyStories** dataset located at `dataset/tiny-stories/train.csv`
- Single column: `text` — short children's stories
- Stories are concatenated into a single corpus separated by `<EOS>` tokens during preprocessing

## Project Structure
```
├── dataset/
│   └── tiny-stories/
│       └── train.csv
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocessing.py      # PreProcessing class: loads CSV, builds corpus
│   ├── tokenization/
│   │   ├── __init__.py
│   │   └── tokenizer.py          # BPETokenizer class: byte-level BPE from scratch
│   └── notebooks/
│       ├── preprocess.ipynb       # Notebook for running preprocessing
│       └── tokenizer.ipynb        # Notebook for training & testing BPE tokenizer
├── venv/                          # Python 3.12 virtual environment
└── CLAUDE.md
```

## Architecture Decisions
- **Context length**: 100 tokens
- **Task**: Next-token prediction (causal language modeling)
- **Tokenization**: Byte-level BPE (vocab_size=4096)
- **Model**: TBD (decoder-only transformer)

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

## Development
- Python 3.12, virtualenv at `./venv`
- Notebooks use `sys.path.insert(0, "../..")` to import from `src/`
- Activate env: `source venv/bin/activate`

## Conventions
- Keep model code in `src/` as Python modules
- Use `src/notebooks/` for experimentation and visualization
- Class-based design for pipeline stages (preprocessing, tokenization, model, training)

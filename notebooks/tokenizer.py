"""Local tokenization pipeline.

Encodes TinyStories + personal-details into token parts under dataset/.
Requires: dataset/tokenizer.json (trained BPE tokenizer from Colab).

Usage:
    cd llm-from-scratch
    python notebooks/tokenizer.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tokenization.tokenizer import BPETokenizer
from src.tokenization.corpus_encoder import CorpusEncoder

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "dataset"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
CSV_PATH = DATA_DIR / "tiny-stories" / "train.csv"
PERSONAL_DIR = DATA_DIR / "personal-details"
TOKENS_PARTS_DIR = DATA_DIR / "tokens_parts"
PERSONAL_PARTS_DIR = DATA_DIR / "tokens_parts_personal"

# ── Settings ───────────────────────────────────────────────────────────
N_WORKERS = 8           # parallel workers for story encoding (use ncpu - 2)
STORIES_PER_CHUNK = 5_000
PERSONAL_REPEATS = 50   # repeat personal details so model sees them often

# ── 1. Load tokenizer ─────────────────────────────────────────────────
print(f"Loading tokenizer from {TOKENIZER_PATH}")
tokenizer = BPETokenizer.load(str(TOKENIZER_PATH))
print(f"Vocab size: {len(tokenizer.vocab)}")

# Quick sanity check
test = "Once upon a time<EOS>"
assert tokenizer.decode(tokenizer.encode(test)) == test
print("Tokenizer roundtrip OK\n")

# ── 2. Encode TinyStories ─────────────────────────────────────────────
print("=" * 60)
print("Encoding TinyStories")
print("=" * 60)

encoder = CorpusEncoder(tokenizer, parts_dir=str(TOKENS_PARTS_DIR))
encoder.encode_parts(
    csv_path=str(CSV_PATH),
    stories_per_chunk=STORIES_PER_CHUNK,
    n_workers=N_WORKERS,
)

# ── 3. Encode Personal Details ────────────────────────────────────────
print("\n" + "=" * 60)
print("Encoding Personal Details")
print("=" * 60)

personal_texts = []
for csv_file in sorted(PERSONAL_DIR.glob("*.csv")):
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        count = 0
        for row in reader:
            if len(row) >= 2:
                q = row[0].strip()
                a = ",".join(row[1:]).strip()
                personal_texts.append(f"Question: {q} Answer: {a}")
                count += 1
    print(f"  {csv_file.name}: {count} rows")

print(f"Total personal Q&A pairs: {len(personal_texts)}")

personal_corpus = "<EOS>".join(personal_texts) + "<EOS>"
personal_ids = tokenizer.encode(personal_corpus)
personal_arr = np.array(personal_ids, dtype=np.uint16)

# Tile and save as parts
personal_repeated = np.tile(personal_arr, PERSONAL_REPEATS)
PERSONAL_PARTS_DIR.mkdir(parents=True, exist_ok=True)

tokens_per_part = 1_000_000
n_parts = max(1, (len(personal_repeated) + tokens_per_part - 1) // tokens_per_part)

for i in range(n_parts):
    start = i * tokens_per_part
    end = min(start + tokens_per_part, len(personal_repeated))
    np.save(PERSONAL_PARTS_DIR / f"part_{i:05d}.npy", personal_repeated[start:end])

print(f"Personal details: {len(personal_arr):,} tokens x {PERSONAL_REPEATS} = {len(personal_repeated):,} tokens")
print(f"Saved as {n_parts} part(s) to {PERSONAL_PARTS_DIR}")

# ── 4. Summary ─────────────────────────────────────────────────────────
story_parts = sorted(TOKENS_PARTS_DIR.glob("part_*.npy"))
personal_parts = sorted(PERSONAL_PARTS_DIR.glob("part_*.npy"))
print(f"\nDone!")
print(f"  Story parts:    {len(story_parts)} files in {TOKENS_PARTS_DIR}")
print(f"  Personal parts: {len(personal_parts)} files in {PERSONAL_PARTS_DIR}")

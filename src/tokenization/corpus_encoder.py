"""Chunked, resumable, parallel corpus encoder.

Tokenizes a CSV corpus in batches, saves intermediate part files for resume
support, and merges into a single numpy array when all parts are done.

Usage:
    encoder = CorpusEncoder(tokenizer, parts_dir="dataset/tokens_parts")
    encoder.encode_parts("train.csv", stories_per_chunk=5_000, n_workers=8)
    tokens = encoder.merge_parts("dataset/tokens.npy")
"""

import re
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

# ---- module-level worker state (inherited by fork) ----
_mp_tokenizer = None
_mp_parts_dir = None
_mp_eos = "<EOS>"


def _encode_chunk(args):
    """Worker: encode a batch of stories and save as part_XXXXX.npy."""
    chunk_idx, stories = args
    chunk_text = _mp_eos.join(stories) + _mp_eos
    ids = _mp_tokenizer.encode_fast(chunk_text)
    arr = np.array(ids, dtype=np.uint16)
    np.save(Path(_mp_parts_dir) / f"part_{chunk_idx:05d}.npy", arr)
    return chunk_idx, len(arr)


class CorpusEncoder:
    """Tokenize an entire CSV corpus with chunked processing, resume, and parallelism.

    Args:
        tokenizer: A trained BPETokenizer instance.
        parts_dir: Directory for intermediate part files (enables resume).
    """

    def __init__(self, tokenizer, parts_dir):
        self.tokenizer = tokenizer
        self.parts_dir = Path(parts_dir)

    def _completed_parts(self):
        """Return set of part indices already saved to disk."""
        done = set()
        if self.parts_dir.exists():
            for f in self.parts_dir.glob("part_*.npy"):
                try:
                    done.add(int(f.stem.split("_")[1]))
                except (IndexError, ValueError):
                    pass
        return done

    def encode_parts(self, csv_path, stories_per_chunk=5_000, n_workers=1):
        """Encode CSV into part files. Resumable — skips already-saved parts.

        Args:
            csv_path:          Path to CSV with a 'text' column.
            stories_per_chunk: Stories per part file (smaller = more resumable).
            n_workers:         Parallel workers (recommended: ncpu - 2). 1 = sequential.
        """
        csv_path = Path(csv_path)
        self.parts_dir.mkdir(parents=True, exist_ok=True)

        # Load stories
        print("Loading stories...")
        stories = pd.read_csv(csv_path, usecols=["text"])["text"].dropna().astype(str).tolist()
        n = len(stories)
        total_parts = (n + stories_per_chunk - 1) // stories_per_chunk
        print(f"{n:,} stories -> {total_parts} parts of {stories_per_chunk:,}\n")

        # Resume detection
        done = self._completed_parts()
        if done:
            print(f"Resuming — {len(done)}/{total_parts} parts already done, skipping\n")

        # Build work items for missing parts only
        work = []
        for i in range(total_parts):
            if i not in done:
                s = i * stories_per_chunk
                work.append((i, stories[s : s + stories_per_chunk]))

        if not work:
            print("All parts already encoded!")
            del stories
            return

        # Set module-level state for worker function
        global _mp_tokenizer, _mp_parts_dir
        _mp_tokenizer = self.tokenizer
        _mp_parts_dir = str(self.parts_dir)

        t0 = time.time()
        completed = len(done)

        if n_workers <= 1:
            # Sequential
            for item in work:
                chunk_idx, n_tok = _encode_chunk(item)
                completed += 1
                elapsed = time.time() - t0
                new_done = completed - len(done)
                eta = elapsed / new_done * (total_parts - completed) if new_done else 0
                print(f"  [{completed}/{total_parts}] part_{chunk_idx:05d} — "
                      f"{n_tok:,} tokens — {elapsed:.0f}s elapsed — ~{eta:.0f}s remaining")
        else:
            # Parallel using fork (children inherit tokenizer via COW memory)
            ctx = mp.get_context("fork")
            with ctx.Pool(n_workers) as pool:
                for chunk_idx, n_tok in pool.imap_unordered(_encode_chunk, work):
                    completed += 1
                    elapsed = time.time() - t0
                    new_done = completed - len(done)
                    eta = elapsed / new_done * (total_parts - completed) if new_done else 0
                    print(f"  [{completed}/{total_parts}] part_{chunk_idx:05d} — "
                          f"{n_tok:,} tokens — {elapsed:.0f}s elapsed — ~{eta:.0f}s remaining")

        del stories, work
        print(f"\nAll parts encoded in {(time.time() - t0) / 60:.1f} min")

    def merge_parts(self, output_path):
        """Concatenate all part files into a single tokens.npy.

        Args:
            output_path: Where to write the merged numpy array.

        Returns:
            The concatenated token array (uint16).
        """
        output_path = Path(output_path)
        part_files = sorted(self.parts_dir.glob("part_*.npy"))
        print(f"Merging {len(part_files)} part files...")

        tokens = np.concatenate([np.load(p) for p in part_files])
        np.save(output_path, tokens)

        print(f"Total tokens: {len(tokens):,}")
        print(f"Saved to {output_path} ({tokens.nbytes / 1e6:.1f} MB)")
        return tokens

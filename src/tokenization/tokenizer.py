"""
Byte-Pair Encoding (BPE) Tokenizer — Revision Summary
=======================================================

HOW BPE WORKS:
  1. Start with a base vocabulary of 256 tokens (one per byte value) + special tokens.
  2. Convert the training corpus into byte sequences.
  3. Repeat until vocab reaches target size:
     a. Count every adjacent pair of tokens across the corpus.
     b. Find the most frequent pair  →  merge it into one new token.
     c. Replace all occurrences of that pair in the corpus with the new token.
  The ordered list of merges IS the tokenizer — it defines how text is compressed.

FILE OVERVIEW (method-by-method):
  _init_base_vocab     — Creates IDs 0-255 (bytes) + special tokens like <EOS>.
  _pre_tokenize        — Splits raw text into chunks (GPT-2 regex) so BPE
                          operates within word boundaries, not across them.
  _build_word_freqs    — Converts each chunk to a byte-ID tuple and counts how
                          many times each unique tuple appears.
  _get_pair_counts     — Scans all tuples, counts every adjacent (id_a, id_b) pair
                          weighted by word frequency.
  _merge_pair          — Given the winning pair, walks every tuple and replaces
                          consecutive (id_a, id_b) with the new merged id.
  train                — Orchestrates the loop: init vocab → build freqs →
                          repeat {count pairs → pick best → merge} until done.
  encode               — Tokenizes new text: start from bytes, then replay merges
                          in priority order (lowest rank = learned first = merge first).
                          Uses "find-min-rank" approach: each iteration scans all
                          adjacent pairs to find the one with lowest rank, then merges
                          it. Clear but re-scans pairs every iteration.
  encode_fast          — Same result, different strategy: iterates the merge list
                          in order and applies each merge if the pair exists. Only
                          walks the token list once per applicable merge instead of
                          re-scanning all pairs each time. Faster in practice because
                          most merges don't apply to any given word and are skipped
                          with a cheap set lookup.
  decode               — Maps token IDs back to bytes/strings and joins them.
  save / load          — JSON serialization of vocab + merge list.
"""

import re
import json
from collections import defaultdict

# GPT-2 style pre-tokenization: splits on word boundaries, contractions, punctuation
PRE_TOKENIZE_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


class BPETokenizer:
    """Byte Pair Encoding tokenizer built from scratch.

    Byte-level BPE: starts with 256 byte tokens as the base vocabulary,
    then iteratively merges the most frequent adjacent pair until the
    target vocabulary size is reached.

    Usage:
        tokenizer = BPETokenizer(vocab_size=4096, special_tokens=["<EOS>"])
        tokenizer.train(corpus)
        ids = tokenizer.encode("Once upon a time")
        text = tokenizer.decode(ids)
    """

    def __init__(self, vocab_size=4096, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<EOS>"]

        self.merges = []          # list of (id_a, id_b) in merge priority order
        self.vocab = {}           # {token_id: bytes | str}
        self.vocab_inverse = {}   # {bytes | str: token_id}
        self.next_id = 0

        # internal training state
        self._word_freqs = None   # {tuple_of_token_ids: count}
        self._merge_rank = None   # {(id_a, id_b): rank} — built lazily for encode

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _init_base_vocab(self):
        self.vocab = {}
        # IDs 0-255: one per byte value
        for i in range(256):
            self.vocab[i] = bytes([i])

        # IDs 256+: special tokens
        for idx, token in enumerate(self.special_tokens):
            self.vocab[256 + idx] = token

        self.vocab_inverse = {v: k for k, v in self.vocab.items()}
        self.next_id = 256 + len(self.special_tokens)
        return self

    def _pre_tokenize(self, text):
        """Split text into chunks that BPE operates on independently.

        Special tokens are isolated as their own chunks so they are never
        split or merged with surrounding text.
        """
        # split on special tokens while keeping them as separate parts
        special_pattern = "|".join(re.escape(st) for st in self.special_tokens)
        parts = re.split(f"({special_pattern})", text)

        result = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                result.append(part)
            else:
                result.extend(PRE_TOKENIZE_PATTERN.findall(part))
        return result

    def _build_word_freqs(self, text):
        """Count unique byte-level token sequences weighted by frequency."""
        words = self._pre_tokenize(text)
        self._word_freqs = defaultdict(int)

        for word in words:
            if word in self.special_tokens:
                token_id = self.vocab_inverse[word]
                self._word_freqs[(token_id,)] += 1
            else:
                byte_seq = tuple(word.encode("utf-8"))
                self._word_freqs[byte_seq] += 1

        return self

    def _get_pair_counts(self):
        pair_counts = defaultdict(int)
        for token_seq, freq in self._word_freqs.items():
            for i in range(len(token_seq) - 1):
                pair_counts[(token_seq[i], token_seq[i + 1])] += freq
        return pair_counts

    def _merge_pair(self, pair, new_id):
        new_word_freqs = {}
        for token_seq, freq in self._word_freqs.items():
            new_seq = []
            i = 0
            while i < len(token_seq):
                if (
                    i < len(token_seq) - 1
                    and token_seq[i] == pair[0]
                    and token_seq[i + 1] == pair[1]
                ):
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(token_seq[i])
                    i += 1
            new_word_freqs[tuple(new_seq)] = freq
        self._word_freqs = new_word_freqs
        return self

    def train(self, corpus, verbose=True):
        """Learn BPE merges from a corpus string.

        Args:
            corpus:  The training text (use a sampled subset for large datasets).
            verbose: Print progress every 100 merges.

        Returns:
            self (fluent interface).
        """
        self._init_base_vocab()
        self._build_word_freqs(corpus)
        self._merge_rank = None  # invalidate cache

        num_merges = self.vocab_size - self.next_id

        for i in range(num_merges):
            pair_counts = self._get_pair_counts()
            if not pair_counts:
                if verbose:
                    print(f"No more pairs to merge at step {i}. Stopping early.")
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            # build the new token by concatenating the two parts
            token_a = self.vocab[best_pair[0]]
            token_b = self.vocab[best_pair[1]]
            if isinstance(token_a, bytes) and isinstance(token_b, bytes):
                new_token = token_a + token_b
            else:
                new_token = str(token_a) + str(token_b)

            new_id = self.next_id
            self.vocab[new_id] = new_token
            self.vocab_inverse[new_token] = new_id
            self.merges.append(best_pair)
            self.next_id += 1

            self._merge_pair(best_pair, new_id)

            if verbose and (i + 1) % 100 == 0:
                display = (
                    new_token.decode("utf-8", errors="replace")
                    if isinstance(new_token, bytes)
                    else new_token
                )
                print(
                    f"Merge {i + 1}/{num_merges}: "
                    f"{best_pair} -> {new_id} ('{display}', count={best_count:,})"
                )

        if verbose:
            print(f"\nTraining complete. Vocabulary size: {len(self.vocab)}")

        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text):
        """Convert text to a list of token IDs."""
        if self._merge_rank is None:
            self._merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}

        words = self._pre_tokenize(text)
        all_ids = []

        for word in words:
            if word in self.special_tokens:
                all_ids.append(self.vocab_inverse[word])
                continue

            # start with byte-level token IDs
            token_ids = list(word.encode("utf-8"))

            # repeatedly merge the lowest-rank (highest-priority) pair
            while len(token_ids) >= 2:
                min_rank = float("inf")
                min_pair = None
                for j in range(len(token_ids) - 1):
                    pair = (token_ids[j], token_ids[j + 1])
                    rank = self._merge_rank.get(pair, float("inf"))
                    if rank < min_rank:
                        min_rank = rank
                        min_pair = pair

                if min_pair is None or min_rank == float("inf"):
                    break

                merged_id = self.vocab_inverse[
                    self.vocab[min_pair[0]] + self.vocab[min_pair[1]]
                    if isinstance(self.vocab[min_pair[0]], bytes)
                    and isinstance(self.vocab[min_pair[1]], bytes)
                    else str(self.vocab[min_pair[0]]) + str(self.vocab[min_pair[1]])
                ]

                new_ids = []
                i = 0
                while i < len(token_ids):
                    if (
                        i < len(token_ids) - 1
                        and token_ids[i] == min_pair[0]
                        and token_ids[i + 1] == min_pair[1]
                    ):
                        new_ids.append(merged_id)
                        i += 2
                    else:
                        new_ids.append(token_ids[i])
                        i += 1
                token_ids = new_ids

            all_ids.extend(token_ids)

        return all_ids

    def encode_fast(self, text):
        words = self._pre_tokenize(text)
        all_ids = []

        for word in words:
            if word in self.special_tokens:
                all_ids.append(self.vocab_inverse[word])
                continue

            # start with byte-level token IDs
            token_ids = list(word.encode("utf-8"))

            # apply merges in training order (merge 0 first, then 1, ...)
            for pair in self.merges:
                if len(token_ids) < 2:
                    break

                # quick check: is this pair even present?
                pair_set = set(zip(token_ids, token_ids[1:]))
                if pair not in pair_set:
                    continue

                # look up the merged token ID
                token_a, token_b = self.vocab[pair[0]], self.vocab[pair[1]]
                if isinstance(token_a, bytes) and isinstance(token_b, bytes):
                    merged_id = self.vocab_inverse[token_a + token_b]
                else:
                    merged_id = self.vocab_inverse[str(token_a) + str(token_b)]

                # walk and replace
                new_ids = []
                i = 0
                while i < len(token_ids):
                    if (
                        i < len(token_ids) - 1
                        and token_ids[i] == pair[0]
                        and token_ids[i + 1] == pair[1]
                    ):
                        new_ids.append(merged_id)
                        i += 2
                    else:
                        new_ids.append(token_ids[i])
                        i += 1
                token_ids = new_ids

            all_ids.extend(token_ids)

        return all_ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids):
        """Convert a list of token IDs back to text."""
        parts = []
        for token_id in ids:
            token = self.vocab[token_id]
            if isinstance(token, bytes):
                parts.append(token.decode("utf-8", errors="replace"))
            else:
                parts.append(token)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """Save tokenizer state to a JSON file."""
        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "merges": [list(m) for m in self.merges],
            "vocab": {
                str(k): list(v) if isinstance(v, bytes) else v
                for k, v in self.vocab.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return self

    @classmethod
    def load(cls, path):
        """Load a trained tokenizer from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            special_tokens=data["special_tokens"],
        )
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.vocab = {}
        for k, v in data["vocab"].items():
            token_id = int(k)
            tokenizer.vocab[token_id] = bytes(v) if isinstance(v, list) else v
        tokenizer.vocab_inverse = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.next_id = max(tokenizer.vocab.keys()) + 1
        return tokenizer

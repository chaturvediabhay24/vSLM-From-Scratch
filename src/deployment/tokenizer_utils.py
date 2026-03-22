"""Utility functions wrapping BPETokenizer for the deployment UI.

Provides visualization helpers (encode with intermediate steps, token display,
stats computation) without modifying the original tokenizer class.
"""

import re


def token_id_to_display(tokenizer, token_id: int) -> str:
    """Convert a token ID to a human-readable string."""
    token = tokenizer.vocab.get(token_id)
    if token is None:
        return f"[UNK:{token_id}]"
    if isinstance(token, bytes):
        return token.decode("utf-8", errors="replace")
    return token


def encode_with_steps(tokenizer, word: str) -> list[dict]:
    """Encode a single pre-tokenized word and capture each merge step.

    Mirrors the logic of BPETokenizer.encode() (lines 237-271) but records
    the intermediate token sequence after each merge.

    Args:
        tokenizer: A loaded BPETokenizer instance.
        word: A single pre-tokenized chunk (NOT a special token).

    Returns:
        List of step dicts, each containing:
            step: int (0 = initial byte state)
            token_ids: list[int]
            token_texts: list[str]
            merged_pair: tuple[int, int] | None
            merged_pair_text: str | None
            merge_rank: int | None
            new_token_id: int | None
            new_token_text: str | None
    """
    # Ensure merge rank is built
    if tokenizer._merge_rank is None:
        tokenizer._merge_rank = {
            pair: rank for rank, pair in enumerate(tokenizer.merges)
        }

    # Start with byte-level token IDs
    token_ids = list(word.encode("utf-8"))

    steps = [
        {
            "step": 0,
            "token_ids": list(token_ids),
            "token_texts": [token_id_to_display(tokenizer, tid) for tid in token_ids],
            "merged_pair": None,
            "merged_pair_text": None,
            "merge_rank": None,
            "new_token_id": None,
            "new_token_text": None,
        }
    ]

    while len(token_ids) >= 2:
        # Find the lowest-rank (highest-priority) pair
        min_rank = float("inf")
        min_pair = None
        for j in range(len(token_ids) - 1):
            pair = (token_ids[j], token_ids[j + 1])
            rank = tokenizer._merge_rank.get(pair, float("inf"))
            if rank < min_rank:
                min_rank = rank
                min_pair = pair

        if min_pair is None or min_rank == float("inf"):
            break

        # Look up the merged token ID (same logic as tokenizer.encode)
        token_a = tokenizer.vocab[min_pair[0]]
        token_b = tokenizer.vocab[min_pair[1]]
        if isinstance(token_a, bytes) and isinstance(token_b, bytes):
            merged_id = tokenizer.vocab_inverse[token_a + token_b]
        else:
            merged_id = tokenizer.vocab_inverse[str(token_a) + str(token_b)]

        # Walk and replace
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

        pair_text_a = token_id_to_display(tokenizer, min_pair[0])
        pair_text_b = token_id_to_display(tokenizer, min_pair[1])

        steps.append(
            {
                "step": len(steps),
                "token_ids": list(token_ids),
                "token_texts": [
                    token_id_to_display(tokenizer, tid) for tid in token_ids
                ],
                "merged_pair": min_pair,
                "merged_pair_text": f"'{pair_text_a}' + '{pair_text_b}'",
                "merge_rank": int(min_rank),
                "new_token_id": merged_id,
                "new_token_text": token_id_to_display(tokenizer, merged_id),
            }
        )

    return steps


def compute_stats(tokenizer) -> dict:
    """Compute aggregate statistics about the tokenizer vocabulary."""
    byte_tokens = 256
    special = len(tokenizer.special_tokens)
    learned = len(tokenizer.vocab) - byte_tokens - special

    # Token lengths: how many bytes each vocab entry represents
    token_lengths = {}
    for tid, token in tokenizer.vocab.items():
        if isinstance(token, bytes):
            length = len(token)
        else:
            length = len(token.encode("utf-8"))
        token_lengths[length] = token_lengths.get(length, 0) + 1

    return {
        "vocab_size": len(tokenizer.vocab),
        "num_merges": len(tokenizer.merges),
        "num_byte_tokens": byte_tokens,
        "num_special_tokens": special,
        "num_learned_tokens": learned,
        "token_lengths": token_lengths,
    }


def compute_compression_ratio(tokenizer, text: str) -> dict:
    """Compute compression stats for a given text."""
    token_ids = tokenizer.encode(text)
    num_bytes = len(text.encode("utf-8"))
    num_tokens = len(token_ids)
    num_words = len(text.split())
    return {
        "num_tokens": num_tokens,
        "num_bytes": num_bytes,
        "num_chars": len(text),
        "num_words": max(num_words, 1),
        "bytes_per_token": round(num_bytes / max(num_tokens, 1), 2),
        "tokens_per_word": round(num_tokens / max(num_words, 1), 2),
        "compression_ratio": round(num_bytes / max(num_tokens, 1), 2),
    }


def char_level_tokenize(text: str) -> list[str]:
    """Character-level baseline tokenization."""
    return list(text)


def word_level_tokenize(text: str) -> list[str]:
    """Word-level baseline tokenization (split on whitespace + punctuation)."""
    return re.findall(r"\w+|[^\w\s]|\s+", text)

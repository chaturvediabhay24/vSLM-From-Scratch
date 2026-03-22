"""Tab 4: Live Training Demo — train a mini tokenizer + edge case showcase."""

import streamlit as st
import pandas as pd

from src.tokenization.tokenizer import BPETokenizer
from src.deployment.styles import GLOBAL_CSS, render_colored_tokens
from src.deployment.tokenizer_utils import (
    token_id_to_display,
    compute_compression_ratio,
)

DEFAULT_CORPUS = (
    "the cat sat on the mat. the cat ate the rat. "
    "the dog sat on the log. the dog ate the frog. "
    "a big cat and a small dog played in the park. "
    "the little cat liked the big dog very much."
)

EDGE_CASES = [
    {
        "label": "Emojis",
        "text": "I love cats! \U0001f63b\U0001f431 So cute! \u2764\ufe0f",
        "note": (
            "Emojis are 3-4 bytes in UTF-8. Since they rarely appear in TinyStories, "
            "they remain as individual byte tokens — no merges learned for them."
        ),
    },
    {
        "label": "Code snippet",
        "text": 'def hello():\n    print("Hello, world!")\n    return 42',
        "note": (
            "Code has very different statistical patterns than children's stories. "
            "Common code tokens like 'def', 'print', 'return' may partially match "
            "English words, but indentation and symbols stay as bytes."
        ),
    },
    {
        "label": "Multilingual",
        "text": "Bonjour le monde! \u4f60\u597d\u4e16\u754c Hello world!",
        "note": (
            "Byte-level BPE handles any Unicode. French shares many English subwords. "
            "Chinese characters are 3 bytes each in UTF-8 and stay as byte tokens "
            "since they never appeared in the English training corpus."
        ),
    },
    {
        "label": "Numbers",
        "text": "The year is 2024 and pi = 3.14159265358979",
        "note": (
            "Digits are single bytes (0x30-0x39). Common digit pairs like '19', '20' "
            "may have learned merges from years in stories. Long decimal sequences stay "
            "mostly unmerged."
        ),
    },
    {
        "label": "Repetition",
        "text": "aaaaaaaaaaaaa bbbbbbbbbbb abababababab",
        "note": (
            "Repeated characters show how BPE merges greedily. If 'aa' was learned as "
            "a merge, 'aaaa' becomes two 'aa' tokens. 'abab' patterns may or may not "
            "have dedicated merges depending on training corpus frequency."
        ),
    },
    {
        "label": "Punctuation heavy",
        "text": 'Wait... Really?! "Yes!!!" she said. (Wow!) [Amazing] {Incredible}',
        "note": (
            "Punctuation marks are individual bytes. Repeated punctuation like '...' or "
            "'!!!' may have learned merges if they appear frequently in stories. "
            "Brackets and braces are rare in TinyStories and stay as byte tokens."
        ),
    },
]


def render_live_training_tab(tokenizer):
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    st.subheader("Live Training & Edge Cases")

    section = st.radio(
        "Section",
        ["Mini Training Demo", "Edge Case Showcase"],
        horizontal=True,
        label_visibility="collapsed",
        key="live_training_section",
    )

    if section == "Mini Training Demo":
        _render_mini_training(tokenizer)
    else:
        _render_edge_cases(tokenizer)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mini Training Demo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _render_mini_training(full_tokenizer):
    st.markdown("#### Train Your Own Mini Tokenizer")
    st.caption(
        "Paste a small corpus and watch BPE learn merge rules in real time. "
        "This proves the tokenizer is built from scratch — you can see every step."
    )

    corpus = st.text_area(
        "Training corpus (paste any text):",
        value=DEFAULT_CORPUS,
        height=120,
        key="mini_corpus",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        target_vocab = st.slider(
            "Target vocab size",
            min_value=260,
            max_value=350,
            value=280,
            step=5,
            key="mini_vocab_size",
            help="256 bytes + 1 special token + N learned merges. Keep small for speed.",
        )
    with col2:
        char_limit = 10_000
        corpus_len = len(corpus)
        if corpus_len > char_limit:
            st.warning(f"Corpus truncated to {char_limit:,} chars for speed.")
            corpus = corpus[:char_limit]
        st.markdown(f"Corpus: **{corpus_len:,}** chars | Will learn **{target_vocab - 257}** merges")

    if st.button("Train!", key="train_btn", type="primary"):
        _run_mini_training(corpus, target_vocab, full_tokenizer)

    # Show previous results if available
    if "mini_tokenizer" in st.session_state and not st.session_state.get("training_in_progress"):
        _show_training_results(full_tokenizer)


def _run_mini_training(corpus: str, target_vocab: int, full_tokenizer):
    """Run BPE training step-by-step with live UI updates."""
    st.session_state["training_in_progress"] = True

    mini_tok = BPETokenizer(vocab_size=target_vocab, special_tokens=["<EOS>"])
    mini_tok._init_base_vocab()
    mini_tok._build_word_freqs(corpus)

    num_merges = target_vocab - mini_tok.next_id
    merge_records = []

    progress = st.progress(0, text="Starting training...")
    status = st.empty()
    merge_table = st.empty()

    for i in range(num_merges):
        pair_counts = mini_tok._get_pair_counts()
        if not pair_counts:
            status.info(f"No more pairs to merge at step {i}. Stopping early.")
            break

        best_pair = max(pair_counts, key=pair_counts.get)
        best_count = pair_counts[best_pair]

        # Create new token (same logic as BPETokenizer.train)
        token_a = mini_tok.vocab[best_pair[0]]
        token_b = mini_tok.vocab[best_pair[1]]
        if isinstance(token_a, bytes) and isinstance(token_b, bytes):
            new_token = token_a + token_b
        else:
            new_token = str(token_a) + str(token_b)

        new_id = mini_tok.next_id
        mini_tok.vocab[new_id] = new_token
        mini_tok.vocab_inverse[new_token] = new_id
        mini_tok.merges.append(best_pair)
        mini_tok.next_id += 1

        mini_tok._merge_pair(best_pair, new_id)

        display = (
            new_token.decode("utf-8", errors="replace")
            if isinstance(new_token, bytes)
            else new_token
        )

        merge_records.append(
            {
                "Merge #": i + 1,
                "Pair": f"{token_id_to_display(mini_tok, best_pair[0])!r} + {token_id_to_display(mini_tok, best_pair[1])!r}",
                "Result": repr(display),
                "ID": new_id,
                "Count": best_count,
            }
        )

        progress.progress(
            (i + 1) / num_merges,
            text=f"Merge {i + 1}/{num_merges}: {repr(display)} (count={best_count:,})",
        )

        # Update table periodically
        if (i + 1) % 5 == 0 or i == num_merges - 1:
            merge_table.dataframe(
                pd.DataFrame(merge_records),
                use_container_width=True,
                hide_index=True,
            )

    progress.progress(1.0, text="Training complete!")
    status.success(
        f"Learned **{len(merge_records)}** merges. "
        f"Final vocab size: **{len(mini_tok.vocab):,}**"
    )

    # Store in session state
    st.session_state["mini_tokenizer"] = mini_tok
    st.session_state["mini_merge_records"] = merge_records
    st.session_state["training_in_progress"] = False


def _show_training_results(full_tokenizer):
    """Show testing panel for the trained mini tokenizer."""
    mini_tok = st.session_state["mini_tokenizer"]

    st.divider()
    st.markdown("#### Test Your Mini Tokenizer")

    test_text = st.text_input(
        "Enter text to tokenize:",
        value="the cat sat on the mat",
        key="mini_test_input",
    )

    if not test_text:
        return

    col_mini, col_full = st.columns(2)

    with col_mini:
        st.markdown("**Your mini tokenizer**")
        mini_ids = mini_tok.encode(test_text)
        st.metric("Tokens", len(mini_ids))
        tokens_data = [
            (tid, token_id_to_display(mini_tok, tid), False) for tid in mini_ids
        ]
        html = render_colored_tokens(tokens_data)
        st.markdown(html, unsafe_allow_html=True)

    with col_full:
        st.markdown("**Full tokenizer (vocab=4096)**")
        full_ids = full_tokenizer.encode(test_text)
        st.metric("Tokens", len(full_ids))
        special_set = set(full_tokenizer.special_tokens)
        tokens_data = [
            (
                tid,
                token_id_to_display(full_tokenizer, tid),
                isinstance(full_tokenizer.vocab.get(tid), str)
                and full_tokenizer.vocab[tid] in special_set,
            )
            for tid in full_ids
        ]
        html = render_colored_tokens(tokens_data)
        st.markdown(html, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Edge Case Showcase
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _render_edge_cases(tokenizer):
    st.markdown("#### Edge Case Showcase")
    st.caption(
        "See how the byte-level BPE tokenizer handles unusual inputs. "
        "Because it starts from raw bytes, it can tokenize anything — no [UNK] tokens."
    )

    for case in EDGE_CASES:
        with st.expander(f"**{case['label']}**: `{case['text'][:50]}{'...' if len(case['text']) > 50 else ''}`"):
            st.code(case["text"], language=None)

            token_ids = tokenizer.encode(case["text"])
            special_set = set(tokenizer.special_tokens)
            tokens_data = [
                (
                    tid,
                    token_id_to_display(tokenizer, tid),
                    isinstance(tokenizer.vocab.get(tid), str)
                    and tokenizer.vocab[tid] in special_set,
                )
                for tid in token_ids
            ]

            html = render_colored_tokens(tokens_data)
            st.markdown(html, unsafe_allow_html=True)

            stats = compute_compression_ratio(tokenizer, case["text"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Tokens", stats["num_tokens"])
            c2.metric("Bytes / Token", stats["bytes_per_token"])
            c3.metric("Tokens / Word", stats["tokens_per_word"])

            st.info(case["note"])

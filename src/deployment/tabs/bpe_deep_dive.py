"""Tab 3: BPE Deep Dive — merge walkthrough, vocab explorer, comparison."""

import streamlit as st
import pandas as pd

from src.deployment.styles import (
    GLOBAL_CSS,
    render_colored_tokens,
    render_step_tokens,
    token_id_to_color,
)
from src.deployment.tokenizer_utils import (
    encode_with_steps,
    token_id_to_display,
    char_level_tokenize,
    word_level_tokenize,
    compute_compression_ratio,
)


def render_deep_dive_tab(tokenizer):
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    st.subheader("BPE Deep Dive")

    section = st.radio(
        "Section",
        ["Merge Walkthrough", "Vocab & Merges Explorer", "Tokenization Comparison"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if section == "Merge Walkthrough":
        _render_merge_walkthrough(tokenizer)
    elif section == "Vocab & Merges Explorer":
        _render_vocab_explorer(tokenizer)
    else:
        _render_comparison(tokenizer)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Merge Walkthrough
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _render_merge_walkthrough(tokenizer):
    st.markdown("#### Step-by-Step Merge Walkthrough")
    st.caption(
        "See exactly how BPE merges byte-level tokens into larger units. "
        "Each step shows the pair being merged and its rank from training."
    )

    text = st.text_input(
        "Enter a word or short phrase:",
        value="unhappiness",
        key="merge_walk_input",
    )

    if not text:
        return

    # Pre-tokenize to get individual words
    words = tokenizer._pre_tokenize(text)
    special_set = set(tokenizer.special_tokens)

    # Filter to non-special words (special tokens have no merge steps)
    non_special_words = [w for w in words if w not in special_set]
    if not non_special_words:
        st.info("Only special tokens found — no merge steps to show.")
        return

    # Word selector if multiple words
    if len(non_special_words) > 1:
        word_labels = [repr(w) for w in non_special_words]
        selected_idx = st.selectbox(
            "Select word to visualize:",
            range(len(non_special_words)),
            format_func=lambda i: word_labels[i],
        )
        selected_word = non_special_words[selected_idx]
    else:
        selected_word = non_special_words[0]

    # Compute merge steps
    steps = encode_with_steps(tokenizer, selected_word)

    if len(steps) <= 1:
        st.info("No merges applied to this word — it's already at byte level.")
        _show_step(steps[0], is_active=True)
        return

    # Step slider
    st.markdown(
        f"**{len(steps) - 1} merge(s) applied** to reduce "
        f"from {len(steps[0]['token_ids'])} byte tokens "
        f"to {len(steps[-1]['token_ids'])} token(s)."
    )

    step_idx = st.slider(
        "Step",
        min_value=0,
        max_value=len(steps) - 1,
        value=len(steps) - 1,
        key="merge_step_slider",
    )

    # Show selected step
    current = steps[step_idx]
    _show_step(current, is_active=True)

    # Show merge info for non-initial steps
    if current["merged_pair"] is not None:
        st.markdown(
            f'<span class="merge-badge pair">'
            f'Merged: {current["merged_pair_text"]}</span> '
            f'<span class="merge-badge rank">'
            f'Rank: {current["merge_rank"]}</span> '
            f'→ New token: **{current["new_token_text"]}** '
            f'(ID {current["new_token_id"]})',
            unsafe_allow_html=True,
        )

    # Expandable full step table
    with st.expander("Show all steps as table"):
        rows = []
        for s in steps:
            rows.append(
                {
                    "Step": s["step"],
                    "Merged Pair": s["merged_pair_text"] or "(initial)",
                    "Rank": s["merge_rank"] if s["merge_rank"] is not None else "-",
                    "New Token": s["new_token_text"] or "-",
                    "New ID": s["new_token_id"] if s["new_token_id"] is not None else "-",
                    "Sequence Length": len(s["token_ids"]),
                    "Tokens": " | ".join(s["token_texts"]),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _show_step(step: dict, is_active: bool = False):
    """Render a single merge step as colored token chips."""
    html = render_step_tokens(
        step["token_ids"],
        step["token_texts"],
        merged_token_id=step.get("new_token_id"),
        is_active=is_active,
    )
    st.markdown(html, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Vocab & Merges Explorer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _render_vocab_explorer(tokenizer):
    st.markdown("#### Vocab & Merges Explorer")
    st.caption(
        f"Browse all {len(tokenizer.merges):,} learned merge rules. "
        "Each merge combines two existing tokens into a new one."
    )

    # Build merge table
    search = st.text_input(
        "Search merges (filter by result token text):",
        key="merge_search",
    )

    rows = []
    for rank, (id_a, id_b) in enumerate(tokenizer.merges):
        text_a = token_id_to_display(tokenizer, id_a)
        text_b = token_id_to_display(tokenizer, id_b)
        result_id = rank + 256 + len(tokenizer.special_tokens)
        result_text = token_id_to_display(tokenizer, result_id)

        if search and search.lower() not in result_text.lower():
            continue

        rows.append(
            {
                "Rank": rank,
                "Token A": repr(text_a),
                "ID A": id_a,
                "Token B": repr(text_b),
                "ID B": id_b,
                "Result": repr(result_text),
                "Result ID": result_id,
            }
        )

    if not rows:
        st.warning("No merges match your search.")
        return

    st.markdown(f"Showing **{len(rows):,}** merge(s)")
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        height=500,
        hide_index=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tokenization Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _render_comparison(tokenizer):
    st.markdown("#### Side-by-Side Comparison")
    st.caption("Compare BPE tokenization against character-level and word-level baselines.")

    text = st.text_area(
        "Input text:",
        value="Once upon a time, there was a little girl named Lily.",
        height=80,
        key="comparison_input",
    )

    if not text:
        return

    # BPE
    bpe_ids = tokenizer.encode(text)
    bpe_tokens = [
        (tid, token_id_to_display(tokenizer, tid)) for tid in bpe_ids
    ]

    # Character-level
    char_tokens = char_level_tokenize(text)

    # Word-level
    word_tokens = word_level_tokenize(text)

    # Display in columns
    col_char, col_bpe, col_word = st.columns(3)

    with col_char:
        st.markdown("**Character-level**")
        st.metric("Tokens", len(char_tokens))
        tokens_data = [(i, ch, False) for i, ch in enumerate(char_tokens)]
        html = render_colored_tokens(tokens_data)
        st.markdown(html, unsafe_allow_html=True)

    with col_bpe:
        st.markdown("**BPE (ours, vocab=4096)**")
        st.metric("Tokens", len(bpe_ids))
        special_set = set(tokenizer.special_tokens)
        tokens_data = [
            (
                tid,
                token_id_to_display(tokenizer, tid),
                isinstance(tokenizer.vocab.get(tid), str)
                and tokenizer.vocab[tid] in special_set,
            )
            for tid in bpe_ids
        ]
        html = render_colored_tokens(tokens_data)
        st.markdown(html, unsafe_allow_html=True)

    with col_word:
        st.markdown("**Word-level**")
        st.metric("Tokens", len(word_tokens))
        tokens_data = [(i, w, False) for i, w in enumerate(word_tokens)]
        html = render_colored_tokens(tokens_data)
        st.markdown(html, unsafe_allow_html=True)

    # Summary comparison
    st.divider()
    st.markdown("**Summary**")
    num_bytes = len(text.encode("utf-8"))
    summary = pd.DataFrame(
        {
            "Method": ["Character-level", "BPE (ours)", "Word-level"],
            "Tokens": [len(char_tokens), len(bpe_ids), len(word_tokens)],
            "Bytes / Token": [
                round(num_bytes / max(len(char_tokens), 1), 2),
                round(num_bytes / max(len(bpe_ids), 1), 2),
                round(num_bytes / max(len(word_tokens), 1), 2),
            ],
        }
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

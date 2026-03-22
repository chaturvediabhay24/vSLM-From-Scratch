"""Tab 5: Stats Dashboard — vocabulary statistics and charts."""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from src.deployment.styles import GLOBAL_CSS
from src.deployment.tokenizer_utils import (
    compute_stats,
    compute_compression_ratio,
)

SAMPLE_TEXT = (
    "Once upon a time, there was a little girl named Lily. She loved to play "
    "outside in the sunshine. One day, she found a big, red ball in the park. "
    '"Look at this ball!" she said to her mom. Her mom smiled and said, '
    '"That is a very nice ball, Lily. You can play with it." Lily was so happy. '
    "She kicked the ball and ran after it. The ball went far, far away. "
    '"Come back, ball!" Lily shouted. But the ball kept rolling. '
    "A friendly dog saw the ball and brought it back to Lily. "
    '"Thank you, doggy!" she said. They played together until the sun went down.'
)


def render_stats_tab(tokenizer):
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    st.subheader("Stats Dashboard")

    stats = compute_stats(tokenizer)

    # ── Top metrics row ─────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Vocab Size", f"{stats['vocab_size']:,}")
    c2.metric("Total Merges", f"{stats['num_merges']:,}")
    c3.metric("Byte Tokens", stats["num_byte_tokens"])
    c4.metric("Learned Tokens", f"{stats['num_learned_tokens']:,}")
    c5.metric("Special Tokens", stats["num_special_tokens"])

    st.divider()

    # ── Charts row ──────────────────────────────────────────────────
    col_chart, col_pie = st.columns(2)

    with col_chart:
        st.markdown("#### Token Length Distribution")
        st.caption("Number of vocabulary entries by their byte length")

        lengths = stats["token_lengths"]
        max_len = max(lengths.keys())
        df = pd.DataFrame(
            {
                "Byte Length": range(1, max_len + 1),
                "Count": [lengths.get(i, 0) for i in range(1, max_len + 1)],
            }
        )
        df = df.set_index("Byte Length")
        st.bar_chart(df, color="#667eea")

    with col_pie:
        st.markdown("#### Token Type Breakdown")
        st.caption("Distribution of token types in the vocabulary")

        fig, ax = plt.subplots(figsize=(5, 5))
        sizes = [
            stats["num_byte_tokens"],
            stats["num_special_tokens"],
            stats["num_learned_tokens"],
        ]
        labels = [
            f"Byte ({stats['num_byte_tokens']})",
            f"Special ({stats['num_special_tokens']})",
            f"Learned ({stats['num_learned_tokens']:,})",
        ]
        colors = ["#4FC3F7", "#FF8A65", "#81C784"]
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 11},
        )
        for autotext in autotexts:
            autotext.set_fontweight("bold")
        ax.set_aspect("equal")
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # ── Compression ratio calculator ────────────────────────────────
    st.markdown("#### Compression Ratio Calculator")
    st.caption("Enter text to see how efficiently the tokenizer compresses it")

    text = st.text_area(
        "Input text for compression analysis",
        value=SAMPLE_TEXT,
        height=120,
        key="stats_text_input",
    )

    if text:
        cr = compute_compression_ratio(tokenizer, text)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tokens", cr["num_tokens"])
        c2.metric("Bytes", cr["num_bytes"])
        c3.metric(
            "Bytes / Token",
            cr["bytes_per_token"],
            delta=f"{cr['bytes_per_token'] - 1:.2f} vs char-level",
            delta_color="normal",
        )
        c4.metric("Tokens / Word", cr["tokens_per_word"])

        # Comparison bar
        st.markdown("**Compression comparison**")
        char_tokens = cr["num_chars"]
        bpe_tokens = cr["num_tokens"]
        word_tokens = cr["num_words"]

        comp_df = pd.DataFrame(
            {
                "Method": ["Character-level", "BPE (ours)", "Word-level"],
                "Tokens": [char_tokens, bpe_tokens, word_tokens],
            }
        )
        comp_df = comp_df.set_index("Method")
        st.bar_chart(comp_df, color="#764ba2")

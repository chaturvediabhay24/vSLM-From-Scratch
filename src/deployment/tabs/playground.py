"""Tab 2: Tokenizer Playground — live tokenization with colored tokens."""

import streamlit as st

from src.deployment.styles import GLOBAL_CSS, render_colored_tokens
from src.deployment.tokenizer_utils import (
    compute_compression_ratio,
    token_id_to_display,
)

EXAMPLE_INPUTS = [
    "Once upon a time, there was a little girl named Lily.",
    "The cat sat on the mat.",
    'She said, "I love you!"',
    "Hello<EOS>World",
    "supercalifragilisticexpialidocious",
    "I love cats! :) \U0001f431",
]


def render_playground_tab(tokenizer):
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    st.subheader("Tokenizer Playground")
    st.caption("Type or paste text below to see how the BPE tokenizer breaks it into tokens.")

    # ── Example buttons ─────────────────────────────────────────────
    st.markdown("**Try these inputs:**")
    cols = st.columns(len(EXAMPLE_INPUTS))
    for i, (col, example) in enumerate(zip(cols, EXAMPLE_INPUTS)):
        with col:
            label = example[:20] + "..." if len(example) > 20 else example
            if st.button(label, key=f"example_{i}", use_container_width=True):
                st.session_state["playground_text"] = example

    # ── Text input ──────────────────────────────────────────────────
    default = st.session_state.get(
        "playground_text",
        "Once upon a time, there was a little girl named Lily.",
    )
    text = st.text_area(
        "Input text",
        value=default,
        height=120,
        key="playground_input",
        label_visibility="collapsed",
    )

    if not text:
        st.info("Enter some text above to see tokenization results.")
        return

    # ── Encode ──────────────────────────────────────────────────────
    token_ids = tokenizer.encode(text)
    special_set = set(tokenizer.special_tokens)

    tokens_display = []
    for tid in token_ids:
        display = token_id_to_display(tokenizer, tid)
        is_special = (
            isinstance(tokenizer.vocab.get(tid), str)
            and tokenizer.vocab[tid] in special_set
        )
        tokens_display.append((tid, display, is_special))

    # ── Colored token visualization ─────────────────────────────────
    st.markdown("#### Tokenized Output")
    st.caption("Hover over tokens to see their ID. Special tokens are highlighted with a red border.")
    html = render_colored_tokens(tokens_display)
    st.markdown(html, unsafe_allow_html=True)

    # ── Live stats ──────────────────────────────────────────────────
    st.markdown("#### Stats")
    stats = compute_compression_ratio(tokenizer, text)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tokens", stats["num_tokens"])
    c2.metric("Characters", stats["num_chars"])
    c3.metric("Bytes (UTF-8)", stats["num_bytes"])
    c4.metric("Bytes / Token", stats["bytes_per_token"])
    c5.metric("Tokens / Word", stats["tokens_per_word"])

    # ── Encode-Decode roundtrip ─────────────────────────────────────
    st.markdown("#### Encode / Decode Roundtrip")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Token IDs**")
        # Show IDs in a scrollable code block
        ids_str = str(token_ids)
        st.code(ids_str, language=None)

    with col_right:
        decoded = tokenizer.decode(token_ids)
        st.markdown("**Decoded Text**")
        st.code(decoded, language=None)

    roundtrip_ok = text == decoded
    if roundtrip_ok:
        st.success("Roundtrip check: **PASSED** — decoded text matches original exactly.")
    else:
        st.error("Roundtrip check: **FAILED** — decoded text differs from original.")
        with st.expander("Show diff"):
            st.text(f"Original:  {repr(text)}")
            st.text(f"Decoded:   {repr(decoded)}")

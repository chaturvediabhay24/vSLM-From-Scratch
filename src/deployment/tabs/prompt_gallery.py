"""Prompt Gallery tab — curated prompts, failure showcase, vocabulary analysis."""

import time

import streamlit as st

from src.deployment.styles import GLOBAL_CSS, _escape_html
from src.deployment.model_utils import generate_full


# ── Curated prompts that showcase the model at its best ──────────────
CURATED_PROMPTS = [
    {
        "category": "Story Starter",
        "prompt": "Once upon a time, there was a little",
        "description": "Classic fairy tale opening — the model's bread and butter",
        "temperature": 0.7,
        "top_k": 40,
    },
    {
        "category": "Cause & Effect",
        "prompt": "The dog was very happy because",
        "description": "Simple cause-effect pattern common in children's stories",
        "temperature": 0.7,
        "top_k": 40,
    },
    {
        "category": "Action Sequence",
        "prompt": "Lily went to the park and saw a",
        "description": "Character + location + action — triggers narrative continuation",
        "temperature": 0.8,
        "top_k": 40,
    },
    {
        "category": "Discovery",
        "prompt": "One day, a boy named Tom found a",
        "description": "Discovery pattern — common TinyStories structure",
        "temperature": 0.8,
        "top_k": 40,
    },
    {
        "category": "Emotion",
        "prompt": "She was sad because her",
        "description": "Emotional setup — model has learned emotion-to-event mappings",
        "temperature": 0.7,
        "top_k": 40,
    },
    {
        "category": "Dialogue",
        "prompt": '"Can I play with you?" asked the little',
        "description": "Dialogue pattern — tests if the model can continue conversations",
        "temperature": 0.8,
        "top_k": 40,
    },
    {
        "category": "Scene Setting",
        "prompt": "The sun was shining and the birds were",
        "description": "Descriptive opening — nature scenes are common in the training data",
        "temperature": 0.7,
        "top_k": 40,
    },
    {
        "category": "Lesson/Moral",
        "prompt": "Mom said, \"You should always be kind because",
        "description": "Moral lessons are a core theme of TinyStories",
        "temperature": 0.7,
        "top_k": 40,
    },
]

# ── Prompts that expose model weaknesses ─────────────────────────────
FAILURE_PROMPTS = [
    {
        "category": "Math / Logic",
        "prompt": "What is 2 + 2? The answer is",
        "explanation": (
            "The model was never trained on math. It generates text that *looks* like an answer "
            "but has no concept of arithmetic. It's a pattern matcher, not a calculator."
        ),
    },
    {
        "category": "Factual Knowledge",
        "prompt": "The capital of France is",
        "explanation": (
            "The model has no world knowledge — only TinyStories patterns. It may generate "
            "a plausible-sounding word but it's not grounded in facts."
        ),
    },
    {
        "category": "Code Generation",
        "prompt": "def fibonacci(n):\n    if n <= 1:\n        return",
        "explanation": (
            "The training data contains zero code. The model will try to continue this as "
            "a story or produce garbled output. It has no concept of programming syntax."
        ),
    },
    {
        "category": "Non-English",
        "prompt": "Il \u00e9tait une fois un petit gar\u00e7on qui",
        "explanation": (
            "TinyStories is English-only. The byte-level tokenizer CAN encode any language "
            "(no [UNK] tokens), but the model never learned French patterns. Output will "
            "drift to English or produce nonsense."
        ),
    },
    {
        "category": "Long Context",
        "prompt": (
            "There was a girl named Anna. She had a cat named Whiskers. One day Anna and "
            "Whiskers went to the beach. They built a big sandcastle together. Then they "
            "went swimming in the ocean. After swimming they ate lunch. Anna had a sandwich "
            "and Whiskers had some fish. Then they played in the sand some more. Anna found "
            "a beautiful shell. She showed it to Whiskers. The cat's name was"
        ),
        "explanation": (
            f"This prompt is long enough to approach the 128-token context window. "
            f"By the time the model needs to recall the cat's name ('Whiskers'), "
            f"that information may have been pushed out of the context window."
        ),
    },
]


def _render_output(prompt: str, generated: str):
    """Render prompt (white) + generated text (green)."""
    prompt_html = _escape_html(prompt)
    gen_html = _escape_html(generated)
    st.markdown(
        f'<div class="generation-output">'
        f'<span class="prompt-text">{prompt_html}</span>'
        f'<span class="generated-text">{gen_html}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_prompt_gallery_tab(model, tokenizer, config, device):
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    section = st.radio(
        "Section",
        ["Curated Gallery", "Failure Showcase", "Vocabulary & Domain"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    if section == "Curated Gallery":
        _render_curated_gallery(model, tokenizer, config, device)
    elif section == "Failure Showcase":
        _render_failure_showcase(model, tokenizer, config, device)
    else:
        _render_vocabulary_domain(tokenizer)


# ── Curated Gallery ──────────────────────────────────────────────────

def _render_curated_gallery(model, tokenizer, config, device):
    st.subheader("Curated Prompt Gallery")
    st.caption(
        "Handpicked prompts that showcase the model at its best. "
        "These are the kinds of patterns TinyStories taught it."
    )

    for i, entry in enumerate(CURATED_PROMPTS):
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{entry['category']}**")
                st.caption(entry["description"])
                st.code(entry["prompt"], language=None)

            with col2:
                st.caption(f"temp={entry['temperature']}, top_k={entry['top_k']}")
                generate_btn = st.button(
                    "Generate", key=f"curated_{i}", use_container_width=True,
                )

            if generate_btn:
                seed = int(time.time() * 1000) % (2**31)
                with st.spinner("Generating..."):
                    full_text, _ = generate_full(
                        model, tokenizer, config, device,
                        prompt=entry["prompt"],
                        max_new_tokens=150,
                        temperature=entry["temperature"],
                        top_k=entry["top_k"],
                        repetition_penalty=1.2,
                        seed=seed,
                    )
                generated = full_text[len(entry["prompt"]):]
                _render_output(entry["prompt"], generated)


# ── Failure Showcase ─────────────────────────────────────────────────

def _render_failure_showcase(model, tokenizer, config, device):
    st.subheader("Failure Showcase")
    st.caption(
        "Prompts where the model struggles or fails entirely. "
        "Showing these demonstrates understanding of what the model actually is."
    )

    st.info(
        "These failures are **expected and by design**. A 5M parameter model trained "
        "only on children's stories cannot do math, speak French, or write code. "
        "That's not a bug — it's a limitation of the training data and model size."
    )

    for i, entry in enumerate(FAILURE_PROMPTS):
        with st.expander(f"{entry['category']}", expanded=(i == 0)):
            st.code(entry["prompt"], language=None)
            st.markdown(f"**Why it fails:** {entry['explanation']}")

            if st.button("Try it anyway", key=f"failure_{i}"):
                seed = int(time.time() * 1000) % (2**31)
                with st.spinner("Generating..."):
                    full_text, _ = generate_full(
                        model, tokenizer, config, device,
                        prompt=entry["prompt"],
                        max_new_tokens=100,
                        temperature=0.8,
                        top_k=40,
                        repetition_penalty=1.2,
                        seed=seed,
                    )
                generated = full_text[len(entry["prompt"]):]
                _render_output(entry["prompt"], generated)


# ── Vocabulary & Domain ──────────────────────────────────────────────

def _render_vocabulary_domain(tokenizer):
    st.subheader("Vocabulary & Domain Analysis")
    st.caption(
        "The tokenizer's learned vocabulary reflects the TinyStories training data. "
        "The most common subwords reveal what patterns the model learned to represent."
    )

    # Show top-N learned tokens (by merge rank = frequency in training data)
    st.markdown("**Top 50 Learned Tokens** (by merge priority — earlier = more frequent)")
    st.markdown(
        "*These are the first 50 byte-pair merges the tokenizer learned. "
        "Lower rank = more frequent in the training corpus.*"
    )

    # Build table of top merges
    rows = []
    num_special = len(tokenizer.special_tokens)
    for rank, (id_a, id_b) in enumerate(tokenizer.merges[:50]):
        result_id = 256 + num_special + rank
        # Decode token texts
        try:
            text_a = tokenizer.vocab[id_a]
            if isinstance(text_a, bytes):
                text_a = text_a.decode("utf-8", errors="replace")
        except (KeyError, Exception):
            text_a = f"[{id_a}]"

        try:
            text_b = tokenizer.vocab[id_b]
            if isinstance(text_b, bytes):
                text_b = text_b.decode("utf-8", errors="replace")
        except (KeyError, Exception):
            text_b = f"[{id_b}]"

        try:
            result_text = tokenizer.vocab[result_id]
            if isinstance(result_text, bytes):
                result_text = result_text.decode("utf-8", errors="replace")
        except (KeyError, Exception):
            result_text = f"[{result_id}]"

        rows.append({
            "Rank": rank + 1,
            "Pair": f"'{text_a}' + '{text_b}'",
            "Result Token": f"'{result_text}'",
            "Token ID": result_id,
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height=500)

    st.divider()

    # Domain observation
    st.markdown("**What the vocabulary tells us:**")
    st.markdown(
        """
        - The earliest merges are common English patterns: spaces + letters,
          common two-letter combinations like "th", "he", "in", "er"
        - Story-specific tokens appear quickly: "the", "and", "was", "she", "her"
        - The vocabulary is heavily biased toward simple English — exactly what TinyStories contains
        - Multi-byte/Unicode merges appear late (high rank), confirming the training data is ASCII-dominant
        """
    )

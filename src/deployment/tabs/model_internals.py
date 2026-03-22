"""Model Internals tab — attention visualization, token probabilities, perplexity."""

import time

import matplotlib.pyplot as plt
import matplotlib
import streamlit as st

from src.deployment.styles import GLOBAL_CSS, _escape_html
from src.deployment.model_utils import (
    extract_attention_weights,
    generate_full,
    compute_perplexity,
)

matplotlib.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "text.color": "#cdd6f4",
    "axes.labelcolor": "#cdd6f4",
    "xtick.color": "#a6adc8",
    "ytick.color": "#a6adc8",
})


def _decode_token(tokenizer, token_id):
    """Safely decode a single token ID to display text."""
    try:
        text = tokenizer.vocab[token_id]
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        # Make whitespace visible
        text = text.replace(" ", "\u00b7").replace("\n", "\\n").replace("\t", "\\t")
        if not text:
            text = "\u2205"
        return text
    except (KeyError, Exception):
        return f"[{token_id}]"


def render_model_internals_tab(model, tokenizer, config, device):
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    section = st.radio(
        "Section",
        ["Attention Visualization", "Token Probabilities", "Perplexity Scorer"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    if section == "Attention Visualization":
        _render_attention_viz(model, tokenizer, config, device)
    elif section == "Token Probabilities":
        _render_token_probabilities(model, tokenizer, config, device)
    else:
        _render_perplexity(model, tokenizer, config, device)


# ── Attention Visualization ──────────────────────────────────────────

def _render_attention_viz(model, tokenizer, config, device):
    st.subheader("Attention Visualization")
    st.caption(
        "See which tokens attend to which other tokens in each layer and head. "
        "Brighter = stronger attention. Each row shows where that token 'looks' "
        "to gather information."
    )

    text = st.text_input(
        "Enter short text (keep under ~40 tokens for readability):",
        value="Once upon a time there was a little girl",
        key="attn_text",
    )

    if not text.strip():
        return

    token_ids = tokenizer.encode(text)

    if len(token_ids) > 60:
        st.warning(
            f"Input has {len(token_ids)} tokens. Attention heatmap will be hard to read. "
            "Consider using shorter text."
        )

    # Layer / head selectors
    c1, c2 = st.columns(2)
    with c1:
        layer = st.selectbox(
            "Layer",
            list(range(config.n_layers)),
            format_func=lambda x: f"Layer {x}",
            key="attn_layer",
        )
    with c2:
        head_options = list(range(config.n_heads)) + ["Average"]
        head = st.selectbox(
            "Head",
            head_options,
            format_func=lambda x: f"Head {x}" if isinstance(x, int) else x,
            key="attn_head",
        )

    if st.button("Visualize Attention", type="primary"):
        with st.spinner("Computing attention weights..."):
            all_weights = extract_attention_weights(model, token_ids, device)

        # Get the attention matrix for selected layer
        layer_weights = all_weights[layer]  # (n_heads, T, T)

        if head == "Average":
            attn_matrix = layer_weights.mean(dim=0).numpy()
            title = f"Layer {layer} — Average across all {config.n_heads} heads"
        else:
            attn_matrix = layer_weights[head].numpy()
            title = f"Layer {layer}, Head {head}"

        # Token labels
        labels = [_decode_token(tokenizer, tid) for tid in token_ids]

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), max(6, len(labels) * 0.4)))
        im = ax.imshow(attn_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=attn_matrix.max())

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=max(6, 12 - len(labels) // 5))
        ax.set_yticklabels(labels, fontsize=max(6, 12 - len(labels) // 5))

        ax.set_xlabel("Key (attended to)", fontsize=11)
        ax.set_ylabel("Query (attending from)", fontsize=11)
        ax.set_title(title, fontsize=13, pad=12)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Explanation
        st.caption(
            "**How to read this:** Each row shows the attention distribution for one token. "
            "The causal mask means tokens can only attend to previous tokens (upper-right "
            "triangle is always zero). Brighter cells indicate stronger attention."
        )


# ── Token Probabilities ──────────────────────────────────────────────

def _render_token_probabilities(model, tokenizer, config, device):
    st.subheader("Token Probability Distribution")
    st.caption(
        "See the top candidate tokens at each generation step. This reveals "
        "how confident the model is and what alternatives it considered."
    )

    prompt = st.text_input(
        "Prompt:",
        value="Once upon a time",
        key="prob_prompt",
    )

    c1, c2 = st.columns(2)
    with c1:
        max_tokens = st.slider("Tokens to generate", 5, 50, 20, 1, key="prob_max")
    with c2:
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.05, key="prob_temp")

    if st.button("Generate with Probability Tracking", type="primary"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        seed = int(time.time() * 1000) % (2**31)
        with st.spinner("Generating..."):
            full_text, steps = generate_full(
                model, tokenizer, config, device,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=50,
                repetition_penalty=1.2,
                track_probabilities=True,
                seed=seed,
            )

        if not steps:
            st.warning("No tokens generated.")
            return

        # Store in session state for step navigation (use _data suffix to avoid
        # colliding with the widget key "prob_prompt")
        st.session_state["prob_steps_data"] = steps
        st.session_state["prob_prompt_data"] = prompt
        st.session_state["prob_full_text_data"] = full_text

    # Display step navigator if data exists
    if not all(k in st.session_state for k in ("prob_steps_data", "prob_prompt_data", "prob_full_text_data")):
        return

    steps = st.session_state["prob_steps_data"]
    prompt_text = st.session_state["prob_prompt_data"]
    full_text = st.session_state["prob_full_text_data"]

    # Show the generated text with current step highlighted
    generated = full_text[len(prompt_text):] if len(full_text) > len(prompt_text) else ""
    prompt_html = _escape_html(prompt_text)
    gen_html = _escape_html(generated)
    st.markdown(
        f'<div class="generation-output">'
        f'<span class="prompt-text">{prompt_html}</span>'
        f'<span class="generated-text">{gen_html}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Step slider
    step_idx = st.slider(
        "Generation step",
        0, len(steps) - 1, 0,
        format="Step %d",
        key="prob_step_slider",
    )

    step = steps[step_idx]
    chosen_id = step["token_id"]
    chosen_text = step["token_text"]

    st.markdown(
        f"**Step {step_idx + 1}:** Chose token **`{_escape_html(chosen_text)}`** "
        f"(ID: {chosen_id})"
    )

    # Probability bar chart
    if step["top_k_probs"]:
        probs = step["top_k_probs"][:10]  # Top 10

        # Build HTML bar chart
        max_prob = max(p for _, _, p in probs) if probs else 1.0
        bars_html = []
        for tid, text, prob in probs:
            is_chosen = tid == chosen_id
            bar_class = "prob-bar chosen" if is_chosen else "prob-bar"
            width_pct = (prob / max(max_prob, 1e-8)) * 100
            label_text = _escape_html(text.replace(" ", "\u00b7").replace("\n", "\\n"))
            if not label_text.strip():
                label_text = "\u2205"
            chosen_marker = " \u2190 chosen" if is_chosen else ""

            bars_html.append(
                f'<div class="prob-bar-container">'
                f'<span class="prob-label">{label_text}</span>'
                f'<div style="flex: 1; background: #313244; border-radius: 4px; overflow: hidden;">'
                f'<div class="{bar_class}" style="width: {width_pct:.1f}%;"></div>'
                f'</div>'
                f'<span class="prob-value">{prob:.1%}{chosen_marker}</span>'
                f'</div>'
            )

        st.markdown("".join(bars_html), unsafe_allow_html=True)
    else:
        st.info("Probability data not available for this step.")


# ── Perplexity Scorer ────────────────────────────────────────────────

def _render_perplexity(model, tokenizer, config, device):
    st.subheader("Perplexity Scorer")
    st.caption(
        "Perplexity measures how 'surprised' the model is by a piece of text. "
        "**Lower = more predictable** (the model has seen similar patterns). "
        "Higher = more surprising or out-of-domain."
    )

    PRESETS = {
        "TinyStories-like (low perplexity expected)": (
            "Once upon a time there was a little girl named Lily. "
            "She loved to play in the park with her friends."
        ),
        "Wikipedia-style (high perplexity expected)": (
            "The European Central Bank announced a 25 basis point rate hike "
            "amid persistent inflationary pressures across the eurozone."
        ),
        "Random/nonsense (very high perplexity)": (
            "Blorf quanx the zimble wock frazzle norp splint."
        ),
        "Custom": "",
    }

    preset = st.selectbox(
        "Choose a preset or enter custom text:",
        list(PRESETS.keys()),
        key="ppl_preset",
    )

    default_text = PRESETS[preset]
    text = st.text_area(
        "Text to score:",
        value=default_text,
        height=100,
        key="ppl_text",
    )

    if st.button("Compute Perplexity", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
            return

        with st.spinner("Computing perplexity..."):
            ppl = compute_perplexity(model, tokenizer, text, config, device)

        if ppl is None:
            st.warning("Text is too short to compute perplexity (need at least 2 tokens).")
            return

        # Color-code the result
        if ppl < 50:
            color = "#a6e3a1"  # green — familiar
            verdict = "The model finds this text very familiar (in-domain)"
        elif ppl < 200:
            color = "#f9e2af"  # yellow — somewhat familiar
            verdict = "Somewhat familiar — partially in-domain"
        else:
            color = "#f38ba8"  # red — out of domain
            verdict = "Very surprising — likely out of domain"

        st.markdown(
            f'<div style="text-align: center; padding: 20px;">'
            f'<h1 style="color: {color}; margin: 0;">{ppl:.1f}</h1>'
            f'<p style="color: #a6adc8; margin-top: 4px;">Perplexity</p>'
            f'<p style="color: {color}; font-size: 14px;">{verdict}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        token_ids = tokenizer.encode(text)
        st.caption(f"Scored on {len(token_ids)} tokens ({len(text)} characters)")

    # Comparison mode
    st.divider()
    st.markdown("**Compare multiple texts side-by-side:**")

    if st.button("Run all presets"):
        cols = st.columns(3)
        preset_items = list(PRESETS.items())[:3]  # Skip "Custom"

        for col, (name, preset_text) in zip(cols, preset_items):
            with col:
                st.markdown(f"**{name.split('(')[0].strip()}**")
                ppl = compute_perplexity(model, tokenizer, preset_text, config, device)
                if ppl is not None:
                    if ppl < 50:
                        color = "#a6e3a1"
                    elif ppl < 200:
                        color = "#f9e2af"
                    else:
                        color = "#f38ba8"
                    st.markdown(
                        f'<div style="text-align: center; padding: 16px; '
                        f'background: #1e1e2e; border-radius: 10px; border: 1px solid #313244;">'
                        f'<h2 style="color: {color}; margin: 0;">{ppl:.1f}</h2>'
                        f'<p style="color: #a6adc8; font-size: 12px;">perplexity</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f'"{preset_text[:60]}..."')

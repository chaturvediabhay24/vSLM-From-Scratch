"""Text Playground tab — core generation demo with streaming, retry, multi-temp."""

import time

import streamlit as st

from src.deployment.styles import GLOBAL_CSS, _escape_html
from src.deployment.model_utils import generate_streaming, generate_full


# ── Example prompts ──────────────────────────────────────────────────
EXAMPLE_PROMPTS = [
    "Once upon a time, there was a little",
    "The dog was very happy because",
    "Lily went to the park and saw a",
    "One day, a boy named Tom found a",
    "The sun was shining and the birds were",
    "She looked at the big red ball and",
]


def _render_generation_output(prompt: str, generated: str):
    """Render prompt (white) + generated text (green) with HTML styling."""
    prompt_html = _escape_html(prompt)
    gen_html = _escape_html(generated)
    html = (
        f'<div class="generation-output">'
        f'<span class="prompt-text">{prompt_html}</span>'
        f'<span class="generated-text">{gen_html}</span>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_text_playground_tab(model, tokenizer, config, device):
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    st.subheader("Text Completion Playground")
    st.caption(
        "Type a story prompt and watch the model complete it, one token at a time. "
        "Adjust generation parameters to see how they affect the output."
    )

    # ── Mode selector ────────────────────────────────────────────────
    mode = st.radio(
        "Mode",
        ["Single Completion", "Compare Temperatures"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    if mode == "Single Completion":
        _render_single_completion(model, tokenizer, config, device)
    else:
        _render_compare_temperatures(model, tokenizer, config, device)


# ── Single Completion ────────────────────────────────────────────────

def _render_single_completion(model, tokenizer, config, device):
    # Example buttons
    st.markdown("**Quick prompts:**")
    cols = st.columns(len(EXAMPLE_PROMPTS))
    for i, (col, example) in enumerate(zip(cols, EXAMPLE_PROMPTS)):
        if col.button(
            example[:25] + "..." if len(example) > 25 else example,
            key=f"example_{i}",
            use_container_width=True,
        ):
            st.session_state["model_prompt"] = example

    # Prompt input
    prompt = st.text_area(
        "Enter your story prompt:",
        value=st.session_state.get("model_prompt", EXAMPLE_PROMPTS[0]),
        height=100,
        key="model_prompt_input",
    )

    # Generation parameters
    st.markdown("**Generation Parameters**")
    p1, p2, p3 = st.columns(3)
    with p1:
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.05,
                                help="Higher = more random, Lower = more focused")
        top_k = st.slider("Top-k", 0, 200, 40, 5,
                          help="Sample from top-k tokens (0 = disabled)")
    with p2:
        top_p = st.slider("Top-p (nucleus)", 0.0, 1.0, 0.0, 0.05,
                          help="Sample from smallest set with cumulative prob >= p (0 = disabled)")
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.2, 0.05,
                                       help="Penalize repeated tokens (1.0 = off)")
    with p3:
        max_new_tokens = st.slider("Max New Tokens", 10, 500, 200, 10,
                                   help="Maximum tokens to generate")

    # Action buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
    generate_clicked = btn_col1.button("Generate", type="primary", use_container_width=True)
    retry_clicked = btn_col2.button("Retry", use_container_width=True,
                                    help="Re-generate with same prompt — different random sampling")

    if generate_clicked or retry_clicked:
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        # Store previous output for retry comparison
        if retry_clicked and "last_generation" in st.session_state:
            st.markdown("**Previous output:**")
            prev = st.session_state["last_generation"]
            _render_generation_output(prev["prompt"], prev["generated"])
            st.markdown("**New output:**")

        # Seed: use current time for randomness, different on each click
        seed = int(time.time() * 1000) % (2**31)

        # Streaming generation
        output_container = st.empty()
        generated_text = ""
        token_count = 0
        start_time = time.time()

        with st.spinner(""):
            for step in generate_streaming(
                model, tokenizer, config, device,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            ):
                full_text = step["full_text_so_far"]
                generated_text = full_text[len(prompt):] if len(full_text) > len(prompt) else ""
                token_count += 1

                # Update display progressively
                prompt_html = _escape_html(prompt)
                gen_html = _escape_html(generated_text)
                output_container.markdown(
                    f'<div class="generation-output">'
                    f'<span class="prompt-text">{prompt_html}</span>'
                    f'<span class="generated-text">{gen_html}</span>'
                    f'<span style="color: #585b70; font-size: 12px;">▌</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        elapsed = time.time() - start_time

        # Final output (no cursor)
        _render_generation_output(prompt, generated_text)

        # Stats
        s1, s2, s3 = st.columns(3)
        s1.metric("Tokens Generated", token_count)
        s2.metric("Time", f"{elapsed:.1f}s")
        s3.metric("Speed", f"{token_count / max(elapsed, 0.001):.1f} tokens/s")

        # Save for retry comparison
        st.session_state["last_generation"] = {
            "prompt": prompt,
            "generated": generated_text,
        }

    # ── Story continuation ───────────────────────────────────────────
    if "last_generation" in st.session_state:
        st.divider()
        if st.button("Continue from last output", help="Use the generated text as the new prompt"):
            prev = st.session_state["last_generation"]
            st.session_state["model_prompt"] = prev["prompt"] + prev["generated"]
            st.rerun()


# ── Compare Temperatures ─────────────────────────────────────────────

def _render_compare_temperatures(model, tokenizer, config, device):
    st.markdown(
        "Generate **3 completions** of the same prompt at different temperatures to see "
        "how randomness affects output."
    )

    prompt = st.text_area(
        "Prompt:",
        value=st.session_state.get("model_prompt", EXAMPLE_PROMPTS[0]),
        height=80,
        key="compare_prompt_input",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        temp_low = st.slider("Low temp", 0.1, 1.0, 0.3, 0.05, key="temp_low")
    with c2:
        temp_mid = st.slider("Mid temp", 0.3, 1.5, 0.8, 0.05, key="temp_mid")
    with c3:
        temp_high = st.slider("High temp", 0.5, 2.0, 1.5, 0.05, key="temp_high")

    p1, p2 = st.columns(2)
    with p1:
        max_tokens = st.slider("Max tokens", 10, 300, 150, 10, key="compare_max")
    with p2:
        top_k = st.slider("Top-k", 0, 200, 40, 5, key="compare_topk")

    if st.button("Generate All Three", type="primary"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        temps = [temp_low, temp_mid, temp_high]
        labels = [
            f"Temp = {temp_low} (focused)",
            f"Temp = {temp_mid} (balanced)",
            f"Temp = {temp_high} (creative)",
        ]

        cols = st.columns(3)
        for col, temp, label in zip(cols, temps, labels):
            with col:
                st.markdown(f"**{label}**")

                seed = int(time.time() * 1000) % (2**31)
                full_text, _ = generate_full(
                    model, tokenizer, config, device,
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_k=top_k,
                    repetition_penalty=1.2,
                    seed=seed,
                )

                generated = full_text[len(prompt):] if len(full_text) > len(prompt) else ""
                _render_generation_output(prompt, generated)

                tok_count = len(tokenizer.encode(generated))
                st.caption(f"{tok_count} tokens generated")

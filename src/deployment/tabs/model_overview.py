"""Model Overview tab — config card, training stats, honest limitations."""

import streamlit as st

from src.deployment.styles import GLOBAL_CSS


def render_model_overview_tab(model, tokenizer, config, device, training_step):
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 2.5em; margin-bottom: 0;">Trained Model</h1>
            <p style="font-size: 1.2em; color: #888; margin-top: 8px;">
                A ~5M parameter decoder-only transformer trained on children's stories
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Model config card ────────────────────────────────────────────
    st.subheader("Model Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Architecture**")
        arch_specs = {
            "Type": "Decoder-only Transformer (GPT-2 pre-norm)",
            "Embedding Dim (d_model)": config.d_model,
            "Attention Heads": config.n_heads,
            "Transformer Layers": config.n_layers,
            "Feed-Forward Dim": f"{config.d_ff:,}",
            "Context Length": f"{config.context_length} tokens",
            "Dropout": config.dropout_rate,
        }
        for k, v in arch_specs.items():
            st.markdown(f"- **{k}:** `{v}`")

    with col2:
        st.markdown("**Tokenizer & Vocab**")
        tok_specs = {
            "Tokenizer": "Byte-level BPE (from scratch)",
            "Vocab Size": f"{config.vocab_size:,}",
            "Base Tokens": "256 (one per byte)",
            "Special Tokens": f"{len(tokenizer.special_tokens)} (<EOS>)",
            "Learned Merges": f"{len(tokenizer.merges):,}",
            "Weight Tying": "Yes (input/output embeddings shared)",
        }
        for k, v in tok_specs.items():
            st.markdown(f"- **{k}:** `{v}`")

    with col3:
        st.markdown("**Training**")
        train_specs = {
            "Training Steps": f"{training_step:,}",
            "Dataset": "TinyStories (~2.1M stories)",
            "Total Tokens": "~424M (424 chunks)",
            "Optimizer": "AdamW (lr=3e-4, wd=0.1)",
            "LR Schedule": "Linear warmup + cosine decay",
            "Mixed Precision": "FP16 (AMP)",
        }
        for k, v in train_specs.items():
            st.markdown(f"- **{k}:** `{v}`")

    st.divider()

    # ── Key metrics ──────────────────────────────────────────────────
    param_count = model.count_parameters()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Parameters", f"{param_count:,}")
    m2.metric("Training Steps", f"{training_step:,}")
    m3.metric("Vocab Size", f"{config.vocab_size:,}")
    m4.metric("Context Window", f"{config.context_length} tokens")
    m5.metric("Checkpoint Size", "61 MB")

    st.divider()

    # ── What this model does ─────────────────────────────────────────
    st.subheader("What This Model Does")

    st.markdown(
        """
        This is an **autoregressive text completion model**. Given a text prompt, it predicts
        one token at a time to continue the text. It was trained on **TinyStories** — a dataset
        of short, simple children's stories.

        **The model learned to:**
        - Continue story beginnings in a coherent narrative style
        - Use simple vocabulary appropriate for children's stories
        - Follow basic narrative patterns (character introduction, action, resolution)
        - Generate grammatically mostly-correct English
        """
    )

    st.info(
        f"**This model was trained on ~424 million tokens of TinyStories data** "
        f"for {training_step:,} steps, resulting in {param_count:,} learned parameters."
    )

    st.divider()

    # ── Honest limitations ───────────────────────────────────────────
    st.subheader("Honest Limitations")

    with st.expander("This is NOT ChatGPT", expanded=True):
        st.markdown(
            """
            This model is fundamentally different from ChatGPT, Claude, or other modern AI assistants:

            | | This Model | ChatGPT/Claude |
            |---|---|---|
            | **Task** | Text completion only | Instruction following + conversation |
            | **Training** | Next-token prediction on stories | Pre-training + RLHF + instruction tuning |
            | **Parameters** | ~5 million | Billions (100B+) |
            | **Training Data** | Children's stories only | Massive internet corpus |
            | **Capability** | Continue a story fragment | Answer questions, write code, reason |

            **This model cannot:**
            - Answer questions or follow instructions
            - Do math, logic, or reasoning
            - Write code or produce structured output
            - Remember anything between sessions
            - Understand context beyond 128 tokens
            """
        )

    with st.expander("Context Window = 128 tokens"):
        st.markdown(
            f"""
            The model can only "see" the **last {config.context_length} tokens** of input at any time.
            For reference:
            - 128 tokens ≈ 50-80 words ≈ 3-5 short sentences
            - Anything before the context window is invisible to the model
            - This means long stories will lose coherence as earlier context drops out

            *Modern models use 8K-128K+ token context windows. Ours is intentionally small
            to keep training feasible on limited hardware.*
            """
        )

    with st.expander("What it's good at"):
        st.markdown(
            """
            - **Continuing short story openings** — "Once upon a time, there was a little..."
            - **Maintaining basic narrative flow** — characters doing things, simple cause-and-effect
            - **Children's story vocabulary** — simple words, names like Lily, Tom, Max
            - **Common story patterns** — park visits, playing with friends, learning lessons
            - **Grammatically reasonable output** — most sentences parse correctly
            """
        )

    with st.expander("Where it struggles"):
        st.markdown(
            """
            - **Long-range coherence** — forgets character names or plot points after ~50 tokens
            - **Factual accuracy** — makes up facts, may say contradictory things
            - **Complex plots** — can't handle multi-character interactions or subplots
            - **Non-story text** — code, math, questions, technical writing all produce nonsense
            - **Rare words** — vocabulary is limited to TinyStories domain
            - **Repetition** — sometimes gets stuck in loops (mitigated by repetition penalty)
            """
        )

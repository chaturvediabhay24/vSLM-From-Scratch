"""Tab 1: About / Landing Page — project overview and design decisions."""

import importlib.util
from pathlib import Path

import streamlit as st

# Import ModelConfig directly from the file to avoid triggering
# src/model/__init__.py which imports torch (may not be installed).
_config_path = Path(__file__).resolve().parent.parent.parent / "model" / "config.py"
_spec = importlib.util.spec_from_file_location("model_config", _config_path)
_config_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config_mod)
ModelConfig = _config_mod.ModelConfig


def render_about_tab():
    cfg = ModelConfig()

    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 2.5em; margin-bottom: 0;">LLM From Scratch</h1>
            <p style="font-size: 1.2em; color: #888; margin-top: 8px;">
                A decoder-only transformer and byte-level BPE tokenizer,
                built entirely from scratch in Python + PyTorch.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Architecture specs ──────────────────────────────────────────
    st.subheader("Model Architecture")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Transformer Config**")
        specs = {
            "Vocab Size": f"{cfg.vocab_size:,}",
            "Context Length": cfg.context_length,
            "Embedding Dim (d_model)": cfg.d_model,
            "Attention Heads": cfg.n_heads,
            "Transformer Layers": cfg.n_layers,
            "Feed-Forward Dim": f"{cfg.d_ff:,}",
            "Dropout": cfg.dropout_rate,
        }
        for k, v in specs.items():
            st.markdown(f"- **{k}:** `{v}`")

        # Approximate parameter count
        params = (
            cfg.vocab_size * cfg.d_model  # token embedding
            + cfg.context_length * cfg.d_model  # positional embedding
            + cfg.n_layers
            * (
                4 * cfg.d_model * cfg.d_model  # QKV + output projection
                + 2 * cfg.d_model  # 2 layernorms (weight + bias each)
                + 2 * cfg.d_model  # layernorm biases
                + cfg.d_model * cfg.d_ff  # MLP fc1
                + cfg.d_ff * cfg.d_model  # MLP fc2
                + cfg.d_ff + cfg.d_model  # MLP biases
            )
            + 2 * cfg.d_model  # final layernorm
            # output projection shares weights with token embedding
        )
        st.metric("Approximate Parameters", f"~{params / 1e6:.1f}M")

    with col2:
        st.markdown("**Pipeline**")
        st.markdown(
            """
            ```
            Raw Text (TinyStories CSV)
                    |
                    v
            1. Preprocessing
               Join stories with <EOS>
                    |
                    v
            2. Tokenization (BPE)
               Train byte-level BPE
               vocab_size = 4,096
                    |
                    v
            3. Corpus Encoding
               Parallel chunked encoding
               -> .npy token arrays
                    |
                    v
            4. Model Training
               Decoder-only transformer
               Next-token prediction
                    |
                    v
            5. Text Generation
               Autoregressive sampling
               Temperature + top-k
            ```
            """
        )

    with col3:
        st.markdown("**Key Features**")
        st.markdown(
            """
            - **Byte-level BPE** — no out-of-vocabulary tokens, handles any Unicode
            - **GPT-2 style pre-norm** — LayerNorm before attention and MLP
            - **Weight tying** — input and output embeddings share parameters
            - **FlashAttention** — uses PyTorch 2.0+ scaled_dot_product_attention
            - **Configurable context** — supports 64 to 512+ token windows
            - **From scratch** — no HuggingFace, no tiktoken, no external tokenizer libs
            """
        )

    st.divider()

    # ── Design decisions ────────────────────────────────────────────
    st.subheader("Design Decisions")

    with st.expander("Why vocab_size = 4,096?"):
        st.markdown(
            """
            The vocabulary size determines how many unique tokens the tokenizer learns.
            It's a tradeoff:

            - **Too small** (e.g., 256 = bytes only): No compression, every character is a token.
              A 100-word story takes ~500 tokens.
            - **Too large** (e.g., 50,000): Great compression but the embedding table
              (`vocab_size x d_model`) becomes huge. For our small model (d_model=256),
              50K vocab = 12.8M params in embeddings alone.
            - **4,096**: Sweet spot for the TinyStories corpus. Achieves ~3.96 bytes/token
              compression ratio. The embedding table is only 4,096 x 256 = ~1M params.
            """
        )

    with st.expander("Why byte-level BPE?"):
        st.markdown(
            """
            Byte-level BPE starts with a base vocabulary of 256 tokens (one per byte value),
            meaning **every possible byte sequence can be encoded**. Benefits:

            - **No `[UNK]` tokens** — emojis, accented characters, even binary data can be tokenized
            - **Language agnostic** — works on English, Chinese, Arabic, code, anything
            - **Simple base** — no need to define a character set, just 0-255
            - **Same approach as GPT-2/GPT-3** — proven to scale
            """
        )

    with st.expander("Why GPT-2 style pre-tokenization?"):
        st.markdown(
            """
            Before BPE merges, we split text using a regex pattern (same as GPT-2):
            ```
            's|'t|'re|'ve|'m|'ll|'d| ?\\w+| ?[^\\s\\w]+|\\s+(?!\\S)|\\s+
            ```

            This ensures BPE operates **within word boundaries**:
            - Contractions like "don't" split into "don" + "'t"
            - Words don't merge across spaces
            - Punctuation stays separate

            Without this, BPE might learn cross-word merges like "e t" (end of one word +
            start of next), which hurt generalization.
            """
        )

    with st.expander("Why a decoder-only transformer?"):
        st.markdown(
            """
            For **next-token prediction** (autoregressive language modeling), a decoder-only
            architecture is the natural fit:

            - **Causal masking** — each position can only attend to previous positions
            - **Same as GPT-2/3/4** — proven architecture for text generation
            - **Simpler than encoder-decoder** — one stack of blocks, no cross-attention
            - **Pre-norm (GPT-2 style)** — LayerNorm before attention/MLP (more stable training)
            """
        )

    st.divider()

    # ── How BPE works ───────────────────────────────────────────────
    st.subheader("How Byte-Pair Encoding Works")
    st.markdown(
        """
        BPE is a **compression algorithm** repurposed for tokenization. The core idea is simple:

        1. **Start** with a base vocabulary of 256 tokens (one per byte value) + special tokens
        2. **Count** every adjacent pair of tokens across the corpus
        3. **Find** the most frequent pair
        4. **Merge** it into a single new token, add to vocabulary
        5. **Replace** all occurrences of that pair in the corpus
        6. **Repeat** steps 2-5 until vocab reaches target size

        The ordered list of merges **IS** the tokenizer. When encoding new text:
        - Convert to bytes
        - Apply merges in the same priority order they were learned
        - Lower rank = learned first = merge first

        > Explore the **BPE Deep Dive** tab to see this process step-by-step on any text!
        """
    )

    st.divider()

    st.markdown(
        """
        <div style="text-align: center; padding: 20px; color: #888;">
            <p>Built as a learning project to understand transformer language models from first principles.</p>
            <p><a href="https://github.com/chaturvediabhay24/vSLM-From-Scratch" target="_blank">
                View on GitHub
            </a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

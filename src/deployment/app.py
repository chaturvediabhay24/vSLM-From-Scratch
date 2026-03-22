"""LLM From Scratch — Streamlit App

Two sections:
  - Tokenizer: existing BPE tokenizer showcase (5 tabs)
  - Model Demo: trained model demo (4 tabs)

Run from project root:
    streamlit run src/deployment/app.py
"""

import sys
from pathlib import Path

# Make src/ importable (same pattern as the project notebooks)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.tokenization.tokenizer import BPETokenizer

# Tokenizer tabs
from src.deployment.tabs.about import render_about_tab
from src.deployment.tabs.playground import render_playground_tab
from src.deployment.tabs.bpe_deep_dive import render_deep_dive_tab
from src.deployment.tabs.live_training import render_live_training_tab
from src.deployment.tabs.stats_dashboard import render_stats_tab

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM From Scratch",
    page_icon="\U0001f9e0",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_tokenizer():
    """Load the trained tokenizer once and cache across sessions."""
    path = PROJECT_ROOT / "dataset" / "tokenizer.json"
    return BPETokenizer.load(str(path))


# ── Section selector ────────────────────────────────────────────────
section = st.radio(
    "Section",
    ["\U0001f9e9 Tokenizer", "\U0001f916 Model Demo"],
    horizontal=True,
    label_visibility="collapsed",
)

# ── Tokenizer section (existing, unchanged) ─────────────────────────
if section == "\U0001f9e9 Tokenizer":
    tokenizer = load_tokenizer()

    tab_about, tab_playground, tab_deep_dive, tab_training, tab_stats = st.tabs(
        [
            "\U0001f3e0 About",
            "\U0001f3ae Playground",
            "\U0001f50d BPE Deep Dive",
            "\U0001f9ea Live Training",
            "\U0001f4ca Stats Dashboard",
        ]
    )

    with tab_about:
        render_about_tab()
    with tab_playground:
        render_playground_tab(tokenizer)
    with tab_deep_dive:
        render_deep_dive_tab(tokenizer)
    with tab_training:
        render_live_training_tab(tokenizer)
    with tab_stats:
        render_stats_tab(tokenizer)

# ── Model Demo section ──────────────────────────────────────────────
elif section == "\U0001f916 Model Demo":
    from src.deployment.model_utils import load_model_and_tokenizer
    from src.deployment.tabs.model_overview import render_model_overview_tab
    from src.deployment.tabs.text_playground import render_text_playground_tab
    from src.deployment.tabs.prompt_gallery import render_prompt_gallery_tab
    from src.deployment.tabs.model_internals import render_model_internals_tab

    with st.spinner("Loading model checkpoint (this may take a few seconds)..."):
        result = load_model_and_tokenizer()

    # Error handling — last element is a string on failure
    if isinstance(result[-1], str) and result[0] is None:
        st.error(result[-1])
        st.stop()

    model, tokenizer, config, device, training_step = result

    tab_overview, tab_playground, tab_gallery, tab_internals = st.tabs(
        [
            "\U0001f4cb Model Overview",
            "\u270d\ufe0f Text Playground",
            "\U0001f3a8 Prompt Gallery",
            "\U0001f52c Model Internals",
        ]
    )

    with tab_overview:
        render_model_overview_tab(model, tokenizer, config, device, training_step)
    with tab_playground:
        render_text_playground_tab(model, tokenizer, config, device)
    with tab_gallery:
        render_prompt_gallery_tab(model, tokenizer, config, device)
    with tab_internals:
        render_model_internals_tab(model, tokenizer, config, device)

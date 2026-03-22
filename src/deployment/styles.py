"""Shared CSS, token coloring, and HTML helpers for the deployment UI."""

GLOBAL_CSS = """
<style>
.token-container {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    padding: 12px;
    background: #1e1e2e;
    border-radius: 8px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 14px;
    line-height: 1.6;
}
.token-chip {
    display: inline-flex;
    align-items: center;
    padding: 4px 8px;
    border-radius: 4px;
    cursor: default;
    position: relative;
    transition: transform 0.1s;
    color: #1e1e2e;
    font-weight: 500;
}
.token-chip:hover {
    transform: scale(1.08);
    z-index: 10;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.token-chip.special {
    border: 2px solid #e74c3c;
    font-weight: 700;
}
.token-chip.special::after {
    content: "SPECIAL";
    position: absolute;
    top: -10px;
    right: -4px;
    font-size: 8px;
    background: #e74c3c;
    color: white;
    padding: 1px 4px;
    border-radius: 3px;
    font-weight: 700;
}
.token-chip.highlight {
    border: 2px solid #f39c12;
    box-shadow: 0 0 8px rgba(243, 156, 18, 0.5);
}
.step-row {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
    align-items: center;
    padding: 8px 12px;
    margin: 4px 0;
    background: #f8f9fa;
    border-radius: 6px;
    border-left: 3px solid #6c757d;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}
.step-row.active {
    background: #fff3cd;
    border-left-color: #f39c12;
}
.merge-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    margin-left: 8px;
}
.merge-badge.rank {
    background: #e3f2fd;
    color: #1565c0;
}
.merge-badge.pair {
    background: #fce4ec;
    color: #c62828;
}
.stat-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 20px;
    color: white;
    text-align: center;
}
.stat-card h3 {
    margin: 0;
    font-size: 14px;
    opacity: 0.9;
}
.stat-card .value {
    font-size: 32px;
    font-weight: 700;
    margin: 8px 0 0 0;
}

/* ── Model demo styles ─────────────────────────────────────────── */
.generated-text {
    color: #a6e3a1;
}
.prompt-text {
    color: #cdd6f4;
}
.generation-output {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 16px;
    line-height: 1.8;
    padding: 20px;
    background: #1e1e2e;
    border-radius: 10px;
    border: 1px solid #313244;
    min-height: 100px;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.model-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #313244 100%);
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #45475a;
}
.model-card h4 {
    color: #cdd6f4;
    margin: 0 0 12px 0;
}
.model-card table {
    width: 100%;
    border-collapse: collapse;
}
.model-card td {
    padding: 6px 12px;
    border-bottom: 1px solid #45475a;
    color: #cdd6f4;
    font-size: 14px;
}
.model-card td:first-child {
    color: #a6adc8;
    font-weight: 500;
}
.limitation-badge {
    display: inline-block;
    background: #f38ba8;
    color: #1e1e2e;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 12px;
    font-weight: 600;
}
.gallery-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}
.gallery-card:hover {
    border-color: #667eea;
}
.prob-bar-container {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}
.prob-bar {
    height: 22px;
    border-radius: 4px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    transition: width 0.3s;
}
.prob-bar.chosen {
    background: linear-gradient(90deg, #a6e3a1, #40a02b);
}
.prob-label {
    min-width: 120px;
    color: #cdd6f4;
    text-align: right;
}
.prob-value {
    min-width: 60px;
    color: #a6adc8;
}
</style>
"""


def token_id_to_color(token_id: int) -> str:
    """Deterministic pastel color from token ID using golden-angle hue spacing."""
    hue = (token_id * 137.508) % 360
    return f"hsl({hue:.0f}, 70%, 82%)"


def _escape_html(text: str) -> str:
    """Escape HTML special characters for safe rendering."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_colored_tokens(
    tokens: list[tuple[int, str, bool]],
    highlight_ids: set[int] | None = None,
    max_tokens: int = 500,
) -> str:
    """Generate HTML for colored token chips.

    Args:
        tokens: list of (token_id, display_text, is_special)
        highlight_ids: set of token IDs to highlight (for merge walkthrough)
        max_tokens: truncate display after this many tokens

    Returns:
        HTML string ready for st.markdown(..., unsafe_allow_html=True)
    """
    if highlight_ids is None:
        highlight_ids = set()

    truncated = len(tokens) > max_tokens
    display_tokens = tokens[:max_tokens]

    spans = []
    for token_id, display_text, is_special in display_tokens:
        color = token_id_to_color(token_id)
        escaped = _escape_html(display_text)

        # Show whitespace visually
        visual = escaped.replace(" ", "&middot;").replace("\n", "\\n").replace("\t", "\\t")
        if not visual:
            visual = "&empty;"

        classes = ["token-chip"]
        if is_special:
            classes.append("special")
        if token_id in highlight_ids:
            classes.append("highlight")

        tooltip = f"ID: {token_id} | Token: '{escaped}'"
        span = (
            f'<span class="{" ".join(classes)}" '
            f'style="background-color: {color};" '
            f'title="{tooltip}">{visual}</span>'
        )
        spans.append(span)

    html = f'<div class="token-container">{"".join(spans)}</div>'
    if truncated:
        html += f'<p style="color: #888; font-size: 12px; margin-top: 4px;">Showing {max_tokens} of {len(tokens)} tokens</p>'

    return html


def render_step_tokens(
    token_ids: list[int],
    display_texts: list[str],
    merged_token_id: int | None = None,
    is_active: bool = False,
) -> str:
    """Render a single merge step as colored token chips.

    Args:
        token_ids: list of token IDs in this step
        display_texts: corresponding display text for each token
        merged_token_id: the token ID that was just created (highlight it)
        is_active: whether this is the currently selected step
    """
    classes = ["step-row"]
    if is_active:
        classes.append("active")

    spans = []
    for tid, text in zip(token_ids, display_texts):
        color = token_id_to_color(tid)
        escaped = _escape_html(text)
        visual = escaped.replace(" ", "&middot;").replace("\n", "\\n")
        if not visual:
            visual = "&empty;"

        extra_class = "highlight" if tid == merged_token_id else ""
        tooltip = f"ID: {tid} | '{escaped}'"
        span = (
            f'<span class="token-chip {extra_class}" '
            f'style="background-color: {color};" '
            f'title="{tooltip}">{visual}</span>'
        )
        spans.append(span)

    return f'<div class="{" ".join(classes)}">{"".join(spans)}</div>'

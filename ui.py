import os

import gradio as gr
import pandas as pd
import requests

# Fix deprecated Starlette constant before Gradio uses it
import starlette.status

if hasattr(starlette.status, "HTTP_422_UNPROCESSABLE_CONTENT"):
    starlette.status.HTTP_422_UNPROCESSABLE_ENTITY = (
        starlette.status.HTTP_422_UNPROCESSABLE_CONTENT
    )

FLASK_BASE_URL = os.environ.get("FLASK_BASE_URL", "http://127.0.0.1:5000")
HTTP_TIMEOUT = 10.0  # seconds


def _build_css() -> str:
    return """
    :root {
      --bg: #15171C;
      --surface: #1E2127;
      --ink: #E8E6E1;
      --muted: #9AA0AC;
      --rule: #2A2D35;
      --accent: #6E8FFF;
      --verdict-ai: #E5707C;
      --verdict-uncertain: #D9A85C;
      --verdict-human: #5FAE7E;
    }

    body, .gradio-container {
      background: var(--bg) !important;
      color: var(--ink);
      font-family: -apple-system, 'Segoe UI', Roboto, Inter, sans-serif;
      font-size: 15px;
      line-height: 1.55;
    }

    .display {
      font-family: Georgia, 'Iowan Old Style', 'Times New Roman', serif;
      font-weight: 600;
    }

    .mono {
      font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    }

    /* Verdict card — the signature element */
    .verdict-card {
      background: var(--surface);
      border-top: 3px solid var(--rule);
      padding: 1rem 1.25rem;
      animation: verdict-draw 240ms ease-out both;
    }
    .verdict-card[data-verdict="ai"]        { border-top-color: var(--verdict-ai); }
    .verdict-card[data-verdict="uncertain"] { border-top-color: var(--verdict-uncertain); }
    .verdict-card[data-verdict="human"]     { border-top-color: var(--verdict-human); }

    .verdict-chip {
      font-family: Georgia, serif;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      animation: chip-fade 80ms ease-out 240ms both;
    }

    @keyframes verdict-draw {
      from { transform: scaleX(0); }
      to   { transform: scaleX(1); }
    }
    @keyframes chip-fade {
      from { opacity: 0; }
      to   { opacity: 1; }
    }

    @media (prefers-reduced-motion: reduce) {
      .verdict-card,
      .verdict-chip {
        animation: none;
      }
    }

    /* Focus ring — accessibility floor */
    :focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }

    /* Force accent color everywhere Gradio's default orange leaks through */
    input[type="radio"],
    input[type="checkbox"] {
      accent-color: var(--accent) !important;
      --checkbox-background-color-selected: var(--accent) !important;
      --checkbox-border-color-selected: var(--accent) !important;
      --checkbox-border-color-hover: var(--accent) !important;
    }
    input[type="radio"]:checked,
    input[type="checkbox"]:checked {
      background-color: var(--accent) !important;
      border-color: var(--accent) !important;
    }
    label:has(input[type="radio"]:checked) {
      --checkbox-label-border-color-selected: var(--accent) !important;
      --checkbox-label-text-color-selected: var(--accent) !important;
    }
    .gradio-container .primary,
    .gradio-container button.primary,
    button[variant="primary"] {
      background: var(--accent) !important;
      border-color: var(--accent) !important;
      color: var(--bg) !important;
    }
    .gradio-container .primary:hover,
    .gradio-container button.primary:hover {
      background: #5A7AE8 !important;
      border-color: #5A7AE8 !important;
    }
    label:has(input[type="radio"]:checked),
    .selected {
      border-color: var(--accent) !important;
      color: var(--accent) !important;
    }

    /* Tabs — numbered markers carry semantic meaning, styled as a stepper */
    .tab-label {
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .tab-label button {
      opacity: 0.55;
      transition: opacity 120ms ease-out;
    }
    .tab-label.selected button,
    .tab-label button[aria-selected="true"] {
      opacity: 1;
      color: var(--accent) !important;
      border-bottom: 2px solid var(--accent) !important;
    }

    /* Form cards — group each tab's fields visually */
    .form-card {
      background: var(--surface);
      border: 1px solid var(--rule);
      border-radius: 6px;
      padding: 1.5rem;
    }

    /* Verdict idle placeholder */
    .verdict-placeholder {
      font-family: Georgia, 'Iowan Old Style', serif;
      font-style: italic;
      color: var(--muted);
      margin: 0;
    }

    /* Example inputs table sits below the form card, not inside it */
    .examples-block {
      margin-top: 1rem;
    }

    /* Log table */
    .log-table table thead {
      font-family: ui-monospace, 'SF Mono', Consolas, monospace;
      border-bottom: 2px solid var(--accent);
    }
    .log-table table tbody tr:nth-child(even) {
      background: var(--surface);
    }
    .log-table table tbody tr:nth-child(odd) {
      background: var(--bg);
    }
    """


DARK_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#15171C",
    body_background_fill_dark="#15171C",
    body_text_color="#E8E6E1",
    body_text_color_dark="#E8E6E1",
    background_fill_primary="#15171C",
    background_fill_primary_dark="#15171C",
    background_fill_secondary="#1E2127",
    background_fill_secondary_dark="#1E2127",
    block_background_fill="#1E2127",
    block_background_fill_dark="#1E2127",
    block_border_color="#2A2D35",
    block_border_color_dark="#2A2D35",
    block_title_text_color="#E8E6E1",
    block_title_text_color_dark="#E8E6E1",
    block_label_text_color="#9AA0AC",
    block_label_text_color_dark="#9AA0AC",
    input_background_fill="#1E2127",
    input_background_fill_dark="#1E2127",
    input_border_color="#2A2D35",
    input_border_color_dark="#2A2D35",
    border_color_primary="#2A2D35",
    border_color_primary_dark="#2A2D35",
    button_primary_background_fill="#6E8FFF",
    button_primary_background_fill_dark="#6E8FFF",
    button_primary_background_fill_hover="#5A7AE8",
    button_primary_background_fill_hover_dark="#5A7AE8",
    button_primary_text_color="#15171C",
    button_primary_text_color_dark="#15171C",
    button_secondary_background_fill="#1E2127",
    button_secondary_background_fill_dark="#1E2127",
    button_secondary_text_color="#E8E6E1",
    button_secondary_text_color_dark="#E8E6E1",
    table_text_color="#E8E6E1",
    table_text_color_dark="#E8E6E1",
    checkbox_background_color="#1E2127",
    checkbox_background_color_dark="#1E2127",
    checkbox_background_color_selected="#6E8FFF",
    checkbox_background_color_selected_dark="#6E8FFF",
    checkbox_border_color="#2A2D35",
    checkbox_border_color_dark="#2A2D35",
    checkbox_border_color_selected="#6E8FFF",
    checkbox_border_color_selected_dark="#6E8FFF",
    checkbox_border_color_hover="#6E8FFF",
    checkbox_border_color_hover_dark="#6E8FFF",
    checkbox_label_background_fill_selected="#1E2127",
    checkbox_label_background_fill_selected_dark="#1E2127",
    checkbox_label_border_color_selected="#6E8FFF",
    checkbox_label_border_color_selected_dark="#6E8FFF",
    checkbox_label_text_color_selected="#6E8FFF",
    checkbox_label_text_color_selected_dark="#6E8FFF",
)


def _health_check(base_url: str) -> tuple[bool, str]:
    """Returns (ok, banner_text). Never raises."""
    try:
        r = requests.get(f"{base_url}/log", timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return True, ""
    except requests.exceptions.ConnectionError:
        return False, (
            f"✗ Flask unreachable at {base_url}. Start it with: python app.py"
        )
    except requests.exceptions.Timeout:
        return False, (
            f"✗ Flask at {base_url} did not respond within {HTTP_TIMEOUT}s. Is it hung?"
        )
    except Exception as e:
        return False, (
            f"✗ Flask at {base_url} returned an error: {type(e).__name__}: {e}"
        )


def _escape(text: str) -> str:
    import html

    return html.escape(str(text))


def _error_html(message: str, view_mode: str) -> str:
    return f"""
    <div class="verdict-card" data-verdict="uncertain" style="background: #FEE;">
      <p class="display" style="font-size: 22px; color: #8B0000; margin: 0 0 0.5rem 0;">
        ERROR
      </p>
      <p style="margin: 0; color: #333333; font-size: 15px; font-weight: 500;">
        {_escape(message)}
      </p>
    </div>
    """


def _render_verdict(body: dict, view_mode: str) -> str:
    """Render the verdict panel HTML. body is the /submit response JSON."""
    label = body.get("label") or {}
    category = label.get("category", "uncertain")
    text = label.get("text", "")
    band = label.get("band")

    # Map category to data-verdict attribute (SPEC-08).
    data_verdict = {
        "likely_ai": "ai",
        "uncertain": "uncertain",
        "likely_human": "human",
    }.get(category, "uncertain")

    chip_text = {
        "likely_ai": "AI",
        "uncertain": "UNCERTAIN",
        "likely_human": "HUMAN",
    }.get(category, "UNCERTAIN")

    chip_color_var = f"var(--verdict-{data_verdict})"

    creator_view = f"""
      <p class="display" style="font-size: 22px; margin: 0 0 0.5rem 0;">
        <span class="verdict-chip" style="color: {chip_color_var};">{chip_text}</span>
      </p>
      <p style="margin: 0; color: var(--muted); font-size: 15px;">
        {_escape(text)}
      </p>
    """

    auditor_view = (
        creator_view
        + """
      <hr style="border: 0; border-top: 1px solid var(--rule); margin: 1rem 0;" />
      <p class="mono" style="font-size: 13px; color: var(--muted); margin: 0;">
        <strong>content_id</strong>: {cid}<br/>
        <strong>attribution</strong>: {attr}<br/>
        <strong>confidence</strong>: {conf}<br/>
        <strong>band</strong>: {band}<br/>
        <strong>category</strong>: {cat}
      </p>
    """.format(
            cid=_escape(body.get("content_id", "—")),
            attr=_escape(body.get("attribution", "—")),
            conf=_escape(body.get("confidence", "—")),
            band=_escape(band or "—"),
            cat=_escape(category),
        )
    )

    body_html = creator_view if view_mode == "Creator view" else auditor_view

    return f"""
    <div class="verdict-card" data-verdict="{data_verdict}">
      {body_html}
    </div>
    """


def handle_submit(text, creator_id, view_mode, creator_id_state, content_id_state):
    """Returns (verdict_html, creator_id_state_value, content_id_state_value)."""
    if not text or not text.strip():
        return _error_html("Text is required.", view_mode), creator_id, ""
    if not creator_id or not creator_id.strip():
        return _error_html("Creator ID is required.", view_mode), creator_id, ""

    try:
        r = requests.post(
            f"{FLASK_BASE_URL}/submit",
            json={"text": text, "creator_id": creator_id},
            timeout=HTTP_TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        return (
            _error_html(f"Flask error: cannot reach {FLASK_BASE_URL}", view_mode),
            creator_id,
            "",
        )
    except requests.exceptions.Timeout:
        return (
            _error_html(f"Flask error: timeout after {HTTP_TIMEOUT}s", view_mode),
            creator_id,
            "",
        )

    if r.status_code == 429:
        return (
            _error_html(f"Slow down — you're sending too many requests. Try again in a moment.", view_mode),
            creator_id,
            "",
        )
    if r.status_code != 200:
        return (
            _error_html(f"Flask error: HTTP {r.status_code}: {r.text}", view_mode),
            creator_id,
            "",
        )

    body = r.json()
    html = _render_verdict(body, view_mode)
    return html, creator_id, body.get("content_id", "")


def handle_appeal(
    content_id, creator_id, reasoning, creator_id_state, content_id_state
):
    if not content_id or not creator_id or not reasoning:
        return "All fields are required.", creator_id_state, content_id_state
    try:
        r = requests.post(
            f"{FLASK_BASE_URL}/appeal",
            json={
                "content_id": content_id,
                "creator_id": creator_id,
                "creator_reasoning": reasoning,
            },
            timeout=HTTP_TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        return (
            f"Flask error: cannot reach {FLASK_BASE_URL}",
            creator_id_state,
            content_id_state,
        )
    except requests.exceptions.Timeout:
        return (
            f"Flask error: timeout after {HTTP_TIMEOUT}s",
            creator_id_state,
            content_id_state,
        )

    if r.status_code == 200:
        body = r.json()
        return (
            f"✓ Appeal filed. Status: **{body.get('status')}**.",
            creator_id_state,
            content_id_state,
        )
    if r.status_code == 403:
        return (
            "✗ Creator ID does not match the original submission.",
            creator_id_state,
            content_id_state,
        )
    if r.status_code == 404:
        return f"✗ Content ID not found.", creator_id_state, content_id_state
    if r.status_code == 409:
        return (
            "✗ An appeal is already under review for this content.",
            creator_id_state,
            content_id_state,
        )
    if r.status_code == 429:
        return (
            "✗ Slow down — you're sending too many requests. Try again in a moment.",
            creator_id_state,
            content_id_state,
        )
    return (
        f"✗ Flask error: HTTP {r.status_code}: {r.text}",
        creator_id_state,
        content_id_state,
    )


def handle_resolve(
    content_id,
    decision,
    corrected_label,
    reviewer_notes,
    creator_id_state,
    content_id_state,
):
    if not content_id or not decision:
        return (
            "Content ID and decision are required.",
            creator_id_state,
            content_id_state,
        )
    payload = {"content_id": content_id, "decision": decision}
    if corrected_label and corrected_label.strip():
        payload["corrected_label"] = corrected_label.strip()
    if reviewer_notes and reviewer_notes.strip():
        payload["reviewer_notes"] = reviewer_notes.strip()

    try:
        r = requests.post(
            f"{FLASK_BASE_URL}/resolve", json=payload, timeout=HTTP_TIMEOUT
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return (
            f"Flask error: cannot reach {FLASK_BASE_URL}",
            creator_id_state,
            content_id_state,
        )

    if r.status_code == 200:
        body = r.json()
        return (
            f"✓ Resolved. New status: **{body.get('status')}**.",
            creator_id_state,
            content_id_state,
        )
    if r.status_code == 409:
        return "✗ No active appeal to resolve.", creator_id_state, content_id_state
    if r.status_code == 429:
        return (
            "✗ Slow down — you're sending too many requests. Try again in a moment.",
            creator_id_state,
            content_id_state,
        )
    return (
        f"✗ Flask error: HTTP {r.status_code}: {r.text}",
        creator_id_state,
        content_id_state,
    )


def _filter_under_review(entries):
    """Group by content_id, take the latest event per group,
    then keep only those whose latest status is 'under_review'."""
    by_content = {}
    for e in entries:  # entries are newest-first
        cid = e.get("content_id")
        if cid and cid not in by_content:
            by_content[cid] = e
    return [e for e in by_content.values() if e.get("status") == "under_review"]


def _compute_analytics(entries):
    """Compute analytics metrics from the full audit log.

    Returns a dict with:
    - appeal_count: number of appeal events
    - resolution_count: number of resolution events
    - classification_count: number of classification events
    - appeal_rate: appeals / classifications (as %)
    - label_distribution: dict of {category: count}
    - under_review_count: number of active appeals
    """
    appeal_count = sum(1 for e in entries if e.get("event_type") == "appeal")
    resolution_count = sum(1 for e in entries if e.get("event_type") == "resolution")
    classification_count = sum(1 for e in entries if e.get("event_type") == "classification")
    under_review_count = sum(1 for e in entries if e.get("status") == "under_review")

    # Label distribution (only from classification events)
    label_dist = {}
    for e in entries:
        if e.get("event_type") == "classification":
            label = e.get("label")
            if label and isinstance(label, dict):
                category = label.get("category", "unknown")
                label_dist[category] = label_dist.get(category, 0) + 1

    # Appeal rate: appeals / classifications
    appeal_rate = (appeal_count / classification_count * 100) if classification_count > 0 else 0

    return {
        "appeal_count": appeal_count,
        "resolution_count": resolution_count,
        "classification_count": classification_count,
        "under_review_count": under_review_count,
        "appeal_rate": appeal_rate,
        "label_distribution": label_dist,
    }


def _format_analytics_markdown(metrics):
    """Format analytics metrics as markdown for display."""
    if not metrics or metrics.get("classification_count", 0) == 0:
        return "### Analytics\nNo submissions yet."

    appeal_rate = metrics.get("appeal_rate", 0)
    under_review = metrics.get("under_review_count", 0)
    label_dist = metrics.get("label_distribution", {})

    # Format label distribution as a readable list
    label_lines = []
    for category in sorted(label_dist.keys()):
        count = label_dist[category]
        label_lines.append(f"  - **{category}**: {count}")

    label_section = "\n".join(label_lines) if label_lines else "  (no labels yet)"

    return f"""### Analytics

| Metric | Value |
|--------|-------|
| **Submissions** | {metrics['classification_count']} |
| **Appeals filed** | {metrics['appeal_count']} |
| **Resolutions** | {metrics['resolution_count']} |
| **Active appeals** | {under_review} |
| **Appeal rate** | {appeal_rate:.1f}% |

#### Label Distribution
{label_section}
"""


_LOG_COLUMNS = [
    "timestamp",
    "event_type",
    "content_id",
    "creator_id",
    "status",
    "label_text",
    "combined_score",
    "degraded_value",
]


def handle_log_refresh(under_review_only, creator_id_state, content_id_state):
    try:
        r = requests.get(f"{FLASK_BASE_URL}/log", timeout=HTTP_TIMEOUT)
    except requests.exceptions.ConnectionError:
        error_md = "### Analytics\nFlask unreachable; unable to fetch metrics."
        return (
            pd.DataFrame([{"error": f"Flask unreachable at {FLASK_BASE_URL}"}]),
            error_md,
            creator_id_state,
            content_id_state,
        )
    except requests.exceptions.Timeout:
        error_md = "### Analytics\nFlask timeout; unable to fetch metrics."
        return (
            pd.DataFrame(
                [{"error": f"Flask timeout after {HTTP_TIMEOUT}s at {FLASK_BASE_URL}"}]
            ),
            error_md,
            creator_id_state,
            content_id_state,
        )

    if r.status_code == 429:
        error_md = "### Analytics\nRate limit exceeded; unable to fetch metrics."
        return (
            pd.DataFrame([{"error": "Slow down — you're sending too many requests. Try again in a moment."}]),
            error_md,
            creator_id_state,
            content_id_state,
        )
    if r.status_code != 200:
        error_md = f"### Analytics\nHTTP {r.status_code}; unable to fetch metrics."
        return (
            pd.DataFrame([{"error": f"HTTP {r.status_code}: {r.text}"}]),
            error_md,
            creator_id_state,
            content_id_state,
        )

    entries = r.json().get("entries", [])

    # Compute metrics from the full log (before any filtering)
    metrics = _compute_analytics(entries)
    metrics_md = _format_analytics_markdown(metrics)

    # Apply filtering only for the displayed table
    if under_review_only:
        entries = _filter_under_review(entries)

    if not entries:
        return (
            pd.DataFrame(columns=_LOG_COLUMNS),
            metrics_md,
            creator_id_state,
            content_id_state,
        )

    # Flatten nested 'label' and 'degraded' dicts into display columns
    flat_entries = []
    for e in entries:
        label = e.get("label") or {}
        degraded = e.get("degraded") or {}
        flat_entries.append(
            {
                "timestamp": e.get("timestamp"),
                "event_type": e.get("event_type"),
                "content_id": e.get("content_id"),
                "creator_id": e.get("creator_id"),
                "status": e.get("status"),
                "label_text": label.get("text", "") if isinstance(label, dict) else "",
                "combined_score": e.get("combined_score"),
                "degraded_value": degraded.get("value", False)
                if isinstance(degraded, dict)
                else False,
            }
        )

    df = pd.DataFrame(flat_entries)
    return df[_LOG_COLUMNS], metrics_md, creator_id_state, content_id_state


_VERDICT_IDLE_HTML = """
<div class="verdict-card" data-verdict="uncertain" style="border-top-color: var(--rule);">
  <p class="verdict-placeholder">Submit text to see a verdict.</p>
</div>
"""


def build_app(initial_banner: str = "") -> tuple:
    css = _build_css()
    with gr.Blocks(title="Provenance Guard") as demo:
        creator_id_state = gr.State(value="")
        content_id_state = gr.State(value="")

        health_banner = gr.Markdown(
            value=initial_banner, elem_classes=["health-banner"]
        )

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab(
                        "01 / Submit text", elem_classes=["tab-label"]
                    ) as submit_tab:
                        gr.Markdown("# 01 / Submit text")
                        with gr.Group(elem_classes=["form-card"]):
                            view_mode_radio = gr.Radio(
                                choices=["Creator view", "Auditor view"],
                                value="Creator view",
                                label="View mode",
                            )
                            text_input = gr.Textbox(lines=8, label="Text", value="")
                            creator_id_input = gr.Textbox(label="Creator ID", value="")
                            submit_button = gr.Button("Submit text", variant="primary")

                        gr.Examples(
                            examples=[
                                [
                                    "ok so i finally tried that new ramen place downtown "
                                    "and honestly? underwhelming. the broth was fine but "
                                    "they put WAY too much sodium in it and i was thirsty "
                                    "for like three hours after. my friend got the spicy "
                                    "version and said it was better. probably won't go "
                                    "back unless someone drags me there",
                                    "ramen_reviewer",
                                ],
                                [
                                    "Artificial intelligence represents a transformative "
                                    "paradigm shift in modern society. It is important to "
                                    "note that while the benefits of AI are numerous, it "
                                    "is equally essential to consider the ethical "
                                    "implications. Furthermore, stakeholders across "
                                    "various sectors must collaborate to ensure "
                                    "responsible deployment.",
                                    "policy_writer",
                                ],
                                [
                                    "The relationship between monetary policy and asset "
                                    "price inflation has been extensively studied in the "
                                    "literature. Central banks face a fundamental tension "
                                    "between their mandate for price stability and the "
                                    "unintended consequences of prolonged low interest "
                                    "rates on equity and real estate valuations.",
                                    "finance_academic",
                                ],
                                [
                                    "I am not doing good homie! It is what it is.",
                                    "jack",
                                ],
                                [
                                    "In conclusion, it is important to note that "
                                    "leveraging synergies across multiple domains can "
                                    "significantly enhance overall productivity and drive "
                                    "sustainable, long-term value creation.",
                                    "ai_writer_01",
                                ],
                                [
                                    "honestly idk man, today was weird lol. my dog "
                                    "ate a sock and i had to take him to the vet, "
                                    "cost me like $200 ugh",
                                    "casual_user",
                                ],
                                [
                                    "Furthermore, it should be noted that the "
                                    "aforementioned considerations necessitate a "
                                    "comprehensive reassessment of existing "
                                    "frameworks in order to optimize outcomes.",
                                    "policy_bot",
                                ],
                            ],
                            inputs=[text_input, creator_id_input],
                            label="Example inputs (click to load)",
                        )

                    with gr.Tab(
                        "02 / File appeal", elem_classes=["tab-label"]
                    ) as appeal_tab:
                        gr.Markdown("# 02 / File appeal")
                        with gr.Group(elem_classes=["form-card"]):
                            appeal_content_id_input = gr.Textbox(label="Content ID")
                            appeal_creator_id_input = gr.Textbox(label="Creator ID")
                            creator_reasoning_input = gr.Textbox(
                                lines=4,
                                label="Why is this classification wrong?",
                                value="",
                            )
                            appeal_button = gr.Button("File appeal", variant="primary")
                        appeal_output = gr.Markdown(value="")

                        appeal_tab.select(
                            fn=lambda cid, crid: (cid, crid),
                            inputs=[content_id_state, creator_id_state],
                            outputs=[appeal_content_id_input, appeal_creator_id_input],
                        )

                    with gr.Tab(
                        "03 / Resolve case", elem_classes=["tab-label"]
                    ) as resolve_tab:
                        gr.Markdown("# 03 / Resolve case")
                        with gr.Group(elem_classes=["form-card"]):
                            resolve_content_id_input = gr.Textbox(label="Content ID")
                            decision_dropdown = gr.Dropdown(
                                choices=["appeal_upheld", "appeal_overturned"],
                                label="Decision",
                            )
                            corrected_label_input = gr.Textbox(
                                label="Corrected label (if overturned)", value=""
                            )
                            reviewer_notes_input = gr.Textbox(
                                lines=3, label="Reviewer notes (optional)", value=""
                            )
                            resolve_button = gr.Button(
                                "Resolve case", variant="primary"
                            )
                        resolve_output = gr.Markdown(value="")

                    with gr.Tab("04 / Log", elem_classes=["tab-label"]) as log_tab:
                        gr.Markdown("# 04 / Log")
                        with gr.Group(elem_classes=["form-card"]):
                            under_review_checkbox = gr.Checkbox(
                                label="Show only items currently under review",
                                value=False,
                            )
                            refresh_button = gr.Button("Refresh log", variant="primary")
                        metrics_output = gr.Markdown(value="### Analytics\nNo data loaded yet.")
                        log_output = gr.Dataframe(
                            value=pd.DataFrame(columns=_LOG_COLUMNS),
                            elem_classes=["log-table"],
                        )

            with gr.Column(scale=2):
                verdict_panel = gr.HTML(
                    value=_VERDICT_IDLE_HTML, elem_classes=["verdict-card"]
                )

        submit_button.click(
            fn=handle_submit,
            inputs=[
                text_input,
                creator_id_input,
                view_mode_radio,
                creator_id_state,
                content_id_state,
            ],
            outputs=[verdict_panel, creator_id_state, content_id_state],
            api_name="submit_text",
        )

        appeal_button.click(
            fn=handle_appeal,
            inputs=[
                appeal_content_id_input,
                appeal_creator_id_input,
                creator_reasoning_input,
                creator_id_state,
                content_id_state,
            ],
            outputs=[appeal_output, creator_id_state, content_id_state],
            api_name="file_appeal",
        )

        resolve_button.click(
            fn=handle_resolve,
            inputs=[
                resolve_content_id_input,
                decision_dropdown,
                corrected_label_input,
                reviewer_notes_input,
                creator_id_state,
                content_id_state,
            ],
            outputs=[resolve_output, creator_id_state, content_id_state],
            api_name="resolve_case",
        )

        refresh_button.click(
            fn=handle_log_refresh,
            inputs=[under_review_checkbox, creator_id_state, content_id_state],
            outputs=[log_output, metrics_output, creator_id_state, content_id_state],
        )

    return demo, css


if __name__ == "__main__":
    ok, banner_text = _health_check(FLASK_BASE_URL)
    app, css = build_app(initial_banner=banner_text if not ok else "")
    app.launch(server_name="127.0.0.1", css=css, theme=DARK_THEME, inbrowser=True)

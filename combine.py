import json

import config


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a float to the inclusive range [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def combine_scores(
    llm_signal: dict[str, any],
    style_signal: dict[str, any],
    degraded_obj: dict[str, any] | None = None,
) -> tuple[float, dict[str, any]]:
    """Combine LLM and stylometric signals into a single confidence score.

    Returns a tuple of (combined_score, degraded_info).
    - combined_score: float in [0, 1]; 0.5 in degraded mode.
    - degraded_info: dict with keys:
        * value: bool – True if degraded mode.
        * reason: str | None – Explanation of degradation.
        * fallback_signal: str | None – Which signal was used as fallback.
        * fallback_score: float | None – The fallback score value.
    """
    # Determine availability of each signal
    llm_ok = (
        llm_signal.get("error") is None and llm_signal.get("ai_probability") is not None
    )
    style_ok = (
        style_signal.get("error") is None
        and style_signal.get("stylometric_score") is not None
    )

    if llm_ok and style_ok:
        combined = (
            0.6 * llm_signal["ai_probability"] + 0.4 * style_signal["stylometric_score"]
        )
        return _clamp(combined), {
            "value": False,
            "reason": None,
            "fallback_signal": None,
            "fallback_score": None,
        }

    # Degraded path
    reason_parts = []
    fallback_signal = None
    fallback_score = None

    if not llm_ok:
        reason_parts.append(
            f"llm: {llm_signal.get('error') or 'missing ai_probability'}"
        )
    if not style_ok:
        reason_parts.append(
            f"style: {style_signal.get('error') or 'missing stylometric_score'}"
        )

    if llm_ok:
        fallback_signal = "llm"
        fallback_score = llm_signal.get("ai_probability")
    elif style_ok:
        fallback_signal = "stylometric"
        fallback_score = style_signal.get("stylometric_score")

    return 0.5, {
        "value": True,
        "reason": "; ".join(reason_parts) or "unknown signal failure",
        "fallback_signal": fallback_signal,
        "fallback_score": fallback_score,
    }


def _load_labels() -> dict[str, str]:
    """Load label template strings from the JSON file defined in config."""
    try:
        with open(config.LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Labels file is optional; fall back to built-in defaults
        return {
            "likely_ai": "{band} probability of AI generation.",
            "likely_human": "{band} probability of human authorship.",
            "uncertain": "Unable to determine authorship with confidence.",
        }


def get_label(combined_score: float, degraded_obj: dict[str, any]) -> dict[str, any]:
    """Map a combined confidence score to a user‑facing label.

    Returns a dict containing:
        * category: one of "likely_ai", "uncertain", "likely_human"
        * text: the rendered label string (with {band} interpolated when appropriate)
        * band: "Low", "Medium", "High" or None (for uncertain)
    """
    # Degraded mode forces uncertain category
    if degraded_obj.get("value"):
        category = "uncertain"
        band = None
    else:
        if combined_score >= 0.75:
            category = "likely_ai"
        elif combined_score >= 0.35:
            category = "uncertain"
        else:
            category = "likely_human"

        # Determine confidence band only for non‑uncertain categories
        if category == "likely_ai":
            distance = combined_score - 0.75
        elif category == "likely_human":
            distance = 0.35 - combined_score
        else:
            distance = None

        if distance is not None:
            if distance >= 0.20:
                band = "High"
            elif distance >= 0.10:
                band = "Medium"
            else:
                band = "Low"
        else:
            band = None

    labels = _load_labels()
    template = labels.get(category, "")
    if category != "uncertain" and band is not None:
        text = template.replace("{band}", band)
    else:
        text = template

    return {"category": category, "text": text, "band": band}

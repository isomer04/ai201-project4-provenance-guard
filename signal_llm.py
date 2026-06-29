import json
import math

from groq import Groq

import config

_client = None


def _client_lazy() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


PROMPT_TEMPLATE = """You are evaluating whether the following text is AI-generated or human-written. Read it carefully, then respond with a JSON object in exactly this shape (no prose, no markdown fences):

{{\"ai_probability\": <float between 0 and 1>, \"rationale\": "<one sentence>"}}

Where ai_probability = 1.0 means you are highly confident the text is AI-generated, and 0.0 means you are highly confident it is human-written. Use the full 0-1 range. Your rationale should mention the strongest cue (in 1 sentence).

Text to evaluate:
\"\"\"
{text}
\"\"\""""


def _parse_json_lenient(raw: str) -> dict:
    """Parse JSON that may be wrapped in markdown fences.
    Raises json.JSONDecodeError if parsing fails.
    """
    s = raw.strip()
    if s.startswith("```"):
        # Remove fences and optional language tag
        s = s.strip("`")
        if "\n" in s:
            first, rest = s.split("\n", 1)
            if first.strip().lower() in {"json", "javascript", "python"}:
                s = rest
    return json.loads(s)


def get_llm_signal(text: str) -> dict:
    """Return a dict with keys: ai_probability (float|None), rationale (str|None), error (str|None).
    Never raises; on any failure returns degraded result with error description.
    """
    if not config.GROQ_API_KEY:
        return {
            "ai_probability": None,
            "rationale": None,
            "error": "GROQ_API_KEY is not configured",
        }
    prompt = PROMPT_TEMPLATE.format(text=text)
    try:
        response = _client_lazy().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=10.0,
        )
        raw = response.choices[0].message.content
        parsed = _parse_json_lenient(raw)
        prob = float(parsed["ai_probability"])
        if not math.isfinite(prob):
            return {
                "ai_probability": None,
                "rationale": None,
                "error": "ai_probability is not finite",
            }
        rationale = str(parsed["rationale"]).strip()
        prob = max(0.0, min(1.0, prob))
        if not rationale:
            return {
                "ai_probability": None,
                "rationale": None,
                "error": "empty rationale",
            }
        return {"ai_probability": prob, "rationale": rationale, "error": None}
    except Exception as e:
        return {
            "ai_probability": None,
            "rationale": None,
            "error": f"signal_llm_failed: {type(e).__name__}: {e}",
        }

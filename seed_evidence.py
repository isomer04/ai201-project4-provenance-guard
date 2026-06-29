"""Seed the audit DB with three contrasting samples and a sample appeal.

The signal scores and labels written here are FIXTURES captured from a
live run on 2026-06-28. They match the values documented in README.md,
and are written verbatim rather than being recomputed, so re-running
this script produces the same audit log regardless of model state,
network conditions, or time of day.

Used only to populate evidence for the README — does not affect normal
runs.

If you want a live (non-fixture) re-test, see _ratelimit_test.py and
run the API manually; that is the workflow that produces fresh values.
"""
import audit
import config


# Fixed fixtures captured on 2026-06-28. Tuple order:
#   (combined_score, label_dict, signals_dict)
_SAMPLES = [
    (
        "evidence_ai_test",
        "Artificial intelligence represents a transformative paradigm shift in "
        "modern society. It is important to note that while the benefits of AI are "
        "numerous, it is equally essential to consider the ethical implications. "
        "Furthermore, stakeholders across various sectors must collaborate to "
        "ensure responsible deployment.",
        0.7501181921306198,
        {
            "category": "likely_ai",
            "text": "This content was likely generated or substantially assisted "
                    "by AI. Our system's confidence in this assessment is Low. "
                    "This estimate may be less reliable for non-native English "
                    "or highly formal writing.",
            "band": "Low",
        },
        {
            "llm": {
                "ai_probability": 0.8,
                "rationale": "The text's overly formal and generic tone, "
                             "combined with its use of buzzwords like 'paradigm "
                             "shift' and 'stakeholders', suggests a high "
                             "likelihood of AI generation.",
                "error": None,
            },
            "stylometric": {
                "sentence_var_score": 0.5258864409796485,
                "ttr_score": 0.5,
                "punctuation_score": 1.0,
                "stylometric_score": 0.6752954803265495,
                "low_data": True,
                "error": None,
            },
        },
    ),
    (
        "evidence_human_test",
        "ok so i finally tried that new ramen place downtown and honestly? "
        "underwhelming. the broth was fine but they put WAY too much sodium in it "
        "and i was thirsty for like three hours after. my friend got the spicy "
        "version and said it was better. probably won't go back unless someone "
        "drags me there",
        0.20861112190666556,
        {
            "category": "likely_human",
            "text": "This content appears to be human-written. Our system's "
                    "confidence in this assessment is Medium. This estimate may "
                    "be less reliable for non-native English or highly formal "
                    "writing.",
            "band": "Medium",
        },
        {
            "llm": {
                "ai_probability": 0.2,
                "rationale": "The text's informal tone, use of colloquial "
                             "expressions, and personal anecdote suggest a high "
                             "likelihood of human authorship, with the "
                             "conversational language and casual criticism of "
                             "the ramen place being particularly indicative of "
                             "a human writer.",
                "error": None,
            },
            "stylometric": {
                "sentence_var_score": 0.2360119857285632,
                "ttr_score": 0.0,
                "punctuation_score": 0.4285714285714285,
                "stylometric_score": 0.2215278047666639,
                "low_data": False,
                "error": None,
            },
        },
    ),
    (
        "evidence_borderline",
        "The relationship between monetary policy and asset price inflation has "
        "been extensively studied in the literature. Central banks face a "
        "fundamental tension between their mandate for price stability and the "
        "unintended consequences of prolonged low interest rates on equity and "
        "real estate valuations.",
        0.5066666666666666,
        {
            "category": "uncertain",
            "text": "We could not confidently determine whether this content "
                    "was AI-generated or human-written. Treat the authorship "
                    "of this piece as unverified.",
            "band": None,
        },
        {
            "llm": {
                "ai_probability": 0.4,
                "rationale": "The text's formal tone, complex sentence "
                             "structure, and use of technical terms like "
                             "'monetary policy' and 'price stability' suggest "
                             "a high level of sophistication, but the absence "
                             "of overly repetitive or formulaic language and "
                             "the presence of nuanced ideas hint at human "
                             "authorship.",
                "error": None,
            },
            "stylometric": {
                "sentence_var_score": 0.5,
                "ttr_score": 0.5,
                "punctuation_score": 1.0,
                "stylometric_score": 0.6666666666666666,
                "low_data": True,
                "error": None,
            },
        },
    ),
]


# Fixed degraded-info object (no degradation in any of the three samples).
_DEGRADED = {
    "value": False,
    "reason": None,
    "fallback_signal": None,
    "fallback_score": None,
}


def main() -> None:
    # Use a separate evidence DB so we don't pollute the shared audit log.
    config.AUDIT_DB_PATH = "data/evidence_audit.db"
    audit.bootstrap_db()

    for cid, text, combined_score, label, signals in _SAMPLES:
        audit.append_event(
            event_type="classification",
            content_id=cid,
            creator_id=cid,
            status="classified",
            text=text,
            signals=signals,
            combined_score=combined_score,
            label=label,
            degraded=_DEGRADED,
            payload={},
            links_to=None,
        )
        print(
            f"{cid}: combined={combined_score:.3f}  "
            f"category={label.get('category')}  band={label.get('band')}"
        )
        print(f"  llm.ai_probability={signals['llm'].get('ai_probability')}")
        print(
            f"  stylometric.stylometric_score="
            f"{signals['stylometric'].get('stylometric_score')}"
        )

    # Sample appeal and resolution against the borderline classification —
    # static data, not derived from the signals above.
    original = audit.get_original_classification_for("evidence_borderline")
    appeal_event_id = audit.append_event(
        event_type="appeal",
        content_id="evidence_borderline",
        creator_id="evidence_borderline",
        status="under_review",
        text=None,
        signals=None,
        combined_score=None,
        label=None,
        degraded=None,
        payload={
            "creator_reasoning": (
                "I am a finance academic and this is an excerpt from a "
                "paper I drafted myself."
            )
        },
        links_to=original.get("event_id"),
    )
    print("\nfiled appeal against evidence_borderline -> under_review")

    audit.append_event(
        event_type="resolution",
        content_id="evidence_borderline",
        creator_id=None,
        status="appeal_overturned",
        text=None,
        signals=None,
        combined_score=None,
        label={
            "category": "likely_human",
            "text": "Manual reviewer corrected label to likely_human",
            "band": None,
        },
        degraded=None,
        payload={
            "resolution_decision": "appeal_overturned",
            "corrected_label": "likely_human",
            "reviewer_notes": (
                "Reviewed; academic prose, no clear AI pattern. Overturned."
            ),
        },
        links_to=appeal_event_id,
    )
    print("resolved evidence_borderline -> appeal_overturned (likely_human)")


if __name__ == "__main__":
    main()

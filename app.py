import os
import uuid

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import audit
import config
from combine import combine_scores, get_label
from signal_llm import get_llm_signal
from signal_stylometry import get_stylometric_signal


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Initialize rate limiter with in-memory storage
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[],
        storage_uri="memory://",
    )
    # Bootstrap the audit DB
    audit.bootstrap_db()

    @app.route("/submit", methods=["POST"])
    @limiter.limit(config.RATE_LIMITS.get("submit", "10 per minute;100 per day"))
    def submit():
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400
        text = data.get("text")
        creator_id = data.get("creator_id")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "'text' must be a non-empty string"}), 400
        if not isinstance(creator_id, str) or not creator_id.strip():
            return jsonify({"error": "'creator_id' must be a non-empty string"}), 400
        # Normalize creator_id once so padded IDs like "alice " match "alice"
        creator_id = creator_id.strip()

        content_id = str(uuid.uuid4())
        # Gather signals (degraded handling inside each)

        llm_signal = get_llm_signal(text)
        style_signal = get_stylometric_signal(text)
        combined_score, degraded_obj = combine_scores(llm_signal, style_signal)
        label = get_label(combined_score, degraded_obj)

        # Append audit event
        event_id = audit.append_event(
            event_type="classification",
            content_id=content_id,
            creator_id=creator_id,
            status="classified",
            text=text,
            signals={"llm": llm_signal, "stylometric": style_signal},
            combined_score=combined_score,
            label=label,
            degraded=degraded_obj,
            payload={},
            links_to=None,
        )

        # Fetch the new event to include a timestamp and return full signals for Auditor view
        latest_event = audit.get_latest_event_for(content_id)
        response = {
            "content_id": content_id,
            "attribution": llm_signal.get("ai_probability"),
            "confidence": combined_score,
            "label": label,
            "signals": {"llm": llm_signal, "stylometric": style_signal},
            "degraded": degraded_obj,
            "event_id": event_id,
            "timestamp": latest_event.get("timestamp") if latest_event else None,
        }
        return jsonify(response), 200

    @app.route("/appeal", methods=["POST"])
    @limiter.limit(config.RATE_LIMITS.get("appeal", "5 per minute;20 per day"))
    def appeal():
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400
        content_id = data.get("content_id")
        creator_id = data.get("creator_id")
        creator_reasoning = data.get("creator_reasoning")
        if not isinstance(content_id, str) or not content_id.strip():
            return jsonify({"error": "'content_id' must be a non-empty string"}), 400
        if not isinstance(creator_id, str) or not creator_id.strip():
            return jsonify({"error": "'creator_id' must be a non-empty string"}), 400
        if not isinstance(creator_reasoning, str) or not creator_reasoning.strip():
            return jsonify(
                {"error": "'creator_reasoning' must be a non-empty string"}
            ), 400
        # Normalize creator_id once so padded IDs like "alice " match "alice"
        creator_id = creator_id.strip()

        # Find original classification event (used for ownership) and latest event (used for active status)
        original = audit.get_original_classification_for(content_id)
        if not original:
            return jsonify({"error": "content_id not found"}), 404
        # Verify ownership against the normalized original classification's creator_id
        original_creator = (original.get("creator_id") or "").strip()
        if original_creator != creator_id:
            return jsonify({"error": "creator_id does not match original creator"}), 403
        # Check if already under review using the latest event
        latest = audit.get_latest_event_for(content_id)
        if latest and latest.get("status") == "under_review":
            return jsonify({"error": "An active appeal already exists"}), 409

        # Append appeal event linking to the original classification
        audit.append_event(
            event_type="appeal",
            content_id=content_id,
            creator_id=creator_id,
            status="under_review",
            text=None,
            signals=None,
            combined_score=None,
            label=None,
            degraded=None,
            payload={"creator_reasoning": creator_reasoning},
            links_to=original.get("event_id"),
        )
        return jsonify(
            {"content_id": content_id, "status": "under_review", "appeal_logged": True}
        ), 200

    @app.route("/resolve", methods=["POST"])
    @limiter.limit(config.RATE_LIMITS.get("resolve", "30 per minute"))
    def resolve():
        valid_categories = {"likely_ai", "uncertain", "likely_human"}
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400
        content_id = data.get("content_id")
        decision = data.get("decision")
        corrected_label = data.get("corrected_label")
        reviewer_notes = data.get("reviewer_notes")
        if not isinstance(content_id, str) or not content_id.strip():
            return jsonify({"error": "'content_id' must be a non-empty string"}), 400
        if decision not in {"appeal_upheld", "appeal_overturned"}:
            return jsonify({"error": "Invalid decision value"}), 400

        latest = audit.get_latest_event_for(content_id)
        if not latest:
            return jsonify({"error": "content_id not found"}), 404
        if latest.get("status") != "under_review":
            return jsonify({"error": "No active appeal to resolve"}), 409

        # Find the appeal event (most recent with type 'appeal')
        events = audit.get_events_for(content_id)
        appeal_event = None
        for ev in reversed(events):
            if ev.get("event_type") == "appeal":
                appeal_event = ev
                break
        if not appeal_event:
            return jsonify({"error": "Appeal event not found"}), 500

        payload = {
            "resolution_decision": decision,
            "corrected_label": corrected_label,
            "reviewer_notes": reviewer_notes,
        }
        label_to_store = None
        if decision == "appeal_overturned":
            if not corrected_label or not corrected_label.strip():
                return jsonify(
                    {"error": "'corrected_label' is required when overturning"}
                ), 400
            corrected_label = corrected_label.strip()
            if corrected_label not in valid_categories:
                return jsonify(
                    {
                        "error": f"'corrected_label' must be one of {sorted(valid_categories)}"
                    }
                ), 400
            label_to_store = {
                "category": corrected_label,
                "text": f"Manual reviewer corrected label to {corrected_label}",
                "band": None,
            }

        audit.append_event(
            event_type="resolution",
            content_id=content_id,
            creator_id=None,
            status=decision,
            text=None,
            signals=None,
            combined_score=None,
            label=label_to_store,
            degraded=None,
            payload=payload,
            links_to=appeal_event.get("event_id"),
        )
        return jsonify(
            {"content_id": content_id, "status": decision, "resolution_logged": True}
        ), 200

    @app.route("/log", methods=["GET"])
    @limiter.limit(config.RATE_LIMITS.get("log", "60 per minute"))
    def log():
        return jsonify({"entries": audit.get_log()}), 200

    return app


if __name__ == "__main__":
    app = create_app()
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = "0.0.0.0" if debug_mode else "127.0.0.1"
    app.run(debug=debug_mode, host=host, port=5000)

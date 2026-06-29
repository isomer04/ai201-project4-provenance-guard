import json
import os

from dotenv import load_dotenv

# Load .env file before reading environment variables
load_dotenv()

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Path to the SQLite audit database (created on first run).
# Can be overridden via the AUDIT_DB_PATH environment variable, which is
# primarily used by the subprocess smoke test to isolate runs from shared
# local state.
AUDIT_DB_PATH = os.environ.get("AUDIT_DB_PATH") or os.path.join(
    BASE_DIR, "data", "audit.db"
)

# Path to the labels JSON file used by the combine module
LABELS_PATH = os.path.join(BASE_DIR, "data", "labels.json")

# Rate limit configurations (per SPEC-06). These are used by Flask-Limiter decorator strings.
# Format: "<limit> per <period>; <limit> per <period>"
RATE_LIMITS = {
    "submit": "10 per minute;100 per day",
    "appeal": "5 per minute;20 per day",
    "resolve": "30 per minute",
    "log": "60 per minute",
}

# Groq API configuration (SPEC-03)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Should be set in .env or environment
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


def load_labels() -> dict[str, str]:
    """Load label templates from JSON file.

    Returns an empty dict if the file is not found.
    """
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

"""
run.py — Single-command launcher for Provenance Guard.

Starts the Flask API in a background daemon thread, waits until it is
accepting connections, then launches the Gradio UI in the main thread.

Usage:
    python run.py
"""

import sys
import threading
import time

import requests

# ── Flask ──────────────────────────────────────────────────────────────────────


def _run_flask(host: str = "127.0.0.1", port: int = 5000):
    from app import create_app

    flask_app = create_app()
    # Use the threaded werkzeug server; reloader must be OFF when running in a thread
    flask_app.run(host=host, port=port, debug=False, use_reloader=False)


def _wait_for_flask(url: str, timeout: float = 15.0) -> bool:
    """Poll until Flask /log responds with HTTP 200 or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=1.0)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.2)
    return False


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from urllib.parse import urlparse

    from ui import FLASK_BASE_URL, _health_check, build_app

    _parsed = urlparse(FLASK_BASE_URL)
    _flask_host = _parsed.hostname or "127.0.0.1"
    _flask_port = _parsed.port or 5000
    flask_url = FLASK_BASE_URL.rstrip("/")

    print("[*] Starting Flask API ...")
    flask_thread = threading.Thread(
        target=_run_flask, args=(_flask_host, _flask_port), daemon=True, name="flask"
    )
    flask_thread.start()

    print(f"    Waiting for Flask at {flask_url} ...", end="", flush=True)
    ready = _wait_for_flask(f"{flask_url}/log")
    if not ready:
        print("\n[!] Flask did not start within 15 s. Check for errors above.")
        sys.exit(1)
    print(" ready.")

    print("[*] Starting Gradio UI ...\n")
    ok, banner_text = _health_check(FLASK_BASE_URL)
    app, css = build_app(initial_banner=banner_text)
    # css is passed to launch() per Gradio 6 API; inbrowser=True auto-opens the tab
    app.launch(server_name="127.0.0.1", css=css, inbrowser=True)

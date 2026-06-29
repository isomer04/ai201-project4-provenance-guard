import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from gradio_client import Client

FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))
GRADIO_PORT = int(os.environ.get("GRADIO_PORT", "7860"))
STARTUP_TIMEOUT = 30  # seconds
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def flask_process():
    """Start Flask in a subprocess; yield; terminate after the module."""
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # so we can kill the whole group
    )
    _wait_for_port(FLASK_PORT, timeout=STARTUP_TIMEOUT)
    yield proc
    _terminate(proc)


@pytest.fixture(scope="module")
def gradio_process():
    """Start Gradio in a subprocess; yield; terminate after the module."""
    proc = subprocess.Popen(
        [sys.executable, "ui.py"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _wait_for_port(GRADIO_PORT, timeout=STARTUP_TIMEOUT)
    yield proc
    _terminate(proc)


@pytest.fixture(scope="module")
def gradio_client(gradio_process, flask_process):
    """gradio_client.Client connected to the running Gradio (Flask must be up too)."""
    url = f"http://localhost:{GRADIO_PORT}"
    return Client(url)


def _wait_for_port(port: int, timeout: int):
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            try:
                s.connect(("localhost", port))
                return
            except ConnectionRefusedError:
                time.sleep(0.5)
    raise TimeoutError(f"Port {port} did not open within {timeout}s")


def _terminate(proc: subprocess.Popen):
    import os
    import signal

    try:
        # Note: os.killpg is Unix-specific. Since OS is Windows, we might need a different approach,
        # but the spec has os.killpg. We'll use proc.terminate() for broader compatibility.
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=5)


def test_submit_returns_verdict(gradio_client):
    """Submit Input 1 from SPEC-05 (clearly-AI); verdict panel shows AI band."""
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("No GROQ_API_KEY provided")

    result = gradio_client.predict(
        "Artificial intelligence represents a transformative paradigm shift in modern society. "
        "It is important to note that while the benefits of AI are numerous, it is equally "
        "essential to consider the ethical implications. Furthermore, stakeholders across "
        "various sectors must collaborate to ensure responsible deployment.",
        "test-user",
        "Creator view",
        "",
        "",
        api_name="/submit_text",
    )
    verdict_html, creator_id_state, content_id_state = result
    assert 'data-verdict="ai"' in verdict_html
    assert "AI" in verdict_html
    assert content_id_state


def test_creator_vs_auditor_view(gradio_client):
    """Same input, Auditor view exposes the raw scores."""
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("No GROQ_API_KEY provided")

    result = gradio_client.predict(
        "Artificial intelligence represents a transformative paradigm shift in modern society. "
        "It is important to note that while the benefits of AI are numerous, it is equally "
        "essential to consider the ethical implications. Furthermore, stakeholders across "
        "various sectors must collaborate to ensure responsible deployment.",
        "test-user",
        "Auditor view",
        "",
        "",
        api_name="/submit_text",
    )
    verdict_html, _, _ = result
    assert "confidence" in verdict_html
    assert "attribution" in verdict_html
    assert "band" in verdict_html


def test_appeal_returns_under_review(gradio_client, flask_process):
    """Submit then appeal; response is under_review."""
    submit_result = gradio_client.predict(
        "ok so i tried that ramen place. underwhelming.",
        "appealer-user",
        "Creator view",
        "",
        "",
        api_name="/submit_text",
    )
    print("SUBMIT RESULT IS:", submit_result)
    _, _, content_id = submit_result
    assert content_id

    appeal_result = gradio_client.predict(
        content_id,
        "appealer-user",
        "I wrote this myself.",
        api_name="/file_appeal",
    )
    status_md, _, _ = appeal_result
    assert "under_review" in status_md
    assert "✓" in status_md or "filed" in status_md.lower()


def test_resolve_overturns(gradio_client):
    """Submit, appeal, resolve with appeal_overturned + corrected_label."""
    submit = gradio_client.predict(
        "Artificial intelligence represents a transformative paradigm shift in modern society.",
        "appealer-user",
        "Creator view",
        "",
        "",
        api_name="/submit_text",
    )
    _, _, content_id = submit
    gradio_client.predict(
        content_id,
        "appealer-user",
        "I wrote this myself.",
        api_name="/file_appeal",
    )
    resolve_result = gradio_client.predict(
        content_id,
        "appeal_overturned",
        "likely_human",
        "Reviewed and corrected.",
        api_name="/resolve_case",
    )
    status_md, _, _ = resolve_result
    assert "appeal_overturned" in status_md


def test_log_returns_entries(gradio_client):
    """Refresh log; result is a DataFrame with at least one row."""
    result = gradio_client.predict(
        False,
        "",
        "",
        api_name="/refresh_log",
    )
    df, _, _ = result
    assert len(df) > 0


def test_no_detection_imports():
    """ui.py does not import signal_llm, signal_stylometry, or combine."""
    code = (PROJECT_ROOT / "ui.py").read_text(encoding="utf-8")
    assert "import signal_llm" not in code
    assert "import signal_stylometry" not in code
    assert "import combine" not in code

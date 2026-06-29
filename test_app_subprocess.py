import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import pytest

# Use 127.0.0.1 to match the IPv4 bind address used by app.py / run.py; "localhost"
# can resolve to ::1 on some systems and miss the server bound on 127.0.0.1.
_APP_URL = "http://127.0.0.1:5000"
_STARTUP_WAIT = 5  # seconds to wait for Flask to become ready


def _wait_for_flask(proc: subprocess.Popen, timeout: float = _STARTUP_WAIT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"{_APP_URL}/log", timeout=1)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    return False


def _terminate(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.communicate(timeout=_STARTUP_WAIT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()


@pytest.fixture(scope="module")
def flask_proc():
    # Point app.py at a test-local audit DB so the smoke test does not append
    # to the shared SQLite audit log used by normal runs. config.AUDIT_DB_PATH
    # honors the AUDIT_DB_PATH environment variable, which is the existing
    # app/bootstrap entrypoint that controls the audit store.
    tmp_dir = tempfile.mkdtemp(prefix="provenance-guard-test-")
    test_db_path = os.path.join(tmp_dir, "audit.db")
    env = os.environ.copy()
    env["AUDIT_DB_PATH"] = test_db_path

    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    ready = _wait_for_flask(proc)
    if not ready:
        _terminate(proc)
        pytest.fail("Flask did not start within the allotted time")
    yield proc
    _terminate(proc)


def test_submit_endpoint(flask_proc):
    """POST /submit with valid payload returns HTTP 200 and a label field."""
    data = json.dumps({"text": "hello world", "creator_id": "test-user"}).encode(
        "utf-8"
    )
    req = urllib.request.Request(
        f"{_APP_URL}/submit",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.getcode()
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        status = e.code
        body = json.loads(e.read().decode())

    assert status == 200, f"Expected 200, got {status}: {body}"
    assert "label" in body, f"'label' missing from response: {body}"
    assert "content_id" in body, f"'content_id' missing from response: {body}"

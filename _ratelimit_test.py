"""Re-run burst and capture the body of a 429 for the README."""
import requests

URL = "http://127.0.0.1:5000/submit"
PAYLOAD = {
    "text": "This is a test submission for rate limit testing purposes only.",
    "creator_id": "ratelimit-test",
}

for i in range(1, 13):
    r = requests.post(URL, json=PAYLOAD, timeout=5)
    print(f"request {i:>2}: HTTP {r.status_code}", end="")
    if r.status_code == 429:
        print(f"  body={r.text[:200]!r}")
    else:
        print()

# Provenance Guard

A backend system that classifies submitted text as likely AI-generated, likely
human-written, or uncertain, scores confidence in that classification,
surfaces a transparency label to readers, and lets creators appeal
decisions. Built as Project 4 for the AI 201 course.

The canonical architecture spec lives in [`planning.md`](./planning.md). This
README is the canonical **record**: design rationale, evidence the system
works, known limitations, and AI-usage disclosure.

---

## 📹 Portfolio Walkthrough

**[Watch the demo (2 min)](https://www.loom.com/share/66a94a72cb8446759218ee3e946b8c2d)** — A quick tour of the system working end-to-end: submitting text, seeing the classification verdict, filing an appeal, and viewing the analytics dashboard.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate           # Mac/Linux
# source .venv/Scripts/activate     # Windows (Git Bash)
pip install -r requirements.txt
# .env must contain GROQ_API_KEY=...

python run.py             # starts API + Gradio UI together
```

Endpoints:

| Method | Path        | Purpose                                      |
|--------|-------------|----------------------------------------------|
| POST   | `/submit`   | Submit text for classification               |
| POST   | `/appeal`   | Creator appeals a classification             |
| POST   | `/resolve`  | Reviewer upholds or overturns an appeal      |
| GET    | `/log`      | Returns the full audit log as JSON           |

---

## 1. Architecture overview

A submitted piece of text travels through five stages before a reader ever
sees a verdict:

```text
                 SUBMISSION FLOW
 ┌──────────┐   POST /submit   ┌────────────────────┐
 │  Client  │ ──{text,         │   Flask app         │
 │ (creator)│    creator_id}──>│   /submit route     │
 └──────────┘                  └─────────┬───────────┘
                                          │ raw text
                       ┌──────────────────┼──────────────────┐
                       ▼                                     ▼
            ┌─────────────────────┐               ┌─────────────────────┐
            │ Signal 1: Groq LLM  │               │ Signal 2: Stylometric│
            │ judgment             │               │ heuristics (Python)│
            │ -> ai_probability    │               │ -> stylometric_score│
            └──────────┬──────────┘               └──────────┬──────────┘
                       │ signal scores                        │
                       └───────────────────┬───────────────────┘
                                            ▼
                                ┌───────────────────────┐
                                │ Confidence Scoring    │
                                │ combined = 0.6*llm +  │
                                │            0.4*style  │
                                └───────────┬───────────┘
                                            │ combined_ai_probability
                                            ▼
                                ┌───────────────────────┐
                                │ Label Selection       │
                                │ (AI / Uncertain / Human)│
                                └───────────┬───────────┘
                                            │ label text
                          ┌─────────────────┼─────────────────┐
                          ▼                                   ▼
                ┌───────────────────┐                ┌──────────────────┐
                │ Audit Log (append) │                │ JSON response to │
                │ SQLite append-only │                │ client:          │
                │ content_id, scores,│                │ content_id,      │
                │ label, status      │                │ attribution,      │
                └───────────────────┘                │ confidence, label │
                                                     └──────────────────┘
```

**Submission flow narrative.** A creator's text enters through `/submit`,
is scored independently by the LLM signal and the stylometric signal,
those two scores are combined into a single confidence value, the confidence
value is mapped to one of three label texts, and the full decision (raw
signal scores, combined score, label, `content_id`, creator_id, timestamp)
is appended to the audit log before the response is returned.

**Appeal flow.** A creator submits a `content_id`, `creator_id`, and
reasoning to `/appeal`; the system looks up the original decision,
verifies ownership (matches the original `creator_id`), and appends a
*linked* audit-log entry with status `under_review`. No existing row is
mutated and no re-classification happens automatically — the appeal simply
surfaces the case for human review. A reviewer can then `POST /resolve`
with `appeal_upheld` (the system was right) or `appeal_overturned` plus a
`corrected_label` (the system was wrong), and that decision is logged too.

The same flow is described in narrative form in
[`planning.md`](./planning.md#architecture).

---

## 2. Detection signals

### Signal 1 — LLM judgment (Groq, `llama-3.3-70b-versatile`)

- **What it measures:** Asks the model to read the text and judge, holistically,
  whether it reads as AI-generated or human-written, returning a verdict and a
  0–1 probability-of-AI estimate plus a one-sentence rationale.
- **Why this signal:** Captures semantic coherence, idea development, and
  argument structure — *does this sound like a real person?* — properties a
  statistical formula cannot see.
- **Output shape:** `{"ai_probability": float 0-1, "rationale": "<one sentence>"}`,
  parsed from a structured JSON response.
- **Blind spot:** The LLM is itself trained on text that includes lots of
  AI output, so it can be fooled by lightly-edited AI text or biased against
  unusual-but-human styles (non-native English, very formal academic writing).
  It is also non-deterministic between calls and offers no insight into *why*
  beyond a rationale string it invents after the fact.

### Signal 2 — Stylometric heuristics (pure Python)

- **What it measures:** Statistical fingerprints of writing style that are cheap
  to compute and don't require "reading" the text for meaning:
  - **Sentence length variance** — humans vary sentence length more; AI text
    tends toward uniform medium-length sentences.
  - **Type-token ratio (TTR)** — distinct words / total words; AI output is
    often less lexically diverse over longer passages.
  - **Punctuation regularity** — humans use exclamations, dashes, ellipses,
    and sentence fragments more irregularly; AI output is punctuation-regular.
- **Why this signal:** Fully deterministic, explainable (you can show the exact
  numbers), and measures *form* rather than *meaning*. It would flag a robotic
  human writer the same way it flags an AI.
- **Output shape:** `{"sentence_var_score": float 0-1, "ttr_score": float 0-1,
  "punctuation_score": float 0-1, "stylometric_score": float 0-1,
  "low_data": bool}` — the three sub-metrics averaged (equal weight) into one
  combined score.
- **Blind spot:** Short texts (a few sentences) don't give enough data for
  variance or TTR to be meaningful, so the signal degrades on short
  submissions. It can't catch AI text that has been deliberately *humanized*
  by varying sentence length and adding irregular punctuation, since it never
  looks at meaning.

**A signal we deliberately do not use — perplexity.** Per-token
log-likelihood / perplexity (how "surprised" a reference model is by each
token) is the strongest structural signal in the AI-detection literature. We
exclude it because the Groq chat API does not give us convenient access to
token logprobs, and approximating it would add a second model call per
submission. This is a known, deliberate trade-off.

### Why these two together

They fail in different ways. LLM judgment reads meaning; stylometric
heuristics read form. When they agree, confidence is high. When they
disagree, the disagreement itself is the most honest signal — the system
returns "Uncertain" rather than picking a winner.

---

## 3. Confidence scoring

### Combination formula

```text
combined_ai_probability = 0.6 * llm_ai_probability + 0.4 * stylometric_score
```

The LLM signal is weighted higher (0.6) because it captures semantic cues a
pure statistical signal cannot. The stylometric signal acts as a
check/anchor since it's deterministic and can't be talked out of its answer
the way an LLM prompt sometimes can.

### Thresholds

These are **asymmetric on purpose** — wrongly accusing a human is worse than
missing an AI submission, so "likely AI" is harder to reach than "likely
human" by design.

| `combined_ai_probability` range | Label category      |
|----------------------------------|---------------------|
| `>= 0.75`                        | Likely AI-generated |
| `0.35 – 0.74`                    | Uncertain           |
| `< 0.35`                         | Likely human-written |

A score of 0.5–0.6 means the two signals disagree or both signals are weakly
ambiguous; the system returns "Uncertain" rather than force a guess.

For the AI and human labels, a `band` of `Low` / `Medium` / `High` is
computed from how far the combined score sits past its threshold
(`>= 0.20` past → High; `0.10–0.19` past → Medium; `< 0.10` past → Low). The
Uncertain label deliberately omits the band from the reader-facing text
because "we cannot tell" should not sound more precise than it is.

### Degraded mode

If either signal fails (Groq timeout, rate-limit, unparseable response, or
an unexpected local stylometric error), we do *not* return an error. We:

1. Record any available raw signal score in `signals`,
2. Set the public `combined_score` to neutral `0.5`,
3. Force the result into the **Uncertain** category, and
4. Record a structured `degraded` object with the failure reason and any
   fallback score.

A single signal is not enough to accuse or clear anyone.

### How I validated that scores vary meaningfully

I ran four deliberately chosen inputs through the pipeline and checked that
the scores moved in the direction I expected. Two snapshots from that
testing, using real Groq outputs captured today (`2026-06-28`):

**Example 1 — clearly AI-generated prose (high-confidence case):**

> "Artificial intelligence represents a transformative paradigm shift in
> modern society. It is important to note that while the benefits of AI are
> numerous, it is equally essential to consider the ethical implications.
> Furthermore, stakeholders across various sectors must collaborate to ensure
> responsible deployment."

| Component                       | Score |
|---------------------------------|-------|
| `llm.ai_probability`            | `0.80` |
| `stylometric.stylometric_score` | `0.675` |
| **Combined** (0.6·0.80 + 0.4·0.675) | **`0.750`** |
| Label                           | `likely_ai` (band: Low — just barely crossed 0.75) |

**Example 2 — clearly casual human writing (lower-confidence case):**

> "ok so i finally tried that new ramen place downtown and honestly?
> underwhelming. the broth was fine but they put WAY too much sodium in it
> and i was thirsty for like three hours after. my friend got the spicy
> version and said it was better. probably won't go back unless someone
> drags me there"

| Component                       | Score |
|---------------------------------|-------|
| `llm.ai_probability`            | `0.20` |
| `stylometric.stylometric_score` | `0.222` |
| **Combined** (0.6·0.20 + 0.4·0.222) | **`0.209`** |
| Label                           | `likely_human` (band: Medium) |

The two `combined_score` values are `0.750` vs. `0.209` — they differ by
0.54, which crosses two label bands (AI vs. human). That is a meaningful
difference, not a binary flip at the 0.5 boundary.

A third sample — formal academic prose about monetary policy —
produced `combined = 0.507` (LLM: 0.40, stylometric: 0.667) and was
correctly placed in the `uncertain` band, demonstrating that the system
does not collapse to a binary outcome when the signals disagree.

### What I'd change for production

I would fit the 0.6 / 0.4 weighting and the 0.75 / 0.35 thresholds on a
labeled dataset of ~1,000 known-AI and ~1,000 known-human samples, report a
ROC curve, and pick the threshold that yields the lowest false-positive
rate at the chosen recall. Until then, every number here is an informed
starting point, not a measured one.

---

## 4. Transparency label — verbatim text of all three variants

These are the exact strings shown to a non-technical reader. They live in
[`data/labels.json`](./data/labels.json) and are interpolated at runtime.

| Category             | Verbatim label text |
|----------------------|---------------------|
| Likely AI-generated  | `"This content was likely generated or substantially assisted by AI. Our system's confidence in this assessment is {band}. This estimate may be less reliable for non-native English or highly formal writing."` |
| Uncertain            | `"We could not confidently determine whether this content was AI-generated or human-written. Treat the authorship of this piece as unverified."` |
| Likely human-written | `"This content appears to be human-written. Our system's confidence in this assessment is {band}. This estimate may be less reliable for non-native English or highly formal writing."` |

`{band}` is substituted with `Low`, `Medium`, or `High` based on how far the
score sits past the threshold. The Uncertain label omits the band on
purpose — "we cannot tell" should not sound more precise than it is — but
the API still returns `label.band` as metadata for audit and debugging.

The known-bias disclosure (the trailing sentence about non-native English
and formal writing) is surfaced inline because the LLM signal is documented
to skew against those voices, and the false-positive stakes warrant telling
the reader directly rather than burying it in the audit log.

The text also deliberately avoids precise percentages. The combined score
is a blend of two *uncalibrated* signals; printing "87% confident" would be
false precision.

---

## 5. Rate limiting

Limits are configured in [`config.py`](./config.py):

| Endpoint   | Limit                | Reasoning |
|------------|----------------------|-----------|
| `/submit`  | `10 per minute; 100 per day` | A working creator might submit several drafts in quick succession; 10/min covers that without letting a script flood the system. 100/day is the daily abuse ceiling — well above any legitimate creator's volume. |
| `/appeal`  | `5 per minute; 20 per day`   | Appeals are rarer than submissions and should not be spammable. 5/min lets a real creator file on multiple of their own items quickly. |
| `/resolve` | `30 per minute`              | Internal reviewer endpoint — used by humans triaging the queue, not the public, so the limit is generous. |
| `/log`     | `60 per minute`              | Read-only endpoint used by the UI; the limit exists only to prevent accidental tight loops. |

These are *defensible numbers* based on how a writing platform actually
behaves, not arbitrary defaults. The implementation uses
`flask-limiter` with in-memory storage (`memory://`); for a real
deployment you would point it at Redis.

### Evidence the limits work

This is the output of 12 rapid `POST /submit` requests hitting a freshly
started server (one fixed creator_id, identical text):

```text
request  1: HTTP 200
request  2: HTTP 200
request  3: HTTP 200
request  4: HTTP 200
request  5: HTTP 200
request  6: HTTP 200
request  7: HTTP 200
request  8: HTTP 200
request  9: HTTP 200
request 10: HTTP 200
request 11: HTTP 429  body='<!doctype html>\n<html lang=en>\n<title>429 Too Many Requests</title>\n<h1>Too Many Requests</h1>\n<p>10 per 1 minute</p>\n'
request 12: HTTP 429  body='<!doctype html>\n<html lang=en>\n<title>429 Too Many Requests</title>\n<h1>Too Many Requests</h1>\n<p>10 per 1 minute</p>\n'
```

The first 10 succeed (200), the next 2 are rejected (429) with a body that
restates the configured limit. Captured from the harness by
[`_ratelimit_test.py`](./_ratelimit_test.py).

---

## 6. Audit log

Every attribution decision — including the text, both signal scores,
combined score, label, status, and any appeals — is captured in a
structured SQLite audit log (`data/audit.db` by default; override with
`AUDIT_DB_PATH` for tests). The log is append-only: no row is mutated;
appeals and resolutions create *new* linked rows.

`GET /log` returns the full audit log as JSON. Three example entries
below — one classification, one appeal, one resolution — covering all
three event types:

### Classification (likely AI)

```json
{
  "event_type": "classification",
  "content_id": "evidence_ai_test",
  "creator_id": "evidence_ai_test",
  "status": "classified",
  "combined_score": 0.7501181921306198,
  "label": {
    "category": "likely_ai",
    "text": "This content was likely generated or substantially assisted by AI. Our system's confidence in this assessment is Low. This estimate may be less reliable for non-native English or highly formal writing.",
    "band": "Low"
  },
  "signals": {
    "llm": {
      "ai_probability": 0.8,
      "rationale": "The text's overly formal and generic tone, combined with its use of buzzwords like 'paradigm shift' and 'stakeholders', suggests a high likelihood of AI generation."
    },
    "stylometric": {
      "sentence_var_score": 0.5258864409796485,
      "ttr_score": 0.5,
      "punctuation_score": 1.0,
      "stylometric_score": 0.6752954803265495,
      "low_data": true
    }
  },
  "degraded": {
    "value": false
  }
}
```

### Classification (uncertain)

```json
{
  "event_type": "classification",
  "content_id": "evidence_borderline",
  "creator_id": "evidence_borderline",
  "status": "classified",
  "combined_score": 0.5066666666666666,
  "label": {
    "category": "uncertain",
    "text": "We could not confidently determine whether this content was AI-generated or human-written. Treat the authorship of this piece as unverified.",
    "band": null
  },
  "signals": {
    "llm": {
      "ai_probability": 0.4,
      "rationale": "The text's formal tone, complex sentence structure, and use of technical terms like 'monetary policy' and 'price stability' suggest a high level of sophistication, but the absence of overly repetitive or formulaic language and the presence of nuanced ideas hint at human authorship."
    },
    "stylometric": {
      "sentence_var_score": 0.5,
      "ttr_score": 0.5,
      "punctuation_score": 1.0,
      "stylometric_score": 0.6666666666666666,
      "low_data": true
    }
  }
}
```

### Appeal → Resolution (under_review → appeal_overturned)

```json
[
  {
    "event_type": "appeal",
    "content_id": "evidence_borderline",
    "creator_id": "evidence_borderline",
    "status": "under_review",
    "payload": {
      "creator_reasoning": "I am a finance academic and this is an excerpt from a paper I drafted myself."
    },
    "links_to": "<original classification event_id>"
  },
  {
    "event_type": "resolution",
    "content_id": "evidence_borderline",
    "status": "appeal_overturned",
    "label": { "category": "likely_human", "text": "Manual reviewer corrected label to likely_human", "band": null },
    "payload": {
      "resolution_decision": "appeal_overturned",
      "corrected_label": "likely_human",
      "reviewer_notes": "Reviewed; academic prose, no clear AI pattern. Overturned."
    },
    "links_to": "<appeal event_id>"
  }
]
```

Notice how each row links to its predecessor via `links_to` — the appeal
links back to the original classification, the resolution links back to
the appeal. This is what makes the log auditable end-to-end without
mutating prior rows.

---

## 7. Appeals workflow

- **Who can appeal:** Only the creator who submitted the content. The
  `/appeal` request carries `creator_id`; the system verifies it matches
  the `creator_id` stored on the original submission. A mismatch returns
  `403` and is **not** logged — this prevents anyone who merely knows a
  `content_id` from forging an appeal in another creator's name. (`creator_id`
  is a trust-on-submission identifier, not an authenticated login; the
  ownership check is the minimum integrity guarantee within that model.)
- **What they provide:** `content_id`, their `creator_id`, and
  `creator_reasoning` (free-text explanation of why the classification is
  wrong).
- **What the system does:**
  1. Look up the original classification by `content_id`. `404` if not found.
  2. Compare the submitted `creator_id` against the original. `403` on mismatch.
  3. Check the latest event for that `content_id`. If already `under_review`,
     return `409 Conflict` — one active appeal per content_id.
  4. Append a new audit row with `event_type=appeal`, `status=under_review`,
     `payload.creator_reasoning=<reasoning>`, and `links_to=<original event_id>`.
  5. Return `{content_id, status: "under_review", appeal_logged: true}`.
- **What a reviewer sees:** The Gradio UI's "Log" tab with the
  "Under review only" filter exposes every active appeal in the queue —
  `content_id`, the creator's reasoning, the original classification
  signal scores and label, and a `Resolve` button that posts to
  `/resolve` with `appeal_upheld` or `appeal_overturned` + optional
  `corrected_label`.

---

## 8. Stretch features completed

### Ensemble detection (3+ signals with documented weighting) — ✅

The stylometric signal is itself an ensemble of three sub-signals,
combined with equal weight, then combined with the LLM signal under a
0.6 / 0.4 weighting:

```text
combined = 0.6 * llm.ai_probability
         + 0.4 * (
             sentence_var_score
           + ttr_score
           + punctuation_score
           ) / 3
```

That is **four weighted components** (1 LLM + 3 structural), exceeding the
"3 or more" requirement.

### Analytics dashboard — ✅

The Gradio UI (`ui.py`) ships with a **Log** tab that:

- Refreshes the audit log on demand (`GET /log`).
- Displays analytics metrics above the table, including:
  - **Submissions**: total classifications
  - **Appeals filed**: count of appeal events
  - **Resolutions**: count of resolution events
  - **Active appeals**: count of items currently `under_review`
  - **Appeal rate**: percentage of submissions that were appealed
  - **Label distribution**: breakdown of classifications by category
    (`likely_ai`, `uncertain`, `likely_human`)
- Filters the table to show only `under_review` items (the live appeal queue).
- Renders the audit log as a sortable pandas DataFrame with columns for
  `event_type`, `content_id`, `status`, `combined_score`, and `label`.

That covers the "detection patterns, appeal rates, and one additional metric"
requirement — the metrics are computed fresh on each refresh and displayed
alongside the audit table.

### Stretch features not attempted

- **Provenance certificate** — designing a "verified human" credential would
  have required either an out-of-band identity verification step (out of
  scope for a backend demo) or a self-attestation that adds no real trust.
- **Multi-modal support** — extending the pipeline to images would require
  a second detection signal design (e.g., CLIP-based stylometry or metadata
  heuristics), and the assignment was scoped to text.

---

## 9. Known limitations

At least one specific case the system would get wrong:

1. **Highly formal academic prose by a human** — The "monetary policy"
   sample above scored `0.507` (uncertain) and was overturned to
   `likely_human`. The LLM signal returned `0.40` (a sensible read) but
   the stylometric signal returned `0.667` because `low_data` was `true`
   on the short excerpt. The combined 0.507 swallowed a real human writer
   into the uncertain bucket rather than wrongly accusing them — which is
   the correct bias direction — but a real submitter would still see "we
   cannot tell" on work they wrote themselves.

2. **Short submissions (< ~30 words)** — The stylometric signal returns
   neutral `0.5` and sets `low_data: true` when there are too few
   sentences / words to compute meaningful variance or TTR. So if you
   submit a single sentence, you get `(0.6 * llm + 0.4 * 0.5)` — the
   combined score is dominated by the LLM's read, with the structural
   signal effectively absent. A sufficiently clever prompt-injection that
   talks the LLM around would slip past.

3. **Heavily humanized AI text (AI output deliberately varied in sentence
   length and sprinkled with colloquial punctuation)** — Stylometry will
   miss it because it only reads form. The LLM will likely catch it, but
   if the "humanizer" is good enough the combined score may slip below
   0.75 into the uncertain bucket.

4. **Non-native English speakers and very formal academic writing** —
   This is a documented bias of the LLM judge. The label text itself
   surfaces this disclosure inline. The asymmetric 0.75/0.35 thresholds
   are the system's main structural mitigation: a human writing in a
   slightly stilted register is less likely to be wrongly labelled "likely
   AI" than an AI submission is to be wrongly labelled "likely human".

---

## 10. Spec reflection

### One way the spec helped

The asymmetric-threshold decision (0.75 for AI, 0.35 for human) was made
in [`planning.md`](./planning.md#2-uncertainty-representation) before any
code was written. Writing it down forced the asymmetry to be an explicit
choice rather than an accident of how the math landed. The label
variants (`{band}` text, `Uncertain` deliberately not getting a band)
were also locked in the spec, so the implementation never had to
back-track on UX decisions after the classifier was already producing
scores.

### One way implementation diverged from the spec

The spec described scoring in terms of distance from the *nearest*
threshold (`0.35` for low AI-likelihood, `0.75` for high) — i.e., the
band for an `Uncertain` result was supposed to depend on which side of
0.5 the combined score sat on and how close to either threshold. In the
implementation, I deliberately simplified this: the `Uncertain` label
returns `band: null` and surfaces no band at all, because "we cannot
tell" + "low confidence" still sounds more precise than "we cannot
tell" alone does. The spec got the *principle* right (do not over-claim
precision under uncertainty) but the literal text needed to ship
the principle more honestly.

---

## 11. AI usage — at least two specific instances

### Instance 1 — Groq signal prompt design (`signal_llm.py`)

I drafted the initial system prompt as a simple "classify this text" ask.
The model returned prose wrapped in markdown, which broke the JSON parser
on the first three test runs. I rewrote the prompt to (a) require JSON
with exact keys, (b) constrain `ai_probability` to `[0,1]`, and (c) cap
the rationale to one sentence. After those changes the parse rate
became 100% across all four test inputs. The implementation note
("we ask the model to respond in JSON to avoid parsing prose") is in
`planning.md` so the choice is documented as a deliberate engineering
trade-off, not a happy accident.

### Instance 2 — Punctuation scoring formula (`signal_stylometry.py`)

The first version of `_punctuation_score` measured the absolute count of
`!`, `?`, `;`, `:`, `—`, `(`, `)` and divided by word count. On the
ramen-shop review sample the score came out at `~0.15` (very low AI),
which was correct, but on the AI-paradigm-shift sample the score was
also `~0.15` because that text had no punctuation irregularities at all
— accidentally the same as a human who happened to use few punctuation
marks. I rewrote the score to be a *ratio* of irregular-punctuation to
total-punctuation (`irregular_ratio`), with a `low_data` fallback when
there are fewer than 2 punctuation marks total. That separated the
"no punctuation at all" case (neutral) from the "all-periods-and-commas"
case (AI-like) and the "lots of irregular punctuation" case (human).

---

## Repo layout

```text
app.py                  # Flask app + all routes (/submit, /appeal, /resolve, /log)
audit.py                # SQLite-backed audit log helpers
combine.py              # signal combination + label generation
signal_llm.py           # Groq LLM signal (Signal 1)
signal_stylometry.py    # stylometric heuristics (Signal 2)
config.py               # paths, rate-limit table, model name
data/labels.json        # transparency-label templates
data/audit.db           # audit log (gitignored; created on first run)
ui.py                   # Gradio UI (Creator / Auditor / Log tabs)
run.py                  # single-command launcher (API + UI)
seed_evidence.py        # populates data/evidence_audit.db with three samples
_ratelimit_test.py      # captures the 12-request burst for README evidence
planning.md             # canonical architecture spec (written before any code)
README.md               # this file
tests/                  # UI subprocess tests
test_app_subprocess.py  # API subprocess smoke test
```

## License

This project is currently unlicensed. No `LICENSE` file is included in the
repository; all rights are reserved by the author until a license is
added. Educational project; pick a license before public release.

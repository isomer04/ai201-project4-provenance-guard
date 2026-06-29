# Provenance Guard — Planning

## 1. Detection Signals

**Design principle — fail safe, not fail open.** Anywhere this system has to make a
decision under uncertainty (unparseable LLM output, signal timeout, short input,
disagreement between signals), the default behavior is the one that is *least harmful
to a wrongly-accused human creator*, not the one that is most useful to a correct
accusation. Concrete applications of this principle are called out in the degraded-mode
fallback below, the asymmetric thresholds (Section 2), and the refusal-to-print
precise percentages (Section 3). This is named once here so it reads as a rule, not as
scattered defensiveness.

Two independent signals, chosen because they fail in different ways — one is semantic
(reads the text like a human judge would), the other is structural (measures the text
without "understanding" it at all).

### Signal 1: LLM Judgment (Groq, llama-3.3-70b-versatile)

- **What it measures:** Asks the model to read the submitted text and judge, holistically,
  whether it reads as AI-generated or human-written, returning a verdict and a 0–1
  probability-of-AI estimate plus a one-sentence rationale.
- **Why it differs from Signal 2:** It captures semantic coherence, idea development,
  argument structure, and "does this sound like a real person" — things no statistical
  formula can see.
- **Output shape:** `{"ai_probability": float 0-1, "rationale": str}` parsed from a
  structured JSON response (we ask the model to respond in JSON to avoid parsing prose).
- **Blind spot:** The LLM is itself a model trained on text that includes lots of AI
  output; it can be fooled by lightly-edited AI text or biased against unusual-but-human
  styles (e.g., non-native English speakers, very formal academic writing, ESL authors).
  It is also non-deterministic between calls and offers no insight into *why* beyond a
  rationale string it invents after the fact.

### Signal 2: Stylometric Heuristics (pure Python)

- **What it measures:** Statistical fingerprints of writing style that are cheap to
  compute and don't require "reading" the text for meaning:
  - **Sentence length variance** — humans vary sentence length more; AI text tends
    toward uniform medium-length sentences.
  - **Type-token ratio (vocabulary diversity)** — distinct words / total words; AI
    output is often less lexically diverse over longer passages.
  - **Punctuation regularity** — humans use exclamations, dashes, ellipses,
    sentence fragments more irregularly; AI output is punctuation-regular. (Note: this
    is *not* "burstiness" in the detection-literature sense, which measures variance in
    per-token perplexity/surprise.)

  **Signal we deliberately do not use — perplexity.** The strongest structural signal in
  the AI-detection literature is per-token log-likelihood / perplexity (how "surprised" a
  reference model is by each token). We exclude it because the Groq chat API does not give
  us convenient access to token logprobs, and approximating it would add a second model
  call per submission. We name this explicitly so the omission is a known, deliberate
  trade-off rather than an oversight.
- **Why it differs from Signal 1:** It is fully deterministic, explainable (you can show
  the exact numbers), and measures *form* rather than *meaning* — it would flag a robotic
  human writer the same way it flags an AI.
- **Output shape:** `{"sentence_var_score": float 0-1, "ttr_score": float 0-1,
  "punctuation_score": float 0-1, "stylometric_score": float 0-1,
  "low_data": bool}` — the three sub-metrics averaged (equal weight) into one
  combined score.

  **Concrete scoring formulas (heuristic, uncalibrated):** all sub-scores are
  AI-likeness scores in `[0, 1]`, where higher means "more AI-like." Tokenization uses
  lowercase word tokens from a regex such as `[A-Za-z0-9']+`; sentence splitting uses
  `.`, `!`, `?`, and newlines as boundaries.

  - `sentence_var_score`: if fewer than 3 sentences or fewer than 30 words, return
    `0.5` and set `low_data: true`. Otherwise compute the coefficient of variation
    of sentence word counts: `cv = stdev(sentence_lengths) / mean(sentence_lengths)`.
    Map uniform sentence length to higher AI-likeness with
    `clamp(1 - (cv / 0.8), 0, 1)`.
  - `ttr_score`: if fewer than 50 words, return `0.5` and set `low_data: true`.
    Otherwise compute `ttr = unique_words / total_words`; low vocabulary diversity is
    more AI-like, so use `clamp((0.55 - ttr) / 0.25, 0, 1)`.
  - `punctuation_score`: if there are fewer than 2 punctuation marks, return `0.5`
    and set `low_data: true`. Otherwise compute
    `irregular_ratio = count(! ? ; : — - … ( )) / total_punctuation`. Regular
    period/comma-heavy punctuation is more AI-like, so use
    `clamp(1 - (irregular_ratio / 0.35), 0, 1)`.

  `stylometric_score` is the arithmetic mean of the three sub-scores. These constants
  are intentionally simple starting points for a demo; M4 validation can adjust them if
  they behave obviously backward on the four test inputs.
- **Blind spot:** Short texts (a few sentences) don't give enough data for variance or
  TTR to be meaningful — the signal degrades on short submissions. It also can't catch
  AI text that has been deliberately "humanized" by varying sentence length and adding
  irregular punctuation, since it never looks at meaning.

### Combining Into One Confidence Score

```
combined_ai_probability = 0.6 * llm_ai_probability + 0.4 * stylometric_score
```

The LLM signal is weighted higher (0.6) because it captures semantic cues a pure
statistical signal cannot, but the stylometric signal acts as a check/anchor since it's
deterministic and can't be talked out of its answer the way an LLM prompt sometimes can.

**Caveat — these weights are uncalibrated.** The 0.6/0.4 split and the 0.75/0.35
thresholds are reasoned defaults, not values fit to labeled data. Before trusting any
output we run a small validation set (≈20–40 known-AI + 20–40 known-human samples) and
report a confusion matrix; the weights/thresholds may be adjusted once, after seeing that
matrix. Until then, every number here is an informed starting point, not a measured one.

**Degraded mode (one signal unavailable).** If either signal fails (Groq timeout,
rate-limit, unparseable response, or an unexpected local stylometric error — see
Resilience below), we do *not* return an error to the creator. We record any available
raw signal score in `signals`, but set the decision-level `combined_score` to neutral
`0.5`, force the result into the **Uncertain** category, and record a structured
`degraded` object with the failure reason and any fallback score. A single signal is not
enough to accuse or clear anyone.

## 2. Uncertainty Representation

In normal mode, `combined_ai_probability` is a single float in [0, 1] representing "how
likely this text is AI-generated." It is **not** rounded to a binary label internally —
the numeric score is stored in the audit log. In degraded mode, `combined_score` is set to
neutral `0.5` for the public decision, while the available raw fallback signal is stored
inside `signals` and `degraded.fallback_score`.

Thresholds (chosen to bias against false positives — wrongly accusing a human is worse
than missing an AI submission):

| Range              | Category            |
|---------------------|----------------------|
| `>= 0.75`           | Likely AI-generated  |
| `0.35 – 0.74`       | Uncertain            |
| `< 0.35`            | Likely human-written |

The "likely AI" band starts higher (0.75) than the "likely human" band's mirror point
(0.65 would be the naive mirror of 0.35) — i.e., the AI band is deliberately *harder to
reach* than the human band, on purpose, to reduce false accusations of human creators.

A score of 0.5–0.6 means: the two signals disagree, or both signals are weakly
ambiguous — the system should say "we can't tell" rather than force a guess.

## 3. Transparency Label Variants

Exact text shown to a non-technical reader, returned in the `label` field of the
`/submit` response:

| Category             | Label text shown to readers |
|----------------------|------------------------------|
| Likely AI-generated  | `"This content was likely generated or substantially assisted by AI. Our system's confidence in this assessment is {band}. This estimate may be less reliable for non-native English or highly formal writing."` |
| Uncertain            | `"We could not confidently determine whether this content was AI-generated or human-written. Treat the authorship of this piece as unverified."` |
| Likely human-written | `"This content appears to be human-written. Our system's confidence in this assessment is {band}. This estimate may be less reliable for non-native English or highly formal writing."` |

For the AI and Human labels, `{band}` is a **coarse confidence band** — `Low`,
`Medium`, or `High` — derived from how far the combined score sits past its threshold,
not a precise percentage. The Uncertain label deliberately omits `{band}` from the
reader-facing text because "we cannot tell" should not sound more precise than it is;
however, the API still returns `label.band` as metadata for debugging and audit review.

We deliberately do **not** print a number like "87% confident": the combined score is a
blend of two *uncalibrated* signals and is not a true probability, so a precise percentage
would be false precision.

Band cutoffs (by distance of the combined score from the **nearest** decision threshold
— for the AI band, distance from 0.75; for the Uncertain band, distance from whichever
of 0.35 or 0.75 is closer; for the Human band, distance from 0.35):

| Distance from threshold | Band   |
|-------------------------|--------|
| `>= 0.20`               | High   |
| `0.10 – 0.19`           | Medium |
| `< 0.10`                | Low    |

We also surface a known-bias disclosure inline in the AI/human labels, because the
LLM signal is documented to skew against non-native English and very formal academic
writing (see Signal 1 blind spot), and the false-positive stakes warrant disclosing it to
the reader directly rather than burying it in the audit log.

## 4. Appeals Workflow

- **Who can appeal:** Only the creator who submitted the content. The `/appeal` request
  carries a `creator_id`, and the system verifies it matches the `creator_id` stored on
  the original submission. A mismatch returns `403` and is **not** logged as an appeal —
  this prevents anyone who merely knows a `content_id` from forging an appeal in another
  creator's name. (In this project `creator_id` is a trust-on-submission identifier, not
  authenticated; the ownership check is the minimum integrity guarantee within that
  scope.)
- **What they provide:** `content_id` + `creator_id` + free-text `creator_reasoning`
  explaining why they believe the classification is wrong.
- **What the system does on receipt:**
  1. Look up the original audit log entry for `content_id`.
  2. Verify the request's `creator_id` matches the submission's `creator_id`
     (else `403`, no state change).
  3. Append a new audit log entry of type `appeal` with `status: "under_review"`,
     linking back to the original decision, storing the creator's reasoning and a
     timestamp. The original classification row is never mutated; current status is
     derived from the latest event for that `content_id`.
  4. Return a confirmation `{content_id, status: "under_review", appeal_logged: true}`.
- **No automated re-classification.** A human reviewer is expected to look at the appeal
  queue (in this project, simply: content whose current status is `under_review`) and
  manually decide. What a reviewer would see: original text, original signals/scores/label,
  and the creator's reasoning, side by side.
- **Resolution states.** An appeal is not complete until a reviewer resolves it. The
  status lifecycle is:

  ```
  classified ──appeal──> under_review ──reviewer──> appeal_upheld   (original label stands)
                                              └────> appeal_overturned (label corrected)
  ```

  Resolution is recorded as another append-only audit event of type `resolution`,
  capturing the reviewer's decision, an optional corrected label, and a timestamp. On
  `appeal_overturned`, the corrected label becomes the content's current public label;
  the original (now-wrong) classification remains in the log for accountability. A demo
  `POST /resolve` endpoint (or admin action) drives this transition — without it, an
  appeal could never close, which is not an appeals process.

## 5. Anticipated Edge Cases

1. **Short-form content (a haiku, a tweet-length excerpt):** Stylometric signal has too
   little data — sentence count of 1–2 makes "sentence length variance" meaningless. The
   stylometric function returns neutral `0.5` sub-scores with `low_data: true`, so the LLM
   signal drives most of the variation for short text; this should be disclosed as a known
   weak point.
2. **Highly repetitive, simple-vocabulary creative writing (e.g., a children's poem or a
   poem using deliberate repetition as a device):** Low type-token ratio and low sentence
   variance look statistically identical to "uniform AI output," even though the
   uniformity here is a deliberate human artistic choice. This is a likely false-positive
   source the stylometric signal cannot distinguish from genuine AI uniformity.
3. **Heavily-edited AI draft (human revises AI output sentence by sentence):** Both
   signals can be partially fooled — stylometrics because the human editing reintroduces
   variance, the LLM because surface polish improves but underlying structure may still
   read as AI. Expect this to land in the "uncertain" band, which is the intended,
   honest outcome rather than a wrong binary guess.

## 6. Rate Limiting

Rate limits protect two things: (a) the **Groq free-tier quota** — every `/submit` makes
one LLM call, and the free tier caps requests per minute and per day; and (b) the
**integrity of the audit log and appeal queue** — a flooder shouldn't be able to pollute
the log or saturate reviewer time with bogus appeals.

### Limits (decided)

| Endpoint   | Limit                    | Reasoning                                                                                                                                                          |
|------------|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `POST /submit`   | `10 per minute; 100 per day` | A genuine writer submitting their own work would not post 10+ pieces in a minute, or 100 in a day — those numbers reflect abuse patterns, not normal usage. The minute limit protects the Groq free-tier RPM; the day limit protects the free-tier daily quota and bounds the audit-log growth rate. |
| `POST /appeal`   | `5 per minute; 20 per day`   | Appeals should be rare — most creators will never appeal, and a single creator typically has only a handful of submissions to dispute. The lower limit reflects that. |
| `GET /log`       | `60 per minute`              | Read endpoint, cheap, no external dependency — generous limit so dashboards/graders can inspect freely.                                                           |
| `POST /resolve`  | `30 per minute`              | Admin action — generous enough for a human reviewer working through a queue, low enough to prevent scripted tampering.                                             |

### Why these numbers specifically

- **10/minute on `/submit`** is exactly what the assignment's Flask-Limiter setup example
  uses (`@limiter.limit("10 per minute;100 per day")`), and it lines up with Groq's
  free-tier RPM for `llama-3.3-70b-versatile`. Going higher risks 429s from Groq, which
  would trigger our degraded-mode fallback unnecessarily.
- **100/day on `/submit`** is a hard ceiling on per-IP consumption of Groq quota. If
  this were deployed for real with a paid tier, this number would scale with budget.
- **The minute and day limits together** defend against two distinct attacker profiles:
  burst floods (the per-minute limit) and slow-drip probing (the per-day limit).
- **Lower limits on `/appeal` than on `/submit`** prevent an attacker who knows many
  `content_id`s from flooding the review queue with bogus appeals.
- **Keying:** Flask-Limiter's default `get_remote_address` (per-IP) is acceptable for
  this project. A real deployment would key on authenticated `creator_id`, but per-IP
  is the documented assignment default and is sufficient for grading.

### Storage

In-memory (`storage_uri="memory://"`) — the assignment's recommended setup. Limiter state
is not durable across restarts, which is fine for a project/demo and matches the
assignment's note. A real deployment would use Redis.

### When a limit triggers

The endpoint returns HTTP `429 Too Many Requests`. We **do not** log rate-limit
rejections to the audit log — the audit log is for *content decisions*, not transport
events. (Rate-limit observability lives in Flask's logs and the README's documented
evidence.)

## 7. Stretch Features

Per the assignment, these are optional (+4 stretch points max). I will commit to
**Ensemble detection** as the primary stretch goal because it is the most directly
additive to the core pipeline — it strengthens an existing required feature rather than
adding a disconnected new one. The other stretch features are recorded here as
"considered, deferred" so the README can note them honestly without overpromising.

### Primary stretch: Ensemble detection (3+ signals)

**What it is:** Extend the detection pipeline from 2 signals to 3+, with a documented
weighting or voting scheme.

**Specific plan:** Add a third signal — **repetition / n-gram regularity** — to the
existing stylometric and LLM signals. Concretely:

- **Signal 3: N-gram repetition regularity.** Compute the rate of repeated
  trigrams (3-word sequences) across the text. AI output tends to repeat
  transitional phrases ("It is important to note," "Furthermore," "In conclusion")
  at a higher rate than human writing, which reuses phrases more opportunistically.
  Pure Python, deterministic, captures a property the other two signals miss
  (phrase-level repetition, not just word-level diversity or sentence variance).
- **New weighting:** `combined = 0.5 * llm + 0.3 * stylometric + 0.2 * ngram_repetition`.
  The LLM keeps the highest weight because it remains the only semantic signal;
  the two structural signals share the remaining weight.
- **Threshold recalibration:** With three signals, the 0.75 / 0.35 boundaries may need
  adjustment. I will re-run the validation set from the Confidence Scoring section
  and re-pick thresholds that preserve the false-positive asymmetry.

**Update rule:** I will update this section with the actual numbers after implementing
and validating. The README will mirror whatever ends up shipping.

### Considered, deferred

- **Provenance certificate:** deferred. Even though Gradio is now the required
  creator/reviewer surface (see Section "UI & Testing Strategy"), it is a
  classification + appeal surface, not a content-publishing surface — there is no
  credential to display on third-party content within this project's scope.
- **Analytics dashboard:** requires either a second endpoint (`GET /stats`) or a static
  page. Could be added cheaply after the core features, but is deprioritized below
  ensemble detection.
- **Multi-modal support:** would require non-text input handling, which changes the
  API contract established in M3. Out of scope for the project's time budget.

## 8. Data Model & Storage

**Storage: SQLite (decided).** Earlier this read "SQLite/JSON" — we commit to SQLite. A
flat JSON file has no concurrency safety, and the appeal flow does a lookup-then-write that
would race under concurrent requests; SQLite gives us atomic writes and transactions for
free with no extra dependency (`sqlite3` ships with Python).

**The audit log is append-only.** Rows are never mutated. "Changing a status" means
appending a new event; the *current* status of a `content_id` is the status of its most
recent event. This preserves a tamper-evident decision history, which is the whole point
of a provenance/accountability system.

**`content_id`** is a server-generated UUID (`uuid4`) returned in the `/submit` response.
The client never supplies it. The same text submitted twice yields two distinct
`content_id`s — we do not dedupe, since identical text from different creators is a
legitimate case.

**Audit event schema** (one table, one row per event):

| Field           | Notes                                                                 |
|-----------------|-----------------------------------------------------------------------|
| `event_id`      | UUID, primary key                                                     |
| `event_type`    | `classification` \| `appeal` \| `resolution`                          |
| `content_id`    | UUID; groups all events for one piece of content                      |
| `creator_id`    | who submitted (classification) / who appealed (appeal); null for admin-only resolution if no reviewer identity is supplied |
| `text`          | original submitted text; stored only on the `classification` event    |
| `signals`       | JSON for classification events: raw scores, rationale, stylometric sub-metrics, `low_data`; null otherwise |
| `combined_score`| float [0,1] for classification events; neutral `0.5` in degraded mode; null for appeal/resolution events |
| `label`         | label text + category + confidence band; corrected label on overturned resolutions when supplied |
| `status`        | `classified` \| `under_review` \| `appeal_upheld` \| `appeal_overturned` |
| `degraded`      | JSON: `{value: bool, reason: str|null, fallback_signal: str|null, fallback_score: float|null}` |
| `payload`       | JSON event-specific details: `creator_reasoning` for appeals; `resolution_decision`, `corrected_label`, and optional reviewer notes for resolutions |
| `links_to`      | `event_id` of the event this one references (appeal→classification, resolution→appeal) |
| `timestamp`     | ISO-8601 UTC                                                          |

`GET /log` returns these events. The current state of a content item is derived by taking
its latest event by timestamp; the appeal queue is the set of `content_id`s whose latest
status is `under_review`.

## 9. Resilience — the Groq dependency

Every `/submit` blocks on a third-party, non-deterministic LLM call, so it is the most
likely point of failure and gets explicit handling. All of the failure-handling choices
below are applications of the Section 1 "fail safe, not fail open" principle — when a
signal is unavailable, the system defaults to *Uncertain* (the label that does the
least harm to a wrongly-accused creator), never to an AI or Human verdict.

- **Timeout:** the Groq call has a bounded timeout (≈10s). On expiry → degraded mode.
- **Retry:** one retry on transient errors (5xx / rate-limit) with a short backoff, then
  degraded mode. We do not retry indefinitely — a slow `/submit` is worse than an honest
  "uncertain."
- **Unparseable response:** the model is asked for JSON but can return prose or
  markdown-fenced JSON. We strip code fences and attempt a tolerant JSON parse; on failure
  → degraded mode. We never let a `json.loads` exception become a 500.
- **Degraded outcome** is defined in the Confidence Scoring section: available raw
  signal(s) are stored, decision-level `combined_score` is set to neutral `0.5`, the
  public label is forced to Uncertain, and the structured `degraded` object records the
  reason and fallback score.

## Architecture

```
                         SUBMISSION FLOW
 ┌──────────┐   POST /submit   ┌────────────────────┐
 │  Client  │ ───{text,        │   Flask app         │
 │ (creator)│     creator_id}─>│   /submit route     │
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
                                │ Confidence Scoring      │
                                │ combined = 0.6*llm +    │
                                │            0.4*style    │
                                └───────────┬─────────────┘
                                            │ combined_ai_probability
                                            ▼
                                ┌───────────────────────┐
                                │ Label Selection         │
                                │ (AI / Uncertain / Human)│
                                └───────────┬─────────────┘
                                            │ label text
                          ┌─────────────────┼─────────────────┐
                          ▼                                   ▼
                ┌───────────────────┐                ┌──────────────────┐
                │ Audit Log (append) │                │ JSON response to │
                │ SQLite append-only  │                │ client: content_id,
                │ content_id, scores, │                │ attribution,      │
                │ label, status       │                │ confidence, label │
                └───────────────────┘                └──────────────────┘


                            APPEAL FLOW
 ┌──────────┐  POST /appeal   ┌────────────────────┐
 │  Client  │ ─{content_id,   │  Flask app          │
 │ (creator)│   creator_id,   │  /appeal route      │
 │          │   reasoning}───>│                     │
 └──────────┘                 └─────────┬───────────┘
                                         │ lookup content_id
                                         ▼
                              ┌──────────────────────┐
                              │ Append appeal event:    │
                              │ status -> under_review  │
                              └──────────┬─────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │ Audit Log (append      │
                              │ appeal entry: reasoning,│
                              │ timestamp, link to       │
                              │ original decision)       │
                              └──────────┬─────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │ JSON response:          │
                              │ {content_id,            │
                              │  status: under_review,  │
                              │  appeal_logged: true}   │
                              └──────────────────────┘
```

**Submission flow narrative:** A creator's text enters through `/submit`, is scored
independently by the LLM signal and the stylometric signal, those two scores are combined
into a single confidence value, the confidence value is mapped to one of three label
texts, and the full decision (both raw signal scores, combined score, label, content_id)
is written to the audit log before the response is returned to the client.

**Appeal flow narrative:** A creator submits a `content_id`, `creator_id`, and reasoning
to `/appeal`; the system looks up the original decision, verifies ownership, and appends
a linked audit log entry whose status is `under_review` — no existing row is mutated and
no re-classification happens automatically. The appeal simply surfaces the case for human
review.

## AI Tool Plan

This is a prompting strategy, not a second source of truth. The canonical requirements
are Sections 1–9 plus the milestone implementation sections below. For each milestone,
provide only the relevant spec sections to the AI tool, review generated code against the
milestone checklist, and edit before using.

- **M3:** provide Signal 1, Resilience, Architecture, and Data Model.
- **M4:** provide Signal 2 formulas, Uncertainty Representation, and Data Model.
- **M5:** provide Transparency Labels, Appeals, Rate Limiting, Data Model, and full UI
  implementation expectations (Section "UI & Testing Strategy" + the locked visual
  tokens). When prompting an AI tool for `ui.py`, include the locked palette, type
  scale, layout, signature element, and copy rules — do not let the tool produce
  default Gradio output.
- **Stretch:** update Detection Signals and Confidence Scoring in this plan first, then
  implement the third signal.

## UI & Testing Strategy

**Flask is the backend API. Gradio is the required user surface.** Both are part of the
deliverable. Flask serves JSON at `/submit`, `/appeal`, `/resolve`, and `/log`. Gradio,
in `ui.py`, is the surface a creator or reviewer actually uses — it is not optional and
not a demo afterthought. The UI calls Flask over HTTP and never imports detection
modules, so the API remains the single source of truth and the UI cannot drift from it.

### Required tabs in `ui.py`

1. **Submit** — `<textarea>` for `text`, `<textbox>` for `creator_id`, Submit button.
   Calls `POST /submit`. Result is rendered in the right-hand verdict panel. The
   `content_id` from the response is auto-populated into the Appeal tab's `content_id`
   field so a user does not have to copy it across tabs.
2. **Appeal** — `<textbox>` for `content_id` (auto-filled from Submit), `<textbox>` for
   `creator_id` (shared state with Submit), `<multiline>` for `creator_reasoning`,
   File appeal button. Calls `POST /appeal`.
3. **Resolve (admin)** — `<textbox>` for `content_id`, `<dropdown>` for `decision`
   (`appeal_upheld` | `appeal_overturned`), optional `<textbox>` for `corrected_label`,
   optional `<textbox>` for `reviewer_notes`, Resolve button. Calls `POST /resolve`.
4. **Log** — Refresh button calls `GET /log`. Renders the audit log as a structured
   table or JSON. Includes a checkbox filter "Show only items currently under_review"
   that filters client-side from the latest event per `content_id` (no new Flask
   endpoint is introduced for this).

### Display modes

A radio at the top of the Submit tab toggles:

- **Creator view** — verdict card shows the label text + confidence band + a one-line
  plain-language explanation of what the band means and its known bias disclaimer. No
  raw scores.
- **Auditor view** — verdict card shows everything: combined score, both raw signal
  scores, stylometric sub-metrics, `low_data` flag, the structured `degraded` object,
  and timestamps.

The Appeal, Resolve, and Log tabs default to Auditor view because they are reviewer/
admin surfaces.

### `ui.py` rules

- Reads `FLASK_BASE_URL` env var (default `http://localhost:5000`).
- Performs a `GET /log` health check on startup and shows a visible banner indicating
  whether Flask responded. If Flask is unreachable, the UI shows a clear startup error
  with the command to start Flask (`python app.py`) — it does not silently fall back to
  direct imports.
- Every Flask call is wrapped in try/except; failures render as a Gradio-formatted
  error (verdict panel turns red, text starts with `"Flask error:"`) rather than
  crashing the page.
- **Visual tokens are applied via `gr.Blocks(css=...)`.** Gradio's default theming
  cannot deliver the locked palette, type scale, verdict band, focus rings, or
  the 240ms verdict-band-draw motion natively. `ui.py` therefore injects a custom
  stylesheet that defines CSS custom properties for the palette tokens (e.g.
  `--bg: #F2F4F7`, `--verdict-ai: #B23A48`), loads `@font-face` rules for Source
  Serif 4 and Inter from `assets/fonts/`, applies the verdict-band signature
  element as a `.verdict-card` class with `border-top: 3px solid var(--verdict-*)`,
  and uses a `@keyframes verdict-draw` rule for the 240ms left-to-right reveal.
  Reduced-motion is respected via a `@media (prefers-reduced-motion: reduce)`
  block that disables the keyframe. The stylesheet is a string passed to
  `gr.Blocks(css=...)` at construction — it is not a separate `.css` file
  referenced from a theme, because Gradio's theme mechanism does not expose
  per-component CSS.

### Visual & UX direction

This subsection pins the aesthetic so the implementer cannot drift into generic
templated output. The frontend-design skill is applied against these tokens during
implementation.

**Subject and audience.** Provenance Guard's UI is used by two kinds of people —
creators checking whether their writing has been flagged, and reviewers working through
an appeal queue. Both need to trust the system at a glance, so the visual language is
*forensic-document meets quiet editorial*: restrained, evidence-first, not
consumer-glossy.

**The one real aesthetic risk.** Type-driven identity. The page is remembered by its
headline scale and a single editorial display face used sparingly on the section
eyebrow and the verdict band, set against a quiet body face. This is a counter-direction
to the dominant "cream + terracotta serif" or "near-black + acid accent" defaults —
instead it leans cool, paper-like, and narrow-palette.

**Palette (locked hex values):**

| Token              | Value      | Use                                                                |
|--------------------|------------|--------------------------------------------------------------------|
| `--bg`             | `#F2F4F7`  | cool paper, off-white page background                              |
| `--surface`        | `#FFFFFF`  | card surface                                                       |
| `--ink`            | `#0F172A`  | primary text                                                       |
| `--muted`          | `#475569`  | secondary text                                                     |
| `--rule`           | `#CBD5E1`  | hairline dividers                                                  |
| `--accent`         | `#1F4FD9`  | active verdict band + focused input ring (cool editorial blue)     |
| `--verdict-ai`     | `#B23A48`  | AI verdict (left border accent + chip only, never full-bleed)      |
| `--verdict-uncertain` | `#8A6D3B` | Uncertain verdict (border accent + chip only)                    |
| `--verdict-human`  | `#2F6B4A`  | Human verdict (border accent + chip only)                          |

**Type (locked families and roles):**

- **Display** (eyebrow, verdict band): editorial serif with optical-size contrast.
  Primary: **Source Serif 4** (open-source via Adobe's GitHub release, no license
  required to bundle). Optional preferred upgrade — **GT Sectra** (Grilli Type,
  commercial license required; only use if a license is held and the font is
  self-hosted or served via a licensed CDN — do not hotlink Grilli's CDN).
  System fallback: `"Iowan Old Style", Georgia, serif`. Display weight 600.
- **Body** (everything else): **Inter** at 15px / 1.55 line-height. Fallback:
  `-apple-system, "Segoe UI", Roboto, sans-serif`. Weight 400; labels 500 uppercase
  tracked +0.04em.
- **Mono** (`content_id`, timestamps, raw scores in Auditor view): `"JetBrains Mono",
  ui-monospace, "SF Mono", Menlo, monospace`.
- **Type scale:** 13 / 15 / 17 / 22 / 32 / 48.
- **Implementation note:** Source Serif 4 and Inter are bundled via `ui.py`'s
  custom CSS injection (see "`ui.py` rules" below) so the UI does not depend on
  the user's local fonts. Source Serif 4 `.woff2` files go in `assets/fonts/`
  and are referenced by `@font-face` in the injected stylesheet.

**Layout (locked):** two-column Gradio `Blocks`. Left column holds the four primary tabs
stacked in sequence (Submit → Appeal → Resolve → Log). Right column is a sticky
"Verdict panel" that always shows the most recent classification result. Tabs use a
numbered marker (01 / 02 / 03 / 04) **because the order carries information** — this
is a sequence of operations, not decorative numbering. On mobile the right panel drops
below the tabs.

**Signature element — the verdict band.** A 3px-tall horizontal rule at the top of
every verdict card, colored with the semantic verdict color, paired with a one-word
uppercase chip ("AI" / "UNCERTAIN" / "HUMAN") set in the display face. This is what
the page is remembered by; it carries the whole decision in one glance and is what
distinguishes the design from generic Gradio defaults.

**Motion.** One orchestrated moment: when a new verdict arrives, the verdict band
draws left-to-right in 240ms, with the chip fading in 80ms after. No other animation.
`prefers-reduced-motion: reduce` disables it.

**Accessibility floor.** Visible 2px `--accent` focus rings on every input. Body text
≥ 4.5:1 contrast against `--bg`. Verdict chips ≥ 4.5:1 contrast against their
background. Logical tab order. Reduced-motion respected.

**Voice and copy rules for UI text.** Active voice, sentence case, named after what
the user does ("Submit text", "File appeal", "Resolve case") — never after endpoints
or internal concepts. The verdict-card subtitle in Creator view is one sentence of
plain-language explanation of the verdict band and its known bias disclaimer, written
to the reader, not to the system.

### Testing surfaces

1. **Function smoke tests:** call `get_llm_signal`, `get_stylometric_signal`,
   `combine_scores`, and `get_label` directly during each milestone.
2. **API tests with curl:** use the documented curl commands for `/submit`, `/appeal`,
   `/resolve`, and `/log`; capture representative JSON for the README.
3. **Rate-limit test:** send 12 rapid `/submit` requests and confirm 10 return `200` and
   2 return `429`.
4. **Gradio manual smoke (required):** run Flask, then run `ui.py`, open the local
   URL, and exercise every tab end-to-end during the M5 checkpoint — Submit, Appeal,
   Resolve, Log refresh, Creator/Auditor toggle, "under_review only" filter. Every
   tab must be exercised, not just Submit.
5. **Gradio automated smoke (stretch):** `tests/test_ui.py` uses the
   [`gradio_client`](https://pypi.org/project/gradio-client/) library to connect to
   a running `ui.py` over its local HTTP endpoint and call each tab's handler
   function directly — `predict(submit_text="…", creator_id="…")`,
   `predict(appeal_content_id="…", …)`, `predict(resolve_content_id="…", …)`,
   `predict(log_refresh=…)`. Assertions: Submit returns a verdict with the
   expected label band for one M4 validation input; Appeal returns a
   `under_review` status; Resolve returns an updated status; Log returns a
   list containing the new event. The test launches Flask + Gradio in
   subprocesses (`subprocess.Popen` with `start_new_session=True`), waits for
   the Gradio URL to respond, runs the assertions, then terminates both. This
   is *not* a replacement for the manual smoke in #4 — it is a CI-friendly
   guard so the UI does not silently regress between the manual runs.

## 10. Milestone 3 — Implementation Steps

The sections above define the prompting strategy and testing surfaces. This section is
the step-by-step recipe for executing Milestone 3 specifically — the first end-to-end
runnable slice of the system.

### Module Layout (decided)

So the build steps below have somewhere to put their output, the project is split
across these files:

```
ai201-project4-provenance-guard/
├── app.py              ← Flask API + all route handlers (/submit, /appeal,
│                          /log, /resolve) + Flask-Limiter wiring
├── config.py           ← constants: GROQ_API_KEY, LLM_MODEL,
│                          VALID_LABEL_CATEGORIES, RATE_LIMITS, AUDIT_DB_PATH,
│                          LABELS (label-text dict from Section 3)
├── signal_llm.py       ← Signal 1: get_llm_signal(text) -> {ai_probability, rationale}
│                          (includes the Groq prompt template + retry/parse logic
│                          from Section 9)
├── signal_stylometry.py← Signal 2: get_stylometric_signal(text) ->
│                          {sentence_var_score, ttr_score, punctuation_score,
│                          stylometric_score, low_data}  [M4]
├── combine.py          ← combine_scores(llm, style, [ngram]) -> float
│                          + get_label(combined_score) -> {category, text, band}
├── audit.py            ← SQLite schema bootstrap + append_event(...)
│                          + get_log() + get_appeal_queue() (filters by status)
├── ui.py               ← required Gradio UI that calls the Flask API over HTTP
│                          (no detection-module imports); see "UI & Testing Strategy"
│                          for tabs, display modes, and visual tokens [M5]
├── requirements.txt    ← flask, flask-limiter, groq, python-dotenv, gradio
└── data/
    └── labels.json     ← label-text dict (M5), editable without touching code
```

Each function named in this planning doc lives in exactly one file above, named in
the comment. The file split matches the milestone split — M3 only writes
`config.py`, `signal_llm.py`, `app.py`, and `audit.py`; `signal_stylometry.py` and
`combine.py` are added in M4; labels, rate limiting, appeals/resolution, and the
required `ui.py` are completed in M5.

**Note on log entry shape:** The starter example in the M3 instructions shows
`"attribution": "likely_ai"` as the stored field. Provenance Guard's commitment in
Section 2 is to *not* round to a categorical label internally — the raw
`combined_score` is what gets stored, and the label text is derived. The audit log
entry therefore stores both `combined_score` (float) and `label` (the user-facing
text + category + band), not a single `attribution` category. This is consistent
with Section 8's schema and preserves the "no false precision" principle.

### Step 1 — Prompt the AI tool with spec sections

Before writing any code, use this `planning.md` + the architecture diagram to prompt
an AI tool. Give it the **Detection Signals** section (Signal 1 only) and the diagram
and ask it to generate:
  1. The Flask app skeleton with the `POST /submit` route stub, and
  2. The first signal function (`get_llm_signal(text) -> dict`).

**Review checklist before pasting the generated code:**
- Function signature matches the spec's description of what the signal returns (a
  score, not a binary flag — see Section 1's "Output shape").
- Flask route structure matches the API contract in Section 8 (`content_id` server-
  generated, JSON request body, JSON response).
- The LLM call uses the timeout + single-retry + tolerant-JSON-parse handling from
  Section 9, not a bare `client.chat.completions.create(...)` with no safeguards.
- Edit before using; don't paste blindly.

### Step 2 — Flask skeleton + `POST /submit` stub

Set up the Flask app. Create a `POST /submit` endpoint that accepts a JSON body with
at minimum a `text` field and a `creator_id` field. For the first cut, have it
return a hardcoded response so the route is verified before any detection logic is
added.

### Step 3 — First detection signal (Signal 1)

Implement the LLM-judgment signal from Section 1. Call Groq with a prompt that asks
for a structured `{ai_probability, rationale}` response (JSON output, see Section 9
parsing rules). **Test it independently before wiring it into the endpoint** —
call `get_llm_signal(text)` directly with 2–3 test inputs (one clearly AI, one clearly
human, one borderline) and inspect each output for:
- `ai_probability` is a float in [0, 1]
- `rationale` is a non-empty string
- A deliberately bad response (e.g. model returns prose) degrades rather than 500s

### Step 4 — Wire Signal 1 into `/submit`

The endpoint now returns a response with at least:
- `content_id` — server-generated UUID (`uuid4`), essential because the appeal
  endpoint needs it
- `attribution` — response-only convenience field for the signal 1 result; normally the
  `ai_probability` float, or `null` if Signal 1 is unavailable in degraded mode
- `confidence` — placeholder for now (will become the decision-level combined score in M4)
- `label` — placeholder for now (will become the real label text in M5)

**Canonical response shape (locked here, applies to all milestones):** The `/submit`
response always carries the four fields above. `attribution` maps to
`signals.llm_ai_probability` when Signal 1 succeeds and is `null` when Signal 1 is
unavailable. `confidence` is the decision-level combined score (float in `[0, 1]`, neutral
`0.5` in degraded mode), and `label` is the derived user-facing text + category + band.
Internally the audit log stores `combined_score`, `label`, full `signals`, and the
structured `degraded` object per Section 8.

```bash
curl -s -X POST http://localhost:5000/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "The sun dipped below the horizon, painting the sky in hues of amber and rose. I sat on the porch, coffee in hand, watching the neighborhood slowly go quiet.", "creator_id": "test-user-1"}' \
  | python -m json.tool
```

You should see a JSON response with `content_id`, `attribution`, `confidence`, and
`label` fields. Save the `content_id` — it will be used to test appeals in Milestone 5.

### Step 5 — Audit log write on every submission

Before moving on, every call to `/submit` writes a structured entry to the log —
timestamp, content_id, creator_id, original text, signal 1 score, and degraded state.
The M3 entry shape:

```json
{
  "event_id": "<uuid>",
  "event_type": "classification",
  "content_id": "<uuid>",
  "creator_id": "test-user-1",
  "timestamp": "2025-04-01T14:32:10.123Z",
  "text": "<original submitted text>",
  "signals": {
    "llm_ai_probability": 0.81,
    "rationale": "Uniform phrasing and generic transitions."
  },
  "combined_score": 0.81,
  "label": {
    "category": "pending_label_generation",
    "text": "Label generation pending Milestone 5.",
    "band": null
  },
  "status": "classified",
  "degraded": {
    "value": false,
    "reason": null,
    "fallback_signal": null,
    "fallback_score": null
  },
  "payload": {},
  "links_to": null
}
```

This is the M3-minimum subset of the Section 8 schema — `signals` contains only the
LLM fields until M4 adds stylometry, and `payload` remains empty until M5 adds appeals
and resolutions. Start simple but use **SQLite per Section 8's decision** — the
concurrency-safety reason holds even in M3, and switching storage backends later would
invalidate the audit log across restarts. Do not use `print()` statements as the log.

### Step 6 — `GET /log` endpoint

Add a `GET /log` endpoint that returns the most recent audit log entries as JSON.
The project requires showing the audit log with at least 3 structured entries —
without a `/log` endpoint (or equivalent), there is no clean way to surface this in
the README. Keep it simple:

```python
return jsonify({"entries": get_log()})
```

In a real system this would require auth; here it is for documentation and grading
visibility. **Rate limiting on `/log` (Section 6: 60/minute) is a forward reference
for M5** — Flask-Limiter wiring is introduced in Section 13 / Step 5, not here. In
M3 the endpoint just needs to return the entries; the `@limiter.limit(...)`
decorator is added when the limiter is wired up in M5.

### Checkpoint

Flask app runs. `POST /submit` returns a JSON response including `content_id`,
attribution result, and a placeholder confidence score. Each submission writes a
structured entry to the audit log. `GET /log` returns those entries as JSON. The
log can be inspected and shows the test submissions.

## 11. Out of Scope / Known Limitations

This project is a graded demo, not a production system. The following items are
**deliberately not solved here**, named explicitly so a reviewer (or future
maintainer) reads them as scoped-out choices rather than as oversights. Each one
would matter in a real deployment but doesn't change the assignment's grading
criteria.

1. **Submission-time `creator_id` is not authenticated.** A client can submit text
   claiming to be any `creator_id` they choose — there is no login, OAuth, or
   identity provider in scope. The ownership check in Section 4 prevents
   *cross-impersonation on appeal* (you can't appeal someone else's content),
   but it does not prevent *impersonation at submission time*. A real
   deployment would gate `/submit` behind authenticated identity. The
   `creator_id` field here is a per-submission self-asserted label, used only
   for the appeal-ownership check and for grouping a creator's submissions in
   the audit log.

2. **Stylometric scoring is heuristic, not calibrated.** The implementation handles
   empty, very short, and punctuation-free inputs by returning neutral `0.5` sub-scores
   with `low_data: true`, and wraps unexpected exceptions in the same degraded-mode path
   used for LLM failures. That prevents crashes, but it does not make the heuristic a
   reliable detector for unusual writing styles. A real deployment would need labeled
   calibration data and bias testing before trusting these scores.

3. **Original text is stored indefinitely in the audit log.** Section 8 stores
   `text` on every `classification` event, which means a submitted text is
   retained for the lifetime of the database. There is no deletion endpoint, no
   retention window, and no "right to erasure" handling. For a real system with
   user-submitted content, this would need a documented retention policy and a
   redaction mechanism. Out of scope for the demo; called out here so the
   choice is not silent.

4. **Per-IP rate-limit keying (Section 6) punishes shared NATs.** A classroom,
   office, or VPN where many users share one IP will see one user's flood shut
   out everyone else on that egress. Per-IP is the assignment's documented
   default and is sufficient for grading; a real deployment would key on
   authenticated `creator_id` (per-item 1 above, this requires auth first).

5. **The Uncertain label omits the band in reader-facing text.** The API still returns
   `label.band` for every classification because it is useful for debugging and audit
   review, but the Uncertain label text does not interpolate `{band}`. A label that
   already says "we can't tell" should not sound more precise than it is.

## 12. Milestone 4 — Implementation Steps

Where Section 10 covered M3 (one signal, hardcoded responses, audit-log write),
this section covers M4: the second signal, real confidence scoring, and the first
end-to-end run of the detection pipeline. The /submit response still uses
placeholder text for `label` (real label text comes in M5), but the *score* is
now real.

### Step 1 — Prompt the AI tool with spec sections

Use this `planning.md` (Sections 1, 2, 8) + the architecture diagram to prompt
an AI tool. Give it the **Detection Signals** section (Signal 2 only),
**Uncertainty Representation**, and the diagram, and ask it to generate:
  1. The second signal function: `get_stylometric_signal(text) -> dict`
     implementing the three sub-metrics from Section 1 (sentence length variance,
     type-token ratio, punctuation regularity), combined with equal weights into
     `stylometric_score` in [0, 1].
  2. The confidence scoring logic: `combine_scores(llm_score, stylometric_score)
     -> float` implementing the `0.6 * llm + 0.4 * style` weighting from Section 1.

**Review checklist before pasting:**
- `combine_scores` returns a float, not a label or category — labels come from
  `get_label()` (Section 3 / M5), not from the scoring function.
- The thresholds `0.75 / 0.35` and the weights `0.6 / 0.4` match Section 2
  *exactly*. AI tools sometimes invent reasonable-looking thresholds that
  silently diverge from spec — correct any mismatch before integrating.
- The signal-2 function returns the three sub-metrics as separate fields plus
  the combined `stylometric_score`, not just the combined score (Section 1's
  "Output shape").
- Edit before using; don't paste blindly.

### Step 2 — Implement Signal 2 standalone

Implement `get_stylometric_signal(text) -> dict` in `signal_stylometry.py`
(per Section 10's module layout). **Test it independently before integration** —
call the function directly with the same inputs you used for Signal 1 and check:
- All three sub-scores are floats in [0, 1].
- `stylometric_score` is the mean of the three sub-scores (equal weights).
- The function does not crash on empty input, very short input, or input with
  no punctuation.

### Step 3 — Implement confidence scoring

Implement `combine_scores(llm_score, stylometric_score) -> float` in `combine.py`.
Replace the placeholder `confidence` in `/submit` with the real combined score.
The response still carries the four-field shape from Section 10 Step 4
(`content_id`, `attribution` = raw signal 1 score, `confidence` = combined score,
`label` = placeholder text until M5).

### Step 4 — Validate scoring with 4 deliberately chosen inputs

The M4 checkpoint requires "at least 4 inputs spanning the confidence range."
Use the literal validation set below — copy these strings directly into
curl/pytest calls. Each pair is annotated with what score *should* land where,
so a miscalibrated pipeline shows up immediately.

**Input 1 — clearly AI-generated (expect HIGH combined score, ideally >= 0.75):**
```text
Artificial intelligence represents a transformative paradigm shift in modern society.
It is important to note that while the benefits of AI are numerous, it is equally
essential to consider the ethical implications. Furthermore, stakeholders across
various sectors must collaborate to ensure responsible deployment.
```

**Input 2 — clearly human-written (expect LOW combined score, ideally < 0.35):**
```text
ok so i finally tried that new ramen place downtown and honestly?
underwhelming. the broth was fine but they put WAY too much sodium in it and
i was thirsty for like three hours after. my friend got the spicy version and
said it was better. probably won't go back unless someone drags me there
```

**Input 3 — borderline: formal human writing (likely mid-to-high stylometric,
LLM signal carries the verdict):**
```text
The relationship between monetary policy and asset price inflation has been
extensively studied in the literature. Central banks face a fundamental tension
between their mandate for price stability and the unintended consequences of
prolonged low interest rates on equity and real estate valuations.
```

**Input 4 — borderline: lightly edited AI output (should land in the Uncertain
band — neither signal can confidently accuse or clear):**
```text
I've been thinking a lot about remote work lately. There are genuine tradeoffs —
flexibility and no commute on one side, isolation and blurred work-life boundaries
on the other. Studies show productivity varies widely by individual and role type.
```

**If any score doesn't match intuition:** print both signal scores separately
and find which one is misbehaving. The LLM signal is non-deterministic — re-run
2–3 times before deciding it's wrong. The stylometric signal is deterministic —
if it's wrong, it's wrong.

### Step 5 — Update audit log to capture both signals

Extend the audit-log write in `audit.py` so each `classification` event stores
both individual signal scores alongside the combined score. The Section 8
schema already has space for this (`signals` field, JSON blob). Minimum M4
shape:

```json
{
  "event_id": "<uuid>",
  "event_type": "classification",
  "content_id": "<uuid>",
  "creator_id": "test-user-1",
  "timestamp": "2025-04-01T14:32:10.123Z",
  "text": "<original submitted text>",
  "signals": {
    "llm_ai_probability": 0.81,
    "stylometric_score": 0.62,
    "stylometric_submetrics": {
      "sentence_var_score": 0.55,
      "ttr_score": 0.71,
      "punctuation_score": 0.60
    },
    "low_data": false,
    "rationale": "Uniform sentence structure; few first-person markers."
  },
  "combined_score": 0.73,
  "label": { "category": "likely_ai", "text": "...", "band": "Low" },
  "status": "classified",
  "degraded": {
    "value": false,
    "reason": null,
    "fallback_signal": null,
    "fallback_score": null
  },
  "payload": {},
  "links_to": null
}
```

`label.text` is still the M3 placeholder text in M4 — M5 swaps it for the real
Section 3 variants.

### Checkpoint

Both detection signals are running and their outputs are combined into a single
confidence score. Submitting clearly AI-generated text produces a noticeably
different score than clearly human-written text. The audit log now records
individual signal scores and the combined result. At least 4 inputs have been
tested spanning the confidence range.

## 13. Milestone 5 — Implementation Steps

Where M3 wired the skeleton and M4 made the score real, M5 turns the system
into something that behaves like a real product: user-facing labels that vary
by score, an appeals process, rate limiting, and a complete audit log. The
transparency label and appeals workflow both depend on the confidence scoring
from M4 — verify that works before starting M5.

### Step 1 — Prompt the AI tool with spec sections

Use this `planning.md` (Sections 3, 4, 6, 8) + the architecture diagram to
prompt an AI tool. Ask it to generate:
  1. A label generation function `get_label(combined_score) -> dict` mapping
     scores to the three label-text variants from Section 3, with the band
     computed from distance-to-nearest-threshold.
  2. The `POST /appeal` endpoint that updates status to `under_review` and
     appends an `appeal` event to the audit log.
  3. The `POST /resolve` admin endpoint that transitions `under_review` to
     `appeal_upheld` or `appeal_overturned`.
  4. Flask-Limiter wiring with the exact limits from Section 6.

**Review checklist before pasting:**
- `get_label()` produces all three label-text variants from Section 3 *verbatim*
  — ask it to print all three and confirm against the spec table.
- The `band` field uses the Section 3 cutoffs (`>= 0.20` / `0.10–0.19` / `< 0.10`
  distance from the nearest threshold), not invented cutoffs.
- `/appeal` returns `403` (not `404` or `400`) when the `creator_id` doesn't
  match the original submission — Section 4's ownership check.
- Rate limits match Section 6 exactly: `/submit` 10/min + 100/day, `/appeal`
  5/min + 20/day, `/log` 60/min, `/resolve` 30/min.
- Flask-Limiter is initialized with `storage_uri="memory://"` (Section 6 + the
  M5 setup note) — without it, you'll see a warning on startup.

### Step 2 — Transparency label variants

Replace the M3/M4 placeholder label text with the three variants from Section 3.
The label returned by `/submit` now changes based on the combined score — different
score bands must produce the corresponding different label text.

**Verification:** run the four M4 test inputs through `/submit` and confirm the
responses use the appropriate label variant for their score bands. Do not require all
four responses to have unique text — there are only three label categories. At minimum,
confirm all three variants are reachable by submitting inputs whose combined scores land
in each of the three bands.

### Step 3 — Appeals workflow

Build `POST /appeal` in `app.py`. The endpoint accepts `content_id`, `creator_id`, and
`creator_reasoning` (Section 4). On receipt:
1. Look up the original audit log entry for `content_id`.
2. Verify the request's `creator_id` matches the submission's `creator_id`
   (else `403`, no state change).
3. If the latest status for this `content_id` is already `under_review`, return `409`
   to prevent duplicate active appeals.
4. Append a new `appeal` event with `links_to` set to the original `event_id`,
   `status: under_review`, and `payload.creator_reasoning` populated.
5. Return `{content_id, status: "under_review", appeal_logged: true}`.

Do **not** implement automated re-classification. A human reviewer is expected
to look at content with `status: under_review` and resolve via `POST /resolve`.

**Test:**

```bash
curl -s -X POST http://localhost:5000/appeal \
  -H "Content-Type: application/json" \
  -d '{"content_id": "PASTE-CONTENT-ID-HERE",
       "creator_id": "test-user-1",
       "creator_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical."}' \
  | python -m json.tool
```

Then `GET /log` and verify the appeal entry shows `status: "under_review"` and
`payload.creator_reasoning` populated.

### Step 4 — Resolution endpoint

Build `POST /resolve` (admin action). Accepts `content_id`, `decision`
(`"appeal_upheld"` or `"appeal_overturned"`), optional `corrected_label`, and optional
`reviewer_notes`. It only resolves content whose latest status is `under_review`; if no
active appeal exists, return `409`. Append a `resolution` event with `links_to` set to
the appeal event's `event_id` and resolution details stored in `payload`. Without this
endpoint, an appeal can never close — which is not an appeals process. This is named in
Section 4 and is the M5 demo for the reviewer side.

### Step 5 — Rate limiting

Wire Flask-Limiter with `storage_uri="memory://"` and apply the limits from
Section 6:

```python
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

@app.route("/submit",   methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit(): ...

@app.route("/appeal",   methods=["POST"])
@limiter.limit("5 per minute;20 per day")
def appeal(): ...

@app.route("/log",      methods=["GET"])
@limiter.limit("60 per minute")
def log(): ...

@app.route("/resolve",  methods=["POST"])
@limiter.limit("30 per minute")
def resolve(): ...
```

**Test:** hammer `/submit` with 12 rapid requests and confirm 10 succeed (200)
and 2 are rejected (429). Capture the status-code output for the README — that's
the grader-facing evidence:

```bash
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:5000/submit \
    -H "Content-Type: application/json" \
    -d '{"text": "This is a test submission for rate limit testing purposes only.", "creator_id": "ratelimit-test"}'
done
```

**Grader-facing evidence list.** The README must include all of the following, not
just some of them, because each one corresponds to a required capability:

1. Curl examples and JSON responses for `/submit`, `/appeal`, `/resolve`, and `/log`
   (from Step 2 of this milestone — re-run them at the M5 checkpoint with the
   final schema).
2. The 12-request rate-limit burst output above (10 × `200`, 2 × `429`).
3. A Gradio UI screenshot of the Submit tab showing one of the four M4 validation
   inputs producing its expected verdict band — e.g., Input 1 (the clearly-AI
   paragraph) rendering with the AI verdict band and label. The screenshot must
   show the verdict-band signature element, not just a generic Gradio panel.
4. A second Gradio UI screenshot showing the Creator/Auditor toggle on the same
   result, demonstrating that Auditor view exposes the raw signal scores and the
   Creator view hides them.
5. A third Gradio UI screenshot showing the Log tab after at least one Submit,
   one Appeal, and one Resolve, with the "under_review only" filter applied.

### Step 6 — Complete audit log

Verify the log captures everything required:
- `timestamp` (ISO-8601 UTC)
- `content_id`
- `creator_id`
- `signals.llm_ai_probability` (raw signal 1 score, on the `classification` event)
- `combined_score` (on the `classification` event)
- both individual signal scores (the full `signals` blob per Section 8)
- structured `degraded` object (`value`, `reason`, `fallback_signal`, `fallback_score`)
- appeal status and `payload.creator_reasoning` (when applicable, on the `appeal` event row)
- resolution decision and optional corrected label/notes in `payload` (when applicable,
  on the `resolution` event row)

Format is structured (SQLite per Section 8), not `print()` statements. Generate
at least 3 entries — minimum: 2 classifications + 1 appeal — to have something
documentable in the README.

### Step 7 — Required Gradio UI

Gradio is no longer optional. `ui.py` is the user-facing surface of this project —
the thing a creator or reviewer actually uses to drive every flow. It is a thin HTTP
client to Flask; it must not import `signal_llm`, `signal_stylometry`, `combine.py`, or
any other detection module. If the UI reimplements detection logic, the API and the UI
will drift, and one of them will be wrong.

**Required contents of `ui.py`:**

- A Gradio `Blocks` app with the four tabs specified in "UI & Testing Strategy":
  Submit, Appeal, Resolve, Log.
- The Submit tab includes the Creator/Auditor radio toggle; the other tabs default
  to Auditor view.
- Reads `FLASK_BASE_URL` from env (default `http://localhost:5000`).
- Performs a `GET /log` health check on startup and shows a visible banner indicating
  whether Flask responded. If Flask is unreachable, the UI surfaces a clear startup
  error with the command to start Flask (`python app.py`) — it does not silently
  fall back to direct imports.
- HTTP client is `requests`. Every Flask call is wrapped in try/except; failures are
  rendered as a Gradio-formatted error (verdict panel turns red, text starts with
  `"Flask error:"`) rather than crashing the page.
- On Submit, the response is rendered in the verdict panel and the `content_id` is
  auto-populated into the Appeal tab's `content_id` field. `creator_id` is shared
  state across Submit, Appeal, and Resolve.
- Resolve tab's decision dropdown only offers the two valid values
  (`appeal_upheld` | `appeal_overturned`). `corrected_label` and `reviewer_notes` are
  optional and only meaningfully applied on `appeal_overturned`.
- Log tab renders the audit log. The "under_review only" checkbox filters
  client-side from the latest event per `content_id` — no new Flask endpoint is
  introduced for this.

**Required visual tokens** (apply the frontend-design skill against these):
- Use the locked palette, type scale, and verdict-band signature element from the
  "Visual & UX direction" subsection. Do not fall back to Gradio defaults.
- Apply the voice/copy rules ("Submit text", "File appeal", "Resolve case", etc.).
- Inject the stylesheet via `gr.Blocks(css=...)` — the locked tokens are *not*
  expressible in Gradio's theme system, so any solution that only touches
  `gr.themes.Soft()` (or similar) is incomplete. The verdict-band motion
  (`@keyframes verdict-draw`, 240ms) and `prefers-reduced-motion: reduce`
  override must both be in the injected CSS.

**Step ordering:** this step is built last in M5, after rate limiting and the complete
audit log are working, because the UI depends on the audit-log schema being final.

**Manual test plan for this step:**
1. Start Flask: `python app.py`
2. In another terminal, start Gradio: `python ui.py`
3. Open the local Gradio URL.
4. Submit each of the four M4 validation inputs through the Submit tab and confirm the
   expected verdict band (AI / HUMAN / UNCERTAIN) in both Creator and Auditor views.
5. File an appeal for one of the AI verdicts and confirm the Log tab shows
   `status: under_review` after refresh.
6. Resolve that appeal with `appeal_upturned` + a `corrected_label` and confirm the
   Log tab's latest event for that `content_id` reflects the new status.
7. Toggle "under_review only" and confirm resolved cases disappear.

**Checkpoint addition:** the M5 checkpoint is not satisfied unless every tab in
`ui.py` is exercised end-to-end against a running Flask backend and produces the
expected behavior. The Creator/Auditor toggle must visibly change the Submit panel;
the verdict-band signature element must be visible on every result card. (The
automated `gradio_client` smoke in `tests/test_ui.py` is a stretch item — see
Testing surfaces #5 — and is not part of the hard checkpoint.)

### Checkpoint

All four production API features are working: the transparency label varies by
confidence level, appeals can be submitted and are reflected in the audit log,
rate limiting triggers when the limit is exceeded (10 succeed / 2 fail on the
12-request burst), and the audit log has at least 3 structured entries covering
submissions and at least one appeal. **AND the Gradio UI launches and drives every
flow (Submit, Appeal, Resolve, Log) end-to-end against a running Flask backend,
without bypassing Flask.** The Creator/Auditor toggle visibly changes the Submit
panel; the verdict-band signature element is visible on every result card; the
visual tokens from the "UI & UX direction" subsection are applied throughout
`ui.py`. All required API flows work end-to-end without workarounds, and they are
all reachable through the Gradio UI, not just through curl.
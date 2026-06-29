import re
import string
from typing import NamedTuple

# Thresholds for stylometric analysis
CV_THRESHOLD = 0.8
IRREGULAR_RATIO_THRESHOLD = 0.35
TTR_BASELINE = 0.55
TTR_RANGE = 0.25
LOW_DATA_GUARD = (3, 30)

# Regex for word tokens and sentence boundaries.
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_SENT_BOUNDARY_RE = re.compile(r"[.!?\n]")


class ScoreResult(NamedTuple):
    """Result of a stylometric component score calculation."""

    score: float
    is_low_data: bool


def _words(text: str) -> list[str]:
    """Extract lowercase word tokens from text."""
    return _WORD_RE.findall(text.lower())


def _sentences(text: str) -> list[str]:
    """Split text into sentences, dropping empty fragments."""
    raw = _SENT_BOUNDARY_RE.split(text)
    return [s.strip() for s in raw if s.strip()]


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(max_val, value))


def _sentence_var_score(sentences: list[str], words: list[str]) -> ScoreResult:
    """Score based on sentence length variation (coefficient of variation).

        Low variation suggests AI-
    like uniformity.
    """
    min_sentences, min_words = LOW_DATA_GUARD
    if len(sentences) < min_sentences or len(words) < min_words:
        return ScoreResult(0.5, True)
    lengths = [len(_words(s)) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    stdev = variance**0.5
    cv = stdev / mean_len if mean_len > 0 else 0.0
    score = _clamp(1 - (cv / CV_THRESHOLD), 0.0, 1.0)
    return ScoreResult(score, False)


def _ttr_score(words: list[str]) -> ScoreResult:
    """Score based on Type-Token Ratio (lexical diversity).

    Lower TTR (less diverse vocabulary) suggests AI generation.
    """
    if len(words) < 50:
        return ScoreResult(0.5, True)
    ttr = len(set(words)) / len(words)
    score = _clamp((TTR_BASELINE - ttr) / TTR_RANGE, 0.0, 1.0)
    return ScoreResult(score, False)


def _punctuation_score(text: str) -> ScoreResult:
    """Score based on irregular punctuation usage.

    Heavy use of irregular punctuation suggests human writing.
    """
    if len(text) == 0:
        return ScoreResult(0.5, True)
    irregular_set = set("!?;:\u2014\u2026()")
    _punct_set = set(string.punctuation) | irregular_set
    total_punct = sum(1 for c in text if c in _punct_set)
    if total_punct < 2:
        return ScoreResult(0.5, True)
    irregular = sum(1 for c in text if c in irregular_set)
    irregular_ratio = irregular / total_punct if total_punct > 0 else 0.0
    score = _clamp(1 - (irregular_ratio / IRREGULAR_RATIO_THRESHOLD), 0.0, 1.0)
    return ScoreResult(score, False)


def get_stylometric_signal(text: str) -> dict:
    """Return the locked output shape for the stylometric signal.

    Combines sentence variation, lexical diversity, and punctuation scores.
    Never raises; on any unexpected error returns degraded defaults.
    """
    try:
        words = _words(text)
        sentences = _sentences(text)
        sv_result = _sentence_var_score(sentences, words)
        ttr_result = _ttr_score(words)
        punct_result = _punctuation_score(text)
        any_low = (
            sv_result.is_low_data or ttr_result.is_low_data or punct_result.is_low_data
        )
        combined = (sv_result.score + ttr_result.score + punct_result.score) / 3.0
        return {
            "sentence_var_score": sv_result.score,
            "ttr_score": ttr_result.score,
            "punctuation_score": punct_result.score,
            "stylometric_score": combined,
            "low_data": any_low,
            "error": None,
        }
    except Exception as e:
        return {
            "sentence_var_score": 0.5,
            "ttr_score": 0.5,
            "punctuation_score": 0.5,
            "stylometric_score": 0.5,
            "low_data": True,
            "error": f"signal_stylometry_unexpected: {type(e).__name__}: {e}",
        }

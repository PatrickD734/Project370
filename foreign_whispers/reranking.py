"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    #We went with strategy 1: Rule based shortening;
    
    target_chars = int(target_duration_s * 15)
    candidates = []

    if len(baseline_es) <= target_chars:
        return []
    
    #Wordy phrases with short equivelants
    Replacements = {
        "en este momento": "ahora",
        "en este instante": "ahora",
        "a causa de": "por",
        "debido a": "por",
        "con el fin de": "para",
        "con la finalidad de": "para",
        "a pesar de que": "aunque",
        "sin embargo": "pero",
        "por lo tanto": "así",
        "es decir": "o sea",
        "de hecho": "además",
        "en realidad": "realmente",
        "a través de": "por",
        "por medio de": "por",
        "en relación con": "sobre",
        "con respecto a": "sobre",
    }

    shortened = baseline_es
    rationals = []
    for long_form, short_form in Replacements.items():
        if long_form in shortened.lower():
            shortened = shortened.lower().replace(long_form, short_form)
            rationals.append(f'"{long_form}" -> "{short_form}"')

    if shortened != baseline_es and len(shortened) <= target_chars:
        candidates.append(TranslationCandidate(
            text=shortened,
            char_count=len(shortened),
            brevity_rationale="Replaced wordy phrases: " + ", ".join(rationals),
        ))

    #Stripping filler words
    FILLERS = [
        "bueno, ", "pues, ", "entonces, ", "claro, ",
        "mira, ", "oye, ", "eh, ", "um, ", "uh, ",
        ", ¿verdad?", ", ¿no?", ", ¿ok?",
    ]
    stripped_rationals = []
    stripped = baseline_es
    for filler in FILLERS:
        if filler in stripped.lower():
            stripped = stripped.lower().replace(filler, "")
            stripped_rationals.append(f'removed "{filler.strip()}"')
        
    if stripped != baseline_es and len(stripped) <= target_chars:
        candidates.append(TranslationCandidate(
            text=stripped,
            char_count=len(stripped),
            brevity_rationale="Stripped fillers: " + ", ".join(stripped_rationals),
        ))

    #Truncating to nearest sentence boundary

    if len(baseline_es) > target_chars:
        truncated = baseline_es[:target_chars]
        for punct in [". ", ", ", " "]:
            idx = truncated.rfind(punct)
            if idx > target_chars * 0.6:
                truncated = truncated[:idx].strip()
                break
        if truncated != baseline_es:
            candidates.append(TranslationCandidate(
            text=truncated,
            char_count=len(truncated),
            brevity_rationale="Truncated to fit time limit",
        ))
    
    return sorted(candidates, key=lambda c: c.char_count)

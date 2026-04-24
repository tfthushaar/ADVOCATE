"""Compute ADVOCATE evaluation metrics with graceful dependency fallbacks."""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_model: "SentenceTransformer | None" = None
_embedding_backend_error: str | None = None


def _get_model() -> "SentenceTransformer | None":
    global _model, _embedding_backend_error
    if _model is not None:
        return _model
    if _embedding_backend_error is not None:
        return None

    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
        return _model
    except Exception as exc:
        _embedding_backend_error = str(exc)
        return None


def embedding_backend_status() -> tuple[bool, str]:
    model = _get_model()
    if model is not None:
        return True, "SentenceTransformer embeddings available"
    return False, _embedding_backend_error or "SentenceTransformer embeddings unavailable"


def compute_svi(gap_report: dict) -> float:
    """Extract or recompute the Strategy Vulnerability Index."""
    if "svi" in gap_report and isinstance(gap_report["svi"], (int, float)):
        return float(gap_report["svi"])

    unrebutted = gap_report.get("unrebutted_count", 0)
    total = gap_report.get("total_opponent_claims", 0)
    if total == 0:
        return 0.0
    return round((unrebutted / total) * 100, 1)


def _tokenise(text: str) -> Counter:
    tokens = [token for token in text.lower().split() if token]
    return Counter(tokens)


def _cosine_from_counters(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    intersection = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in intersection)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _fallback_divergence(employer_text: str, employee_text: str) -> float:
    """Cheap lexical fallback if embedding dependencies are unavailable."""
    similarity = _cosine_from_counters(_tokenise(employer_text), _tokenise(employee_text))
    return round(max(0.0, 1.0 - similarity), 4)


def compute_adversarial_divergence(employer_args: dict, employee_args: dict) -> float:
    """Compute divergence between the two sides' arguments."""

    def args_to_text(arguments: dict) -> str:
        claims = arguments.get("claims", [])
        return " ".join(
            f"{claim.get('issue', '')} {claim.get('rule', '')} {claim.get('application', '')}"
            for claim in claims
        )

    employer_text = args_to_text(employer_args)
    employee_text = args_to_text(employee_args)
    if not employer_text or not employee_text:
        return 0.0

    model = _get_model()
    if model is None:
        return _fallback_divergence(employer_text, employee_text)

    try:
        embeddings = model.encode([employer_text, employee_text])
        first, second = embeddings[0], embeddings[1]
        cosine_similarity = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second) + 1e-10))
        return round(max(0.0, 1.0 - cosine_similarity), 4)
    except Exception:
        return _fallback_divergence(employer_text, employee_text)


def compute_rule_validity_rate(evaluation: dict) -> float:
    """Compute the overall verified citation rate across both sides."""
    all_scores = evaluation.get("employer_scores", []) + evaluation.get("employee_scores", [])
    if not all_scores:
        return 0.0
    valid = sum(1 for score in all_scores if score.get("rule_validity", 0) == 1)
    return round(valid / len(all_scores) * 100, 1)


def compute_batch_metrics(results: list[dict]) -> dict:
    """Compute batch metrics over a list of pipeline results."""
    svi_by_case = {}
    divergence_by_case = {}
    rule_validity_by_case = {}
    outcome_correct = []
    winner_svis = []
    loser_svis = []

    for result in results:
        case_id = result.get("case_id", "unknown")
        gap_report = result.get("gap_report", {})
        employer_args = result.get("employer_args", {})
        employee_args = result.get("employee_args", {})
        evaluation = result.get("evaluation", {})
        ground_truth = result.get("ground_truth_outcome", "")

        weaker_side = gap_report.get("weaker_side", "")
        weaker_svi = compute_svi(gap_report)
        stronger_svi = round(max(0.0, weaker_svi - evaluation.get("score_delta", 0) * 10), 1)
        employer_svi = weaker_svi if weaker_side == "employer" else stronger_svi
        employee_svi = weaker_svi if weaker_side == "employee" else stronger_svi

        svi_by_case[case_id] = {"employer_svi": employer_svi, "employee_svi": employee_svi}
        divergence_by_case[case_id] = compute_adversarial_divergence(employer_args, employee_args)
        rule_validity_by_case[case_id] = compute_rule_validity_rate(evaluation)

        if ground_truth == "employer_wins":
            actual_loser_svi = employee_svi
            actual_winner_svi = employer_svi
        elif ground_truth == "employee_wins":
            actual_loser_svi = employer_svi
            actual_winner_svi = employee_svi
        else:
            continue

        outcome_correct.append(actual_loser_svi > actual_winner_svi)
        loser_svis.append(actual_loser_svi)
        winner_svis.append(actual_winner_svi)

    outcome_alignment = round(sum(outcome_correct) / len(outcome_correct) * 100, 1) if outcome_correct else 0.0

    return {
        "svi_by_case": svi_by_case,
        "divergence_by_case": divergence_by_case,
        "rule_validity_by_case": rule_validity_by_case,
        "outcome_alignment_pct": outcome_alignment,
        "n_cases_with_outcome": len(outcome_correct),
        "summary_stats": {
            "mean_svi_winner": round(float(np.mean(winner_svis)), 2) if winner_svis else 0.0,
            "mean_svi_loser": round(float(np.mean(loser_svis)), 2) if loser_svis else 0.0,
            "mean_divergence": round(float(np.mean(list(divergence_by_case.values()))), 4) if divergence_by_case else 0.0,
            "mean_rule_validity_rate": round(float(np.mean(list(rule_validity_by_case.values()))), 1)
            if rule_validity_by_case
            else 0.0,
        },
    }

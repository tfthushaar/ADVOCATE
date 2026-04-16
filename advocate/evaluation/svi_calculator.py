"""
svi_calculator.py
Computes the Strategy Vulnerability Index (SVI) and Adversarial Divergence Score
for a batch of pipeline runs (used in the validation harness).

SVI Formula:
  SVI = (unrebutted_opponent_claims / total_opponent_claims) × 100

Adversarial Divergence Score:
  Cosine distance between employer and employee argument embeddings on the same case.
  Higher = more divergent = better adversarial separation.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def compute_svi(gap_report: dict) -> float:
    """
    Extract the SVI from a gap report dict.
    Falls back to recomputing from raw fields if needed.
    """
    if "svi" in gap_report and isinstance(gap_report["svi"], (int, float)):
        return float(gap_report["svi"])

    unrebutted = gap_report.get("unrebutted_count", 0)
    total = gap_report.get("total_opponent_claims", 0)
    if total == 0:
        return 0.0
    return round((unrebutted / total) * 100, 1)


def compute_adversarial_divergence(employer_args: dict, employee_args: dict) -> float:
    """
    Compute the Adversarial Divergence Score between the two agents' arguments.
    Returns cosine similarity (0=identical, 1=orthogonal, values near 0 indicate convergence).

    A high score (close to 1) indicates the architectural isolation is working —
    the agents are producing genuinely different arguments.
    """
    model = _get_model()

    def args_to_text(args: dict) -> str:
        claims = args.get("claims", [])
        return " ".join(
            f"{c.get('issue', '')} {c.get('rule', '')} {c.get('application', '')}"
            for c in claims
        )

    employer_text = args_to_text(employer_args)
    employee_text = args_to_text(employee_args)

    if not employer_text or not employee_text:
        return 0.0

    embeddings = model.encode([employer_text, employee_text])
    e1, e2 = embeddings[0], embeddings[1]

    # Cosine similarity
    cos_sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10))
    # Divergence = 1 - similarity (higher = more different)
    divergence = round(1.0 - cos_sim, 4)
    return divergence


def compute_rule_validity_rate(evaluation: dict) -> float:
    """
    Compute the overall Rule Validity Rate across both sides.
    = % of claims whose cited case passed the cosine similarity threshold.
    """
    all_scores = evaluation.get("employer_scores", []) + evaluation.get("employee_scores", [])
    if not all_scores:
        return 0.0
    valid = sum(1 for s in all_scores if s.get("rule_validity", 0) == 1)
    return round(valid / len(all_scores) * 100, 1)


def compute_batch_metrics(results: list[dict]) -> dict:
    """
    Compute aggregate metrics over a batch of pipeline runs.

    Args:
        results: List of dicts, each with keys:
                 {case_id, employer_args, employee_args, evaluation, gap_report, ground_truth_outcome}
                 ground_truth_outcome: "employer_wins" | "employee_wins"

    Returns:
        {
          svi_by_case:            {case_id: {employer_svi, employee_svi}},
          divergence_by_case:     {case_id: float},
          rule_validity_by_case:  {case_id: float},
          outcome_alignment:      % of cases where higher-SVI side lost (as expected),
          summary_stats:          {mean_svi_winner, mean_svi_loser, mean_divergence}
        }
    """
    svi_by_case = {}
    divergence_by_case = {}
    rule_validity_by_case = {}
    outcome_correct = []
    winner_svis = []
    loser_svis = []

    for result in results:
        cid = result.get("case_id", "unknown")
        gap = result.get("gap_report", {})
        emp_args = result.get("employer_args", {})
        ee_args = result.get("employee_args", {})
        evaluation = result.get("evaluation", {})
        ground_truth = result.get("ground_truth_outcome", "")

        # SVI — the gap_report gives us the weaker side's SVI
        weaker_side = gap.get("weaker_side", "")
        svi_value = compute_svi(gap)

        # Reconstruct both-side SVI approximation:
        # gap_report computes SVI for the WEAKER side only.
        # Stronger side's SVI is implicitly lower (they had better coverage).
        stronger_svi = round(max(0.0, svi_value - evaluation.get("score_delta", 0) * 10), 1)
        weaker_svi = svi_value
        employer_svi = weaker_svi if weaker_side == "employer" else stronger_svi
        employee_svi = weaker_svi if weaker_side == "employee" else stronger_svi

        svi_by_case[cid] = {"employer_svi": employer_svi, "employee_svi": employee_svi}
        divergence_by_case[cid] = compute_adversarial_divergence(emp_args, ee_args)
        rule_validity_by_case[cid] = compute_rule_validity_rate(evaluation)

        # Outcome alignment: losing side should have higher SVI
        if ground_truth == "employer_wins":
            predicted_loser = "employee"
            actual_loser_svi = employee_svi
            actual_winner_svi = employer_svi
        elif ground_truth == "employee_wins":
            predicted_loser = "employer"
            actual_loser_svi = employer_svi
            actual_winner_svi = employee_svi
        else:
            continue

        is_correct = actual_loser_svi > actual_winner_svi
        outcome_correct.append(is_correct)
        loser_svis.append(actual_loser_svi)
        winner_svis.append(actual_winner_svi)

    outcome_alignment = (
        round(sum(outcome_correct) / len(outcome_correct) * 100, 1)
        if outcome_correct else 0.0
    )

    return {
        "svi_by_case": svi_by_case,
        "divergence_by_case": divergence_by_case,
        "rule_validity_by_case": rule_validity_by_case,
        "outcome_alignment_pct": outcome_alignment,
        "n_cases_with_outcome": len(outcome_correct),
        "summary_stats": {
            "mean_svi_winner": round(float(np.mean(winner_svis)), 2) if winner_svis else 0.0,
            "mean_svi_loser": round(float(np.mean(loser_svis)), 2) if loser_svis else 0.0,
            "mean_divergence": round(
                float(np.mean(list(divergence_by_case.values()))), 4
            ) if divergence_by_case else 0.0,
            "mean_rule_validity_rate": round(
                float(np.mean(list(rule_validity_by_case.values()))), 1
            ) if rule_validity_by_case else 0.0,
        },
    }

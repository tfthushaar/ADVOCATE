"""Agent 4: score claims on the fixed IRAC rubric."""

from __future__ import annotations

import json

from dotenv import load_dotenv

from advocate.llm.client import chat_completion
from advocate.rag.retriever import verify_citation
from advocate.settings import get_default_model

load_dotenv()

DEFAULT_MODEL = get_default_model()

EVALUATOR_SYSTEM = """You are a strict, impartial legal argument evaluator.
You score legal claims using a fixed rubric. You MUST follow the scoring rules exactly.
Output ONLY valid JSON.

RUBRIC:
- issue_clarity (0 or 1): Award 1 if the claim clearly and correctly identifies the specific legal issue in dispute. Award 0 if vague, incorrect, or missing.
- application_logic (0, 1, or 2): Award 2 if the argument explicitly and logically connects the cited rule to the specific case facts with no logical gaps. Award 1 if the connection is made but with gaps or generalities. Award 0 if no logical connection is demonstrated.
- rebuttal_coverage (0 or 1): Award 1 if this claim directly addresses at least one of the opponent's specific claims. Award 0 if no opponent claim is addressed.

NOTE: rule_validity is NOT scored by you - it is verified programmatically and provided to you.

Output schema:
{
  "claim_id": "string",
  "side": "employer" | "employee",
  "issue_clarity": 0 or 1,
  "rule_validity": 0 or 1,
  "application_logic": 0, 1, or 2,
  "rebuttal_coverage": 0 or 1,
  "total_score": number,
  "issue_clarity_reason": "string",
  "application_logic_reason": "string",
  "rebuttal_coverage_reason": "string"
}"""


def _score_claim_llm(
    claim: dict,
    side: str,
    opponent_claims: list[dict],
    rule_validity_score: int,
    case: dict,
    model: str,
) -> dict:
    opponent_summary = "\n".join(
        f"  [{item['claim_id']}] Issue: {item.get('issue', '')} | Rule: {item.get('rule', '')}"
        for item in opponent_claims
    )
    prompt = (
        f"Score the following {side.upper()} claim using the rubric.\n\n"
        "CLAIM TO SCORE:\n"
        f"  claim_id: {claim.get('claim_id')}\n"
        f"  issue: {claim.get('issue', '')}\n"
        f"  rule: {claim.get('rule', '')}\n"
        f"  cited_case: {claim.get('cited_case', '')}\n"
        f"  application: {claim.get('application', '')}\n"
        f"  conclusion: {claim.get('conclusion', '')}\n\n"
        f"PROGRAMMATIC rule_validity score (already determined): {rule_validity_score}\n\n"
        "CASE FACTS (for application logic check):\n"
        + "\n".join(f"  - {fact}" for fact in case.get("facts", []))
        + "\n\n"
        f"OPPONENT CLAIMS (for rebuttal coverage check):\n{opponent_summary}\n\n"
        "Score this claim and output ONLY JSON."
    )

    raw, _ = chat_completion(
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        model=model,
        max_tokens=1024,
    )

    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {
            "claim_id": claim.get("claim_id"),
            "side": side,
            "issue_clarity": 0,
            "rule_validity": rule_validity_score,
            "application_logic": 0,
            "rebuttal_coverage": 0,
            "issue_clarity_reason": "parse error",
            "application_logic_reason": "parse error",
            "rebuttal_coverage_reason": "parse error",
        }

    scores["rule_validity"] = rule_validity_score
    scores["side"] = side
    scores["total_score"] = (
        scores.get("issue_clarity", 0)
        + scores.get("rule_validity", 0)
        + scores.get("application_logic", 0)
        + scores.get("rebuttal_coverage", 0)
    )
    return scores


def evaluate(case: dict, employer_args: dict, employee_args: dict, model: str | None = None) -> dict:
    selected_model = model or DEFAULT_MODEL
    employer_claims = employer_args.get("claims", [])
    employee_claims = employee_args.get("claims", [])

    def score_side(claims: list[dict], side: str, opponent_claims: list[dict]) -> list[dict]:
        scored_claims = []
        for claim in claims:
            cited_case = claim.get("cited_case", "")
            is_valid, best_score = verify_citation(cited_case) if cited_case else (False, 0.0)
            scores = _score_claim_llm(claim, side, opponent_claims, int(is_valid), case, selected_model)
            scores["citation_similarity"] = best_score
            scored_claims.append(scores)
        return scored_claims

    employer_scores = score_side(employer_claims, "employer", employee_claims)
    employee_scores = score_side(employee_claims, "employee", employer_claims)

    def average_total(scores: list[dict]) -> float:
        if not scores:
            return 0.0
        return round(sum(score.get("total_score", 0) for score in scores) / len(scores), 3)

    def dimension_average(scores: list[dict]) -> dict[str, float]:
        if not scores:
            return {
                "issue_clarity": 0,
                "rule_validity": 0,
                "application_logic": 0,
                "rebuttal_coverage": 0,
            }
        count = len(scores)
        return {
            key: round(sum(score.get(key, 0) for score in scores) / count, 3)
            for key in ("issue_clarity", "rule_validity", "application_logic", "rebuttal_coverage")
        }

    employer_average = average_total(employer_scores)
    employee_average = average_total(employee_scores)

    return {
        "employer_scores": employer_scores,
        "employee_scores": employee_scores,
        "employer_avg": employer_average,
        "employee_avg": employee_average,
        "employer_dimension_avg": dimension_average(employer_scores),
        "employee_dimension_avg": dimension_average(employee_scores),
        "stronger_side": "employer" if employer_average >= employee_average else "employee",
        "score_delta": round(abs(employer_average - employee_average), 3),
        "evaluator_model": selected_model,
    }

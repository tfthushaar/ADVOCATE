"""
irac_evaluator.py  —  Agent 4: IRAC Evaluator
Scores each claim on a fixed 4-dimension rubric (max 5 pts per claim).
Rule Validity is verified programmatically (cosine similarity ≥ 0.75).
"""

import os
import json
from dotenv import load_dotenv
from advocate.llm.client import chat_completion
from advocate.rag.retriever import verify_citation

load_dotenv()

DEFAULT_MODEL = os.getenv("ADVOCATE_MODEL", "claude-sonnet-4-6")

EVALUATOR_SYSTEM = """You are a strict, impartial legal argument evaluator.
You score legal claims using a fixed rubric. You MUST follow the scoring rules exactly.
Output ONLY valid JSON.

RUBRIC:
- issue_clarity (0 or 1): Award 1 if the claim clearly and correctly identifies the specific legal issue in dispute. Award 0 if vague, incorrect, or missing.
- application_logic (0, 1, or 2): Award 2 if the argument explicitly and logically connects the cited rule to the specific case facts with no logical gaps. Award 1 if the connection is made but with gaps or generalities. Award 0 if no logical connection is demonstrated.
- rebuttal_coverage (0 or 1): Award 1 if this claim directly addresses at least one of the opponent's specific claims. Award 0 if no opponent claim is addressed.

NOTE: rule_validity is NOT scored by you — it is verified programmatically and provided to you.

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
        f"  [{c['claim_id']}] Issue: {c.get('issue', '')} | Rule: {c.get('rule', '')}"
        for c in opponent_claims
    )
    prompt = (
        f"Score the following {side.upper()} claim using the rubric.\n\n"
        f"CLAIM TO SCORE:\n"
        f"  claim_id: {claim.get('claim_id')}\n"
        f"  issue: {claim.get('issue', '')}\n"
        f"  rule: {claim.get('rule', '')}\n"
        f"  cited_case: {claim.get('cited_case', '')}\n"
        f"  application: {claim.get('application', '')}\n"
        f"  conclusion: {claim.get('conclusion', '')}\n\n"
        f"PROGRAMMATIC rule_validity score (already determined): {rule_validity_score}\n\n"
        "CASE FACTS (for application logic check):\n"
        + "\n".join(f"  - {f}" for f in case.get("facts", [])) + "\n\n"
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


def evaluate(
    case: dict,
    employer_args: dict,
    employee_args: dict,
    model: str | None = None,
) -> dict:
    """
    Score all claims from both sides using the IRAC rubric.

    Args:
        case:          Parsed case dict.
        employer_args: Employer agent output.
        employee_args: Employee agent output.
        model:         LLM model ID for the evaluator. Defaults to ADVOCATE_MODEL.
    """
    model = model or DEFAULT_MODEL
    employer_claims = employer_args.get("claims", [])
    employee_claims = employee_args.get("claims", [])

    def score_side(claims, side, opponent_claims):
        scored = []
        for claim in claims:
            cited = claim.get("cited_case", "")
            is_valid, best_score = verify_citation(cited) if cited else (False, 0.0)
            scores = _score_claim_llm(claim, side, opponent_claims, int(is_valid), case, model)
            scores["citation_similarity"] = best_score
            scored.append(scores)
        return scored

    employer_scores = score_side(employer_claims, "employer", employee_claims)
    employee_scores = score_side(employee_claims, "employee", employer_claims)

    def avg(scores):
        return round(sum(s.get("total_score", 0) for s in scores) / len(scores), 3) if scores else 0.0

    def dim_avg(scores):
        if not scores:
            return {"issue_clarity": 0, "rule_validity": 0, "application_logic": 0, "rebuttal_coverage": 0}
        n = len(scores)
        return {k: round(sum(s.get(k, 0) for s in scores) / n, 3)
                for k in ("issue_clarity", "rule_validity", "application_logic", "rebuttal_coverage")}

    emp_avg = avg(employer_scores)
    ee_avg = avg(employee_scores)

    return {
        "employer_scores": employer_scores,
        "employee_scores": employee_scores,
        "employer_avg": emp_avg,
        "employee_avg": ee_avg,
        "employer_dimension_avg": dim_avg(employer_scores),
        "employee_dimension_avg": dim_avg(employee_scores),
        "stronger_side": "employer" if emp_avg >= ee_avg else "employee",
        "score_delta": round(abs(emp_avg - ee_avg), 3),
        "evaluator_model": model,
    }

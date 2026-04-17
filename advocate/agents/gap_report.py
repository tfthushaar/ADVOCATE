"""
gap_report.py  —  Agent 5: Strategy Gap Report
Computes the SVI and produces a ranked vulnerability list for the weaker side.

SVI = (unrebutted opponent claims / total opponent claims) × 100
"""

import os
import json
from dotenv import load_dotenv
from advocate.llm.client import chat_completion

load_dotenv()

DEFAULT_MODEL = os.getenv("ADVOCATE_MODEL", "claude-sonnet-4-6")

GAP_SYSTEM_PROMPT = """You are a senior legal strategy consultant.
You have been given the IRAC-scored arguments from both sides of an employment wrongful termination case.
Your task is to identify the strategic gaps — claims by the stronger side that the weaker side failed to address.

For each gap:
1. Explain WHY the unaddressed claim is dangerous for the weaker side.
2. Suggest a specific counter-argument the weaker side should have made.
3. Estimate severity: HIGH (would likely determine outcome), MEDIUM (significant), LOW (marginal).

Output ONLY valid JSON.

Output schema:
{
  "weaker_side": "employer" | "employee",
  "svi": number,
  "total_opponent_claims": number,
  "unrebutted_count": number,
  "gaps": [
    {
      "gap_rank": number,
      "opponent_claim_id": "string",
      "opponent_issue": "string",
      "opponent_rule": "string",
      "severity": "HIGH" | "MEDIUM" | "LOW",
      "why_dangerous": "string",
      "suggested_counter": "string",
      "suggested_case_type": "string"
    }
  ],
  "overall_strategy_assessment": "string — 2-3 sentence overall assessment",
  "top_priority_action": "string — single most important pre-trial action"
}"""


def generate_gap_report(
    case: dict,
    employer_args: dict,
    employee_args: dict,
    evaluation: dict,
    model: str | None = None,
) -> dict:
    """
    Generate the Strategy Gap Report for the weaker side.

    Args:
        case:          Parsed case dict.
        employer_args: Employer agent output.
        employee_args: Employee agent output.
        evaluation:    IRAC evaluation output.
        model:         LLM model ID. Defaults to ADVOCATE_MODEL.
    """
    model = model or DEFAULT_MODEL
    stronger_side = evaluation["stronger_side"]
    weaker_side = "employee" if stronger_side == "employer" else "employer"

    stronger_claims = (
        employer_args.get("claims", []) if stronger_side == "employer"
        else employee_args.get("claims", [])
    )
    weaker_scores = (
        evaluation["employee_scores"] if weaker_side == "employee"
        else evaluation["employer_scores"]
    )
    total_opponent_claims = len(stronger_claims)

    stronger_claims_str = "\n".join(
        f"[{c.get('claim_id')}] Issue: {c.get('issue', '')}\n"
        f"  Rule: {c.get('rule', '')}\n"
        f"  Application: {c.get('application', '')[:300]}\n"
        f"  Conclusion: {c.get('conclusion', '')}"
        for c in stronger_claims
    )
    weaker_scores_str = "\n".join(
        f"[{s.get('claim_id')}] Total: {s.get('total_score')}/5 | "
        f"Rebuttal: {s.get('rebuttal_coverage')}/1 | "
        f"Reason: {s.get('rebuttal_coverage_reason', '')[:200]}"
        for s in weaker_scores
    )

    prompt = (
        f"CASE: {case.get('plaintiff')} v. {case.get('defendant')} | "
        f"Termination: {case.get('termination_reason')} | "
        f"Jurisdiction: {case.get('jurisdiction')}\n\n"
        f"SCORE SUMMARY:\n"
        f"  Stronger ({stronger_side}): {evaluation[f'{stronger_side}_avg']}/5\n"
        f"  Weaker ({weaker_side}): {evaluation[f'{weaker_side}_avg']}/5\n\n"
        f"STRONGER SIDE ({stronger_side.upper()}) CLAIMS:\n{stronger_claims_str}\n\n"
        f"WEAKER SIDE ({weaker_side.upper()}) SCORES:\n{weaker_scores_str}\n\n"
        f"Total stronger-side claims: {total_opponent_claims}\n\n"
        "Identify all unaddressed vulnerabilities and generate the gap report JSON."
    )

    raw, latency = chat_completion(
        messages=[
            {"role": "system", "content": GAP_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
        max_tokens=4096,
    )

    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        report = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gap Report Agent ({model}) returned invalid JSON: {e}\nRaw:\n{raw}") from e

    # Always recompute SVI from formula (don't trust LLM's arithmetic)
    unrebutted = report.get("unrebutted_count", 0)
    report["svi"] = round((unrebutted / total_opponent_claims) * 100, 1) if total_opponent_claims else 0.0
    report["total_opponent_claims"] = total_opponent_claims
    report["weaker_side"] = weaker_side
    report["stronger_side"] = stronger_side
    report["model"] = model
    report["latency_s"] = latency
    report["score_summary"] = {
        "employer_avg": evaluation["employer_avg"],
        "employee_avg": evaluation["employee_avg"],
        "employer_dimension_avg": evaluation["employer_dimension_avg"],
        "employee_dimension_avg": evaluation["employee_dimension_avg"],
    }
    return report

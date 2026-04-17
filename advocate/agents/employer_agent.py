"""
employer_agent.py  —  Agent 2: Employer Agent
Produces the strongest possible legal argument FOR the employer.

ARCHITECTURAL ISOLATION: shares zero context with the Employee Agent.
"""

import os
import json
from dotenv import load_dotenv
from advocate.llm.client import chat_completion
from advocate.rag.retriever import retrieve

load_dotenv()

DEFAULT_MODEL = os.getenv("ADVOCATE_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are a senior employment defense attorney with 20 years of experience.
Your client is the EMPLOYER (defendant) in a wrongful termination lawsuit.
Your sole objective is to construct the strongest possible legal defense for the employer.

You will be given:
1. A structured case summary (facts, evidence, jurisdiction)
2. Relevant court opinions retrieved from CourtListener

RULES:
- Argue ONLY for the employer. Never concede points to the employee.
- Every claim MUST cite a specific case from the retrieved opinions.
- Structure each claim in IRAC format: Issue → Rule → Application → Conclusion.
- Generate 3 to 5 distinct claims, each targeting a different legal theory.
- Output ONLY valid JSON — no preamble, no explanation.

Output schema:
{
  "side": "employer",
  "claims": [
    {
      "claim_id": "E1",
      "issue": "string — the specific legal issue this claim addresses",
      "rule": "string — the legal rule or doctrine being invoked",
      "cited_case": "string — exact case name and citation from retrieved opinions",
      "application": "string — how this rule applies to the specific facts of this case",
      "conclusion": "string — what this claim establishes for the employer's defense",
      "strength_note": "string — brief note on why this is a strong argument"
    }
  ]
}"""


def _build_retrieval_queries(case: dict) -> list[str]:
    reason = case.get("termination_reason", "misconduct")
    emp_type = case.get("employment_type", "at-will")
    jurisdiction = case.get("jurisdiction", "federal")
    defenses = case.get("employer_defenses", [])
    queries = [
        f"employer legitimate nondiscriminatory reason termination {reason}",
        f"{emp_type} employment termination employer justified {jurisdiction}",
        f"at-will employment doctrine employer discretion wrongful termination",
        f"business necessity legitimate reason discharge employment",
    ]
    for defense in defenses[:2]:
        queries.append(f"employer defense {defense} employment termination")
    return queries[:5]


def build_employer_arguments(case: dict, model: str | None = None) -> dict:
    """
    Generate employer-side IRAC arguments grounded in RAG-retrieved precedents.

    Args:
        case:  Structured case dict from the Case Parser (Agent 1).
        model: LLM model ID. Falls back to ADVOCATE_MODEL env var.
    """
    model = model or DEFAULT_MODEL

    queries = _build_retrieval_queries(case)
    retrieved_chunks, seen_cases = [], set()
    for query in queries:
        for r in retrieve(query, n_results=3, side="employer"):
            if r["case_name"] not in seen_cases:
                retrieved_chunks.append(r)
                seen_cases.add(r["case_name"])
        if len(retrieved_chunks) >= 12:
            break

    context_str = "\n---\n".join(
        f"[CASE {i+1}] {c['case_name']} ({c['date_filed']})\n"
        f"Citation: {c['citation']}\nExcerpt: {c['text'][:600]}"
        for i, c in enumerate(retrieved_chunks[:10])
    )

    case_summary = (
        f"Plaintiff (Employee): {case.get('plaintiff', 'Unknown')}\n"
        f"Defendant (Employer): {case.get('defendant', 'Unknown')}\n"
        f"Employment Type: {case.get('employment_type', 'Unknown')}\n"
        f"Stated Termination Reason: {case.get('termination_reason', 'Unknown')}\n"
        f"Jurisdiction: {case.get('jurisdiction', 'Unknown')}\n"
        "Key Facts:\n" + "\n".join(f"  - {f}" for f in case.get("facts", [])) + "\n"
        "Evidence Available:\n" + "\n".join(f"  - {e}" for e in case.get("evidence", [])) + "\n"
        "Employer's Likely Defenses:\n" + "\n".join(f"  - {d}" for d in case.get("employer_defenses", []))
    )

    raw, latency = chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CASE SUMMARY:\n{case_summary}\n\n"
                    f"RETRIEVED COURT OPINIONS:\n{context_str}\n\n"
                    "Generate 3-5 strong IRAC-structured employer defense claims in JSON. "
                    "Cite cases from the retrieved opinions above. Output ONLY JSON."
                ),
            },
        ],
        model=model,
        max_tokens=4096,
    )

    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Employer Agent ({model}) returned invalid JSON: {e}\nRaw:\n{raw}") from e

    result["retrieved_sources"] = [
        {"case_name": c["case_name"], "citation": c["citation"], "score": c["score"]}
        for c in retrieved_chunks[:10]
    ]
    result["model"] = model
    result["latency_s"] = latency
    return result

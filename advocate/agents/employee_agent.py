"""
employee_agent.py  —  Agent 3: Employee Agent
Produces the strongest possible legal argument FOR the employee.

ARCHITECTURAL ISOLATION: shares zero context with the Employer Agent.
"""

import os
import json
from dotenv import load_dotenv
from advocate.llm.client import chat_completion
from advocate.rag.retriever import retrieve

load_dotenv()

DEFAULT_MODEL = os.getenv("ADVOCATE_MODEL", "gpt-4o")

SYSTEM_PROMPT = """You are a senior plaintiff's employment attorney with 20 years of experience.
Your client is the EMPLOYEE (plaintiff) in a wrongful termination lawsuit.
Your sole objective is to construct the strongest possible legal case for the employee.

You will be given:
1. A structured case summary (facts, evidence, jurisdiction)
2. Relevant court opinions retrieved from CourtListener

RULES:
- Argue ONLY for the employee. Never concede points to the employer.
- Every claim MUST cite a specific case from the retrieved opinions.
- Structure each claim in IRAC format: Issue → Rule → Application → Conclusion.
- Generate 3 to 5 distinct claims, each targeting a different legal theory.
- Output ONLY valid JSON — no preamble, no explanation.

Output schema:
{
  "side": "employee",
  "claims": [
    {
      "claim_id": "P1",
      "issue": "string — the specific legal issue this claim addresses",
      "rule": "string — the legal rule, statute, or doctrine being invoked",
      "cited_case": "string — exact case name and citation from retrieved opinions",
      "application": "string — how this rule applies to the specific facts of this case",
      "conclusion": "string — what this claim establishes for the employee's case",
      "strength_note": "string — brief note on why this is a strong argument"
    }
  ]
}"""


def _build_retrieval_queries(case: dict) -> list[str]:
    characteristics = case.get("protected_characteristics", [])
    jurisdiction = case.get("jurisdiction", "federal")
    claims = case.get("employee_claims", [])
    queries = [
        "employee wrongful termination discrimination retaliation plaintiff victory",
        "Title VII ADEA ADA protected class termination employee rights",
        f"retaliation protected activity employee termination unlawful {jurisdiction}",
        "pretext employer stated reason false discriminatory termination",
    ]
    for char in characteristics[:2]:
        queries.append(f"{char} discrimination termination employee plaintiff")
    for claim in claims[:2]:
        queries.append(f"{claim} employee rights wrongful termination case")
    return queries[:5]


def build_employee_arguments(case: dict, model: str | None = None) -> dict:
    """
    Generate employee-side IRAC arguments grounded in RAG-retrieved precedents.

    Args:
        case:  Structured case dict from the Case Parser (Agent 1).
               NOTE: Must NOT contain any output from the Employer Agent.
        model: LLM model ID. Falls back to ADVOCATE_MODEL env var.
    """
    model = model or DEFAULT_MODEL

    queries = _build_retrieval_queries(case)
    retrieved_chunks, seen_cases = [], set()
    for query in queries:
        for r in retrieve(query, n_results=3, side="employee"):
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
        "Employee's Legal Claims:\n" + "\n".join(f"  - {c}" for c in case.get("employee_claims", [])) + "\n"
        "Protected Characteristics at Issue:\n" + "\n".join(f"  - {p}" for p in case.get("protected_characteristics", []))
    )

    raw, latency = chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CASE SUMMARY:\n{case_summary}\n\n"
                    f"RETRIEVED COURT OPINIONS:\n{context_str}\n\n"
                    "Generate 3-5 strong IRAC-structured employee plaintiff claims in JSON. "
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
        raise ValueError(f"Employee Agent ({model}) returned invalid JSON: {e}\nRaw:\n{raw}") from e

    result["retrieved_sources"] = [
        {"case_name": c["case_name"], "citation": c["citation"], "score": c["score"]}
        for c in retrieved_chunks[:10]
    ]
    result["model"] = model
    result["latency_s"] = latency
    return result

"""Agent 3: build the strongest employee-side IRAC arguments."""

from __future__ import annotations

import json

from dotenv import load_dotenv

from advocate.llm.client import chat_completion
from advocate.rag.retriever import retrieve
from advocate.settings import get_default_model

load_dotenv()

DEFAULT_MODEL = get_default_model()

SYSTEM_PROMPT = """You are a senior plaintiff's employment attorney with 20 years of experience.
Your client is the EMPLOYEE (plaintiff) in a wrongful termination lawsuit.
Your sole objective is to construct the strongest possible legal case for the employee.

You will be given:
1. A structured case summary (facts, evidence, jurisdiction)
2. Relevant court opinions retrieved from CourtListener

RULES:
- Argue ONLY for the employee. Never concede points to the employer.
- Every claim MUST cite a specific case from the retrieved opinions.
- Structure each claim in IRAC format: Issue -> Rule -> Application -> Conclusion.
- Generate 3 to 5 distinct claims, each targeting a different legal theory.
- Output ONLY valid JSON - no preamble, no explanation.

Output schema:
{
  "side": "employee",
  "claims": [
    {
      "claim_id": "P1",
      "issue": "string - the specific legal issue this claim addresses",
      "rule": "string - the legal rule, statute, or doctrine being invoked",
      "cited_case": "string - exact case name and citation from retrieved opinions",
      "application": "string - how this rule applies to the specific facts of this case",
      "conclusion": "string - what this claim establishes for the employee's case",
      "strength_note": "string - brief note on why this is a strong argument"
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
    for characteristic in characteristics[:2]:
        queries.append(f"{characteristic} discrimination termination employee plaintiff")
    for claim in claims[:2]:
        queries.append(f"{claim} employee rights wrongful termination case")
    return queries[:5]


def build_employee_arguments(case: dict, model: str | None = None) -> dict:
    selected_model = model or DEFAULT_MODEL

    queries = _build_retrieval_queries(case)
    retrieved_chunks: list[dict] = []
    seen_cases: set[str] = set()

    for query in queries:
        for result in retrieve(query, n_results=3, side="employee"):
            if result["case_name"] not in seen_cases:
                retrieved_chunks.append(result)
                seen_cases.add(result["case_name"])
        if len(retrieved_chunks) >= 12:
            break

    context = "\n---\n".join(
        f"[CASE {index + 1}] {chunk['case_name']} ({chunk['date_filed']})\n"
        f"Citation: {chunk['citation']}\n"
        f"Excerpt: {chunk['text'][:600]}"
        for index, chunk in enumerate(retrieved_chunks[:10])
    )

    case_summary = (
        f"Plaintiff (Employee): {case.get('plaintiff', 'Unknown')}\n"
        f"Defendant (Employer): {case.get('defendant', 'Unknown')}\n"
        f"Employment Type: {case.get('employment_type', 'Unknown')}\n"
        f"Stated Termination Reason: {case.get('termination_reason', 'Unknown')}\n"
        f"Jurisdiction: {case.get('jurisdiction', 'Unknown')}\n"
        "Key Facts:\n"
        + "\n".join(f"  - {fact}" for fact in case.get("facts", []))
        + "\nEvidence Available:\n"
        + "\n".join(f"  - {item}" for item in case.get("evidence", []))
        + "\nEmployee's Legal Claims:\n"
        + "\n".join(f"  - {claim}" for claim in case.get("employee_claims", []))
        + "\nProtected Characteristics at Issue:\n"
        + "\n".join(f"  - {item}" for item in case.get("protected_characteristics", []))
    )

    raw, latency = chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CASE SUMMARY:\n{case_summary}\n\n"
                    f"RETRIEVED COURT OPINIONS:\n{context}\n\n"
                    "Generate 3-5 strong IRAC-structured employee plaintiff claims in JSON. "
                    "Cite cases from the retrieved opinions above. Output ONLY JSON."
                ),
            },
        ],
        model=selected_model,
        max_tokens=4096,
    )

    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Employee Agent ({selected_model}) returned invalid JSON: {exc}\nRaw:\n{raw}") from exc

    result["retrieved_sources"] = [
        {"case_name": chunk["case_name"], "citation": chunk["citation"], "score": chunk["score"]}
        for chunk in retrieved_chunks[:10]
    ]
    result["model"] = selected_model
    result["latency_s"] = latency
    return result

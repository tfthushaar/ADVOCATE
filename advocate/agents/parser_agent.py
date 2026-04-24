"""Agent 1: parse a free-text case brief into structured JSON."""

from __future__ import annotations

import json

from dotenv import load_dotenv

from advocate.llm.client import chat_completion
from advocate.settings import get_default_model

load_dotenv()

DEFAULT_MODEL = get_default_model()

SYSTEM_PROMPT = """You are a precise legal case parser specialising in US employment law.
Your task is to extract structured information from free-text case briefs about wrongful termination.

You MUST output ONLY valid JSON - no explanation, no preamble, no trailing text.
If a field cannot be determined from the brief, use null for scalars or [] for arrays.

Output schema:
{
  "plaintiff": "string - the employee's name or identifier",
  "defendant": "string - the employer's name",
  "employment_type": "string - e.g. 'at-will', 'contract', 'union', 'public sector'",
  "termination_reason": "string - stated reason for termination as given by employer",
  "jurisdiction": "string - US federal circuit or state, e.g. '9th Circuit', 'California'",
  "facts": ["string - key factual allegations, one per element"],
  "evidence": ["string - each piece of evidence mentioned, one per element"],
  "employee_claims": ["string - legal theories the employee would likely raise"],
  "employer_defenses": ["string - defenses the employer would likely raise"],
  "protected_characteristics": ["string - any protected class characteristics at issue, e.g. 'race', 'age', 'disability'"],
  "timeline": "string - key dates or sequence if mentioned, else null"
}"""


def parse_case(case_brief: str, model: str | None = None) -> dict:
    selected_model = model or DEFAULT_MODEL

    raw, _ = chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Parse the following employment wrongful termination case brief into structured JSON.\n\n"
                    f"CASE BRIEF:\n{case_brief}\n\n"
                    "Output ONLY the JSON object."
                ),
            },
        ],
        model=selected_model,
        max_tokens=2048,
    )

    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Case Parser ({selected_model}) returned invalid JSON: {exc}\nRaw:\n{raw}") from exc

    for field in ("facts", "evidence", "employee_claims", "employer_defenses", "protected_characteristics"):
        if field not in parsed or parsed[field] is None:
            parsed[field] = []

    return parsed

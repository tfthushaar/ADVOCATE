"""
parser_agent.py  —  Agent 1: Case Parser
Converts a free-text case brief into a structured JSON representation.
"""

import os
import json
from dotenv import load_dotenv
from advocate.llm.client import chat_completion

load_dotenv()

DEFAULT_MODEL = os.getenv("ADVOCATE_MODEL", "gpt-4o")

SYSTEM_PROMPT = """You are a precise legal case parser specialising in US employment law.
Your task is to extract structured information from free-text case briefs about wrongful termination.

You MUST output ONLY valid JSON — no explanation, no preamble, no trailing text.
If a field cannot be determined from the brief, use null for scalars or [] for arrays.

Output schema:
{
  "plaintiff": "string — the employee's name or identifier",
  "defendant": "string — the employer's name",
  "employment_type": "string — e.g. 'at-will', 'contract', 'union', 'public sector'",
  "termination_reason": "string — stated reason for termination as given by employer",
  "jurisdiction": "string — US federal circuit or state, e.g. '9th Circuit', 'California'",
  "facts": ["string — key factual allegations, one per element"],
  "evidence": ["string — each piece of evidence mentioned, one per element"],
  "employee_claims": ["string — legal theories the employee would likely raise"],
  "employer_defenses": ["string — defenses the employer would likely raise"],
  "protected_characteristics": ["string — any protected class characteristics at issue, e.g. 'race', 'age', 'disability'"],
  "timeline": "string — key dates/sequence if mentioned, else null"
}"""


def parse_case(case_brief: str, model: str | None = None) -> dict:
    """
    Parse a free-text case brief into a structured dict.

    Args:
        case_brief: Raw text of the case brief.
        model:      LLM model ID. Falls back to ADVOCATE_MODEL env var.

    Returns:
        Parsed case dict conforming to the schema above.
    """
    model = model or DEFAULT_MODEL

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
        model=model,
        max_tokens=2048,
    )

    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Case Parser ({model}) returned invalid JSON: {e}\nRaw:\n{raw}") from e

    for field in ("facts", "evidence", "employee_claims", "employer_defenses", "protected_characteristics"):
        if field not in parsed or parsed[field] is None:
            parsed[field] = []

    return parsed

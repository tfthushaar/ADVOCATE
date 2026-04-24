"""Run batch validation with Anthropic and save the output JSON."""

from __future__ import annotations

import argparse
import json
import os

from advocate.evaluation.validate import run_validation
from advocate.settings import get_setting

os.environ["ADVOCATE_MODEL"] = "claude-sonnet-4-6"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch validation with Anthropic Claude Sonnet")
    parser.add_argument(
        "--output",
        type=str,
        default="anthropic_research_results.json",
        help="Path to save the JSON output results.",
    )
    args = parser.parse_args()

    anthropic_key = get_setting("ANTHROPIC_API_KEY", "")
    courtlistener_token = get_setting("COURTLISTENER_API_TOKEN", "")

    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY is not set in the environment or secrets.")
        return

    if not courtlistener_token:
        print("WARNING: COURTLISTENER_API_TOKEN is missing. Retrieval may fail if you rebuild the index.")

    print("Starting batch validation using Anthropic model (claude-sonnet-4-6)...")
    results = run_validation("./advocate/data/test_scenarios", model="claude-sonnet-4-6")

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\n[Success] Research results written to:")
    print(f" -> {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

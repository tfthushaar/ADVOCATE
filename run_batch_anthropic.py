"""
run_batch_anthropic.py

A standalone script to execute the ADVOCATE multi-agent pipeline over all 10
test scenarios using Anthropic's Claude Sonnet model, simulating a research
run and saving the validation results to a JSON file. This JSON can then be
used to write an IEEE-format research paper.
"""
import os
import json
import argparse
from pathlib import Path

# Force the environment to use Claude Sonnet
os.environ["ADVOCATE_MODEL"] = "claude-sonnet-4-6"

from advocate.evaluation.validate import run_validation

def main():
    parser = argparse.ArgumentParser(description="Run Batch Validation with Anthropic Claude Sonnet")
    parser.add_argument(
        "--output", 
        type=str, 
        default="anthropic_research_results.json",
        help="Path to save the JSON output results."
    )
    args = parser.parse_args()

    # Verify keys are present
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    cl_token = os.getenv("COURTLISTENER_API_TOKEN", "")

    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY is not set in the environment or .env file.")
        print("Please enter it in your .env file and run this script again.")
        return
        
    if not cl_token:
        print("WARNING: COURTLISTENER_API_TOKEN is missing. The retrieval stage might fail.")

    print(f"Starting Batch Validation using Anthropic model (claude-sonnet-4-6)...")
    results = run_validation("./advocate/data/test_scenarios")
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[Success] Research results securely written to:")
    print(f" -> {os.path.abspath(args.output)}")
    print("You can now use these results to write your IEEE-format research paper.")

if __name__ == "__main__":
    main()

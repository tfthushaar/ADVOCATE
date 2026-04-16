"""
validate.py
Runs the ADVOCATE pipeline over the held-out test scenarios and performs
a Wilcoxon signed-rank test to validate the SVI metric.

Usage:
  python -m advocate.evaluation.validate
  python -m advocate.evaluation.validate --scenarios ./advocate/data/test_scenarios
"""

import json
import argparse
import sys
from pathlib import Path
from scipy import stats
from advocate.pipeline.advocate_graph import run_pipeline
from advocate.evaluation.svi_calculator import (
    compute_svi,
    compute_adversarial_divergence,
    compute_rule_validity_rate,
    compute_batch_metrics,
)


def load_scenarios(scenarios_dir: str) -> list[dict]:
    """Load all test scenario JSON files from the given directory."""
    path = Path(scenarios_dir)
    scenarios = []
    for f in sorted(path.glob("*.json")):
        with open(f) as fp:
            scenario = json.load(fp)
        if "case_brief" in scenario:
            scenarios.append(scenario)
    return scenarios


def run_validation(scenarios_dir: str = "./advocate/data/test_scenarios") -> dict:
    """
    Run the full pipeline on all scenarios and validate SVI against outcomes.

    Returns:
        Validation results dict with Wilcoxon test results.
    """
    scenarios = load_scenarios(scenarios_dir)
    if not scenarios:
        print(f"No scenarios found in {scenarios_dir}. Add JSON files first.")
        return {}

    print(f"[validate] Running pipeline on {len(scenarios)} test scenarios …\n")
    results = []

    for i, scenario in enumerate(scenarios):
        case_id = scenario.get("case_id", f"scenario_{i+1}")
        ground_truth = scenario.get("ground_truth_outcome", "")
        print(f"  [{i+1}/{len(scenarios)}] {case_id} (outcome: {ground_truth}) …")

        try:
            state = run_pipeline(scenario["case_brief"])
            results.append({
                "case_id": case_id,
                "ground_truth_outcome": ground_truth,
                "employer_args": state.get("employer_args", {}),
                "employee_args": state.get("employee_args", {}),
                "evaluation": state.get("evaluation", {}),
                "gap_report": state.get("gap_report", {}),
                "errors": state.get("errors", []),
            })
            svi = compute_svi(state.get("gap_report", {}))
            weaker = state.get("gap_report", {}).get("weaker_side", "?")
            print(f"    SVI={svi}% (weaker: {weaker}) | errors: {state.get('errors', [])}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"case_id": case_id, "ground_truth_outcome": ground_truth, "error": str(e)})

    print("\n[validate] Computing batch metrics …")
    batch = compute_batch_metrics([r for r in results if "error" not in r])

    # Wilcoxon signed-rank test: loser SVI vs winner SVI
    winner_svis, loser_svis = [], []
    for r in results:
        if "error" in r or not r.get("ground_truth_outcome"):
            continue
        cid = r["case_id"]
        svi_pair = batch["svi_by_case"].get(cid, {})
        gt = r["ground_truth_outcome"]

        if gt == "employer_wins":
            winner_svis.append(svi_pair.get("employer_svi", 0))
            loser_svis.append(svi_pair.get("employee_svi", 0))
        elif gt == "employee_wins":
            winner_svis.append(svi_pair.get("employee_svi", 0))
            loser_svis.append(svi_pair.get("employer_svi", 0))

    wilcoxon_result = None
    if len(loser_svis) >= 3:
        try:
            stat, p_value = stats.wilcoxon(loser_svis, winner_svis, alternative="greater")
            wilcoxon_result = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "significant": p_value < 0.05,
                "n_pairs": len(loser_svis),
                "mean_loser_svi": round(sum(loser_svis) / len(loser_svis), 2),
                "mean_winner_svi": round(sum(winner_svis) / len(winner_svis), 2),
            }
        except Exception as e:
            wilcoxon_result = {"error": str(e)}
    else:
        wilcoxon_result = {"error": f"Need at least 3 paired observations, got {len(loser_svis)}"}

    output = {
        "n_scenarios": len(scenarios),
        "n_successful": len([r for r in results if "error" not in r]),
        "batch_metrics": batch,
        "wilcoxon_test": wilcoxon_result,
        "per_case_results": [
            {
                "case_id": r.get("case_id"),
                "ground_truth": r.get("ground_truth_outcome"),
                "svi": batch["svi_by_case"].get(r.get("case_id"), {}),
                "divergence": batch["divergence_by_case"].get(r.get("case_id")),
                "rule_validity_rate": batch["rule_validity_by_case"].get(r.get("case_id")),
                "errors": r.get("errors", []),
            }
            for r in results
        ],
    }

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Scenarios run:          {output['n_scenarios']}")
    print(f"Successful:             {output['n_successful']}")
    print(f"Outcome alignment:      {batch['outcome_alignment_pct']}%")
    print(f"Mean divergence score:  {batch['summary_stats']['mean_divergence']}")
    print(f"Mean rule validity:     {batch['summary_stats']['mean_rule_validity_rate']}%")
    if wilcoxon_result and "p_value" in wilcoxon_result:
        print(f"\nWilcoxon test (loser SVI > winner SVI):")
        print(f"  Statistic: {wilcoxon_result['statistic']}")
        print(f"  p-value:   {wilcoxon_result['p_value']}")
        print(f"  Significant (p<0.05): {wilcoxon_result['significant']}")
        print(f"  Mean loser SVI:  {wilcoxon_result['mean_loser_svi']}%")
        print(f"  Mean winner SVI: {wilcoxon_result['mean_winner_svi']}%")
    print("=" * 60)

    return output


def main():
    parser = argparse.ArgumentParser(description="ADVOCATE validation harness")
    parser.add_argument(
        "--scenarios",
        default="./advocate/data/test_scenarios",
        help="Path to test scenarios directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save results JSON",
    )
    args = parser.parse_args()

    results = run_validation(args.scenarios)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

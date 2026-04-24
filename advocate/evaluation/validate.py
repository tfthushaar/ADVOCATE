"""Run held-out scenario validation and compute batch metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import stats

from advocate.evaluation.svi_calculator import compute_batch_metrics, compute_svi
from advocate.pipeline.advocate_graph import run_pipeline


def load_scenarios(scenarios_dir: str) -> list[dict]:
    path = Path(scenarios_dir)
    scenarios = []
    for file_path in sorted(path.glob("*.json")):
        with file_path.open(encoding="utf-8") as handle:
            scenario = json.load(handle)
        if "case_brief" in scenario:
            scenarios.append(scenario)
    return scenarios


def run_validation(scenarios_dir: str = "./advocate/data/test_scenarios", model: str | None = None) -> dict:
    scenarios = load_scenarios(scenarios_dir)
    if not scenarios:
        print(f"No scenarios found in {scenarios_dir}. Add JSON files first.")
        return {}

    print(f"[validate] Running pipeline on {len(scenarios)} test scenarios...\n")
    results = []

    for index, scenario in enumerate(scenarios, start=1):
        case_id = scenario.get("case_id", f"scenario_{index}")
        ground_truth = scenario.get("ground_truth_outcome", "")
        print(f"  [{index}/{len(scenarios)}] {case_id} (outcome: {ground_truth})...")

        try:
            state = run_pipeline(scenario["case_brief"], model=model)
            results.append(
                {
                    "case_id": case_id,
                    "ground_truth_outcome": ground_truth,
                    "employer_args": state.get("employer_args", {}),
                    "employee_args": state.get("employee_args", {}),
                    "evaluation": state.get("evaluation", {}),
                    "gap_report": state.get("gap_report", {}),
                    "errors": state.get("errors", []),
                },
            )
            svi = compute_svi(state.get("gap_report", {}))
            weaker = state.get("gap_report", {}).get("weaker_side", "?")
            print(f"    SVI={svi}% (weaker: {weaker}) | errors: {state.get('errors', [])}")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            results.append(
                {
                    "case_id": case_id,
                    "ground_truth_outcome": ground_truth,
                    "error": str(exc),
                },
            )

    print("\n[validate] Computing batch metrics...")
    batch = compute_batch_metrics([result for result in results if "error" not in result])

    winner_svis: list[float] = []
    loser_svis: list[float] = []
    for result in results:
        if "error" in result or not result.get("ground_truth_outcome"):
            continue

        case_id = result["case_id"]
        svi_pair = batch["svi_by_case"].get(case_id, {})
        ground_truth = result["ground_truth_outcome"]

        if ground_truth == "employer_wins":
            winner_svis.append(svi_pair.get("employer_svi", 0))
            loser_svis.append(svi_pair.get("employee_svi", 0))
        elif ground_truth == "employee_wins":
            winner_svis.append(svi_pair.get("employee_svi", 0))
            loser_svis.append(svi_pair.get("employer_svi", 0))

    wilcoxon_result = None
    if len(loser_svis) >= 3:
        try:
            statistic, p_value = stats.wilcoxon(loser_svis, winner_svis, alternative="greater")
            wilcoxon_result = {
                "statistic": round(float(statistic), 4),
                "p_value": round(float(p_value), 4),
                "significant": bool(p_value < 0.05),
                "n_pairs": len(loser_svis),
                "mean_loser_svi": round(sum(loser_svis) / len(loser_svis), 2),
                "mean_winner_svi": round(sum(winner_svis) / len(winner_svis), 2),
            }
        except Exception as exc:
            wilcoxon_result = {"error": str(exc)}
    else:
        wilcoxon_result = {"error": f"Need at least 3 paired observations, got {len(loser_svis)}"}

    output = {
        "n_scenarios": len(scenarios),
        "n_successful": len([result for result in results if "error" not in result]),
        "batch_metrics": batch,
        "wilcoxon_test": wilcoxon_result,
        "per_case_results": [
            {
                "case_id": result.get("case_id"),
                "ground_truth": result.get("ground_truth_outcome"),
                "svi": batch["svi_by_case"].get(result.get("case_id"), {}),
                "divergence": batch["divergence_by_case"].get(result.get("case_id")),
                "rule_validity_rate": batch["rule_validity_by_case"].get(result.get("case_id")),
                "errors": result.get("errors", []),
            }
            for result in results
        ],
    }

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Scenarios run:          {output['n_scenarios']}")
    print(f"Successful:             {output['n_successful']}")
    print(f"Outcome alignment:      {batch['outcome_alignment_pct']}%")
    print(f"Mean divergence score:  {batch['summary_stats']['mean_divergence']}")
    print(f"Mean rule validity:     {batch['summary_stats']['mean_rule_validity_rate']}%")
    if wilcoxon_result and "p_value" in wilcoxon_result:
        print("\nWilcoxon test (loser SVI > winner SVI):")
        print(f"  Statistic: {wilcoxon_result['statistic']}")
        print(f"  p-value:   {wilcoxon_result['p_value']}")
        print(f"  Significant (p<0.05): {wilcoxon_result['significant']}")
        print(f"  Mean loser SVI:  {wilcoxon_result['mean_loser_svi']}%")
        print(f"  Mean winner SVI: {wilcoxon_result['mean_winner_svi']}%")
    print("=" * 60)

    return output


def main() -> None:
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
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

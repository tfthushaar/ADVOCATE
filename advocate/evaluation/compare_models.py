"""Run the ADVOCATE pipeline across multiple models and compare metrics."""

from __future__ import annotations

import time

from advocate.evaluation.svi_calculator import (
    compute_adversarial_divergence,
    compute_rule_validity_rate,
    compute_svi,
)
from advocate.pipeline.advocate_graph import run_pipeline


def run_comparison(case_brief: str, model_ids: list[str], progress_callback=None) -> dict:
    results: dict[str, dict] = {}

    for model_id in model_ids:
        if progress_callback:
            progress_callback(model_id, "running")

        started_at = time.perf_counter()
        try:
            state = run_pipeline(case_brief, model=model_id)
            elapsed = round(time.perf_counter() - started_at, 2)

            evaluation = state.get("evaluation", {})
            gap_report = state.get("gap_report", {})
            employer_args = state.get("employer_args", {})
            employee_args = state.get("employee_args", {})
            employer_dimensions = evaluation.get("employer_dimension_avg", {})
            employee_dimensions = evaluation.get("employee_dimension_avg", {})

            def avg_dimension(key: str) -> float:
                return round((employer_dimensions.get(key, 0) + employee_dimensions.get(key, 0)) / 2, 3)

            employer_average = evaluation.get("employer_avg", 0)
            employee_average = evaluation.get("employee_avg", 0)
            overall_average = round((employer_average + employee_average) / 2, 3)

            results[model_id] = {
                "model": model_id,
                "status": "success",
                "employer_avg_score": employer_average,
                "employee_avg_score": employee_average,
                "overall_avg_score": overall_average,
                "rule_validity_rate": compute_rule_validity_rate(evaluation),
                "adversarial_divergence": compute_adversarial_divergence(employer_args, employee_args),
                "svi": compute_svi(gap_report),
                "issue_clarity_avg": avg_dimension("issue_clarity"),
                "application_logic_avg": avg_dimension("application_logic"),
                "rebuttal_coverage_avg": avg_dimension("rebuttal_coverage"),
                "total_latency_s": elapsed,
                "error_count": len(state.get("errors", [])),
                "stronger_side": evaluation.get("stronger_side", "-"),
                "weaker_side": gap_report.get("weaker_side", "-"),
                "errors": state.get("errors", []),
                "_state": state,
            }

            if progress_callback:
                progress_callback(model_id, "done")

        except Exception as exc:
            elapsed = round(time.perf_counter() - started_at, 2)
            results[model_id] = {
                "model": model_id,
                "status": "error",
                "error_message": str(exc),
                "total_latency_s": elapsed,
                "overall_avg_score": 0,
                "rule_validity_rate": 0,
                "adversarial_divergence": 0,
                "svi": 0,
                "error_count": 1,
            }
            if progress_callback:
                progress_callback(model_id, f"error: {exc}")

    successful_models = [model_id for model_id in model_ids if results.get(model_id, {}).get("status") == "success"]

    higher_is_better = [
        "overall_avg_score",
        "employer_avg_score",
        "employee_avg_score",
        "rule_validity_rate",
        "adversarial_divergence",
        "issue_clarity_avg",
        "application_logic_avg",
        "rebuttal_coverage_avg",
    ]
    lower_is_better = ["svi", "total_latency_s", "error_count"]

    rankings: dict[str, list[str]] = {}
    winners: dict[str, str] = {}

    for metric in higher_is_better:
        ranked = sorted(successful_models, key=lambda model_id: results[model_id].get(metric, 0), reverse=True)
        rankings[metric] = ranked
        if ranked:
            winners[metric] = ranked[0]

    for metric in lower_is_better:
        ranked = sorted(successful_models, key=lambda model_id: results[model_id].get(metric, float("inf")))
        rankings[metric] = ranked
        if ranked:
            winners[metric] = ranked[0]

    summary_table = []
    for model_id in model_ids:
        result = results.get(model_id, {})
        summary_table.append(
            {
                "Model": model_id,
                "Status": result.get("status", "error"),
                "Overall Score": result.get("overall_avg_score", "-"),
                "Rule Validity %": result.get("rule_validity_rate", "-"),
                "Divergence": result.get("adversarial_divergence", "-"),
                "SVI %": result.get("svi", "-"),
                "Issue Clarity": result.get("issue_clarity_avg", "-"),
                "App. Logic": result.get("application_logic_avg", "-"),
                "Rebuttal": result.get("rebuttal_coverage_avg", "-"),
                "Latency (s)": result.get("total_latency_s", "-"),
                "Errors": result.get("error_count", "-"),
            },
        )

    return {
        "case_brief": case_brief,
        "models_run": model_ids,
        "results": results,
        "rankings": rankings,
        "winner": winners,
        "summary_table": summary_table,
    }


def best_model_overall(comparison: dict) -> str | None:
    results = comparison.get("results", {})
    successful = {model_id: result for model_id, result in results.items() if result.get("status") == "success"}
    if not successful:
        return None

    def normalise(values: list[float], invert: bool = False) -> list[float]:
        minimum = min(values)
        maximum = max(values)
        if maximum == minimum:
            return [0.5] * len(values)
        normalised = [(value - minimum) / (maximum - minimum) for value in values]
        return [1 - score for score in normalised] if invert else normalised

    models = list(successful.keys())
    metric_weights = {
        "overall_avg_score": (0.35, False),
        "rule_validity_rate": (0.25, False),
        "adversarial_divergence": (0.20, False),
        "svi": (0.10, True),
        "total_latency_s": (0.10, True),
    }

    composite_scores = {model_id: 0.0 for model_id in models}
    for metric, (weight, invert) in metric_weights.items():
        raw_values = [successful[model_id].get(metric, 0) for model_id in models]
        for model_id, score in zip(models, normalise(raw_values, invert=invert)):
            composite_scores[model_id] += weight * score

    return max(composite_scores, key=lambda model_id: composite_scores[model_id])

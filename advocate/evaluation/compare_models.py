"""
compare_models.py
Runs the ADVOCATE pipeline with multiple LLM models on the same case brief
and aggregates comparative metrics across all models.

Metrics computed per model:
  - employer_avg_score      : avg IRAC score for employer arguments (0–5)
  - employee_avg_score      : avg IRAC score for employee arguments (0–5)
  - overall_avg_score       : mean of both sides
  - rule_validity_rate      : % of citations verified in RAG (0–100)
  - adversarial_divergence  : cosine distance between employer/employee outputs (0–1)
  - svi                     : Strategy Vulnerability Index for weaker side (0–100%)
  - issue_clarity_avg       : avg issue clarity score (0–1)
  - application_logic_avg   : avg application logic score (0–2)
  - rebuttal_coverage_avg   : avg rebuttal coverage score (0–1)
  - total_latency_s         : wall-clock seconds for the full pipeline run
  - error_count             : number of pipeline errors encountered
"""

import time
from advocate.pipeline.advocate_graph import run_pipeline
from advocate.evaluation.svi_calculator import (
    compute_svi,
    compute_adversarial_divergence,
    compute_rule_validity_rate,
)


def run_comparison(
    case_brief: str,
    model_ids: list[str],
    progress_callback=None,
) -> dict:
    """
    Run the pipeline for each model and collect comparative metrics.

    Args:
        case_brief:        The case brief text (same for all models).
        model_ids:         List of model ID strings to compare.
        progress_callback: Optional callable(model_id, status_str) for UI progress.

    Returns:
        {
          "case_brief": str,
          "models_run": [str],
          "results": {model_id: per_model_result_dict},
          "rankings": {metric_name: [model_id ordered best→worst]},
          "winner": {metric_name: model_id},
          "summary_table": [{"model": ..., metric1: ..., ...}]
        }
    """
    results: dict[str, dict] = {}

    for model_id in model_ids:
        if progress_callback:
            progress_callback(model_id, "running")

        t0 = time.perf_counter()
        try:
            state = run_pipeline(case_brief, model=model_id)
            elapsed = round(time.perf_counter() - t0, 2)

            evaluation = state.get("evaluation", {})
            gap = state.get("gap_report", {})
            emp_args = state.get("employer_args", {})
            ee_args = state.get("employee_args", {})

            # Dimension averages (both sides combined)
            emp_dim = evaluation.get("employer_dimension_avg", {})
            ee_dim = evaluation.get("employee_dimension_avg", {})

            def avg_dim(key):
                return round(
                    (emp_dim.get(key, 0) + ee_dim.get(key, 0)) / 2, 3
                )

            employer_avg = evaluation.get("employer_avg", 0)
            employee_avg = evaluation.get("employee_avg", 0)
            overall_avg = round((employer_avg + employee_avg) / 2, 3)

            results[model_id] = {
                "model": model_id,
                "status": "success",
                "employer_avg_score": employer_avg,
                "employee_avg_score": employee_avg,
                "overall_avg_score": overall_avg,
                "rule_validity_rate": compute_rule_validity_rate(evaluation),
                "adversarial_divergence": compute_adversarial_divergence(emp_args, ee_args),
                "svi": compute_svi(gap),
                "issue_clarity_avg": avg_dim("issue_clarity"),
                "application_logic_avg": avg_dim("application_logic"),
                "rebuttal_coverage_avg": avg_dim("rebuttal_coverage"),
                "total_latency_s": elapsed,
                "error_count": len(state.get("errors", [])),
                "stronger_side": evaluation.get("stronger_side", "—"),
                "weaker_side": gap.get("weaker_side", "—"),
                "errors": state.get("errors", []),
                # Full state for drill-down
                "_state": state,
            }

            if progress_callback:
                progress_callback(model_id, "done")

        except Exception as e:
            elapsed = round(time.perf_counter() - t0, 2)
            results[model_id] = {
                "model": model_id,
                "status": "error",
                "error_message": str(e),
                "total_latency_s": elapsed,
                "overall_avg_score": 0,
                "rule_validity_rate": 0,
                "adversarial_divergence": 0,
                "svi": 0,
                "error_count": 1,
            }
            if progress_callback:
                progress_callback(model_id, f"error: {e}")

    # ── Rankings ──────────────────────────────────────────────────────────────
    # For each metric, rank models from best to worst.
    # "best" definition differs by metric:
    #   higher is better: overall_avg_score, rule_validity_rate, adversarial_divergence,
    #                     issue_clarity_avg, application_logic_avg, rebuttal_coverage_avg
    #   lower is better:  svi (lower vulnerability = better), total_latency_s, error_count

    successful = [m for m in model_ids if results.get(m, {}).get("status") == "success"]

    higher_is_better = [
        "overall_avg_score", "employer_avg_score", "employee_avg_score",
        "rule_validity_rate", "adversarial_divergence",
        "issue_clarity_avg", "application_logic_avg", "rebuttal_coverage_avg",
    ]
    lower_is_better = ["svi", "total_latency_s", "error_count"]

    rankings: dict[str, list[str]] = {}
    winner: dict[str, str] = {}

    for metric in higher_is_better:
        ranked = sorted(successful, key=lambda m: results[m].get(metric, 0), reverse=True)
        rankings[metric] = ranked
        if ranked:
            winner[metric] = ranked[0]

    for metric in lower_is_better:
        ranked = sorted(successful, key=lambda m: results[m].get(metric, float("inf")))
        rankings[metric] = ranked
        if ranked:
            winner[metric] = ranked[0]

    # ── Summary table (list of dicts, one row per model) ─────────────────────
    summary_table = []
    for m in model_ids:
        r = results.get(m, {})
        summary_table.append({
            "Model": m,
            "Status": r.get("status", "error"),
            "Overall Score": r.get("overall_avg_score", "—"),
            "Rule Validity %": r.get("rule_validity_rate", "—"),
            "Divergence": r.get("adversarial_divergence", "—"),
            "SVI %": r.get("svi", "—"),
            "Issue Clarity": r.get("issue_clarity_avg", "—"),
            "App. Logic": r.get("application_logic_avg", "—"),
            "Rebuttal": r.get("rebuttal_coverage_avg", "—"),
            "Latency (s)": r.get("total_latency_s", "—"),
            "Errors": r.get("error_count", "—"),
        })

    return {
        "case_brief": case_brief,
        "models_run": model_ids,
        "results": results,
        "rankings": rankings,
        "winner": winner,
        "summary_table": summary_table,
    }


def best_model_overall(comparison: dict) -> str | None:
    """
    Pick the single best overall model using a weighted composite score.

    Weights (all normalised to 0–1 range before weighting):
      overall_avg_score    × 0.35
      rule_validity_rate   × 0.25
      adversarial_divergence × 0.20
      svi (inverted)       × 0.10
      latency (inverted)   × 0.10
    """
    results = comparison.get("results", {})
    successful = {m: r for m, r in results.items() if r.get("status") == "success"}
    if not successful:
        return None

    def _norm(values: list[float], invert=False) -> list[float]:
        mn, mx = min(values), max(values)
        if mx == mn:
            return [0.5] * len(values)
        normed = [(v - mn) / (mx - mn) for v in values]
        return [1 - n for n in normed] if invert else normed

    models = list(successful.keys())
    metrics = {
        "overall_avg_score": (0.35, False),
        "rule_validity_rate": (0.25, False),
        "adversarial_divergence": (0.20, False),
        "svi": (0.10, True),
        "total_latency_s": (0.10, True),
    }

    composite = {m: 0.0 for m in models}
    for metric, (weight, invert) in metrics.items():
        raw = [successful[m].get(metric, 0) for m in models]
        normed = _norm(raw, invert=invert)
        for m, score in zip(models, normed):
            composite[m] += weight * score

    return max(composite, key=lambda m: composite[m])

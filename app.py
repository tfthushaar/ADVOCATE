"""Deploy-ready Streamlit entrypoint for the ADVOCATE application."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from advocate.evaluation.compare_models import best_model_overall, run_comparison
from advocate.evaluation.validate import run_validation
from advocate.evaluation.svi_calculator import (
    compute_adversarial_divergence,
    compute_rule_validity_rate,
    compute_svi,
)
from advocate.llm.client import AVAILABLE_MODELS, is_model_available, provider_env_key_for_model
from advocate.pipeline.advocate_graph import run_pipeline
from advocate.rag.retriever import collection_size, index_ready
from advocate.settings import get_default_model, get_setting, supabase_is_configured
from advocate.store import SupabaseStore

APP_ROOT = Path(__file__).parent
SCENARIOS_DIR = APP_ROOT / "advocate" / "data" / "test_scenarios"
RESEARCH_RESULTS_PATH = APP_ROOT / "anthropic_research_results.json"
SECRETS_EXAMPLE_PATH = APP_ROOT / ".streamlit" / "secrets.toml.example"
SCHEMA_PATH = APP_ROOT / "supabase" / "schema.sql"
PROVIDER_SETTINGS = (
    ("OPENAI_API_KEY", "OpenAI"),
    ("ANTHROPIC_API_KEY", "Anthropic"),
    ("GOOGLE_API_KEY", "Google"),
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        :root {
            --ink: #1f2937;
            --ink-soft: #52606d;
            --paper: #f5efe5;
            --paper-deep: #ebe0cf;
            --panel: rgba(255, 252, 247, 0.82);
            --accent: #8c4a2f;
            --accent-dark: #6b341d;
            --accent-soft: #d8b08c;
            --emerald: #2f6f5d;
            --crimson: #9d3c2c;
            --border: rgba(107, 52, 29, 0.14);
            --shadow: 0 18px 40px rgba(63, 34, 18, 0.08);
        }

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(216, 176, 140, 0.35), transparent 28%),
                radial-gradient(circle at top left, rgba(140, 74, 47, 0.08), transparent 30%),
                linear-gradient(180deg, #f8f3eb 0%, #f2eadf 100%);
        }

        .block-container {
            max-width: 1320px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            background: linear-gradient(135deg, rgba(255, 249, 240, 0.96), rgba(247, 237, 220, 0.92));
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 2.1rem 2.3rem;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
            margin-bottom: 1.4rem;
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: auto -8% -30% auto;
            width: 240px;
            height: 240px;
            background: radial-gradient(circle, rgba(140, 74, 47, 0.14), transparent 65%);
            pointer-events: none;
        }

        .eyebrow {
            display: inline-block;
            background: rgba(140, 74, 47, 0.10);
            color: var(--accent-dark);
            border: 1px solid rgba(140, 74, 47, 0.18);
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .hero h1, .hero h2, .hero h3 {
            font-family: 'Fraunces', serif;
            margin: 0.7rem 0 0.2rem 0;
            color: #2b2118;
        }

        .hero-copy {
            color: var(--ink-soft);
            max-width: 760px;
            line-height: 1.65;
            font-size: 1rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 0.9rem;
            margin: 1rem 0 1.25rem 0;
        }

        .metric-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow);
        }

        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--ink-soft);
            font-weight: 700;
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2b2118;
            margin-top: 0.35rem;
        }

        .metric-note {
            color: var(--ink-soft);
            font-size: 0.88rem;
            margin-top: 0.35rem;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.15rem 1.2rem;
            box-shadow: var(--shadow);
        }

        .subtle {
            color: var(--ink-soft);
        }

        .claim-card {
            background: rgba(255, 255, 255, 0.66);
            border: 1px solid rgba(107, 52, 29, 0.10);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.8rem;
        }

        .claim-title {
            font-weight: 700;
            color: var(--accent-dark);
            margin-bottom: 0.35rem;
        }

        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border: 1px solid rgba(107, 52, 29, 0.12);
            background: rgba(255, 255, 255, 0.72);
            color: var(--accent-dark);
        }

        .setup-list {
            line-height: 1.8;
            color: var(--ink-soft);
        }

        .auth-card {
            background: rgba(255, 251, 245, 0.92);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1.5rem;
            box-shadow: var(--shadow);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            background: rgba(255,255,255,0.55);
            padding: 0.35rem;
            border-radius: 14px;
            border: 1px solid var(--border);
        }

        .stTabs [data-baseweb="tab"] {
            font-weight: 700;
            color: var(--ink-soft);
            border-radius: 10px !important;
            padding: 0.7rem 1rem !important;
        }

        .stTabs [aria-selected="true"] {
            background: #fff7ec !important;
            color: var(--accent-dark) !important;
            border: 1px solid rgba(140, 74, 47, 0.18);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div>'
        "</div>"
    )


def init_session_state() -> None:
    st.session_state.setdefault("auth_user", None)
    st.session_state.setdefault("single_case_brief", "")
    st.session_state.setdefault("compare_case_brief", "")
    st.session_state.setdefault("_last_single_scenario", "Custom brief")
    st.session_state.setdefault("_last_compare_scenario", "Custom brief")
    st.session_state.setdefault("latest_single_result", None)
    st.session_state.setdefault("latest_comparison", None)
    st.session_state.setdefault("latest_batch_result", None)
    for env_key, _provider in PROVIDER_SETTINGS:
        st.session_state.setdefault(env_key, "")


@st.cache_resource(show_spinner=False)
def get_store() -> SupabaseStore:
    return SupabaseStore()


@st.cache_data(show_spinner=False)
def load_scenarios() -> dict[str, dict]:
    scenarios = {"Custom brief": {"case_brief": "", "ground_truth_outcome": None}}
    for file_path in sorted(SCENARIOS_DIR.glob("*.json")):
        with file_path.open(encoding="utf-8") as handle:
            scenarios[file_path.stem] = json.load(handle)
    return scenarios


@st.cache_data(show_spinner=False)
def load_research_results() -> dict:
    with RESEARCH_RESULTS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def app_user() -> dict | None:
    return st.session_state.get("auth_user")


def configured_models() -> list[str]:
    return [model_id for model_id in AVAILABLE_MODELS if is_model_available(model_id)]


def clear_session_provider_keys() -> None:
    for env_key, _provider in PROVIDER_SETTINGS:
        st.session_state[env_key] = ""


def default_model_index(models: list[str]) -> int:
    default_model = get_default_model()
    return models.index(default_model) if default_model in models else 0


def format_model_name(model_id: str) -> str:
    info = AVAILABLE_MODELS.get(model_id)
    if not info:
        return model_id
    return f"{info['display']} ({info['provider']})"


def friendly_timestamp(value: str | None) -> str:
    if not value:
        return "Unknown time"
    try:
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned).strftime("%d %b %Y, %I:%M %p UTC")
    except Exception:
        return value


def brief_title(case_brief: str, prefix: str = "") -> str:
    first_line = " ".join(case_brief.strip().split())
    truncated = first_line[:80] + ("..." if len(first_line) > 80 else "")
    return f"{prefix}{truncated or 'Untitled case'}"


def ensure_scenario_state(state_key: str, marker_key: str, selected_name: str, scenarios: dict[str, dict]) -> None:
    if st.session_state.get(marker_key) != selected_name:
        st.session_state[state_key] = scenarios[selected_name].get("case_brief", "")
        st.session_state[marker_key] = selected_name


def pipeline_summary(state: dict, model: str) -> dict:
    evaluation = state.get("evaluation", {})
    gap_report = state.get("gap_report", {})
    return {
        "model": model,
        "stronger_side": evaluation.get("stronger_side", "-"),
        "weaker_side": gap_report.get("weaker_side", "-"),
        "svi": compute_svi(gap_report),
        "rule_validity_rate": compute_rule_validity_rate(evaluation),
        "divergence": compute_adversarial_divergence(
            state.get("employer_args", {}),
            state.get("employee_args", {}),
        ),
        "warning_count": len(state.get("errors", [])),
        "steps_completed": state.get("steps_completed", []),
    }


def comparison_summary(comparison: dict) -> dict:
    winner = best_model_overall(comparison)
    return {
        "winner": winner or "-",
        "models_run": len(comparison.get("models_run", [])),
        "successful_models": len(
            [
                result
                for result in comparison.get("results", {}).values()
                if result.get("status") == "success"
            ],
        ),
    }


def batch_summary(result: dict, model: str) -> dict:
    metrics = result.get("batch_metrics", {}).get("summary_stats", {})
    return {
        "model": model,
        "n_successful": result.get("n_successful", 0),
        "n_scenarios": result.get("n_scenarios", 0),
        "mean_divergence": metrics.get("mean_divergence", 0),
        "mean_rule_validity_rate": metrics.get("mean_rule_validity_rate", 0),
        "outcome_alignment_pct": result.get("batch_metrics", {}).get("outcome_alignment_pct", 0),
    }


def save_run(store: SupabaseStore, *, run_mode: str, title: str, model: str, case_brief: str, summary: dict, result: dict) -> None:
    user = app_user()
    if not user:
        return
    status = "warning" if summary.get("warning_count", 0) else "success"
    if run_mode == "compare":
        status = "success"
    try:
        store.save_analysis(
            user_id=user["id"],
            title=title,
            model=model,
            run_mode=run_mode,
            status=status,
            case_brief=case_brief,
            summary=summary,
            result=result,
            errors=result.get("errors", []) if isinstance(result, dict) else [],
        )
    except Exception as exc:
        st.warning(f"Run finished, but saving to Supabase failed: {exc}")


def render_claims(label: str, claims: list[dict]) -> None:
    st.markdown(f"#### {label}")
    if not claims:
        st.info("No claims were returned for this side.")
        return
    for claim in claims:
        st.markdown(
            f"""
            <div class="claim-card">
                <div class="claim-title">[{claim.get("claim_id", "-")}] {claim.get("issue", "Untitled issue")}</div>
                <div><strong>Rule:</strong> {claim.get("rule", "-")}</div>
                <div><strong>Cited Case:</strong> {claim.get("cited_case", "-")}</div>
                <div><strong>Application:</strong> {claim.get("application", "-")}</div>
                <div><strong>Conclusion:</strong> {claim.get("conclusion", "-")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_single_result(model: str, state: dict) -> None:
    summary = pipeline_summary(state, model)
    evaluation = state.get("evaluation", {})
    gap_report = state.get("gap_report", {})
    parsed_case = state.get("parsed_case", {})

    st.markdown("### Latest Analysis")
    st.markdown(
        "<div class='metric-grid'>"
        + metric_card("Model", format_model_name(model), "Active analysis model")
        + metric_card("Stronger Side", str(summary["stronger_side"]).title(), "Higher average IRAC score")
        + metric_card("SVI", f"{summary['svi']}%", "Lower is strategically safer")
        + metric_card("Rule Validity", f"{summary['rule_validity_rate']}%", "Verified citation rate")
        + "</div>",
        unsafe_allow_html=True,
    )

    if state.get("errors"):
        st.warning("Pipeline finished with warnings: " + "; ".join(state["errors"]))

    overview_tab, arguments_tab, gaps_tab, raw_tab = st.tabs(
        ["Case Map", "Arguments", "Gap Report", "Raw JSON"],
    )

    with overview_tab:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Parsed Case")
            st.write(
                {
                    "plaintiff": parsed_case.get("plaintiff"),
                    "defendant": parsed_case.get("defendant"),
                    "employment_type": parsed_case.get("employment_type"),
                    "termination_reason": parsed_case.get("termination_reason"),
                    "jurisdiction": parsed_case.get("jurisdiction"),
                },
            )
            st.markdown("#### Key Facts")
            st.write(parsed_case.get("facts", []))
        with right:
            st.markdown("#### Evaluation Snapshot")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Side": "Employer",
                            "Average Score": evaluation.get("employer_avg", 0),
                            "Issue Clarity": evaluation.get("employer_dimension_avg", {}).get("issue_clarity", 0),
                            "Rule Validity": evaluation.get("employer_dimension_avg", {}).get("rule_validity", 0),
                            "Application Logic": evaluation.get("employer_dimension_avg", {}).get("application_logic", 0),
                            "Rebuttal Coverage": evaluation.get("employer_dimension_avg", {}).get("rebuttal_coverage", 0),
                        },
                        {
                            "Side": "Employee",
                            "Average Score": evaluation.get("employee_avg", 0),
                            "Issue Clarity": evaluation.get("employee_dimension_avg", {}).get("issue_clarity", 0),
                            "Rule Validity": evaluation.get("employee_dimension_avg", {}).get("rule_validity", 0),
                            "Application Logic": evaluation.get("employee_dimension_avg", {}).get("application_logic", 0),
                            "Rebuttal Coverage": evaluation.get("employee_dimension_avg", {}).get("rebuttal_coverage", 0),
                        },
                    ],
                ),
                hide_index=True,
                use_container_width=True,
            )

    with arguments_tab:
        employer_col, employee_col = st.columns(2)
        with employer_col:
            render_claims("Employer Claims", state.get("employer_args", {}).get("claims", []))
        with employee_col:
            render_claims("Employee Claims", state.get("employee_args", {}).get("claims", []))

    with gaps_tab:
        st.markdown(
            "<div class='metric-grid'>"
            + metric_card("Weaker Side", str(gap_report.get("weaker_side", "-")).title(), "Side with greater vulnerability")
            + metric_card("Unrebutted Claims", f"{gap_report.get('unrebutted_count', 0)}", "Claims left insufficiently addressed")
            + metric_card("Top Priority", gap_report.get("top_priority_action", "-"), "Most important next move")
            + metric_card("Warnings", str(len(state.get("errors", []))), "Pipeline execution warnings")
            + "</div>",
            unsafe_allow_html=True,
        )
        if gap_report.get("overall_strategy_assessment"):
            st.info(gap_report["overall_strategy_assessment"])
        for gap in sorted(gap_report.get("gaps", []), key=lambda item: item.get("gap_rank", 999)):
            with st.expander(f"Gap #{gap.get('gap_rank', '?')} - {gap.get('severity', 'UNKNOWN')}"):
                st.write(
                    {
                        "opponent_issue": gap.get("opponent_issue"),
                        "opponent_rule": gap.get("opponent_rule"),
                        "why_dangerous": gap.get("why_dangerous"),
                        "suggested_counter": gap.get("suggested_counter"),
                        "suggested_case_type": gap.get("suggested_case_type"),
                    },
                )

    with raw_tab:
        st.json(state)


def render_comparison_result(comparison: dict) -> None:
    winner = best_model_overall(comparison)
    results = comparison.get("results", {})

    st.markdown("### Latest Comparison")
    st.markdown(
        "<div class='metric-grid'>"
        + metric_card("Best Overall", format_model_name(winner) if winner else "-", "Weighted benchmark winner")
        + metric_card("Models Run", str(len(comparison.get("models_run", []))), "Models included in the comparison")
        + metric_card(
            "Successful Runs",
            str(len([item for item in results.values() if item.get("status") == "success"])),
            "Models that completed without fatal errors",
        )
        + metric_card("Comparison Mode", "Multi-model", "Shared case brief benchmark"),
        "</div>",
        unsafe_allow_html=True,
    )

    summary_df = pd.DataFrame(comparison.get("summary_table", []))
    if not summary_df.empty:
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    successful = {
        model_id: result
        for model_id, result in results.items()
        if result.get("status") == "success"
    }
    if len(successful) >= 2:
        chart_df = pd.DataFrame(
            [
                {
                    "Model": format_model_name(model_id),
                    "Overall Score": result.get("overall_avg_score", 0),
                    "Rule Validity": result.get("rule_validity_rate", 0),
                    "SVI": result.get("svi", 0),
                    "Latency": result.get("total_latency_s", 0),
                }
                for model_id, result in successful.items()
            ],
        )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.bar(
                    chart_df,
                    x="Model",
                    y="Overall Score",
                    color="Model",
                    title="Overall IRAC Score",
                ),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                px.bar(
                    chart_df,
                    x="Model",
                    y="SVI",
                    color="Model",
                    title="Strategy Vulnerability Index",
                ),
                use_container_width=True,
            )

    for model_id in comparison.get("models_run", []):
        result = results.get(model_id, {})
        with st.expander(format_model_name(model_id)):
            if result.get("status") != "success":
                st.error(result.get("error_message", "Unknown error"))
            else:
                render_single_result(model_id, result.get("_state", {}))


def render_batch_result(model: str, validation: dict) -> None:
    batch_metrics = validation.get("batch_metrics", {})
    summary_stats = batch_metrics.get("summary_stats", {})
    wilcoxon = validation.get("wilcoxon_test", {})

    st.markdown("### Latest Batch Validation")
    st.markdown(
        "<div class='metric-grid'>"
        + metric_card("Model", format_model_name(model), "Validation model")
        + metric_card("Scenarios", str(validation.get("n_successful", 0)), "Successful scenario executions")
        + metric_card("Alignment", f"{batch_metrics.get('outcome_alignment_pct', 0)}%", "Expected loser had higher SVI")
        + metric_card("Rule Validity", f"{summary_stats.get('mean_rule_validity_rate', 0)}%", "Mean verified citation rate")
        + "</div>",
        unsafe_allow_html=True,
    )

    if "p_value" in wilcoxon:
        st.info(
            f"Wilcoxon statistic {wilcoxon['statistic']} with p-value {wilcoxon['p_value']} "
            f"across {wilcoxon['n_pairs']} paired outcomes.",
        )
    elif wilcoxon.get("error"):
        st.warning(wilcoxon["error"])

    rows = []
    for case in validation.get("per_case_results", []):
        svi = case.get("svi", {})
        rows.append(
            {
                "Case": case.get("case_id"),
                "Ground Truth": case.get("ground_truth"),
                "Employer SVI": svi.get("employer_svi", 0),
                "Employee SVI": svi.get("employee_svi", 0),
                "Divergence": case.get("divergence"),
                "Rule Validity": case.get("rule_validity_rate"),
            },
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_history_tab(store: SupabaseStore) -> None:
    st.markdown("### Saved History")
    records = store.list_analyses(app_user()["id"], limit=30)
    if not records:
        st.info("You do not have any saved analyses yet.")
        return

    lookup = {record["id"]: record for record in records}
    selected_id = st.selectbox(
        "Open a saved run",
        options=list(lookup),
        format_func=lambda run_id: (
            f"{friendly_timestamp(lookup[run_id].get('created_at'))} | "
            f"{lookup[run_id].get('run_mode', 'single').title()} | "
            f"{lookup[run_id].get('title', 'Untitled run')}"
        ),
    )
    record = lookup[selected_id]
    summary = record.get("summary", {})

    st.markdown(
        "<div class='metric-grid'>"
        + metric_card("Saved Run", record.get("title", "-"), friendly_timestamp(record.get("created_at")))
        + metric_card("Mode", str(record.get("run_mode", "-")).title(), f"Status: {record.get('status', '-')}")
        + metric_card("Model", format_model_name(record.get("model", "-")), "Stored with the run")
        + metric_card("Highlights", ", ".join(f"{key}: {value}" for key, value in list(summary.items())[:2]) or "-", "Saved summary snapshot")
        + "</div>",
        unsafe_allow_html=True,
    )

    if record.get("run_mode") == "compare":
        render_comparison_result(record.get("result", {}))
    elif record.get("run_mode") == "batch":
        render_batch_result(record.get("model", "-"), record.get("result", {}))
    else:
        render_single_result(record.get("model", "-"), record.get("result", {}))


def render_research_tab() -> None:
    research = load_research_results()
    batch_metrics = research.get("batch_metrics", {})
    summary_stats = batch_metrics.get("summary_stats", {})
    wilcoxon = research.get("wilcoxon_test", {})

    st.markdown(
        """
        <div class="hero">
            <span class="eyebrow">Research Snapshot</span>
            <h2>Benchmark data ships with the app</h2>
            <p class="hero-copy">
                This tab surfaces the bundled research dataset so the deployed app still has a useful narrative layer
                even before a user runs their first live analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='metric-grid'>"
        + metric_card("Outcome Alignment", f"{batch_metrics.get('outcome_alignment_pct', 0)}%", "Held-out scenario agreement")
        + metric_card("Mean Divergence", str(summary_stats.get("mean_divergence", 0)), "Adversarial separation score")
        + metric_card("Mean Rule Validity", f"{summary_stats.get('mean_rule_validity_rate', 0)}%", "Average verified citation rate")
        + metric_card(
            "Wilcoxon p-value",
            str(wilcoxon.get("p_value", "-")),
            "Lower than 0.05 suggests significant winner/loser separation",
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    case_rows = []
    for case in research.get("per_case_results", []):
        svi = case.get("svi", {})
        case_rows.append(
            {
                "Case": case.get("case_id"),
                "Ground Truth": case.get("ground_truth"),
                "Employer SVI": svi.get("employer_svi", 0),
                "Employee SVI": svi.get("employee_svi", 0),
                "Divergence": case.get("divergence", 0),
                "Rule Validity": case.get("rule_validity_rate", 0),
            },
        )

    if case_rows:
        case_df = pd.DataFrame(case_rows)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.scatter(
                    case_df,
                    x="Divergence",
                    y="Rule Validity",
                    color="Ground Truth",
                    hover_name="Case",
                    title="Divergence vs Rule Validity",
                ),
                use_container_width=True,
            )
        with right:
            svi_long = case_df.melt(
                id_vars=["Case", "Ground Truth", "Divergence", "Rule Validity"],
                value_vars=["Employer SVI", "Employee SVI"],
                var_name="Side",
                value_name="SVI",
            )
            st.plotly_chart(
                px.box(
                    svi_long,
                    x="Side",
                    y="SVI",
                    color="Side",
                    title="SVI Distribution by Side",
                ),
                use_container_width=True,
            )
        st.dataframe(case_df, use_container_width=True, hide_index=True)


def render_setup_tab(store: SupabaseStore) -> None:
    ok, message = store.healthcheck()
    provider_rows = []
    for env_key, provider in PROVIDER_SETTINGS:
        configured = any(
            provider_env_key_for_model(model_id) == env_key and is_model_available(model_id)
            for model_id in AVAILABLE_MODELS
        )
        provider_rows.append(
            {
                "Provider": provider,
                "Configured": "Yes" if configured else "No",
                "Key": env_key,
            },
        )

    st.markdown("### Deployment Checklist")
    st.markdown(
        """
        <div class="panel">
            <div class="setup-list">
                1. Create the Supabase tables with the SQL in <code>supabase/schema.sql</code>.<br/>
                2. Add <code>SUPABASE_URL</code> and <code>SUPABASE_SERVICE_ROLE_KEY</code> to your Streamlit secrets.<br/>
                3. Add at least one provider key so the pipeline can call an LLM.<br/>
                4. Optionally build or mount a Chroma index for grounded case retrieval.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(pd.DataFrame(provider_rows), use_container_width=True, hide_index=True)

    st.markdown(
        "<div class='metric-grid'>"
        + metric_card("Supabase", "Connected" if ok else "Issue", message)
        + metric_card("RAG Index", "Ready" if index_ready() else "Not Ready", f"{collection_size()} chunks detected")
        + metric_card("Default Model", get_default_model(), "Model used when none is selected")
        + metric_card("Recommended Entry Point", "streamlit run app.py", "Single deployable Streamlit app")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Secrets Template")
    st.code(SECRETS_EXAMPLE_PATH.read_text(encoding="utf-8"), language="toml")

    st.markdown("#### Supabase Schema")
    st.code(SCHEMA_PATH.read_text(encoding="utf-8"), language="sql")


def render_workspace_tab(store: SupabaseStore) -> None:
    scenarios = load_scenarios()
    available_models = configured_models()

    single_tab, compare_tab, batch_tab = st.tabs(
        ["Single Analysis", "Compare Models", "Batch Validation"],
    )

    with single_tab:
        st.markdown("### Analyze a Case")
        if not available_models:
            st.error("No model providers are configured yet. Add provider keys in Streamlit secrets first.")
        scenario_name = st.selectbox("Scenario", options=list(scenarios), key="single_scenario")
        ensure_scenario_state("single_case_brief", "_last_single_scenario", scenario_name, scenarios)

        selected_model = st.selectbox(
            "Model",
            options=available_models or list(AVAILABLE_MODELS),
            format_func=format_model_name,
            index=default_model_index(available_models) if available_models else 0,
        )
        custom_model = st.text_input(
            "Custom model id (optional)",
            placeholder="Use this if your deployment uses a model not listed above.",
        )
        effective_model = custom_model.strip() or selected_model

        if scenarios[scenario_name].get("ground_truth_outcome"):
            st.caption(f"Ground truth for the sample scenario: {scenarios[scenario_name]['ground_truth_outcome']}")

        st.text_area(
            "Case brief",
            key="single_case_brief",
            height=220,
            placeholder="Describe the employee, employer, facts, evidence, and termination context.",
        )

        if st.button("Run analysis", type="primary", use_container_width=True):
            env_key = provider_env_key_for_model(effective_model)
            if not env_key or not is_model_available(effective_model):
                st.error(f"The provider key for `{effective_model}` is not configured.")
            elif not st.session_state["single_case_brief"].strip():
                st.error("Enter a case brief first.")
            else:
                with st.spinner(f"Running the ADVOCATE pipeline with {effective_model}..."):
                    state = run_pipeline(st.session_state["single_case_brief"], model=effective_model)
                    st.session_state["latest_single_result"] = {
                        "model": effective_model,
                        "state": state,
                    }
                    summary = pipeline_summary(state, effective_model)
                    save_run(
                        store,
                        run_mode="single",
                        title=brief_title(st.session_state["single_case_brief"]),
                        model=effective_model,
                        case_brief=st.session_state["single_case_brief"],
                        summary=summary,
                        result=state,
                    )

        latest_single = st.session_state.get("latest_single_result")
        if latest_single:
            render_single_result(latest_single["model"], latest_single["state"])

    with compare_tab:
        st.markdown("### Benchmark Multiple Models")
        compare_options = available_models if available_models else list(AVAILABLE_MODELS)
        compare_scenario = st.selectbox("Scenario", options=list(scenarios), key="compare_scenario")
        ensure_scenario_state("compare_case_brief", "_last_compare_scenario", compare_scenario, scenarios)
        selected_models = st.multiselect(
            "Models to compare",
            options=compare_options,
            default=compare_options[:2],
            format_func=format_model_name,
        )
        st.text_area(
            "Shared case brief",
            key="compare_case_brief",
            height=200,
            placeholder="Use the same brief for all selected models.",
        )
        if st.button("Run comparison", use_container_width=True):
            if len(selected_models) < 2:
                st.error("Select at least two models to compare.")
            elif not st.session_state["compare_case_brief"].strip():
                st.error("Enter a case brief to compare.")
            else:
                with st.spinner("Running comparison benchmark..."):
                    comparison = run_comparison(
                        case_brief=st.session_state["compare_case_brief"],
                        model_ids=selected_models,
                    )
                    st.session_state["latest_comparison"] = comparison
                    save_run(
                        store,
                        run_mode="compare",
                        title=brief_title(st.session_state["compare_case_brief"], prefix="Comparison - "),
                        model=", ".join(selected_models),
                        case_brief=st.session_state["compare_case_brief"],
                        summary=comparison_summary(comparison),
                        result=comparison,
                    )

        latest_comparison = st.session_state.get("latest_comparison")
        if latest_comparison:
            render_comparison_result(latest_comparison)

    with batch_tab:
        st.markdown("### Batch Validation")
        st.caption("This reruns the held-out scenarios. It is useful for evaluation, but it will cost more tokens.")
        batch_model = st.selectbox(
            "Validation model",
            options=available_models or list(AVAILABLE_MODELS),
            format_func=format_model_name,
            key="batch_model",
        )
        if st.button("Run batch validation", use_container_width=True):
            if not is_model_available(batch_model):
                st.error("Configure the provider key for the selected model first.")
            else:
                with st.spinner("Running held-out validation scenarios..."):
                    validation = run_validation(str(SCENARIOS_DIR), model=batch_model)
                    st.session_state["latest_batch_result"] = {
                        "model": batch_model,
                        "result": validation,
                    }
                    save_run(
                        store,
                        run_mode="batch",
                        title=f"Batch validation - {batch_model}",
                        model=batch_model,
                        case_brief="Held-out validation set",
                        summary=batch_summary(validation, batch_model),
                        result=validation,
                    )

        latest_batch = st.session_state.get("latest_batch_result")
        if latest_batch:
            render_batch_result(latest_batch["model"], latest_batch["result"])


def render_sidebar(store: SupabaseStore) -> None:
    user = app_user()
    ok, message = store.healthcheck()
    with st.sidebar:
        st.markdown("## ADVOCATE")
        st.caption("Authenticated legal strategy workspace")
        st.markdown(f"**Signed in as:** `{user['username']}`")
        if st.button("Log out", use_container_width=True):
            st.session_state["auth_user"] = None
            st.session_state["latest_single_result"] = None
            st.session_state["latest_comparison"] = None
            st.session_state["latest_batch_result"] = None
            st.rerun()

        st.divider()
        st.markdown("### Session API Keys")
        st.caption("Paste your own provider keys here to use them for this session only.")
        for env_key, provider in PROVIDER_SETTINGS:
            session_value = st.session_state.get(env_key, "")
            using_default = not session_value and bool(get_setting(env_key))
            note = (
                "Using this session key"
                if session_value
                else ("Using deployment default" if using_default else "Not configured")
            )
            st.text_input(
                f"{provider} API Key",
                key=env_key,
                type="password",
                placeholder=f"Paste a {provider} key for this session only",
                help=f"Overrides the deployment-level {env_key} for your current session only.",
            )
            st.caption(note)

        if st.button("Clear Session API Keys", use_container_width=True):
            clear_session_provider_keys()
            st.rerun()

        st.divider()
        st.markdown("### System Status")
        st.write(
            {
                "Supabase": "Connected" if ok else f"Issue: {message}",
                "RAG index": f"{collection_size()} chunks" if index_ready() else "Not built",
                "Configured models": len(configured_models()),
            },
        )


def render_missing_supabase() -> None:
    st.markdown(
        """
        <div class="hero">
            <span class="eyebrow">Setup Required</span>
            <h1>ADVOCATE needs Supabase before anyone can sign in</h1>
            <p class="hero-copy">
                This deployment uses Supabase as the persistent user and run-history database. Add the
                required secrets, apply the bundled SQL schema, and then reload the app.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(SECRETS_EXAMPLE_PATH.read_text(encoding="utf-8"), language="toml")
    st.code(SCHEMA_PATH.read_text(encoding="utf-8"), language="sql")


def render_auth_page(store: SupabaseStore) -> None:
    ok, message = store.healthcheck()
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown(
            """
            <div class="hero">
                <span class="eyebrow">Deploy Ready</span>
                <h1>ADVOCATE now ships with persistent profiles and saved case history</h1>
                <p class="hero-copy">
                    Create a profile with just a username and password, sign in, and keep your case work in
                    Supabase instead of losing it on every app restart.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='metric-grid'>"
            + metric_card("Authentication", "Username + Password", "Simple profile creation for each user")
            + metric_card("Persistence", "Supabase", "Run history and profiles survive redeploys")
            + metric_card("Frontend", "Streamlit", "Single-script deployment flow")
            + metric_card("Backend", "ADVOCATE Pipeline", "Existing legal analysis engine preserved")
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="panel">
                <span class="status-pill">Supabase status</span>
                <p class="subtle" style="margin-top:0.8rem;">
                    {"Connection healthy." if ok else f"Connection issue: {message}"}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown("### Sign in or create a profile")
        sign_in_tab, sign_up_tab = st.tabs(["Sign in", "Create profile"])

        with sign_in_tab:
            with st.form("sign_in_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", use_container_width=True)
            if submitted:
                try:
                    user = store.authenticate_user(username, password)
                    st.session_state["auth_user"] = asdict(user)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        with sign_up_tab:
            with st.form("sign_up_form"):
                new_username = st.text_input("Username", key="new_username")
                new_password = st.text_input("Password", type="password", key="new_password")
                submitted = st.form_submit_button("Create profile", use_container_width=True)
            if submitted:
                try:
                    user = store.create_user(new_username, new_password)
                    st.session_state["auth_user"] = asdict(user)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        st.markdown("</div>", unsafe_allow_html=True)


def render_authenticated_app(store: SupabaseStore) -> None:
    render_sidebar(store)
    user = app_user()
    st.markdown(
        f"""
        <div class="hero">
            <span class="eyebrow">Workspace</span>
            <h1>Welcome back, {user["username"]}</h1>
            <p class="hero-copy">
                Run new case analyses, compare models, revisit saved work, and inspect the bundled research
                benchmark from a single authenticated Streamlit deployment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    workspace_tab, history_tab, research_tab, setup_tab = st.tabs(
        ["Workspace", "My History", "Research", "Setup"],
    )

    with workspace_tab:
        render_workspace_tab(store)
    with history_tab:
        render_history_tab(store)
    with research_tab:
        render_research_tab()
    with setup_tab:
        render_setup_tab(store)


def main() -> None:
    st.set_page_config(
        page_title="ADVOCATE",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    init_session_state()

    if not supabase_is_configured():
        render_missing_supabase()
        return

    store = get_store()
    if not app_user():
        render_auth_page(store)
        return

    render_authenticated_app(store)


if __name__ == "__main__":
    main()

"""
app.py  —  ADVOCATE Streamlit UI
Adversarial Pre-Trial Simulation with Multi-Model Comparison
"""

import os
import sys
import json
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ADVOCATE",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title { font-size:2.4rem; font-weight:700; color:#1a2744; margin-bottom:0; }
.subtitle   { font-size:1rem; color:#5a6a8a; margin-top:0; }
.provider-openai    { color:#10a37f; font-weight:600; }
.provider-anthropic { color:#c96442; font-weight:600; }
.provider-google    { color:#4285f4; font-weight:600; }
.winner-badge {
    background:#fef3c7; border:1px solid #f59e0b;
    border-radius:6px; padding:2px 8px; font-weight:600; color:#92400e;
}
.metric-card {
    background:#f8f9fc; border-radius:8px;
    padding:12px 16px; margin:4px 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar — API Keys + RAG ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ ADVOCATE")
    st.markdown("*Multi-Model Adversarial Pre-Trial Simulation*")
    st.divider()

    st.markdown("### API Keys")
    openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    anthropic_key = st.text_input("Anthropic API Key", value=os.getenv("ANTHROPIC_API_KEY", ""), type="password")
    google_key = st.text_input("Google API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key

    any_key = openai_key or anthropic_key or google_key

    st.divider()
    st.markdown("### RAG Index")
    index_path = Path(os.getenv("CHROMA_PERSIST_PATH", "./advocate/data/chroma_db"))
    if index_path.exists():
        try:
            from advocate.rag.retriever import collection_size
            n = collection_size()
            st.success(f"Index ready: {n:,} chunks")
        except Exception as e:
            st.warning(f"Index check failed: {e}")
    else:
        st.warning("Index not built yet.")

    if st.button("Build / Rebuild Index", use_container_width=True):
        with st.spinner("Fetching CourtListener opinions…"):
            try:
                from advocate.rag.build_index import main as build_main
                build_main()
                st.success("Index built!")
                st.rerun()
            except Exception as e:
                st.error(f"Build failed: {e}")

    # Show which Gemini models are actually accessible with the current key
    if google_key:
        st.divider()
        st.markdown("### Available Gemini Models")
        if st.button("List models for my key", use_container_width=True):
            with st.spinner("Querying Gemini API…"):
                try:
                    from advocate.llm.client import list_gemini_models
                    available_gemini = list_gemini_models()
                    if available_gemini:
                        st.session_state["gemini_models"] = available_gemini
                    else:
                        st.warning("No models found — check your GOOGLE_API_KEY.")
                except Exception as e:
                    st.error(f"Could not list models: {e}")

        if "gemini_models" in st.session_state:
            for m in st.session_state["gemini_models"]:
                st.caption(f"• {m}")

    st.divider()
    st.markdown("""
**Providers supported**
- 🟢 OpenAI: GPT-4o, GPT-4o Mini, GPT-4 Turbo
- 🟠 Anthropic: Claude Sonnet/Opus/Haiku
- 🔵 Google: Gemini 2.0 Flash / Flash Lite, 1.5 Pro / Flash

Pipeline: LangGraph · RAG: CourtListener + ChromaDB
""")


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">⚖️ ADVOCATE</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Adversarial Verdict Analysis through Coordinated Agent-based Trial Emulation '
    '— Multi-Model Benchmark</p>',
    unsafe_allow_html=True,
)
st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_single, tab_compare, tab_batch = st.tabs([
    "Single Model Run", "Multi-Model Comparison", "Batch Validation"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Model Run
# ════════════════════════════════════════════════════════════════════════════
with tab_single:
    from advocate.llm.client import AVAILABLE_MODELS, is_model_available

    col_cfg, col_brief = st.columns([1, 2])

    with col_cfg:
        st.markdown("### Model & Scenario")

        # Model picker
        available = [m for m in AVAILABLE_MODELS if is_model_available(m)]
        all_models = list(AVAILABLE_MODELS.keys())
        model_display = {
            m: f"{AVAILABLE_MODELS[m]['display']} ({AVAILABLE_MODELS[m]['provider']})"
            for m in all_models
        }
        selected_model = st.selectbox(
            "Model",
            all_models,
            format_func=lambda m: model_display[m],
            index=0,
        )
        if not is_model_available(selected_model):
            st.warning(f"Set `{AVAILABLE_MODELS[selected_model]['env_key']}` in sidebar to use this model.")

        # Scenario loader
        scenario_dir = Path("./advocate/data/test_scenarios")
        scenario_files = sorted(scenario_dir.glob("*.json")) if scenario_dir.exists() else []
        scenario_names = ["Custom input"] + [f.stem for f in scenario_files]
        selected_scenario = st.selectbox("Scenario", scenario_names)

        if selected_scenario != "Custom input":
            with open(scenario_dir / f"{selected_scenario}.json") as f:
                sd = json.load(f)
            default_brief = sd.get("case_brief", "")
            st.info(f"Ground truth: **{sd.get('ground_truth_outcome', '?')}**")
        else:
            default_brief = ""

    with col_brief:
        st.markdown("### Case Brief")
        case_brief_single = st.text_area(
            "Enter the case brief:",
            value=default_brief,
            height=280,
            placeholder="Describe the case: parties, facts, evidence, termination reason, jurisdiction…",
        )

    run_single = st.button(
        "Run Pipeline",
        type="primary",
        disabled=not (is_model_available(selected_model) and case_brief_single.strip()),
    )

    if run_single:
        with st.spinner(f"Running 5-agent pipeline with **{model_display[selected_model]}**…"):
            t0 = time.time()
            try:
                from advocate.pipeline.advocate_graph import run_pipeline
                state = run_pipeline(case_brief_single, model=selected_model)
                elapsed = round(time.time() - t0, 1)
                st.session_state["single_result"] = state
                st.session_state["single_model"] = selected_model
                st.success(f"Done in {elapsed}s. Results below.")
                if state.get("errors"):
                    st.warning(f"Warnings: {state['errors']}")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.exception(e)

    if "single_result" in st.session_state:
        state = st.session_state["single_result"]
        model_used = st.session_state.get("single_model", "")
        parsed = state.get("parsed_case", {})
        evaluation = state.get("evaluation", {})
        gap = state.get("gap_report", {})

        st.divider()
        st.markdown(f"#### Results — `{model_used}`")

        # Case summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Plaintiff", parsed.get("plaintiff", "—"))
        c2.metric("Defendant", parsed.get("defendant", "—"))
        c3.metric("Jurisdiction", parsed.get("jurisdiction", "—"))
        c4.metric("Employment Type", parsed.get("employment_type", "—"))

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**IRAC Scores**")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Employer Avg", f"{evaluation.get('employer_avg', 0)}/5")
            s2.metric("Employee Avg", f"{evaluation.get('employee_avg', 0)}/5")
            s3.metric("Stronger Side", evaluation.get("stronger_side", "—").upper())
            s4.metric("Delta", evaluation.get("score_delta", 0))

        with col_r:
            st.markdown("**Strategy Gap Report**")
            g1, g2, g3 = st.columns(3)
            g1.metric("SVI", f"{gap.get('svi', 0)}%")
            g2.metric("Weaker Side", gap.get("weaker_side", "—").upper())
            g3.metric("Unrebutted Claims", f"{gap.get('unrebutted_count', 0)}/{gap.get('total_opponent_claims', 0)}")

        if gap.get("overall_strategy_assessment"):
            st.info(gap["overall_strategy_assessment"])
        if gap.get("top_priority_action"):
            st.warning(f"**Top Priority:** {gap['top_priority_action']}")

        # Arguments side by side
        col_emp, col_ee = st.columns(2)
        with col_emp:
            st.markdown("**Employer Claims**")
            for c in state.get("employer_args", {}).get("claims", []):
                with st.expander(f"[{c.get('claim_id')}] {c.get('issue', '')[:70]}"):
                    st.markdown(f"**Rule:** {c.get('rule', '')}")
                    st.markdown(f"**Cited:** `{c.get('cited_case', '')}`")
                    st.markdown(f"**Application:** {c.get('application', '')}")
                    st.markdown(f"**Conclusion:** {c.get('conclusion', '')}")
        with col_ee:
            st.markdown("**Employee Claims**")
            for c in state.get("employee_args", {}).get("claims", []):
                with st.expander(f"[{c.get('claim_id')}] {c.get('issue', '')[:70]}"):
                    st.markdown(f"**Rule:** {c.get('rule', '')}")
                    st.markdown(f"**Cited:** `{c.get('cited_case', '')}`")
                    st.markdown(f"**Application:** {c.get('application', '')}")
                    st.markdown(f"**Conclusion:** {c.get('conclusion', '')}")

        # Gap list
        gaps = gap.get("gaps", [])
        if gaps:
            st.markdown("**Ranked Vulnerabilities**")
            for g in sorted(gaps, key=lambda x: x.get("gap_rank", 99)):
                sev = g.get("severity", "MEDIUM")
                icon = "🔴" if sev == "HIGH" else ("🟡" if sev == "MEDIUM" else "🟢")
                with st.expander(f"#{g.get('gap_rank')} {icon} [{sev}] {g.get('opponent_issue', '')[:75]}"):
                    st.markdown(f"**Why dangerous:** {g.get('why_dangerous', '')}")
                    st.markdown(f"**Suggested counter:** {g.get('suggested_counter', '')}")

        with st.expander("Full JSON output"):
            st.json({k: v for k, v in state.items() if k != "_state"})


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Multi-Model Comparison
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    from advocate.llm.client import AVAILABLE_MODELS, is_model_available

    st.markdown("### Select Models to Compare")
    st.markdown(
        "Run the same case through multiple LLMs and compare them on "
        "IRAC quality, citation validity, adversarial divergence, SVI, and latency."
    )

    # Model selection — grouped by provider
    providers = {"OpenAI": "🟢", "Anthropic": "🟠", "Google": "🔵"}
    selected_models = []

    cols = st.columns(len(providers))
    for col, (provider, icon) in zip(cols, providers.items()):
        with col:
            st.markdown(f"**{icon} {provider}**")
            provider_models = [m for m, v in AVAILABLE_MODELS.items() if v["provider"] == provider]
            for m in provider_models:
                available = is_model_available(m)
                label = AVAILABLE_MODELS[m]["display"]
                checked = st.checkbox(
                    label,
                    key=f"cmp_{m}",
                    value=(m == "gpt-4o" and available),
                    disabled=not available,
                    help=f"Requires {AVAILABLE_MODELS[m]['env_key']}" if not available else AVAILABLE_MODELS[m]["description"],
                )
                if checked:
                    selected_models.append(m)

    st.divider()

    # Scenario / brief input
    col_scen, col_b = st.columns([1, 2])
    with col_scen:
        scenario_dir = Path("./advocate/data/test_scenarios")
        scenario_files = sorted(scenario_dir.glob("*.json")) if scenario_dir.exists() else []
        scenario_names_cmp = ["Custom input"] + [f.stem for f in scenario_files]
        sel_scen_cmp = st.selectbox("Scenario", scenario_names_cmp, key="cmp_scenario")
        if sel_scen_cmp != "Custom input":
            with open(scenario_dir / f"{sel_scen_cmp}.json") as f:
                sd_cmp = json.load(f)
            default_cmp = sd_cmp.get("case_brief", "")
            st.info(f"Ground truth: **{sd_cmp.get('ground_truth_outcome', '?')}**")
        else:
            default_cmp = ""

    with col_b:
        st.markdown("### Case Brief")
        case_brief_cmp = st.text_area(
            "Case brief for comparison:",
            value=default_cmp,
            height=220,
            key="cmp_brief",
            placeholder="Enter the employment wrongful termination case brief…",
        )

    if len(selected_models) < 2:
        st.info("Select at least 2 models above to run a comparison.")

    run_cmp = st.button(
        f"Compare {len(selected_models)} Models",
        type="primary",
        disabled=(len(selected_models) < 2 or not case_brief_cmp.strip()),
    )

    if run_cmp:
        progress_area = st.empty()
        status_map = {m: "queued" for m in selected_models}

        def update_progress(model_id, status):
            status_map[model_id] = status
            lines = [
                f"{'✅' if s == 'done' else ('⏳' if s == 'running' else ('❌' if 'error' in str(s) else '🕐'))} "
                f"`{m}` — {s}"
                for m, s in status_map.items()
            ]
            progress_area.markdown("\n\n".join(lines))

        with st.spinner("Running comparison pipeline…"):
            try:
                from advocate.evaluation.compare_models import run_comparison, best_model_overall
                comparison = run_comparison(
                    case_brief=case_brief_cmp,
                    model_ids=selected_models,
                    progress_callback=update_progress,
                )
                st.session_state["comparison"] = comparison
                progress_area.empty()
                st.success("Comparison complete!")
            except Exception as e:
                st.error(f"Comparison failed: {e}")
                st.exception(e)

    if "comparison" in st.session_state:
        from advocate.evaluation.compare_models import best_model_overall
        cmp = st.session_state["comparison"]
        results = cmp["results"]
        summary = cmp["summary_table"]
        winner = cmp["winner"]
        rankings = cmp["rankings"]

        st.divider()
        st.markdown("## Comparison Results")

        # Best model overall
        best = best_model_overall(cmp)
        if best:
            best_info = AVAILABLE_MODELS.get(best, {})
            st.markdown(
                f"### Overall Best Model: "
                f"**{best_info.get('display', best)}** ({best_info.get('provider', '')})",
            )
            st.caption("Composite score weighted: IRAC quality 35% · Rule validity 25% · Divergence 20% · SVI 10% · Speed 10%")

        st.divider()

        # ── Summary table ──────────────────────────────────────────────────
        st.markdown("### Score Summary Table")

        import pandas as pd

        # Build display dataframe with winner highlighting
        df_rows = []
        for row in summary:
            m = row["Model"]
            info = AVAILABLE_MODELS.get(m, {})
            display_name = info.get("display", m)
            provider = info.get("provider", "")
            df_rows.append({
                "Model": f"{display_name} ({provider})",
                "Status": row["Status"],
                "Overall Score (/5)": row["Overall Score"],
                "Rule Validity (%)": row["Rule Validity %"],
                "Divergence (0-1)": row["Divergence"],
                "SVI (%) ↓": row["SVI %"],
                "Issue Clarity": row["Issue Clarity"],
                "App. Logic (/2)": row["App. Logic"],
                "Rebuttal (/1)": row["Rebuttal"],
                "Latency (s) ↓": row["Latency (s)"],
            })

        df = pd.DataFrame(df_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.caption("↓ = lower is better for that metric")

        # ── Per-metric winner badges ───────────────────────────────────────
        st.divider()
        st.markdown("### Per-Metric Winners")

        metric_labels = {
            "overall_avg_score": ("Overall IRAC Score", "higher"),
            "rule_validity_rate": ("Rule Validity Rate", "higher"),
            "adversarial_divergence": ("Adversarial Divergence", "higher"),
            "svi": ("Strategy Vulnerability Index", "lower"),
            "issue_clarity_avg": ("Issue Clarity", "higher"),
            "application_logic_avg": ("Application Logic", "higher"),
            "rebuttal_coverage_avg": ("Rebuttal Coverage", "higher"),
            "total_latency_s": ("Response Speed", "lower"),
        }

        cols = st.columns(4)
        for i, (metric, (label, direction)) in enumerate(metric_labels.items()):
            w = winner.get(metric)
            if w:
                w_info = AVAILABLE_MODELS.get(w, {})
                w_display = w_info.get("display", w)
                w_val = results[w].get(metric, "—")
                cols[i % 4].metric(
                    label=f"{label} ({'↑' if direction == 'higher' else '↓'})",
                    value=w_display,
                    delta=f"{w_val}" if isinstance(w_val, (int, float)) else str(w_val),
                )

        # ── Side-by-side bar charts ────────────────────────────────────────
        st.divider()
        st.markdown("### Visual Comparison")

        successful = {m: r for m, r in results.items() if r.get("status") == "success"}
        if len(successful) >= 2:
            chart_metrics = [
                ("overall_avg_score", "Overall IRAC Score (max 5)"),
                ("rule_validity_rate", "Rule Validity Rate (%)"),
                ("adversarial_divergence", "Adversarial Divergence (0–1)"),
                ("svi", "SVI — lower is better (%)"),
                ("total_latency_s", "Latency — lower is better (s)"),
            ]

            for metric, title in chart_metrics:
                chart_data = {
                    AVAILABLE_MODELS.get(m, {}).get("display", m): r.get(metric, 0)
                    for m, r in successful.items()
                }
                st.markdown(f"**{title}**")
                st.bar_chart(chart_data)

        # ── Per-model drill-down ───────────────────────────────────────────
        st.divider()
        st.markdown("### Per-Model Argument Detail")

        for m in cmp["models_run"]:
            r = results.get(m, {})
            info = AVAILABLE_MODELS.get(m, {})
            label = f"{info.get('display', m)} ({info.get('provider', '')})"

            if r.get("status") != "success":
                with st.expander(f"❌ {label} — {r.get('error_message', 'error')}"):
                    st.error(r.get("error_message", "Unknown error"))
                continue

            state = r.get("_state", {})
            with st.expander(f"✅ {label} — Overall: {r.get('overall_avg_score')}/5 · SVI: {r.get('svi')}%"):
                emp_claims = state.get("employer_args", {}).get("claims", [])
                ee_claims = state.get("employee_args", {}).get("claims", [])

                col_e, col_ee = st.columns(2)
                with col_e:
                    st.markdown("**Employer Claims**")
                    for c in emp_claims:
                        st.markdown(
                            f"**[{c.get('claim_id')}]** {c.get('issue', '')}  \n"
                            f"*Rule:* {c.get('rule', '')}  \n"
                            f"*Cited:* `{c.get('cited_case', '')}`"
                        )
                        st.divider()
                with col_ee:
                    st.markdown("**Employee Claims**")
                    for c in ee_claims:
                        st.markdown(
                            f"**[{c.get('claim_id')}]** {c.get('issue', '')}  \n"
                            f"*Rule:* {c.get('rule', '')}  \n"
                            f"*Cited:* `{c.get('cited_case', '')}`"
                        )
                        st.divider()

                gap_r = state.get("gap_report", {})
                if gap_r.get("overall_strategy_assessment"):
                    st.info(f"**Strategy Assessment:** {gap_r['overall_strategy_assessment']}")
                if gap_r.get("top_priority_action"):
                    st.warning(f"**Top Priority:** {gap_r['top_priority_action']}")

        with st.expander("Full comparison JSON"):
            # Remove _state keys before serialising (too large)
            export = {
                k: v for k, v in cmp.items() if k != "results"
            }
            export["results"] = {
                m: {kk: vv for kk, vv in r.items() if kk != "_state"}
                for m, r in results.items()
            }
            st.json(export)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Batch Validation
# ════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("## Batch SVI Validation")
    st.markdown(
        "Run all 10 test scenarios through a single model and perform a "
        "Wilcoxon signed-rank test to validate the SVI metric against ground-truth outcomes."
    )

    from advocate.llm.client import AVAILABLE_MODELS, is_model_available

    all_models_batch = list(AVAILABLE_MODELS.keys())
    available_batch = [m for m in all_models_batch if is_model_available(m)]

    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        batch_model = st.selectbox(
            "Model for batch run",
            all_models_batch,
            format_func=lambda m: f"{AVAILABLE_MODELS[m]['display']} ({AVAILABLE_MODELS[m]['provider']})",
            index=0,
        )
        if not is_model_available(batch_model):
            st.warning(f"Set `{AVAILABLE_MODELS[batch_model]['env_key']}` in sidebar.")
        output_path = st.text_input("Save results to:", value="./validation_results.json")
        run_batch = st.button(
            "Run Batch Validation",
            type="primary",
            disabled=not is_model_available(batch_model),
        )

    if run_batch:
        with st.spinner(f"Running batch validation with {batch_model}…"):
            try:
                import os as _os
                _os.environ["ADVOCATE_MODEL"] = batch_model
                from advocate.evaluation.validate import run_validation
                batch_results = run_validation("./advocate/data/test_scenarios")
                if output_path:
                    with open(output_path, "w") as f:
                        json.dump(batch_results, f, indent=2)
                st.session_state["batch_results"] = batch_results
                st.success(f"Done — {batch_results.get('n_successful')}/{batch_results.get('n_scenarios')} scenarios succeeded.")
            except Exception as e:
                st.error(f"Batch failed: {e}")
                st.exception(e)

    if "batch_results" in st.session_state:
        br = st.session_state["batch_results"]
        bm = br.get("batch_metrics", {})
        st.divider()
        st.markdown("### Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Outcome Alignment", f"{bm.get('outcome_alignment_pct', 0)}%")
        m2.metric("Mean Divergence", bm.get("summary_stats", {}).get("mean_divergence", 0))
        m3.metric("Mean Rule Validity", f"{bm.get('summary_stats', {}).get('mean_rule_validity_rate', 0)}%")
        m4.metric("Scenarios", br.get("n_successful", 0))

        wx = br.get("wilcoxon_test", {})
        st.markdown("### Wilcoxon Signed-Rank Test")
        if "p_value" in wx:
            w1, w2, w3 = st.columns(3)
            w1.metric("Statistic", wx.get("statistic"))
            w2.metric("p-value", wx.get("p_value"))
            w3.metric("Significant (p<0.05)", "Yes" if wx.get("significant") else "No")
            st.markdown(
                f"Mean loser SVI: **{wx.get('mean_loser_svi')}%** | "
                f"Mean winner SVI: **{wx.get('mean_winner_svi')}%**"
            )
        else:
            st.warning(wx.get("error", "Test could not be computed."))

        with st.expander("Per-case breakdown"):
            st.json(br.get("per_case_results", []))

# ADVOCATE
**Adversarial Verdict Analysis through Coordinated Agent-based Trial Emulation**

A multi-agent agentic AI system that simulates adversarial pre-trial argumentation for employment wrongful termination cases — and benchmarks multiple LLMs head-to-head on structured legal reasoning.

---

## What It Does

ADVOCATE runs the same employment law case through a 5-agent pipeline and evaluates LLM outputs on four objective dimensions. The **Multi-Model Comparison** mode lets you pit GPT-4o, Claude, and Gemini against each other on the same case and see which model reasons most rigorously.

---

## Architecture

```
Case Brief Input
       │
       ▼
[Agent 1: Case Parser] ──► Structured JSON
       │
   ┌───┴───┐  (architecturally isolated — zero shared context)
   ▼       ▼
[Agent 2]  [Agent 3]
Employer   Employee
Agent      Agent
   │       │
   └───┬───┘
       ▼
[Agent 4: IRAC Evaluator] ──► Rubric scores (0–5 per claim)
       │
       ▼
[Agent 5: Strategy Gap Report] ──► SVI + ranked vulnerability list
```

**Key constraint:** Agents 2 and 3 share zero conversational context, preventing consensus collapse. Each retrieves independently from the same RAG index using differently-framed queries.

---

## Supported Models

| Provider | Models | API Key Env Var |
|---|---|---|
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` | `OPENAI_API_KEY` |
| **Anthropic** | `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| **Google** | `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash` | `GOOGLE_API_KEY` |

You only need to set the API keys for the providers you want to use. The routing is automatic based on model ID prefix.

---

## Evaluation Metrics (per model)

| Metric | Description | Direction |
|---|---|---|
| **Overall IRAC Score** | Avg rubric score across all claims (0–5) | Higher = better |
| **Rule Validity Rate** | % of cited cases verified in ChromaDB (cosine ≥ 0.75) | Higher = better |
| **Adversarial Divergence** | Cosine distance between employer/employee outputs (0–1) | Higher = better |
| **Issue Clarity** | Avg issue identification score (0–1) | Higher = better |
| **Application Logic** | Avg rule-to-facts connection score (0–2) | Higher = better |
| **Rebuttal Coverage** | Avg opponent-claim response score (0–1) | Higher = better |
| **SVI** | Strategy Vulnerability Index — % of opponent claims unrebutted | Lower = better |
| **Latency** | Total pipeline wall-clock time (seconds) | Lower = better |

### IRAC Rubric Detail

| Dimension | What It Checks | Score |
|---|---|---|
| Issue Clarity | Did the agent correctly identify the legal issue? | 0 or 1 |
| Rule Validity | Is the cited case in the RAG index? *(programmatic)* | 0 or 1 |
| Application Logic | Does the argument logically connect rule to facts? | 0, 1, or 2 |
| Rebuttal Coverage | Does this claim address the opponent's key point? | 0 or 1 |

**Rule Validity is the only dimension scored programmatically** — a cosine similarity search against ChromaDB ensures hallucinated citations score 0.

### Overall Best Model (Composite Score)

When comparing multiple models, a weighted composite score determines the overall winner:

| Component | Weight |
|---|---|
| Overall IRAC Score | 35% |
| Rule Validity Rate | 25% |
| Adversarial Divergence | 20% |
| SVI (inverted) | 10% |
| Speed (inverted) | 10% |

### Strategy Vulnerability Index (SVI)

```
SVI = (unrebutted opponent claims / total opponent claims) × 100
```
Range: 0% (fully rebutted all opponent arguments) → 100% (addressed none).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env — add whichever provider keys you have:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
#   GOOGLE_API_KEY=AIza...
```

### 3. Build the RAG index (one-time, ~5 minutes)

Fetches 60–80 US federal employment wrongful termination opinions from CourtListener (free, no API key) and stores them in ChromaDB.

```bash
python -m advocate.rag.build_index
```

### 4. Launch the app

```bash
streamlit run app.py
```

### 5. Run batch SVI validation (optional)

```bash
python -m advocate.evaluation.validate --output validation_results.json
```

---

## Project Structure

```
advocate/
├── llm/
│   └── client.py           # Unified OpenAI / Anthropic / Gemini routing
├── data/
│   ├── raw_cases/          # CourtListener JSON responses (cached)
│   ├── processed_cases/    # Cleaned opinion text
│   └── test_scenarios/     # 10 synthetic case briefs + ground truth outcomes
├── rag/
│   ├── build_index.py      # Fetch → chunk → embed → ChromaDB
│   └── retriever.py        # Cosine similarity search + citation verification
├── agents/
│   ├── parser_agent.py     # Agent 1: case brief → structured JSON
│   ├── employer_agent.py   # Agent 2: employer-side IRAC arguments
│   ├── employee_agent.py   # Agent 3: employee-side IRAC arguments
│   ├── irac_evaluator.py   # Agent 4: rubric scoring
│   └── gap_report.py       # Agent 5: SVI + vulnerability list
├── pipeline/
│   └── advocate_graph.py   # LangGraph orchestration (model threaded through state)
└── evaluation/
    ├── svi_calculator.py   # SVI, Adversarial Divergence, batch metrics
    ├── compare_models.py   # Multi-model comparison runner + composite ranking
    └── validate.py         # Wilcoxon validation harness
app.py                      # Streamlit UI (3 tabs)
requirements.txt
.env.example
```

---

## UI Tabs

| Tab | Description |
|---|---|
| **Single Model Run** | Run the pipeline with one model; see full argument detail + gap report |
| **Multi-Model Comparison** | Select 2–9 models, run same case, compare on all metrics with charts and rankings |
| **Batch Validation** | Run all 10 test scenarios, compute Wilcoxon test for SVI validity |

---

## Research Contributions

1. **Adversarial Architectural Separation** — independent context window isolation prevents consensus collapse in multi-agent legal reasoning
2. **IRAC Rubric as Agent Evaluation Framework** — structured, reproducible rubric independent of case outcome
3. **Strategy Vulnerability Index (SVI)** — novel quantitative metric for pre-trial legal strategy weakness
4. **Multi-Model LLM Benchmark** — objective comparison of GPT, Claude, and Gemini on structured legal argumentation using the same RAG-grounded evaluation framework

---

## Technology Stack

| Component | Tool |
|---|---|
| LLM Providers | OpenAI, Anthropic, Google (via unified client) |
| Orchestration | LangGraph |
| Vector Store | ChromaDB (local) |
| Embeddings | all-MiniLM-L6-v2 |
| Data Source | CourtListener REST API v4 (free) |
| Frontend | Streamlit |
| Language | Python 3.10+ |

# ADVOCATE

ADVOCATE stands for **Adversarial Verdict Analysis through Coordinated Agent-based Trial Emulation**.

It is a Streamlit application and research-oriented legal reasoning framework for **employment wrongful termination analysis**. Instead of asking one model for one answer, ADVOCATE runs a case through a **five-agent adversarial pipeline** that separates plaintiff-side and defense-side reasoning, scores both sides on a structured legal rubric, and surfaces the weakest parts of the overall strategy.

Live app: [advocate-pretrial-simulator.streamlit.app](https://advocate-pretrial-simulator.streamlit.app/)

## What This Project Is

At a high level, ADVOCATE is built for a simple question:

> If we force an AI system to reason like both sides of a dispute, score those arguments rigorously, and then identify what one side failed to answer, can we produce something more useful than a single-shot legal summary?

This project answers that by combining:

- a **five-agent legal reasoning architecture**
- an **adversarial workflow** instead of a single-response workflow
- **IRAC-based evaluation**
- a **Strategy Vulnerability Index (SVI)** for identifying strategic gaps
- optional **RAG grounding** over employment-law opinions
- a deployable **Streamlit + Supabase** application for interactive use

The current app supports:

- username/password account creation and sign-in
- persistent saved run history with Supabase
- single-case analysis
- multi-model comparison
- batch validation over the included scenario set
- a bundled research/benchmark tab
- per-session provider API key entry in the sidebar

## Why This Exists

Single-model legal outputs often collapse into one narrative. That is not ideal for adversarial legal reasoning, where the real value often lies in:

- seeing the strongest arguments for **both** sides
- detecting what one side **failed to rebut**
- identifying whether a position is persuasive because it is strong, or just because the opposing side was underdeveloped
- benchmarking different models on the **same** structured legal task

ADVOCATE is designed to be useful for:

- **pre-trial strategy exploration**
- **legal education and classroom simulations**
- **AI legal reasoning research**
- **model benchmarking**
- **stress-testing argument quality**
- **identifying vulnerabilities in a litigation narrative**

It is not a substitute for licensed legal advice. It is a structured analysis and benchmarking tool.

## Core Idea: Five-Agent Adversarial Reasoning

Instead of one model producing one answer, ADVOCATE uses five coordinated agents:

1. **Case Parser**
2. **Employer Agent**
3. **Employee Agent**
4. **IRAC Evaluator**
5. **Strategy Gap Report Agent**

That gives the system a workflow closer to adversarial legal preparation than a normal chat interaction.

## The Five-Agent Framework

### Agent 1: Case Parser

The parser converts a free-text case brief into structured legal input.

It extracts:

- plaintiff
- defendant
- employment type
- stated termination reason
- jurisdiction
- key facts
- evidence
- likely employee claims
- likely employer defenses
- protected characteristics
- timeline

Why this matters:

- downstream agents reason over a consistent case representation
- evaluation becomes more reproducible
- custom user briefs become easier to compare across runs and models

### Agent 2: Employer Agent

This agent acts like employer-side counsel and generates the strongest employer defense it can.

Its role:

- argue only for the employer
- retrieve relevant authority when available
- produce 3-5 structured IRAC claims
- cite case authority from retrieved material

### Agent 3: Employee Agent

This agent acts like employee-side counsel and generates the strongest plaintiff-side case it can.

Its role:

- argue only for the employee
- retrieve relevant authority when available
- produce 3-5 structured IRAC claims
- cite case authority from retrieved material

### Architectural Separation Between Agents 2 and 3

This is one of the most important design choices in the project.

The employer and employee agents are intentionally kept **architecturally isolated**. They do not share each other's reasoning chain while generating arguments.

Why that matters:

- it reduces consensus collapse
- it encourages genuinely divergent arguments
- it better simulates adversarial legal reasoning
- it makes the resulting gap analysis more meaningful

If both sides shared the same evolving context window, they would tend to converge on the same narrative. That is the opposite of what an adversarial legal framework needs.

### Agent 4: IRAC Evaluator

Once both sides generate arguments, the evaluator scores them using a structured rubric.

Current scoring dimensions include:

- **Issue Clarity**
- **Rule Validity**
- **Application Logic**
- **Rebuttal Coverage**

This creates a more disciplined evaluation loop than "which side sounds better?"

### Agent 5: Strategy Gap Report Agent

This final agent identifies the strategic gaps in the weaker side's case.

It produces:

- weaker side
- unrebutted opponent claims
- ranked vulnerabilities
- severity
- suggested counters
- overall strategy assessment
- top-priority action

This is where ADVOCATE becomes especially useful in practice: it does not stop at scoring. It tries to answer:

> Where is the strategy actually vulnerable?

## End-to-End Flow

```text
Raw Case Brief
    |
    v
Case Parser
    |
    v
Structured Case Representation
    |
    +--> Employer Agent
    |
    +--> Employee Agent
    |
    v
IRAC Evaluator
    |
    v
Strategy Gap Report
```

## Why the Adversarial Design Matters

The project is built around the idea that legal reasoning quality is not just about producing a plausible answer. It is about:

- identifying the issue correctly
- citing valid authority
- applying law to facts coherently
- answering the opponent's strongest claims

That makes ADVOCATE different from a generic "legal summary" tool. Its value is often in the **structure of disagreement**.

## Strategy Vulnerability Index (SVI)

ADVOCATE uses an SVI-style gap measure to estimate strategic weakness.

In simple terms:

- the more important opponent claims go unrebutted
- the more vulnerable that side's strategy is

The app uses SVI to help answer:

- which side is more exposed?
- how complete is the rebuttal coverage?
- where should trial preparation focus next?

## IRAC Evaluation

IRAC is a natural fit here because it imposes legal-argument discipline.

The evaluator looks at:

- **Issue**: Did the argument identify the legal question cleanly?
- **Rule**: Was the cited authority grounded and valid?
- **Application**: Did the agent connect the rule to the actual facts?
- **Rebuttal**: Did it meaningfully answer the opposing side?

This makes the framework useful not only for generating arguments, but also for **measuring** them.

## Retrieval and Grounding

ADVOCATE can use a local ChromaDB-backed retrieval layer populated from employment-law opinions.

When the RAG layer is available, it helps with:

- citation grounding
- legal precedent retrieval
- rule-validity scoring

When it is not available:

- the app still loads
- auth and persistence still work
- the research tab still works
- live reasoning can still run
- retrieval returns empty results and citation grounding becomes weaker

This is deliberate. The app now degrades gracefully instead of crashing at startup if optional RAG dependencies are unavailable.

## What the App Is Good For

### 1. Pre-trial reasoning support

A user can paste a brief and quickly inspect:

- strongest employer-side arguments
- strongest employee-side arguments
- which side appears stronger under the rubric
- what that side failed to address

### 2. Legal education

The framework is useful for:

- classroom simulations
- IRAC practice
- employment-law issue spotting
- argument comparison exercises

### 3. Model benchmarking

Because the same structured pipeline can be run across multiple models, ADVOCATE is useful for comparing:

- overall legal reasoning quality
- citation validity
- adversarial divergence
- vulnerability exposure
- latency

### 4. Prompt-architecture research

The system is also a useful case study in:

- agent separation
- structured evaluation
- adversarial prompting
- orchestration of multiple reasoning roles

## Current Deployment Architecture

The public app is now a deployable Streamlit application with persistent storage.

```text
Streamlit UI
    |
    +-- Supabase persistence/auth layer
    |     - app_users
    |     - analysis_runs
    |
    +-- ADVOCATE pipeline
    |     - parse_case
    |     - employer_agent
    |     - employee_agent
    |     - irac_evaluator
    |     - gap_report
    |
    +-- Optional local RAG layer
          - ChromaDB
          - employment-law case chunks
```

## Persistent Data Model

Supabase stores two main application tables:

### `app_users`

Contains:

- username
- password hash
- created timestamp
- last login timestamp

### `analysis_runs`

Contains:

- owning user
- run mode
- title
- selected model(s)
- case brief
- summary JSON
- result JSON
- timestamps

Schema file:

- [`supabase/schema.sql`](supabase/schema.sql)

## App Areas

### Workspace

Signed-in users can:

- run single-case analysis
- compare multiple models on one case
- run batch validation on bundled scenarios

### My History

Users can reopen prior runs saved in Supabase.

### Research

This tab exposes the bundled research dataset and benchmark visualizations from:

- `anthropic_research_results.json`

### Setup

This tab surfaces deployment health, including:

- Supabase status
- retrieval backend status
- embedding backend status
- configured model providers
- schema and secrets reference blocks

## Supported Model Providers

The current code supports:

- OpenAI
- Anthropic
- Google Gemini

Provider routing is handled in:

- [`advocate/llm/client.py`](advocate/llm/client.py)

Users can supply model keys in two ways:

- deployment-level secrets
- per-session keys pasted into the Streamlit sidebar

## Deployment Notes

Recommended Streamlit deployment settings:

- entrypoint: `app.py`
- Python version: `3.11`

Minimum secrets for auth + persistence:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your_backend_secret_key"
ADVOCATE_MODEL = "gpt-4o-mini"
```

If you want a deployment-wide default model provider:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your_backend_secret_key"
OPENAI_API_KEY = "sk-..."
ADVOCATE_MODEL = "gpt-4o-mini"
```

If you want users to bring their own model key:

- deploy only with the Supabase secrets
- do not add provider secrets
- users can paste their own provider key in the app sidebar

## Local Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy local config:

```bash
copy .env.example .env
```

Run the app:

```bash
streamlit run app.py
```

Optional: build the local retrieval index:

```bash
python -m advocate.rag.build_index
```

## Security Notes

- Never commit real secrets to the repository.
- Never expose backend Supabase secrets in browser-side code.
- If a backend secret was posted in chat, screenshots, or public text, rotate it immediately.
- Streamlit secrets are the right place for deployment-time secrets.

## Limitations

ADVOCATE is useful, but it has important limitations:

- it is not legal advice
- it depends on the quality of the input brief
- it can still inherit model hallucinations
- retrieval quality depends on the local index being available
- the current legal domain focus is wrongful termination / employment-law style analysis

## Key Files

- [`app.py`](app.py): main Streamlit app
- [`advocate/pipeline/advocate_graph.py`](advocate/pipeline/advocate_graph.py): 5-agent orchestration
- [`advocate/agents/parser_agent.py`](advocate/agents/parser_agent.py): case parser
- [`advocate/agents/employer_agent.py`](advocate/agents/employer_agent.py): employer-side advocate
- [`advocate/agents/employee_agent.py`](advocate/agents/employee_agent.py): employee-side advocate
- [`advocate/agents/irac_evaluator.py`](advocate/agents/irac_evaluator.py): rubric scorer
- [`advocate/agents/gap_report.py`](advocate/agents/gap_report.py): strategic vulnerability analyzer
- [`advocate/store.py`](advocate/store.py): Supabase persistence
- [`advocate/auth.py`](advocate/auth.py): username/password auth helpers
- [`advocate/settings.py`](advocate/settings.py): unified settings and session-secret helpers
- [`supabase/schema.sql`](supabase/schema.sql): database schema

## Summary

ADVOCATE is not just a UI for calling an LLM. It is a structured adversarial reasoning system for employment-law disputes.

Its main contribution is the combination of:

- role-separated adversarial generation
- structured IRAC evaluation
- vulnerability-focused strategic analysis
- deployment-ready persistent application infrastructure

That makes it useful both as:

- a practical pre-trial exploration tool
- and a research framework for studying legal reasoning quality across models

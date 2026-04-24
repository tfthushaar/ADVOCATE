# ADVOCATE

ADVOCATE is a Streamlit application for adversarial employment-law case analysis. It combines a five-agent legal reasoning pipeline with persistent user accounts and saved run history backed by Supabase.

Live app: [advocate-pretrial-simulator.streamlit.app](https://advocate-pretrial-simulator.streamlit.app/)

## Overview

The current app supports:

- Streamlit as the deployable frontend
- Supabase-backed user profiles and persistent saved history
- Username + password sign-up and sign-in
- Single-case analysis runs
- Multi-model comparison runs
- Batch validation runs over the bundled scenarios
- A research/benchmark tab using the included `anthropic_research_results.json`
- Optional per-session provider API key entry directly in the Streamlit sidebar

The underlying ADVOCATE pipeline is still the same core flow:

```text
parse_case
  -> employer_agent
  -> employee_agent
  -> irac_evaluator
  -> gap_report
```

## Live Deployment

The public deployment is here:

- [https://advocate-pretrial-simulator.streamlit.app/](https://advocate-pretrial-simulator.streamlit.app/)

### What works without a model API key

If the app is deployed with only Supabase secrets configured:

- users can create accounts
- users can sign in
- users can browse the bundled research tab
- users can use persistence and saved run history

But live ADVOCATE pipeline runs will only work when either:

- the deployment has a provider key in Streamlit secrets, or
- the signed-in user pastes their own provider key into the sidebar for that session

## Architecture

```text
Streamlit UI (app.py)
    |
    +-- Supabase persistence/auth layer
    |     - app_users
    |     - analysis_runs
    |
    +-- ADVOCATE pipeline
          parse_case
            -> employer_agent
            -> employee_agent
            -> irac_evaluator
            -> gap_report
    |
    +-- Optional RAG layer
          ChromaDB + CourtListener-derived employment-law cases
```

## Persistent Data Model

Supabase stores two app tables:

- `app_users`
  - username
  - password hash
  - created/login timestamps
- `analysis_runs`
  - owning user
  - run title
  - mode (`single`, `compare`, `batch`)
  - selected model(s)
  - case brief
  - saved summary JSON
  - full saved result JSON

Schema file:

- [`supabase/schema.sql`](supabase/schema.sql)

## Core Files

- [`app.py`](app.py): main authenticated Streamlit app
- [`app_simulation.py`](app_simulation.py): compatibility entrypoint that launches the main app
- [`advocate/store.py`](advocate/store.py): Supabase storage layer
- [`advocate/auth.py`](advocate/auth.py): username/password hashing and validation
- [`advocate/settings.py`](advocate/settings.py): environment, Streamlit secrets, and session-setting helpers
- [`advocate/pipeline/advocate_graph.py`](advocate/pipeline/advocate_graph.py): main ADVOCATE graph orchestration
- [`advocate/llm/client.py`](advocate/llm/client.py): provider routing for OpenAI, Anthropic, and Gemini
- [`supabase/schema.sql`](supabase/schema.sql): SQL required for persistent database tables
- [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example): deployment secrets template
- [`.env.example`](.env.example): local env template

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure secrets

For local development:

```bash
copy .env.example .env
```

For Streamlit deployment, add secrets in TOML format using:

- [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)

### 3. Set up Supabase

1. Create a Supabase project.
2. Open `SQL Editor`.
3. Run the SQL from [`supabase/schema.sql`](supabase/schema.sql).
4. Copy your project URL.
5. Copy your backend Supabase secret key.
6. Add both to Streamlit secrets.

Important:

- `SUPABASE_URL` should be your project URL, like `https://your-project.supabase.co`
- `SUPABASE_SERVICE_ROLE_KEY` should be a backend secret key
- do not use the publishable key for `SUPABASE_SERVICE_ROLE_KEY`

## Streamlit Secrets

### Minimum deployment secrets

This is enough to deploy the app with authentication and persistence:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your_backend_secret_key"
ADVOCATE_MODEL = "gpt-4o-mini"
```

### Deployment with a default model provider

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your_backend_secret_key"
OPENAI_API_KEY = "sk-..."
ADVOCATE_MODEL = "gpt-4o-mini"
```

You can swap `OPENAI_API_KEY` for:

- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

### Bring-your-own-key deployment

If you do not want to store a model provider key in deployment secrets, that is supported too.

In that setup:

- keep the Supabase secrets configured
- deploy without `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`
- users can paste their own provider key into the sidebar after signing in

## Local Development

### Run the app

```bash
streamlit run app.py
```

### Optional: build the local RAG index

```bash
python -m advocate.rag.build_index
```

This step is optional. Without a local Chroma index, the app still works, but retrieval-grounded citation support is weaker.

## Streamlit Community Cloud Deployment

Recommended settings:

- Entry point: `app.py`
- Python version: `3.11`
- Secrets: paste TOML into the Streamlit deployment secrets box

### Recommended deployment flow

1. Push repo updates to GitHub.
2. Connect the repo in Streamlit Community Cloud.
3. Set entrypoint to `app.py`.
4. Select Python `3.11`.
5. Paste your secrets.
6. Deploy.

## App Areas

### Workspace

Main signed-in workspace where users can:

- run a single case analysis
- compare multiple models on one case
- run batch validation on the bundled scenario set

### My History

Per-user saved history from Supabase, including:

- saved single runs
- saved comparisons
- saved batch validations

### Research

Displays the bundled benchmark dataset from:

- `anthropic_research_results.json`

### Setup

Shows deployment health, including:

- Supabase connectivity
- RAG index status
- configured providers
- secrets/schema reference blocks

## Authentication Model

The app currently uses app-managed authentication with:

- username
- password
- password hashing in Python
- Supabase as persistent storage

This is not using Supabase Auth's hosted sign-in UI. Instead, the Streamlit app manages sign-up and sign-in itself and stores user records in the `app_users` table.

## RAG and Retrieval Notes

The legal retrieval layer still uses local Chroma persistence.

Default path resolution checks:

- `./advocate/data/chroma_db`
- `./advocate/advocate/data/chroma_db`

If no index is available:

- the app still loads
- auth and persistence still work
- research tab still works
- pipeline runs may be less grounded because retrieval returns empty results

## Supported Model Providers

The current app supports:

- OpenAI
- Anthropic
- Google Gemini

Provider routing is handled in:

- [`advocate/llm/client.py`](advocate/llm/client.py)

## Security Notes

- Never expose your Supabase backend secret in browser-side code.
- Never commit real secrets to the repository.
- If a backend secret was pasted into chat, screenshots, or public text, rotate it before deployment.
- Streamlit secrets are the right place for deployment-time secret storage.

## Example Secrets Block

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your_backend_secret_key"
OPENAI_API_KEY = "sk-..."
ADVOCATE_MODEL = "gpt-4o-mini"
COURTLISTENER_API_TOKEN = "your_courtlistener_token"
```

## Repository Status

The repository is currently set up for:

- authenticated Streamlit deployment
- persistent Supabase-backed storage
- optional deployment-level model credentials
- optional per-session user-supplied model credentials

## License / Usage

No explicit license file is currently included in the repository. Add one if you want to make reuse terms explicit.

# ADVOCATE

ADVOCATE is a Streamlit application for adversarial employment-law case analysis. It now ships with:

- Streamlit as the deployable frontend
- Supabase as the persistent database for user profiles and saved run history
- Username + password sign-up and sign-in
- Saved single runs, model comparisons, and batch validations per user
- The existing 5-agent ADVOCATE legal-analysis pipeline

## What changed

The repo started as a research/demo project with local-only state and no user accounts. It has been refactored into a single authenticated Streamlit app in [`app.py`](app.py) backed by Supabase.

Persistent storage now covers:

- `app_users`: usernames and hashed passwords
- `analysis_runs`: saved run history and JSON results per user

## Architecture

```
Streamlit UI (app.py)
    |
    +-- Supabase auth/profile store
    |     - app_users
    |     - analysis_runs
    |
    +-- ADVOCATE pipeline
          parse_case
            -> employer_agent
            -> employee_agent
            -> irac_evaluator
            -> gap_report
```

## Required secrets

Use either environment variables locally or Streamlit secrets in deployment.

See:

- [`.env.example`](.env.example)
- [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)

Minimum required for the deployed app:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your-service-role-key"
OPENAI_API_KEY = "sk-..."
```

You can swap `OPENAI_API_KEY` for `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY` if you want to run a different provider instead.

## Supabase setup

1. Create a Supabase project.
2. Open the SQL editor.
3. Run the SQL in [`supabase/schema.sql`](supabase/schema.sql).
4. Add `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` to your Streamlit secrets.

The app uses the service-role key server-side from Streamlit. Do not expose it in client-side code.

## Local development

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy the env template and fill in your secrets:

```bash
copy .env.example .env
```

3. Optional: build the RAG index:

```bash
python -m advocate.rag.build_index
```

4. Run the app:

```bash
streamlit run app.py
```

## Deployment

This repo is ready for Streamlit Community Cloud or any Streamlit-compatible host.

Recommended deploy target:

- Entry point: `app.py`
- Python version: 3.10+
- Secrets: add the values from `.streamlit/secrets.toml.example`

## Main app areas

- `Workspace`: run single analyses, compare models, and batch validations
- `My History`: reopen saved runs from Supabase
- `Research`: inspect the bundled benchmark dataset
- `Setup`: verify Supabase, provider keys, and RAG status

## Notes on retrieval

The legal-retrieval layer still uses a local Chroma index. If no index is available, the app still runs, but retrieval-grounded citation support will be weaker. For the best experience, build the index before deployment or mount one at `CHROMA_PERSIST_PATH`.

## Key files

- [`app.py`](app.py)
- [`advocate/store.py`](advocate/store.py)
- [`advocate/auth.py`](advocate/auth.py)
- [`advocate/settings.py`](advocate/settings.py)
- [`supabase/schema.sql`](supabase/schema.sql)

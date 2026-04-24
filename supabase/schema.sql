create extension if not exists pgcrypto;

create table if not exists public.app_users (
    id uuid primary key default gen_random_uuid(),
    username text not null unique,
    password_hash text not null,
    created_at timestamptz not null default timezone('utc', now()),
    last_login_at timestamptz
);

alter table public.app_users enable row level security;

create table if not exists public.analysis_runs (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references public.app_users(id) on delete cascade,
    title text not null,
    model text not null,
    run_mode text not null default 'single',
    status text not null default 'success',
    case_brief text not null,
    summary jsonb not null default '{}'::jsonb,
    result jsonb not null default '{}'::jsonb,
    errors jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default timezone('utc', now())
);

alter table public.analysis_runs enable row level security;

create index if not exists analysis_runs_user_created_idx
    on public.analysis_runs (user_id, created_at desc);

comment on table public.app_users is
'Server-managed Streamlit auth records. Use the service role key from the backend only.';

comment on table public.analysis_runs is
'Persistent ADVOCATE run history keyed to the authenticated Streamlit user.';

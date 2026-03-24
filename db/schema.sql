-- =============================================================================
-- Deepfake Detector — Supabase Schema
-- Run in the Supabase SQL editor (Dashboard → SQL editor → New query)
-- =============================================================================

create extension if not exists "pgcrypto";

-- -----------------------------------------------------------------------------
-- detections
-- One row per submitted media file (image / video / audio).
-- -----------------------------------------------------------------------------
create table if not exists public.detections (
    id          uuid          primary key default gen_random_uuid(),
    job_id      text          unique not null,
    media_type  text,                                    -- 'image' | 'video' | 'audio'
    media_url   text,                                    -- public URL or storage path
    is_fake     boolean,
    confidence  float,
    heatmap_url text,                                    -- Grad-CAM overlay URL
    status      text          default 'pending',         -- pending | processing | done | error
    created_at  timestamptz   default now()
);

create index if not exists idx_detections_job_id     on public.detections (job_id);
create index if not exists idx_detections_status     on public.detections (status);
create index if not exists idx_detections_created_at on public.detections (created_at desc);

-- -----------------------------------------------------------------------------
-- agent_logs
-- One row per investigation run by the LangChain agent.
-- References detections via job_id.
-- -----------------------------------------------------------------------------
create table if not exists public.agent_logs (
    id         uuid        primary key default gen_random_uuid(),
    job_id     text        references public.detections (job_id) on delete cascade,
    tool_calls jsonb,      -- array of {tool, input, output} objects
    summary    text,       -- final agent verdict / narrative
    sources    jsonb,      -- array of URLs returned by Tavily
    created_at timestamptz default now()
);

create index if not exists idx_agent_logs_job_id     on public.agent_logs (job_id);
create index if not exists idx_agent_logs_created_at on public.agent_logs (created_at desc);

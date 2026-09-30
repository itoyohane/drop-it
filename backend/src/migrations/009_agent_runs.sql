-- Durable Agent-run baseline.  Claim fencing and versioned checkpoints are
-- deliberately added by 010 so this migration remains the original 009
-- schema used by existing deployments.
CREATE TABLE agent_runs (
    run_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    route TEXT NOT NULL,
    user_text TEXT NOT NULL,
    history TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'running',
    error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE agent_run_steps (
    run_id TEXT NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    step TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, step)
);

CREATE INDEX idx_agent_runs_scope_updated
    ON agent_runs(project_id, conversation_id, updated_at);
CREATE INDEX idx_agent_run_steps_run_created
    ON agent_run_steps(run_id, created_at);

ALTER TABLE messages ADD COLUMN agent_run_id TEXT;
CREATE UNIQUE INDEX idx_messages_agent_run_role
    ON messages(agent_run_id, role)
    WHERE agent_run_id IS NOT NULL;

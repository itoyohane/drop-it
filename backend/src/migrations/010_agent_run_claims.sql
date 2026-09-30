-- Short leases and fencing.  SqliteRepository applies this migration with a
-- compatibility-aware upgrader so databases that already received an early
-- 009 implementation (with some of these columns) are upgraded safely.
ALTER TABLE agent_runs ADD COLUMN latest_step TEXT;
ALTER TABLE agent_runs ADD COLUMN latest_version INTEGER NOT NULL DEFAULT 0;
ALTER TABLE agent_runs ADD COLUMN claim_owner TEXT;
ALTER TABLE agent_runs ADD COLUMN claim_token INTEGER NOT NULL DEFAULT 0;
ALTER TABLE agent_runs ADD COLUMN claim_expires_at REAL;

-- Rebuild the original unversioned step table.  Existing rows become version 0.
ALTER TABLE agent_run_steps RENAME TO agent_run_steps_legacy;
CREATE TABLE agent_run_steps (
    run_id TEXT NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    step TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, step, version)
);
INSERT INTO agent_run_steps (run_id, step, version, state, created_at)
SELECT run_id, step, 0, state, created_at FROM agent_run_steps_legacy;
DROP TABLE agent_run_steps_legacy;
CREATE INDEX idx_agent_run_steps_run_created
    ON agent_run_steps(run_id, created_at);

ALTER TABLE playlists ADD COLUMN agent_run_id TEXT;
CREATE UNIQUE INDEX idx_playlists_agent_run
    ON playlists(agent_run_id)
    WHERE agent_run_id IS NOT NULL;

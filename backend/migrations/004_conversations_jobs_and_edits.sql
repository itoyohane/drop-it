CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

INSERT INTO conversations (id, project_id, title, created_at, updated_at)
SELECT 'default-' || id, id, '主对话', created_at, updated_at FROM projects;

ALTER TABLE messages ADD COLUMN conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE;

UPDATE messages
SET conversation_id = 'default-' || project_id
WHERE conversation_id IS NULL;

CREATE INDEX idx_conversations_project_updated ON conversations(project_id, updated_at);
CREATE INDEX idx_messages_conversation_created ON messages(conversation_id, created_at);

CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_jobs_project_updated ON jobs(project_id, updated_at);
CREATE INDEX idx_jobs_status ON jobs(status);

ALTER TABLE tracks ADD COLUMN updated_at TEXT;
UPDATE tracks SET updated_at = created_at WHERE updated_at IS NULL;

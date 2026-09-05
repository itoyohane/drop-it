ALTER TABLE projects ADD COLUMN scope TEXT NOT NULL DEFAULT 'project';

CREATE TABLE library_sources (
    id TEXT PRIMARY KEY,
    folder_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE project_sources (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES library_sources(id) ON DELETE CASCADE,
    added_at TEXT NOT NULL,
    PRIMARY KEY (project_id, source_id)
);

CREATE TABLE source_tracks (
    source_id TEXT NOT NULL REFERENCES library_sources(id) ON DELETE CASCADE,
    track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    added_at TEXT NOT NULL,
    PRIMARY KEY (source_id, track_id)
);

CREATE INDEX idx_project_sources_project ON project_sources(project_id);
CREATE INDEX idx_source_tracks_source ON source_tracks(source_id);

INSERT INTO projects (id, name, description, created_at, updated_at, scope)
VALUES ('global-chat', '普通对话', 'Uses the complete local library', datetime('now'), datetime('now'), 'global');

INSERT INTO conversations (id, project_id, title, created_at, updated_at)
VALUES ('global-chat-default', 'global-chat', '普通对话', datetime('now'), datetime('now'));

INSERT INTO library_sources (id, folder_key, name, status, created_at, updated_at)
SELECT 'legacy-' || p.id, 'legacy:' || p.id, p.name || '（历史曲库）', 'ready', p.created_at, p.updated_at
FROM projects p
WHERE p.scope = 'project'
  AND EXISTS (SELECT 1 FROM project_tracks pt WHERE pt.project_id = p.id);

INSERT INTO project_sources (project_id, source_id, added_at)
SELECT p.id, 'legacy-' || p.id, p.created_at
FROM projects p
WHERE p.scope = 'project'
  AND EXISTS (SELECT 1 FROM project_tracks pt WHERE pt.project_id = p.id);

INSERT INTO source_tracks (source_id, track_id, added_at)
SELECT 'legacy-' || pt.project_id, pt.track_id, pt.added_at
FROM project_tracks pt;

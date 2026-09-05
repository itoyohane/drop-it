ALTER TABLE tracks RENAME TO tracks_legacy;

CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE tracks (
    id TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    filename TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    duration_sec INTEGER NOT NULL DEFAULT 0,
    bpm REAL NOT NULL DEFAULT 0,
    musical_key TEXT NOT NULL DEFAULT 'Unknown',
    camelot_key TEXT NOT NULL DEFAULT '—',
    energy REAL NOT NULL DEFAULT 0,
    bpm_confidence REAL NOT NULL DEFAULT 0,
    key_confidence REAL NOT NULL DEFAULT 0,
    analysis_status TEXT NOT NULL DEFAULT 'pending',
    analysis_error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE project_tracks (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    added_at TEXT NOT NULL,
    PRIMARY KEY (project_id, track_id)
);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_events TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE playlists (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_project_tracks_project ON project_tracks(project_id);
CREATE INDEX idx_messages_project_created ON messages(project_id, created_at);
CREATE INDEX idx_playlists_project_updated ON playlists(project_id, updated_at);

INSERT INTO projects (id, name, description, created_at, updated_at)
SELECT 'legacy-project', 'Imported library', 'Migrated from the original DropIt library', datetime('now'), datetime('now')
WHERE EXISTS (SELECT 1 FROM tracks_legacy LIMIT 1);

INSERT INTO tracks (
    id, file_hash, title, artist, filename, path, duration_sec, bpm,
    musical_key, camelot_key, energy, bpm_confidence, key_confidence,
    analysis_status, created_at
)
SELECT id, id, title, artist, substr(path, length(rtrim(path, replace(path, '\\', ''))) + 1), path,
       duration_sec, bpm, musical_key, musical_key, energy, 0, 0, 'analyzed', datetime('now')
FROM tracks_legacy;

INSERT INTO project_tracks (project_id, track_id, added_at)
SELECT 'legacy-project', id, datetime('now') FROM tracks_legacy;

DROP TABLE tracks_legacy;

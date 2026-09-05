ALTER TABLE tracks ADD COLUMN analyzer TEXT NOT NULL DEFAULT '';
ALTER TABLE tracks ADD COLUMN analysis_details TEXT NOT NULL DEFAULT '{}';
ALTER TABLE tracks ADD COLUMN embedding_status TEXT NOT NULL DEFAULT 'pending';
ALTER TABLE tracks ADD COLUMN embedding_model TEXT NOT NULL DEFAULT '';
ALTER TABLE tracks ADD COLUMN embedding_error TEXT;

-- CLAP audio vectors only. Text queries use the same encoder space in memory.
CREATE TABLE music_embeddings (
    track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (track_id, model)
);

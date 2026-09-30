CREATE TABLE tracks (
    id TEXT PRIMARY KEY,
    title TEXT,
    artist TEXT,
    path TEXT UNIQUE,
    duration_sec INTEGER,
    bpm REAL,
    musical_key TEXT,
    energy REAL,
    mood TEXT,
    role TEXT
);

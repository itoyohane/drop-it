-- Vector payloads moved out of SQLite into the persistent Chroma store.
-- Existing BLOBs cannot be trusted across providers/dimensions, so force a
-- reindex the next time the affected tracks are analyzed.
DELETE FROM music_embeddings;

UPDATE tracks
SET embedding_status = 'pending',
    embedding_model = '',
    embedding_error = '向量库已切换到 Chroma，请重新分析'
WHERE embedding_status = 'ready';

ALTER TABLE tracks ADD COLUMN description TEXT NOT NULL DEFAULT '';
ALTER TABLE tracks ADD COLUMN description_model TEXT NOT NULL DEFAULT '';

-- Essentia features and CLAP audio vectors are incompatible with the librosa text-RAG pipeline.
DELETE FROM music_embeddings;

UPDATE tracks
SET analysis_status = 'pending',
    analysis_error = '音频分析栈已升级为 librosa，请重新分析',
    analyzer = '',
    analysis_details = '{}',
    description = '',
    description_model = '',
    embedding_status = 'pending',
    embedding_model = '',
    embedding_error = NULL;

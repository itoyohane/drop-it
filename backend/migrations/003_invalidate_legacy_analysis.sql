-- The v0 prototype generated these values from a deterministic random seed.
-- Preserve track identity and metadata, but never present the old values as measured audio facts.
UPDATE tracks
SET bpm = 0,
    musical_key = 'Unknown',
    camelot_key = '—',
    energy = 0,
    analysis_status = 'pending',
    analysis_error = '旧版特征不是音频实测值，请重新导入或调用分析工具'
WHERE analysis_status = 'analyzed'
  AND bpm_confidence = 0
  AND key_confidence = 0;

-- One search, as the app runs it: embed the text, score every segment, filter, keep the top 12.
WITH q AS (
    SELECT catalog_call('function.scenario-embed-text', 'walking down the aisle between workbenches') AS v
)
SELECT s.segment_id, s.worker_id, s.clip_index, s.start_s, s.activity, s.motion,
       cosine_similarity_vector(s.embedding, q.v) AS score
FROM read_parquet('{app}/data/segments.parquet') AS s
CROSS JOIN q
WHERE s.motion >= 38
  AND s.activity = 'walking'
ORDER BY score DESC
LIMIT 12

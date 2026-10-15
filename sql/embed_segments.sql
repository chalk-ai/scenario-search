-- Batch job: embed every segment on the GPU ScalingGroup, through the Chalk function.
-- The function sends the segment's four frames to SigLIP 2 and returns their normalized mean.
-- batch.py saves the result as data/segments_embedded.parquet.
SELECT m.segment_id, m.worker_id, m.clip_index, m.start_s, m.duration_s, m.motion,
       catalog_call('function.scenario-embed-segment', m.segment_id) AS embedding
FROM read_parquet('{meta}') AS m

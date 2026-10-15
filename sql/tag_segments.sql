-- Zero-shot activity tags. Each label is one text prompt, embedded with the same model as the segments.
-- Raw cosine scores carry a per-prompt bias, so each label's scores are z-scored across all segments;
-- a segment takes its best label when that z is at least 1.5, and 'bench_work' otherwise.
WITH prompts AS (
    SELECT * FROM (VALUES
        ('walking',       'walking down the aisle between workbenches'),
        ('parts_bin',     'reaching into a bin of metal parts'),
        ('watch_tray',    'a tray full of round watch faces'),
        ('phone_pairing', 'pairing a smartwatch with a phone'),
        ('film_peeling',  'peeling protective film off small parts'),
        ('strap_fitting', 'attaching a strap to a smartwatch'),
        ('packaging',     'packing a smartwatch into a printed retail box')
    ) AS t(activity, prompt)
),
labels AS (
    SELECT activity, catalog_call('function.scenario-embed-text', prompt) AS v FROM prompts
),
scored AS (
    SELECT s.segment_id, l.activity, cosine_similarity_vector(s.embedding, l.v) AS sim
    FROM read_parquet('{embedded}') AS s
    CROSS JOIN labels AS l
),
z AS (
    SELECT segment_id, activity,
           (sim - avg(sim) OVER (PARTITION BY activity)) / stddev_pop(sim) OVER (PARTITION BY activity) AS z
    FROM scored
),
best AS (
    SELECT segment_id, activity, z,
           row_number() OVER (PARTITION BY segment_id ORDER BY z DESC) AS rank
    FROM z
)
SELECT s.segment_id, s.worker_id, s.clip_index, s.start_s, s.duration_s, s.motion,
       CASE WHEN b.z >= 1.5 THEN b.activity ELSE 'bench_work' END AS activity,
       s.embedding
FROM read_parquet('{embedded}') AS s
JOIN best AS b ON b.segment_id = s.segment_id AND b.rank = 1

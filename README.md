# Scenario search over egocentric video, on Chalk

Describe a moment in first-person video ("reaching into a bin of metal parts", "a smartwatch screen
lit up during testing") and get back the ten-second clips that match. A vision-language model on a
GPU embeds every clip once; each search is **one ChalkSQL query** that embeds the text with the same
model, ranks every clip by cosine similarity, filters on metadata and returns the top 12. The same
search also runs as a Chalk DataFrame.

Live demo: <https://chalk.ai/demos/scenario-search>

```sql
WITH q AS (
    SELECT catalog_call('function.scenario-embed-text', 'walking down the aisle between workbenches') AS v
)
SELECT s.segment_id, s.worker_id, s.clip_index, s.start_s, s.activity,
       cosine_similarity_vector(s.embedding, q.v) AS score
FROM read_parquet('https://<app>/data/segments.parquet') AS s
CROSS JOIN q
WHERE s.motion >= 38 AND s.activity = 'walking'
ORDER BY score DESC
LIMIT 12
```

## How it fits together

```
download.py ─► prep_segments.py ─► out/{clips,thumbs,frames,data}
                                        │ deploy_app.py --upload
                                        ▼
                          Chalk Volume "scenario-search-media"
                                        │ mounted at /media
                                        ▼
   scenario-search app (CPU ScalingGroup) ◄──── frames over HTTPS ──── scenario-embed (GPU ScalingGroup)
   serves clips, frames and parquet;                                    SigLIP 2: /embed/text, /embed/segments
   runs every search as one ChalkSQL query                                        ▲
           │                                                                      │
           ▼                                                                      │
   catalog_call('function.scenario-embed-text' | 'function.scenario-embed-segment')
       two thin Chalk functions (embed_functions.py) that forward to the GPU group
```

- **Footage:** the 416 clips of [Voxel51/Egocentric_10K_subset](https://huggingface.co/datasets/Voxel51/Egocentric_10K_subset)
  (Apache-2.0), a curated slice of [builddotai/Egocentric-10K](https://huggingface.co/datasets/builddotai/Egocentric-10K)
  by Build AI: eight workers on a smartwatch assembly line, three-minute head-camera clips. Each clip
  is cut into 10-second segments at 0, 30, 60, 90, 120 and 150 s (2,492 segments; clips shorter than
  three minutes yield fewer), re-encoded to 640 px h264 for browsers.
- **Model:** [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384)
  (Apache-2.0), pinned to one revision, 1,152-dim embeddings. A segment's vector is the normalized mean
  of four frames (1.25, 3.75, 6.25 and 8.75 s in); a query's vector comes from the text tower.
- **Metadata:** worker, clip, start time, a camera-motion score (mean frame difference at 4 fps) and a
  zero-shot activity tag. The tag is computed in SQL: each segment's similarity to seven activity
  prompts is z-scored per prompt across all segments, and the best prompt wins if its z is at least
  1.5 (`bench_work` otherwise). It is approximate.

## Files

| File | What it does |
| --- | --- |
| `download.py` | Downloads the 416 source clips (about 35 GB) at a pinned dataset revision |
| `prep_segments.py` | Cuts segments, transcodes clips, extracts thumbnails and frames, writes `data/segments_meta.parquet` |
| `deploy_app.py` | Uploads media to the Volume; deploys the search app (`app/`) as a CPU ScalingGroup with workload identity |
| `deploy_embedder.py` | Deploys SigLIP 2 (`embedder/server.py`) as a GPU ScalingGroup that reads frames from the app |
| `embed_functions.py`, `deploy_functions.py` | The two Chalk functions that ChalkSQL and DataFrames call by name |
| `sql/embed_segments.sql` | Batch job: every segment through `function.scenario-embed-segment` |
| `sql/tag_segments.sql` | Zero-shot activity tags with window functions |
| `sql/search.sql` | One search, as the app runs it |
| `batch.py` | Runs both batch queries and publishes the results to the Volume |
| `search_dataframe.py` | The same search with the Chalk DataFrame API, run as a job in the environment |
| `example.py` | Calls the deployed app's `/api/search` |

## Run it

Needs `ffmpeg`, [uv](https://docs.astral.sh/uv/) and your existing `chalk login`. Pick a random
`EMBED_TOKEN`; the GPU group only answers requests that carry it.

```bash
export EMBED_TOKEN=$(openssl rand -hex 16)
./download.py --out raw/
./prep_segments.py --raw raw/ --out out/
./deploy_app.py --upload out/ # media to the Volume, then the app
./deploy_embedder.py # SigLIP 2 on a GPU
./deploy_functions.py
./batch.py # embed + tag every segment
./search_dataframe.py "a tray full of round watch faces"
./example.py https://<app web_url> "hands holding a lit-up watch screen"
```

The query engine reads its function catalog when the environment is deployed: after the first
`deploy_functions.py`, deploy (or re-activate) the environment once so `catalog_call` can see the new
names. Redeploying an existing function needs no such step.

## API

`POST /api/search` with `{"preset": "<id>"}` (ids in `app/presets.json`) or
`{"query": "<text>", "filters": {"worker": "worker_003", "activity": "walking", "motion": "steady|moving"}}`
returns the rows, the exact SQL that ran and its wall time. Filter values come from closed sets, and
free text only ever lands inside one escaped SQL string literal.

## Measured

On Chalk staging (GCP), 2026-10-15, one NVIDIA L4 for the model:

| Step | Measured |
| --- | --- |
| Batch embedding, `sql/embed_segments.sql` | 2,492 segments (9,968 frames) in 201.8 s, one query: about 12 segments/s |
| Zero-shot tags, `sql/tag_segments.sql` | 4.6 s for 2,492 segments x 7 prompts, window functions included |
| One search, uncached (`/api/search`, six free-text queries) | ChalkSQL 1.22 to 1.26 s (median 1.23 s); 1.35 s at the client |
| One search, cached | 123 ms at the client |
| Model time on the GPU | about 50 ms per text query, about 0.1 s per segment (four frames) |

The batch embeddings match a local run of the same model to cosine similarity 1.0000 on every segment.

## Notes

- **Limits.** The gateway hides client addresses, so limits are replica-wide: 20 free-text searches
  and 120 searches per minute in total, four ChalkSQL queries in flight, identical searches cached
  for ten minutes. The GPU group allows 600 text embeddings per minute.
- **Batch load.** The engine sends a batch's rows to the function in parallel. Each function replica
  lets four calls reach the GPU group at a time (`EMBED_IN_FLIGHT`), and the GPU group admits four
  segment requests at a time, so queues stay short and nothing times out. Raise both for more
  throughput on a bigger GPU.
- **Filter before you embed.** The engine evaluates a `catalog_call` projection on every scanned row
  before applying `WHERE` or `LIMIT`, so filtering the batch query does not save GPU work. Filter the
  input file (or a subquery you materialize) instead.
- **Versioned parquet URLs.** The engine caches `read_parquet` results by URL, so the app and
  `batch.py` add `?v=<content hash>` whenever a file changes.
- **Frames over HTTPS.** The GPU group fetches frames from the app instead of mounting the Volume:
  in the staging cluster this was built on, GPU nodes could not mount Volumes.
- **Volume mounts are snapshots.** A ScalingGroup mounts the Volume version current at deploy time,
  which is why `batch.py` redeploys the app after each upload.
- **Search scale.** Ranking reads all 2,492 vectors (about 18 MB of parquet) per query. That is fine
  here; for millions of clips, keep the vectors in an index and push the similarity down to it.
- **Delete it:** `ScalingGroup.from_name("scenario-embed").delete()` (and `scenario-search`,
  `scenario-embed-text`, `scenario-embed-segment`) with the same `CHALK_*` environment.

## License

Code: Apache-2.0 (see `LICENSE`). Footage: Apache-2.0, from Egocentric-10K by Build AI via the
Voxel51 subset. Model weights: Apache-2.0, Google.

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported.
The examples use the current environment selected by your Chalk login.

"""Scenario search over egocentric factory video: every search is one ChalkSQL query.

The query embeds the visitor's text with a Chalk function (catalog_call), ranks the segments' stored
SigLIP 2 vectors with cosine_similarity_vector, filters on segment metadata, and returns the top
segments. The page's clips, thumbnails, frames and the embeddings parquet are served from a Chalk
Volume mounted at MEDIA_DIR.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Literal

import httpx
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

WEB = Path(os.environ.get("WEB_DIR", "/web"))
MEDIA = Path(os.environ.get("MEDIA_DIR", "/media"))
API = os.environ["CHALK_API_SERVER"].rstrip("/")
ENV_ID = os.environ["CHALK_ENVIRONMENT_ID"]


def versioned(url: str, path: Path) -> str:
    """`url` tagged with a hash of the file it serves: the engine caches read_parquet results by URL."""
    if not path.is_file():
        return url
    return f"{url}?v={hashlib.md5(path.read_bytes()).hexdigest()[:12]}"


# The parquet ChalkSQL reads: this app's own /data/segments.parquet, over HTTPS.
SEGMENTS_PARQUET = versioned(os.environ["SEGMENTS_PARQUET"], MEDIA / "data" / "segments.parquet")
MAX_QUERY_CHARS = 120
# Replica-wide budgets: the gateway hides client addresses, so limits apply to all visitors together.
FREE_TEXT_PER_MINUTE = int(os.environ.get("FREE_TEXT_PER_MINUTE", "20"))
SEARCHES_PER_MINUTE = int(os.environ.get("SEARCHES_PER_MINUTE", "120"))
PRESET_CACHE_SECONDS = 600
CACHE_SIZE = 256
TOP_K = 12

WORKERS = tuple(f"worker_{i:03d}" for i in range(1, 9))
ACTIVITIES = tuple(json.loads((WEB / "presets.json").read_text())["activities"])

SQL_TEMPLATE = """WITH q AS (
    SELECT catalog_call('function.scenario-embed-text', '{text}') AS v
)
SELECT s.segment_id, s.worker_id, s.clip_index, s.start_s, s.activity, s.motion,
       cosine_similarity_vector(s.embedding, q.v) AS score
FROM read_parquet('{parquet}') AS s
CROSS JOIN q{where}
ORDER BY score DESC
LIMIT {k}"""

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://([a-z0-9-]+\.)*chalk\.(ai|dev)|http://localhost(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)
for sub in ("clips", "thumbs", "frames", "data"):
    if (MEDIA / sub).is_dir():
        app.mount(f"/{sub}", StaticFiles(directory=MEDIA / sub), name=sub)

_client = httpx.AsyncClient(timeout=120)
_token: dict[str, Any] = {"value": None, "expires": 0.0}
_inflight = asyncio.Semaphore(4)
_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_calls: dict[str, deque[float]] = {"free": deque(), "all": deque()}


async def token() -> str:
    # chalk_identity=True mounts a short-lived workload JWT and rotates it in place, so it is re-read
    # on every call. Client credentials are only a local-run fallback.
    identity_file = os.environ.get("CHALK_WEB_IDENTITY_TOKEN_FILE")
    if identity_file and Path(identity_file).exists():
        return Path(identity_file).read_text().strip()
    if _token["value"] and time.time() < _token["expires"] - 300:
        return _token["value"]
    r = await _client.post(
        f"{API}/v1/oauth/token",
        json={
            "client_id": os.environ["CHALK_CLIENT_ID"],
            "client_secret": os.environ["CHALK_CLIENT_SECRET"],
            "grant_type": "client_credentials",
        },
    )
    r.raise_for_status()
    body = r.json()
    _token.update(value=body["access_token"], expires=time.time() + float(body.get("expires_in", 3600)))
    return _token["value"]


def sql_literal(text: str) -> str:
    # Visitor text only ever lands inside one single-quoted SQL literal: control characters are dropped
    # and quotes doubled, so it cannot end the literal.
    return re.sub(r"[\x00-\x1f\x7f]", " ", text).replace("'", "''")


class Filters(BaseModel):
    worker: Literal[WORKERS] | None = None  # type: ignore[valid-type]
    activity: Literal[ACTIVITIES] | None = None  # type: ignore[valid-type]
    motion: Literal["steady", "moving"] | None = None


class SearchRequest(BaseModel):
    preset: str | None = None
    query: str | None = Field(default=None, max_length=400)
    filters: Filters = Filters()


def where_clause(f: Filters) -> str:
    # Filter values come from closed sets validated above, never from free text.
    terms = []
    if f.worker:
        terms.append(f"s.worker_id = '{f.worker}'")
    if f.activity:
        terms.append(f"s.activity = '{f.activity}'")
    if f.motion:
        # Quartiles of mean frame difference over all segments: a steady camera vs one on the move.
        terms.append("s.motion < 24" if f.motion == "steady" else "s.motion >= 38")
    return "\nWHERE " + "\n  AND ".join(terms) if terms else ""


def take(kind: str, limit: int) -> None:
    now = time.monotonic()
    calls = _calls[kind]
    while calls and now - calls[0] > 60:
        calls.popleft()
    if len(calls) >= limit:
        raise HTTPException(429, "Too many searches right now; try a preset or wait a minute.")
    calls.append(now)


async def run_sql(query: str) -> tuple[list[dict[str, Any]], float]:
    t0 = time.perf_counter()
    r = await _client.post(
        f"{API}/chalk.protosql.v1.SqlService/ExecuteSqlQuery",
        headers={
            "Authorization": f"Bearer {await token()}",
            "X-Chalk-Env-Id": ENV_ID,
            "x-chalk-deployment-type": "engine-grpc",
        },
        json={"query": query, "syncOptions": {}},
    )
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    body = r.json()
    if "parquet" not in body:
        raise HTTPException(502, f"ChalkSQL error: {json.dumps(body.get('errors', body))[:300]}")
    rows = pq.read_table(io.BytesIO(base64.b64decode(body["parquet"]))).to_pylist()
    for row in rows:
        row["score"] = round(float(row["score"]), 4)
        row["motion"] = round(float(row["motion"]), 1)
        row["clip_url"] = f"clips/{row['segment_id']}.mp4"
        row["thumb_url"] = f"thumbs/{row['segment_id']}.jpg"
    return rows, ms


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/data")
async def data() -> dict[str, str]:
    # The versioned URL to read the embeddings from, for ChalkSQL or DataFrame clients.
    return {"segments_parquet": SEGMENTS_PARQUET}


@app.get("/api/presets")
async def presets() -> JSONResponse:
    return JSONResponse(json.loads((WEB / "presets.json").read_text()))


@app.post("/api/search")
async def search(req: SearchRequest) -> JSONResponse:
    preset_list = json.loads((WEB / "presets.json").read_text())["presets"]
    if req.preset is not None:
        preset = next((p for p in preset_list if p["id"] == req.preset), None)
        if preset is None:
            raise HTTPException(404, "unknown preset")
        text, filters = preset["query"], Filters(**preset.get("filters", {}))
    else:
        text, filters = " ".join((req.query or "").split())[:MAX_QUERY_CHARS], req.filters
        if not text:
            raise HTTPException(400, "empty search")
    key = json.dumps([text.lower(), filters.model_dump()], sort_keys=True)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < PRESET_CACHE_SECONDS:
        _cache.move_to_end(key)
        return JSONResponse({**hit[1], "cached": True})
    take("all", SEARCHES_PER_MINUTE)
    if req.preset is None:
        take("free", FREE_TEXT_PER_MINUTE)
    sql = SQL_TEMPLATE.format(text=sql_literal(text), parquet=SEGMENTS_PARQUET, where=where_clause(filters), k=TOP_K)
    async with _inflight:
        rows, ms = await run_sql(sql)
    result = {"query": text, "filters": filters.model_dump(), "rows": rows, "sql": sql, "sql_ms": round(ms)}
    _cache[key] = (time.time(), result)
    if len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return JSONResponse({**result, "cached": False})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), proxy_headers=True)

"""Run one ChalkSQL query over the Connect API and return the result as an Arrow table."""

from __future__ import annotations

import base64
import io
import json
import os
import time

import httpx
import pyarrow as pa
import pyarrow.parquet as pq


def run_sql(query: str, timeout: float = 1800) -> tuple[pa.Table, float]:
    """Run `query` in CHALK_ENVIRONMENT_ID; return the result and the wall time in seconds."""
    api = os.environ["CHALK_API_SERVER"].rstrip("/")
    token = httpx.post(
        f"{api}/v1/oauth/token",
        json={
            "client_id": os.environ["CHALK_CLIENT_ID"],
            "client_secret": os.environ["CHALK_CLIENT_SECRET"],
            "grant_type": "client_credentials",
        },
        timeout=30,
    ).json()["access_token"]
    t0 = time.perf_counter()
    r = httpx.post(
        f"{api}/chalk.protosql.v1.SqlService/ExecuteSqlQuery",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Chalk-Env-Id": os.environ["CHALK_ENVIRONMENT_ID"],
            "x-chalk-deployment-type": "engine-grpc",
        },
        json={"query": query, "syncOptions": {}},
        timeout=timeout,
    )
    seconds = time.perf_counter() - t0
    body = r.json()
    if "parquet" not in body:
        raise RuntimeError(f"ChalkSQL failed (HTTP {r.status_code}): {json.dumps(body)[:2000]}")
    return pq.read_table(io.BytesIO(base64.b64decode(body["parquet"]))), seconds

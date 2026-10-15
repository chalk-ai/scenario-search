#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5", "httpx==0.28.1", "pyarrow==21.0.0"]
# ///
"""Embed and tag every segment with two ChalkSQL queries, and publish the result to the app.

    ./batch.py --env-id <sql environment> --out out/

1. sql/embed_segments.sql sends every segment through function.scenario-embed-segment (the GPU).
2. sql/tag_segments.sql adds a zero-shot activity tag per segment from text-prompt similarities.
Each result goes to the Volume under data/, and the app is redeployed so its mount includes it:
ChalkSQL reads the parquet back from the app over HTTPS, at a URL versioned by the file's hash,
because the engine caches read_parquet results by URL.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from chalk_env import load_chalk_env
from chalk_sql import run_sql

HERE = Path(__file__).resolve().parent


def publish(out: Path, env_id: str, app: str, name: str) -> str:
    """Upload out/data, redeploy the app, and return the file's versioned URL once the app serves it."""
    subprocess.run(
        [str(HERE / "deploy_app.py"), "--env-id", env_id, "--upload", str(out), "--only", "data"], check=True
    )
    digest = hashlib.md5((out / "data" / name).read_bytes()).hexdigest()
    url = f"{app}/data/{name}"
    # The previous app revision keeps serving for a while after a redeploy; wait for the new bytes.
    for _ in range(60):
        r = httpx.get(url, timeout=120)
        if r.status_code == 200 and hashlib.md5(r.content).hexdigest() == digest:
            return f"{url}?v={digest[:12]}"
        time.sleep(10)
    raise RuntimeError(f"{url} never served the uploaded file")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--app", default=None, help="app URL (default: the predicted scenario-search URL)")
    args = parser.parse_args()
    load_chalk_env(args.env_id)
    app = args.app or f"https://{args.env_id}-scenario-search.staging.internal.gcp.chalk.dev"
    data = args.out / "data"

    # One query over every segment. A WHERE on this query would not save GPU work: the engine evaluates
    # the catalog_call projection on every scanned row before filtering.
    meta = f"{app}/data/segments_meta.parquet"
    meta += f"?v={hashlib.md5((data / 'segments_meta.parquet').read_bytes()).hexdigest()[:12]}"
    embedded, seconds = run_sql((HERE / "sql/embed_segments.sql").read_text().replace("{meta}", meta))
    print(f"embedded {embedded.num_rows} segments in {seconds:.1f}s", flush=True)
    pq.write_table(embedded, data / "segments_embedded.parquet")
    embedded_url = publish(args.out, args.env_id, app, "segments_embedded.parquet")

    tagged, seconds = run_sql((HERE / "sql/tag_segments.sql").read_text().replace("{embedded}", embedded_url))
    print(f"tagged {tagged.num_rows} segments in {seconds:.1f}s")
    pq.write_table(tagged, data / "segments.parquet")
    publish(args.out, args.env_id, app, "segments.parquet")


if __name__ == "__main__":
    main()

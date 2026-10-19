#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkpy[chalkdf]==2.162.5"]
# ///
"""The same scenario search as the page's SQL, written with the Chalk DataFrame API.

    ./search_dataframe.py "a tray full of round watch faces" [--activity walking] \
        [--app https://<app web_url>]

The plan runs as a job in that environment: it embeds the text on the GPU through the Chalk function,
scores every stored segment embedding, filters on metadata, and returns the top 12.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

APP = "https://tmnmc9beyujew-scenario-search.staging.internal.gcp.chalk.dev"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--activity", default=None)
    parser.add_argument("--worker", default=None)
    parser.add_argument("--app", default=os.environ.get("SCENARIO_SEARCH_APP", APP))
    args = parser.parse_args()
    # The app hands out the embeddings' current versioned URL: the engine caches scans by URL.
    with urllib.request.urlopen(f"{args.app.rstrip('/')}/api/data", timeout=30) as r:
        segments_url = json.load(r)["segments_parquet"]

    import pyarrow as pa
    from chalk import functions as F
    from chalk.features import _
    from chalkdf import DataFrame

    # One row: the query and its embedding, computed once by the deployed function.
    query = DataFrame.from_dict({"query": [args.query]}).with_columns(
        {"v": F.catalog_call("function.scenario-embed-text", _.query, output_type=pa.list_(pa.float64()))}
    )
    segments = DataFrame.scan([segments_url])
    if args.activity:
        segments = segments.filter(_.activity == args.activity)
    if args.worker:
        segments = segments.filter(_.worker_id == args.worker)
    top = (
        segments.join(query, how="cross")
        .with_columns({"score": F.cosine_similarity(_.embedding, _.v)})
        .select("segment_id", "worker_id", "clip_index", "start_s", "activity", "score")
        .order_by(("score", "desc"))
        .slice(0, 12)
    )
    for row in top.run(remote=True).to_arrow().to_pylist():
        print(f"{row['score']:.4f}  {row['segment_id']}  {row['worker_id']}  {row['activity']}")


if __name__ == "__main__":
    main()

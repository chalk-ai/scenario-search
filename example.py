#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx==0.28.1"]
# ///
"""Search the deployed app over HTTP: ./example.py https://<app> "a tray full of round watch faces" [activity]"""

from __future__ import annotations

import sys

import httpx


def main() -> None:
    app, query = sys.argv[1].rstrip("/"), sys.argv[2]
    filters = {"activity": sys.argv[3]} if len(sys.argv) > 3 else {}
    r = httpx.post(f"{app}/api/search", json={"query": query, "filters": filters}, timeout=120)
    r.raise_for_status()
    body = r.json()
    print(f"ChalkSQL {body['sql_ms']} ms{' (cached)' if body['cached'] else ''}")
    for row in body["rows"]:
        print(f"{row['score']:.3f}  {row['segment_id']}  {row['activity']:<14}  {app}/{row['clip_url']}")


if __name__ == "__main__":
    main()

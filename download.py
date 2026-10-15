#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx==0.28.1"]
# ///
"""Download the footage: Voxel51/Egocentric_10K_subset (Apache-2.0), pinned to one revision.

    ./download.py --out raw/

416 three-minute 1080p clips from one factory (about 35 GB). The subset is a curated slice of
builddotai/Egocentric-10K.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

REPO = "https://huggingface.co/datasets/Voxel51/Egocentric_10K_subset/resolve"
REVISION = "8a136de2d005a7041d80d2f78ac0478ce5f2fb54"


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("raw"))
    parser.add_argument("--parallel", type=int, default=12)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    samples = httpx.get(f"{REPO}/{REVISION}/samples.json", follow_redirects=True, timeout=120).json()["samples"]
    paths = [s["filepath"] for s in samples]
    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(lambda p: fetch(f"{REPO}/{REVISION}/{p}", args.out / Path(p).name), paths))
    print(f"{len(paths)} clips in {args.out}")


if __name__ == "__main__":
    main()

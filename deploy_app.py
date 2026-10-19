#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Upload the media to a Chalk Volume and deploy the search app (app/) as a CPU ScalingGroup.

    ./deploy_app.py --upload out/ # clips/ thumbs/ frames/ data/
    ./deploy_app.py # redeploy the app only

The app runs every search as one ChalkSQL query in the same environment, authenticated with the
ScalingGroup's own workload identity, so it holds no client secret.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
VOLUME = "scenario-search-media"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", type=Path, default=None, help="local dir holding clips/ thumbs/ frames/ data/")
    parser.add_argument("--only", nargs="*", default=None, help="subdirectories of --upload to send")
    parser.add_argument("--no-deploy", action="store_true")
    args = parser.parse_args()
    from chalkcompute import Image, ReadinessProbe, ScalingGroup, Volume

    volume = Volume(VOLUME)
    if args.upload:
        for sub in args.only or ("clips", "thumbs", "frames", "data"):
            t0 = time.monotonic()
            volume.put_dir(args.upload / sub, sub)
            print(f"uploaded {sub}/ in {time.monotonic() - t0:.0f}s")
    if args.no_deploy:
        return

    image = (
        Image.debian_slim(python_version="3.12")
        .pip_install(["httpx==0.28.1", "pyarrow==21.0.0",
                "chalkcompute==2.13.5", "fastapi==0.116.1", "uvicorn[standard]==0.35.0"])
        .add_local_file(str(HERE / "app/server.py"), "/app/server.py", strategy="copy")
        .add_local_file(str(HERE / "app/index.html"), "/web/index.html", strategy="copy")
        .add_local_file(str(HERE / "app/presets.json"), "/web/presets.json", strategy="copy")
        .workdir("/app")
    )
    group = ScalingGroup(
        image=image,
        name='scenario-search',
        cpu="2",
        memory="4Gi",
        env={
            "PORT": "8080",
            "WEB_DIR": "/web",
            "MEDIA_DIR": "/media",
        },
        entrypoint=["python", "/app/server.py"],
        port=8080,
        readiness_probe=ReadinessProbe.http("/healthz", period_seconds=5),
        volumes=[volume.mount("/media")],
        chalk_identity=True,
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        group.deploy(build_timeout=1800.0, ready_timeout=1200.0)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", group.web_url)


if __name__ == "__main__":
    main()

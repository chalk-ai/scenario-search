#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Deploy the SigLIP 2 embedder (embedder/server.py) as a GPU ScalingGroup.

    EMBED_TOKEN=... ./deploy_embedder.py

The embedder serves text and segment embeddings over HTTPS, reading segment frames from the app's
/frames (the media Volume, served by deploy_app.py). The Chalk functions in deploy_functions.py call
it, so ChalkSQL and chalkdf reach the model through catalog_call.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    service_url = ScalingGroup.from_name("scenario-search").web_url

    image = (
        # The CUDA torch stack comes with the base image; pip only adds the light packages on top.
        Image.base("pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime")
        .pip_install(
            [
                "transformers==4.57.1",
                "sentencepiece==0.2.1",
                "protobuf==6.32.1",
                "pillow==11.3.0",
                "httpx==0.28.1",
                "fastapi==0.116.1",
                "uvicorn[standard]==0.35.0",
            ]
        )
        .add_local_file(str(HERE / "embedder/server.py"), "/app/server.py", strategy="copy")
        .workdir("/app")
    )
    group = ScalingGroup(
        image=image,
        name='scenario-embed',
        gpu='nvidia-l4',
        cpu="4",
        memory="16Gi",
        # The 4.5 GB of weights download at startup (about a minute) instead of riding in the image.
        env={
            "PORT": "8080",
            "HF_HOME": "/tmp/hf",
            # Frames over HTTPS from the app: in this environment, GPU nodes cannot mount Volumes.
            "FRAME_URL_TEMPLATE": service_url.rstrip("/") + "/frames/{segment_id}_{k}.jpg",
            "EMBED_TOKEN": os.environ.get("EMBED_TOKEN", ""),
        },
        entrypoint=["python", "/app/server.py"],
        port=8080,
        readiness_probe=ReadinessProbe.http("/status", period_seconds=5),
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        group.deploy(build_timeout=2400.0, ready_timeout=1800.0)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", group.web_url)


if __name__ == "__main__":
    main()

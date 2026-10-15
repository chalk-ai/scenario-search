#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Deploy the embedding model as two Chalk functions, so ChalkSQL and chalkdf can call it.

    EMBED_TOKEN=... ./deploy_functions.py --env-id <sql environment> --embedder https://<embedder web_url>

Every query engine in the environment then reaches them by catalog name:

    SELECT catalog_call('function.scenario-embed-text', 'a tray full of round watch faces')
    SELECT catalog_call('function.scenario-embed-segment', 'w003_c017_t060')

Both return a 1152-dim unit vector (list<float64>). The functions (embed_functions.py) are thin: they
forward to the SigLIP 2 ScalingGroup from deploy_embedder.py, which runs the model on a GPU.
"""

from __future__ import annotations

import argparse
import os

from chalk_env import load_chalk_env
from embed_functions import scenario_embed_segment, scenario_embed_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--embedder", required=True, help="web_url of the scenario-embed ScalingGroup")
    args = parser.parse_args()
    load_chalk_env(args.env_id)
    import chalkcompute
    from chalkcompute import Image, RemoteFunction

    image = Image.debian_slim(python_version="3.12").pip_install(["httpx==0.28.1"])
    env = {"EMBEDDER_URL": args.embedder, "EMBED_TOKEN": os.environ.get("EMBED_TOKEN", "")}
    for fn, name in (
        (scenario_embed_text, "scenario-embed-text"),
        (scenario_embed_segment, "scenario-embed-segment"),
    ):
        try:
            chalkcompute.function(
                image=image,
                name=name,
                # Batch rows wait on the GPU in parallel; a 1 CPU / 2Gi replica fell over under the full table.
                cpu="2",
                memory="8Gi",
                env=env,
                min_replicas=1,
                max_replicas=3,
                # One SQL batch can be thousands of rows; the batched shim forwards them on a thread pool.
                max_batching_size=16,
            )(fn)
        except Exception as exc:
            # The decorator polls readiness for a fixed 300 s; a cold image pull can outlast it while the
            # rollout continues server-side.
            if "not ready" not in str(exc):
                raise
            print(f"{name}: {exc}; waiting longer")
            RemoteFunction.from_name(name).wait_ready(timeout=1200)
        print(f"deployed function.{name}")


if __name__ == "__main__":
    main()

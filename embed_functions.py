"""The Chalk functions behind catalog_call: thin forwards to the SigLIP 2 embedder ScalingGroup.

This module is copied into each function's container on its own, so it imports nothing local.
"""

from __future__ import annotations

import os
import threading
import time

# The engine sends a batch's rows to this function in parallel, many batches at once. Each replica
# lets only this many calls reach the GPU group at a time, so its queue stays short: queued calls
# that outlive the client timeout would be retried while the abandoned originals still hold their
# place, and the batch would grind to a halt.
_IN_FLIGHT = threading.BoundedSemaphore(int(os.environ.get("EMBED_IN_FLIGHT", "4")))


def _post(path: str, body: dict[str, list[str]]) -> list[float]:
    import httpx  # installed in the function image, not needed by the deploy script

    # A SQL batch fails as a unit, so a busy embedder (429, its per-minute budget) or a brief 5xx is
    # retried with backoff instead of failing every row in the batch.
    for attempt in range(6):
        with _IN_FLIGHT:
            r = httpx.post(
                os.environ["EMBEDDER_URL"].rstrip("/") + path,
                json=body,
                headers={"x-embed-token": os.environ.get("EMBED_TOKEN", "")},
                timeout=60,
            )
        if r.status_code != 429 and r.status_code < 500:
            break
        time.sleep(min(20.0, 1.0 * 2**attempt))
    r.raise_for_status()
    return [float(x) for x in r.json()["vectors"][0]]


def scenario_embed_text(text: str) -> list[float]:
    """SigLIP 2 text embedding of `text`: a 1152-dim unit vector in the same space as the segments."""
    return _post("/embed/text", {"texts": [text or ""]})


def scenario_embed_segment(segment_id: str) -> list[float]:
    """Mean SigLIP 2 image embedding of a segment's four frames, normalized: a 1152-dim unit vector."""
    return _post("/embed/segments", {"segment_ids": [segment_id]})

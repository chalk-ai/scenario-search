"""SigLIP 2 image-text embeddings on one GPU, for scenario search over video segments.

POST /embed/text      {"texts": ["carrying a tray of parts"]}       -> {"vectors": [[...1152]], ...}
POST /embed/segments  {"segment_ids": ["w003_c017_t060"]}          -> {"vectors": [[...1152]], ...}
GET  /status

A segment's vector is the normalized mean of its four frames' image embeddings. Frames come from
FRAME_URL_TEMPLATE (e.g. "https://<app>/frames/{segment_id}_{k}.jpg") or, when that is unset, from
FRAME_DIR (a mounted copy of the media Volume), so callers name segments, never URLs or paths.
Requests need the x-embed-token header when EMBED_TOKEN is set.
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoProcessor

MODEL_ID = os.environ.get("MODEL_ID", "google/siglip2-so400m-patch14-384")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "e8e487298228002f3d8a82e0cd5c8ea9c567f57f")
FRAME_URL_TEMPLATE = os.environ.get("FRAME_URL_TEMPLATE", "")
FRAME_DIR = Path(os.environ.get("FRAME_DIR", "/media/frames"))
# Segment requests beyond this wait in line before fetching frames: a batch job sends many at once,
# and unbounded fetches exhaust the connection pool long before the GPU is the bottleneck.
SEGMENT_CONCURRENCY = int(os.environ.get("SEGMENT_CONCURRENCY", "4"))
EMBED_TOKEN = os.environ.get("EMBED_TOKEN", "")
FRAMES_PER_SEGMENT = 4
SEGMENT_ID = re.compile(r"^w\d{3}_c\d{3}_t\d{3}$")
# Replica-wide budgets: the gateway hides client addresses, so limits apply to all callers together.
# Text calls follow visitors' searches; segment calls come from the batch job.
TEXT_PER_MINUTE = int(os.environ.get("TEXT_PER_MINUTE", "600"))
SEGMENTS_PER_MINUTE = int(os.environ.get("SEGMENTS_PER_MINUTE", "6000"))

state: dict[str, Any] = {}


class Embedder:
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=dtype)
        self.model = self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.dtype = dtype
        # One GPU runs one forward pass at a time.
        self.lock = threading.Lock()
        for _ in range(3):
            self.text(["warm up"])
            self.images([Image.new("RGB", (384, 384))])

    @torch.no_grad()
    def text(self, texts: list[str]) -> list[list[float]]:
        inputs = self.processor(text=texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        with self.lock:
            v = self.model.get_text_features(**inputs.to(self.device)).float()
        return torch.nn.functional.normalize(v, dim=-1).cpu().tolist()

    @torch.no_grad()
    def images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        with self.lock:
            v = self.model.get_image_features(pixel_values=inputs["pixel_values"].to(self.device, self.dtype)).float()
        return torch.nn.functional.normalize(v, dim=-1).cpu()


class Budget:
    def __init__(self, per_minute: int) -> None:
        super().__init__()
        self.per_minute = per_minute
        self.calls: deque[float] = deque()

    def take(self) -> None:
        now = time.monotonic()
        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()
        if len(self.calls) >= self.per_minute:
            raise HTTPException(429, "embedding budget exhausted; try again in a minute")
        self.calls.append(now)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # uvicorn listens only after this returns, so the replica turns ready with the model warm.
    state["embedder"] = await asyncio.to_thread(Embedder)
    state["http"] = httpx.AsyncClient(timeout=30)
    state["segment_slots"] = asyncio.Semaphore(SEGMENT_CONCURRENCY)
    state["budgets"] = {"text": Budget(TEXT_PER_MINUTE), "segments": Budget(SEGMENTS_PER_MINUTE)}
    yield
    await state["http"].aclose()


app = FastAPI(lifespan=lifespan)


def check(token: str | None, budget: str) -> None:
    if EMBED_TOKEN and token != EMBED_TOKEN:
        raise HTTPException(401, "missing or wrong x-embed-token")
    state["budgets"][budget].take()


class TextRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


class SegmentRequest(BaseModel):
    segment_ids: list[str] = Field(min_length=1, max_length=32)


@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse({"ok": True, "model": MODEL_ID, "revision": MODEL_REVISION, "device": state["embedder"].device})


@app.post("/embed/text")
async def embed_text(req: TextRequest, x_embed_token: str | None = Header(default=None)) -> JSONResponse:
    check(x_embed_token, "text")
    texts = [t[:200] for t in req.texts]
    t0 = time.perf_counter()
    vectors = await asyncio.to_thread(state["embedder"].text, texts)
    return JSONResponse({"vectors": vectors, "model_ms": round((time.perf_counter() - t0) * 1000, 1)})


async def load_frame(segment_id: str, k: int) -> Image.Image:
    if FRAME_URL_TEMPLATE:
        r = await state["http"].get(FRAME_URL_TEMPLATE.format(segment_id=segment_id, k=k))
        if r.status_code != 200:
            raise HTTPException(404, f"frame {segment_id}_{k} not found")
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    path = FRAME_DIR / f"{segment_id}_{k}.jpg"
    if not path.is_file():
        raise HTTPException(404, f"frame {segment_id}_{k} not found")
    return await asyncio.to_thread(lambda: Image.open(path).convert("RGB"))


@app.post("/embed/segments")
async def embed_segments(req: SegmentRequest, x_embed_token: str | None = Header(default=None)) -> JSONResponse:
    check(x_embed_token, "segments")
    for s in req.segment_ids:
        if not SEGMENT_ID.match(s):
            raise HTTPException(422, f"bad segment id {s!r}")
    async with state["segment_slots"]:
        images = await asyncio.gather(*(load_frame(s, k) for s in req.segment_ids for k in range(FRAMES_PER_SEGMENT)))
        t0 = time.perf_counter()
        frames = await asyncio.to_thread(state["embedder"].images, list(images))
    means = frames.reshape(len(req.segment_ids), FRAMES_PER_SEGMENT, -1).mean(dim=1)
    vectors = torch.nn.functional.normalize(means, dim=-1).tolist()
    return JSONResponse({"vectors": vectors, "model_ms": round((time.perf_counter() - t0) * 1000, 1)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))

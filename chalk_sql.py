"""Run ChalkSQL with ambient SDK authentication and return an Arrow table."""

from __future__ import annotations

import time

import pyarrow as pa
from chalkcompute import SQLClient


def run_sql(query: str, timeout: float = 1800) -> tuple[pa.Table, float]:
    """Run `query` in the selected environment; return the table and elapsed seconds."""
    started = time.perf_counter()
    with SQLClient() as client:
        table = client.query(query, timeout=timeout).to_arrow()
    return table, time.perf_counter() - started

"""Latency-budgeted model selection.

A model is only useful inline if it decides within the real-time budget. Each
candidate gets ``cost = alpha * (1 - macro_F1) + beta * inference_ms`` and the
cheapest candidate within the budget wins.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from qos_ml.features import rows_to_features


def measure_inference_ms(model: Any, rows: list[dict[str, Any]], repeats: int = 100) -> float:
    """Mean time to classify ONE flow, as the live controller does it (ms)."""
    samples = rows_to_features(rows[:repeats])
    if not samples:
        raise ValueError("need at least one row to measure inference time")
    model.predict_proba(pd.DataFrame([samples[0]]))  # warm-up
    start = time.perf_counter()
    for sample in samples:
        model.predict_proba(pd.DataFrame([sample]))
    return (time.perf_counter() - start) * 1000 / len(samples)


def select_model(
    candidates: list[dict[str, Any]],
    *,
    budget_ms: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.02,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Return (chosen name, candidates annotated with cost and within_budget).

    ``candidates`` need ``name``, ``macro_f1`` and ``inference_ms``. If nothing
    fits the budget, the fastest model is chosen so the loop never stalls.
    """
    table = []
    for candidate in candidates:
        cost = alpha * (1 - candidate["macro_f1"]) + beta * candidate["inference_ms"]
        within = budget_ms is None or candidate["inference_ms"] <= budget_ms
        table.append({**candidate, "cost": cost, "within_budget": within})
    if not table:
        return None, table
    allowed = [row for row in table if row["within_budget"]] or [min(table, key=lambda r: r["inference_ms"])]
    return min(allowed, key=lambda row: row["cost"])["name"], table

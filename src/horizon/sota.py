"""SOTA frontier flag computation.

Lives in its own module (no pandas/sklearn imports) so it can be tested
without pulling in the full analysis pipeline's dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_sota_flags(
    results: dict[str, Any], metric_key: str
) -> dict[str, bool]:
    """Mark a model SOTA if its <metric_key> estimate matches or beats the
    running maximum across all earlier release dates. Models that share a
    release date all see the same updated running max, so within a date only
    the top-scoring model(s) are flagged SOTA.

    Args:
        results: mapping of model id -> dict with `release_date` and
            `metrics[metric_key]["estimate"]`.
        metric_key: e.g. "p50_horizon_length" or "p80_horizon_length".

    Returns:
        Dict mapping each model id to a SOTA flag at the requested threshold.
    """
    by_date: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for model, model_data in results.items():
        horizon = model_data["metrics"][metric_key]["estimate"]
        by_date[model_data["release_date"]].append((model, horizon))

    flags: dict[str, bool] = {}
    highest_so_far = float("-inf")
    for release_date in sorted(by_date.keys()):
        items = by_date[release_date]
        highest_so_far = max(highest_so_far, max(h for _, h in items))
        for model, horizon in items:
            flags[model] = horizon >= highest_so_far
    return flags

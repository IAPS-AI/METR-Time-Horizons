"""Tests for the per-threshold SOTA flag computation.

These guard the bug class that motivated `is_sota_p50` / `is_sota_p80`:
a single per-model SOTA flag computed against p50 was being reused on
the p80 dashboard view, mislabeling the frontier model.
"""

from __future__ import annotations

import yaml

from horizon.sota import compute_sota_flags as _compute_sota_flags


def _model(release_date: str, p50: float, p80: float) -> dict:
    return {
        "release_date": release_date,
        "metrics": {
            "p50_horizon_length": {"estimate": p50},
            "p80_horizon_length": {"estimate": p80},
        },
    }


def test_p50_and_p80_can_disagree() -> None:
    """A model can lead p50 without leading p80, and vice versa."""
    results = {
        "A": _model("2025-01-01", p50=10.0, p80=1.0),
        "B": _model("2025-02-01", p50=5.0, p80=2.0),
        "C": _model("2025-03-01", p50=12.0, p80=1.5),
    }
    assert _compute_sota_flags(results, "p50_horizon_length") == {
        "A": True,
        "B": False,
        "C": True,
    }
    assert _compute_sota_flags(results, "p80_horizon_length") == {
        "A": True,
        "B": True,
        "C": False,
    }


def test_same_date_ties_both_flagged() -> None:
    """Models tying the running max on the same release date both count as SOTA."""
    results = {
        "A": _model("2025-01-01", p50=10.0, p80=1.0),
        "B": _model("2025-02-01", p50=11.0, p80=2.0),
        "B2": _model("2025-02-01", p50=11.0, p80=2.0),
        "C": _model("2025-03-01", p50=8.0, p80=1.5),
    }
    flags = _compute_sota_flags(results, "p50_horizon_length")
    assert flags["A"] is True
    assert flags["B"] is True
    assert flags["B2"] is True
    assert flags["C"] is False


def test_first_model_is_always_sota() -> None:
    results = {"only": _model("2025-01-01", p50=1.0, p80=0.5)}
    assert _compute_sota_flags(results, "p50_horizon_length") == {"only": True}
    assert _compute_sota_flags(results, "p80_horizon_length") == {"only": True}


def test_strict_improvement_required_on_later_date() -> None:
    """A later model whose horizon ties the running max IS still SOTA (>= semantics)."""
    results = {
        "A": _model("2025-01-01", p50=5.0, p80=1.0),
        "B": _model("2025-02-01", p50=5.0, p80=1.0),
    }
    flags = _compute_sota_flags(results, "p50_horizon_length")
    assert flags == {"A": True, "B": True}


def test_sota_frontier_is_monotone_non_decreasing() -> None:
    """By construction, walking SOTA models in date order yields a non-decreasing horizon."""
    results = {
        "A": _model("2025-01-01", p50=10.0, p80=1.0),
        "B": _model("2025-02-01", p50=5.0, p80=2.0),
        "C": _model("2025-03-01", p50=12.0, p80=1.5),
        "D": _model("2025-04-01", p50=20.0, p80=3.0),
    }
    for metric in ("p50_horizon_length", "p80_horizon_length"):
        flags = _compute_sota_flags(results, metric)
        sota_in_order = sorted(
            (m["release_date"], m["metrics"][metric]["estimate"])
            for k, m in results.items()
            if flags[k]
        )
        horizons = [h for _, h in sota_in_order]
        assert horizons == sorted(horizons), (
            f"{metric} SOTA frontier not monotone non-decreasing: {horizons}"
        )


def test_checked_in_yaml_files_have_both_flags(tmp_path: object) -> None:
    """The shipped YAMLs must carry both per-threshold flags so the dashboard
    and the Python overlay loader can pick the right one."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    for name in ("benchmark_results_1_0.yaml", "benchmark_results_1_1.yaml"):
        data = yaml.safe_load((repo / name).read_text())
        results = data["results"]
        assert results, f"{name} has no results"
        for model, model_data in results.items():
            metrics = model_data["metrics"]
            assert "is_sota_p50" in metrics, f"{name}:{model} missing is_sota_p50"
            assert "is_sota_p80" in metrics, f"{name}:{model} missing is_sota_p80"


def test_checked_in_yaml_p50_flag_matches_recomputed() -> None:
    """`is_sota_p50` (and the legacy `is_sota` alias) must equal a fresh
    computation from the horizons in the YAML — proves the migration was
    consistent and that nothing has drifted since."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    for name in ("benchmark_results_1_0.yaml", "benchmark_results_1_1.yaml"):
        data = yaml.safe_load((repo / name).read_text())
        results = data["results"]
        recomputed_p50 = _compute_sota_flags(results, "p50_horizon_length")
        recomputed_p80 = _compute_sota_flags(results, "p80_horizon_length")
        for model, flag in recomputed_p50.items():
            stored = results[model]["metrics"]["is_sota_p50"]
            assert stored == flag, f"{name}:{model} is_sota_p50 stored={stored} recomputed={flag}"
            # Legacy alias should match p50 by definition
            assert results[model]["metrics"]["is_sota"] == flag, (
                f"{name}:{model} legacy is_sota out of sync with is_sota_p50"
            )
        for model, flag in recomputed_p80.items():
            stored = results[model]["metrics"]["is_sota_p80"]
            assert stored == flag, f"{name}:{model} is_sota_p80 stored={stored} recomputed={flag}"


def test_checked_in_yaml_p80_frontier_ends_correctly_v1_1() -> None:
    """Regression test for the original bug report: on the v1.1 p80 frontier,
    the most-recent SOTA model should be the one with the highest p80 horizon
    in date order — and that should be `gemini_3_1_pro` as of the data this
    test was written against. If a newer model ships and overtakes it on p80,
    update this test to the new frontier model."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    data = yaml.safe_load((repo / "benchmark_results_1_1.yaml").read_text())
    results = data["results"]

    sota_p80 = sorted(
        (
            (model_data["release_date"], model)
            for model, model_data in results.items()
            if model_data["metrics"]["is_sota_p80"]
        ),
    )
    assert sota_p80, "No p80 SOTA models found in v1.1 YAML"
    latest_date, latest_model = sota_p80[-1]
    # Sanity: latest p80 SOTA model has the max p80 horizon overall (monotone frontier).
    max_p80 = max(
        m["metrics"]["p80_horizon_length"]["estimate"] for m in results.values()
    )
    assert (
        results[latest_model]["metrics"]["p80_horizon_length"]["estimate"] == max_p80
    ), f"latest p80 SOTA ({latest_model}) does not have the global max p80"

from __future__ import annotations

from pathlib import Path

import numpy as np

from synthetic_bit_sequence_majority_rule.io.configs import load_default_config
from synthetic_bit_sequence_majority_rule.services.runner import run_pipeline


def test_default_pipeline_smoke() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_default_config(project_root / "configs" / "default.yaml")
    result = run_pipeline(cfg, run_id="test_run")

    assert "selected" in result.branches
    assert result.metadata["formula_kmax"] == 5
    assert result.metadata["reduced_k_values"] == [3, 5]
    assert set(result.selected_branch.majority_results) == {"euclidean", "chebyshev", "canberra", "manhattan"}
    assert set(result.selected_branch.final_comparison.to_frame()["Metric"]) == {"euclidean", "chebyshev", "canberra", "manhattan"}
    assert not result.selected_branch.final_comparison.to_frame().empty


def test_ionosfera_pipeline_supports_long_binary_sequences() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_default_config(project_root / "configs" / "default.yaml")
    cfg.dataset.path = (
        project_root
        / "datasets"
        / "quantitative"
        / "ionosfera"
        / "Ionosfera (350, 33, 2).csv"
    )
    cfg.dataset.alternate_paths = []
    cfg.dataset.format = "csv"
    cfg.dataset.has_header = False
    cfg.metrics.enabled = ["euclidean"]
    cfg.run.save_outputs = False

    result = run_pipeline(cfg, run_id="ionosfera_test_run")
    branch = result.selected_branch
    majority = branch.majority_results["euclidean"]

    assert result.metadata["formula_kmax"] == 247
    assert result.metadata["reduced_k_values"] == list(range(3, 248, 2))
    assert len(majority.reduced_k_values) == 123
    assert all(len(sequence) == 123 for sequence in majority.binary_sequences)
    assert majority.decimal_values.dtype == object
    assert any(value > np.iinfo(np.int64).max for value in majority.decimal_values)
    assert branch.statistics_results["euclidean"].rows
    assert len(branch.stability_results["euclidean"].rows) == 123

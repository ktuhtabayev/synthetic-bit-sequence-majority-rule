from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from synthetic_bit_sequence_majority_rule.io.configs import load_default_config
from synthetic_bit_sequence_majority_rule.io.writers import prune_full_run_outputs
from synthetic_bit_sequence_majority_rule.services.analysis import (
    build_comparison_from_run,
    write_analysis_outputs,
)
from synthetic_bit_sequence_majority_rule.services.runner import (
    make_run_id,
    run_none_minmax_comparison,
    run_pipeline,
)


def test_prune_full_run_outputs_keeps_latest_three_and_preserves_quick_runs(tmp_path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    for index in range(5):
        path = runs_root / f"default_run_20260608_12000{index}"
        path.mkdir()
        os.utime(path, (1000 + index, 1000 + index))

    quick_path = runs_root / "quick_distance_test"
    quick_path.mkdir()
    os.utime(quick_path, (1, 1))

    removed = prune_full_run_outputs(runs_root, keep_latest=3)

    remaining = sorted(path.name for path in runs_root.iterdir())
    assert remaining == [
        "default_run_20260608_120002",
        "default_run_20260608_120003",
        "default_run_20260608_120004",
        "quick_distance_test",
    ]
    assert sorted(path.name for path in removed) == [
        "default_run_20260608_120000",
        "default_run_20260608_120001",
    ]


def test_run_ids_started_in_the_same_second_are_distinct() -> None:
    run_ids = {make_run_id("default_run") for _ in range(50)}

    assert len(run_ids) == 50
    assert all(run_id.startswith("default_run_") for run_id in run_ids)


def test_gui_exports_are_never_pruned_and_command_line_runs_keep_three(tmp_path) -> None:
    cfg = load_default_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    cfg.run.output_root = tmp_path
    result = run_pipeline(cfg)
    comparison = build_comparison_from_run(run_none_minmax_comparison(cfg, result))

    for _ in range(4):
        write_analysis_outputs(replace(result, run_id=make_run_id("default_run")), comparison, gui_export=True)
    for _ in range(4):
        write_analysis_outputs(replace(result, run_id=make_run_id("default_run")), comparison)

    assert len(list((tmp_path / "gui").iterdir())) == 4
    assert len([path for path in tmp_path.iterdir() if path.name.startswith("default_run_")]) == 3

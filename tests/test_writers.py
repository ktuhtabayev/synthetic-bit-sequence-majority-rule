from __future__ import annotations

import os

from synthetic_bit_sequence_majority_rule.io.writers import prune_full_run_outputs


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

from __future__ import annotations

from pathlib import Path

import pytest

from synthetic_bit_sequence_majority_rule.cli import build_parser, config_from_args, main
from synthetic_bit_sequence_majority_rule.io.configs import load_dataset_catalog, load_default_config
from synthetic_bit_sequence_majority_rule.paths import PROJECT_ROOT


def _config(*argv: str):
    return config_from_args(build_parser().parse_args(argv))


def test_no_options_runs_the_config_file_as_is() -> None:
    assert _config() == load_default_config()


def test_a_preset_and_overrides_apply_to_one_run() -> None:
    entry = load_dataset_catalog()["ionosfera_csv"]

    cfg = _config(
        "--preset", "ionosfera_csv",
        "--normalization", "zscore",
        "--metrics", "euclidean, Canberra",
        "--no-save",
    )

    assert cfg.dataset.path == PROJECT_ROOT / entry["path"]
    assert cfg.dataset.has_header is False
    assert cfg.dataset.alternate_paths == [PROJECT_ROOT / path for path in entry["alternate_paths"]]
    assert cfg.preprocessing.normalization.mode == "zscore"
    assert cfg.metrics.enabled == ["euclidean", "canberra"]
    assert cfg.run.save_outputs is False


def test_a_dataset_path_is_relative_to_the_current_folder_and_matches_its_preset(monkeypatch) -> None:
    monkeypatch.chdir(PROJECT_ROOT / "datasets")

    cfg = _config("--dataset", "default.dat")

    assert cfg.dataset.path == PROJECT_ROOT / "datasets" / "default.dat"
    assert cfg.dataset.format == "dat"
    # The default_dat preset supplies the CSV as its alternate.
    assert cfg.dataset.alternate_paths == [PROJECT_ROOT / "datasets" / "default.csv"]


def test_invalid_options_are_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--metrics", "euclidean,cosine"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--preset", "default_csv", "--dataset", "x.csv"])
    with pytest.raises(SystemExit, match="Unknown preset 'nope'"):
        _config("--preset", "nope")


def test_list_presets(capsys) -> None:
    assert main(["--list-presets"]) == 0

    out = capsys.readouterr().out
    assert "default_csv" in out
    assert "ionosfera_dat" in out


def test_a_run_works_from_any_folder_and_restores_it(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["--no-save", "--metrics", "euclidean"]) == 0

    assert Path.cwd() == tmp_path
    assert not any(tmp_path.iterdir())
    out = capsys.readouterr().out
    assert "Final comparison for selected branch" in out
    assert "Finished successfully." in out


def test_a_failed_run_reports_the_error_and_exits_nonzero(tmp_path: Path, capsys) -> None:
    assert main(["--dataset", str(tmp_path / "missing.csv"), "--no-save"]) == 1

    err = capsys.readouterr().err
    assert "Run failed" in err
    assert "load_dataset" in err

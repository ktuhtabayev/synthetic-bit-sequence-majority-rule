from __future__ import annotations

from synthetic_bit_sequence_majority_rule.domain.params import AppConfig
from synthetic_bit_sequence_majority_rule.io.configs import (
    apply_dataset_selection,
    config_to_dict,
    find_dataset_preset,
    load_dataset_catalog,
    load_default_config,
    load_experiments_config,
    load_named_experiment,
)
from synthetic_bit_sequence_majority_rule.paths import PROJECT_ROOT

SECTIONS = (
    "project",
    "run",
    "dataset",
    "preprocessing",
    "metrics",
    "neighbors",
    "k_values",
    "majority_rule",
    "binary_sequence",
    "decimal_encoding",
    "statistics",
    "exports",
    "notes",
)


def test_omitted_config_keys_fall_back_to_the_dataclass_defaults() -> None:
    # Every section is present but empty, so every field is read through its default.
    config = AppConfig.from_dict({section: {} for section in SECTIONS})

    assert config == AppConfig()
    assert config.run.run_name == "default_run"
    assert str(config.run.output_root).replace("\\", "/") == "outputs/runs"


def test_config_to_dict_reads_back_to_an_equal_config() -> None:
    config = load_default_config()

    as_dict = config_to_dict(config)

    assert AppConfig.from_dict(as_dict) == config
    assert list(as_dict) == list(SECTIONS)
    assert as_dict["dataset"]["path"] == str(config.dataset.path)
    assert as_dict["statistics"]["tie_rule"] == list(config.statistics.tie_rule.rules)


def test_default_config_paths_do_not_depend_on_the_current_folder(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert load_default_config() == load_default_config(PROJECT_ROOT / "configs" / "default.yaml")
    assert "default_csv" in load_dataset_catalog()
    experiments = load_experiments_config()
    assert len(experiments) == 6
    # Experiments inherit the base config's sections they do not override.
    assert all(cfg.dataset.object_name_prefix == "S" for cfg in experiments)
    named = load_named_experiment("zscore_all_metrics_default_dat")
    assert named.preprocessing.normalization.mode == "zscore"
    assert named.dataset.has_header is False


def test_find_dataset_preset_matches_relative_and_absolute_paths() -> None:
    catalog = load_dataset_catalog()

    assert find_dataset_preset(catalog, "datasets/default.dat", PROJECT_ROOT) == "default_dat"
    assert find_dataset_preset(catalog, PROJECT_ROOT / "datasets" / "default.csv", PROJECT_ROOT) == "default_csv"
    assert find_dataset_preset(catalog, "datasets/unknown.csv", PROJECT_ROOT) is None


def test_apply_dataset_selection_takes_options_and_alternates_only_from_the_preset() -> None:
    catalog = load_dataset_catalog()
    config = load_default_config()

    apply_dataset_selection(
        config,
        catalog["ionosfera_csv"]["path"],
        project_root=PROJECT_ROOT,
        preset=catalog["ionosfera_csv"],
    )
    assert config.dataset.path == PROJECT_ROOT / catalog["ionosfera_csv"]["path"]
    assert config.dataset.format == "csv"
    assert config.dataset.has_header is False
    assert config.dataset.alternate_paths == [
        PROJECT_ROOT / path for path in catalog["ionosfera_csv"]["alternate_paths"]
    ]

    apply_dataset_selection(config, "elsewhere/data.dat", project_root=PROJECT_ROOT)
    assert config.dataset.format == "dat"
    assert config.dataset.has_header is False
    assert config.dataset.alternate_paths == []

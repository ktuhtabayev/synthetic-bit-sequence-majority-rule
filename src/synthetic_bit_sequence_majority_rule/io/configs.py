from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from synthetic_bit_sequence_majority_rule.domain.errors import (
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
)
from synthetic_bit_sequence_majority_rule.domain.params import AppConfig, StatisticsTieRuleConfig
from synthetic_bit_sequence_majority_rule.paths import DEFAULT_CONFIG_PATH, EXPERIMENTS_CONFIG_PATH


# ============================================================
# Small helpers
# ============================================================

def _ensure_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(
            message=f"Expected '{name}' to be a mapping, got {type(value).__name__}.",
            section=name,
        )
    return value


def _read_yaml_file(path: str | Path) -> Mapping[str, Any]:
    yaml_path = Path(path)

    if not yaml_path.exists():
        raise ConfigFileNotFoundError(yaml_path)

    try:
        text = yaml_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise ConfigParseError(yaml_path, f"Unable to read file: {exc}") from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigParseError(yaml_path, f"YAML parsing error: {exc}") from exc

    if data is None:
        data = {}

    return _ensure_mapping(data, str(yaml_path))


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)

    for key, override_value in override.items():
        base_value = result.get(key)

        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            result[key] = _deep_merge_dicts(base_value, override_value)
        else:
            result[key] = override_value

    return result


def _normalize_dataset_section_for_dat_rules(config_dict: Mapping[str, Any]) -> dict[str, Any]:
    """DAT files are headerless matrix files, so a DAT dataset defaults to has_header: false."""
    config_dict = dict(config_dict)
    dataset = config_dict.get("dataset")
    if not isinstance(dataset, Mapping):
        return config_dict

    dataset_copy = dict(dataset)
    if str(dataset_copy.get("format", "")).lower() == "dat":
        dataset_copy.setdefault("has_header", False)
        dataset_copy.setdefault("delimiter", ",")

    config_dict["dataset"] = dataset_copy
    return config_dict


def _read_experiments(path: str | Path) -> list[Any]:
    raw = _read_yaml_file(path)

    if "experiments" not in raw:
        raise ConfigValidationError(
            message="Missing required top-level key 'experiments'.",
            section="experiments",
        )

    experiments = raw["experiments"]
    if not isinstance(experiments, list):
        raise ConfigValidationError(
            message="'experiments' must be a list.",
            section="experiments",
        )
    return experiments


def _experiment_base(base_config_path: str | Path) -> dict[str, Any]:
    """The config every experiment overrides; empty when the base file is absent."""
    if not Path(base_config_path).exists():
        return {}
    return _normalize_dataset_section_for_dat_rules(_read_yaml_file(base_config_path))


def _experiment_config(base_raw: Mapping[str, Any], experiment: Mapping[str, Any], idx: int) -> AppConfig:
    merged = _normalize_dataset_section_for_dat_rules(_deep_merge_dicts(base_raw, experiment))
    try:
        return AppConfig.from_dict(merged)
    except Exception as exc:
        experiment_name = experiment.get("name", f"index_{idx}")
        raise ConfigValidationError(
            message=f"Experiment '{experiment_name}' is invalid: {exc}",
            section=f"experiments[{idx}]",
        ) from exc


# ============================================================
# Public API
# ============================================================

def load_app_config(path: str | Path) -> AppConfig:
    raw = _normalize_dataset_section_for_dat_rules(_read_yaml_file(path))

    try:
        return AppConfig.from_dict(raw)
    except Exception as exc:
        raise ConfigValidationError(
            message=str(exc),
            section="root",
        ) from exc


def load_default_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    return load_app_config(path)


def load_dataset_catalog(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, dict[str, Any]]:
    """
    Return the dataset_catalog presets from the config file, keyed by preset
    name. Returns an empty dict when the section is missing.
    """
    raw = _read_yaml_file(path)
    catalog = raw.get("dataset_catalog")
    if not isinstance(catalog, Mapping):
        return {}

    presets: dict[str, dict[str, Any]] = {}
    for name, entry in catalog.items():
        if isinstance(entry, Mapping) and entry.get("path"):
            presets[str(name)] = dict(entry)
    return presets


def find_dataset_preset(
    catalog: Mapping[str, Mapping[str, Any]],
    dataset_path: str | Path,
    project_root: Path,
) -> str | None:
    """Name of the first catalog preset whose path is dataset_path, if any."""
    target = project_root / dataset_path
    for name, entry in catalog.items():
        if project_root / str(entry["path"]) == target:
            return name
    return None


def apply_dataset_selection(
    config: AppConfig,
    dataset_path: str | Path,
    *,
    project_root: Path,
    preset: Mapping[str, Any] | None = None,
) -> None:
    """
    Point config.dataset at one dataset file.

    Relative paths resolve against project_root, so a run does not depend on
    the current folder. The format comes from the file extension unless the
    preset names one. Alternate paths are other formats of the same dataset,
    so only the preset for this file may supply them: keeping the config's own
    alternates would silently load its default dataset whenever this file is
    missing.
    """
    dataset = project_root / dataset_path
    config.dataset.path = dataset
    suffix = dataset.suffix.lower().lstrip(".")
    if suffix:
        config.dataset.format = suffix
        config.dataset.has_header = suffix != "dat"

    preset = preset or {}
    config.dataset.alternate_paths = [
        project_root / str(path) for path in preset.get("alternate_paths") or []
    ]
    if "format" in preset:
        config.dataset.format = str(preset["format"]).lower()
    if "has_header" in preset:
        config.dataset.has_header = bool(preset["has_header"])
    if "delimiter" in preset:
        config.dataset.delimiter = str(preset["delimiter"])


def load_experiments_config(
    path: str | Path = EXPERIMENTS_CONFIG_PATH,
    *,
    base_config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[AppConfig]:
    experiments = _read_experiments(path)
    base_raw = _experiment_base(base_config_path)

    configs: list[AppConfig] = []
    for idx, exp in enumerate(experiments):
        if not isinstance(exp, Mapping):
            raise ConfigValidationError(
                message=f"Experiment at index {idx} must be a mapping.",
                section=f"experiments[{idx}]",
            )
        configs.append(_experiment_config(base_raw, exp, idx))

    if not configs:
        raise ConfigValidationError(
            message="No experiments were found in the experiments list.",
            section="experiments",
        )

    return configs


def load_named_experiment(
    experiment_name: str,
    experiments_path: str | Path = EXPERIMENTS_CONFIG_PATH,
    *,
    base_config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> AppConfig:
    for idx, exp in enumerate(_read_experiments(experiments_path)):
        if isinstance(exp, Mapping) and str(exp.get("name", "")).strip() == experiment_name:
            return _experiment_config(_experiment_base(base_config_path), exp, idx)

    raise ConfigValidationError(
        message=f"Experiment '{experiment_name}' was not found.",
        section="experiments",
    )


# ============================================================
# Debug / convenience helpers
# ============================================================

def _to_plain(value: Any) -> Any:
    """Config value -> YAML/JSON-ready value, in the shape AppConfig.from_dict reads."""
    if isinstance(value, StatisticsTieRuleConfig):
        # The config file spells statistics.tie_rule as a plain list.
        return list(value.rules)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Every config section as plain data; AppConfig.from_dict reads it back to an equal config."""
    return _to_plain(config)


def describe_config(config: AppConfig) -> str:
    return (
        f"Project={config.project.name} | "
        f"Dataset={config.dataset.path} ({config.dataset.format}) | "
        f"Normalization={config.preprocessing.normalization.mode} | "
        f"Metrics={', '.join(config.metrics.enabled)} | "
        f"FullK={config.k_values.full.start}..{config.k_values.full.end} | "
        f"ReducedKMode={config.k_values.reduced.mode} | "
        f"BinaryOrderMode={config.binary_sequence.reduced_order_mode}"
    )

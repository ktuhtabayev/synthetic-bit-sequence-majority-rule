from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from synthetic_bit_sequence_majority_rule.domain.errors import (
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
)
from synthetic_bit_sequence_majority_rule.domain.params import AppConfig


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


def _normalize_dataset_section_for_dat_rules(config_dict: dict[str, Any]) -> dict[str, Any]:
    dataset = config_dict.get("dataset")
    if not isinstance(dataset, Mapping):
        return config_dict

    dataset_copy = dict(dataset)
    fmt = str(dataset_copy.get("format", "")).lower()

    if fmt == "dat":
        dataset_copy.setdefault("has_header", False)
        dataset_copy.setdefault("delimiter", ",")
        dataset_copy.setdefault("first_row_contains_shape", True)
        dataset_copy.setdefault("last_row_contains_feature_signs", True)
        dataset_copy.setdefault("quantitative_feature_sign", 1)
        dataset_copy.setdefault("nominal_feature_sign", 0)

    config_dict = dict(config_dict)
    config_dict["dataset"] = dataset_copy
    return config_dict


# ============================================================
# Public API
# ============================================================

def load_app_config(path: str | Path) -> AppConfig:
    raw = dict(_read_yaml_file(path))
    raw = _normalize_dataset_section_for_dat_rules(raw)

    try:
        return AppConfig.from_dict(raw)
    except Exception as exc:
        raise ConfigValidationError(
            message=str(exc),
            section="root",
        ) from exc


def load_default_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    return load_app_config(path)


def load_experiments_config(path: str | Path = "configs/experiments.yaml") -> list[AppConfig]:
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

    default_config_path = Path("configs/default.yaml")
    base_raw: dict[str, Any] = {}
    if default_config_path.exists():
        base_raw = dict(_read_yaml_file(default_config_path))
        base_raw = _normalize_dataset_section_for_dat_rules(base_raw)

    configs: list[AppConfig] = []

    for idx, exp in enumerate(experiments):
        if not isinstance(exp, Mapping):
            raise ConfigValidationError(
                message=f"Experiment at index {idx} must be a mapping.",
                section=f"experiments[{idx}]",
            )

        merged = _deep_merge_dicts(base_raw, exp)
        merged = _normalize_dataset_section_for_dat_rules(merged)

        try:
            cfg = AppConfig.from_dict(merged)
        except Exception as exc:
            experiment_name = exp.get("name", f"index_{idx}")
            raise ConfigValidationError(
                message=f"Experiment '{experiment_name}' is invalid: {exc}",
                section=f"experiments[{idx}]",
            ) from exc

        configs.append(cfg)

    if not configs:
        raise ConfigValidationError(
            message="No experiments were found in the experiments list.",
            section="experiments",
        )

    return configs


def load_named_experiment(
    experiment_name: str,
    experiments_path: str | Path = "configs/experiments.yaml",
) -> AppConfig:
    raw = _read_yaml_file(experiments_path)

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

    for idx, exp in enumerate(experiments):
        if not isinstance(exp, Mapping):
            continue
        if str(exp.get("name", "")).strip() == experiment_name:
            default_config_path = Path("configs/default.yaml")
            base_raw: dict[str, Any] = {}
            if default_config_path.exists():
                base_raw = dict(_read_yaml_file(default_config_path))
                base_raw = _normalize_dataset_section_for_dat_rules(base_raw)

            merged = _deep_merge_dicts(base_raw, exp)
            merged = _normalize_dataset_section_for_dat_rules(merged)

            try:
                return AppConfig.from_dict(merged)
            except Exception as exc:
                raise ConfigValidationError(
                    message=f"Experiment '{experiment_name}' is invalid: {exc}",
                    section=f"experiments[{idx}]",
                ) from exc

    raise ConfigValidationError(
        message=f"Experiment '{experiment_name}' was not found.",
        section="experiments",
    )


# ============================================================
# Debug / convenience helpers
# ============================================================

def config_to_dict(config: AppConfig) -> dict[str, Any]:
    return {
        "project": {
            "name": config.project.name,
            "version": config.project.version,
        },
        "run": {
            "run_name": config.run.run_name,
            "save_outputs": config.run.save_outputs,
            "output_root": str(config.run.output_root),
        },
        "dataset": {
            "path": str(config.dataset.path),
            "alternate_paths": [str(p) for p in config.dataset.alternate_paths],
            "format": config.dataset.format,
            "supported_formats": list(config.dataset.supported_formats),
            "delimiter": config.dataset.delimiter,
            "has_header": config.dataset.has_header,
            "object_id_column": config.dataset.object_id_column,
            "object_name_prefix": config.dataset.object_name_prefix,
            "feature_columns": list(config.dataset.feature_columns),
            "class_column": config.dataset.class_column,
            "label_mapping": dict(config.dataset.label_mapping),
        },
        "preprocessing": {
            "enabled": config.preprocessing.enabled,
            "normalization": {
                "mode": config.preprocessing.normalization.mode,
                "apply_before_distance": config.preprocessing.normalization.apply_before_distance,
            },
        },
        "metrics": {
            "enabled": list(config.metrics.enabled),
        },
        "neighbors": {
            "tie_break_rule": config.neighbors.tie_break_rule,
            "exclude_self": config.neighbors.exclude_self,
        },
        "k_values": {
            "full": {
                "start": config.k_values.full.start,
                "end": config.k_values.full.end,
            },
            "reduced": {
                "mode": config.k_values.reduced.mode,
            },
        },
        "majority_rule": {
            "threshold": config.majority_rule.threshold,
            "comparison": config.majority_rule.comparison,
        },
        "binary_sequence": {
            "use_reduced_k_values": config.binary_sequence.use_reduced_k_values,
            "reduced_order_mode": config.binary_sequence.reduced_order_mode,
        },
        "decimal_encoding": {
            "enabled": config.decimal_encoding.enabled,
            "bit_order": config.decimal_encoding.bit_order,
        },
        "statistics": {
            "enabled": config.statistics.enabled,
            "dominant_class_rule": config.statistics.dominant_class_rule,
            "purity_formula": config.statistics.purity_formula,
            "tie_rule": list(config.statistics.tie_rule.rules),
        },
        "exports": {
            "distance_matrices": config.exports.distance_matrices,
            "neighbors_tables": config.exports.neighbors_tables,
            "full_a_matrices": config.exports.full_a_matrices,
            "reduced_a_matrices": config.exports.reduced_a_matrices,
            "full_b_matrices": config.exports.full_b_matrices,
            "reduced_b_matrices": config.exports.reduced_b_matrices,
            "stats_tables": config.exports.stats_tables,
            "final_comparison": config.exports.final_comparison,
            "excel": config.exports.excel,
        },
        "notes": {
            "problem": config.notes.problem,
            "limitation": config.notes.limitation,
            "switching_rule": config.notes.switching_rule,
        },
    }


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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from synthetic_bit_sequence_majority_rule.domain.errors import NormalizationError
from synthetic_bit_sequence_majority_rule.domain.params import NormalizationConfig
from synthetic_bit_sequence_majority_rule.domain.schema import (
    LoadedDataset,
    NormalizedDataset,
    ensure_loaded_dataset,
)


# ============================================================
# Registry types
# ============================================================

NormalizationFunction = Callable[[LoadedDataset], NormalizedDataset]


# ============================================================
# Internal helper dataclasses
# ============================================================

@dataclass(slots=True)
class NormalizationSummary:
    """
    Lightweight summary for logging / reporting.
    """
    mode: str
    changed: bool
    n_objects: int
    n_features: int
    parameters: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Small helpers
# ============================================================

_ensure_dataset = ensure_loaded_dataset


def _validate_numeric_matrix(X: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise NormalizationError(mode=mode, reason="Feature matrix must be 2-dimensional.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise NormalizationError(mode=mode, reason="Feature matrix must not be empty.")
    if not np.isfinite(arr).all():
        raise NormalizationError(mode=mode, reason="Feature matrix contains NaN or infinite values.")
    return arr


def _build_normalized_dataset(
    dataset: LoadedDataset,
    mode: str,
    X_normalized: np.ndarray,
    parameters: dict[str, Any],
) -> NormalizedDataset:
    return NormalizedDataset(
        dataset=dataset,
        mode=mode,
        X_normalized=X_normalized,
        parameters=parameters,
    )


# ============================================================
# Normalization methods
# ============================================================

def normalize_none(dataset: LoadedDataset) -> NormalizedDataset:
    """
    No normalization.

    This explicitly represents the branch where the original dataset
    is used as-is.
    """
    dataset = _ensure_dataset(dataset)
    X = _validate_numeric_matrix(dataset.X, mode="none")

    return _build_normalized_dataset(
        dataset=dataset,
        mode="none",
        X_normalized=X.copy(),
        parameters={
            "applied": False,
            "reason": "No normalization selected; original dataset is used as-is.",
        },
    )


def normalize_minmax(dataset: LoadedDataset) -> NormalizedDataset:
    """
    Min-Max Normalization (Rescaling)

    Formula:
        x' = (x - min(x)) / (max(x) - min(x))

    Constant columns are safely mapped to 0.0.
    """
    dataset = _ensure_dataset(dataset)
    X = _validate_numeric_matrix(dataset.X, mode="minmax")

    col_min = np.min(X, axis=0)
    col_max = np.max(X, axis=0)
    col_range = col_max - col_min

    safe_range = np.where(col_range == 0.0, 1.0, col_range)
    X_norm = (X - col_min) / safe_range

    constant_mask = col_range == 0.0
    if np.any(constant_mask):
        X_norm[:, constant_mask] = 0.0

    return _build_normalized_dataset(
        dataset=dataset,
        mode="minmax",
        X_normalized=X_norm,
        parameters={
            "applied": True,
            "column_min": col_min.tolist(),
            "column_max": col_max.tolist(),
            "column_range": col_range.tolist(),
            "constant_columns": np.where(constant_mask)[0].tolist(),
        },
    )


def normalize_zscore(dataset: LoadedDataset) -> NormalizedDataset:
    """
    Z-score Normalization (Standardization)

    Formula:
        x' = (x - mu) / sigma

    Constant columns are safely mapped to 0.0.
    """
    dataset = _ensure_dataset(dataset)
    X = _validate_numeric_matrix(dataset.X, mode="zscore")

    col_mean = np.mean(X, axis=0)
    col_std = np.std(X, axis=0, ddof=0)

    safe_std = np.where(col_std == 0.0, 1.0, col_std)
    X_norm = (X - col_mean) / safe_std

    constant_mask = col_std == 0.0
    if np.any(constant_mask):
        X_norm[:, constant_mask] = 0.0

    return _build_normalized_dataset(
        dataset=dataset,
        mode="zscore",
        X_normalized=X_norm,
        parameters={
            "applied": True,
            "column_mean": col_mean.tolist(),
            "column_std": col_std.tolist(),
            "constant_columns": np.where(constant_mask)[0].tolist(),
        },
    )


# ============================================================
# Registry
# ============================================================

_NORMALIZATION_REGISTRY: dict[str, NormalizationFunction] = {
    "none": normalize_none,
    "minmax": normalize_minmax,
    "zscore": normalize_zscore,
}


def register_normalization_method(name: str, fn: NormalizationFunction) -> None:
    """
    Register a new normalization method dynamically.

    This keeps the module future-extendable without rewriting
    the central dispatcher.
    """
    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError("Normalization method name cannot be empty.")
    if not callable(fn):
        raise TypeError("Normalization function must be callable.")
    _NORMALIZATION_REGISTRY[normalized_name] = fn


def get_available_normalization_methods() -> list[str]:
    """
    Return the sorted list of registered normalization methods.
    """
    return sorted(_NORMALIZATION_REGISTRY.keys())


# ============================================================
# Public API
# ============================================================

def apply_normalization(
    dataset: LoadedDataset,
    normalization_config: NormalizationConfig,
) -> NormalizedDataset:
    """
    Apply normalization according to the provided config.

    Current supported modes:
    - none
    - minmax
    - zscore

    Future modes can be added via register_normalization_method().
    """
    dataset = _ensure_dataset(dataset)

    mode = normalization_config.mode.strip().lower()
    if mode not in _NORMALIZATION_REGISTRY:
        raise NormalizationError(
            mode=mode,
            reason=(
                f"Unsupported normalization mode. "
                f"Available modes: {', '.join(get_available_normalization_methods())}"
            ),
        )

    try:
        result = _NORMALIZATION_REGISTRY[mode](dataset)
    except NormalizationError:
        raise
    except Exception as exc:
        raise NormalizationError(mode=mode, reason=str(exc)) from exc

    return result


def prepare_dataset_variants(
    dataset: LoadedDataset,
    normalization_config: NormalizationConfig,
) -> dict[str, LoadedDataset]:
    """
    Prepare dataset variants for downstream stages.

    Returns a dictionary that always contains:
    - 'raw'      : original dataset
    - 'selected' : dataset actually selected by default.yaml normalization mode

    If mode != 'none', it also contains:
    - 'normalized' : normalized dataset

    Branch semantics used across the project:
    - raw        = original dataset as loaded
    - selected   = dataset used by the current pipeline run
    - normalized = explicit normalized branch when normalization is active
    """
    dataset = _ensure_dataset(dataset)

    raw_dataset = dataset.copy()
    selected_normalized = apply_normalization(dataset, normalization_config)
    selected_dataset = selected_normalized.to_loaded_dataset()

    variants: dict[str, LoadedDataset] = {
        "raw": raw_dataset,
        "selected": selected_dataset,
    }

    if normalization_config.mode.strip().lower() != "none":
        variants["normalized"] = selected_dataset

    return variants


def should_use_normalized_dataset(normalization_config: NormalizationConfig) -> bool:
    """
    Helper for downstream logic.
    """
    return normalization_config.mode.strip().lower() != "none"


def build_normalization_summary(
    normalized_dataset: NormalizedDataset,
) -> NormalizationSummary:
    """
    Build a small summary object for logging / export.
    """
    return NormalizationSummary(
        mode=normalized_dataset.mode,
        changed=normalized_dataset.mode != "none",
        n_objects=normalized_dataset.n_objects,
        n_features=normalized_dataset.n_features,
        parameters=dict(normalized_dataset.parameters),
    )


def normalized_dataset_to_frame(
    normalized_dataset: NormalizedDataset,
    include_object_label: bool = True,
) -> pd.DataFrame:
    """
    Convert a normalized dataset to a DataFrame for export/debugging.
    """
    df = pd.DataFrame(
        normalized_dataset.X_normalized,
        columns=normalized_dataset.feature_names,
    )
    if include_object_label:
        df.insert(0, "Object", normalized_dataset.object_labels)
    df[normalized_dataset.class_column] = normalized_dataset.y
    return df


def compare_raw_vs_normalized(
    dataset: LoadedDataset,
    normalization_config: NormalizationConfig,
) -> dict[str, pd.DataFrame]:
    """
    Convenience helper for debugging/export.

    Returns:
    - 'raw_frame'
    - 'selected_frame'
    - optionally 'normalized_frame'
    """
    variants = prepare_dataset_variants(dataset, normalization_config)

    result: dict[str, pd.DataFrame] = {
        "raw_frame": variants["raw"].to_frame(),
        "selected_frame": variants["selected"].to_frame(),
    }

    if "normalized" in variants:
        result["normalized_frame"] = variants["normalized"].to_frame()

    return result
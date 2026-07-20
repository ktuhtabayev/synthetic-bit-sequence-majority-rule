from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from synthetic_bit_sequence_majority_rule.domain.errors import DistanceComputationError
from synthetic_bit_sequence_majority_rule.domain.schema import (
    DistanceMatrixResult,
    LoadedDataset,
    dataset_normalized_flag,
    ensure_loaded_dataset,
)


# ============================================================
# Registry types
# ============================================================

DistanceFunction = Callable[[LoadedDataset], DistanceMatrixResult]


# ============================================================
# Internal summary dataclass
# ============================================================

@dataclass(slots=True)
class DistanceSummary:
    metric_name: str
    n_objects: int
    normalized: bool
    min_distance: float
    max_distance: float
    diagonal_zero: bool
    symmetric: bool
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Small helpers
# ============================================================

_ensure_dataset = ensure_loaded_dataset
_normalized_flag_from_dataset = dataset_normalized_flag


def _ensure_numeric_matrix(dataset: LoadedDataset, metric_name: str) -> np.ndarray:
    X = np.asarray(dataset.X, dtype=float)
    if X.ndim != 2:
        raise DistanceComputationError(metric=metric_name, reason="Feature matrix must be 2-dimensional.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise DistanceComputationError(metric=metric_name, reason="Feature matrix must not be empty.")
    if not np.isfinite(X).all():
        raise DistanceComputationError(metric=metric_name, reason="Feature matrix contains NaN or infinite values.")
    return X


def _build_result(
    dataset: LoadedDataset,
    metric_name: str,
    matrix: np.ndarray,
    extra_metadata: dict[str, Any] | None = None,
) -> DistanceMatrixResult:
    metadata = {
        "source_path": dataset.source_path,
        "source_format": dataset.source_format,
        "normalization_mode": dataset.metadata.get("normalization_mode", "none"),
        "with_normalization": _normalized_flag_from_dataset(dataset),
        "n_objects": dataset.n_objects,
        "n_features": dataset.n_features,
        "class_counts": dict(dataset.class_counts),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return DistanceMatrixResult(
        metric_name=metric_name,
        matrix=matrix,
        object_labels=list(dataset.object_labels),
        normalized=_normalized_flag_from_dataset(dataset),
        metadata=metadata,
    )


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    sq_norms = np.sum(X * X, axis=1, keepdims=True)
    squared = sq_norms + sq_norms.T - 2.0 * (X @ X.T)
    squared = np.maximum(squared, 0.0)
    return np.sqrt(squared)


def _pairwise_chebyshev(X: np.ndarray) -> np.ndarray:
    diff = np.abs(X[:, None, :] - X[None, :, :])
    return np.max(diff, axis=2)


def _pairwise_manhattan(X: np.ndarray) -> np.ndarray:
    diff = np.abs(X[:, None, :] - X[None, :, :])
    return np.sum(diff, axis=2)


def _pairwise_canberra(X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    abs_diff = np.abs(X[:, None, :] - X[None, :, :])
    denom = np.abs(X[:, None, :]) + np.abs(X[None, :, :])

    zero_denom_mask = denom == 0.0
    safe_denom = np.where(zero_denom_mask, 1.0, denom)
    frac = abs_diff / safe_denom
    frac[zero_denom_mask] = 0.0

    matrix = np.sum(frac, axis=2)

    metadata = {
        "canberra_zero_denominator_pairs": int(np.sum(zero_denom_mask)),
        "canberra_safe_zero_handling": True,
    }
    return matrix, metadata


# ============================================================
# Metric implementations
# ============================================================

def compute_euclidean(dataset: LoadedDataset) -> DistanceMatrixResult:
    metric_name = "euclidean"
    dataset = _ensure_dataset(dataset)
    X = _ensure_numeric_matrix(dataset, metric_name)

    try:
        matrix = _pairwise_euclidean(X)
    except Exception as exc:
        raise DistanceComputationError(metric=metric_name, reason=str(exc)) from exc

    np.fill_diagonal(matrix, 0.0)
    return _build_result(dataset, metric_name, matrix)


def compute_chebyshev(dataset: LoadedDataset) -> DistanceMatrixResult:
    metric_name = "chebyshev"
    dataset = _ensure_dataset(dataset)
    X = _ensure_numeric_matrix(dataset, metric_name)

    try:
        matrix = _pairwise_chebyshev(X)
    except Exception as exc:
        raise DistanceComputationError(metric=metric_name, reason=str(exc)) from exc

    np.fill_diagonal(matrix, 0.0)
    return _build_result(dataset, metric_name, matrix)


def compute_manhattan(dataset: LoadedDataset) -> DistanceMatrixResult:
    metric_name = "manhattan"
    dataset = _ensure_dataset(dataset)
    X = _ensure_numeric_matrix(dataset, metric_name)

    try:
        matrix = _pairwise_manhattan(X)
    except Exception as exc:
        raise DistanceComputationError(metric=metric_name, reason=str(exc)) from exc

    np.fill_diagonal(matrix, 0.0)
    return _build_result(dataset, metric_name, matrix)


def compute_canberra(dataset: LoadedDataset) -> DistanceMatrixResult:
    metric_name = "canberra"
    dataset = _ensure_dataset(dataset)
    X = _ensure_numeric_matrix(dataset, metric_name)

    try:
        matrix, extra_metadata = _pairwise_canberra(X)
    except Exception as exc:
        raise DistanceComputationError(metric=metric_name, reason=str(exc)) from exc

    np.fill_diagonal(matrix, 0.0)
    return _build_result(dataset, metric_name, matrix, extra_metadata=extra_metadata)


# ============================================================
# Registry
# ============================================================

_DISTANCE_REGISTRY: dict[str, DistanceFunction] = {
    "euclidean": compute_euclidean,
    "chebyshev": compute_chebyshev,
    "manhattan": compute_manhattan,
    "canberra": compute_canberra,
}


def register_distance_metric(name: str, fn: DistanceFunction) -> None:
    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError("Metric name cannot be empty.")
    if not callable(fn):
        raise TypeError("Distance metric function must be callable.")
    _DISTANCE_REGISTRY[normalized_name] = fn


def get_available_distance_metrics() -> list[str]:
    return sorted(_DISTANCE_REGISTRY.keys())


# ============================================================
# Public API
# ============================================================

def compute_distance_matrix(dataset: LoadedDataset, metric_name: str) -> DistanceMatrixResult:
    dataset = _ensure_dataset(dataset)
    metric = str(metric_name).strip().lower()

    if metric not in _DISTANCE_REGISTRY:
        raise DistanceComputationError(
            metric=metric,
            reason=(
                f"Unsupported metric. Available metrics: "
                f"{', '.join(get_available_distance_metrics())}"
            ),
        )

    try:
        result = _DISTANCE_REGISTRY[metric](dataset)
    except DistanceComputationError:
        raise
    except Exception as exc:
        raise DistanceComputationError(metric=metric, reason=str(exc)) from exc

    return result


def compute_multiple_distance_matrices(
    dataset: LoadedDataset,
    metric_names: list[str],
) -> dict[str, DistanceMatrixResult]:
    if not metric_names:
        raise ValueError("metric_names must not be empty.")

    results: dict[str, DistanceMatrixResult] = {}
    for metric_name in metric_names:
        metric = str(metric_name).strip().lower()
        results[metric] = compute_distance_matrix(dataset, metric)
    return results


def build_distance_summary(result: DistanceMatrixResult) -> DistanceSummary:
    matrix = result.matrix
    diagonal = np.diag(matrix)

    if matrix.shape[0] > 1:
        off_diag_mask = ~np.eye(matrix.shape[0], dtype=bool)
        off_diag_values = matrix[off_diag_mask]
        min_distance = float(np.min(off_diag_values))
        max_distance = float(np.max(off_diag_values))
    else:
        min_distance = 0.0
        max_distance = 0.0

    return DistanceSummary(
        metric_name=result.metric_name,
        n_objects=result.n_objects,
        normalized=result.normalized,
        min_distance=min_distance,
        max_distance=max_distance,
        diagonal_zero=bool(np.allclose(diagonal, 0.0, atol=1e-12)),
        symmetric=bool(np.allclose(matrix, matrix.T, atol=1e-12)),
        metadata=dict(result.metadata),
    )


def distance_result_to_frame(result: DistanceMatrixResult) -> pd.DataFrame:
    return result.to_frame()


def compare_distance_runs(
    raw_dataset: LoadedDataset,
    selected_dataset: LoadedDataset,
    metric_names: list[str],
    normalized_dataset: LoadedDataset | None = None,
) -> dict[str, dict[str, DistanceMatrixResult]]:
    """
    Convenience helper for experiments that need conceptual branches:
    - raw
    - selected
    - normalized (optional)

    Branch semantics:
    - raw       : original dataset as loaded
    - selected  : dataset actually used by the current pipeline run
    - normalized: explicit normalized dataset branch when normalization mode is not none
    """
    results: dict[str, dict[str, DistanceMatrixResult]] = {
        "raw": compute_multiple_distance_matrices(raw_dataset, metric_names),
        "selected": compute_multiple_distance_matrices(selected_dataset, metric_names),
    }

    if normalized_dataset is not None:
        results["normalized"] = compute_multiple_distance_matrices(normalized_dataset, metric_names)

    return results

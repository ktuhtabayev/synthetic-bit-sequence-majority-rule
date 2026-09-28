from __future__ import annotations

from typing import Any, Callable

import numpy as np

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


# Upper bound on each (rows, m, n) float64 temporary the broadcast metrics
# allocate; Canberra holds about five at once. Rows are processed in blocks
# under this size, and every block computes the same elements the full
# broadcast would, so the result is identical and only peak memory changes
# (German, 1000 x 20: 684 MB as one broadcast, about 126 MB in blocks).
_BLOCK_BYTES = 16 * 1024 * 1024


def _row_blocks(n_objects: int, n_features: int) -> list[slice]:
    bytes_per_row = max(1, n_objects * n_features * np.dtype(float).itemsize)
    rows_per_block = max(1, _BLOCK_BYTES // bytes_per_row)
    return [
        slice(start, min(start + rows_per_block, n_objects))
        for start in range(0, n_objects, rows_per_block)
    ]


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    sq_norms = np.sum(X * X, axis=1, keepdims=True)
    squared = sq_norms + sq_norms.T - 2.0 * (X @ X.T)
    squared = np.maximum(squared, 0.0)
    return np.sqrt(squared)


def _pairwise_chebyshev(X: np.ndarray) -> np.ndarray:
    matrix = np.empty((X.shape[0], X.shape[0]), dtype=float)
    for rows in _row_blocks(*X.shape):
        matrix[rows] = np.max(np.abs(X[rows, None, :] - X[None, :, :]), axis=2)
    return matrix


def _pairwise_manhattan(X: np.ndarray) -> np.ndarray:
    matrix = np.empty((X.shape[0], X.shape[0]), dtype=float)
    for rows in _row_blocks(*X.shape):
        matrix[rows] = np.sum(np.abs(X[rows, None, :] - X[None, :, :]), axis=2)
    return matrix


def _pairwise_canberra(X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.empty((X.shape[0], X.shape[0]), dtype=float)
    zero_denominator_pairs = 0

    for rows in _row_blocks(*X.shape):
        abs_diff = np.abs(X[rows, None, :] - X[None, :, :])
        denom = np.abs(X[rows, None, :]) + np.abs(X[None, :, :])

        zero_denom_mask = denom == 0.0
        safe_denom = np.where(zero_denom_mask, 1.0, denom)
        frac = abs_diff / safe_denom
        frac[zero_denom_mask] = 0.0

        matrix[rows] = np.sum(frac, axis=2)
        zero_denominator_pairs += int(np.sum(zero_denom_mask))

    metadata = {
        "canberra_zero_denominator_pairs": zero_denominator_pairs,
        "canberra_safe_zero_handling": True,
    }
    return matrix, metadata


# ============================================================
# Metric implementations
# ============================================================

def _compute_metric(
    dataset: LoadedDataset,
    metric_name: str,
    pairwise: Callable[[np.ndarray], np.ndarray | tuple[np.ndarray, dict[str, Any]]],
) -> DistanceMatrixResult:
    dataset = _ensure_dataset(dataset)
    X = _ensure_numeric_matrix(dataset, metric_name)

    try:
        computed = pairwise(X)
    except Exception as exc:
        raise DistanceComputationError(metric=metric_name, reason=str(exc)) from exc

    matrix, extra_metadata = computed if isinstance(computed, tuple) else (computed, None)
    np.fill_diagonal(matrix, 0.0)
    return _build_result(dataset, metric_name, matrix, extra_metadata=extra_metadata)


def compute_euclidean(dataset: LoadedDataset) -> DistanceMatrixResult:
    return _compute_metric(dataset, "euclidean", _pairwise_euclidean)


def compute_chebyshev(dataset: LoadedDataset) -> DistanceMatrixResult:
    return _compute_metric(dataset, "chebyshev", _pairwise_chebyshev)


def compute_manhattan(dataset: LoadedDataset) -> DistanceMatrixResult:
    return _compute_metric(dataset, "manhattan", _pairwise_manhattan)


def compute_canberra(dataset: LoadedDataset) -> DistanceMatrixResult:
    return _compute_metric(dataset, "canberra", _pairwise_canberra)


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

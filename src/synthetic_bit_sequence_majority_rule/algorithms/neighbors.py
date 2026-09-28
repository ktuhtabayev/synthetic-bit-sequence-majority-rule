from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from synthetic_bit_sequence_majority_rule.domain.errors import NeighborConstructionError
from synthetic_bit_sequence_majority_rule.domain.params import NeighborsConfig
from synthetic_bit_sequence_majority_rule.domain.schema import (
    DistanceMatrixResult,
    NeighborTableResult,
)


# ============================================================
# Small helpers
# ============================================================

def _ensure_distance_result(result: DistanceMatrixResult) -> DistanceMatrixResult:
    if not isinstance(result, DistanceMatrixResult):
        raise TypeError("Expected a DistanceMatrixResult instance.")
    return result


def _validate_neighbors_config(config: NeighborsConfig) -> None:
    if config.tie_break_rule != "index_ascending":
        raise NeighborConstructionError(
            metric="unknown",
            reason=f"Unsupported tie_break_rule '{config.tie_break_rule}'.",
        )
    if not config.exclude_self:
        raise NeighborConstructionError(
            metric="unknown",
            reason="exclude_self must be true for the current problem definition.",
        )


# ============================================================
# Core construction
# ============================================================

def build_neighbor_table(
    distance_result: DistanceMatrixResult,
    neighbors_config: NeighborsConfig,
) -> NeighborTableResult:
    """
    Build ordered neighbor labels and distances from a distance matrix.

    Current supported rule:
    - sort by ascending distance
    - ties resolved by ascending original object index
    - self is excluded

    This stage is upstream of the majority / k-rule stage, so it does not
    depend on formula-based k_max. It always builds the full ordered neighbor
    list of length (m - 1).
    """
    result = _ensure_distance_result(distance_result)
    _validate_neighbors_config(neighbors_config)

    object_labels = list(result.object_labels)
    matrix = result.matrix
    n_objects = result.n_objects

    if n_objects <= 1:
        raise NeighborConstructionError(
            metric=result.metric_name,
            reason="At least two objects are required to build neighbors.",
        )

    # A stable sort orders each row by ascending distance and breaks ties by
    # ascending original index. The matrix is validated as finite float, so
    # this is exactly the (distance, index) ordering.
    order = np.argsort(matrix, axis=1, kind="stable")
    if neighbors_config.exclude_self:
        # Every row holds its own index exactly once, so dropping it leaves m - 1.
        not_self = order != np.arange(n_objects)[:, np.newaxis]
        order = order[not_self].reshape(n_objects, n_objects - 1)

    neighbor_labels: list[list[str]] = np.asarray(object_labels, dtype=object)[order].tolist()
    neighbor_distances: list[list[float]] = np.take_along_axis(matrix, order, axis=1).tolist()

    return NeighborTableResult(
        metric_name=result.metric_name,
        object_labels=object_labels,
        neighbor_labels=neighbor_labels,
        neighbor_distances=neighbor_distances,
        tie_break_rule=neighbors_config.tie_break_rule,
        metadata={
            "source_metric": result.metric_name,
            "normalization_mode": result.metadata.get("normalization_mode", "none"),
            "with_normalization": result.metadata.get("with_normalization", False),
            "n_objects": result.n_objects,
            "neighbor_count": result.n_objects - 1,
            "tie_break_rule": neighbors_config.tie_break_rule,
            "exclude_self": neighbors_config.exclude_self,
        },
    )


def build_multiple_neighbor_tables(
    distance_results: dict[str, DistanceMatrixResult],
    neighbors_config: NeighborsConfig,
) -> dict[str, NeighborTableResult]:
    if not distance_results:
        raise ValueError("distance_results must not be empty.")

    results: dict[str, NeighborTableResult] = {}
    for metric_name, distance_result in distance_results.items():
        results[metric_name] = build_neighbor_table(distance_result, neighbors_config)
    return results


def neighbor_combined_frame(result: NeighborTableResult) -> pd.DataFrame:
    """
    Build one combined frame with:
    Object | NN1 | d1 | NN2 | d2 | ...

    Useful for export and quick comparison with Excel.
    """
    shape = (result.n_objects, result.neighbor_count)
    labels = np.asarray(result.neighbor_labels, dtype=object).reshape(shape)
    distances = np.asarray(result.neighbor_distances, dtype=float).reshape(shape)
    data: dict[str, Any] = {"Object": list(result.object_labels)}

    for idx in range(result.neighbor_count):
        rank = idx + 1
        data[f"NN{rank}"] = labels[:, idx]
        data[f"d{rank}"] = distances[:, idx]

    return pd.DataFrame(data)

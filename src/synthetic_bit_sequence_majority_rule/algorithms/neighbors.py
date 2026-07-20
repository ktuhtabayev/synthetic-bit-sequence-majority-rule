from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from synthetic_bit_sequence_majority_rule.domain.errors import NeighborConstructionError
from synthetic_bit_sequence_majority_rule.domain.params import NeighborsConfig
from synthetic_bit_sequence_majority_rule.domain.schema import (
    DistanceMatrixResult,
    NeighborTableResult,
)


# ============================================================
# Internal summary dataclass
# ============================================================

@dataclass(slots=True)
class NeighborSummary:
    metric_name: str
    n_objects: int
    neighbor_count: int
    tie_break_rule: str
    exclude_self: bool
    normalized: bool
    metadata: dict[str, Any] = field(default_factory=dict)


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

    neighbor_labels: list[list[str]] = []
    neighbor_distances: list[list[float]] = []

    for row_idx in range(n_objects):
        pairs: list[tuple[float, int, str]] = []

        for col_idx in range(n_objects):
            if neighbors_config.exclude_self and row_idx == col_idx:
                continue

            distance_value = float(matrix[row_idx, col_idx])
            label = object_labels[col_idx]
            pairs.append((distance_value, col_idx, label))

        # primary key: distance
        # secondary key: original index (stable tie-break)
        pairs.sort(key=lambda x: (x[0], x[1]))

        row_neighbor_labels = [item[2] for item in pairs]
        row_neighbor_distances = [item[0] for item in pairs]

        neighbor_labels.append(row_neighbor_labels)
        neighbor_distances.append(row_neighbor_distances)

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


def compare_neighbor_runs(
    raw_distance_results: dict[str, DistanceMatrixResult],
    selected_distance_results: dict[str, DistanceMatrixResult],
    neighbors_config: NeighborsConfig,
    normalized_distance_results: dict[str, DistanceMatrixResult] | None = None,
) -> dict[str, dict[str, NeighborTableResult]]:
    """
    Build neighbor tables for conceptual branches:
    - raw
    - selected
    - normalized (optional)

    Branch semantics:
    - raw       : original dataset
    - selected  : dataset actually selected by default.yaml normalization mode
    - normalized: explicit normalized dataset branch when normalization mode is not none
    """
    results: dict[str, dict[str, NeighborTableResult]] = {
        "raw": build_multiple_neighbor_tables(raw_distance_results, neighbors_config),
        "selected": build_multiple_neighbor_tables(selected_distance_results, neighbors_config),
    }

    if normalized_distance_results is not None:
        results["normalized"] = build_multiple_neighbor_tables(
            normalized_distance_results,
            neighbors_config,
        )

    return results


# ============================================================
# Summary / conversion helpers
# ============================================================

def build_neighbor_summary(result: NeighborTableResult) -> NeighborSummary:
    return NeighborSummary(
        metric_name=result.metric_name,
        n_objects=result.n_objects,
        neighbor_count=result.neighbor_count,
        tie_break_rule=result.tie_break_rule,
        exclude_self=True,
        normalized=bool(result.metadata.get("with_normalization", False)),
        metadata=dict(result.metadata),
    )


def neighbor_labels_to_frame(result: NeighborTableResult) -> pd.DataFrame:
    return result.labels_frame()


def neighbor_distances_to_frame(result: NeighborTableResult) -> pd.DataFrame:
    return result.distances_frame()


def neighbor_combined_frame(result: NeighborTableResult) -> pd.DataFrame:
    """
    Build one combined frame with:
    Object | NN1 | d1 | NN2 | d2 | ...

    Useful for export and quick comparison with Excel.
    """
    n_neighbors = result.neighbor_count
    data: dict[str, list[Any]] = {"Object": list(result.object_labels)}

    for idx in range(n_neighbors):
        rank = idx + 1
        data[f"NN{rank}"] = [row[idx] for row in result.neighbor_labels]
        data[f"d{rank}"] = [row[idx] for row in result.neighbor_distances]

    return pd.DataFrame(data)
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from synthetic_bit_sequence_majority_rule.algorithms.statistics import (
    ComplexityTableResult,
    StabilityTableResult,
)


PCA_3D_UNAVAILABLE_MESSAGE = "PCA 3D requires at least 3 metrics and 3 stability coordinates."
PCA_NOISE_TOLERANCE = 1e-12
PCA_OVERLAP_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class MetaObjectsResult:
    frame: pd.DataFrame
    k_mapping: dict[str, int] = field(default_factory=dict)
    identity_columns: list[str] = field(default_factory=lambda: ["Metric"])


@dataclass(frozen=True, slots=True)
class PCAProjectionResult:
    requested_components: int
    available: bool
    frame: pd.DataFrame
    explained_variance_ratio: list[float] = field(default_factory=list)
    message: str = ""


@dataclass(frozen=True, slots=True)
class NormalizationComparisonResult:
    summary_frame: pd.DataFrame
    meta_objects: MetaObjectsResult
    pca_2d: PCAProjectionResult
    pca_3d: PCAProjectionResult


def build_meta_objects_from_stability(
    stability_results: dict[str, StabilityTableResult],
) -> MetaObjectsResult:
    all_k_values = sorted(
        {
            int(row.k_values[-1])
            for result in stability_results.values()
            for row in result.rows
            if row.k_values
        }
    )
    k_mapping = {f"y{idx}": k_value for idx, k_value in enumerate(all_k_values, start=1)}

    rows: list[dict[str, object]] = []
    for metric_name, result in stability_results.items():
        stability_by_k = {
            int(row.k_values[-1]): float(row.stability)
            for row in result.rows
            if row.k_values
        }
        row_data: dict[str, object] = {"Metric": metric_name.title()}
        for column, k_value in k_mapping.items():
            row_data[column] = stability_by_k.get(k_value, np.nan)
        rows.append(row_data)

    return MetaObjectsResult(
        frame=pd.DataFrame(rows),
        k_mapping=k_mapping,
        identity_columns=["Metric"],
    )


def build_normalization_meta_objects(
    raw_results: dict[str, StabilityTableResult],
    minmax_results: dict[str, StabilityTableResult],
) -> MetaObjectsResult:
    if list(raw_results) != list(minmax_results):
        raise ValueError("None and MinMax branches must contain the same metrics in the same order.")

    raw_meta = build_meta_objects_from_stability(raw_results)
    minmax_meta = build_meta_objects_from_stability(minmax_results)
    if raw_meta.k_mapping != minmax_meta.k_mapping:
        raise ValueError("None and MinMax branches must use identical stability k-coordinates.")

    raw_by_metric = raw_meta.frame.set_index("Metric")
    minmax_by_metric = minmax_meta.frame.set_index("Metric")
    rows: list[dict[str, object]] = []
    for metric in raw_meta.frame["Metric"].astype(str):
        for normalization, source in (
            ("None", raw_by_metric),
            ("MinMax", minmax_by_metric),
        ):
            row: dict[str, object] = {
                "Metric": metric,
                "Normalization": normalization,
            }
            for column in raw_meta.k_mapping:
                row[column] = float(source.loc[metric, column])
            rows.append(row)

    return MetaObjectsResult(
        frame=pd.DataFrame(rows),
        k_mapping=dict(raw_meta.k_mapping),
        identity_columns=["Metric", "Normalization"],
    )


def build_complexity_comparison_frame(
    raw_result: ComplexityTableResult,
    minmax_result: ComplexityTableResult,
    *,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    raw = raw_result.to_frame().rename(
        columns={
            "ComponentCount": "ComponentCountNone",
            "MeanStability": "MeanStabilityNone",
            "Complexity": "ComplexityNone",
            "Interpretation": "InterpretationNone",
        }
    )
    minmax = minmax_result.to_frame().rename(
        columns={
            "ComponentCount": "ComponentCountMinMax",
            "MeanStability": "MeanStabilityMinMax",
            "Complexity": "ComplexityMinMax",
            "Interpretation": "InterpretationMinMax",
        }
    )
    frame = raw.merge(minmax, on="Metric", how="inner", validate="one_to_one")
    if len(frame) != len(raw) or len(frame) != len(minmax):
        raise ValueError("None and MinMax complexity tables must contain identical metrics.")
    if not (
        frame["ComponentCountNone"].to_numpy()
        == frame["ComponentCountMinMax"].to_numpy()
    ).all():
        raise ValueError("None and MinMax complexity scores must use the same component count.")

    frame.insert(1, "ComponentCount", frame.pop("ComponentCountNone"))
    frame = frame.drop(columns=["ComponentCountMinMax"])
    frame["DeltaComplexity"] = frame["ComplexityMinMax"] - frame["ComplexityNone"]

    def effect(delta: float) -> str:
        if delta > tolerance:
            return "MinMax increased complexity"
        if delta < -tolerance:
            return "MinMax reduced complexity"
        return "No meaningful change"

    frame["Effect"] = frame["DeltaComplexity"].map(effect)
    return frame


def build_normalization_comparison(
    raw_stability: dict[str, StabilityTableResult],
    minmax_stability: dict[str, StabilityTableResult],
    raw_complexity: ComplexityTableResult,
    minmax_complexity: ComplexityTableResult,
) -> NormalizationComparisonResult:
    meta_objects = build_normalization_meta_objects(raw_stability, minmax_stability)
    return NormalizationComparisonResult(
        summary_frame=build_complexity_comparison_frame(raw_complexity, minmax_complexity),
        meta_objects=meta_objects,
        pca_2d=apply_pca_to_meta_objects(meta_objects, requested_components=2),
        pca_3d=apply_pca_to_meta_objects(meta_objects, requested_components=3),
    )


def apply_pca_to_meta_objects(
    meta_objects: MetaObjectsResult,
    requested_components: int,
) -> PCAProjectionResult:
    value_columns = list(meta_objects.k_mapping.keys())
    if not value_columns or meta_objects.frame.empty:
        return PCAProjectionResult(
            requested_components=requested_components,
            available=False,
            frame=pd.DataFrame(),
            message=_unavailable_message(requested_components),
        )

    values = meta_objects.frame[value_columns].to_numpy(dtype=float)
    if np.isnan(values).any():
        column_means = np.nanmean(values, axis=0)
        values = np.where(np.isnan(values), column_means, values)

    max_components = min(values.shape[0], values.shape[1])
    if max_components < requested_components:
        return PCAProjectionResult(
            requested_components=requested_components,
            available=False,
            frame=pd.DataFrame(),
            message=_unavailable_message(requested_components),
        )

    pca = PCA(n_components=requested_components)
    coordinates = clean_pca_coordinates(pca.fit_transform(values))

    data: dict[str, list[object]] = {
        column: meta_objects.frame[column].tolist()
        for column in meta_objects.identity_columns
    }
    for component_idx in range(requested_components):
        data[f"PC{component_idx + 1}"] = coordinates[:, component_idx].tolist()
    data["ExplainedVarianceRatio"] = [float(sum(pca.explained_variance_ratio_))] * len(meta_objects.frame)

    return PCAProjectionResult(
        requested_components=requested_components,
        available=True,
        frame=pd.DataFrame(data),
        explained_variance_ratio=[float(value) for value in pca.explained_variance_ratio_],
    )


def clean_pca_coordinates(
    coordinates: np.ndarray,
    tolerance: float = PCA_NOISE_TOLERANCE,
) -> np.ndarray:
    cleaned = np.asarray(coordinates, dtype=float).copy()
    cleaned[np.abs(cleaned) < tolerance] = 0.0
    return cleaned


def near_zero_axis_notes(
    frame: pd.DataFrame,
    *,
    tolerance: float = PCA_NOISE_TOLERANCE,
) -> list[str]:
    notes: list[str] = []
    for column in [col for col in frame.columns if str(col).startswith("PC")]:
        values = frame[column].to_numpy(dtype=float)
        if values.size and float(np.max(values) - np.min(values)) <= tolerance:
            notes.append(
                f"{column} has near-zero variance; visible separation on this axis is not meaningful."
            )
    return notes


def label_offsets_for_points(
    points: Sequence[tuple[float, ...]],
    *,
    tolerance: float = PCA_OVERLAP_TOLERANCE,
) -> list[tuple[int, int]]:
    offsets = [(5, 5) for _ in points]
    groups: list[list[int]] = []

    for idx, point in enumerate(points):
        matched = False
        for group in groups:
            first = points[group[0]]
            if len(point) == len(first) and all(abs(a - b) <= tolerance for a, b in zip(point, first)):
                group.append(idx)
                matched = True
                break
        if not matched:
            groups.append([idx])

    pattern = [
        (5, 5),
        (5, -15),
        (-55, 5),
        (-55, -15),
        (20, 18),
        (20, -28),
        (-80, 18),
        (-80, -28),
    ]
    for group in groups:
        if len(group) == 1:
            continue
        for order, idx in enumerate(group):
            offsets[idx] = pattern[order % len(pattern)]

    return offsets


def _unavailable_message(requested_components: int) -> str:
    if requested_components == 3:
        return PCA_3D_UNAVAILABLE_MESSAGE
    return f"PCA {requested_components}D requires at least {requested_components} metrics and {requested_components} stability coordinates."

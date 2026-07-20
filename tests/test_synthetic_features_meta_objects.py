from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from synthetic_bit_sequence_majority_rule.algorithms.statistics import (
    StabilityRow,
    StabilityTableResult,
    build_complexity_table,
)
from synthetic_bit_sequence_majority_rule.algorithms.meta_objects import (
    PCA_3D_UNAVAILABLE_MESSAGE,
    apply_pca_to_meta_objects,
    build_normalization_comparison,
    build_normalization_meta_objects,
    build_meta_objects_from_stability,
    clean_pca_coordinates,
    label_offsets_for_points,
    near_zero_axis_notes,
)
from synthetic_bit_sequence_majority_rule.gui.synthetic_features import (
    build_synthetic_binary_frame,
    build_synthetic_decimal_frame,
)


def make_synthetic_feature_result() -> SimpleNamespace:
    return SimpleNamespace(
        object_labels=["S1", "S2"],
        b_reduced=np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            ]
        ),
        reduced_k_values=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        classes=np.array([1, 2]),
    )


def test_synthetic_binary_prefix_table() -> None:
    frame = build_synthetic_binary_frame(make_synthetic_feature_result())

    assert list(frame.columns[:5]) == ["Object", "b3", "b3, b5", "b3, ..., b7", "b3, ..., b9"]
    assert frame.loc[0, "b3"] == "1"
    assert frame.loc[0, "b3, b5"] == "11"
    assert frame.loc[0, "b3, ..., b7"] == "111"
    assert frame.loc[1, "b3, ..., b21"] == "1111110000"
    assert frame.loc[1, "Class"] == 2


def test_synthetic_decimal_prefix_table() -> None:
    frame = build_synthetic_decimal_frame(make_synthetic_feature_result())

    assert frame.loc[0, "b3"] == 1
    assert frame.loc[0, "b3, b5"] == 3
    assert frame.loc[0, "b3, ..., b7"] == 7
    assert frame.loc[0, "b3, ..., b21"] == 1023
    assert frame.loc[1, "b3, ..., b21"] == 1008


def make_stability_result(metric_name: str, values: list[float]) -> StabilityTableResult:
    reduced_k = [3, 5, 7]
    return StabilityTableResult(
        metric_name=metric_name,
        rows=[
            StabilityRow(
                metric_name=metric_name,
                representation=f"prefix_{idx}",
                k_values=reduced_k[:idx],
                unique_values=idx,
                object_count=42,
                stability=value,
                interpretation="High" if value > 0.8 else "Satisfactory",
            )
            for idx, value in enumerate(values, start=1)
        ],
    )


def test_meta_object_stability_matrix_orders_metrics_and_k_values() -> None:
    meta = build_meta_objects_from_stability(
        {
            "euclidean": make_stability_result("euclidean", [0.61, 0.64, 0.70]),
            "canberra": make_stability_result("canberra", [0.52, 0.58, 0.90]),
        }
    )

    assert meta.k_mapping == {"y1": 3, "y2": 5, "y3": 7}
    assert list(meta.frame.columns) == ["Metric", "y1", "y2", "y3"]
    assert meta.frame["Metric"].tolist() == ["Euclidean", "Canberra"]
    assert meta.frame.loc[1, "y3"] == 0.90


def test_pca_2d_table_generation() -> None:
    meta = build_meta_objects_from_stability(
        {
            "euclidean": make_stability_result("euclidean", [0.61, 0.64, 0.70]),
            "chebyshev": make_stability_result("chebyshev", [0.63, 0.66, 0.72]),
            "canberra": make_stability_result("canberra", [0.52, 0.58, 0.90]),
        }
    )

    result = apply_pca_to_meta_objects(meta, requested_components=2)

    assert result.available is True
    assert list(result.frame.columns) == ["Metric", "PC1", "PC2", "ExplainedVarianceRatio"]
    assert len(result.frame) == 3
    assert len(result.explained_variance_ratio) == 2


def test_pca_3d_unavailable_for_small_default_style_input() -> None:
    meta = build_meta_objects_from_stability(
        {
            "euclidean": make_stability_result("euclidean", [0.61, 0.64, 0.70]),
            "canberra": make_stability_result("canberra", [0.52, 0.58, 0.90]),
        }
    )

    result = apply_pca_to_meta_objects(meta, requested_components=3)

    assert result.available is False
    assert result.message == PCA_3D_UNAVAILABLE_MESSAGE


def test_pca_coordinate_cleanup_removes_tiny_noise_only() -> None:
    cleaned = clean_pca_coordinates(
        np.array(
            [
                [1e-17, 0.25],
                [-9e-13, -0.5],
                [2e-12, 0.0],
            ]
        )
    )

    assert cleaned[0, 0] == 0.0
    assert cleaned[1, 0] == 0.0
    assert cleaned[2, 0] == 2e-12
    assert cleaned[0, 1] == 0.25


def test_near_zero_axis_notes_report_flat_components() -> None:
    meta = build_meta_objects_from_stability(
        {
            "euclidean": make_stability_result("euclidean", [0.61, 0.64, 0.70]),
            "chebyshev": make_stability_result("chebyshev", [0.63, 0.66, 0.72]),
            "canberra": make_stability_result("canberra", [0.52, 0.58, 0.90]),
            "manhattan": make_stability_result("manhattan", [0.61, 0.64, 0.70]),
        }
    )
    result = apply_pca_to_meta_objects(meta, requested_components=3)

    notes = near_zero_axis_notes(result.frame)

    assert any(note.startswith("PC3 has near-zero variance") for note in notes)


def test_label_offsets_for_overlapping_points_are_distinct() -> None:
    offsets = label_offsets_for_points(
        [
            (0.0, 0.0),
            (0.0, 0.0),
            (1e-10, -1e-10),
            (1.0, 1.0),
        ]
    )

    assert len(set(offsets[:3])) == 3
    assert offsets[3] == (5, 5)


def test_none_vs_minmax_meta_objects_use_one_shared_pca_identity_space() -> None:
    raw = {
        "euclidean": make_stability_result("euclidean", [0.8, 0.85, 0.9]),
        "chebyshev": make_stability_result("chebyshev", [0.7, 0.75, 0.8]),
        "canberra": make_stability_result("canberra", [0.6, 0.65, 0.7]),
        "manhattan": make_stability_result("manhattan", [0.75, 0.8, 0.85]),
    }
    minmax = {
        "euclidean": make_stability_result("euclidean", [0.65, 0.7, 0.75]),
        "chebyshev": make_stability_result("chebyshev", [0.6, 0.65, 0.7]),
        "canberra": make_stability_result("canberra", [0.55, 0.6, 0.65]),
        "manhattan": make_stability_result("manhattan", [0.7, 0.75, 0.8]),
    }

    meta = build_normalization_meta_objects(raw, minmax)
    comparison = build_normalization_comparison(
        raw,
        minmax,
        build_complexity_table(raw),
        build_complexity_table(minmax),
    )

    assert meta.identity_columns == ["Metric", "Normalization"]
    assert len(meta.frame) == 8
    assert meta.frame.groupby("Metric")["Normalization"].apply(set).eq({"None", "MinMax"}).all()
    assert comparison.pca_2d.available is True
    assert comparison.pca_3d.available is True
    assert list(comparison.pca_2d.frame.columns[:2]) == ["Metric", "Normalization"]
    assert len(comparison.pca_2d.frame) == 8
    assert (comparison.summary_frame["DeltaComplexity"] > 0).all()
    assert comparison.summary_frame["Effect"].eq("MinMax increased complexity").all()

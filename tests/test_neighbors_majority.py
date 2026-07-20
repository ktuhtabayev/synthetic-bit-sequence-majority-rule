from __future__ import annotations

import numpy as np

from synthetic_bit_sequence_majority_rule.algorithms.majority import (
    _decimal_values_from_sequences,
    a_full_frame,
    a_reduced_frame,
    b_full_frame,
    b_reduced_frame,
    build_majority_matrices,
    compute_formula_based_kmax,
)
from synthetic_bit_sequence_majority_rule.algorithms.neighbors import build_neighbor_table
from synthetic_bit_sequence_majority_rule.domain.params import (
    BinarySequenceConfig,
    DecimalEncodingConfig,
    KValuesConfig,
    MajorityRuleConfig,
    NeighborsConfig,
)
from synthetic_bit_sequence_majority_rule.domain.schema import (
    DistanceMatrixResult,
    LoadedDataset,
    NeighborTableResult,
)


def test_neighbors_exclude_self_and_break_ties_by_index() -> None:
    distance = DistanceMatrixResult(
        metric_name="euclidean",
        matrix=np.array(
            [
                [0.0, 1.0, 1.0, 2.0],
                [1.0, 0.0, 3.0, 3.0],
                [1.0, 3.0, 0.0, 2.0],
                [2.0, 3.0, 2.0, 0.0],
            ]
        ),
        object_labels=["S1", "S2", "S3", "S4"],
    )

    result = build_neighbor_table(distance, NeighborsConfig())

    assert result.neighbor_labels[0] == ["S2", "S3", "S4"]
    assert "S1" not in result.neighbor_labels[0]
    assert result.neighbor_labels[1] == ["S1", "S3", "S4"]


def make_majority_dataset() -> LoadedDataset:
    return LoadedDataset(
        source_path=None,
        source_format="test",
        X=np.arange(20, dtype=float).reshape(10, 2),
        y=np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2]),
        object_labels=[f"S{i}" for i in range(1, 11)],
        feature_names=["x1", "x2"],
        class_column="Class",
    )


def make_neighbor_result_for_majority() -> NeighborTableResult:
    labels = [f"S{i}" for i in range(1, 11)]
    first_row = ["S7", "S2", "S3", "S8", "S9", "S4", "S5", "S6", "S10"]
    rows = [first_row]
    for label in labels[1:6]:
        rows.append([candidate for candidate in ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"] if candidate != label])
    for label in labels[6:]:
        rows.append([candidate for candidate in ["S7", "S8", "S9", "S10", "S1", "S2", "S3", "S4", "S5", "S6"] if candidate != label])

    return NeighborTableResult(
        metric_name="euclidean",
        object_labels=labels,
        neighbor_labels=rows,
        neighbor_distances=[[float(i) for i in range(1, 10)] for _ in labels],
    )


def test_majority_dynamic_kmax_a_b_sequences_and_decimal_encoding() -> None:
    dataset = make_majority_dataset()
    neighbors = make_neighbor_result_for_majority()

    result = build_majority_matrices(
        dataset=dataset,
        neighbor_result=neighbors,
        k_values_config=KValuesConfig(),
        majority_rule_config=MajorityRuleConfig(),
        binary_sequence_config=BinarySequenceConfig(),
        decimal_encoding_config=DecimalEncodingConfig(),
    )

    assert compute_formula_based_kmax(dataset) == 5
    assert result.full_k_values == [1, 2, 3, 4, 5]
    assert result.reduced_k_values == [3, 5]
    assert result.a_full.shape == (10, 5)
    assert result.a_reduced.shape == (10, 2)
    assert result.b_full.shape == (10, 5)
    assert result.b_reduced.shape == (10, 2)

    assert result.same_class_indicators[0, :5].tolist() == [0, 1, 1, 0, 0]
    assert result.a_full[0].tolist() == [0, 1, 2, 2, 2]
    assert result.b_full[0].tolist() == [0, 0, 1, 0, 0]
    assert result.b_reduced[0].tolist() == [1, 0]
    assert result.binary_sequences[0] == "10"
    assert int(result.decimal_values[0]) == 2

    assert list(a_full_frame(result).columns) == ["Object", "Class", "a1", "a2", "a3", "a4", "a5"]
    assert list(a_reduced_frame(result).columns) == ["Object", "Class", "a3", "a5"]
    assert list(b_full_frame(result).columns) == ["Object", "Class", "b1", "b2", "b3", "b4", "b5"]
    assert list(b_reduced_frame(result).columns) == ["Object", "Class", "b3", "b5"]


def test_decimal_encoding_supports_123_bit_sequences() -> None:
    sequence = "1" * 123
    values = _decimal_values_from_sequences([sequence], DecimalEncodingConfig())

    assert values.dtype == object
    assert values.shape == (1,)
    assert values[0] == (1 << 123) - 1
    assert values[0] > np.iinfo(np.int64).max

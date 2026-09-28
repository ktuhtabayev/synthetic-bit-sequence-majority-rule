from __future__ import annotations

import numpy as np
import pytest

from synthetic_bit_sequence_majority_rule.algorithms.majority import (
    _decimal_values_from_sequences,
    _same_class_indicator_matrix,
    a_full_frame,
    a_reduced_frame,
    b_full_frame,
    b_reduced_frame,
    bit_strings,
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


def _reference_neighbor_order(matrix: np.ndarray) -> list[list[int]]:
    """The defining rule: other objects by ascending distance, ties by ascending index."""
    n = matrix.shape[0]
    return [
        [col for _, col in sorted((float(matrix[row, col]), col) for col in range(n) if col != row)]
        for row in range(n)
    ]


def _random_tied_distance_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    # Few distinct integer values, so most rows contain many exact ties.
    upper = np.triu(rng.integers(1, 4, size=(n, n)).astype(float), k=1)
    return upper + upper.T


def test_neighbor_order_matches_the_reference_rule_on_heavily_tied_distances() -> None:
    rng = np.random.default_rng(7)
    for n in (2, 3, 9, 40):
        matrix = _random_tied_distance_matrix(rng, n)
        labels = [f"S{i}" for i in range(1, n + 1)]
        result = build_neighbor_table(
            DistanceMatrixResult(metric_name="manhattan", matrix=matrix, object_labels=labels),
            NeighborsConfig(),
        )

        expected = _reference_neighbor_order(matrix)
        assert result.neighbor_labels == [[labels[col] for col in row] for row in expected]
        assert result.neighbor_distances == [
            [float(matrix[r, col]) for col in row] for r, row in enumerate(expected)
        ]


def test_same_class_indicators_match_the_neighbor_classes() -> None:
    rng = np.random.default_rng(11)
    n = 30
    labels = [f"S{i}" for i in range(1, n + 1)]
    y = np.array([1, 2] + rng.integers(1, 3, size=n - 2).tolist())
    dataset = LoadedDataset(
        source_path=None,
        source_format="test",
        X=rng.normal(size=(n, 3)),
        y=y,
        object_labels=labels,
        feature_names=["x1", "x2", "x3"],
        class_column="Class",
    )
    neighbors = build_neighbor_table(
        DistanceMatrixResult(
            metric_name="euclidean",
            matrix=_random_tied_distance_matrix(rng, n),
            object_labels=labels,
        ),
        NeighborsConfig(),
    )

    indicators = _same_class_indicator_matrix(dataset, neighbors)

    index = {label: i for i, label in enumerate(labels)}
    expected = [
        [1 if y[index[label]] == y[row] else 0 for label in neighbors.neighbor_labels[row]]
        for row in range(n)
    ]
    assert indicators.tolist() == expected


def test_neighbor_table_names_the_first_invalid_label() -> None:
    labels = ["S1", "S2", "S3"]
    distances = [[1.0, 2.0]] * 3

    def table(rows: list[list[str]]) -> NeighborTableResult:
        return NeighborTableResult(
            metric_name="euclidean",
            object_labels=labels,
            neighbor_labels=rows,
            neighbor_distances=distances,
        )

    with pytest.raises(ValueError, match="Unknown neighbor label 'S9' found in row 0"):
        table([["S2", "S9"], ["S1", "S3"], ["S1", "S2"]])
    with pytest.raises(ValueError, match="Self-neighbor found in row 1 for object 'S2'"):
        table([["S2", "S3"], ["S2", "S3"], ["S1", "S2"]])
    with pytest.raises(ValueError, match="Duplicate neighbor labels found in row 2"):
        table([["S2", "S3"], ["S1", "S3"], ["S1", "S1"]])


def test_bit_strings_render_each_row_in_column_order() -> None:
    assert bit_strings(np.array([[1, 0, 1], [0, 0, 0]])) == ["101", "000"]
    assert bit_strings(np.array([[1.0, 0.0]])) == ["10"]
    assert bit_strings(np.zeros((2, 0), dtype=int)) == ["", ""]
    with pytest.raises(ValueError, match="0/1"):
        bit_strings(np.array([[0, 2]]))


def test_decimal_encoding_supports_123_bit_sequences() -> None:
    sequence = "1" * 123
    values = _decimal_values_from_sequences([sequence], DecimalEncodingConfig())

    assert values.dtype == object
    assert values.shape == (1,)
    assert values[0] == (1 << 123) - 1
    assert values[0] > np.iinfo(np.int64).max

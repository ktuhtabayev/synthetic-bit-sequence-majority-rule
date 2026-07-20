from __future__ import annotations

import numpy as np

from synthetic_bit_sequence_majority_rule.algorithms.distances import compute_distance_matrix
from synthetic_bit_sequence_majority_rule.domain.schema import LoadedDataset


def make_distance_dataset() -> LoadedDataset:
    return LoadedDataset(
        source_path=None,
        source_format="test",
        X=np.array(
            [
                [1.0, 1.0],
                [4.0, 5.0],
                [1.0, 5.0],
            ]
        ),
        y=np.array([1, 1, 2]),
        object_labels=["S1", "S2", "S3"],
        feature_names=["x1", "x2"],
        class_column="Class",
    )


def test_euclidean_distance_formula() -> None:
    result = compute_distance_matrix(make_distance_dataset(), "euclidean")

    assert np.isclose(result.matrix[0, 1], 5.0)
    assert np.isclose(result.matrix[0, 2], 4.0)
    assert np.isclose(result.matrix[1, 2], 3.0)
    assert np.allclose(np.diag(result.matrix), 0.0)


def test_chebyshev_distance_formula() -> None:
    result = compute_distance_matrix(make_distance_dataset(), "chebyshev")

    assert np.isclose(result.matrix[0, 1], 4.0)
    assert np.isclose(result.matrix[0, 2], 4.0)
    assert np.isclose(result.matrix[1, 2], 3.0)


def test_manhattan_distance_formula() -> None:
    result = compute_distance_matrix(make_distance_dataset(), "manhattan")

    assert np.isclose(result.matrix[0, 1], 7.0)
    assert np.isclose(result.matrix[0, 2], 4.0)
    assert np.isclose(result.matrix[1, 2], 3.0)


def test_canberra_distance_formula() -> None:
    result = compute_distance_matrix(make_distance_dataset(), "canberra")

    expected_s1_s2 = abs(1 - 4) / (abs(1) + abs(4)) + abs(1 - 5) / (abs(1) + abs(5))
    expected_s1_s3 = abs(1 - 1) / (abs(1) + abs(1)) + abs(1 - 5) / (abs(1) + abs(5))
    assert np.isclose(result.matrix[0, 1], expected_s1_s2)
    assert np.isclose(result.matrix[0, 2], expected_s1_s3)
    assert np.allclose(result.matrix, result.matrix.T)

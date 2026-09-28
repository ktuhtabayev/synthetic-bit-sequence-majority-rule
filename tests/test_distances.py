from __future__ import annotations

import numpy as np
import pytest

from synthetic_bit_sequence_majority_rule.algorithms import distances
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


def _one_shot_kernels(X: np.ndarray) -> dict[str, np.ndarray]:
    """Each metric as a single full (m, m, n) broadcast, with no row blocks."""
    abs_diff = np.abs(X[:, None, :] - X[None, :, :])
    denom = np.abs(X[:, None, :]) + np.abs(X[None, :, :])
    frac = abs_diff / np.where(denom == 0.0, 1.0, denom)
    frac[denom == 0.0] = 0.0
    return {
        "chebyshev": np.max(abs_diff, axis=2),
        "manhattan": np.sum(abs_diff, axis=2),
        "canberra": np.sum(frac, axis=2),
    }


@pytest.mark.parametrize("block_bytes", [1, 200, 10_000])
def test_row_blocked_kernels_equal_the_one_shot_broadcast_exactly(monkeypatch, block_bytes) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(37, 5))
    X[:, 2] = 0.0  # zero Canberra denominators
    X[::4, 3] = -X[::4, 3]
    dataset = LoadedDataset(
        source_path=None,
        source_format="test",
        X=X,
        y=np.array([1, 2] * 18 + [1]),
        object_labels=[f"S{i}" for i in range(1, 38)],
        feature_names=[f"x{i}" for i in range(1, 6)],
        class_column="Class",
    )
    monkeypatch.setattr(distances, "_BLOCK_BYTES", block_bytes)
    assert len(distances._row_blocks(37, 5)) > 1 or block_bytes >= 37 * 5 * 8

    expected = _one_shot_kernels(X)
    for metric, matrix in expected.items():
        np.fill_diagonal(matrix, 0.0)
        assert np.array_equal(compute_distance_matrix(dataset, metric).matrix, matrix), metric

    zero_pairs = compute_distance_matrix(dataset, "canberra").metadata["canberra_zero_denominator_pairs"]
    assert zero_pairs == 37 * 37  # column 2 is zero for every pair

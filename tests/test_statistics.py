from __future__ import annotations

import numpy as np
import pytest

from synthetic_bit_sequence_majority_rule.algorithms.statistics import (
    StabilityRow,
    StabilityTableResult,
    build_complexity_table,
    build_final_comparison,
    build_membership_table,
    build_sequence_statistics,
    build_stability_table,
    stability_interpretation,
)
from synthetic_bit_sequence_majority_rule.domain.errors import StatisticsComputationError
from synthetic_bit_sequence_majority_rule.domain.params import StatisticsConfig
from synthetic_bit_sequence_majority_rule.domain.schema import MajorityMatricesResult


def make_majority_result() -> MajorityMatricesResult:
    classes = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    b_reduced = np.array(
        [
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [0, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ]
    )
    binary_sequences = ["".join(str(x) for x in row) for row in b_reduced]
    decimal_values = np.array([int(seq, 2) for seq in binary_sequences])
    return MajorityMatricesResult(
        metric_name="chebyshev",
        object_labels=[f"S{i}" for i in range(1, 11)],
        classes=classes,
        same_class_indicators=np.ones((10, 9), dtype=int),
        a_full=np.ones((10, 5), dtype=int),
        a_reduced=np.ones((10, 2), dtype=int),
        b_full=np.ones((10, 5), dtype=int),
        b_reduced=b_reduced,
        full_k_values=[1, 2, 3, 4, 5],
        reduced_k_values=[3, 5],
        binary_sequences=binary_sequences,
        decimal_values=decimal_values,
    )


def make_long_majority_result() -> MajorityMatricesResult:
    classes = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    b_reduced = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
        ]
    )
    binary_sequences = ["".join(str(x) for x in row) for row in b_reduced]
    decimal_values = np.array([int(seq, 2) for seq in binary_sequences])
    return MajorityMatricesResult(
        metric_name="euclidean",
        object_labels=[f"S{i}" for i in range(1, 11)],
        classes=classes,
        same_class_indicators=np.ones((10, 9), dtype=int),
        a_full=np.ones((10, 9), dtype=int),
        a_reduced=np.ones((10, 4), dtype=int),
        b_full=np.ones((10, 9), dtype=int),
        b_reduced=b_reduced,
        full_k_values=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        reduced_k_values=[3, 5, 7, 9],
        binary_sequences=binary_sequences,
        decimal_values=decimal_values,
    )


def test_sequence_statistics_rank_by_purity_first() -> None:
    stats = build_sequence_statistics(make_majority_result(), StatisticsConfig())

    assert stats.rows[0].binary_sequence == "01"
    assert stats.rows[0].winner_class == 2
    assert stats.rows[0].purity == 1.0
    assert stats.rows[1].binary_sequence == "11"
    assert stats.rows[1].frequency == 9


def test_membership_and_stability_for_prefixes() -> None:
    membership = build_membership_table(make_majority_result())
    frame = membership.to_frame()

    b3_value_1 = frame[(frame["Representation"] == "b3") & (frame["Decimal"] == 1)].iloc[0]
    assert b3_value_1["CountK1"] == 6
    assert b3_value_1["CountK2"] == 3
    assert round(float(b3_value_1["Membership"]), 6) == 0.571429

    stability = build_stability_table(membership, object_count=10)
    stability_frame = stability.to_frame()
    assert set(stability_frame["Representation"]) == {"b3", "b3, b5"}
    assert set(stability_frame["KValues"]) == {"3", "3, 5"}
    assert round(float(stability_frame.iloc[0]["Stability"]), 6) == 0.614286
    assert stability_interpretation(0.5) == "Not distinguishable"
    assert stability_interpretation(0.55) == "Poor stability"
    assert stability_interpretation(0.7) == "Satisfactory"
    assert stability_interpretation(0.9) == "High"


def test_final_comparison_uses_top_ranked_row() -> None:
    stats = build_sequence_statistics(make_majority_result(), StatisticsConfig())
    comparison = build_final_comparison({"chebyshev": stats})
    frame = comparison.to_frame()

    assert frame.loc[0, "Metric"] == "chebyshev"
    assert frame.loc[0, "BinarySequence"] == "01"
    assert frame.loc[0, "WinnerClass"] == 2


def test_membership_and_stability_use_compact_prefix_display_labels() -> None:
    membership = build_membership_table(make_long_majority_result())
    membership_frame = membership.to_frame()

    assert list(membership_frame["Representation"].drop_duplicates()) == [
        "b3",
        "b3, b5",
        "b3, ..., b7",
        "b3, ..., b9",
    ]
    assert list(membership_frame["KValues"].drop_duplicates()) == [
        "3",
        "3, 5",
        "3, ..., 7",
        "3, ..., 9",
    ]

    stability = build_stability_table(membership, object_count=10)
    stability_frame = stability.to_frame()
    assert stability_frame["Representation"].tolist() == [
        "b3",
        "b3, b5",
        "b3, ..., b7",
        "b3, ..., b9",
    ]
    assert stability_frame["KValues"].tolist() == [
        "3",
        "3, 5",
        "3, ..., 7",
        "3, ..., 9",
    ]


def _reference_membership_rows(result: MajorityMatricesResult) -> list[tuple]:
    """The defining rule: group objects by each prefix string, rows in ascending decimal."""
    total_k1 = int((result.classes == 1).sum())
    total_k2 = int((result.classes == 2).sum())
    rows = []
    for width in range(1, len(result.reduced_k_values) + 1):
        grouped: dict[str, list[int]] = {}
        for bits, cls in zip(result.b_reduced.astype(int), result.classes.astype(int)):
            counts = grouped.setdefault("".join(str(int(bit)) for bit in bits[:width]), [0, 0])
            counts[0 if cls == 1 else 1] += 1
        for sequence in sorted(grouped, key=lambda item: (int(item, 2), item)):
            n1, n2 = grouped[sequence]
            f1, f2 = n1 / total_k1, n2 / total_k2
            membership = 0.0 if f1 + f2 == 0.0 else f1 / (f1 + f2)
            rows.append((width, sequence, int(sequence, 2), n1, n2, n1 + n2, membership))
    return rows


def test_membership_table_matches_the_reference_prefix_grouping() -> None:
    rng = np.random.default_rng(3)
    n, width = 80, 11
    classes = np.array([1, 2] + rng.integers(1, 3, size=n - 2).tolist())
    # Correlated bits give a realistic mix of shared and unique prefixes.
    b_reduced = (rng.random((n, width)) < np.linspace(0.2, 0.8, width)).astype(int)
    b_reduced[: n // 4] = b_reduced[0]
    sequences = ["".join(map(str, row)) for row in b_reduced]
    result = MajorityMatricesResult(
        metric_name="canberra",
        object_labels=[f"S{i}" for i in range(1, n + 1)],
        classes=classes,
        same_class_indicators=np.ones((n, n - 1), dtype=int),
        a_full=np.ones((n, 2 * width + 1), dtype=int),
        a_reduced=np.ones((n, width), dtype=int),
        b_full=np.ones((n, 2 * width + 1), dtype=int),
        b_reduced=b_reduced,
        full_k_values=list(range(1, 2 * width + 2)),
        reduced_k_values=list(range(3, 2 * width + 2, 2)),
        binary_sequences=sequences,
        decimal_values=np.array([int(seq, 2) for seq in sequences]),
    )

    membership = build_membership_table(result)

    actual = [
        (
            len(row.k_values),
            row.binary_sequence,
            row.decimal,
            row.count_k1,
            row.count_k2,
            row.frequency,
            row.membership,
        )
        for row in membership.rows
    ]
    # Exact equality, floats included: the arithmetic must be unchanged.
    assert actual == _reference_membership_rows(result)
    assert all(type(row.count_k1) is int and type(row.decimal) is int for row in membership.rows)


def _stability_result(metric: str, values: list[float]) -> StabilityTableResult:
    return StabilityTableResult(
        metric_name=metric,
        rows=[
            StabilityRow(
                metric_name=metric,
                representation=f"prefix_{index}",
                k_values=[3 + 2 * index],
                unique_values=2,
                object_count=10,
                stability=value,
                interpretation="",
            )
            for index, value in enumerate(values)
        ],
    )


def test_complexity_formula_and_boundary_interpretations() -> None:
    result = build_complexity_table(
        {
            "minimal": _stability_result("minimal", [1.0, 1.0]),
            "simple": _stability_result("simple", [0.5, 1.0]),
            "difficult": _stability_result("difficult", [0.5, 0.7]),
            "maximal": _stability_result("maximal", [0.5, 0.5]),
        }
    ).to_frame().set_index("Metric")

    assert result.loc["minimal", "Complexity"] == 0.0
    assert result.loc["minimal", "Interpretation"].startswith("Minimal complexity")
    assert result.loc["simple", "Complexity"] == 0.25
    assert result.loc["simple", "Interpretation"] == "Simple classification task"
    assert result.loc["difficult", "Complexity"] == pytest.approx(0.4)
    assert result.loc["difficult", "Interpretation"] == "Difficult classification task"
    assert result.loc["maximal", "Complexity"] == 0.5
    assert result.loc["maximal", "Interpretation"].startswith("Maximal ambiguity")


def test_complexity_rejects_stability_outside_theoretical_interval() -> None:
    with pytest.raises(StatisticsComputationError, match="theoretical interval"):
        build_complexity_table({"invalid": _stability_result("invalid", [0.49, 0.8])})

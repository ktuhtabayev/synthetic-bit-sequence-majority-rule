from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from synthetic_bit_sequence_majority_rule.algorithms.majority import bit_strings
from synthetic_bit_sequence_majority_rule.domain.errors import StatisticsComputationError
from synthetic_bit_sequence_majority_rule.domain.params import StatisticsConfig
from synthetic_bit_sequence_majority_rule.domain.schema import (
    BinarySequenceStatsRow,
    FinalComparisonResult,
    FinalComparisonRow,
    MajorityMatricesResult,
    StatisticsTableResult,
)


@dataclass(slots=True)
class MembershipRow:
    metric_name: str
    representation: str
    # One tuple shared by every row of a representation; a table holds many rows
    # per representation, so per-row copies would dominate its memory.
    k_values: tuple[int, ...]
    binary_sequence: str
    decimal: int
    count_k1: int
    count_k2: int
    frequency: int
    membership: float


@dataclass(slots=True)
class MembershipTableResult:
    metric_name: str
    rows: list[MembershipRow]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Metric": row.metric_name,
                    "Representation": format_representation(row.k_values),
                    "KValues": format_k_values(row.k_values),
                    "BinarySequence": row.binary_sequence,
                    "Decimal": row.decimal,
                    "CountK1": row.count_k1,
                    "CountK2": row.count_k2,
                    "Frequency": row.frequency,
                    "Membership": row.membership,
                }
                for row in self.rows
            ]
        )


@dataclass(slots=True)
class StabilityRow:
    metric_name: str
    representation: str
    k_values: list[int]
    unique_values: int
    object_count: int
    stability: float
    interpretation: str


@dataclass(slots=True)
class StabilityTableResult:
    metric_name: str
    rows: list[StabilityRow]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Metric": row.metric_name,
                    "Representation": format_representation(row.k_values),
                    "KValues": format_k_values(row.k_values),
                    "UniqueValues": row.unique_values,
                    "ObjectCount": row.object_count,
                    "Stability": row.stability,
                    "Interpretation": row.interpretation,
                }
                for row in self.rows
            ]
        )


@dataclass(slots=True)
class ComplexityRow:
    metric_name: str
    component_count: int
    mean_stability: float
    complexity: float
    interpretation: str


@dataclass(slots=True)
class ComplexityTableResult:
    rows: list[ComplexityRow]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Metric": row.metric_name,
                    "ComponentCount": row.component_count,
                    "MeanStability": row.mean_stability,
                    "Complexity": row.complexity,
                    "Interpretation": row.interpretation,
                }
                for row in self.rows
            ]
        )


# Long prefixes show only their ends, so these read just the first and last k
# (they run once per membership and stability row).
def format_representation(k_values: Sequence[int]) -> str:
    if not k_values:
        return ""
    if len(k_values) <= 2:
        return ", ".join(f"b{k}" for k in k_values)
    return f"b{k_values[0]}, ..., b{k_values[-1]}"


def format_k_values(k_values: Sequence[int]) -> str:
    if not k_values:
        return ""
    if len(k_values) <= 2:
        return ", ".join(str(k) for k in k_values)
    return f"{k_values[0]}, ..., {k_values[-1]}"


def _ensure_majority_result(result: MajorityMatricesResult) -> MajorityMatricesResult:
    if not isinstance(result, MajorityMatricesResult):
        raise TypeError("Expected a MajorityMatricesResult instance.")
    return result


def _class_counts(classes: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(classes.astype(int), return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def _dominant_class(count_k1: int, count_k2: int, config: StatisticsConfig) -> int:
    if count_k1 > count_k2:
        return 1
    if count_k2 > count_k1:
        return 2
    if config.dominant_class_rule == "tie_zero":
        return 0
    return 1


def _purity(count_k1: int, count_k2: int) -> float:
    total = count_k1 + count_k2
    if total == 0:
        return 0.0
    return max(count_k1, count_k2) / total


def _status(count_k1: int, count_k2: int) -> str:
    if count_k1 > 0 and count_k2 > 0:
        return "Mixed"
    if count_k1 == 0 and count_k2 == 0:
        return "Empty"
    return "Pure"


def _sort_stats_rows(
    rows: list[BinarySequenceStatsRow],
    tie_rules: list[str],
) -> list[BinarySequenceStatsRow]:
    def key(row: BinarySequenceStatsRow) -> tuple[Any, ...]:
        parts: list[Any] = []
        for rule in tie_rules:
            if rule == "higher_purity":
                parts.append(-row.purity)
            elif rule == "higher_frequency":
                parts.append(-row.frequency)
            elif rule == "smaller_decimal":
                parts.append(row.decimal)
            elif rule == "smaller_binary_sequence":
                parts.append(row.binary_sequence)
        return tuple(parts)

    return sorted(rows, key=key)


def build_sequence_statistics(
    result: MajorityMatricesResult,
    statistics_config: StatisticsConfig,
) -> StatisticsTableResult:
    """
    Build distribution statistics for reduced binary sequences.

    Each unique reduced bit chain is counted by class K1/K2, converted to
    decimal, assigned a dominant class, and ranked by the configured tie rule.
    """
    result = _ensure_majority_result(result)

    try:
        grouped: dict[str, dict[str, int]] = {}
        decimal_by_sequence: dict[str, int] = {}

        for sequence, decimal, cls in zip(
            result.binary_sequences,
            [int(value) for value in result.decimal_values.tolist()],
            result.classes.astype(int).tolist(),
        ):
            if sequence not in grouped:
                grouped[sequence] = {"count_k1": 0, "count_k2": 0}
                decimal_by_sequence[sequence] = int(decimal)

            if cls == 1:
                grouped[sequence]["count_k1"] += 1
            elif cls == 2:
                grouped[sequence]["count_k2"] += 1
            else:
                raise StatisticsComputationError(
                    metric=result.metric_name,
                    reason=f"Unsupported class label {cls}.",
                )

        rows: list[BinarySequenceStatsRow] = []
        for sequence, counts in grouped.items():
            count_k1 = counts["count_k1"]
            count_k2 = counts["count_k2"]
            frequency = count_k1 + count_k2
            rows.append(
                BinarySequenceStatsRow(
                    metric_name=result.metric_name,
                    binary_sequence=sequence,
                    decimal=decimal_by_sequence[sequence],
                    count_k1=count_k1,
                    count_k2=count_k2,
                    frequency=frequency,
                    winner_class=_dominant_class(count_k1, count_k2, statistics_config),
                    purity=_purity(count_k1, count_k2),
                    status=_status(count_k1, count_k2),
                    winner_rank=None,
                )
            )

        ranked = _sort_stats_rows(rows, statistics_config.tie_rule.rules)
        for idx, row in enumerate(ranked, start=1):
            row.winner_rank = idx

        return StatisticsTableResult(
            metric_name=result.metric_name,
            rows=ranked,
            normalization_mode=str(result.metadata.get("normalization_mode", "none")),
            with_normalization=bool(result.metadata.get("with_normalization", False)),
            metadata={
                "source_metric": result.metric_name,
                "reduced_k_values": list(result.reduced_k_values),
                "tie_rule": list(statistics_config.tie_rule.rules),
            },
        )
    except StatisticsComputationError:
        raise
    except Exception as exc:
        raise StatisticsComputationError(metric=result.metric_name, reason=str(exc)) from exc


def build_multiple_sequence_statistics(
    majority_results: dict[str, MajorityMatricesResult],
    statistics_config: StatisticsConfig,
) -> dict[str, StatisticsTableResult]:
    return {
        metric_name: build_sequence_statistics(result, statistics_config)
        for metric_name, result in majority_results.items()
    }


def _membership_value(
    count_k1: int,
    count_k2: int,
    total_k1: int,
    total_k2: int,
) -> float:
    """
    Membership of value mu to K1:

        f(mu) = (n1(mu)/|K1|) / ((n1(mu)/|K1|) + (n2(mu)/|K2|))

    This matches the experiment spreadsheet formula. When every object
    shares one value, f(mu)=0.5 because both classes are represented at
    their base rates.
    """
    if total_k1 <= 0 or total_k2 <= 0:
        raise StatisticsComputationError(reason="Class totals must be positive.")

    normalized_k1 = count_k1 / total_k1
    normalized_k2 = count_k2 / total_k2
    denominator = normalized_k1 + normalized_k2
    if denominator == 0.0:
        return 0.0
    return normalized_k1 / denominator


def build_membership_table(result: MajorityMatricesResult) -> MembershipTableResult:
    """
    Build membership rows for every reduced-prefix representation.

    For reduced k values [3,5,7], this creates representations b3, b3_b5,
    and b3_b5_b7. The final one corresponds to the full reduced sequence.
    """
    result = _ensure_majority_result(result)
    counts = _class_counts(result.classes)
    total_k1 = counts.get(1, 0)
    total_k2 = counts.get(2, 0)
    rows: list[MembershipRow] = []

    try:
        bits = result.b_reduced.astype(np.int64)
        in_k1 = result.classes.astype(int) == 1
        # An object's sequence for the first `width` reduced k values is a prefix
        # of its full sequence.
        full_sequences = bit_strings(bits)

        # Objects sharing a prefix are one group. Groups are numbered in ascending
        # order of their prefix, so appending a bit as rank * 2 + bit keeps the
        # numbering in ascending order of the longer prefix: ascending decimal,
        # which for equal-length bit strings is also ascending string order.
        rank = np.zeros(len(full_sequences), dtype=np.int64)
        for width in range(1, len(result.reduced_k_values) + 1):
            k_values = tuple(result.reduced_k_values[:width])
            representation = "_".join(f"b{k}" for k in k_values)

            _, first_member, rank = np.unique(
                rank * 2 + bits[:, width - 1],
                return_index=True,
                return_inverse=True,
            )
            group_count = len(first_member)
            counts_k1 = np.bincount(rank[in_k1], minlength=group_count).tolist()
            counts_k2 = np.bincount(rank[~in_k1], minlength=group_count).tolist()

            for group, member in enumerate(first_member.tolist()):
                sequence = full_sequences[member][:width]
                count_k1 = counts_k1[group]
                count_k2 = counts_k2[group]
                rows.append(
                    MembershipRow(
                        metric_name=result.metric_name,
                        representation=representation,
                        k_values=k_values,
                        binary_sequence=sequence,
                        decimal=int(sequence, 2) if sequence else 0,
                        count_k1=count_k1,
                        count_k2=count_k2,
                        frequency=count_k1 + count_k2,
                        membership=_membership_value(
                            count_k1=count_k1,
                            count_k2=count_k2,
                            total_k1=total_k1,
                            total_k2=total_k2,
                        ),
                    )
                )

        return MembershipTableResult(
            metric_name=result.metric_name,
            rows=rows,
            metadata={
                "class_counts": counts,
                "reduced_k_values": list(result.reduced_k_values),
                "membership_formula": "(n1(mu)/|K1|) / ((n1(mu)/|K1|) + (n2(mu)/|K2|))",
            },
        )
    except StatisticsComputationError:
        raise
    except Exception as exc:
        raise StatisticsComputationError(metric=result.metric_name, reason=str(exc)) from exc


def stability_interpretation(value: float) -> str:
    if value <= 0.5:
        return "Not distinguishable"
    if value <= 0.6:
        return "Poor stability"
    if value <= 0.8:
        return "Satisfactory"
    return "High"


def build_stability_table(
    membership_result: MembershipTableResult,
    object_count: int,
) -> StabilityTableResult:
    rows: list[StabilityRow] = []

    if not membership_result.rows:
        return StabilityTableResult(metric_name=membership_result.metric_name, rows=rows)

    grouped: dict[tuple[int, ...], list[MembershipRow]] = {}
    for item in membership_result.rows:
        grouped.setdefault(tuple(item.k_values), []).append(item)

    for k_values_key, group in grouped.items():
        stability = 0.0
        for item in group:
            membership = float(item.membership)
            frequency = int(item.frequency)
            if membership >= 0.5:
                stability += frequency * membership
            else:
                stability += frequency * (1.0 - membership)
        stability = stability / object_count if object_count else 0.0

        first = group[0]
        k_values = list(k_values_key)
        rows.append(
            StabilityRow(
                metric_name=membership_result.metric_name,
                representation=first.representation,
                k_values=k_values,
                unique_values=len(group),
                object_count=int(object_count),
                stability=float(stability),
                interpretation=stability_interpretation(float(stability)),
            )
        )

    return StabilityTableResult(
        metric_name=membership_result.metric_name,
        rows=rows,
        metadata=dict(membership_result.metadata),
    )


def complexity_interpretation(value: float, *, tolerance: float = 1e-12) -> str:
    if value < -tolerance or value > 0.5 + tolerance:
        raise StatisticsComputationError(
            reason=f"Complexity must be in [0, 0.5], got {value}.",
        )
    if abs(value) <= tolerance:
        return "Minimal complexity (completely stable separation)"
    if value <= 0.25 + tolerance:
        return "Simple classification task"
    if value < 0.5 - tolerance:
        return "Difficult classification task"
    return "Maximal ambiguity (no stable separation)"


def build_complexity_table(
    stability_results: dict[str, StabilityTableResult],
    *,
    tolerance: float = 1e-12,
) -> ComplexityTableResult:
    rows: list[ComplexityRow] = []

    for metric_name, result in stability_results.items():
        if not result.rows:
            continue

        values = np.asarray([row.stability for row in result.rows], dtype=float)
        if not np.isfinite(values).all():
            raise StatisticsComputationError(
                metric=metric_name,
                reason="Stability values must be finite.",
            )
        if np.any(values < 0.5 - tolerance) or np.any(values > 1.0 + tolerance):
            raise StatisticsComputationError(
                metric=metric_name,
                reason="Stability values must be in the theoretical interval [0.5, 1].",
            )

        values = np.clip(values, 0.5, 1.0)
        mean_stability = float(np.mean(values))
        complexity = float(np.clip(1.0 - mean_stability, 0.0, 0.5))
        rows.append(
            ComplexityRow(
                metric_name=metric_name,
                component_count=len(values),
                mean_stability=mean_stability,
                complexity=complexity,
                interpretation=complexity_interpretation(
                    complexity,
                    tolerance=tolerance,
                ),
            )
        )

    return ComplexityTableResult(
        rows=rows,
        metadata={
            "formula": "C(Q) = 1 - (1/t) * sum_i(g_i)",
            "component_symbol": "t",
            "stability_interval": [0.5, 1.0],
            "complexity_interval": [0.0, 0.5],
        },
    )


def build_required_result_tables(
    result: MajorityMatricesResult,
    statistics_config: StatisticsConfig,
) -> tuple[StatisticsTableResult, MembershipTableResult, StabilityTableResult]:
    statistics = build_sequence_statistics(result, statistics_config)
    membership = build_membership_table(result)
    stability = build_stability_table(membership, object_count=len(result.object_labels))
    return statistics, membership, stability


def build_final_comparison(
    statistics_results: dict[str, StatisticsTableResult],
) -> FinalComparisonResult:
    rows: list[FinalComparisonRow] = []

    for metric_name, result in statistics_results.items():
        if not result.rows:
            continue
        winner = min(
            result.rows,
            key=lambda row: row.winner_rank if row.winner_rank is not None else 10**9,
        )
        rows.append(
            FinalComparisonRow(
                metric_name=metric_name,
                winner_class=winner.winner_class,
                binary_sequence=winner.binary_sequence,
                decimal=winner.decimal,
                frequency=winner.frequency,
                purity=winner.purity,
            )
        )

    rows.sort(key=lambda row: row.metric_name)
    return FinalComparisonResult(
        rows=rows,
        metadata={
            "meaning": "Best binary sequence per metric according to the configured statistics tie rule.",
        },
    )

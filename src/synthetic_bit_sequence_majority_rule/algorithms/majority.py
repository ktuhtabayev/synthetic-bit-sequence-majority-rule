from __future__ import annotations


import numpy as np
import pandas as pd

from synthetic_bit_sequence_majority_rule.domain.errors import MajorityRuleError
from synthetic_bit_sequence_majority_rule.domain.params import (
    BinarySequenceConfig,
    DecimalEncodingConfig,
    KValuesConfig,
    MajorityRuleConfig,
)
from synthetic_bit_sequence_majority_rule.domain.schema import (
    LoadedDataset,
    MajorityMatricesResult,
    NeighborTableResult,
    dataset_normalized_flag,
    ensure_loaded_dataset,
)


# ============================================================
# Small helpers
# ============================================================

_ensure_dataset = ensure_loaded_dataset
_normalized_flag_from_dataset = dataset_normalized_flag


def _ensure_neighbor_result(result: NeighborTableResult) -> NeighborTableResult:
    if not isinstance(result, NeighborTableResult):
        raise TypeError("Expected a NeighborTableResult instance.")
    return result


def _validate_alignment(dataset: LoadedDataset, neighbor_result: NeighborTableResult) -> None:
    if dataset.object_labels != neighbor_result.object_labels:
        raise MajorityRuleError(
            metric=neighbor_result.metric_name,
            reason="Object label order mismatch between dataset and neighbor result.",
        )

    expected_neighbor_count = dataset.n_objects - 1
    if neighbor_result.neighbor_count != expected_neighbor_count:
        raise MajorityRuleError(
            metric=neighbor_result.metric_name,
            reason=(
                f"Neighbor count mismatch. Expected {expected_neighbor_count}, "
                f"got {neighbor_result.neighbor_count}."
            ),
        )


# ============================================================
# Dynamic k-rule resolution
# ============================================================

def compute_formula_based_kmax(dataset: LoadedDataset) -> int:
    """
    New rule:

        k_max = min(m - 1, 2 * min_i |K_i| - 3)

    where:
        m = number of objects
        |K_i| = class counts
    """
    dataset = _ensure_dataset(dataset)

    if not dataset.class_counts:
        raise MajorityRuleError(
            metric="unknown",
            reason="Dataset class counts are missing.",
        )

    min_class_size = min(int(v) for v in dataset.class_counts.values())
    formula_kmax = 2 * min_class_size - 3
    physical_kmax = dataset.n_objects - 1
    kmax = min(physical_kmax, formula_kmax)

    if kmax < 1:
        raise MajorityRuleError(
            metric="unknown",
            reason=(
                f"k_max became invalid ({kmax}). "
                f"Computed from class counts {dataset.class_counts}."
            ),
        )

    return kmax


def compute_full_k_values(
    dataset: LoadedDataset,
    k_values_config: KValuesConfig,
) -> list[int]:
    """
    Current rule from config:
        full.start must be 1
        full.end must be 'auto_formula'

    Final full list:
        [1, 2, ..., k_max]
    """
    if k_values_config.full.start != 1:
        raise MajorityRuleError(
            metric="unknown",
            reason=f"k_values.full.start must currently be 1, got {k_values_config.full.start}.",
        )

    if k_values_config.full.end != "auto_formula":
        raise MajorityRuleError(
            metric="unknown",
            reason="k_values.full.end must currently be 'auto_formula'.",
        )

    kmax = compute_formula_based_kmax(dataset)
    return list(range(1, kmax + 1))


def compute_reduced_k_values(
    full_k_values: list[int],
    k_values_config: KValuesConfig,
    binary_sequence_config: BinarySequenceConfig,
) -> list[int]:
    """
    Current rule from config:
        k_values.reduced.mode = odd_from_3
        binary_sequence.reduced_order_mode = follow_reduced_k_values

    Final reduced list:
        odd values from 3 to k_max
    """
    if k_values_config.reduced.mode != "odd_from_3":
        raise MajorityRuleError(
            metric="unknown",
            reason="k_values.reduced.mode must currently be 'odd_from_3'.",
        )

    if not binary_sequence_config.use_reduced_k_values:
        raise MajorityRuleError(
            metric="unknown",
            reason="binary_sequence.use_reduced_k_values must currently be true.",
        )

    if binary_sequence_config.reduced_order_mode != "follow_reduced_k_values":
        raise MajorityRuleError(
            metric="unknown",
            reason="binary_sequence.reduced_order_mode must currently be 'follow_reduced_k_values'.",
        )

    reduced = [k for k in full_k_values if k >= 3 and (k % 2 == 1)]

    if not reduced:
        raise MajorityRuleError(
            metric="unknown",
            reason=(
                "Reduced k values are empty. "
                "This usually means formula-based k_max < 3 for the current dataset."
            ),
        )

    return reduced


# ============================================================
# Same-class indicators
# ============================================================

def _same_class_indicator_matrix(
    dataset: LoadedDataset,
    neighbor_result: NeighborTableResult,
) -> np.ndarray:
    """
    Build same-class indicator matrix:
    1 if neighbor class == target class, else 0

    This part remains valid under the new rule.
    """
    object_to_index = {label: idx for idx, label in enumerate(dataset.object_labels)}

    try:
        neighbor_indices = np.asarray(
            [[object_to_index[label] for label in row] for row in neighbor_result.neighbor_labels],
            dtype=np.intp,
        ).reshape(dataset.n_objects, dataset.n_objects - 1)
    except KeyError as exc:
        raise MajorityRuleError(
            metric=neighbor_result.metric_name,
            reason=f"Unknown neighbor label '{exc.args[0]}'.",
        ) from exc

    classes = dataset.y.astype(int)
    return (classes[neighbor_indices] == classes[:, np.newaxis]).astype(int)


# ============================================================
# A(S) and B(S)
# ============================================================

def _build_a_all(same_class_indicators: np.ndarray) -> np.ndarray:
    """
    Build cumulative a-values for all possible neighbor ranks:
        a1, a2, ..., a_(m-1)
    """
    return np.cumsum(same_class_indicators, axis=1)


def _select_k_columns(a_all: np.ndarray, k_values: list[int]) -> np.ndarray:
    """
    Select columns corresponding to requested k values.
    Since column indices are zero-based but k starts from 1,
    the selected column index is (k - 1).
    """
    col_indices = [k - 1 for k in k_values]
    return a_all[:, col_indices]


def _majority_binary_from_a(
    a_matrix: np.ndarray,
    k_values: list[int],
    majority_rule_config: MajorityRuleConfig,
) -> np.ndarray:
    """
    Build binary majority matrix b(S) from A(S).
    """
    if a_matrix.shape[1] != len(k_values):
        raise ValueError("a_matrix column count must match k_values length.")

    k_array = np.asarray(k_values, dtype=float).reshape(1, -1)
    ratios = a_matrix / k_array

    if majority_rule_config.comparison == "strict_greater":
        return (ratios > majority_rule_config.threshold).astype(int)

    if majority_rule_config.comparison == "greater_or_equal":
        return (ratios >= majority_rule_config.threshold).astype(int)

    raise MajorityRuleError(
        metric="unknown",
        reason=f"Unsupported majority comparison '{majority_rule_config.comparison}'.",
    )


def bit_strings(bits: np.ndarray) -> list[str]:
    """
    One '0'/'1' string per row of a 0/1 matrix, in column order.

    A prefix of a row's string is the bit string of that row's first columns,
    so callers that need every prefix width slice these instead of rebuilding.
    """
    matrix = np.asarray(bits).astype(np.uint8, copy=False)
    if matrix.ndim != 2:
        raise ValueError("bits must be a 2D array.")
    if matrix.size and matrix.max() > 1:
        raise ValueError("bits must contain only 0/1 values.")
    if matrix.shape[1] == 0:
        return [""] * matrix.shape[0]
    # ASCII '0' is 48, so bit + 48 is its character; each row's bytes form one string.
    chars = np.ascontiguousarray(matrix + ord("0"))
    return [row.decode("ascii") for row in chars.view(f"S{matrix.shape[1]}").ravel()]


def _binary_sequences_from_b_reduced(b_reduced: np.ndarray) -> list[str]:
    return bit_strings(b_reduced)


def _binary_to_decimal(sequence: str, bit_order: str) -> int:
    if not sequence:
        return 0

    if any(ch not in {"0", "1"} for ch in sequence):
        raise ValueError(f"Invalid binary sequence: {sequence}")

    if bit_order == "left_to_right":
        return int(sequence, 2)

    if bit_order == "right_to_left":
        return int(sequence[::-1], 2)

    raise ValueError(f"Unsupported bit order: {bit_order}")


def _decimal_values_from_sequences(
    binary_sequences: list[str],
    decimal_encoding_config: DecimalEncodingConfig,
) -> np.ndarray:
    if not decimal_encoding_config.enabled:
        return np.asarray([0] * len(binary_sequences), dtype=object)

    return np.asarray(
        [
            _binary_to_decimal(seq, decimal_encoding_config.bit_order)
            for seq in binary_sequences
        ],
        dtype=object,
    )


# ============================================================
# Public API
# ============================================================

def build_majority_matrices(
    dataset: LoadedDataset,
    neighbor_result: NeighborTableResult,
    k_values_config: KValuesConfig,
    majority_rule_config: MajorityRuleConfig,
    binary_sequence_config: BinarySequenceConfig,
    decimal_encoding_config: DecimalEncodingConfig,
) -> MajorityMatricesResult:
    """
    Build majority-rule outputs for one metric under the new dynamic rule.

    New rule:
        full    = 1..k_max
        reduced = odd values from 3 to k_max
    """
    dataset = _ensure_dataset(dataset)
    neighbor_result = _ensure_neighbor_result(neighbor_result)

    _validate_alignment(dataset, neighbor_result)

    metric_name = neighbor_result.metric_name
    full_k_values = compute_full_k_values(dataset, k_values_config)
    reduced_k_values = compute_reduced_k_values(
        full_k_values,
        k_values_config,
        binary_sequence_config,
    )
    formula_kmax = full_k_values[-1]

    try:
        same_class_indicators = _same_class_indicator_matrix(dataset, neighbor_result)
        a_all = _build_a_all(same_class_indicators)

        # IMPORTANT:
        # A FULL is now only k = 1..k_max
        a_full = _select_k_columns(a_all, full_k_values)

        # IMPORTANT:
        # A REDUCED is odd values from 3 to k_max
        a_reduced = _select_k_columns(a_all, reduced_k_values)

        # IMPORTANT:
        # B FULL and B REDUCED must follow the same selected k-columns
        b_full = _majority_binary_from_a(a_full, full_k_values, majority_rule_config)
        b_reduced = _majority_binary_from_a(a_reduced, reduced_k_values, majority_rule_config)

        binary_sequences = _binary_sequences_from_b_reduced(b_reduced)
        decimal_values = _decimal_values_from_sequences(binary_sequences, decimal_encoding_config)

    except MajorityRuleError:
        raise
    except Exception as exc:
        raise MajorityRuleError(metric=metric_name, reason=str(exc)) from exc

    return MajorityMatricesResult(
        metric_name=metric_name,
        object_labels=list(dataset.object_labels),
        classes=dataset.y.copy(),
        same_class_indicators=same_class_indicators,
        a_full=a_full,
        a_reduced=a_reduced,
        b_full=b_full,
        b_reduced=b_reduced,
        full_k_values=full_k_values,
        reduced_k_values=reduced_k_values,
        binary_sequences=binary_sequences,
        decimal_values=decimal_values,
        metadata={
            "source_metric": metric_name,
            "normalization_mode": dataset.metadata.get("normalization_mode", "none"),
            "with_normalization": _normalized_flag_from_dataset(dataset),
            "majority_threshold": majority_rule_config.threshold,
            "majority_comparison": majority_rule_config.comparison,
            "bit_order": decimal_encoding_config.bit_order,
            "formula_kmax": formula_kmax,
            "class_counts": dict(dataset.class_counts),
            "reduced_rule": "odd_from_3",
        },
    )


def build_multiple_majority_matrices(
    dataset: LoadedDataset,
    neighbor_results: dict[str, NeighborTableResult],
    k_values_config: KValuesConfig,
    majority_rule_config: MajorityRuleConfig,
    binary_sequence_config: BinarySequenceConfig,
    decimal_encoding_config: DecimalEncodingConfig,
) -> dict[str, MajorityMatricesResult]:
    if not neighbor_results:
        raise ValueError("neighbor_results must not be empty.")

    results: dict[str, MajorityMatricesResult] = {}
    for metric_name, neighbor_result in neighbor_results.items():
        results[metric_name] = build_majority_matrices(
            dataset=dataset,
            neighbor_result=neighbor_result,
            k_values_config=k_values_config,
            majority_rule_config=majority_rule_config,
            binary_sequence_config=binary_sequence_config,
            decimal_encoding_config=decimal_encoding_config,
        )
    return results


# ============================================================
# Summary / conversion helpers
# ============================================================

def same_class_indicator_frame(result: MajorityMatricesResult) -> pd.DataFrame:
    """
    Full same-class indicator frame always has m-1 columns:
        s1, s2, ..., s_(m-1)

    This stage is upstream of the formula-based k truncation.
    """
    full_neighbor_count = result.same_class_indicators.shape[1]
    columns = [f"s{k}" for k in range(1, full_neighbor_count + 1)]
    df = pd.DataFrame(result.same_class_indicators.astype(int), columns=columns)
    df.insert(0, "Class", result.classes.astype(int))
    df.insert(0, "Object", result.object_labels)
    return df


def a_full_frame(result: MajorityMatricesResult) -> pd.DataFrame:
    return result.a_full_frame().reset_index()


def a_reduced_frame(result: MajorityMatricesResult) -> pd.DataFrame:
    return result.a_reduced_frame().reset_index()


def b_full_frame(result: MajorityMatricesResult) -> pd.DataFrame:
    return result.b_full_frame().reset_index()


def b_reduced_frame(result: MajorityMatricesResult) -> pd.DataFrame:
    return result.b_reduced_frame().reset_index()

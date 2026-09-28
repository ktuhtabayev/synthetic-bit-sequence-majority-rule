from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from synthetic_bit_sequence_majority_rule.algorithms.majority import bit_strings
from synthetic_bit_sequence_majority_rule.algorithms.statistics import format_representation
from synthetic_bit_sequence_majority_rule.domain.schema import MajorityMatricesResult


def build_synthetic_binary_frame(result: MajorityMatricesResult) -> pd.DataFrame:
    data: dict[str, Sequence[object]] = {"Object": list(result.object_labels)}

    # Each representation is a prefix of the full reduced sequence.
    full_sequences = bit_strings(result.b_reduced)
    for width in range(1, len(result.reduced_k_values) + 1):
        column = format_representation(result.reduced_k_values[:width])
        data[column] = [sequence[:width] for sequence in full_sequences]

    data["Class"] = result.classes.astype(int).tolist()
    return pd.DataFrame(data)


def build_synthetic_decimal_frame(result: MajorityMatricesResult) -> pd.DataFrame:
    data: dict[str, Sequence[object]] = {"Object": list(result.object_labels)}

    # Left-to-right decimal of a prefix one bit longer is value * 2 + bit.
    # Python ints keep long sequences exact.
    bits = result.b_reduced.astype(int)
    values = [0] * len(result.object_labels)
    for width in range(1, len(result.reduced_k_values) + 1):
        column = format_representation(result.reduced_k_values[:width])
        values = [value * 2 + bit for value, bit in zip(values, bits[:, width - 1].tolist())]
        data[column] = values

    data["Class"] = result.classes.astype(int).tolist()
    return pd.DataFrame(data)

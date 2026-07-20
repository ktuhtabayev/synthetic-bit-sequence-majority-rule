from __future__ import annotations

import pandas as pd

from synthetic_bit_sequence_majority_rule.algorithms.statistics import format_representation
from synthetic_bit_sequence_majority_rule.domain.schema import MajorityMatricesResult


def build_synthetic_binary_frame(result: MajorityMatricesResult) -> pd.DataFrame:
    data: dict[str, list[object]] = {"Object": list(result.object_labels)}

    for width in range(1, len(result.reduced_k_values) + 1):
        k_values = result.reduced_k_values[:width]
        column = format_representation(k_values)
        data[column] = [
            "".join(str(int(bit)) for bit in row[:width])
            for row in result.b_reduced.astype(int)
        ]

    data["Class"] = result.classes.astype(int).tolist()
    return pd.DataFrame(data)


def build_synthetic_decimal_frame(result: MajorityMatricesResult) -> pd.DataFrame:
    data: dict[str, list[object]] = {"Object": list(result.object_labels)}

    for width in range(1, len(result.reduced_k_values) + 1):
        k_values = result.reduced_k_values[:width]
        column = format_representation(k_values)
        values: list[int] = []
        for row in result.b_reduced.astype(int):
            sequence = "".join(str(int(bit)) for bit in row[:width])
            values.append(int(sequence, 2) if sequence else 0)
        data[column] = values

    data["Class"] = result.classes.astype(int).tolist()
    return pd.DataFrame(data)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd


# ============================================================
# Small validation helpers
# ============================================================

def _ensure_non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")
    return value.strip()


def _ensure_list_of_str(value: Any, name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"All items in {name} must be strings.")
        if not item.strip():
            raise ValueError(f"All items in {name} must be non-empty strings.")
        result.append(item.strip())
    return result


def _ensure_numpy_1d_int(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    try:
        arr = arr.astype(int)
    except Exception as exc:
        raise TypeError(f"{name} must be convertible to int.") from exc
    return arr


def _ensure_numpy_1d_python_int(value: Any, name: str) -> np.ndarray:
    """Return a 1D object array of arbitrary-precision Python integers."""
    arr = np.asarray(value, dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    try:
        values = [int(item) for item in arr.tolist()]
    except Exception as exc:
        raise TypeError(f"{name} must be convertible to int.") from exc
    if any(item < 0 for item in values):
        raise ValueError(f"{name} must contain non-negative integers.")
    return np.asarray(values, dtype=object)


def _ensure_numpy_2d_numeric(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    try:
        arr = arr.astype(float)
    except Exception as exc:
        raise TypeError(f"{name} must be numeric.") from exc
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _ensure_same_length(expected: int, actual: int, left_name: str, right_name: str) -> None:
    if expected != actual:
        raise ValueError(
            f"Length mismatch: {left_name} has length {expected}, "
            f"but {right_name} has length {actual}."
        )


# ============================================================
# Loaded dataset
# ============================================================

@dataclass(slots=True)
class LoadedDataset:
    source_path: str | None
    source_format: str
    X: np.ndarray
    y: np.ndarray
    object_labels: list[str]
    feature_names: list[str]
    class_column: str
    raw_frame: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_format = _ensure_non_empty_str(
            self.source_format,
            "LoadedDataset.source_format",
        ).lower()
        self.X = _ensure_numpy_2d_numeric(self.X, "LoadedDataset.X")
        self.y = _ensure_numpy_1d_int(self.y, "LoadedDataset.y")
        self.object_labels = _ensure_list_of_str(
            self.object_labels,
            "LoadedDataset.object_labels",
        )
        self.feature_names = _ensure_list_of_str(
            self.feature_names,
            "LoadedDataset.feature_names",
        )
        self.class_column = _ensure_non_empty_str(
            self.class_column,
            "LoadedDataset.class_column",
        )

        if self.raw_frame is not None and not isinstance(self.raw_frame, pd.DataFrame):
            raise TypeError("LoadedDataset.raw_frame must be a pandas DataFrame or None.")

        n_objects, n_features = self.X.shape

        _ensure_same_length(n_objects, len(self.y), "X rows", "y")
        _ensure_same_length(n_objects, len(self.object_labels), "X rows", "object_labels")
        _ensure_same_length(n_features, len(self.feature_names), "X columns", "feature_names")

        if len(set(self.object_labels)) != len(self.object_labels):
            raise ValueError("LoadedDataset.object_labels must be unique.")

        unique_classes = set(self.y.tolist())
        if unique_classes != {1, 2}:
            raise ValueError("LoadedDataset.y must contain exactly classes {1, 2}.")

    @property
    def n_objects(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def classes(self) -> list[int]:
        return sorted(set(int(v) for v in self.y.tolist()))

    @property
    def class_counts(self) -> dict[int, int]:
        values, counts = np.unique(self.y, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    @property
    def valid_neighbor_count(self) -> int:
        return self.n_objects - 1

    def to_frame(self) -> pd.DataFrame:
        if self.raw_frame is not None:
            # return self.raw_frame.copy()
            df = self.raw_frame.copy()

            if "Object" not in df.columns:
                df.insert(0, "Object", self.object_labels)

            return df

        df = pd.DataFrame(self.X, columns=self.feature_names)
        df.insert(0, "Object", self.object_labels)
        df[self.class_column] = self.y
        return df

    def copy(self) -> "LoadedDataset":
        return LoadedDataset(
            source_path=self.source_path,
            source_format=self.source_format,
            X=self.X.copy(),
            y=self.y.copy(),
            object_labels=list(self.object_labels),
            feature_names=list(self.feature_names),
            class_column=self.class_column,
            raw_frame=None if self.raw_frame is None else self.raw_frame.copy(),
            metadata=dict(self.metadata),
        )


def ensure_loaded_dataset(dataset: Any) -> LoadedDataset:
    if not isinstance(dataset, LoadedDataset):
        raise TypeError("Expected a LoadedDataset instance.")
    return dataset


def dataset_normalized_flag(dataset: LoadedDataset) -> bool:
    """True when the dataset's metadata records an active normalization mode."""
    mode = dataset.metadata.get("normalization_mode")
    return isinstance(mode, str) and mode.strip().lower() != "none"


# ============================================================
# Normalized dataset
# ============================================================

@dataclass(slots=True)
class NormalizedDataset:
    dataset: LoadedDataset
    mode: str
    X_normalized: np.ndarray
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, LoadedDataset):
            raise TypeError("NormalizedDataset.dataset must be a LoadedDataset.")
        self.mode = _ensure_non_empty_str(self.mode, "NormalizedDataset.mode").lower()
        self.X_normalized = _ensure_numpy_2d_numeric(
            self.X_normalized,
            "NormalizedDataset.X_normalized",
        )

        if self.X_normalized.shape != self.dataset.X.shape:
            raise ValueError("NormalizedDataset.X_normalized shape must match dataset.X shape.")

    @property
    def n_objects(self) -> int:
        return self.dataset.n_objects

    @property
    def n_features(self) -> int:
        return self.dataset.n_features

    @property
    def object_labels(self) -> list[str]:
        return list(self.dataset.object_labels)

    @property
    def feature_names(self) -> list[str]:
        return list(self.dataset.feature_names)

    @property
    def y(self) -> np.ndarray:
        return self.dataset.y.copy()

    @property
    def class_column(self) -> str:
        return self.dataset.class_column

    def to_loaded_dataset(self) -> LoadedDataset:
        metadata = dict(self.dataset.metadata)
        metadata["normalization_mode"] = self.mode
        metadata["normalization_parameters"] = dict(self.parameters)

        return LoadedDataset(
            source_path=self.dataset.source_path,
            source_format=self.dataset.source_format,
            X=self.X_normalized.copy(),
            y=self.dataset.y.copy(),
            object_labels=list(self.dataset.object_labels),
            feature_names=list(self.dataset.feature_names),
            class_column=self.dataset.class_column,
            raw_frame=None,
            metadata=metadata,
        )


# ============================================================
# Distance / neighbors / majority / statistics schemas
# ============================================================

@dataclass(slots=True)
class DistanceMatrixResult:
    """
    Stores one distance matrix for one metric.
    """
    metric_name: str
    matrix: np.ndarray
    object_labels: list[str]
    normalized: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metric_name = _ensure_non_empty_str(
            self.metric_name,
            "DistanceMatrixResult.metric_name",
        ).lower()
        self.matrix = _ensure_numpy_2d_numeric(self.matrix, "DistanceMatrixResult.matrix")
        self.object_labels = _ensure_list_of_str(
            self.object_labels,
            "DistanceMatrixResult.object_labels",
        )

        n_rows, n_cols = self.matrix.shape
        if n_rows != n_cols:
            raise ValueError("DistanceMatrixResult.matrix must be square.")
        _ensure_same_length(
            n_rows,
            len(self.object_labels),
            "DistanceMatrixResult.matrix size",
            "DistanceMatrixResult.object_labels",
        )

        if not np.allclose(self.matrix, self.matrix.T, atol=1e-12):
            raise ValueError("DistanceMatrixResult.matrix must be symmetric.")

        if not np.allclose(np.diag(self.matrix), 0.0, atol=1e-12):
            raise ValueError("DistanceMatrixResult.matrix diagonal must be zero.")

        if np.any(self.matrix < -1e-12):
            raise ValueError("DistanceMatrixResult.matrix must not contain negative distances.")

    @property
    def n_objects(self) -> int:
        return self.matrix.shape[0]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix, index=self.object_labels, columns=self.object_labels)


@dataclass(slots=True)
class NeighborTableResult:
    """
    Ordered neighbor labels and distances for one metric.
    """
    metric_name: str
    object_labels: list[str]
    neighbor_labels: list[list[str]]
    neighbor_distances: list[list[float]]
    tie_break_rule: str = "index_ascending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metric_name = _ensure_non_empty_str(
            self.metric_name,
            "NeighborTableResult.metric_name",
        ).lower()
        self.object_labels = _ensure_list_of_str(
            self.object_labels,
            "NeighborTableResult.object_labels",
        )

        n_objects = len(self.object_labels)
        if n_objects == 0:
            raise ValueError("NeighborTableResult.object_labels must not be empty.")

        if len(self.neighbor_labels) != n_objects:
            raise ValueError("NeighborTableResult.neighbor_labels row count mismatch.")
        if len(self.neighbor_distances) != n_objects:
            raise ValueError("NeighborTableResult.neighbor_distances row count mismatch.")

        expected_neighbors = n_objects - 1
        object_set = set(self.object_labels)

        for i in range(n_objects):
            labels_row = self.neighbor_labels[i]
            dists_row = self.neighbor_distances[i]

            if len(labels_row) != expected_neighbors:
                raise ValueError(
                    f"NeighborTableResult.neighbor_labels[{i}] must contain {expected_neighbors} neighbors."
                )
            if len(dists_row) != expected_neighbors:
                raise ValueError(
                    f"NeighborTableResult.neighbor_distances[{i}] must contain {expected_neighbors} distances."
                )

            for lbl in labels_row:
                if lbl not in object_set:
                    raise ValueError(f"Unknown neighbor label '{lbl}' found in row {i}.")
                if lbl == self.object_labels[i]:
                    raise ValueError(f"Self-neighbor found in row {i} for object '{lbl}'.")

            if len(set(labels_row)) != len(labels_row):
                raise ValueError(f"Duplicate neighbor labels found in row {i}.")

            if any(not np.isfinite(float(d)) for d in dists_row):
                raise ValueError(f"Non-finite neighbor distance found in row {i}.")
            if any(float(d) < -1e-12 for d in dists_row):
                raise ValueError(f"Negative neighbor distance found in row {i}.")

    @property
    def n_objects(self) -> int:
        return len(self.object_labels)

    @property
    def neighbor_count(self) -> int:
        return self.n_objects - 1

    def labels_frame(self) -> pd.DataFrame:
        columns = [f"NN{i}" for i in range(1, self.neighbor_count + 1)]
        df = pd.DataFrame(self.neighbor_labels, index=self.object_labels, columns=columns)
        df.index.name = "Object"
        return df

    def distances_frame(self) -> pd.DataFrame:
        columns = [f"d{i}" for i in range(1, self.neighbor_count + 1)]
        df = pd.DataFrame(self.neighbor_distances, index=self.object_labels, columns=columns)
        df.index.name = "Object"
        return df


@dataclass(slots=True)
class MajorityMatricesResult:
    """
    Full and reduced majority-rule outputs for one metric.

    New rule:
    - same_class_indicators keeps the full physical neighbor width (n_objects - 1)
    - a_full / b_full use only full_k_values
    - a_reduced / b_reduced use only reduced_k_values
    """
    metric_name: str
    object_labels: list[str]
    classes: np.ndarray

    same_class_indicators: np.ndarray   # shape: (n_objects, n_neighbors_physical)
    a_full: np.ndarray                  # shape: (n_objects, len(full_k_values))
    a_reduced: np.ndarray               # shape: (n_objects, len(reduced_k_values))
    b_full: np.ndarray                  # shape: (n_objects, len(full_k_values))
    b_reduced: np.ndarray               # shape: (n_objects, len(reduced_k_values))

    full_k_values: list[int]
    reduced_k_values: list[int]

    binary_sequences: list[str]
    decimal_values: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metric_name = _ensure_non_empty_str(
            self.metric_name,
            "MajorityMatricesResult.metric_name",
        ).lower()
        self.object_labels = _ensure_list_of_str(
            self.object_labels,
            "MajorityMatricesResult.object_labels",
        )
        self.classes = _ensure_numpy_1d_int(
            self.classes,
            "MajorityMatricesResult.classes",
        )
        self.same_class_indicators = _ensure_numpy_2d_numeric(
            self.same_class_indicators,
            "MajorityMatricesResult.same_class_indicators",
        )
        self.a_full = _ensure_numpy_2d_numeric(self.a_full, "MajorityMatricesResult.a_full")
        self.a_reduced = _ensure_numpy_2d_numeric(self.a_reduced, "MajorityMatricesResult.a_reduced")
        self.b_full = _ensure_numpy_2d_numeric(self.b_full, "MajorityMatricesResult.b_full")
        self.b_reduced = _ensure_numpy_2d_numeric(self.b_reduced, "MajorityMatricesResult.b_reduced")
        self.full_k_values = [int(k) for k in self.full_k_values]
        self.reduced_k_values = [int(k) for k in self.reduced_k_values]
        self.decimal_values = _ensure_numpy_1d_python_int(
            self.decimal_values,
            "MajorityMatricesResult.decimal_values",
        )

        n_objects = len(self.object_labels)
        _ensure_same_length(n_objects, len(self.classes), "object_labels", "classes")
        _ensure_same_length(n_objects, self.same_class_indicators.shape[0], "object_labels", "same_class_indicators rows")
        _ensure_same_length(n_objects, self.a_full.shape[0], "object_labels", "a_full rows")
        _ensure_same_length(n_objects, self.a_reduced.shape[0], "object_labels", "a_reduced rows")
        _ensure_same_length(n_objects, self.b_full.shape[0], "object_labels", "b_full rows")
        _ensure_same_length(n_objects, self.b_reduced.shape[0], "object_labels", "b_reduced rows")
        _ensure_same_length(n_objects, len(self.binary_sequences), "object_labels", "binary_sequences")
        _ensure_same_length(n_objects, len(self.decimal_values), "object_labels", "decimal_values")

        expected_neighbors = n_objects - 1

        # same_class_indicators always keeps the full physical width
        if self.same_class_indicators.shape[1] != expected_neighbors:
            raise ValueError("same_class_indicators must have n_objects - 1 columns.")

        # full_k_values are dynamic but must be valid
        if not self.full_k_values:
            raise ValueError("full_k_values must not be empty.")
        if len(set(self.full_k_values)) != len(self.full_k_values):
            raise ValueError("full_k_values must be unique.")
        if sorted(self.full_k_values) != self.full_k_values:
            raise ValueError("full_k_values must be sorted in ascending order.")
        if self.full_k_values[0] != 1:
            raise ValueError("full_k_values must start at 1.")
        if any(k < 1 or k > expected_neighbors for k in self.full_k_values):
            raise ValueError("All full_k_values must lie in [1, n_objects - 1].")
        if self.full_k_values != list(range(1, self.full_k_values[-1] + 1)):
            raise ValueError("full_k_values must be consecutive starting from 1.")

        # a_full / b_full match the dynamic full_k_values length
        if self.a_full.shape[1] != len(self.full_k_values):
            raise ValueError("a_full column count must match full_k_values length.")
        if self.b_full.shape[1] != len(self.full_k_values):
            raise ValueError("b_full column count must match full_k_values length.")

        # reduced_k_values are dynamic and must be a subset of full_k_values
        if not self.reduced_k_values:
            raise ValueError("reduced_k_values must not be empty.")
        if len(set(self.reduced_k_values)) != len(self.reduced_k_values):
            raise ValueError("reduced_k_values must be unique.")
        if sorted(self.reduced_k_values) != self.reduced_k_values:
            raise ValueError("reduced_k_values must be sorted in ascending order.")
        if any(k < 1 or k > expected_neighbors for k in self.reduced_k_values):
            raise ValueError("All reduced_k_values must lie in [1, n_objects - 1].")
        if not set(self.reduced_k_values).issubset(set(self.full_k_values)):
            raise ValueError("reduced_k_values must be a subset of full_k_values.")

        if self.a_reduced.shape[1] != len(self.reduced_k_values):
            raise ValueError("a_reduced column count must match reduced_k_values length.")
        if self.b_reduced.shape[1] != len(self.reduced_k_values):
            raise ValueError("b_reduced column count must match reduced_k_values length.")

        if not set(np.unique(self.same_class_indicators)).issubset({0.0, 1.0}):
            raise ValueError("same_class_indicators must contain only 0/1 values.")
        if not set(np.unique(self.b_full)).issubset({0.0, 1.0}):
            raise ValueError("b_full must contain only 0/1 values.")
        if not set(np.unique(self.b_reduced)).issubset({0.0, 1.0}):
            raise ValueError("b_reduced must contain only 0/1 values.")

        if set(self.classes.tolist()) != {1, 2}:
            raise ValueError("classes must contain exactly {1, 2}.")

        expected_seq_len = len(self.reduced_k_values)
        for seq in self.binary_sequences:
            if (
                not isinstance(seq, str)
                or len(seq) != expected_seq_len
                or any(ch not in {"0", "1"} for ch in seq)
            ):
                raise ValueError(
                    "Each binary sequence must be a bit-string matching reduced_k_values length."
                )

    def a_full_frame(self) -> pd.DataFrame:
        cols = [f"a{k}" for k in self.full_k_values]
        df = pd.DataFrame(self.a_full.astype(int), index=self.object_labels, columns=cols)
        df.insert(0, "Class", self.classes.astype(int))
        df.index.name = "Object"
        return df

    def a_reduced_frame(self) -> pd.DataFrame:
        cols = [f"a{k}" for k in self.reduced_k_values]
        df = pd.DataFrame(self.a_reduced.astype(int), index=self.object_labels, columns=cols)
        df.insert(0, "Class", self.classes.astype(int))
        df.index.name = "Object"
        return df

    def b_full_frame(self) -> pd.DataFrame:
        cols = [f"b{k}" for k in self.full_k_values]
        df = pd.DataFrame(self.b_full.astype(int), index=self.object_labels, columns=cols)
        df.insert(0, "Class", self.classes.astype(int))
        df.index.name = "Object"
        return df

    def b_reduced_frame(self) -> pd.DataFrame:
        cols = [f"b{k}" for k in self.reduced_k_values]
        df = pd.DataFrame(self.b_reduced.astype(int), index=self.object_labels, columns=cols)
        df.insert(0, "Class", self.classes.astype(int))
        df.index.name = "Object"
        return df


# ============================================================
# Statistics result schemas
# ============================================================

@dataclass(slots=True)
class BinarySequenceStatsRow:
    metric_name: str
    binary_sequence: str
    decimal: int
    count_k1: int
    count_k2: int
    frequency: int
    winner_class: int
    purity: float
    status: str
    winner_rank: int | None = None

    def __post_init__(self) -> None:
        self.metric_name = _ensure_non_empty_str(
            self.metric_name,
            "BinarySequenceStatsRow.metric_name",
        ).lower()
        self.binary_sequence = _ensure_non_empty_str(
            self.binary_sequence,
            "BinarySequenceStatsRow.binary_sequence",
        )
        if any(ch not in {"0", "1"} for ch in self.binary_sequence):
            raise ValueError("binary_sequence must contain only 0/1.")
        self.decimal = int(self.decimal)
        self.count_k1 = int(self.count_k1)
        self.count_k2 = int(self.count_k2)
        self.frequency = int(self.frequency)
        self.winner_class = int(self.winner_class)
        self.purity = float(self.purity)
        self.status = _ensure_non_empty_str(self.status, "BinarySequenceStatsRow.status")

        if self.frequency < 0:
            raise ValueError("frequency must be non-negative.")
        if self.count_k1 < 0 or self.count_k2 < 0:
            raise ValueError("Class counts must be non-negative.")
        if self.count_k1 + self.count_k2 != self.frequency:
            raise ValueError("count_k1 + count_k2 must equal frequency.")
        if self.winner_class not in {0, 1, 2}:
            raise ValueError("winner_class must be one of {0, 1, 2}.")
        if not (0.0 <= self.purity <= 1.0):
            raise ValueError("purity must lie in [0, 1].")
        if self.winner_rank is not None:
            self.winner_rank = int(self.winner_rank)
            if self.winner_rank <= 0:
                raise ValueError("winner_rank must be positive when provided.")


@dataclass(slots=True)
class StatisticsTableResult:
    metric_name: str
    rows: list[BinarySequenceStatsRow]
    normalization_mode: str
    with_normalization: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metric_name = _ensure_non_empty_str(
            self.metric_name,
            "StatisticsTableResult.metric_name",
        ).lower()
        self.normalization_mode = _ensure_non_empty_str(
            self.normalization_mode,
            "StatisticsTableResult.normalization_mode",
        ).lower()

        if not isinstance(self.rows, list):
            raise TypeError("rows must be a list of BinarySequenceStatsRow.")
        for row in self.rows:
            if not isinstance(row, BinarySequenceStatsRow):
                raise TypeError("All rows must be BinarySequenceStatsRow instances.")
            if row.metric_name != self.metric_name:
                raise ValueError("Each row.metric_name must match StatisticsTableResult.metric_name.")

    def to_frame(self) -> pd.DataFrame:
        data = []
        for row in self.rows:
            data.append(
                {
                    "Metric": row.metric_name,
                    "BinarySequence": row.binary_sequence,
                    "Decimal": row.decimal,
                    "CountK1": row.count_k1,
                    "CountK2": row.count_k2,
                    "Frequency": row.frequency,
                    "WinnerClass": row.winner_class,
                    "Purity": row.purity,
                    "Status": row.status,
                    "WinnerRank": row.winner_rank,
                }
            )
        return pd.DataFrame(data)


@dataclass(slots=True)
class FinalComparisonRow:
    metric_name: str
    winner_class: int
    binary_sequence: str
    decimal: int
    frequency: int
    purity: float

    def __post_init__(self) -> None:
        self.metric_name = _ensure_non_empty_str(
            self.metric_name,
            "FinalComparisonRow.metric_name",
        ).lower()
        self.binary_sequence = _ensure_non_empty_str(
            self.binary_sequence,
            "FinalComparisonRow.binary_sequence",
        )
        if any(ch not in {"0", "1"} for ch in self.binary_sequence):
            raise ValueError("binary_sequence must contain only 0/1.")
        self.winner_class = int(self.winner_class)
        self.decimal = int(self.decimal)
        self.frequency = int(self.frequency)
        self.purity = float(self.purity)

        if self.winner_class not in {0, 1, 2}:
            raise ValueError("winner_class must be one of {0, 1, 2}.")
        if self.frequency < 0:
            raise ValueError("frequency must be non-negative.")
        if not (0.0 <= self.purity <= 1.0):
            raise ValueError("purity must lie in [0, 1].")


@dataclass(slots=True)
class FinalComparisonResult:
    rows: list[FinalComparisonRow]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.rows, list):
            raise TypeError("rows must be a list of FinalComparisonRow.")
        for row in self.rows:
            if not isinstance(row, FinalComparisonRow):
                raise TypeError("All rows must be FinalComparisonRow instances.")

    def to_frame(self) -> pd.DataFrame:
        data = []
        for row in self.rows:
            data.append(
                {
                    "Metric": row.metric_name,
                    "WinnerClass": row.winner_class,
                    "BinarySequence": row.binary_sequence,
                    "Decimal": row.decimal,
                    "Frequency": row.frequency,
                    "Purity": row.purity,
                }
            )
        return pd.DataFrame(data)


# ============================================================
# Convenience constructors
# ============================================================

def build_object_labels(count: int, prefix: str = "S") -> list[str]:
    if count <= 0:
        raise ValueError("count must be positive.")
    _ensure_non_empty_str(prefix, "prefix")
    return [f"{prefix}{i}" for i in range(1, count + 1)]


def build_loaded_dataset_from_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    class_column: str,
    object_labels: Sequence[str] | None = None,
    source_path: str | None = None,
    source_format: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> LoadedDataset:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    feature_cols = _ensure_list_of_str(list(feature_columns), "feature_columns")
    class_col = _ensure_non_empty_str(class_column, "class_column")

    missing = [col for col in feature_cols + [class_col] if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in frame: {missing}")

    if object_labels is None:
        labels = build_object_labels(len(frame), prefix="S")
    else:
        labels = _ensure_list_of_str(list(object_labels), "object_labels")
        _ensure_same_length(len(frame), len(labels), "frame rows", "object_labels")

    X = frame.loc[:, feature_cols].to_numpy(dtype=float)
    y = frame.loc[:, class_col].to_numpy(dtype=int)

    return LoadedDataset(
        X=X,
        y=y,
        feature_names=feature_cols,
        object_labels=labels,
        class_column=class_col,
        source_path=source_path,
        source_format=source_format if source_format is not None else "csv",
        raw_frame=frame.copy(),
        metadata={} if metadata is None else dict(metadata),
    )




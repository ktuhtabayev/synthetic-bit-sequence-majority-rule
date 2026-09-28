from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from synthetic_bit_sequence_majority_rule.domain.errors import (
    DatasetFileNotFoundError,
    DatasetParseError,
    DatasetValidationError,
    InvalidClassLabelError,
    InvalidFeatureMatrixError,
    MissingColumnError,
    ObjectLabelError,
    UnsupportedDatasetFormatError,
)
from synthetic_bit_sequence_majority_rule.domain.params import DatasetConfig
from synthetic_bit_sequence_majority_rule.domain.schema import (
    LoadedDataset,
    build_object_labels,
)

logger = logging.getLogger(__name__)


# ============================================================
# Dataset structure metadata
# ============================================================

@dataclass(slots=True)
class DatasetStructureInfo:
    """
    Captures structural facts discovered during dataset loading.

    General project-wide structural rules:
    - first row may contain: m, n, c
    - final sign row may contain feature type signs
      where 1 = quantitative, 0 = nominal
    """
    has_shape_row: bool
    has_feature_sign_row: bool
    m_objects: int | None
    n_features: int | None
    n_classes: int | None
    feature_signs: list[int]
    quantitative_feature_indices: list[int]
    nominal_feature_indices: list[int]

    @property
    def all_features_quantitative(self) -> bool:
        return bool(self.feature_signs) and all(sign == 1 for sign in self.feature_signs)

    @property
    def has_nominal_features(self) -> bool:
        return len(self.nominal_feature_indices) > 0


# ============================================================
# Small helpers
# ============================================================

def _read_text_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="cp1251").splitlines()
        except Exception as exc:
            raise DatasetParseError(path, f"Unable to read text file: {exc}") from exc
    except Exception as exc:
        raise DatasetParseError(path, f"Unable to read text file: {exc}") from exc


def _normalize_format(fmt: str) -> str:
    return fmt.strip().lower()


def _split_line(line: str, delimiter: str) -> list[str]:
    """
    Robust tokenization for matrix-style datasets.

    Handles:
    - commas
    - spaces
    - tabs
    - mixed whitespace
    - trailing commas / trailing empty cells
    """
    text = line.strip()
    if not text:
        return []

    # For matrix-style datasets, support mixed delimiters robustly.
    # We intentionally split on commas and any whitespace.
    tokens = re.split(r"[,\s]+", text)

    # Remove empty tokens caused by trailing delimiters.
    return [tok for tok in tokens if tok != ""]


def _try_parse_int_tokens(tokens: list[str]) -> list[int] | None:
    try:
        return [int(float(tok)) for tok in tokens]
    except Exception:
        return None


def _is_numeric_row(tokens: list[str]) -> bool:
    try:
        for tok in tokens:
            float(tok)
    except ValueError:
        return False
    return True


def _is_shape_row(tokens: list[str]) -> bool:
    ints = _try_parse_int_tokens(tokens)
    if ints is None or len(ints) != 3:
        return False
    m, n, c = ints
    return m > 0 and n > 0 and c > 0


def _is_feature_sign_row(tokens: list[str], expected_n_features: int | None) -> bool:
    ints = _try_parse_int_tokens(tokens)
    if ints is None:
        return False
    if len(ints) == 0:
        return False
    if expected_n_features is not None and len(ints) != expected_n_features:
        return False
    return all(x in (0, 1) for x in ints)


def _coerce_numeric_frame(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    try:
        out = df.apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise InvalidFeatureMatrixError(
            reason=f"Non-numeric values found where numeric values were expected: {exc}",
            dataset_path=path,
        ) from exc

    if out.empty:
        raise InvalidFeatureMatrixError(reason="Dataset is empty after parsing.", dataset_path=path)

    if out.isna().any().any():
        raise InvalidFeatureMatrixError(
            reason="Parsed dataset contains NaN values.",
            dataset_path=path,
        )

    return out


def _validate_shape_against_data(
    path: Path,
    discovered_m: int | None,
    discovered_n: int | None,
    discovered_c: int | None,
    actual_m: int,
    actual_n: int,
    actual_classes: set[int],
) -> None:
    if discovered_m is not None and discovered_m != actual_m:
        raise DatasetValidationError(
            message=f"Shape row mismatch for m: expected {discovered_m}, found {actual_m}.",
            dataset_path=path,
        )

    if discovered_n is not None and discovered_n != actual_n:
        raise DatasetValidationError(
            message=f"Shape row mismatch for n: expected {discovered_n}, found {actual_n}.",
            dataset_path=path,
        )

    if discovered_c is not None and discovered_c != len(actual_classes):
        raise DatasetValidationError(
            message=f"Shape row mismatch for c: expected {discovered_c}, found {len(actual_classes)}.",
            dataset_path=path,
        )


def _apply_label_mapping(y_raw: Iterable[Any], label_mapping: Mapping[int, int], path: Path) -> np.ndarray:
    mapped: list[int] = []
    found_raw_labels: list[Any] = []

    for value in y_raw:
        try:
            raw_label = int(float(value))
        except Exception:
            raw_label = value

        found_raw_labels.append(raw_label)

        if raw_label not in label_mapping:
            raise InvalidClassLabelError(
                found_labels=sorted(set(found_raw_labels), key=lambda x: str(x)),
                dataset_path=path,
            )

        mapped.append(int(label_mapping[raw_label]))

    mapped_arr = np.asarray(mapped, dtype=int)
    classes = set(mapped_arr.tolist())
    if classes != {1, 2}:
        raise InvalidClassLabelError(
            found_labels=sorted(classes),
            dataset_path=path,
        )

    return mapped_arr


def _feature_sign_metadata(feature_signs: list[int]) -> tuple[list[int], list[int]]:
    quantitative = [i for i, sign in enumerate(feature_signs) if sign == 1]
    nominal = [i for i, sign in enumerate(feature_signs) if sign == 0]
    return quantitative, nominal


def _detect_matrix_style_text_file(lines: list[str], delimiter: str) -> bool:
    """
    Detect whether a text file's lines should be treated as matrix-style.

    Strong signals:
    - first row is shape row (m, n, c)
    - last row is feature-sign row

    A first row with non-numeric tokens is a header, which the matrix-style
    parser cannot read, so such a file is never matrix-style.
    """
    lines = [line for line in lines if line.strip()]
    if not lines:
        return False

    rows = [_split_line(line, delimiter) for line in lines]
    rows = [row for row in rows if row]

    if not rows:
        return False

    if _is_shape_row(rows[0]):
        return True

    if not _is_numeric_row(rows[0]):
        return False

    if _is_feature_sign_row(rows[-1], expected_n_features=None):
        return True

    return False


# ============================================================
# General matrix-style text parser
# ============================================================

def _parse_matrix_style_dataset(
    path: Path,
    delimiter: str,
    label_mapping: Mapping[int, int],
    object_name_prefix: str,
    lines: list[str] | None = None,
) -> LoadedDataset:
    """
    General loader for matrix-style datasets.

    Supported project-wide structural rules:
    - optional first row: m, n, c
    - optional final sign row: feature type signs (1 quantitative, 0 nominal)

    Data rows are assumed to contain:
    - n feature values
    - 1 class value

    lines are the file's lines when the caller has already read it.
    """
    if lines is None:
        lines = _read_text_lines(path)
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise DatasetParseError(path, "File is empty.")

    parsed_rows = [_split_line(line, delimiter) for line in lines]
    parsed_rows = [row for row in parsed_rows if row]

    if not parsed_rows:
        raise DatasetParseError(path, "No usable rows found after tokenization.")

    first_row = parsed_rows[0]
    has_shape_row = _is_shape_row(first_row)

    discovered_m: int | None = None
    discovered_n: int | None = None
    discovered_c: int | None = None

    data_start_idx = 0
    if has_shape_row:
        parsed_shape = _try_parse_int_tokens(first_row)
        if parsed_shape is None:
            raise DatasetParseError(path, "Shape row exists but could not be parsed.")
        discovered_m, discovered_n, discovered_c = parsed_shape
        data_start_idx = 1

    remaining_rows = parsed_rows[data_start_idx:]
    if not remaining_rows:
        raise DatasetParseError(path, "No data rows found after removing shape row.")

    expected_n_features = discovered_n

    last_row = remaining_rows[-1]
    has_feature_sign_row = _is_feature_sign_row(last_row, expected_n_features)

    feature_signs: list[int]
    if has_feature_sign_row:
        ints = _try_parse_int_tokens(last_row)
        assert ints is not None
        feature_signs = ints
        data_rows = remaining_rows[:-1]
    else:
        data_rows = remaining_rows
        if expected_n_features is None:
            inferred_width = len(data_rows[0])
            if inferred_width < 2:
                raise DatasetParseError(path, "Each data row must contain at least one feature and one class.")
            expected_n_features = inferred_width - 1
        feature_signs = [1] * expected_n_features

    if not data_rows:
        raise DatasetParseError(path, "No object rows found after removing metadata rows.")

    if expected_n_features is None:
        inferred_width = len(data_rows[0])
        if inferred_width < 2:
            raise DatasetParseError(path, "Each data row must contain at least one feature and one class.")
        expected_n_features = inferred_width - 1

    expected_total_columns = expected_n_features + 1
    for row_idx, row in enumerate(data_rows):
        if len(row) != expected_total_columns:
            raise DatasetParseError(
                path,
                (
                    f"Data row {row_idx + 1} has {len(row)} columns, "
                    f"expected {expected_total_columns} (= {expected_n_features} features + 1 class)."
                ),
            )

    feature_names = [f"x{i}" for i in range(1, expected_n_features + 1)]
    columns = feature_names + ["Class"]
    data_frame = pd.DataFrame(data_rows, columns=columns)
    data_frame = _coerce_numeric_frame(data_frame, path)

    X = data_frame.loc[:, feature_names].to_numpy(dtype=float)
    y = _apply_label_mapping(data_frame["Class"].tolist(), label_mapping, path)

    object_labels = build_object_labels(len(data_frame), prefix=object_name_prefix)

    quantitative, nominal = _feature_sign_metadata(feature_signs)
    structure_info = DatasetStructureInfo(
        has_shape_row=has_shape_row,
        has_feature_sign_row=has_feature_sign_row,
        m_objects=discovered_m,
        n_features=expected_n_features,
        n_classes=discovered_c,
        feature_signs=feature_signs,
        quantitative_feature_indices=quantitative,
        nominal_feature_indices=nominal,
    )

    _validate_shape_against_data(
        path=path,
        discovered_m=discovered_m,
        discovered_n=expected_n_features if has_shape_row else None,
        discovered_c=discovered_c,
        actual_m=X.shape[0],
        actual_n=X.shape[1],
        actual_classes=set(y.tolist()),
    )

    return LoadedDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        object_labels=object_labels,
        class_column="Class",
        source_path=str(path),
        source_format=path.suffix.lower().lstrip("."),
        raw_frame=data_frame.copy(),
        metadata={
            "has_shape_row": structure_info.has_shape_row,
            "has_feature_sign_row": structure_info.has_feature_sign_row,
            "m_objects": structure_info.m_objects,
            "n_features": structure_info.n_features,
            "n_classes": structure_info.n_classes,
            "feature_signs": structure_info.feature_signs,
            "quantitative_feature_indices": structure_info.quantitative_feature_indices,
            "nominal_feature_indices": structure_info.nominal_feature_indices,
            "all_features_quantitative": structure_info.all_features_quantitative,
            "has_nominal_features": structure_info.has_nominal_features,
        },
    )


# ============================================================
# Headered tabular CSV / Excel parser
# ============================================================

def _parse_headered_table_dataset(
    frame: pd.DataFrame,
    path: Path,
    dataset_config: DatasetConfig,
) -> LoadedDataset:
    required_columns = list(dataset_config.feature_columns) + [dataset_config.class_column]
    missing = [col for col in required_columns if col not in frame.columns]
    if missing:
        raise MissingColumnError(missing_columns=missing, dataset_path=path)

    if dataset_config.object_id_column:
        if dataset_config.object_id_column not in frame.columns:
            raise MissingColumnError(
                missing_columns=[dataset_config.object_id_column],
                dataset_path=path,
            )
        object_labels = frame[dataset_config.object_id_column].astype(str).tolist()
        if len(set(object_labels)) != len(object_labels):
            raise ObjectLabelError(
                reason=f"Duplicate values found in object_id_column '{dataset_config.object_id_column}'.",
                dataset_path=path,
            )
    else:
        object_labels = build_object_labels(len(frame), prefix=dataset_config.object_name_prefix)

    feature_frame = frame.loc[:, dataset_config.feature_columns].copy()
    feature_frame = _coerce_numeric_frame(feature_frame, path)

    y = _apply_label_mapping(frame[dataset_config.class_column].tolist(), dataset_config.label_mapping, path)

    feature_signs = [1] * len(dataset_config.feature_columns)
    quantitative, nominal = _feature_sign_metadata(feature_signs)

    return LoadedDataset(
        X=feature_frame.to_numpy(dtype=float),
        y=y,
        feature_names=list(dataset_config.feature_columns),
        object_labels=object_labels,
        class_column=dataset_config.class_column,
        source_path=str(path),
        source_format=path.suffix.lower().lstrip("."),
        raw_frame=frame.copy(),
        metadata={
            "has_shape_row": False,
            "has_feature_sign_row": False,
            "m_objects": len(frame),
            "n_features": len(dataset_config.feature_columns),
            "n_classes": len(set(y.tolist())),
            "feature_signs": feature_signs,
            "quantitative_feature_indices": quantitative,
            "nominal_feature_indices": nominal,
            "all_features_quantitative": True,
            "has_nominal_features": False,
        },
    )


# ============================================================
# File-format-specific loaders
# ============================================================

def _load_csv(path: Path, dataset_config: DatasetConfig) -> LoadedDataset:
    """
    CSV loader.

    Supports:
    1. matrix-style datasets
    2. headered table datasets

    has_header=False forces the matrix-style parse for CSVs whose first
    row is plain data, since a headered parse would swallow that row.
    """
    lines = _read_text_lines(path)
    if not dataset_config.has_header or _detect_matrix_style_text_file(lines, dataset_config.delimiter):
        return _parse_matrix_style_dataset(
            path=path,
            delimiter=dataset_config.delimiter,
            label_mapping=dataset_config.label_mapping,
            object_name_prefix=dataset_config.object_name_prefix,
            lines=lines,
        )

    # Otherwise try headered CSV.
    try:
        frame = pd.read_csv(path, delimiter=dataset_config.delimiter)
    except Exception as exc:
        raise DatasetParseError(path, f"Unable to parse CSV file: {exc}") from exc

    return _parse_headered_table_dataset(frame=frame, path=path, dataset_config=dataset_config)


def _load_dat(path: Path, dataset_config: DatasetConfig) -> LoadedDataset:
    """
    DAT loader.

    Uses the same general matrix-style rules.
    Delimiter is handled robustly by tokenization, so mixed tabs/spaces are fine.
    """
    return _parse_matrix_style_dataset(
        path=path,
        delimiter=dataset_config.delimiter,
        label_mapping=dataset_config.label_mapping,
        object_name_prefix=dataset_config.object_name_prefix,
    )


def _load_excel(path: Path, dataset_config: DatasetConfig) -> LoadedDataset:
    try:
        frame = pd.read_excel(path)
    except Exception as exc:
        raise DatasetParseError(path, f"Unable to parse Excel file: {exc}") from exc

    return _parse_headered_table_dataset(frame=frame, path=path, dataset_config=dataset_config)


# ============================================================
# Public API
# ============================================================

def load_dataset(dataset_config: DatasetConfig) -> LoadedDataset:
    """
    Main dataset loader.

    Currently supported and extendable:
    - csv
    - dat
    - xlsx / xls

    General dataset structural rules supported for text-like datasets:
    - optional shape row: m, n, c
    - optional final sign row: feature signs
    """
    path = Path(dataset_config.path)
    if not path.exists():
        raise DatasetFileNotFoundError(path)

    fmt = _normalize_format(dataset_config.format)
    if fmt not in set(dataset_config.supported_formats):
        raise UnsupportedDatasetFormatError(fmt, dataset_config.supported_formats)

    if fmt == "csv":
        dataset = _load_csv(path, dataset_config)
    elif fmt == "dat":
        dataset = _load_dat(path, dataset_config)
    elif fmt in {"xlsx", "xls"}:
        dataset = _load_excel(path, dataset_config)
    else:
        raise UnsupportedDatasetFormatError(fmt, dataset_config.supported_formats)

    _validate_loaded_dataset_against_config(dataset, dataset_config)
    return dataset


def try_alternate_dataset_paths(dataset_config: DatasetConfig) -> LoadedDataset:
    """
    Try the primary dataset path first, then alternate paths.
    Useful when the config allows both default.csv and default.dat.
    """
    candidate_paths = [dataset_config.path] + list(dataset_config.alternate_paths)
    errors: list[str] = []

    for candidate in candidate_paths:
        candidate_path = Path(candidate)
        suffix = candidate_path.suffix.lower().lstrip(".")

        cfg = DatasetConfig(
            path=candidate_path,
            alternate_paths=list(dataset_config.alternate_paths),
            format=suffix if suffix else dataset_config.format,
            supported_formats=list(dataset_config.supported_formats),
            delimiter=dataset_config.delimiter,
            has_header=dataset_config.has_header if suffix != "dat" else False,
            object_id_column=dataset_config.object_id_column,
            object_name_prefix=dataset_config.object_name_prefix,
            feature_columns=list(dataset_config.feature_columns),
            class_column=dataset_config.class_column,
            label_mapping=dict(dataset_config.label_mapping),
        )

        try:
            dataset = load_dataset(cfg)
        except Exception as exc:
            errors.append(f"{candidate_path}: {exc}")
            continue

        if errors:
            logger.warning(
                "Loaded alternate dataset %s after failures:\n%s",
                candidate_path,
                "\n".join(errors),
            )
        return dataset

    raise DatasetParseError(
        dataset_config.path,
        "All candidate dataset paths failed:\n" + "\n".join(errors),
    )


def infer_dataset_format_from_path(path: str | Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    if not suffix:
        raise DatasetParseError(path, "Unable to infer dataset format from file extension.")
    return suffix


def load_dataset_from_path(
    path: str | Path,
    *,
    delimiter: str = ",",
    has_header: bool = True,
    object_name_prefix: str = "S",
    feature_columns: list[str] | None = None,
    class_column: str = "Class",
    label_mapping: Mapping[int, int] | None = None,
    supported_formats: list[str] | None = None,
) -> LoadedDataset:
    """
    Convenience loader for ad-hoc use outside the full config pipeline.
    """
    path_obj = Path(path)
    fmt = infer_dataset_format_from_path(path_obj)

    cfg = DatasetConfig(
        path=path_obj,
        alternate_paths=[],
        format=fmt,
        supported_formats=supported_formats or ["csv", "dat", "xlsx", "xls"],
        delimiter=delimiter,
        has_header=has_header if fmt != "dat" else False,
        object_id_column=None,
        object_name_prefix=object_name_prefix,
        feature_columns=feature_columns or ["x1", "x2", "x3", "x4", "x5", "x6"],
        class_column=class_column,
        label_mapping=dict(label_mapping or {1: 1, 2: 2}),
    )
    return load_dataset(cfg)


# ============================================================
# Validation
# ============================================================

def _validate_loaded_dataset_against_config(
    dataset: LoadedDataset,
    dataset_config: DatasetConfig,
) -> None:
    if dataset.n_objects <= 0:
        raise DatasetValidationError(
            message="Loaded dataset contains zero objects.",
            dataset_path=dataset.source_path,
        )

    if dataset.n_features <= 0:
        raise DatasetValidationError(
            message="Loaded dataset contains zero features.",
            dataset_path=dataset.source_path,
        )

    if set(dataset.y.tolist()) != {1, 2}:
        raise InvalidClassLabelError(
            found_labels=sorted(set(dataset.y.tolist())),
            dataset_path=dataset.source_path,
        )

    if len(dataset.object_labels) != dataset.n_objects:
        raise ObjectLabelError(
            reason="Number of object labels does not match number of rows.",
            dataset_path=dataset.source_path,
        )

    if len(set(dataset.object_labels)) != len(dataset.object_labels):
        raise ObjectLabelError(
            reason="Object labels are not unique.",
            dataset_path=dataset.source_path,
        )

    if len(dataset.feature_names) != dataset.n_features:
        raise DatasetValidationError(
            message="Number of feature names does not match number of feature columns.",
            dataset_path=dataset.source_path,
        )

    if dataset.metadata.get("n_features") is not None:
        declared_n = int(dataset.metadata["n_features"])
        if declared_n != dataset.n_features:
            raise DatasetValidationError(
                message=f"Declared feature count {declared_n} does not match loaded feature count {dataset.n_features}.",
                dataset_path=dataset.source_path,
            )

    if dataset.metadata.get("m_objects") is not None:
        declared_m = int(dataset.metadata["m_objects"])
        if declared_m != dataset.n_objects:
            raise DatasetValidationError(
                message=f"Declared object count {declared_m} does not match loaded object count {dataset.n_objects}.",
                dataset_path=dataset.source_path,
            )

    if dataset.metadata.get("n_classes") is not None:
        declared_c = int(dataset.metadata["n_classes"])
        actual_c = len(set(dataset.y.tolist()))
        if declared_c != actual_c:
            raise DatasetValidationError(
                message=f"Declared class count {declared_c} does not match loaded class count {actual_c}.",
                dataset_path=dataset.source_path,
            )

    feature_signs = dataset.metadata.get("feature_signs")
    if feature_signs is not None:
        if len(feature_signs) != dataset.n_features:
            raise DatasetValidationError(
                message="Feature-sign row length does not match number of features.",
                dataset_path=dataset.source_path,
            )
        if any(sign not in (0, 1) for sign in feature_signs):
            raise DatasetValidationError(
                message="Feature signs must contain only 0 or 1.",
                dataset_path=dataset.source_path,
            )

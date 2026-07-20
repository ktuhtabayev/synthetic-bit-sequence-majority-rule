from __future__ import annotations

from pathlib import Path
from typing import Any


class SyntheticBitSequenceError(Exception):
    """
    Base exception for the whole project.

    All custom project exceptions should inherit from this class so callers
    can catch one umbrella error type when needed.
    """


# ============================================================
# Configuration errors
# ============================================================

class ConfigError(SyntheticBitSequenceError):
    """
    Raised when configuration loading, parsing, or validation fails.
    """


class ConfigFileNotFoundError(ConfigError):
    """
    Raised when a requested configuration file does not exist.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        super().__init__(f"Configuration file not found: {self.path}")


class ConfigParseError(ConfigError):
    """
    Raised when a configuration file exists but cannot be parsed.
    """

    def __init__(self, path: str | Path, reason: str) -> None:
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"Failed to parse configuration file '{self.path}': {self.reason}")


class ConfigValidationError(ConfigError):
    """
    Raised when configuration content is syntactically readable
    but semantically invalid.
    """

    def __init__(self, message: str, section: str | None = None) -> None:
        self.section = section
        prefix = f"[{section}] " if section else ""
        super().__init__(f"{prefix}{message}")


# ============================================================
# Dataset / schema errors
# ============================================================

class DatasetError(SyntheticBitSequenceError):
    """
    Base class for dataset-related failures.
    """


class DatasetFileNotFoundError(DatasetError):
    """
    Raised when the dataset file path does not exist.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        super().__init__(f"Dataset file not found: {self.path}")


class UnsupportedDatasetFormatError(DatasetError):
    """
    Raised when the dataset file format is not supported by the loader.
    """

    def __init__(self, fmt: str, supported_formats: list[str] | tuple[str, ...]) -> None:
        self.fmt = fmt
        self.supported_formats = list(supported_formats)
        super().__init__(
            f"Unsupported dataset format '{self.fmt}'. "
            f"Supported formats: {', '.join(self.supported_formats)}"
        )


class DatasetParseError(DatasetError):
    """
    Raised when a dataset file exists but cannot be parsed correctly.
    """

    def __init__(self, path: str | Path, reason: str) -> None:
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"Failed to parse dataset '{self.path}': {self.reason}")


class DatasetValidationError(DatasetError):
    """
    Raised when parsed dataset content does not satisfy the expected schema.
    """

    def __init__(self, message: str, dataset_path: str | Path | None = None) -> None:
        self.dataset_path = Path(dataset_path) if dataset_path is not None else None
        prefix = f"[{self.dataset_path}] " if self.dataset_path is not None else ""
        super().__init__(f"{prefix}{message}")


class MissingColumnError(DatasetValidationError):
    """
    Raised when one or more required columns are missing in the dataset.
    """

    def __init__(
        self,
        missing_columns: list[str] | tuple[str, ...],
        dataset_path: str | Path | None = None,
    ) -> None:
        self.missing_columns = list(missing_columns)
        message = f"Missing required columns: {', '.join(self.missing_columns)}"
        super().__init__(message=message, dataset_path=dataset_path)


class InvalidClassLabelError(DatasetValidationError):
    """
    Raised when class labels are not valid for the current two-class setup.
    """

    def __init__(self, found_labels: list[Any], dataset_path: str | Path | None = None) -> None:
        self.found_labels = found_labels
        message = (
            "Invalid class labels for the current problem. "
            f"Expected exactly two mapped classes {{1, 2}}, found: {self.found_labels}"
        )
        super().__init__(message=message, dataset_path=dataset_path)


class InvalidFeatureMatrixError(DatasetValidationError):
    """
    Raised when feature matrix values are non-numeric, empty, or malformed.
    """

    def __init__(self, reason: str, dataset_path: str | Path | None = None) -> None:
        self.reason = reason
        super().__init__(message=f"Invalid feature matrix: {self.reason}", dataset_path=dataset_path)


class ObjectLabelError(DatasetValidationError):
    """
    Raised when object labels are missing, duplicated, or malformed.
    """

    def __init__(self, reason: str, dataset_path: str | Path | None = None) -> None:
        self.reason = reason
        super().__init__(message=f"Invalid object labels: {self.reason}", dataset_path=dataset_path)


# ============================================================
# Preprocessing / normalization errors
# ============================================================

class PreprocessingError(SyntheticBitSequenceError):
    """
    Base class for preprocessing-related errors.
    """


class NormalizationError(PreprocessingError):
    """
    Raised when normalization fails.
    """

    def __init__(self, mode: str, reason: str) -> None:
        self.mode = mode
        self.reason = reason
        super().__init__(f"Normalization failed for mode '{self.mode}': {self.reason}")


# ============================================================
# Distance / neighbors / majority-rule errors
# ============================================================

class AlgorithmError(SyntheticBitSequenceError):
    """
    Base class for algorithmic failures.
    """


class DistanceComputationError(AlgorithmError):
    """
    Raised when a distance matrix cannot be computed.
    """

    def __init__(self, metric: str, reason: str) -> None:
        self.metric = metric
        self.reason = reason
        super().__init__(f"Distance computation failed for metric '{self.metric}': {self.reason}")


class NeighborConstructionError(AlgorithmError):
    """
    Raised when ordered neighbors cannot be constructed correctly.
    """

    def __init__(self, metric: str, reason: str) -> None:
        self.metric = metric
        self.reason = reason
        super().__init__(f"Neighbor construction failed for metric '{self.metric}': {self.reason}")


class MajorityRuleError(AlgorithmError):
    """
    Raised when A(S), b(S), binary sequences, or decimal encoding fails.
    """

    def __init__(self, reason: str, metric: str | None = None) -> None:
        self.metric = metric
        self.reason = reason
        prefix = f"[{metric}] " if metric else ""
        super().__init__(f"{prefix}Majority-rule computation failed: {self.reason}")


class StatisticsComputationError(AlgorithmError):
    """
    Raised when statistics or final comparison generation fails.
    """

    def __init__(self, reason: str, metric: str | None = None) -> None:
        self.metric = metric
        self.reason = reason
        prefix = f"[{metric}] " if metric else ""
        super().__init__(f"{prefix}Statistics computation failed: {self.reason}")


# ============================================================
# IO / writing errors
# ============================================================

class IOErrorBase(SyntheticBitSequenceError):
    """
    Base class for read/write failures inside the project.
    """


class OutputWriteError(IOErrorBase):
    """
    Raised when a file cannot be written to disk.
    """

    def __init__(self, path: str | Path, reason: str) -> None:
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"Failed to write output file '{self.path}': {self.reason}")


class OutputDirectoryError(IOErrorBase):
    """
    Raised when an output directory cannot be created or used.
    """

    def __init__(self, path: str | Path, reason: str) -> None:
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"Failed to prepare output directory '{self.path}': {self.reason}")


class SerializationError(IOErrorBase):
    """
    Raised when serialization to JSON/CSV/Excel or similar fails.
    """

    def __init__(self, target: str, reason: str) -> None:
        self.target = target
        self.reason = reason
        super().__init__(f"Serialization failed for '{self.target}': {self.reason}")


# ============================================================
# Service / pipeline errors
# ============================================================

class ServiceError(SyntheticBitSequenceError):
    """
    Base class for orchestration-layer failures.
    """


class PipelineExecutionError(ServiceError):
    """
    Raised when the end-to-end pipeline fails at runtime.
    """

    def __init__(self, stage: str, reason: str) -> None:
        self.stage = stage
        self.reason = reason
        super().__init__(f"Pipeline failed at stage '{self.stage}': {self.reason}")


class ExperimentExecutionError(ServiceError):
    """
    Raised when a named experiment run fails.
    """

    def __init__(self, experiment_name: str, reason: str) -> None:
        self.experiment_name = experiment_name
        self.reason = reason
        super().__init__(f"Experiment '{self.experiment_name}' failed: {self.reason}")
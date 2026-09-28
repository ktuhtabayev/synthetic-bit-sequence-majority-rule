from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast


NormalizationMode = Literal["none", "minmax", "zscore"]
DistanceMetricName = Literal["euclidean", "chebyshev", "canberra", "manhattan"]
TieBreakRule = Literal["index_ascending"]
MajorityComparison = Literal["strict_greater", "greater_or_equal"]
BitOrder = Literal["left_to_right", "right_to_left"]
DominantClassRule = Literal["larger_count", "tie_zero"]
PurityFormula = Literal["dominant_count_div_total"]


def _ensure_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"'{name}' must be a mapping, got {type(value).__name__}.")
    return value


def _ensure_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"'{name}' must be a sequence, got {type(value).__name__}.")
    return value


def _to_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _to_list_of_str(value: Sequence[Any] | None, name: str) -> list[str]:
    if value is None:
        return []
    seq = _ensure_sequence(value, name)
    result: list[str] = []
    for item in seq:
        if not isinstance(item, str):
            raise TypeError(f"All items in '{name}' must be strings.")
        result.append(item)
    return result


@dataclass(slots=True)
class ProjectConfig:
    name: str = "synthetic-bit-sequence-majority-rule"
    version: str = "0.1.0"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Project name cannot be empty.")
        if not self.version.strip():
            raise ValueError("Project version cannot be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ProjectConfig":
        if data is None:
            return cls()
        d = _ensure_mapping(data, "project")
        defaults = cls()
        cfg = cls(
            name=str(d.get("name", defaults.name)),
            version=str(d.get("version", defaults.version)),
        )
        cfg.validate()
        return cfg


@dataclass(slots=True)
class RunConfig:
    run_name: str = "default_run"
    save_outputs: bool = True
    output_root: Path = Path("outputs/runs")

    def validate(self) -> None:
        if not self.run_name.strip():
            raise ValueError("run.run_name cannot be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RunConfig":
        if data is None:
            return cls()
        d = _ensure_mapping(data, "run")
        defaults = cls()
        cfg = cls(
            run_name=str(d.get("run_name", defaults.run_name)),
            save_outputs=bool(d.get("save_outputs", defaults.save_outputs)),
            output_root=_to_path(d.get("output_root")) or defaults.output_root,
        )
        cfg.validate()
        return cfg


@dataclass(slots=True)
class DatasetConfig:
    path: Path = Path("datasets/default.csv")
    alternate_paths: list[Path] = field(default_factory=lambda: [Path("datasets/default.dat")])
    format: str = "csv"
    supported_formats: list[str] = field(default_factory=lambda: ["csv", "dat"])
    delimiter: str = ","
    has_header: bool = True

    object_id_column: str | None = None
    object_name_prefix: str = "S"

    feature_columns: list[str] = field(default_factory=lambda: ["x1", "x2", "x3", "x4", "x5", "x6"])
    class_column: str = "Class"
    label_mapping: dict[int, int] = field(default_factory=lambda: {1: 1, 2: 2})

    def validate(self) -> None:
        if not str(self.path):
            raise ValueError("dataset.path cannot be empty.")
        if not self.supported_formats:
            raise ValueError("dataset.supported_formats cannot be empty.")
        if self.format not in self.supported_formats:
            raise ValueError(
                f"dataset.format='{self.format}' is not in supported_formats={self.supported_formats}."
            )
        if self.format == "csv" and not self.delimiter:
            raise ValueError("dataset.delimiter cannot be empty for csv format.")
        if not self.object_name_prefix.strip():
            raise ValueError("dataset.object_name_prefix cannot be empty.")
        if not self.feature_columns:
            raise ValueError("dataset.feature_columns cannot be empty.")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("dataset.feature_columns contains duplicates.")
        if not self.class_column.strip():
            raise ValueError("dataset.class_column cannot be empty.")
        if self.class_column in self.feature_columns:
            raise ValueError("dataset.class_column must not be included in dataset.feature_columns.")
        if not self.label_mapping:
            raise ValueError("dataset.label_mapping cannot be empty.")
        normalized_values = set(self.label_mapping.values())
        if normalized_values != {1, 2}:
            raise ValueError(
                "dataset.label_mapping values must map to exactly {1, 2} for the current two-class problem."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "DatasetConfig":
        if data is None:
            cfg = cls()
            cfg.validate()
            return cfg

        d = _ensure_mapping(data, "dataset")
        defaults = cls()

        raw_alternate = d.get("alternate_paths", [str(p) for p in defaults.alternate_paths])
        alt_paths = [Path(p) for p in raw_alternate if p is not None]

        raw_label_mapping = d.get("label_mapping", defaults.label_mapping)
        if not isinstance(raw_label_mapping, Mapping):
            raise TypeError("dataset.label_mapping must be a mapping.")
        label_mapping: dict[int, int] = {int(k): int(v) for k, v in raw_label_mapping.items()}

        cfg = cls(
            path=_to_path(d.get("path")) or defaults.path,
            alternate_paths=alt_paths,
            format=str(d.get("format", defaults.format)).lower(),
            supported_formats=[str(x).lower() for x in d.get("supported_formats", defaults.supported_formats)],
            delimiter=str(d.get("delimiter", defaults.delimiter)),
            has_header=bool(d.get("has_header", defaults.has_header)),
            object_id_column=d.get("object_id_column"),
            object_name_prefix=str(d.get("object_name_prefix", defaults.object_name_prefix)),
            feature_columns=_to_list_of_str(d.get("feature_columns", defaults.feature_columns), "dataset.feature_columns"),
            class_column=str(d.get("class_column", defaults.class_column)),
            label_mapping=label_mapping,
        )
        cfg.validate()
        return cfg


@dataclass(slots=True)
class NormalizationConfig:
    mode: NormalizationMode = "none"
    apply_before_distance: bool = True

    def validate(self) -> None:
        allowed = {"none", "minmax", "zscore"}
        if self.mode not in allowed:
            raise ValueError(f"preprocessing.normalization.mode must be one of {sorted(allowed)}.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "NormalizationConfig":
        if data is None:
            cfg = cls()
            cfg.validate()
            return cfg
        d = _ensure_mapping(data, "preprocessing.normalization")
        defaults = cls()
        cfg = cls(
            mode=cast(NormalizationMode, str(d.get("mode", defaults.mode)).lower()),
            apply_before_distance=bool(d.get("apply_before_distance", defaults.apply_before_distance)),
        )
        cfg.validate()
        return cfg


@dataclass(slots=True)
class PreprocessingConfig:
    enabled: bool = True
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    def validate(self) -> None:
        self.normalization.validate()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PreprocessingConfig":
        if data is None:
            cfg = cls()
            cfg.validate()
            return cfg
        d = _ensure_mapping(data, "preprocessing")
        defaults = cls()
        cfg = cls(
            enabled=bool(d.get("enabled", defaults.enabled)),
            normalization=NormalizationConfig.from_dict(d.get("normalization")),
        )
        cfg.validate()
        return cfg


@dataclass(slots=True)
class MetricsConfig:
    enabled: list[DistanceMetricName] = field(default_factory=lambda: ["euclidean", "chebyshev", "canberra", "manhattan"])

    def validate(self) -> None:
        allowed = {"euclidean", "chebyshev", "canberra", "manhattan"}
        if not self.enabled:
            raise ValueError("metrics.enabled cannot be empty.")
        invalid = [m for m in self.enabled if m not in allowed]
        if invalid:
            raise ValueError(f"metrics.enabled contains invalid metrics: {invalid}.")
        if len(set(self.enabled)) != len(self.enabled):
            raise ValueError("metrics.enabled contains duplicates.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MetricsConfig":
        if data is None:
            cfg = cls()
            cfg.validate()
            return cfg
        d = _ensure_mapping(data, "metrics")
        defaults = cls()
        cfg = cls(enabled=[cast(DistanceMetricName, str(x).lower()) for x in d.get("enabled", defaults.enabled)])
        cfg.validate()
        return cfg


@dataclass(slots=True)
class NeighborsConfig:
    tie_break_rule: TieBreakRule = "index_ascending"
    exclude_self: bool = True

    def validate(self) -> None:
        if self.tie_break_rule != "index_ascending":
            raise ValueError("neighbors.tie_break_rule must currently be 'index_ascending'.")
        if not self.exclude_self:
            raise ValueError("neighbors.exclude_self must remain true for the current problem definition.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "NeighborsConfig":
        if data is None:
            cfg = cls()
            cfg.validate()
            return cfg
        d = _ensure_mapping(data, "neighbors")
        defaults = cls()
        cfg = cls(
            tie_break_rule=cast(TieBreakRule, str(d.get("tie_break_rule", defaults.tie_break_rule)).lower()),
            exclude_self=bool(d.get("exclude_self", defaults.exclude_self)),
        )
        cfg.validate()
        return cfg


@dataclass(slots=True)
class FullKRangeConfig:
    start: int = 1
    end: str = "auto_formula"

    def validate(self) -> None:
        if self.start != 1:
            raise ValueError("k_values.full.start must currently be 1.")
        if self.end != "auto_formula":
            raise ValueError("k_values.full.end must currently be 'auto_formula'.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "FullKRangeConfig":
        if data is None:
            cfg = cls()
            cfg.validate()
            return cfg
        d = _ensure_mapping(data, "k_values.full")
        defaults = cls()
        cfg = cls(
            start=int(d.get("start", defaults.start)),
            end=str(d.get("end", defaults.end)),
        )
        cfg.validate()
        return cfg


@dataclass(slots=True)
class ReducedKConfig:
    mode: str = "odd_from_3"

    def validate(self) -> None:
        if self.mode != "odd_from_3":
            raise ValueError("k_values.reduced.mode must currently be 'odd_from_3'.")

    @classmethod
    def from_dict(cls, data: Any) -> "ReducedKConfig":
        if data is None:
            cfg = cls()
            cfg.validate()
            return cfg

        if not isinstance(data, Mapping):
            raise TypeError("k_values.reduced must be a mapping with a 'mode' field.")

        d = _ensure_mapping(data, "k_values.reduced")
        defaults = cls()
        cfg = cls(mode=str(d.get("mode", defaults.mode)).lower())
        cfg.validate()
        return cfg


@dataclass(slots=True)
class KValuesConfig:
    full: FullKRangeConfig = field(default_factory=FullKRangeConfig)
    reduced: ReducedKConfig = field(default_factory=ReducedKConfig)

    def validate(self) -> None:
        self.full.validate()
        self.reduced.validate()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "KValuesConfig":
        if data is None:
            cfg = cls()
            cfg.validate()
            return cfg

        d = _ensure_mapping(data, "k_values")
        cfg = cls(
            full=FullKRangeConfig.from_dict(d.get("full")),
            reduced=ReducedKConfig.from_dict(d.get("reduced")),
        )
        cfg.validate()
        return cfg


@dataclass(slots=True)
class MajorityRuleConfig:
    threshold: float = 0.5
    comparison: MajorityComparison = "strict_greater"

    def validate(self) -> None:
        allowed = {"strict_greater", "greater_or_equal"}
        if self.comparison not in allowed:
            raise ValueError(f"majority_rule.comparison must be one of {sorted(allowed)}.")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("majority_rule.threshold must be between 0 and 1.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MajorityRuleConfig":
        if data is None:
            cfg = cls()
            cfg.validate()
            return cfg
        d = _ensure_mapping(data, "majority_rule")
        defaults = cls()
        cfg = cls(
            threshold=float(d.get("threshold", defaults.threshold)),
            comparison=cast(MajorityComparison, str(d.get("comparison", defaults.comparison)).lower()),
        )
        cfg.validate()
        return cfg


@dataclass(slots=True)
class BinarySequenceConfig:
    use_reduced_k_values: bool = True
    reduced_order_mode: str = "follow_reduced_k_values"

    def validate(self) -> None:
        if not self.use_reduced_k_values:
            raise ValueError("binary_sequence.use_reduced_k_values must currently be true.")
        if self.reduced_order_mode != "follow_reduced_k_values":
            raise ValueError(
                "binary_sequence.reduced_order_mode must currently be 'follow_reduced_k_values'."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "BinarySequenceConfig":
        if data is None:
            cfg = cls()
            cfg.validate()
            return cfg

        d = _ensure_mapping(data, "binary_sequence")
        defaults = cls()
        cfg = cls(
            use_reduced_k_values=bool(d.get("use_reduced_k_values", defaults.use_reduced_k_values)),
            reduced_order_mode=str(d.get("reduced_order_mode", defaults.reduced_order_mode)).lower(),
        )
        cfg.validate()
        return cfg


@dataclass(slots=True)
class DecimalEncodingConfig:
    enabled: bool = True
    bit_order: BitOrder = "left_to_right"

    def validate(self) -> None:
        allowed = {"left_to_right", "right_to_left"}
        if self.bit_order not in allowed:
            raise ValueError(f"decimal_encoding.bit_order must be one of {sorted(allowed)}.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "DecimalEncodingConfig":
        if data is None:
            cfg = cls()
            cfg.validate()
            return cfg
        d = _ensure_mapping(data, "decimal_encoding")
        defaults = cls()
        cfg = cls(
            enabled=bool(d.get("enabled", defaults.enabled)),
            bit_order=cast(BitOrder, str(d.get("bit_order", defaults.bit_order)).lower()),
        )
        cfg.validate()
        return cfg


@dataclass(slots=True)
class StatisticsTieRuleConfig:
    rules: list[str] = field(
        default_factory=lambda: [
            "higher_purity",
            "higher_frequency",
            "smaller_decimal",
            "smaller_binary_sequence",
        ]
    )

    def validate(self) -> None:
        allowed = {
            "higher_purity",
            "higher_frequency",
            "smaller_decimal",
            "smaller_binary_sequence",
        }
        if not self.rules:
            raise ValueError("statistics.tie_rule cannot be empty.")
        invalid = [rule for rule in self.rules if rule not in allowed]
        if invalid:
            raise ValueError(f"statistics.tie_rule contains invalid entries: {invalid}.")
        if len(set(self.rules)) != len(self.rules):
            raise ValueError("statistics.tie_rule contains duplicates.")

    @classmethod
    def from_dict(cls, data: Sequence[Any] | None) -> "StatisticsTieRuleConfig":
        if data is None:
            cfg = cls()
            cfg.validate()
            return cfg
        cfg = cls(rules=_to_list_of_str(data, "statistics.tie_rule"))
        cfg.validate()
        return cfg


@dataclass(slots=True)
class StatisticsConfig:
    enabled: bool = True
    dominant_class_rule: DominantClassRule = "larger_count"
    purity_formula: PurityFormula = "dominant_count_div_total"
    tie_rule: StatisticsTieRuleConfig = field(default_factory=StatisticsTieRuleConfig)

    def validate(self) -> None:
        dominant_allowed = {"larger_count", "tie_zero"}
        purity_allowed = {"dominant_count_div_total"}

        if self.dominant_class_rule not in dominant_allowed:
            raise ValueError(
                f"statistics.dominant_class_rule must be one of {sorted(dominant_allowed)}."
            )
        if self.purity_formula not in purity_allowed:
            raise ValueError(
                f"statistics.purity_formula must be one of {sorted(purity_allowed)}."
            )
        self.tie_rule.validate()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "StatisticsConfig":
        if data is None:
            cfg = cls()
            cfg.validate()
            return cfg
        d = _ensure_mapping(data, "statistics")
        defaults = cls()
        cfg = cls(
            enabled=bool(d.get("enabled", defaults.enabled)),
            dominant_class_rule=cast(
                DominantClassRule,
                str(d.get("dominant_class_rule", defaults.dominant_class_rule)).lower(),
            ),
            purity_formula=cast(
                PurityFormula,
                str(d.get("purity_formula", defaults.purity_formula)).lower(),
            ),
            tie_rule=StatisticsTieRuleConfig.from_dict(d.get("tie_rule")),
        )
        cfg.validate()
        return cfg


@dataclass(slots=True)
class ExportsConfig:
    distance_matrices: bool = True
    neighbors_tables: bool = True
    full_a_matrices: bool = True
    reduced_a_matrices: bool = True
    full_b_matrices: bool = True
    reduced_b_matrices: bool = True
    stats_tables: bool = True
    final_comparison: bool = True

    # CSV tables and run_info.json are always written; Excel is optional.
    excel: bool = False

    def validate(self) -> None:
        return

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ExportsConfig":
        if data is None:
            cfg = cls()
            cfg.validate()
            return cfg
        d = _ensure_mapping(data, "exports")
        defaults = cls()
        cfg = cls(
            distance_matrices=bool(d.get("distance_matrices", defaults.distance_matrices)),
            neighbors_tables=bool(d.get("neighbors_tables", defaults.neighbors_tables)),
            full_a_matrices=bool(d.get("full_a_matrices", defaults.full_a_matrices)),
            reduced_a_matrices=bool(d.get("reduced_a_matrices", defaults.reduced_a_matrices)),
            full_b_matrices=bool(d.get("full_b_matrices", defaults.full_b_matrices)),
            reduced_b_matrices=bool(d.get("reduced_b_matrices", defaults.reduced_b_matrices)),
            stats_tables=bool(d.get("stats_tables", defaults.stats_tables)),
            final_comparison=bool(d.get("final_comparison", defaults.final_comparison)),
            excel=bool(d.get("excel", defaults.excel)),
        )
        cfg.validate()
        return cfg


@dataclass(slots=True)
class NotesConfig:
    problem: str = ""
    limitation: str = ""
    switching_rule: str = ""

    def validate(self) -> None:
        return

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "NotesConfig":
        if data is None:
            return cls()
        d = _ensure_mapping(data, "notes")
        cfg = cls(
            problem=str(d.get("problem", "")),
            limitation=str(d.get("limitation", "")),
            switching_rule=str(d.get("switching_rule", "")),
        )
        cfg.validate()
        return cfg


@dataclass(slots=True)
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    run: RunConfig = field(default_factory=RunConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    neighbors: NeighborsConfig = field(default_factory=NeighborsConfig)
    k_values: KValuesConfig = field(default_factory=KValuesConfig)
    majority_rule: MajorityRuleConfig = field(default_factory=MajorityRuleConfig)
    binary_sequence: BinarySequenceConfig = field(default_factory=BinarySequenceConfig)
    decimal_encoding: DecimalEncodingConfig = field(default_factory=DecimalEncodingConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    exports: ExportsConfig = field(default_factory=ExportsConfig)
    notes: NotesConfig = field(default_factory=NotesConfig)

    def validate(self) -> None:
        self.project.validate()
        self.run.validate()
        self.dataset.validate()
        self.preprocessing.validate()
        self.metrics.validate()
        self.neighbors.validate()
        self.k_values.validate()
        self.majority_rule.validate()
        self.binary_sequence.validate()
        self.decimal_encoding.validate()
        self.statistics.validate()
        self.exports.validate()
        self.notes.validate()

        if self.dataset.format == "dat" and self.dataset.has_header:
            raise ValueError(
                "dataset.has_header should normally be false for DAT datasets in this project."
            )

    @property
    def enabled_metrics(self) -> list[str]:
        return list(self.metrics.enabled)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppConfig":
        d = _ensure_mapping(data, "root config")
        cfg = cls(
            project=ProjectConfig.from_dict(d.get("project")),
            run=RunConfig.from_dict(d.get("run")),
            dataset=DatasetConfig.from_dict(d.get("dataset")),
            preprocessing=PreprocessingConfig.from_dict(d.get("preprocessing")),
            metrics=MetricsConfig.from_dict(d.get("metrics")),
            neighbors=NeighborsConfig.from_dict(d.get("neighbors")),
            k_values=KValuesConfig.from_dict(d.get("k_values")),
            majority_rule=MajorityRuleConfig.from_dict(d.get("majority_rule")),
            binary_sequence=BinarySequenceConfig.from_dict(d.get("binary_sequence")),
            decimal_encoding=DecimalEncodingConfig.from_dict(d.get("decimal_encoding")),
            statistics=StatisticsConfig.from_dict(d.get("statistics")),
            exports=ExportsConfig.from_dict(d.get("exports")),
            notes=NotesConfig.from_dict(d.get("notes")),
        )
        cfg.validate()
        return cfg

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from synthetic_bit_sequence_majority_rule.algorithms.distances import compare_distance_runs
from synthetic_bit_sequence_majority_rule.algorithms.majority import (
    compare_majority_runs,
    compute_formula_based_kmax,
    compute_full_k_values,
    compute_reduced_k_values,
)
from synthetic_bit_sequence_majority_rule.algorithms.neighbors import compare_neighbor_runs
from synthetic_bit_sequence_majority_rule.algorithms.normalization import prepare_dataset_variants
from synthetic_bit_sequence_majority_rule.algorithms.statistics import (
    ComplexityTableResult,
    MembershipTableResult,
    StabilityTableResult,
    build_complexity_table,
    build_final_comparison,
    build_membership_table,
    build_sequence_statistics,
    build_stability_table,
)
from synthetic_bit_sequence_majority_rule.domain.errors import PipelineExecutionError
from synthetic_bit_sequence_majority_rule.domain.params import AppConfig
from synthetic_bit_sequence_majority_rule.domain.schema import (
    DistanceMatrixResult,
    FinalComparisonResult,
    LoadedDataset,
    MajorityMatricesResult,
    NeighborTableResult,
    StatisticsTableResult,
)
from synthetic_bit_sequence_majority_rule.io.configs import load_default_config
from synthetic_bit_sequence_majority_rule.io.loaders import try_alternate_dataset_paths


@dataclass(slots=True)
class PipelineBranchResult:
    branch_name: str
    dataset: LoadedDataset
    distance_results: dict[str, DistanceMatrixResult]
    neighbor_results: dict[str, NeighborTableResult]
    majority_results: dict[str, MajorityMatricesResult]
    statistics_results: dict[str, StatisticsTableResult]
    membership_results: dict[str, MembershipTableResult]
    stability_results: dict[str, StabilityTableResult]
    complexity_result: ComplexityTableResult
    final_comparison: FinalComparisonResult


@dataclass(slots=True)
class PipelineRunResult:
    config: AppConfig
    source_dataset: LoadedDataset
    branches: dict[str, PipelineBranchResult]
    run_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_branch(self) -> PipelineBranchResult:
        return self.branches["selected"]


def make_run_id(run_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name)
    return f"{safe_name}_{timestamp}"


def run_pipeline(config: AppConfig, run_id: str | None = None) -> PipelineRunResult:
    try:
        dataset = try_alternate_dataset_paths(config.dataset)
    except Exception as exc:
        raise PipelineExecutionError(stage="load_dataset", reason=str(exc)) from exc

    try:
        formula_kmax = compute_formula_based_kmax(dataset)
        full_k_values = compute_full_k_values(dataset, config.k_values)
        reduced_k_values = compute_reduced_k_values(
            full_k_values,
            config.k_values,
            config.binary_sequence,
        )
    except Exception as exc:
        raise PipelineExecutionError(stage="resolve_k_values", reason=str(exc)) from exc

    try:
        dataset_variants = prepare_dataset_variants(dataset, config.preprocessing.normalization)
    except Exception as exc:
        raise PipelineExecutionError(stage="normalization", reason=str(exc)) from exc

    try:
        distance_results = compare_distance_runs(
            raw_dataset=dataset_variants["raw"],
            selected_dataset=dataset_variants["selected"],
            normalized_dataset=dataset_variants.get("normalized"),
            metric_names=config.enabled_metrics,
        )
    except Exception as exc:
        raise PipelineExecutionError(stage="distances", reason=str(exc)) from exc

    try:
        neighbor_results = compare_neighbor_runs(
            raw_distance_results=distance_results["raw"],
            selected_distance_results=distance_results["selected"],
            normalized_distance_results=distance_results.get("normalized"),
            neighbors_config=config.neighbors,
        )
    except Exception as exc:
        raise PipelineExecutionError(stage="neighbors", reason=str(exc)) from exc

    try:
        majority_results = compare_majority_runs(
            raw_dataset=dataset_variants["raw"],
            raw_neighbor_results=neighbor_results["raw"],
            selected_dataset=dataset_variants["selected"],
            selected_neighbor_results=neighbor_results["selected"],
            normalized_dataset=dataset_variants.get("normalized"),
            normalized_neighbor_results=neighbor_results.get("normalized"),
            k_values_config=config.k_values,
            majority_rule_config=config.majority_rule,
            binary_sequence_config=config.binary_sequence,
            decimal_encoding_config=config.decimal_encoding,
        )
    except Exception as exc:
        raise PipelineExecutionError(stage="majority", reason=str(exc)) from exc

    branches: dict[str, PipelineBranchResult] = {}
    for branch_name, branch_majority_results in majority_results.items():
        try:
            statistics_results: dict[str, StatisticsTableResult] = {}
            membership_results: dict[str, MembershipTableResult] = {}
            stability_results: dict[str, StabilityTableResult] = {}

            for metric_name, majority_result in branch_majority_results.items():
                stats = build_sequence_statistics(majority_result, config.statistics)
                membership = build_membership_table(majority_result)
                stability = build_stability_table(
                    membership,
                    object_count=len(majority_result.object_labels),
                )
                statistics_results[metric_name] = stats
                membership_results[metric_name] = membership
                stability_results[metric_name] = stability

            final_comparison = build_final_comparison(statistics_results)
            complexity_result = build_complexity_table(stability_results)
            branches[branch_name] = PipelineBranchResult(
                branch_name=branch_name,
                dataset=dataset_variants[branch_name],
                distance_results=distance_results[branch_name],
                neighbor_results=neighbor_results[branch_name],
                majority_results=branch_majority_results,
                statistics_results=statistics_results,
                membership_results=membership_results,
                stability_results=stability_results,
                complexity_result=complexity_result,
                final_comparison=final_comparison,
            )
        except Exception as exc:
            raise PipelineExecutionError(
                stage=f"statistics:{branch_name}",
                reason=str(exc),
            ) from exc

    actual_run_id = run_id or make_run_id(config.run.run_name)
    return PipelineRunResult(
        config=config,
        source_dataset=dataset,
        branches=branches,
        run_id=actual_run_id,
        metadata={
            "formula_kmax": formula_kmax,
            "full_k_values": full_k_values,
            "reduced_k_values": reduced_k_values,
            "normalization_mode": config.preprocessing.normalization.mode,
            "metrics": list(config.enabled_metrics),
        },
    )


def run_pipeline_from_config(
    config_path: str | Path = "configs/default.yaml",
    run_id: str | None = None,
) -> PipelineRunResult:
    config = load_default_config(config_path)
    return run_pipeline(config=config, run_id=run_id)


def run_none_minmax_comparison(
    config: AppConfig,
    current_result: PipelineRunResult | None = None,
) -> PipelineRunResult:
    if (
        current_result is not None
        and current_result.config.preprocessing.normalization.mode == "minmax"
        and "normalized" in current_result.branches
    ):
        return current_result

    comparison_config = deepcopy(config)
    comparison_config.preprocessing.normalization.mode = "minmax"
    comparison_config.run.save_outputs = False
    comparison_config.validate()
    return run_pipeline(
        comparison_config,
        run_id=f"{current_result.run_id if current_result else 'comparison'}_none_vs_minmax",
    )

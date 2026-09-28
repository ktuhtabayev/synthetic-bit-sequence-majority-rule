from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from synthetic_bit_sequence_majority_rule.algorithms.distances import (
    compute_multiple_distance_matrices,
)
from synthetic_bit_sequence_majority_rule.algorithms.majority import (
    build_multiple_majority_matrices,
    compute_formula_based_kmax,
    compute_full_k_values,
    compute_reduced_k_values,
)
from synthetic_bit_sequence_majority_rule.algorithms.neighbors import (
    build_multiple_neighbor_tables,
)
from synthetic_bit_sequence_majority_rule.algorithms.normalization import prepare_dataset_variants
from synthetic_bit_sequence_majority_rule.algorithms.statistics import (
    ComplexityTableResult,
    MembershipTableResult,
    StabilityTableResult,
    build_complexity_table,
    build_final_comparison,
    build_required_result_tables,
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
from synthetic_bit_sequence_majority_rule.paths import DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)


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
    # The random suffix keeps two runs started in the same second in separate folders.
    return f"{safe_name}_{timestamp}_{uuid4().hex[:8]}"


@contextmanager
def _stage(stage: str) -> Iterator[None]:
    """Report a failure inside the block as a PipelineExecutionError for `stage`."""
    started = perf_counter()
    try:
        yield
    except Exception as exc:
        raise PipelineExecutionError(stage=stage, reason=str(exc)) from exc
    logger.debug("Stage %s finished in %.3fs.", stage, perf_counter() - started)


def _compute_branch(
    branch_name: str,
    dataset: LoadedDataset,
    config: AppConfig,
) -> PipelineBranchResult:
    """Run distances -> neighbors -> majority -> statistics for one dataset variant."""
    with _stage(f"distances:{branch_name}"):
        distance_results = compute_multiple_distance_matrices(dataset, config.enabled_metrics)

    with _stage(f"neighbors:{branch_name}"):
        neighbor_results = build_multiple_neighbor_tables(distance_results, config.neighbors)

    with _stage(f"majority:{branch_name}"):
        majority_results = build_multiple_majority_matrices(
            dataset=dataset,
            neighbor_results=neighbor_results,
            k_values_config=config.k_values,
            majority_rule_config=config.majority_rule,
            binary_sequence_config=config.binary_sequence,
            decimal_encoding_config=config.decimal_encoding,
        )

    with _stage(f"statistics:{branch_name}"):
        statistics_results: dict[str, StatisticsTableResult] = {}
        membership_results: dict[str, MembershipTableResult] = {}
        stability_results: dict[str, StabilityTableResult] = {}

        for metric_name, majority_result in majority_results.items():
            (
                statistics_results[metric_name],
                membership_results[metric_name],
                stability_results[metric_name],
            ) = build_required_result_tables(majority_result, config.statistics)

        final_comparison = build_final_comparison(statistics_results)
        complexity_result = build_complexity_table(stability_results)

    return PipelineBranchResult(
        branch_name=branch_name,
        dataset=dataset,
        distance_results=distance_results,
        neighbor_results=neighbor_results,
        majority_results=majority_results,
        statistics_results=statistics_results,
        membership_results=membership_results,
        stability_results=stability_results,
        complexity_result=complexity_result,
        final_comparison=final_comparison,
    )


def run_pipeline(
    config: AppConfig,
    run_id: str | None = None,
    *,
    reuse_branches: Mapping[str, PipelineBranchResult] | None = None,
) -> PipelineRunResult:
    """
    Load the configured dataset and compute every branch for every enabled metric.

    reuse_branches supplies branches an earlier run already computed for the
    same dataset and config; they are taken as-is instead of recomputed.
    """
    with _stage("load_dataset"):
        dataset = try_alternate_dataset_paths(config.dataset)

    with _stage("resolve_k_values"):
        formula_kmax = compute_formula_based_kmax(dataset)
        full_k_values = compute_full_k_values(dataset, config.k_values)
        reduced_k_values = compute_reduced_k_values(
            full_k_values,
            config.k_values,
            config.binary_sequence,
        )

    with _stage("normalization"):
        dataset_variants = prepare_dataset_variants(dataset, config.preprocessing.normalization)

    # Branch pairs with identical feature matrices share one computation:
    # - mode == none : selected is the raw dataset used as-is
    # - mode != none : normalized is the same dataset as selected
    normalization_mode = config.preprocessing.normalization.mode.strip().lower()
    branch_aliases = (
        {"selected": "raw"} if normalization_mode == "none" else {"normalized": "selected"}
    )
    computed_names = [name for name in dataset_variants if name not in branch_aliases]

    computed_branches: dict[str, PipelineBranchResult] = {}
    for branch_name in computed_names:
        reused = (reuse_branches or {}).get(branch_name)
        if reused is not None:
            logger.debug("Branch '%s' reused from an earlier run.", branch_name)
            computed_branches[branch_name] = reused
        else:
            computed_branches[branch_name] = _compute_branch(
                branch_name,
                dataset_variants[branch_name],
                config,
            )

    branches: dict[str, PipelineBranchResult] = {}
    for branch_name in dataset_variants:
        source = computed_branches.get(branch_name)
        if source is None:
            source = replace(
                computed_branches[branch_aliases[branch_name]],
                branch_name=branch_name,
                dataset=dataset_variants[branch_name],
            )
        branches[branch_name] = source

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
    config_path: str | Path = DEFAULT_CONFIG_PATH,
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

    # The raw branch does not depend on the normalization mode, so a run made
    # with this same config already holds it.
    reuse_branches = None
    if current_result is not None and current_result.config == config:
        reuse_branches = {"raw": current_result.branches["raw"]}

    return run_pipeline(
        comparison_config,
        run_id=f"{current_result.run_id if current_result else 'comparison'}_none_vs_minmax",
        reuse_branches=reuse_branches,
    )

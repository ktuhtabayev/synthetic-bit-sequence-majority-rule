from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synthetic_bit_sequence_majority_rule.algorithms.meta_objects import (
    NormalizationComparisonResult,
    build_normalization_comparison,
)
from synthetic_bit_sequence_majority_rule.domain.params import AppConfig
from synthetic_bit_sequence_majority_rule.io.writers import (
    write_normalization_comparison_outputs,
    write_pipeline_outputs,
)
from synthetic_bit_sequence_majority_rule.services.runner import (
    PipelineRunResult,
    run_none_minmax_comparison,
    run_pipeline,
)


@dataclass(slots=True)
class FullAnalysisResult:
    """Everything one Run produces: pipeline result, comparison, output folder."""

    pipeline: PipelineRunResult
    comparison: NormalizationComparisonResult
    output_dir: Path | None


def build_comparison_from_run(
    comparison_run: PipelineRunResult,
) -> NormalizationComparisonResult:
    return build_normalization_comparison(
        comparison_run.branches["raw"].stability_results,
        comparison_run.branches["normalized"].stability_results,
        comparison_run.branches["raw"].complexity_result,
        comparison_run.branches["normalized"].complexity_result,
    )


def write_analysis_outputs(
    pipeline: PipelineRunResult,
    comparison: NormalizationComparisonResult,
    project_root: Path | None = None,
) -> Path:
    output_root = Path(pipeline.config.run.output_root)
    if project_root is not None:
        output_root = Path(project_root) / output_root
    output_dir = write_pipeline_outputs(pipeline, output_root)
    write_normalization_comparison_outputs(comparison, output_dir)
    return output_dir


def run_full_analysis(
    config: AppConfig,
    *,
    project_root: Path | None = None,
    write_outputs: bool = True,
) -> FullAnalysisResult:
    """
    Single entry point shared by the CLI runner and the GUI:
    run the configured pipeline, derive the paired none-vs-minmax
    comparison, and optionally write all outputs.
    """
    pipeline = run_pipeline(config)
    comparison_run = run_none_minmax_comparison(config, pipeline)
    comparison = build_comparison_from_run(comparison_run)

    output_dir: Path | None = None
    if write_outputs:
        output_dir = write_analysis_outputs(pipeline, comparison, project_root)

    return FullAnalysisResult(
        pipeline=pipeline,
        comparison=comparison,
        output_dir=output_dir,
    )

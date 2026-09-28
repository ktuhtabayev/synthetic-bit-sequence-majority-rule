from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from synthetic_bit_sequence_majority_rule.algorithms.majority import (
    a_full_frame,
    a_reduced_frame,
    b_full_frame,
    b_reduced_frame,
    same_class_indicator_frame,
)
from synthetic_bit_sequence_majority_rule.domain.errors import OutputDirectoryError, OutputWriteError
from synthetic_bit_sequence_majority_rule.io.configs import config_to_dict

if TYPE_CHECKING:
    # Annotation only: the io layer must not import the services layer at runtime.
    from synthetic_bit_sequence_majority_rule.algorithms.meta_objects import (
        NormalizationComparisonResult,
    )
    from synthetic_bit_sequence_majority_rule.services.runner import (
        PipelineBranchResult,
        PipelineRunResult,
    )


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise OutputDirectoryError(out, str(exc)) from exc
    return out


def prune_full_run_outputs(
    output_root: str | Path,
    *,
    keep_latest: int = 3,
    run_prefix: str = "default_run_",
) -> list[Path]:
    """
    Remove older full pipeline run folders while preserving quick/manual outputs.

    Only directories directly under output_root whose names start with
    default_run_ are considered. This intentionally leaves quick_* folders and
    any other project resources untouched.
    """
    root = Path(output_root)
    if keep_latest < 0:
        raise ValueError("keep_latest must be non-negative.")
    if not root.exists():
        return []

    root_resolved = root.resolve()
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith(run_prefix)
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    removed: list[Path] = []
    for path in candidates[keep_latest:]:
        resolved = path.resolve()
        if root_resolved not in resolved.parents:
            raise OutputDirectoryError(path, "Refusing to remove a path outside the output root.")
        try:
            shutil.rmtree(resolved)
        except Exception as exc:
            raise OutputDirectoryError(path, str(exc)) from exc
        removed.append(path)

    return removed


def _write_text(path: Path, content: str) -> None:
    try:
        path.write_text(content, encoding="utf-8")
    except Exception as exc:
        raise OutputWriteError(path, str(exc)) from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        raise OutputWriteError(path, str(exc)) from exc


def _write_frame(path: Path, frame: pd.DataFrame, *, index: bool = True) -> None:
    try:
        frame.to_csv(path, index=index, encoding="utf-8")
    except Exception as exc:
        raise OutputWriteError(path, str(exc)) from exc


def _write_frame_no_index(path: Path, frame: pd.DataFrame) -> None:
    _write_frame(path, frame, index=False)


def _sheet_name(name: str) -> str:
    invalid_chars = set("[]:*?/\\")
    cleaned = "".join(ch if ch not in invalid_chars else "_" for ch in name)
    return cleaned[:31] or "Sheet"


def _write_branch_excel(path: Path, branch: PipelineBranchResult) -> None:
    try:
        with pd.ExcelWriter(path) as writer:
            branch.dataset.to_frame().to_excel(writer, sheet_name="Dataset", index=False)
            branch.final_comparison.to_frame().to_excel(writer, sheet_name="Final", index=False)
            branch.complexity_result.to_frame().to_excel(
                writer,
                sheet_name="Complexity",
                index=False,
            )

            for metric_name, stats_result in branch.statistics_results.items():
                stats_result.to_frame().to_excel(
                    writer,
                    sheet_name=_sheet_name(f"Stats_{metric_name}"),
                    index=False,
                )
            for metric_name, stability_result in branch.stability_results.items():
                stability_result.to_frame().to_excel(
                    writer,
                    sheet_name=_sheet_name(f"Stability_{metric_name}"),
                    index=False,
                )
            for metric_name, majority_result in branch.majority_results.items():
                b_reduced_frame(majority_result).to_excel(
                    writer,
                    sheet_name=_sheet_name(f"B_reduced_{metric_name}"),
                    index=False,
                )
    except Exception as exc:
        raise OutputWriteError(path, str(exc)) from exc


def _stats_payload(result: PipelineRunResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "metadata": dict(result.metadata),
        "config": config_to_dict(result.config),
        "branches": {
            branch_name: {
                "dataset_source_path": branch.dataset.source_path,
                "dataset_source_format": branch.dataset.source_format,
                "n_objects": branch.dataset.n_objects,
                "n_features": branch.dataset.n_features,
                "class_counts": branch.dataset.class_counts,
                "metrics": list(branch.majority_results.keys()),
            }
            for branch_name, branch in result.branches.items()
        },
    }


def write_pipeline_outputs(
    result: PipelineRunResult,
    output_root: str | Path | None = None,
    *,
    prune_old_runs: bool = True,
) -> Path:
    root = Path(output_root) if output_root is not None else Path(result.config.run.output_root)
    run_dir = ensure_output_dir(root / result.run_id)

    _write_text(run_dir / "dataset_path.txt", str(result.source_dataset.source_path or ""))
    _write_json(run_dir / "run_info.json", _stats_payload(result))

    for branch_name, branch in result.branches.items():
        branch_dir = ensure_output_dir(run_dir / branch_name)

        _write_frame_no_index(branch_dir / "dataset.csv", branch.dataset.to_frame())

        if result.config.exports.distance_matrices:
            distance_dir = ensure_output_dir(branch_dir / "distances")
            for metric_name, distance_result in branch.distance_results.items():
                _write_frame(distance_dir / f"{metric_name}.csv", distance_result.to_frame())

        if result.config.exports.neighbors_tables:
            neighbors_dir = ensure_output_dir(branch_dir / "neighbors")
            for metric_name, neighbor_result in branch.neighbor_results.items():
                _write_frame(neighbors_dir / f"{metric_name}_labels.csv", neighbor_result.labels_frame())
                _write_frame(neighbors_dir / f"{metric_name}_distances.csv", neighbor_result.distances_frame())

        majority_dir = ensure_output_dir(branch_dir / "majority")
        for metric_name, majority_result in branch.majority_results.items():
            _write_frame_no_index(
                majority_dir / f"{metric_name}_same_class_indicators.csv",
                same_class_indicator_frame(majority_result),
            )
            if result.config.exports.full_a_matrices:
                _write_frame_no_index(majority_dir / f"{metric_name}_a_full.csv", a_full_frame(majority_result))
            if result.config.exports.reduced_a_matrices:
                _write_frame_no_index(majority_dir / f"{metric_name}_a_reduced.csv", a_reduced_frame(majority_result))
            if result.config.exports.full_b_matrices:
                _write_frame_no_index(majority_dir / f"{metric_name}_b_full.csv", b_full_frame(majority_result))
            if result.config.exports.reduced_b_matrices:
                _write_frame_no_index(majority_dir / f"{metric_name}_b_reduced.csv", b_reduced_frame(majority_result))

        if result.config.exports.stats_tables:
            stats_dir = ensure_output_dir(branch_dir / "statistics")
            for metric_name, stats_result in branch.statistics_results.items():
                _write_frame_no_index(stats_dir / f"{metric_name}_sequence_statistics.csv", stats_result.to_frame())
            for metric_name, membership_result in branch.membership_results.items():
                _write_frame_no_index(stats_dir / f"{metric_name}_membership.csv", membership_result.to_frame())
            for metric_name, stability_result in branch.stability_results.items():
                _write_frame_no_index(stats_dir / f"{metric_name}_stability.csv", stability_result.to_frame())
            _write_frame_no_index(
                stats_dir / "complexity.csv",
                branch.complexity_result.to_frame(),
            )

        if result.config.exports.final_comparison:
            _write_frame_no_index(branch_dir / "final_comparison.csv", branch.final_comparison.to_frame())

        if result.config.exports.excel:
            _write_branch_excel(branch_dir / "summary.xlsx", branch)

    if prune_old_runs and result.run_id.startswith("default_run_"):
        prune_full_run_outputs(root, keep_latest=3)

    return run_dir


def write_normalization_comparison_outputs(
    comparison: NormalizationComparisonResult,
    run_dir: str | Path,
) -> Path:
    comparison_dir = ensure_output_dir(Path(run_dir) / "normalization_comparison")
    _write_frame_no_index(comparison_dir / "complexity_comparison.csv", comparison.summary_frame)
    _write_frame_no_index(comparison_dir / "meta_objects.csv", comparison.meta_objects.frame)
    if comparison.pca_2d.available:
        _write_frame_no_index(comparison_dir / "pca_2d.csv", comparison.pca_2d.frame)
    if comparison.pca_3d.available:
        _write_frame_no_index(comparison_dir / "pca_3d.csv", comparison.pca_3d.frame)
    _write_json(
        comparison_dir / "comparison_info.json",
        {
            "baseline_normalization": "none",
            "comparison_normalization": "minmax",
            "delta_formula": "DeltaComplexity = ComplexityMinMax - ComplexityNone",
            "k_mapping": comparison.meta_objects.k_mapping,
            "pca_2d_explained_variance_ratio": comparison.pca_2d.explained_variance_ratio,
            "pca_3d_explained_variance_ratio": comparison.pca_3d.explained_variance_ratio,
        },
    )
    return comparison_dir

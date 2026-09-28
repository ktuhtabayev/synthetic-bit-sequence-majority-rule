"""
Command-line runner for the full analysis.

With no options it runs configs/default.yaml exactly as configured. Options
override the dataset, normalization, metrics, and output location for one run
without editing the config file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from synthetic_bit_sequence_majority_rule.domain.errors import SyntheticBitSequenceError
from synthetic_bit_sequence_majority_rule.domain.params import AppConfig
from synthetic_bit_sequence_majority_rule.io.configs import (
    apply_dataset_selection,
    find_dataset_preset,
    load_dataset_catalog,
    load_default_config,
)
from synthetic_bit_sequence_majority_rule.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT

AVAILABLE_METRICS = ("euclidean", "chebyshev", "canberra", "manhattan")
NORMALIZATION_MODES = ("none", "minmax", "zscore")

logger = logging.getLogger(__name__)


def _metric_list(text: str) -> list[str]:
    metrics = [item.strip().lower() for item in text.split(",") if item.strip()]
    unknown = [metric for metric in metrics if metric not in AVAILABLE_METRICS]
    if unknown or not metrics:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated subset of {', '.join(AVAILABLE_METRICS)}"
        )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthetic-bit-sequence",
        description=(
            "Build synthetic bit-sequence features with the nearest-neighbor majority "
            "rule and compare metrics. Without options, runs the config file as-is."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="config file (default: configs/default.yaml)",
    )
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument("--preset", help="dataset_catalog entry to run, e.g. ionosfera_csv")
    dataset.add_argument("--dataset", type=Path, help="dataset file to run (csv, dat, xlsx, xls)")
    parser.add_argument("--normalization", choices=NORMALIZATION_MODES)
    parser.add_argument(
        "--metrics",
        type=_metric_list,
        help=f"comma-separated metrics, e.g. euclidean,canberra (from {', '.join(AVAILABLE_METRICS)})",
    )
    parser.add_argument("--output-root", type=Path, help="folder that receives the run folder")
    parser.add_argument("--no-save", action="store_true", help="compute and print only; write nothing")
    parser.add_argument("--list-presets", action="store_true", help="list dataset presets and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="log stage timings and tracebacks")
    return parser


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def config_from_args(args: argparse.Namespace, project_root: Path = PROJECT_ROOT) -> AppConfig:
    """
    The config file with the command-line overrides applied.

    Paths given on the command line are relative to the current folder; paths
    inside the config file are relative to project_root.
    """
    config = load_default_config(args.config)

    if args.preset is not None or args.dataset is not None:
        catalog = load_dataset_catalog(args.config)
        if args.preset is not None:
            if args.preset not in catalog:
                raise SystemExit(
                    f"Unknown preset '{args.preset}'. Available: {', '.join(catalog) or 'none'}"
                )
            preset_name: str | None = args.preset
            dataset_path: Path = project_root / str(catalog[args.preset]["path"])
        else:
            dataset_path = args.dataset.resolve()
            preset_name = find_dataset_preset(catalog, dataset_path, project_root)
        apply_dataset_selection(
            config,
            dataset_path,
            project_root=project_root,
            preset=catalog[preset_name] if preset_name is not None else None,
        )

    if args.normalization is not None:
        config.preprocessing.normalization.mode = args.normalization
    if args.metrics is not None:
        config.metrics.enabled = args.metrics
    if args.output_root is not None:
        config.run.output_root = args.output_root.resolve()
    if args.no_save:
        config.run.save_outputs = False

    config.validate()
    return config


def _print_presets(config_path: Path) -> None:
    catalog = load_dataset_catalog(config_path)
    if not catalog:
        print("No dataset_catalog presets in", config_path)
        return
    width = max(len(name) for name in catalog)
    for name, entry in catalog.items():
        print(f"{name:<{width}}  {entry['path']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Resolve command-line paths against the folder the command was typed in,
    # before the run switches to the project root.
    args.config = args.config.resolve()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.list_presets:
        _print_presets(args.config)
        return 0

    # Imported here so --help and --list-presets stay fast.
    from synthetic_bit_sequence_majority_rule.services.analysis import run_full_analysis
    from synthetic_bit_sequence_majority_rule.services.report_builder import (
        build_run_summary_frame,
    )

    print("=" * 80)
    print("SYNTHETIC BIT SEQUENCE MAJORITY RULE")
    print("=" * 80)
    print(f"Config: {args.config}")

    try:
        # Paths in the config file are relative to the project root, so the run
        # works the same whichever folder it is launched from.
        with _working_directory(PROJECT_ROOT):
            config = config_from_args(args)
            analysis = run_full_analysis(
                config,
                project_root=PROJECT_ROOT,
                write_outputs=config.run.save_outputs,
            )
    except SyntheticBitSequenceError as exc:
        sys.stdout.flush()  # keep the banner ahead of the error when both are piped
        if args.verbose:
            logger.exception("Run failed.")
        print(f"\nRun failed: {exc}", file=sys.stderr)
        return 1

    summary = build_run_summary_frame(analysis.pipeline)

    print("\nRun summary")
    print("-" * 80)
    print(summary.to_string(index=False))

    if analysis.output_dir is not None:
        print("\nOutputs saved")
        print("-" * 80)
        print(analysis.output_dir)

    print("\nFinal comparison for selected branch")
    print("-" * 80)
    print(analysis.pipeline.selected_branch.final_comparison.to_frame().to_string(index=False))

    print("\nFinished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

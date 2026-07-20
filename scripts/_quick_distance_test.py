from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ============================================================
# Make src/ importable when running this script directly
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from synthetic_bit_sequence_majority_rule.io.configs import load_default_config  # noqa: E402
from synthetic_bit_sequence_majority_rule.io.loaders import try_alternate_dataset_paths  # noqa: E402
from synthetic_bit_sequence_majority_rule.algorithms.normalization import (  # noqa: E402
    prepare_dataset_variants,
)
from synthetic_bit_sequence_majority_rule.algorithms.distances import (  # noqa: E402
    build_distance_summary,
    compare_distance_runs,
    distance_result_to_frame,
)


# ============================================================
# Helpers
# ============================================================

def print_separator(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_frame(df: pd.DataFrame, title: str, decimals: int = 6) -> None:
    print(f"\n{title}")
    print("-" * 80)
    df_to_print = df.copy()
    numeric_cols = df_to_print.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        df_to_print[numeric_cols] = df_to_print[numeric_cols].round(decimals)
    print(df_to_print.to_string())


def make_run_id(prefix: str = "quick_distance_test") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def round_numeric_frame(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        out[numeric_cols] = out[numeric_cols].round(decimals)
    return out


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_dataframe(path: Path, df: pd.DataFrame, decimals: int = 6) -> None:
    df_to_save = round_numeric_frame(df, decimals)
    df_to_save.to_csv(path, index=True, encoding="utf-8")


def summary_to_dict(summary: Any) -> dict[str, Any]:
    return {
        "metric_name": summary.metric_name,
        "n_objects": summary.n_objects,
        "normalized": summary.normalized,
        "min_distance": summary.min_distance,
        "max_distance": summary.max_distance,
        "diagonal_zero": summary.diagonal_zero,
        "symmetric": summary.symmetric,
        "metadata": summary.metadata,
    }


def load_quick_test_settings() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    quick_tests = raw.get("quick_tests", {}) or {}
    distances = quick_tests.get("distances", {}) or {}
    metrics = raw.get("metrics", {}) or {}

    return {
        "print_tables": bool(quick_tests.get("print_tables", True)),
        "round_decimals": int(quick_tests.get("round_decimals", 6)),
        "save_outputs": bool(distances.get("save_outputs", True)),
        "output_subdir": str(distances.get("output_subdir", "quick_distance_test")),
        "metrics_to_test": list(metrics.get("enabled", ["euclidean", "chebyshev", "canberra"])),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    settings = load_quick_test_settings()
    print_tables = settings["print_tables"]
    round_decimals = settings["round_decimals"]
    save_outputs = settings["save_outputs"]
    output_subdir = settings["output_subdir"]
    metrics_to_test = settings["metrics_to_test"]

    print_separator("QUICK DISTANCE TEST")

    # ------------------------------------------------------------
    # 1) Load config and dataset
    # ------------------------------------------------------------
    print("\n[1] Loading config and dataset ...")
    cfg = load_default_config(PROJECT_ROOT / "configs" / "default.yaml")
    dataset = try_alternate_dataset_paths(cfg.dataset)
    print("OK: dataset loaded successfully")
    print(f"Source path         : {dataset.source_path}")
    print(f"Source format       : {dataset.source_format}")
    print(f"Objects             : {dataset.n_objects}")
    print(f"Features            : {dataset.n_features}")
    print(f"Classes             : {dataset.classes}")
    print(f"Class counts        : {dataset.class_counts}")
    print(f"Normalization mode  : {cfg.preprocessing.normalization.mode}")
    print(f"Metrics to test     : {metrics_to_test}")

    # ------------------------------------------------------------
    # 2) Prepare raw / selected / normalized dataset variants
    # ------------------------------------------------------------
    print("\n[2] Preparing raw / selected / normalized dataset variants ...")
    variants = prepare_dataset_variants(dataset, cfg.preprocessing.normalization)

    raw_dataset = variants["raw"]
    selected_dataset = variants["selected"]
    normalized_dataset = variants.get("normalized")

    print("OK: dataset variants prepared")
    print(f"Has normalized branch : {'yes' if normalized_dataset is not None else 'no'}")

    # ------------------------------------------------------------
    # 3) Compute distances
    # ------------------------------------------------------------
    print("\n[3] Computing distance matrices ...")
    results = compare_distance_runs(
        raw_dataset=raw_dataset,
        selected_dataset=selected_dataset,
        normalized_dataset=normalized_dataset,
        metric_names=metrics_to_test,
    )
    print("OK: distance matrices computed")

    # ------------------------------------------------------------
    # 4) Print summaries and optional tables
    # ------------------------------------------------------------
    for branch_name, branch_results in results.items():
        print_separator(f"BRANCH: {branch_name.upper()}")

        for metric_name, result in branch_results.items():
            summary = build_distance_summary(result)

            print(f"\nMetric: {metric_name}")
            print("-" * 80)
            print(f"Objects       : {summary.n_objects}")
            print(f"Normalized    : {summary.normalized}")
            print(f"Min distance  : {summary.min_distance:.{round_decimals}f}")
            print(f"Max distance  : {summary.max_distance:.{round_decimals}f}")
            print(f"Diagonal zero : {summary.diagonal_zero}")
            print(f"Symmetric     : {summary.symmetric}")
            print(f"Metadata      : {summary.metadata}")

            if print_tables:
                df = distance_result_to_frame(result)
                print_frame(df, f"{branch_name.upper()} - {metric_name.upper()} DISTANCE MATRIX", round_decimals)

    # ------------------------------------------------------------
    # 5) Save outputs
    # ------------------------------------------------------------
    if save_outputs:
        run_id = make_run_id()
        run_dir = ensure_dir(PROJECT_ROOT / "outputs" / "runs" / output_subdir / run_id)

        write_text(run_dir / "dataset_path.txt", str(dataset.source_path or ""))
        write_text(run_dir / "normalization_mode.txt", cfg.preprocessing.normalization.mode)
        write_text(run_dir / "metrics.txt", ", ".join(metrics_to_test))

        master_payload: dict[str, Any] = {
            "dataset_source_path": dataset.source_path,
            "dataset_source_format": dataset.source_format,
            "n_objects": dataset.n_objects,
            "n_features": dataset.n_features,
            "classes": dataset.classes,
            "class_counts": dataset.class_counts,
            "normalization_mode": cfg.preprocessing.normalization.mode,
            "metrics": metrics_to_test,
            "branch_rule": {
                "raw": "original dataset",
                "selected": "dataset actually selected by default.yaml normalization mode",
                "normalized": "explicit normalized dataset branch when normalization mode is not none",
            },
            "branches": {},
        }

        for branch_name, branch_results in results.items():
            branch_dir = ensure_dir(run_dir / branch_name)
            master_payload["branches"][branch_name] = {}

            for metric_name, result in branch_results.items():
                df = distance_result_to_frame(result)
                save_dataframe(branch_dir / f"distance_{metric_name}.csv", df, round_decimals)

                summary = build_distance_summary(result)
                master_payload["branches"][branch_name][metric_name] = summary_to_dict(summary)

        write_json(run_dir / "run_info.json", master_payload)

        print("\n[4] Outputs saved")
        print("-" * 80)
        print(f"Run directory : {run_dir}")

    print_separator("QUICK DISTANCE TEST FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
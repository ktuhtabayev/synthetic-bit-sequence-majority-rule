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
    compare_distance_runs,
)
from synthetic_bit_sequence_majority_rule.algorithms.neighbors import (  # noqa: E402
    build_neighbor_summary,
    compare_neighbor_runs,
    neighbor_combined_frame,
    neighbor_distances_to_frame,
    neighbor_labels_to_frame,
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
    print(df_to_print.to_string(index=False))


def make_run_id(prefix: str = "quick_neighbors_test") -> str:
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
    df_to_save.to_csv(path, index=False, encoding="utf-8")


def summary_to_dict(summary: Any) -> dict[str, Any]:
    return {
        "metric_name": summary.metric_name,
        "n_objects": summary.n_objects,
        "neighbor_count": summary.neighbor_count,
        "tie_break_rule": summary.tie_break_rule,
        "exclude_self": summary.exclude_self,
        "normalized": summary.normalized,
        "metadata": summary.metadata,
    }


def load_quick_test_settings() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    quick_tests = raw.get("quick_tests", {}) or {}
    neighbors = quick_tests.get("neighbors", {}) or {}
    metrics = raw.get("metrics", {}) or {}

    return {
        "print_tables": bool(quick_tests.get("print_tables", True)),
        "round_decimals": int(quick_tests.get("round_decimals", 6)),
        "save_outputs": bool(neighbors.get("save_outputs", True)),
        "output_subdir": str(neighbors.get("output_subdir", "quick_neighbors_test")),
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

    print_separator("QUICK NEIGHBORS TEST")

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
    print(f"Valid neighbors     : {dataset.valid_neighbor_count}")
    print(f"Normalization mode  : {cfg.preprocessing.normalization.mode}")
    print(f"Metrics to test     : {metrics_to_test}")
    print(f"Tie break rule      : {cfg.neighbors.tie_break_rule}")
    print(f"Exclude self        : {cfg.neighbors.exclude_self}")

    # ------------------------------------------------------------
    # 2) Prepare dataset variants
    # ------------------------------------------------------------
    print("\n[2] Preparing raw / selected / normalized dataset variants ...")
    variants = prepare_dataset_variants(dataset, cfg.preprocessing.normalization)

    raw_dataset = variants["raw"]
    selected_dataset = variants["selected"]
    normalized_dataset = variants.get("normalized")

    print("OK: dataset variants prepared")
    print(f"Has normalized branch : {'yes' if normalized_dataset is not None else 'no'}")

    # ------------------------------------------------------------
    # 3) Compute distance results first
    # ------------------------------------------------------------
    print("\n[3] Computing distance matrices for neighbor ordering ...")
    distance_results = compare_distance_runs(
        raw_dataset=raw_dataset,
        selected_dataset=selected_dataset,
        normalized_dataset=normalized_dataset,
        metric_names=metrics_to_test,
    )
    print("OK: distance matrices computed")

    # ------------------------------------------------------------
    # 4) Build neighbor tables
    # ------------------------------------------------------------
    print("\n[4] Building neighbor tables ...")
    neighbor_results = compare_neighbor_runs(
        raw_distance_results=distance_results["raw"],
        selected_distance_results=distance_results["selected"],
        normalized_distance_results=distance_results.get("normalized"),
        neighbors_config=cfg.neighbors,
    )
    print("OK: neighbor tables constructed")

    # ------------------------------------------------------------
    # 5) Print summaries and optional tables
    # ------------------------------------------------------------
    for branch_name, branch_results in neighbor_results.items():
        print_separator(f"BRANCH: {branch_name.upper()}")

        for metric_name, result in branch_results.items():
            summary = build_neighbor_summary(result)

            print(f"\nMetric: {metric_name}")
            print("-" * 80)
            print(f"Objects         : {summary.n_objects}")
            print(f"Neighbor count  : {summary.neighbor_count}")
            print(f"Tie break rule  : {summary.tie_break_rule}")
            print(f"Exclude self    : {summary.exclude_self}")
            print(f"Normalized      : {summary.normalized}")
            print(f"Metadata        : {summary.metadata}")

            if print_tables:
                combined_df = neighbor_combined_frame(result)
                print_frame(
                    combined_df,
                    f"{branch_name.upper()} - {metric_name.upper()} NEIGHBOR TABLE",
                    round_decimals,
                )

    # ------------------------------------------------------------
    # 6) Save outputs
    # ------------------------------------------------------------
    if save_outputs:
        run_id = make_run_id()
        run_dir = ensure_dir(PROJECT_ROOT / "outputs" / "runs" / output_subdir / run_id)

        write_text(run_dir / "dataset_path.txt", str(dataset.source_path or ""))
        write_text(run_dir / "normalization_mode.txt", cfg.preprocessing.normalization.mode)
        write_text(run_dir / "metrics.txt", ", ".join(metrics_to_test))
        write_text(run_dir / "tie_break_rule.txt", cfg.neighbors.tie_break_rule)

        master_payload: dict[str, Any] = {
            "dataset_source_path": dataset.source_path,
            "dataset_source_format": dataset.source_format,
            "n_objects": dataset.n_objects,
            "n_features": dataset.n_features,
            "classes": dataset.classes,
            "class_counts": dataset.class_counts,
            "valid_neighbors": dataset.valid_neighbor_count,
            "normalization_mode": cfg.preprocessing.normalization.mode,
            "metrics": metrics_to_test,
            "tie_break_rule": cfg.neighbors.tie_break_rule,
            "exclude_self": cfg.neighbors.exclude_self,
            "branch_rule": {
                "raw": "original dataset",
                "selected": "dataset actually selected by default.yaml normalization mode",
                "normalized": "explicit normalized dataset branch when normalization mode is not none",
            },
            "note": "Neighbor ordering is independent of the later formula-based k_max rule. It always builds the full ordered list of m-1 neighbors.",
            "branches": {},
        }

        for branch_name, branch_results in neighbor_results.items():
            branch_dir = ensure_dir(run_dir / branch_name)
            master_payload["branches"][branch_name] = {}

            for metric_name, result in branch_results.items():
                labels_df = neighbor_labels_to_frame(result)
                distances_df = neighbor_distances_to_frame(result)
                combined_df = neighbor_combined_frame(result)

                save_dataframe(branch_dir / f"neighbors_{metric_name}_labels.csv", labels_df, round_decimals)
                save_dataframe(branch_dir / f"neighbors_{metric_name}_distances.csv", distances_df, round_decimals)
                save_dataframe(branch_dir / f"neighbors_{metric_name}_combined.csv", combined_df, round_decimals)

                summary = build_neighbor_summary(result)
                master_payload["branches"][branch_name][metric_name] = summary_to_dict(summary)

        write_json(run_dir / "run_info.json", master_payload)

        print("\n[5] Outputs saved")
        print("-" * 80)
        print(f"Run directory : {run_dir}")

    print_separator("QUICK NEIGHBORS TEST FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
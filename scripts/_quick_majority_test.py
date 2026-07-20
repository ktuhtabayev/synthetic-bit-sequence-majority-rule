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
    compare_neighbor_runs,
)
from synthetic_bit_sequence_majority_rule.algorithms.majority import (  # noqa: E402
    a_full_frame,
    a_reduced_frame,
    b_full_frame,
    b_reduced_frame,
    build_majority_summary,
    compare_majority_runs,
    compute_formula_based_kmax,
    compute_full_k_values,
    compute_reduced_k_values,
    same_class_indicator_frame,
)
from synthetic_bit_sequence_majority_rule.algorithms.statistics import (  # noqa: E402
    build_final_comparison,
    build_membership_table,
    build_multiple_sequence_statistics,
    build_stability_table,
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


def make_run_id(prefix: str = "quick_majority_test") -> str:
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
        "full_k_values": summary.full_k_values,
        "reduced_k_values": summary.reduced_k_values,
        "normalization_mode": summary.normalization_mode,
        "with_normalization": summary.with_normalization,
        "metadata": summary.metadata,
    }


def load_quick_test_settings() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    quick_tests = raw.get("quick_tests", {}) or {}
    majority = quick_tests.get("majority", {}) or {}
    metrics = raw.get("metrics", {}) or {}

    return {
        "print_tables": bool(quick_tests.get("print_tables", True)),
        "round_decimals": int(quick_tests.get("round_decimals", 6)),
        "save_outputs": bool(majority.get("save_outputs", True)),
        "output_subdir": str(majority.get("output_subdir", "quick_majority_test")),
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

    print_separator("QUICK MAJORITY TEST")

    # ------------------------------------------------------------
    # 1) Load config and dataset
    # ------------------------------------------------------------
    print("\n[1] Loading config and dataset ...")
    cfg = load_default_config(PROJECT_ROOT / "configs" / "default.yaml")
    dataset = try_alternate_dataset_paths(cfg.dataset)

    formula_kmax = compute_formula_based_kmax(dataset)
    full_k_values = compute_full_k_values(dataset, cfg.k_values)
    reduced_k_values = compute_reduced_k_values(full_k_values, cfg.k_values, cfg.binary_sequence)

    print("OK: dataset loaded successfully")
    print(f"Source path         : {dataset.source_path}")
    print(f"Source format       : {dataset.source_format}")
    print(f"Objects             : {dataset.n_objects}")
    print(f"Features            : {dataset.n_features}")
    print(f"Classes             : {dataset.classes}")
    print(f"Class counts        : {dataset.class_counts}")
    print(f"Normalization mode  : {cfg.preprocessing.normalization.mode}")
    print(f"Metrics to test     : {metrics_to_test}")
    print(f"Formula-based k_max : {formula_kmax}")
    print(f"Full k values    : {full_k_values}")
    print(f"Reduced k values : {reduced_k_values}")

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
    # 3) Distances
    # ------------------------------------------------------------
    print("\n[3] Computing distance matrices ...")
    distance_results = compare_distance_runs(
        raw_dataset=raw_dataset,
        selected_dataset=selected_dataset,
        normalized_dataset=normalized_dataset,
        metric_names=metrics_to_test,
    )
    print("OK: distance matrices computed")

    # ------------------------------------------------------------
    # 4) Neighbors
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
    # 5) Majority
    # ------------------------------------------------------------
    print("\n[5] Building majority-rule matrices ...")
    majority_results = compare_majority_runs(
        raw_dataset=raw_dataset,
        raw_neighbor_results=neighbor_results["raw"],
        selected_dataset=selected_dataset,
        selected_neighbor_results=neighbor_results["selected"],
        normalized_dataset=normalized_dataset,
        normalized_neighbor_results=neighbor_results.get("normalized"),
        k_values_config=cfg.k_values,
        majority_rule_config=cfg.majority_rule,
        binary_sequence_config=cfg.binary_sequence,
        decimal_encoding_config=cfg.decimal_encoding,
    )
    print("OK: majority-rule matrices constructed")

    # ------------------------------------------------------------
    # 6) Statistics / required-result tables
    # ------------------------------------------------------------
    statistics_results: dict[str, Any] = {}
    final_comparison_results: dict[str, Any] = {}

    if cfg.statistics.enabled:
        print("\n[6] Building sequence statistics and final comparisons ...")
        for branch_name, branch_results in majority_results.items():
            branch_stats = build_multiple_sequence_statistics(
                majority_results=branch_results,
                statistics_config=cfg.statistics,
            )
            statistics_results[branch_name] = branch_stats
            final_comparison_results[branch_name] = build_final_comparison(branch_stats)
        print("OK: statistics and final comparisons constructed")

    # ------------------------------------------------------------
    # 7) Print summaries and optional tables
    # ------------------------------------------------------------
    for branch_name, branch_results in majority_results.items():
        print_separator(f"BRANCH: {branch_name.upper()}")

        for metric_name, result in branch_results.items():
            summary = build_majority_summary(result)

            print(f"\nMetric: {metric_name}")
            print("-" * 80)
            print(f"Objects              : {summary.n_objects}")
            print(f"Full k values        : {summary.full_k_values}")
            print(f"Reduced k values     : {summary.reduced_k_values}")
            print(f"Normalization mode   : {summary.normalization_mode}")
            print(f"With normalization   : {summary.with_normalization}")
            print(f"Metadata             : {summary.metadata}")

            if print_tables:
                print_frame(
                    same_class_indicator_frame(result),
                    f"{branch_name.upper()} - {metric_name.upper()} SAME-CLASS INDICATORS",
                    round_decimals,
                )
                print_frame(
                    a_full_frame(result),
                    f"{branch_name.upper()} - {metric_name.upper()} A FULL",
                    round_decimals,
                )
                print_frame(
                    a_reduced_frame(result),
                    f"{branch_name.upper()} - {metric_name.upper()} A REDUCED",
                    round_decimals,
                )
                print_frame(
                    b_full_frame(result),
                    f"{branch_name.upper()} - {metric_name.upper()} B FULL",
                    round_decimals,
                )
                print_frame(
                    b_reduced_frame(result),
                    f"{branch_name.upper()} - {metric_name.upper()} B REDUCED",
                    round_decimals,
                )
                if cfg.statistics.enabled:
                    print_frame(
                        statistics_results[branch_name][metric_name].to_frame(),
                        f"{branch_name.upper()} - {metric_name.upper()} BINARY SEQUENCE STATISTICS",
                        round_decimals,
                    )
                    membership = build_membership_table(result)
                    stability = build_stability_table(
                        membership,
                        object_count=len(result.object_labels),
                    )
                    print_frame(
                        membership.to_frame(),
                        f"{branch_name.upper()} - {metric_name.upper()} MEMBERSHIP",
                        round_decimals,
                    )
                    print_frame(
                        stability.to_frame(),
                        f"{branch_name.upper()} - {metric_name.upper()} STABILITY",
                        round_decimals,
                    )

        if print_tables and cfg.statistics.enabled:
            print_frame(
                final_comparison_results[branch_name].to_frame(),
                f"{branch_name.upper()} - FINAL COMPARISON",
                round_decimals,
            )

    # ------------------------------------------------------------
    # 8) Save outputs
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
            "formula_kmax": formula_kmax,
            "full_k_values": full_k_values,
            "reduced_k_values": reduced_k_values,
            "branch_rule": {
                "raw": "original dataset",
                "selected": "dataset actually selected by default.yaml normalization mode",
                "normalized": "explicit normalized dataset branch when normalization mode is not none",
            },
            "note": "Same-class indicators are built for the full ordered neighbor list, while A/B full-reduced tables follow the new dynamic formula-based k rule.",
            "statistics_enabled": cfg.statistics.enabled,
            "branches": {},
        }

        for branch_name, branch_results in majority_results.items():
            branch_dir = ensure_dir(run_dir / branch_name)
            master_payload["branches"][branch_name] = {}

            for metric_name, result in branch_results.items():
                save_dataframe(
                    branch_dir / f"{metric_name}_same_class_indicators.csv",
                    same_class_indicator_frame(result),
                    round_decimals,
                )
                save_dataframe(
                    branch_dir / f"{metric_name}_a_full.csv",
                    a_full_frame(result),
                    round_decimals,
                )
                save_dataframe(
                    branch_dir / f"{metric_name}_a_reduced.csv",
                    a_reduced_frame(result),
                    round_decimals,
                )
                save_dataframe(
                    branch_dir / f"{metric_name}_b_full.csv",
                    b_full_frame(result),
                    round_decimals,
                )
                save_dataframe(
                    branch_dir / f"{metric_name}_b_reduced.csv",
                    b_reduced_frame(result),
                    round_decimals,
                )

                if cfg.statistics.enabled:
                    stats_result = statistics_results[branch_name][metric_name]
                    save_dataframe(
                        branch_dir / f"{metric_name}_binary_sequence_statistics.csv",
                        stats_result.to_frame(),
                        round_decimals,
                    )

                    membership = build_membership_table(result)
                    stability = build_stability_table(
                        membership,
                        object_count=len(result.object_labels),
                    )
                    save_dataframe(
                        branch_dir / f"{metric_name}_membership.csv",
                        membership.to_frame(),
                        round_decimals,
                    )
                    save_dataframe(
                        branch_dir / f"{metric_name}_stability.csv",
                        stability.to_frame(),
                        round_decimals,
                    )

                summary = build_majority_summary(result)
                metric_payload = summary_to_dict(summary)
                if cfg.statistics.enabled:
                    metric_payload["statistics_rows"] = (
                        statistics_results[branch_name][metric_name]
                        .to_frame()
                        .to_dict(orient="records")
                    )
                master_payload["branches"][branch_name][metric_name] = metric_payload

            if cfg.statistics.enabled:
                final_frame = final_comparison_results[branch_name].to_frame()
                save_dataframe(
                    branch_dir / "final_comparison.csv",
                    final_frame,
                    round_decimals,
                )
                master_payload["branches"][branch_name]["final_comparison"] = (
                    final_frame.to_dict(orient="records")
                )

        write_json(run_dir / "run_info.json", master_payload)

        print("\n[8] Outputs saved")
        print("-" * 80)
        print(f"Run directory : {run_dir}")

    print_separator("QUICK MAJORITY TEST FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()

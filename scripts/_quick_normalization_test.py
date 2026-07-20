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
    apply_normalization,
    build_normalization_summary,
    compare_raw_vs_normalized,
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
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df_to_print = df.copy()
    if len(numeric_cols) > 0:
        df_to_print[numeric_cols] = df_to_print[numeric_cols].round(decimals)
    print(df_to_print.to_string(index=False))


def make_run_id(prefix: str = "quick_normalization_test") -> str:
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


def summary_to_dict(summary: Any) -> dict[str, Any]:
    return {
        "mode": summary.mode,
        "changed": summary.changed,
        "n_objects": summary.n_objects,
        "n_features": summary.n_features,
        "parameters": summary.parameters,
    }


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_dataframe(path: Path, df: pd.DataFrame, decimals: int = 6) -> None:
    df_to_save = round_numeric_frame(df, decimals)
    df_to_save.to_csv(path, index=False, encoding="utf-8")


def load_quick_test_settings() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    quick_tests = raw.get("quick_tests", {}) or {}
    normalization = quick_tests.get("normalization", {}) or {}

    return {
        "print_tables": bool(quick_tests.get("print_tables", True)),
        "round_decimals": int(quick_tests.get("round_decimals", 6)),
        "save_outputs": bool(normalization.get("save_outputs", True)),
        "output_subdir": str(normalization.get("output_subdir", "quick_normalization_test")),
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

    print_separator("QUICK NORMALIZATION TEST")

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

    # ------------------------------------------------------------
    # 2) Apply normalization from default.yaml
    # ------------------------------------------------------------
    print("\n[2] Applying normalization from default.yaml ...")
    normalized = apply_normalization(dataset, cfg.preprocessing.normalization)
    summary = build_normalization_summary(normalized)
    comparison = compare_raw_vs_normalized(dataset, cfg.preprocessing.normalization)

    print("OK: normalization completed")

    print("\nSummary")
    print("-" * 80)
    print(f"Mode        : {summary.mode}")
    print(f"Changed     : {summary.changed}")
    print(f"Objects     : {summary.n_objects}")
    print(f"Features    : {summary.n_features}")
    print("Parameters  :")
    for key, value in summary.parameters.items():
        print(f"  {key}: {value}")

    # ------------------------------------------------------------
    # 3) Optional terminal tables
    # ------------------------------------------------------------
    if print_tables:
        print_frame(comparison["raw_frame"], "RAW FRAME", round_decimals)
        print_frame(comparison["selected_frame"], "SELECTED FRAME", round_decimals)

        if "normalized_frame" in comparison:
            print_frame(comparison["normalized_frame"], "NORMALIZED FRAME", round_decimals)

    # ------------------------------------------------------------
    # 4) Save outputs
    # ------------------------------------------------------------
    if save_outputs:
        run_id = make_run_id()
        run_dir = ensure_dir(PROJECT_ROOT / "outputs" / "runs" / output_subdir / run_id)

        raw_frame = comparison["raw_frame"]
        selected_frame = comparison["selected_frame"]

        save_dataframe(run_dir / "raw_dataset.csv", raw_frame, round_decimals)
        save_dataframe(run_dir / "selected_dataset.csv", selected_frame, round_decimals)

        if "normalized_frame" in comparison:
            save_dataframe(
                run_dir / "normalized_dataset.csv",
                comparison["normalized_frame"],
                round_decimals,
            )

        write_text(run_dir / "dataset_path.txt", str(dataset.source_path or ""))
        write_text(run_dir / "normalization_mode.txt", cfg.preprocessing.normalization.mode)

        metadata_payload = {
            "dataset_source_path": dataset.source_path,
            "dataset_source_format": dataset.source_format,
            "n_objects": dataset.n_objects,
            "n_features": dataset.n_features,
            "classes": dataset.classes,
            "class_counts": dataset.class_counts,
            "feature_names": dataset.feature_names,
            "object_labels": dataset.object_labels,
            "dataset_metadata": dataset.metadata,
            "normalization_summary": summary_to_dict(summary),
            "branch_rule": {
                "raw": "original dataset",
                "selected": "dataset actually selected by default.yaml normalization mode",
                "normalized": "explicit normalized branch when normalization mode is not none",
            },
            "with_normalization": cfg.preprocessing.normalization.mode.strip().lower() != "none",
        }
        write_json(run_dir / "run_info.json", metadata_payload)

        print("\n[3] Outputs saved")
        print("-" * 80)
        print(f"Run directory : {run_dir}")

    print_separator("QUICK NORMALIZATION TEST FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
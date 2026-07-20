from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint


# ============================================================
# Make src/ importable when running this script directly
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from synthetic_bit_sequence_majority_rule.io.configs import (  # noqa: E402
    describe_config,
    load_default_config,
    load_experiments_config,
)
from synthetic_bit_sequence_majority_rule.io.loaders import try_alternate_dataset_paths  # noqa: E402


def main() -> None:
    print("=" * 80)
    print("QUICK LOAD TEST")
    print("=" * 80)

    # ------------------------------------------------------------
    # 1) Load default config
    # ------------------------------------------------------------
    print("\n[1] Loading default config ...")
    cfg = load_default_config(PROJECT_ROOT / "configs" / "default.yaml")
    print("OK: default config loaded")
    print("Config summary:")
    print(describe_config(cfg))

    # ------------------------------------------------------------
    # 2) Load experiments config
    # ------------------------------------------------------------
    print("\n[2] Loading experiments config ...")
    experiments = load_experiments_config(PROJECT_ROOT / "configs" / "experiments.yaml")
    print(f"OK: experiments config loaded ({len(experiments)} experiment(s))")

    print("\nExperiment names / summaries:")
    for idx, exp_cfg in enumerate(experiments, start=1):
        print(f"  {idx}. {describe_config(exp_cfg)}")

    # ------------------------------------------------------------
    # 3) Try loading dataset from primary / alternate paths
    # ------------------------------------------------------------
    print("\n[3] Loading dataset from default config ...")
    dataset = try_alternate_dataset_paths(cfg.dataset)
    print("OK: dataset loaded successfully")

    # ------------------------------------------------------------
    # 4) Print dataset summary
    # ------------------------------------------------------------
    print("\nDataset summary")
    print("-" * 80)
    print(f"Source path       : {dataset.source_path}")
    print(f"Source format     : {dataset.source_format}")
    print(f"Objects (m)       : {dataset.n_objects}")
    print(f"Features (n)      : {dataset.n_features}")
    print(f"Classes           : {dataset.classes}")
    print(f"Valid neighbors   : {dataset.valid_neighbor_count}")
    print(f"Class counts      : {dataset.class_counts}")

    print("\nObject labels:")
    print(dataset.object_labels)

    print("\nFeature names:")
    print(dataset.feature_names)

    print("\nClass column:")
    print(dataset.class_column)

    # ------------------------------------------------------------
    # 5) Print metadata discovered from dataset structure
    # ------------------------------------------------------------
    print("\nDataset metadata")
    print("-" * 80)
    pprint(dataset.metadata)

    # ------------------------------------------------------------
    # 6) Show first few rows as DataFrame
    # ------------------------------------------------------------
    print("\nDataset preview")
    print("-" * 80)
    preview_df = dataset.to_frame()
    print(preview_df.head(10).to_string(index=False))

    # ------------------------------------------------------------
    # 7) Explicit checks tied to your current dataset style
    # ------------------------------------------------------------
    print("\nSanity checks for the current project dataset style")
    print("-" * 80)

    if dataset.metadata.get("m_objects") is not None:
        print(f"Declared m_objects       : {dataset.metadata.get('m_objects')}")
    else:
        print("Declared m_objects       : not explicitly provided in file")

    if dataset.metadata.get("n_features") is not None:
        print(f"Declared n_features      : {dataset.metadata.get('n_features')}")
    else:
        print("Declared n_features      : not explicitly provided in file")

    if dataset.metadata.get("n_classes") is not None:
        print(f"Declared n_classes       : {dataset.metadata.get('n_classes')}")
    else:
        print("Declared n_classes       : not explicitly provided in file")

    print(f"Has shape row            : {dataset.metadata.get('has_shape_row')}")
    print(f"Has feature sign row     : {dataset.metadata.get('has_feature_sign_row')}")
    print(f"Feature signs            : {dataset.metadata.get('feature_signs')}")
    print(f"Quantitative indices     : {dataset.metadata.get('quantitative_feature_indices')}")
    print(f"Nominal indices          : {dataset.metadata.get('nominal_feature_indices')}")
    print(f"All quantitative         : {dataset.metadata.get('all_features_quantitative')}")
    print(f"Has nominal features     : {dataset.metadata.get('has_nominal_features')}")

    # ------------------------------------------------------------
    # 8) Final note
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("QUICK LOAD TEST FINISHED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
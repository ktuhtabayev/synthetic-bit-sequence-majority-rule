from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from synthetic_bit_sequence_majority_rule.io.configs import load_default_config  # noqa: E402
from synthetic_bit_sequence_majority_rule.services.analysis import run_full_analysis  # noqa: E402
from synthetic_bit_sequence_majority_rule.services.report_builder import (  # noqa: E402
    build_run_summary_frame,
)


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "default.yaml"

    print("=" * 80)
    print("SYNTHETIC BIT SEQUENCE MAJORITY RULE")
    print("=" * 80)
    print(f"Config: {config_path}")

    config = load_default_config(config_path)
    analysis = run_full_analysis(
        config,
        project_root=PROJECT_ROOT,
        write_outputs=config.run.save_outputs,
    )
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


if __name__ == "__main__":
    main()

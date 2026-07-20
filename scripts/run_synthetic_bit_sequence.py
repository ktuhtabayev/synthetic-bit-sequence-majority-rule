from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from synthetic_bit_sequence_majority_rule.gui.meta_objects import (  # noqa: E402
    build_normalization_comparison,
)
from synthetic_bit_sequence_majority_rule.io.writers import (  # noqa: E402
    write_normalization_comparison_outputs,
    write_pipeline_outputs,
)
from synthetic_bit_sequence_majority_rule.services.report_builder import (  # noqa: E402
    build_run_summary_frame,
)
from synthetic_bit_sequence_majority_rule.services.runner import (  # noqa: E402
    run_none_minmax_comparison,
    run_pipeline_from_config,
)


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "default.yaml"

    print("=" * 80)
    print("SYNTHETIC BIT SEQUENCE MAJORITY RULE")
    print("=" * 80)
    print(f"Config: {config_path}")

    result = run_pipeline_from_config(config_path)
    comparison_run = run_none_minmax_comparison(result.config, result)
    comparison = build_normalization_comparison(
        comparison_run.branches["raw"].stability_results,
        comparison_run.branches["normalized"].stability_results,
        comparison_run.branches["raw"].complexity_result,
        comparison_run.branches["normalized"].complexity_result,
    )
    summary = build_run_summary_frame(result)

    print("\nRun summary")
    print("-" * 80)
    print(summary.to_string(index=False))

    if result.config.run.save_outputs:
        run_dir = write_pipeline_outputs(result, PROJECT_ROOT / result.config.run.output_root)
        write_normalization_comparison_outputs(comparison, run_dir)
        print("\nOutputs saved")
        print("-" * 80)
        print(run_dir)

    print("\nFinal comparison for selected branch")
    print("-" * 80)
    print(result.selected_branch.final_comparison.to_frame().to_string(index=False))

    print("\nFinished successfully.")


if __name__ == "__main__":
    main()

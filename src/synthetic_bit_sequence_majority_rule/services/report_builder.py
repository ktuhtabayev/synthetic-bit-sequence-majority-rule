from __future__ import annotations

import pandas as pd

from synthetic_bit_sequence_majority_rule.services.runner import PipelineRunResult


def build_run_summary_frame(result: PipelineRunResult) -> pd.DataFrame:
    rows = []
    for branch_name, branch in result.branches.items():
        rows.append(
            {
                "RunId": result.run_id,
                "Branch": branch_name,
                "Dataset": branch.dataset.source_path,
                "Objects": branch.dataset.n_objects,
                "Features": branch.dataset.n_features,
                "ClassCounts": str(branch.dataset.class_counts),
                "Normalization": branch.dataset.metadata.get("normalization_mode", "none"),
                "Metrics": ", ".join(branch.majority_results.keys()),
                "KMax": result.metadata.get("formula_kmax"),
                "ReducedK": ",".join(str(k) for k in result.metadata.get("reduced_k_values", [])),
            }
        )
    return pd.DataFrame(rows)

from __future__ import annotations

from synthetic_bit_sequence_majority_rule.algorithms.statistics import (
    StabilityRow,
    StabilityTableResult,
)
from synthetic_bit_sequence_majority_rule.gui.stability_plot import (
    StabilityPlotPoint,
    StabilityPlotSeries,
    build_stability_conclusion,
    prepare_stability_plot_series,
)


def make_stability_result(metric_name: str, values: list[tuple[list[int], float]]) -> StabilityTableResult:
    return StabilityTableResult(
        metric_name=metric_name,
        rows=[
            StabilityRow(
                metric_name=metric_name,
                representation="_".join(f"b{k}" for k in k_values),
                k_values=k_values,
                unique_values=2,
                object_count=10,
                stability=stability,
                interpretation="High" if stability > 0.8 else "Satisfactory",
            )
            for k_values, stability in values
        ],
    )


def test_prepare_stability_plot_series_extracts_current_k_values_in_order() -> None:
    series = prepare_stability_plot_series(
        {
            "euclidean": make_stability_result(
                "euclidean",
                [
                    ([3, 5, 7], 0.72),
                    ([3], 0.61),
                    ([3, 5], 0.65),
                ],
            ),
            "canberra": make_stability_result(
                "canberra",
                [
                    ([3], 0.55),
                    ([3, 5], 0.62),
                ],
            ),
        }
    )

    assert [item.metric_name for item in series] == ["euclidean", "canberra"]
    assert [point.k_value for point in series[0].points] == [3, 5, 7]
    assert [point.stability for point in series[0].points] == [0.61, 0.65, 0.72]
    assert [point.k_value for point in series[1].points] == [3, 5]


def test_build_stability_conclusion_reports_best_trend_high_and_disagreement() -> None:
    series = [
        StabilityPlotSeries(
            metric_name="euclidean",
            points=[
                StabilityPlotPoint(3, 0.55, "Poor stability"),
                StabilityPlotPoint(5, 0.70, "Satisfactory"),
                StabilityPlotPoint(7, 0.90, "High"),
            ],
        ),
        StabilityPlotSeries(
            metric_name="canberra",
            points=[
                StabilityPlotPoint(3, 0.50, "Not distinguishable"),
                StabilityPlotPoint(5, 0.52, "Poor stability"),
                StabilityPlotPoint(7, 0.53, "Poor stability"),
            ],
        ),
    ]

    conclusion = build_stability_conclusion(series)

    assert "Best observed stability: Euclidean = 0.900000 at k=7" in conclusion
    assert "Euclidean improves" in conclusion
    assert "Euclidean first reaches High at k=7" in conclusion
    assert "Canberra does not reach High stability" in conclusion
    assert "Warning: metrics differ strongly at k=7" in conclusion


def test_build_stability_conclusion_detects_high_plateau() -> None:
    conclusion = build_stability_conclusion(
        [
            StabilityPlotSeries(
                metric_name="chebyshev",
                points=[
                    StabilityPlotPoint(3, 0.62, "Satisfactory"),
                    StabilityPlotPoint(5, 0.88, "High"),
                    StabilityPlotPoint(7, 0.89, "High"),
                    StabilityPlotPoint(9, 0.90, "High"),
                ],
            )
        ]
    )

    assert "Chebyshev improves and plateaus at high stability" in conclusion


def test_build_stability_conclusion_handles_empty_data() -> None:
    assert build_stability_conclusion([]) == "No stability data is available yet."

from __future__ import annotations

from dataclasses import dataclass

from synthetic_bit_sequence_majority_rule.algorithms.statistics import StabilityTableResult


@dataclass(frozen=True, slots=True)
class StabilityPlotPoint:
    k_value: int
    stability: float
    interpretation: str


@dataclass(frozen=True, slots=True)
class StabilityPlotSeries:
    metric_name: str
    points: list[StabilityPlotPoint]


def prepare_stability_plot_series(
    stability_results: dict[str, StabilityTableResult],
) -> list[StabilityPlotSeries]:
    series: list[StabilityPlotSeries] = []

    for metric_name, result in stability_results.items():
        points = [
            StabilityPlotPoint(
                k_value=int(row.k_values[-1]),
                stability=float(row.stability),
                interpretation=row.interpretation,
            )
            for row in result.rows
            if row.k_values
        ]
        points.sort(key=lambda point: point.k_value)
        series.append(StabilityPlotSeries(metric_name=metric_name, points=points))

    return series


def build_stability_conclusion(series: list[StabilityPlotSeries]) -> str:
    points_with_metric = [
        (item.metric_name, point)
        for item in series
        for point in item.points
    ]
    if not points_with_metric:
        return "No stability data is available yet."

    best_metric, best_point = max(
        points_with_metric,
        key=lambda item: (item[1].stability, -item[1].k_value, item[0]),
    )
    lines = [
        (
            f"Best observed stability: {best_metric.title()} = "
            f"{best_point.stability:.6f} at k={best_point.k_value} "
            f"({best_point.interpretation})."
        )
    ]

    trend_lines: list[str] = []
    high_lines: list[str] = []
    for item in series:
        if not item.points:
            continue

        first = item.points[0]
        last = item.points[-1]
        delta = last.stability - first.stability
        if delta > 0.05:
            trend = "improves"
        elif delta < -0.05:
            trend = "drops"
        else:
            trend = "stays nearly flat"

        plateau = ""
        if len(item.points) >= 3:
            tail = item.points[-3:]
            tail_values = [point.stability for point in tail]
            if max(tail_values) - min(tail_values) <= 0.020000001 and sum(tail_values) / len(tail_values) > 0.8:
                plateau = " and plateaus at high stability"

        trend_lines.append(
            (
                f"{item.metric_name.title()} {trend}{plateau} "
                f"from {first.stability:.6f} to {last.stability:.6f}."
            )
        )

        first_high = next((point for point in item.points if point.stability > 0.8), None)
        if first_high is None:
            high_lines.append(f"{item.metric_name.title()} does not reach High stability.")
        else:
            high_lines.append(f"{item.metric_name.title()} first reaches High at k={first_high.k_value}.")

    if trend_lines:
        lines.append("Trend: " + " ".join(trend_lines))
    if high_lines:
        lines.append("High threshold: " + " ".join(high_lines))

    warning = _metric_disagreement_warning(series)
    if warning:
        lines.append(warning)

    return "\n".join(lines)


def _metric_disagreement_warning(series: list[StabilityPlotSeries]) -> str:
    by_k: dict[int, list[tuple[str, float]]] = {}
    for item in series:
        for point in item.points:
            by_k.setdefault(point.k_value, []).append((item.metric_name, point.stability))

    strongest: tuple[int, float] | None = None
    for k_value, values in by_k.items():
        if len(values) < 2:
            continue
        stabilities = [value for _, value in values]
        spread = max(stabilities) - min(stabilities)
        if strongest is None or spread > strongest[1]:
            strongest = (k_value, spread)

    if strongest is not None and strongest[1] >= 0.15:
        return (
            f"Warning: metrics differ strongly at k={strongest[0]} "
            f"(spread {strongest[1]:.6f})."
        )
    return ""

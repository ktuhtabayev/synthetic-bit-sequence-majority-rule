from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QHeaderView  # noqa: E402

from synthetic_bit_sequence_majority_rule.gui.app import (  # noqa: E402
    GRAY_APP_STYLESHEET,
    MainWindow,
    frame_to_table,
)
from synthetic_bit_sequence_majority_rule.io.configs import load_default_config  # noqa: E402
from synthetic_bit_sequence_majority_rule.services.runner import run_pipeline  # noqa: E402


def test_gui_tables_do_not_stretch_last_column_and_remain_resizable() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    table = frame_to_table(
        pd.DataFrame(
            {
                "Object": ["S1", "S2"],
                "x1": [0.1234567, 1.0],
                "Class": [1, 2],
            }
        )
    )
    header = table.horizontalHeader()
    model = table.model()

    assert header.stretchLastSection() is False
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Interactive
    assert header.sectionResizeMode(model.columnCount() - 1) == QHeaderView.ResizeMode.Interactive


def test_gui_table_preserves_source_order_and_sorts_object_names_naturally() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    source_order = ["S1", "S2", "S10", "S3"]
    table = frame_to_table(
        pd.DataFrame(
            {
                "Object": source_order,
                "Class": [1, 1, 2, 1],
            }
        )
    )
    model = table.model()

    assert table.horizontalHeader().sortIndicatorSection() == -1
    assert model.frame["Object"].tolist() == source_order

    table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    assert model.frame["Object"].tolist() == ["S1", "S2", "S3", "S10"]

    table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
    assert model.frame["Object"].tolist() == ["S10", "S3", "S2", "S1"]

    model.sort(-1)
    assert model.frame["Object"].tolist() == source_order


def test_gui_table_values_are_center_aligned_for_text_and_numbers() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    table = frame_to_table(
        pd.DataFrame(
            {
                "Metric": ["euclidean", "canberra"],
                "Decimal": [1023, 7],
                "Status": ["Pure", "Mixed"],
            }
        )
    )
    expected_alignment = int(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    model = table.model()

    for row in range(model.rowCount()):
        for column in range(model.columnCount()):
            alignment = model.data(model.index(row, column), Qt.ItemDataRole.TextAlignmentRole)
            assert int(alignment) == expected_alignment


def test_gui_stylesheet_uses_light_gray_theme_with_dark_text() -> None:
    assert "#eeeeee" in GRAY_APP_STYLESHEET
    assert "#fafafa" in GRAY_APP_STYLESHEET
    assert "color: #202020" in GRAY_APP_STYLESHEET


def test_metrics_dropdown_initializes_from_config_and_updates_text() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    window = MainWindow(Path.cwd())

    assert window.normalization.currentText() == "none"
    assert window._selected_metric_names() == ["euclidean", "chebyshev", "canberra", "manhattan"]
    assert window.metrics_button.text() == "Metrics: All"

    window.metric_actions["canberra"].setChecked(False)
    window._update_metric_button_text()
    assert window._selected_metric_names() == ["euclidean", "chebyshev", "manhattan"]
    assert window.metrics_button.text() == "Metrics: 3 selected"

    window.metric_actions["chebyshev"].setChecked(False)
    window.metric_actions["manhattan"].setChecked(False)
    window._update_metric_button_text()
    assert window.metrics_button.text() == "Metrics: Euclidean"


def test_metrics_dropdown_falls_back_to_euclidean_when_all_unchecked() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    window = MainWindow(Path.cwd())
    for action in window.metric_actions.values():
        action.setChecked(False)
    window._update_metric_button_text()

    cfg = window._config_from_controls()

    assert window.metrics_button.text() == "Metrics: Euclidean (default)"
    assert cfg.metrics.enabled == ["euclidean"]


def test_stability_plot_tab_exists_and_refreshes_after_populate() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    cfg = load_default_config(Path.cwd() / "configs" / "default.yaml")
    result = run_pipeline(cfg, run_id="gui_stability_plot_test")
    window = MainWindow(Path.cwd())
    window.last_result = result
    window.populate_tabs(result)
    window.refresh_stability_plot()

    tab_names = [window.tabs.tabText(index) for index in range(window.tabs.count())]
    assert "Stability Plot" in tab_names
    assert window.stability_plot_canvas is not None
    assert window.stability_conclusion is not None
    assert "Best observed stability" in window.stability_conclusion.toPlainText()


def test_synthetic_features_and_meta_objects_tabs_exist_after_populate() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    cfg = load_default_config(Path.cwd() / "configs" / "default.yaml")
    cfg.preprocessing.normalization.mode = "minmax"
    result = run_pipeline(cfg, run_id="gui_new_views_test")
    window = MainWindow(Path.cwd())
    window.last_result = result
    window.normalization_comparison = window._build_normalization_comparison(result)
    window.populate_tabs(result)

    tab_names = [window.tabs.tabText(index) for index in range(window.tabs.count())]
    assert "Synthetic Features Space" in tab_names
    assert "Meta Objects" in tab_names

    synthetic_tab = window.tabs.widget(tab_names.index("Synthetic Features Space"))
    synthetic_metric_tabs = synthetic_tab.layout().itemAt(0).widget()
    first_metric_tabs = synthetic_metric_tabs.widget(0)
    assert first_metric_tabs.tabText(0) == "Binary"
    assert first_metric_tabs.tabText(1) == "Decimal"

    meta_tab = window.tabs.widget(tab_names.index("Meta Objects"))
    meta_controls = meta_tab.layout().itemAt(0).layout()
    assert meta_controls.itemAt(0).widget().text() == "Save PCA 2D PNG"
    assert meta_controls.itemAt(1).widget().text() == "Save PCA 3D PNG"
    meta_tabs = meta_tab.layout().itemAt(1).widget()
    assert [meta_tabs.tabText(index) for index in range(meta_tabs.count())] == [
        "Based on Stability",
        "Complexity C(Q)",
        "Applying PCA (2D)",
        "Applying PCA (3D)",
        "Visualization (2D)",
        "Visualization (3D)",
        "None vs MinMax",
    ]
    assert "pca_2d" in window.meta_object_figures
    comparison_tab = meta_tabs.widget(6)
    comparison_tabs = comparison_tab.layout().itemAt(1).widget()
    assert [comparison_tabs.tabText(index) for index in range(comparison_tabs.count())] == [
        "Summary",
        "PCA 2D",
        "PCA 3D",
    ]
    assert "normalization_comparison_2d" in window.meta_object_figures

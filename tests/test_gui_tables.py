from __future__ import annotations

import os
import threading

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QHeaderView, QMessageBox  # noqa: E402

from synthetic_bit_sequence_majority_rule.gui import app as gui_app  # noqa: E402
from synthetic_bit_sequence_majority_rule.gui.app import (  # noqa: E402
    _display_value,
    AnalysisWorker,
    MainWindow,
    frame_to_table,
    run_output_dir,
)
from synthetic_bit_sequence_majority_rule.gui.theme import THEME, build_stylesheet  # noqa: E402
from synthetic_bit_sequence_majority_rule.io.configs import load_default_config  # noqa: E402
from synthetic_bit_sequence_majority_rule.services.analysis import (  # noqa: E402
    build_comparison_from_run,
)
from synthetic_bit_sequence_majority_rule.domain.errors import PipelineExecutionError  # noqa: E402
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


def test_gui_stylesheet_is_built_from_theme_colors() -> None:
    stylesheet = build_stylesheet()
    assert THEME["window_bg"] in stylesheet
    assert THEME["accent"] in stylesheet
    assert f"color: {THEME['text']}" in stylesheet
    assert "#primaryButton" in stylesheet


def test_metrics_dropdown_initializes_from_config_and_updates_text() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    window = MainWindow(Path.cwd(), restore_settings=False)

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

    window = MainWindow(Path.cwd(), restore_settings=False)
    for action in window.metric_actions.values():
        action.setChecked(False)
    window._update_metric_button_text()

    cfg = window._config_from_controls()

    assert window.metrics_button.text() == "Metrics: Euclidean (default)"
    assert cfg.metrics.enabled == ["euclidean"]


def test_dataset_preset_dropdown_fills_path_and_tracks_custom_edits() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    window = MainWindow(Path.cwd(), restore_settings=False)

    preset_names = [
        window.dataset_preset.itemText(index)
        for index in range(window.dataset_preset.count())
    ]
    assert preset_names[0] == "Custom..."
    assert "default_csv" in preset_names
    assert "ionosfera_dat" in preset_names

    # The default config path matches the default_csv preset.
    assert window.dataset_preset.currentText() == "default_csv"

    row = preset_names.index("ionosfera_dat")
    window.dataset_preset.setCurrentIndex(row)
    window._on_dataset_preset_selected(row)
    assert window.dataset_path.text().endswith("Ionosfera (350, 33, 2).dat")

    window.dataset_path.setText(str(Path.cwd() / "datasets" / "nonexistent.csv"))
    window._sync_preset_to_path()
    assert window.dataset_preset.currentText() == "Custom..."


def test_stability_plot_tab_exists_and_refreshes_after_populate() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    cfg = load_default_config(Path.cwd() / "configs" / "default.yaml")
    result = run_pipeline(cfg, run_id="gui_stability_plot_test")
    window = MainWindow(Path.cwd(), restore_settings=False)
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
    window = MainWindow(Path.cwd(), restore_settings=False)
    window.last_result = result
    window.normalization_comparison = build_comparison_from_run(result)
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


def _run_in_worker(tmp_path: Path):
    """Run the GUI worker synchronously, with outputs redirected to tmp_path."""
    cfg = load_default_config(Path.cwd() / "configs" / "default.yaml")
    cfg.run.output_root = str(tmp_path)
    worker = AnalysisWorker(cfg, Path.cwd())
    finished = []
    failures = []
    worker.finished_ok.connect(finished.append)
    worker.failed.connect(failures.append)
    worker.run()
    assert not failures and len(finished) == 1, failures
    return finished[0]


def _finished_window(tmp_path: Path, monkeypatch) -> MainWindow:
    app = QApplication.instance() or QApplication([])
    assert app is not None
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)

    window = MainWindow(Path.cwd(), restore_settings=False)
    window._on_analysis_finished(_run_in_worker(tmp_path))
    return window


def test_a_finished_run_writes_nothing_until_export(tmp_path, monkeypatch) -> None:
    window = _finished_window(tmp_path, monkeypatch)

    assert window.last_output_dir is None
    assert not window.open_output_button.isEnabled()
    assert "Output folder: Not exported" in window.status_text.toPlainText()
    assert not any(tmp_path.iterdir())


def test_export_writes_the_run_folder_and_enables_open_output(tmp_path, monkeypatch) -> None:
    window = _finished_window(tmp_path, monkeypatch)
    expected_dir = run_output_dir(window.last_result, window.project_root)
    assert expected_dir == tmp_path / "gui" / window.last_result.run_id

    window.export_last_result()
    # Export runs in the background with Run and Export disabled until it ends.
    assert not window.run_button.isEnabled() and not window.export_button.isEnabled()
    _wait_for_background_task(window)

    assert window.last_output_dir == expected_dir
    assert (expected_dir / "run_info.json").is_file()
    assert (expected_dir / "normalization_comparison" / "meta_objects.csv").is_file()
    assert window.open_output_button.isEnabled()
    assert window.run_button.isEnabled() and window.export_button.isEnabled()
    assert f"Output folder: {expected_dir}" in window.status_text.toPlainText()


def _wait_for_background_task(window: MainWindow) -> None:
    worker = window.active_worker
    assert worker is not None
    assert worker.wait(60_000)
    # Deliver the worker's queued signals to the window.
    QApplication.processEvents()
    assert window.active_worker is None


def test_saving_a_plot_writes_only_that_png_and_enables_open_output(tmp_path, monkeypatch) -> None:
    window = _finished_window(tmp_path, monkeypatch)
    expected_dir = run_output_dir(window.last_result, window.project_root)

    window.save_stability_plot()

    assert window.last_output_dir == expected_dir
    assert [path.relative_to(expected_dir) for path in expected_dir.rglob("*") if path.is_file()] == [
        Path("selected") / "stability_plot.png"
    ]
    assert window.open_output_button.isEnabled()
    assert f"Output folder: {expected_dir}" in window.status_text.toPlainText()

    # The next Run has not been exported yet, so the button greys out again.
    window._on_analysis_finished(_run_in_worker(tmp_path))
    assert window.last_output_dir is None
    assert not window.open_output_button.isEnabled()


def test_a_missing_preset_file_fails_instead_of_loading_the_default_dataset() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    window = MainWindow(Path.cwd(), restore_settings=False)
    window.dataset_catalog = {
        "missing_xlsx": {"path": "datasets/missing/Missing.xlsx", "alternate_paths": [], "format": "xlsx"}
    }
    window.dataset_path.setText(str(Path.cwd() / "datasets" / "missing" / "Missing.xlsx"))

    cfg = window._config_from_controls()

    # The config's own alternate (datasets/default.dat) must not leak in.
    assert cfg.dataset.alternate_paths == []
    with pytest.raises(PipelineExecutionError, match="not found"):
        run_pipeline(cfg, run_id="gui_missing_dataset_test")


def test_a_custom_dataset_path_has_no_alternates() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    window = MainWindow(Path.cwd(), restore_settings=False)
    window.dataset_path.setText(str(Path.cwd() / "datasets" / "nowhere.csv"))

    assert window._config_from_controls().dataset.alternate_paths == []


def test_a_preset_supplies_its_own_loading_options_and_alternates() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    window = MainWindow(Path.cwd(), restore_settings=False)
    preset_names = [window.dataset_preset.itemText(i) for i in range(window.dataset_preset.count())]
    row = preset_names.index("ionosfera_csv")
    window._on_dataset_preset_selected(row)

    cfg = window._config_from_controls()

    entry = window.dataset_catalog["ionosfera_csv"]
    assert cfg.dataset.path == Path.cwd() / entry["path"]
    assert cfg.dataset.has_header is False
    assert cfg.dataset.alternate_paths == [Path.cwd() / path for path in entry["alternate_paths"]]


def test_a_run_does_not_depend_on_the_launch_folder(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    project_root = Path.cwd()
    window = MainWindow(project_root, restore_settings=False)
    monkeypatch.chdir(tmp_path)

    result = run_pipeline(window._config_from_controls(), run_id="gui_launch_folder_test")

    assert Path(result.source_dataset.source_path) == project_root / "datasets" / "default.csv"


def test_closing_the_window_waits_for_a_running_analysis(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None
    release = threading.Event()

    def slow_analysis(*args, **kwargs):
        release.wait(timeout=10)
        raise RuntimeError("stopped by test")

    monkeypatch.setattr(gui_app, "run_full_analysis", slow_analysis)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: None)
    window = MainWindow(Path.cwd(), restore_settings=False)
    window.run_current_pipeline()
    worker = window.active_worker
    assert worker is not None and worker.isRunning()

    threading.Timer(0.2, release.set).start()
    window.close()

    # Closing blocked until the thread finished instead of destroying it mid-run.
    assert worker.isFinished()


def test_a_failed_plot_save_is_reported_instead_of_raising(tmp_path, monkeypatch) -> None:
    window = _finished_window(tmp_path, monkeypatch)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args[-1]))
    monkeypatch.setattr(
        window.stability_plot_figure,
        "savefig",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("read-only folder")),
    )

    window.save_stability_plot()

    assert errors == ["read-only folder"]
    assert window.last_output_dir is None
    assert not window.open_output_button.isEnabled()


def test_table_cells_show_the_same_text_as_the_frame_values_before_and_after_sorting() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None
    frame = pd.DataFrame(
        {
            "Object": ["S2", "S10", "S1"],
            "Distance": [0.1234567891, float("nan"), 2.0],
            "Count": [3, 1, 2],
            "Decimal": [(1 << 123) - 1, 5, 0],  # Python ints beyond int64
            "Flag": [True, False, True],
        }
    )
    table = frame_to_table(frame)
    model = table.model()

    def shown() -> list[list[str]]:
        return [
            [model.data(model.index(row, col)) for col in range(model.columnCount())]
            for row in range(model.rowCount())
        ]

    def expected() -> list[list[str]]:
        current = model.frame
        return [
            [_display_value(current.iat[row, col], 6) for col in range(current.shape[1])]
            for row in range(current.shape[0])
        ]

    assert shown() == expected()
    assert shown()[0][1] == "0.123457" and shown()[1][1] == "" and shown()[0][3] == str((1 << 123) - 1)
    table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    assert model.frame["Object"].tolist() == ["S1", "S2", "S10"]
    assert shown() == expected()


def test_small_tables_fit_their_columns_when_first_shown() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None
    table = frame_to_table(pd.DataFrame({"Label": ["a fairly long cell value that needs room"]}))
    width_before = table.columnWidth(0)

    table.show()
    QApplication.processEvents()

    assert table.columnWidth(0) > width_before
    table.close()

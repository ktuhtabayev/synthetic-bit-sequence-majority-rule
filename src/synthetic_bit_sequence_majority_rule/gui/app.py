from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from synthetic_bit_sequence_majority_rule.algorithms.majority import (
    a_full_frame,
    a_reduced_frame,
    b_full_frame,
    b_reduced_frame,
    same_class_indicator_frame,
)
from synthetic_bit_sequence_majority_rule.algorithms.neighbors import neighbor_combined_frame
from synthetic_bit_sequence_majority_rule.algorithms.meta_objects import (
    NormalizationComparisonResult,
    PCAProjectionResult,
    apply_pca_to_meta_objects,
    build_meta_objects_from_stability,
    label_offsets_for_points,
    near_zero_axis_notes,
)
from synthetic_bit_sequence_majority_rule.gui.stability_plot import (
    build_stability_conclusion,
    prepare_stability_plot_series,
)
from synthetic_bit_sequence_majority_rule.gui.theme import (
    STABILITY_BANDS,
    THEME,
    build_stylesheet,
)
from synthetic_bit_sequence_majority_rule.gui.synthetic_features import (
    build_synthetic_binary_frame,
    build_synthetic_decimal_frame,
)
from synthetic_bit_sequence_majority_rule.io.configs import (
    apply_dataset_selection,
    find_dataset_preset,
    load_dataset_catalog,
    load_default_config,
)
from synthetic_bit_sequence_majority_rule.paths import PROJECT_ROOT
from synthetic_bit_sequence_majority_rule.services.analysis import (
    FullAnalysisResult,
    analysis_output_dir,
    run_full_analysis,
    write_analysis_outputs,
)
from synthetic_bit_sequence_majority_rule.services.runner import (
    PipelineBranchResult,
    PipelineRunResult,
)


try:
    from PyQt6.QtCore import (
        QAbstractTableModel,
        QModelIndex,
        QSettings,
        QThread,
        QUrl,
        Qt,
        pyqtSignal,
    )
    from PyQt6.QtGui import QAction, QDesktopServices
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QComboBox,
        QFileDialog,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMenu,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QTabWidget,
        QTableView,
        QTextEdit,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - depends on local GUI install
    raise RuntimeError("PyQt6 is required to launch the desktop GUI.") from exc


logger = logging.getLogger(__name__)

T = TypeVar("T")


def _display_value(value: Any, decimals: int) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return str(value)


def _natural_sort_key(value: Any) -> tuple[object, ...]:
    if pd.isna(value):
        return (2, ())
    if isinstance(value, Real):
        return (0, float(value))

    parts = tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", str(value))
        if part
    )
    return (1, parts)


class DataFrameTableModel(QAbstractTableModel):
    """Expose a DataFrame lazily so large research tables remain responsive."""

    def __init__(self, frame: pd.DataFrame, *, decimals: int = 6) -> None:
        super().__init__()
        # reset_index returns a new frame, and sorting replaces self.frame rather
        # than editing it, so neither needs another copy.
        self._source_frame = frame.reset_index(drop=True)
        self.frame = self._source_frame
        self.decimals = decimals

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    @frame.setter
    def frame(self, frame: pd.DataFrame) -> None:
        # Qt asks for cells one at a time, often several times each, so cells
        # are read from one array per column: an O(1) lookup that yields the
        # same scalars as frame.iat. Columns are fetched on first use, since a
        # wide table only ever shows a few of them.
        self._frame = frame
        self._columns: list[np.ndarray | None] = [None] * frame.shape[1]
        self._row_count, self._column_count = frame.shape

    def _column(self, position: int) -> np.ndarray:
        values = self._columns[position]
        if values is None:
            values = self._columns[position] = self._frame.iloc[:, position].to_numpy()
        return values

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else self._row_count

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else self._column_count

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return _display_value(self._column(index.column())[index.row()], self.decimals)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return None

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self.frame.columns[section])
        return str(section + 1)

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        if column == -1:
            if self._frame is self._source_frame:
                return  # already in source order
            sorted_frame = self._source_frame
        elif 0 <= column < self.columnCount():
            values = self.frame.iloc[:, column].tolist()
            positions = sorted(
                range(len(values)),
                key=lambda index: _natural_sort_key(values[index]),
                reverse=order == Qt.SortOrder.DescendingOrder,
            )
            sorted_frame = self.frame.iloc[positions].reset_index(drop=True)
        else:
            return

        self.layoutAboutToBeChanged.emit()
        self.frame = sorted_frame
        self.layoutChanged.emit()


class _FitOnShowTableView(QTableView):
    """
    Resizes its columns to their contents the first time it is shown.

    Measuring every cell is the costly part of building a table, and a Run
    builds dozens of tables across tabs the user may never open.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fit_pending = False

    def fit_columns_when_shown(self) -> None:
        self._fit_pending = True

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._fit_pending:
            self._fit_pending = False
            self.resizeColumnsToContents()
        super().showEvent(event)


def frame_to_table(frame: pd.DataFrame, *, decimals: int = 6) -> QTableView:
    model = DataFrameTableModel(frame, decimals=decimals)
    table = _FitOnShowTableView()
    table.setModel(model)
    table.setAlternatingRowColors(True)
    table.setWordWrap(False)
    table.setSortingEnabled(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

    horizontal = table.horizontalHeader()
    vertical = table.verticalHeader()
    assert horizontal is not None and vertical is not None  # a QTableView always has both
    horizontal.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    horizontal.setStretchLastSection(False)
    vertical.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
    vertical.setDefaultSectionSize(26)
    if model.rowCount() * model.columnCount() <= 50_000:
        table.fit_columns_when_shown()
    else:
        for col_idx, column in enumerate(model.frame.columns):
            table.setColumnWidth(col_idx, min(180, max(72, len(str(column)) * 9 + 24)))
    horizontal.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
    table.setSortingEnabled(True)
    return table


def _distance_frame(distance_result) -> pd.DataFrame:
    frame = distance_result.to_frame().reset_index()
    return frame.rename(columns={"index": "Object"})


def _document_tabs(pages: Iterable[tuple[str, QWidget]]) -> QTabWidget:
    tabs = QTabWidget()
    tabs.setDocumentMode(True)
    for title, page in pages:
        tabs.addTab(page, title)
    return tabs


def _metric_tabs(results: Mapping[str, T], build_page: Callable[[T], QWidget]) -> QTabWidget:
    """One tab per metric, titled with the metric name."""
    return _document_tabs(
        (metric_name.title(), build_page(result)) for metric_name, result in results.items()
    )


def _to_frame_table(result) -> QTableView:
    return frame_to_table(result.to_frame())


def _majority_views(result) -> QTabWidget:
    return _document_tabs(
        [
            ("Same Class", frame_to_table(same_class_indicator_frame(result))),
            ("A Full", frame_to_table(a_full_frame(result))),
            ("A Reduced", frame_to_table(a_reduced_frame(result))),
            ("B Full", frame_to_table(b_full_frame(result))),
            ("B Reduced", frame_to_table(b_reduced_frame(result))),
        ]
    )


def _canvas_widget(figure: Figure) -> QWidget:
    figure.tight_layout()
    canvas = FigureCanvasQTAgg(figure)
    widget = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(canvas, 1)
    widget.setLayout(layout)
    canvas.draw_idle()
    return widget


class TaskWorker(QThread):
    """Run one task off the GUI thread so the window stays responsive."""

    finished_ok = pyqtSignal(object)  # the task's return value
    failed = pyqtSignal(str)

    def __init__(self, task: Callable[[], object], description: str, parent=None) -> None:
        super().__init__(parent)
        self._task = task
        self._description = description

    def run(self) -> None:  # pragma: no cover - thread entry point
        try:
            result = self._task()
        except Exception as exc:
            logger.exception("%s failed.", self._description)
            self.failed.emit(str(exc))
        else:
            self.finished_ok.emit(result)


class AnalysisWorker(TaskWorker):
    """Run the full analysis; it emits a FullAnalysisResult and writes nothing until Export."""

    def __init__(self, config, project_root: Path, parent=None) -> None:
        super().__init__(
            lambda: run_full_analysis(config, project_root=project_root, write_outputs=False),
            "Analysis run",
            parent,
        )


def run_output_dir(result: PipelineRunResult, project_root: Path) -> Path:
    """The folder Export writes this run to (and where saved plots go)."""
    return analysis_output_dir(result, project_root, gui_export=True)


class MainWindow(QMainWindow):
    AVAILABLE_METRICS = ("euclidean", "chebyshev", "canberra", "manhattan")

    def __init__(self, project_root: Path, restore_settings: bool = True) -> None:
        super().__init__()
        self.project_root = project_root
        self.config_path = project_root / "configs" / "default.yaml"
        self.last_result: PipelineRunResult | None = None
        self.normalization_comparison: NormalizationComparisonResult | None = None
        self.last_output_dir: Path | None = None
        self.stability_plot_figure: Figure | None = None
        self.stability_plot_canvas: FigureCanvasQTAgg | None = None
        self.stability_conclusion: QTextEdit | None = None
        self.meta_object_figures: dict[str, Figure] = {}
        self.active_worker: TaskWorker | None = None  # the Run or Export in progress
        self.restore_settings = restore_settings
        self.settings = QSettings("synthetic-bit-sequence-majority-rule", "DesktopApp")
        self.dataset_catalog = load_dataset_catalog(self.config_path)

        self.setWindowTitle("Synthetic Bit Sequence Majority Rule")
        self.resize(1440, 860)

        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("Dataset path from config")
        self.dataset_path.textEdited.connect(self._on_dataset_path_edited)

        self.dataset_preset = QComboBox()
        self.dataset_preset.addItem("Custom...")
        for preset_name in self.dataset_catalog:
            self.dataset_preset.addItem(preset_name)
        self.dataset_preset.activated.connect(self._on_dataset_preset_selected)

        self.normalization = QComboBox()
        self.normalization.addItems(["none", "minmax", "zscore"])

        self.metric_menu = QMenu(self)
        self.metric_actions: dict[str, QAction] = {}
        for metric_name in self.AVAILABLE_METRICS:
            action = QAction(metric_name.title(), self)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked=False: self._update_metric_button_text())
            self.metric_menu.addAction(action)
            self.metric_actions[metric_name] = action

        self.metrics_button = QToolButton()
        self.metrics_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.metrics_button.setMenu(self.metric_menu)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(120)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        browse_button = QPushButton("Dataset...")
        browse_button.clicked.connect(self.choose_dataset)

        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self.run_current_pipeline)

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_last_result)

        self.open_output_button = QPushButton("Open Output Folder")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self.open_output_folder)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # busy indicator
        self.progress_bar.setMaximumWidth(160)
        self.progress_bar.setVisible(False)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Dataset"))
        controls.addWidget(self.dataset_preset)
        controls.addWidget(self.dataset_path, 1)
        controls.addWidget(browse_button)
        controls.addWidget(QLabel("Normalization"))
        controls.addWidget(self.normalization)
        controls.addWidget(self.metrics_button)
        controls.addWidget(self.run_button)
        controls.addWidget(self.export_button)
        controls.addWidget(self.open_output_button)
        controls.addWidget(self.progress_bar)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.status_text)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)
        self.load_defaults()
        if self.restore_settings:
            self._restore_saved_state()

    def load_defaults(self) -> None:
        cfg = load_default_config(self.config_path)
        self.dataset_path.setText(str(self.project_root / cfg.dataset.path))
        self.normalization.setCurrentText(cfg.preprocessing.normalization.mode)
        for metric, action in self.metric_actions.items():
            action.setChecked(metric in cfg.enabled_metrics)
        self._update_metric_button_text()
        self._sync_preset_to_path()

    def _restore_saved_state(self) -> None:
        # Keep dataset, normalization, and metrics at their configured defaults.
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def _save_state(self) -> None:
        self.settings.setValue("window/geometry", self.saveGeometry())

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.active_worker is not None and self.active_worker.isRunning():
            # Destroying a running QThread aborts the process, and neither a run
            # nor an export can stop midway, so let it finish before the window goes.
            self.status_text.setPlainText("Finishing the current task before closing...")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.active_worker.wait()
            finally:
                QApplication.restoreOverrideCursor()
        if self.restore_settings:
            self._save_state()
        super().closeEvent(event)

    def _on_dataset_preset_selected(self, index: int) -> None:
        preset_name = self.dataset_preset.itemText(index)
        entry = self.dataset_catalog.get(preset_name)
        if entry is None:
            return
        self.dataset_path.setText(str(self.project_root / str(entry["path"])))

    def _on_dataset_path_edited(self, _text: str) -> None:
        self._sync_preset_to_path()

    def _matching_preset_name(self) -> str | None:
        """The catalog preset whose path is the current path, if any."""
        current = self.dataset_path.text().strip()
        if not current:
            return None
        return find_dataset_preset(self.dataset_catalog, current, self.project_root)

    def _sync_preset_to_path(self) -> None:
        """Show the matching preset name for the current path, else Custom."""
        name = self._matching_preset_name()
        row = self.dataset_preset.findText(name) if name is not None else -1
        self.dataset_preset.setCurrentIndex(max(row, 0))

    def _selected_metric_names(self) -> list[str]:
        return [
            metric
            for metric, action in self.metric_actions.items()
            if action.isChecked()
        ]

    def _update_metric_button_text(self) -> None:
        selected = self._selected_metric_names()
        if len(selected) == len(self.metric_actions):
            label = "Metrics: All"
        elif len(selected) == 1:
            label = f"Metrics: {selected[0].title()}"
        elif len(selected) > 1:
            label = f"Metrics: {len(selected)} selected"
        else:
            label = "Metrics: Euclidean (default)"
        self.metrics_button.setText(label)

    def choose_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose dataset",
            str(self.project_root / "datasets"),
            "Datasets (*.csv *.dat *.xlsx *.xls);;All files (*.*)",
        )
        if path:
            self.dataset_path.setText(path)
            self._sync_preset_to_path()

    def _config_from_controls(self):
        cfg = load_default_config(self.config_path)
        dataset_text = self.dataset_path.text().strip()
        if dataset_text:
            preset_name = self._matching_preset_name()
            apply_dataset_selection(
                cfg,
                dataset_text,
                project_root=self.project_root,
                preset=self.dataset_catalog[preset_name] if preset_name is not None else None,
            )

        cfg.preprocessing.normalization.mode = self.normalization.currentText()
        metrics = self._selected_metric_names()
        cfg.metrics.enabled = metrics or ["euclidean"]
        cfg.validate()
        return cfg

    def _busy(self) -> bool:
        return self.active_worker is not None and self.active_worker.isRunning()

    def _start_worker(
        self,
        worker: TaskWorker,
        on_success: Callable[[Any], None],
        on_failure: Callable[[str], None],
        status: str,
    ) -> None:
        """Run a Run or Export task in the background, one at a time."""
        self.run_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_text.setPlainText(status)

        worker.finished_ok.connect(on_success)
        worker.failed.connect(on_failure)
        worker.finished.connect(self._on_worker_done)
        # Each task gets a fresh worker; release it once its thread ends.
        worker.finished.connect(worker.deleteLater)
        self.active_worker = worker
        worker.start()

    def _on_worker_done(self) -> None:
        self.active_worker = None  # its deleteLater is pending
        self.run_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def run_current_pipeline(self) -> None:
        if self._busy():
            return
        try:
            cfg = self._config_from_controls()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid configuration", str(exc))
            self.status_text.setPlainText(str(exc))
            return

        self._start_worker(
            AnalysisWorker(cfg, self.project_root, parent=self),
            self._on_analysis_finished,
            self._on_analysis_failed,
            "Running pipeline and None vs MinMax comparison...",
        )

    def _on_analysis_finished(self, analysis: FullAnalysisResult) -> None:
        self.last_result = analysis.pipeline
        self.normalization_comparison = analysis.comparison
        self.last_output_dir = None
        self.populate_tabs(analysis.pipeline)
        self.open_output_button.setEnabled(False)
        self.status_text.setPlainText(self._status_summary(analysis.pipeline, None))

    def _on_analysis_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Run failed", message)
        self.status_text.setPlainText(message)

    def populate_tabs(self, result: PipelineRunResult) -> None:
        self.tabs.clear()
        branch = result.selected_branch
        self.tabs.addTab(frame_to_table(branch.dataset.to_frame()), "Dataset")
        self.tabs.addTab(
            _metric_tabs(branch.distance_results, lambda item: frame_to_table(_distance_frame(item))),
            "Distances",
        )
        self.tabs.addTab(
            _metric_tabs(branch.neighbor_results, lambda item: frame_to_table(neighbor_combined_frame(item))),
            "Neighbors",
        )
        self.tabs.addTab(_metric_tabs(branch.majority_results, _majority_views), "Majority A/B")
        self.tabs.addTab(_metric_tabs(branch.statistics_results, _to_frame_table), "Statistics")
        self.tabs.addTab(_metric_tabs(branch.membership_results, _to_frame_table), "Membership")
        self.tabs.addTab(_metric_tabs(branch.stability_results, _to_frame_table), "Stability")
        self.tabs.addTab(self._stability_plot_tab(), "Stability Plot")
        self.tabs.addTab(self._synthetic_features_space_tab(branch), "Synthetic Features Space")
        self.tabs.addTab(self._meta_objects_tab(branch), "Meta Objects")
        self.tabs.addTab(frame_to_table(branch.final_comparison.to_frame()), "Final Comparison")

    def _stability_plot_tab(self) -> QWidget:
        widget = QWidget()

        refresh_button = QPushButton("Refresh Plot")
        refresh_button.clicked.connect(self.refresh_stability_plot)

        save_button = QPushButton("Save Plot PNG")
        save_button.clicked.connect(self.save_stability_plot)

        controls = QHBoxLayout()
        controls.addWidget(refresh_button)
        controls.addWidget(save_button)
        controls.addStretch(1)

        self.stability_plot_figure = Figure(figsize=(8, 4.8), facecolor=THEME["figure_bg"])
        self.stability_plot_canvas = FigureCanvasQTAgg(self.stability_plot_figure)

        self.stability_conclusion = QTextEdit()
        self.stability_conclusion.setReadOnly(True)
        self.stability_conclusion.setMaximumHeight(115)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.stability_plot_canvas, 1)
        layout.addWidget(self.stability_conclusion)

        widget.setLayout(layout)
        self.refresh_stability_plot()
        return widget

    def refresh_stability_plot(self) -> None:
        if self.stability_plot_figure is None or self.stability_plot_canvas is None:
            return

        self.stability_plot_figure.clear()
        axis = self.stability_plot_figure.add_subplot(111)
        axis.set_facecolor(THEME["axes_bg"])
        for low, high, color, alpha in STABILITY_BANDS:
            axis.axhspan(low, high, color=color, alpha=alpha)
        for threshold in (0.5, 0.6, 0.8):
            axis.axhline(threshold, color=THEME["plot_grid_line"], linewidth=0.9, linestyle="--")

        if self.last_result is None:
            axis.text(
                0.5,
                0.5,
                "Run the pipeline to draw stability patterns.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            if self.stability_conclusion is not None:
                self.stability_conclusion.setPlainText("No stability data is available yet.")
        else:
            series = prepare_stability_plot_series(self.last_result.selected_branch.stability_results)
            all_k_values: set[int] = set()
            for item in series:
                if not item.points:
                    continue
                x_values = [point.k_value for point in item.points]
                y_values = [point.stability for point in item.points]
                all_k_values.update(x_values)
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=2,
                    label=item.metric_name.title(),
                )
            if all_k_values:
                axis.set_xticks(sorted(all_k_values))
            axis.legend(loc="best")
            if self.stability_conclusion is not None:
                self.stability_conclusion.setPlainText(build_stability_conclusion(series))

        axis.set_title("Stability by reduced k value")
        axis.set_xlabel("Reduced k")
        axis.set_ylabel("Stability")
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, alpha=0.25)
        self.stability_plot_figure.tight_layout()
        self.stability_plot_canvas.draw_idle()

    def save_stability_plot(self) -> None:
        if self.stability_plot_figure is None:
            QMessageBox.information(self, "No plot", "Run the pipeline and refresh the plot first.")
            return
        if self.last_result is None:
            QMessageBox.information(self, "No plot", "Run the pipeline first.")
            return

        self._save_figure(self.stability_plot_figure, "selected", "stability_plot.png")

    def _save_figure(self, figure: Figure, subfolder: str, filename: str) -> None:
        """Save one plot into this run's export folder, which need not be exported yet."""
        assert self.last_result is not None
        output_dir = run_output_dir(self.last_result, self.project_root)
        target_path = output_dir / subfolder / filename
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(target_path, dpi=160, bbox_inches="tight")
        except Exception as exc:
            logger.exception("Saving plot %s failed.", target_path)
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        self.last_output_dir = output_dir
        self.open_output_button.setEnabled(True)
        self._refresh_status()
        QMessageBox.information(self, "Plot saved", str(target_path))

    def _synthetic_features_space_tab(self, branch: PipelineBranchResult) -> QWidget:
        widget = QWidget()

        metric_tabs = _metric_tabs(
            branch.majority_results,
            lambda result: _document_tabs(
                [
                    ("Binary", frame_to_table(build_synthetic_binary_frame(result))),
                    ("Decimal", frame_to_table(build_synthetic_decimal_frame(result))),
                ]
            ),
        )

        layout = QVBoxLayout()
        layout.addWidget(metric_tabs, 1)
        widget.setLayout(layout)
        return widget

    def _meta_objects_tab(self, branch: PipelineBranchResult) -> QWidget:
        widget = QWidget()
        self.meta_object_figures = {}

        meta_objects = build_meta_objects_from_stability(branch.stability_results)
        pca_2d = apply_pca_to_meta_objects(meta_objects, requested_components=2)
        pca_3d = apply_pca_to_meta_objects(meta_objects, requested_components=3)

        save_2d_button = QPushButton("Save PCA 2D PNG")
        save_2d_button.clicked.connect(lambda: self.save_meta_object_plot("pca_2d"))
        save_3d_button = QPushButton("Save PCA 3D PNG")
        save_3d_button.clicked.connect(lambda: self.save_meta_object_plot("pca_3d"))

        controls = QHBoxLayout()
        controls.addWidget(save_2d_button)
        controls.addWidget(save_3d_button)
        controls.addStretch(1)

        tabs = _document_tabs(
            [
                ("Based on Stability", frame_to_table(meta_objects.frame)),
                ("Complexity C(Q)", frame_to_table(branch.complexity_result.to_frame())),
                ("Applying PCA (2D)", self._pca_table_widget(pca_2d)),
                ("Applying PCA (3D)", self._pca_table_widget(pca_3d)),
                ("Visualization (2D)", self._pca_plot_widget(pca_2d, "pca_2d")),
                ("Visualization (3D)", self._pca_plot_widget(pca_3d, "pca_3d")),
                ("None vs MinMax", self._normalization_comparison_widget()),
            ]
        )

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(tabs, 1)
        widget.setLayout(layout)
        return widget

    def _normalization_comparison_widget(self) -> QWidget:
        if self.normalization_comparison is None:
            return self._message_widget(
                "Run the pipeline from the GUI to calculate the None vs MinMax comparison."
            )

        comparison = self.normalization_comparison
        save_2d_button = QPushButton("Save Comparison 2D PNG")
        save_2d_button.clicked.connect(
            lambda: self.save_meta_object_plot("normalization_comparison_2d")
        )
        save_3d_button = QPushButton("Save Comparison 3D PNG")
        save_3d_button.clicked.connect(
            lambda: self.save_meta_object_plot("normalization_comparison_3d")
        )

        controls = QHBoxLayout()
        controls.addWidget(save_2d_button)
        controls.addWidget(save_3d_button)
        controls.addStretch(1)

        tabs = _document_tabs(
            [
                ("Summary", frame_to_table(comparison.summary_frame)),
                (
                    "PCA 2D",
                    self._normalization_comparison_plot_widget(
                        comparison.pca_2d,
                        "normalization_comparison_2d",
                    ),
                ),
                (
                    "PCA 3D",
                    self._normalization_comparison_plot_widget(
                        comparison.pca_3d,
                        "normalization_comparison_3d",
                    ),
                ),
            ]
        )

        widget = QWidget()
        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(tabs, 1)
        widget.setLayout(layout)
        return widget

    def _pca_table_widget(self, result: PCAProjectionResult) -> QWidget:
        if result.available:
            return frame_to_table(result.frame)
        return self._message_widget(result.message)

    def _pca_plot_widget(self, result: PCAProjectionResult, figure_key: str) -> QWidget:
        if not result.available:
            return self._message_widget(result.message)

        figure = Figure(figsize=(7.5, 4.8), facecolor=THEME["figure_bg"])
        self.meta_object_figures[figure_key] = figure
        is_3d = result.requested_components == 3
        # Any: the stubs type add_subplot as 2-D Axes even with projection="3d".
        axis: Any = figure.add_subplot(111, projection="3d") if is_3d else figure.add_subplot(111)
        axis.set_facecolor(THEME["axes_bg"])

        frame = result.frame
        if is_3d:
            points = [
                (float(row["PC1"]), float(row["PC2"]), float(row["PC3"]))
                for _, row in frame.iterrows()
            ]
            offsets = label_offsets_for_points(points)
            x_range = max(float(frame["PC1"].max() - frame["PC1"].min()), 1.0)
            y_range = max(float(frame["PC2"].max() - frame["PC2"].min()), 1.0)
            z_range = max(float(frame["PC3"].max() - frame["PC3"].min()), 1.0)
            axis.scatter(frame["PC1"], frame["PC2"], frame["PC3"], s=70)
            for idx, (_, row) in enumerate(frame.iterrows()):
                dx, dy = offsets[idx]
                axis.text(
                    float(row["PC1"]) + (dx / 1000.0) * x_range,
                    float(row["PC2"]) + (dy / 1000.0) * y_range,
                    float(row["PC3"]) + ((dx - dy) / 1600.0) * z_range,
                    str(row["Metric"]),
                )
            axis.set_zlabel("PC 3")
        else:
            points_2d = [
                (float(row["PC1"]), float(row["PC2"]))
                for _, row in frame.iterrows()
            ]
            offsets = label_offsets_for_points(points_2d)
            axis.scatter(frame["PC1"], frame["PC2"], s=70)
            for idx, (_, row) in enumerate(frame.iterrows()):
                axis.annotate(
                    str(row["Metric"]),
                    (row["PC1"], row["PC2"]),
                    xytext=offsets[idx],
                    textcoords="offset points",
                )

        axis.set_title(f"Meta Objects PCA ({result.requested_components}D)")
        axis.set_xlabel("PC 1")
        axis.set_ylabel("PC 2")
        axis.grid(True, alpha=0.25)
        notes = near_zero_axis_notes(frame)
        if notes:
            if is_3d:
                axis.text2D(
                    0.01,
                    0.01,
                    "\n".join(notes),
                    transform=axis.transAxes,
                    fontsize=9,
                    color=THEME["plot_note"],
                    va="bottom",
                )
            else:
                axis.text(
                    0.01,
                    0.01,
                    "\n".join(notes),
                    transform=axis.transAxes,
                    fontsize=9,
                    color=THEME["plot_note"],
                    va="bottom",
                )
            for column in [col for col in ("PC1", "PC2", "PC3") if col in frame.columns]:
                values = frame[column]
                if float(values.max() - values.min()) <= 1e-12:
                    if column == "PC1":
                        axis.set_xlim(-1.0, 1.0)
                    elif column == "PC2":
                        axis.set_ylim(-1.0, 1.0)
                    elif column == "PC3" and is_3d:
                        axis.set_zlim(-1.0, 1.0)
        return _canvas_widget(figure)

    def _normalization_comparison_plot_widget(
        self,
        result: PCAProjectionResult,
        figure_key: str,
    ) -> QWidget:
        if not result.available:
            return self._message_widget(result.message)

        figure = Figure(figsize=(7.5, 4.8), facecolor=THEME["figure_bg"])
        self.meta_object_figures[figure_key] = figure
        is_3d = result.requested_components == 3
        # Any: the stubs type add_subplot as 2-D Axes even with projection="3d".
        axis: Any = figure.add_subplot(111, projection="3d") if is_3d else figure.add_subplot(111)
        axis.set_facecolor(THEME["axes_bg"])
        frame = result.frame
        metrics = frame["Metric"].drop_duplicates().astype(str).tolist()
        colors = {
            metric: f"C{index % 10}"
            for index, metric in enumerate(metrics)
        }

        for metric in metrics:
            group = frame[frame["Metric"] == metric]
            color = colors[metric]
            if len(group) == 2:
                if is_3d:
                    axis.plot(group["PC1"], group["PC2"], group["PC3"], color=color, alpha=0.6)
                else:
                    axis.plot(group["PC1"], group["PC2"], color=color, alpha=0.6)

        for _, row in frame.iterrows():
            metric = str(row["Metric"])
            normalization = str(row["Normalization"])
            marker = "o" if normalization == "None" else "^"
            color = colors[metric]
            if is_3d:
                axis.scatter(row["PC1"], row["PC2"], row["PC3"], color=color, marker=marker, s=75)
            else:
                axis.scatter(row["PC1"], row["PC2"], color=color, marker=marker, s=75)

        metric_handles = [
            Line2D([0], [0], color=colors[metric], linewidth=2, label=metric)
            for metric in metrics
        ]
        mode_handles = [
            Line2D([0], [0], marker="o", color="#444444", linestyle="None", label="None"),
            Line2D([0], [0], marker="^", color="#444444", linestyle="None", label="MinMax"),
        ]
        first_legend = axis.legend(handles=metric_handles, loc="upper left", title="Metrics")
        axis.add_artist(first_legend)
        axis.legend(handles=mode_handles, loc="upper right", title="Normalization")
        axis.set_xlabel("PC 1")
        axis.set_ylabel("PC 2")
        if is_3d:
            axis.set_zlabel("PC 3", labelpad=10)
            axis.view_init(elev=22, azim=-58)
        axis.grid(True, alpha=0.25)
        notes = near_zero_axis_notes(frame)
        if notes:
            note_text = "\n".join(notes)
            if is_3d:
                axis.text2D(0.01, 0.01, note_text, transform=axis.transAxes, fontsize=9)
            else:
                axis.text(0.01, 0.01, note_text, transform=axis.transAxes, fontsize=9)
        return _canvas_widget(figure)

    def _message_widget(self, message: str) -> QWidget:
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(message)
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(text)
        widget.setLayout(layout)
        return widget

    def save_meta_object_plot(self, figure_key: str) -> None:
        figure = self.meta_object_figures.get(figure_key)
        if figure is None:
            QMessageBox.information(self, "No plot", "This PCA visualization is not available.")
            return
        if self.last_result is None:
            QMessageBox.information(self, "No plot", "Run the pipeline first.")
            return

        subfolder = (
            "normalization_comparison"
            if figure_key.startswith("normalization_comparison_")
            else "selected"
        )
        self._save_figure(figure, subfolder, f"{figure_key}.png")

    def _refresh_status(self) -> None:
        if self.last_result is not None:
            self.status_text.setPlainText(self._status_summary(self.last_result, self.last_output_dir))

    def _status_summary(self, result: PipelineRunResult, output_dir: Path | None) -> str:
        branch = result.selected_branch
        dataset_path = result.source_dataset.source_path or self.dataset_path.text()
        metrics = ", ".join(branch.majority_results.keys())
        output_text = str(output_dir) if output_dir is not None else "Not exported"
        return (
            f"Run: {result.run_id}\n"
            f"Dataset: {dataset_path}\n"
            f"Normalization: {branch.dataset.metadata.get('normalization_mode', 'none')}\n"
            f"Metrics: {metrics}\n"
            f"k_max: {result.metadata.get('formula_kmax')}\n"
            f"Reduced k: {result.metadata.get('reduced_k_values')}\n"
            f"Output folder: {output_text}"
        )

    def export_last_result(self) -> None:
        if self.last_result is None or self.normalization_comparison is None:
            QMessageBox.information(self, "Nothing to export", "Run the pipeline first.")
            return
        if self._busy():
            return

        result, comparison = self.last_result, self.normalization_comparison
        project_root = self.project_root
        self._start_worker(
            TaskWorker(
                lambda: write_analysis_outputs(result, comparison, project_root, gui_export=True),
                "Export",
                parent=self,
            ),
            self._on_export_finished,
            self._on_export_failed,
            "Exporting results...",
        )

    def _on_export_finished(self, output_dir: Path) -> None:
        self.last_output_dir = output_dir
        self.open_output_button.setEnabled(True)
        self._refresh_status()
        QMessageBox.information(self, "Export complete", str(output_dir))

    def _on_export_failed(self, message: str) -> None:
        self._refresh_status()
        QMessageBox.critical(self, "Export failed", message)

    def open_output_folder(self) -> None:
        if self.last_output_dir is None:
            QMessageBox.information(self, "No output folder", "Export first.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output_dir)))


def main() -> None:  # pragma: no cover - GUI entrypoint
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    project_root = PROJECT_ROOT
    app = QApplication([])
    app.setOrganizationName("synthetic-bit-sequence-majority-rule")
    app.setApplicationName("DesktopApp")
    app.setStyle("Fusion")
    app.setStyleSheet(build_stylesheet())
    window = MainWindow(project_root)
    window.show()
    app.exec()

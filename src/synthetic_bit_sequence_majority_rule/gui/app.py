from __future__ import annotations

import re
from numbers import Real
from pathlib import Path
from typing import Callable

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
from synthetic_bit_sequence_majority_rule.gui.meta_objects import (
    NormalizationComparisonResult,
    PCAProjectionResult,
    apply_pca_to_meta_objects,
    build_normalization_comparison,
    build_meta_objects_from_stability,
    label_offsets_for_points,
    near_zero_axis_notes,
)
from synthetic_bit_sequence_majority_rule.gui.stability_plot import (
    build_stability_conclusion,
    prepare_stability_plot_series,
)
from synthetic_bit_sequence_majority_rule.gui.synthetic_features import (
    build_synthetic_binary_frame,
    build_synthetic_decimal_frame,
)
from synthetic_bit_sequence_majority_rule.io.configs import load_default_config
from synthetic_bit_sequence_majority_rule.io.writers import (
    write_normalization_comparison_outputs,
    write_pipeline_outputs,
)
from synthetic_bit_sequence_majority_rule.services.runner import (
    PipelineBranchResult,
    PipelineRunResult,
    run_none_minmax_comparison,
    run_pipeline,
)


def _require_pyqt6() -> tuple[object, ...]:
    try:
        from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QUrl, Qt
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
            QPushButton,
            QTabWidget,
            QTableView,
            QTextEdit,
            QToolButton,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:  # pragma: no cover - depends on local GUI install
        raise RuntimeError("PyQt6 is required to launch the desktop GUI.") from exc

    return (
        QUrl,
        Qt,
        QAbstractTableModel,
        QModelIndex,
        QAction,
        QDesktopServices,
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
        QPushButton,
        QTabWidget,
        QTableView,
        QTextEdit,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )


(
    QUrl,
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QAction,
    QDesktopServices,
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
    QPushButton,
    QTabWidget,
    QTableView,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
) = _require_pyqt6()


GRAY_APP_STYLESHEET = """
QMainWindow,
QWidget {
    background-color: #eeeeee;
    color: #202020;
}

QLineEdit,
QComboBox,
QTextEdit,
QTabWidget::pane,
QTableView {
    background-color: #fafafa;
    color: #202020;
    border: 1px solid #b8b8b8;
}

QPushButton,
QToolButton,
QTabBar::tab {
    background-color: #dddddd;
    color: #202020;
    border: 1px solid #aaaaaa;
    padding: 5px 10px;
}

QPushButton:hover,
QToolButton:hover,
QTabBar::tab:selected {
    background-color: #f4f4f4;
}

QHeaderView::section {
    background-color: #d6d6d6;
    color: #202020;
    border: 1px solid #b8b8b8;
    padding: 4px;
}

QTableView::item {
    background-color: #ffffff;
}

QTableView::item:alternate {
    background-color: #f0f0f0;
}

QTableView::item:selected {
    background-color: #1f7fcf;
    color: #ffffff;
}

QScrollBar:vertical,
QScrollBar:horizontal {
    background-color: #e0e0e0;
}

QMenu {
    background-color: #fafafa;
    color: #202020;
    border: 1px solid #aaaaaa;
}

QMenu::item:selected {
    background-color: #dcecff;
    color: #202020;
}
"""


def _display_value(value: object, decimals: int) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return str(value)


def _natural_sort_key(value: object) -> tuple[object, ...]:
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
        self._source_frame = frame.reset_index(drop=True).copy()
        self.frame = self._source_frame.copy()
        self.decimals = decimals

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return _display_value(
                self.frame.iat[index.row(), index.column()],
                self.decimals,
            )
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
            sorted_frame = self._source_frame.copy()
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


def frame_to_table(frame: pd.DataFrame, *, decimals: int = 6) -> QTableView:
    model = DataFrameTableModel(frame, decimals=decimals)
    table = QTableView()
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
    horizontal.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    horizontal.setStretchLastSection(False)
    vertical = table.verticalHeader()
    vertical.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
    vertical.setDefaultSectionSize(26)
    if model.rowCount() * model.columnCount() <= 50_000:
        table.resizeColumnsToContents()
    else:
        for col_idx, column in enumerate(model.frame.columns):
            table.setColumnWidth(col_idx, min(180, max(72, len(str(column)) * 9 + 24)))
    horizontal.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
    table.setSortingEnabled(True)
    return table


def _distance_frame(distance_result) -> pd.DataFrame:
    frame = distance_result.to_frame().reset_index()
    return frame.rename(columns={"index": "Object"})


class MainWindow(QMainWindow):
    AVAILABLE_METRICS = ("euclidean", "chebyshev", "canberra", "manhattan")

    def __init__(self, project_root: Path) -> None:
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

        self.setWindowTitle("Synthetic Bit Sequence Majority Rule")
        self.resize(1440, 860)

        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("Dataset path from config")

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

        run_button = QPushButton("Run")
        run_button.clicked.connect(self.run_current_pipeline)

        export_button = QPushButton("Export")
        export_button.clicked.connect(self.export_last_result)

        self.open_output_button = QPushButton("Open Output Folder")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self.open_output_folder)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Dataset"))
        controls.addWidget(self.dataset_path, 1)
        controls.addWidget(browse_button)
        controls.addWidget(QLabel("Normalization"))
        controls.addWidget(self.normalization)
        controls.addWidget(self.metrics_button)
        controls.addWidget(run_button)
        controls.addWidget(export_button)
        controls.addWidget(self.open_output_button)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.status_text)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)
        self.load_defaults()

    def load_defaults(self) -> None:
        cfg = load_default_config(self.config_path)
        self.dataset_path.setText(str(self.project_root / cfg.dataset.path))
        self.normalization.setCurrentText(cfg.preprocessing.normalization.mode)
        for metric, action in self.metric_actions.items():
            action.setChecked(metric in cfg.enabled_metrics)
        self._update_metric_button_text()

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

    def _config_from_controls(self):
        cfg = load_default_config(self.config_path)
        dataset = Path(self.dataset_path.text().strip())
        if dataset:
            try:
                cfg.dataset.path = dataset.relative_to(self.project_root)
            except ValueError:
                cfg.dataset.path = dataset
            suffix = dataset.suffix.lower().lstrip(".")
            if suffix:
                cfg.dataset.format = suffix
                cfg.dataset.has_header = suffix != "dat"

        cfg.preprocessing.normalization.mode = self.normalization.currentText()
        metrics = self._selected_metric_names()
        cfg.metrics.enabled = metrics or ["euclidean"]
        cfg.validate()
        return cfg

    def run_current_pipeline(self) -> None:
        try:
            cfg = self._config_from_controls()
            self.status_text.setPlainText("Running pipeline and None vs MinMax comparison...")
            result = run_pipeline(cfg)
            comparison_run = run_none_minmax_comparison(cfg, result)
            self.normalization_comparison = self._build_normalization_comparison(
                comparison_run
            )
            self.last_result = result
            self.populate_tabs(result)
            self.last_output_dir = write_pipeline_outputs(
                result,
                self.project_root / result.config.run.output_root,
            )
            write_normalization_comparison_outputs(
                self.normalization_comparison,
                self.last_output_dir,
            )
            self.open_output_button.setEnabled(True)
            self.status_text.setPlainText(self._status_summary(result, self.last_output_dir))
        except Exception as exc:
            QMessageBox.critical(self, "Run failed", str(exc))
            self.status_text.setPlainText(str(exc))

    @staticmethod
    def _build_normalization_comparison(
        comparison_run: PipelineRunResult,
    ) -> NormalizationComparisonResult:
        raw = comparison_run.branches["raw"]
        minmax = comparison_run.branches["normalized"]
        return build_normalization_comparison(
            raw.stability_results,
            minmax.stability_results,
            raw.complexity_result,
            minmax.complexity_result,
        )

    def populate_tabs(self, result: PipelineRunResult) -> None:
        self.tabs.clear()
        branch = result.selected_branch
        self.tabs.addTab(frame_to_table(branch.dataset.to_frame()), "Dataset")
        self.tabs.addTab(frame_to_table(branch.final_comparison.to_frame()), "Final Comparison")
        self.tabs.addTab(self._distance_tabs(branch), "Distances")
        self.tabs.addTab(self._neighbor_tabs(branch), "Neighbors")
        self.tabs.addTab(self._majority_tabs(branch), "Majority A/B")
        self.tabs.addTab(self._simple_metric_tabs(branch.statistics_results, lambda item: item.to_frame()), "Statistics")
        self.tabs.addTab(self._simple_metric_tabs(branch.membership_results, lambda item: item.to_frame()), "Membership")
        self.tabs.addTab(self._simple_metric_tabs(branch.stability_results, lambda item: item.to_frame()), "Stability")
        self.tabs.addTab(self._stability_plot_tab(), "Stability Plot")
        self.tabs.addTab(self._synthetic_features_space_tab(branch), "Synthetic Features Space")
        self.tabs.addTab(self._meta_objects_tab(branch), "Meta Objects")

    def _simple_metric_tabs(
        self,
        mapping: dict[str, object],
        frame_builder: Callable[[object], pd.DataFrame],
    ) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        for metric_name, item in mapping.items():
            tabs.addTab(frame_to_table(frame_builder(item)), metric_name.title())
        return tabs

    def _distance_tabs(self, branch: PipelineBranchResult) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        for metric_name, distance_result in branch.distance_results.items():
            tabs.addTab(frame_to_table(_distance_frame(distance_result)), metric_name.title())
        return tabs

    def _neighbor_tabs(self, branch: PipelineBranchResult) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        for metric_name, neighbor_result in branch.neighbor_results.items():
            tabs.addTab(frame_to_table(neighbor_combined_frame(neighbor_result)), metric_name.title())
        return tabs

    def _majority_tabs(self, branch: PipelineBranchResult) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        for metric_name, majority_result in branch.majority_results.items():
            metric_tabs = QTabWidget()
            metric_tabs.setDocumentMode(True)
            metric_tabs.addTab(frame_to_table(same_class_indicator_frame(majority_result)), "Same Class")
            metric_tabs.addTab(frame_to_table(a_full_frame(majority_result)), "A Full")
            metric_tabs.addTab(frame_to_table(a_reduced_frame(majority_result)), "A Reduced")
            metric_tabs.addTab(frame_to_table(b_full_frame(majority_result)), "B Full")
            metric_tabs.addTab(frame_to_table(b_reduced_frame(majority_result)), "B Reduced")
            tabs.addTab(metric_tabs, metric_name.title())
        return tabs

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

        self.stability_plot_figure = Figure(figsize=(8, 4.8), facecolor="#fafafa")
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
        axis.set_facecolor("#ffffff")
        axis.axhspan(0.0, 0.5, color="#f5b7b1", alpha=0.18)
        axis.axhspan(0.5, 0.6, color="#f9e79f", alpha=0.25)
        axis.axhspan(0.6, 0.8, color="#abebc6", alpha=0.22)
        axis.axhspan(0.8, 1.0, color="#aed6f1", alpha=0.24)
        for threshold in (0.5, 0.6, 0.8):
            axis.axhline(threshold, color="#777777", linewidth=0.9, linestyle="--")

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
        if self.last_output_dir is None:
            QMessageBox.information(self, "No output folder", "Run or export first.")
            return

        target_dir = self.last_output_dir / "selected"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / "stability_plot.png"
        self.stability_plot_figure.savefig(target_path, dpi=160, bbox_inches="tight")
        QMessageBox.information(self, "Plot saved", str(target_path))

    def _synthetic_features_space_tab(self, branch: PipelineBranchResult) -> QWidget:
        widget = QWidget()

        metric_tabs = QTabWidget()
        metric_tabs.setDocumentMode(True)
        for metric_name, majority_result in branch.majority_results.items():
            kind_tabs = QTabWidget()
            kind_tabs.setDocumentMode(True)
            kind_tabs.addTab(
                frame_to_table(build_synthetic_binary_frame(majority_result)),
                "Binary",
            )
            kind_tabs.addTab(
                frame_to_table(build_synthetic_decimal_frame(majority_result)),
                "Decimal",
            )
            metric_tabs.addTab(kind_tabs, metric_name.title())

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

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(frame_to_table(meta_objects.frame), "Based on Stability")
        tabs.addTab(
            frame_to_table(branch.complexity_result.to_frame()),
            "Complexity C(Q)",
        )
        tabs.addTab(self._pca_table_widget(pca_2d), "Applying PCA (2D)")
        tabs.addTab(self._pca_table_widget(pca_3d), "Applying PCA (3D)")
        tabs.addTab(self._pca_plot_widget(pca_2d, "pca_2d"), "Visualization (2D)")
        tabs.addTab(self._pca_plot_widget(pca_3d, "pca_3d"), "Visualization (3D)")
        tabs.addTab(self._normalization_comparison_widget(), "None vs MinMax")

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

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(frame_to_table(comparison.summary_frame), "Summary")
        tabs.addTab(
            self._normalization_comparison_plot_widget(
                comparison.pca_2d,
                "normalization_comparison_2d",
            ),
            "PCA 2D",
        )
        tabs.addTab(
            self._normalization_comparison_plot_widget(
                comparison.pca_3d,
                "normalization_comparison_3d",
            ),
            "PCA 3D",
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

        figure = Figure(figsize=(7.5, 4.8), facecolor="#fafafa")
        self.meta_object_figures[figure_key] = figure
        is_3d = result.requested_components == 3
        axis = figure.add_subplot(111, projection="3d") if is_3d else figure.add_subplot(111)
        axis.set_facecolor("#ffffff")

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
            points = [
                (float(row["PC1"]), float(row["PC2"]))
                for _, row in frame.iterrows()
            ]
            offsets = label_offsets_for_points(points)
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
                    color="#555555",
                    va="bottom",
                )
            else:
                axis.text(
                    0.01,
                    0.01,
                    "\n".join(notes),
                    transform=axis.transAxes,
                    fontsize=9,
                    color="#555555",
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
        figure.tight_layout()

        canvas = FigureCanvasQTAgg(figure)
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(canvas, 1)
        widget.setLayout(layout)
        canvas.draw_idle()
        return widget

    def _normalization_comparison_plot_widget(
        self,
        result: PCAProjectionResult,
        figure_key: str,
    ) -> QWidget:
        if not result.available:
            return self._message_widget(result.message)

        figure = Figure(figsize=(7.5, 4.8), facecolor="#fafafa")
        self.meta_object_figures[figure_key] = figure
        is_3d = result.requested_components == 3
        axis = figure.add_subplot(111, projection="3d") if is_3d else figure.add_subplot(111)
        axis.set_facecolor("#ffffff")
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
        figure.tight_layout()

        canvas = FigureCanvasQTAgg(figure)
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(canvas, 1)
        widget.setLayout(layout)
        canvas.draw_idle()
        return widget

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
        if self.last_output_dir is None:
            QMessageBox.information(self, "No output folder", "Run or export first.")
            return

        if figure_key.startswith("normalization_comparison_"):
            target_dir = self.last_output_dir / "normalization_comparison"
        else:
            target_dir = self.last_output_dir / "selected"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{figure_key}.png"
        figure.savefig(target_path, dpi=160, bbox_inches="tight")
        QMessageBox.information(self, "Plot saved", str(target_path))

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
        if self.last_result is None:
            QMessageBox.information(self, "Nothing to export", "Run the pipeline first.")
            return
        try:
            self.last_output_dir = write_pipeline_outputs(
                self.last_result,
                self.project_root / self.last_result.config.run.output_root,
            )
            if self.normalization_comparison is not None:
                write_normalization_comparison_outputs(
                    self.normalization_comparison,
                    self.last_output_dir,
                )
            self.open_output_button.setEnabled(True)
            self.status_text.setPlainText(self._status_summary(self.last_result, self.last_output_dir))
            QMessageBox.information(self, "Export complete", str(self.last_output_dir))
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def open_output_folder(self) -> None:
        if self.last_output_dir is None:
            QMessageBox.information(self, "No output folder", "Run or export first.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output_dir)))


def main() -> None:  # pragma: no cover - GUI entrypoint
    project_root = Path(__file__).resolve().parents[3]
    app = QApplication([])
    app.setStyleSheet(GRAY_APP_STYLESHEET)
    window = MainWindow(project_root)
    window.show()
    app.exec()

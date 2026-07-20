"""Single source of truth for GUI colors, shared by the Qt stylesheet and matplotlib."""

from __future__ import annotations


THEME: dict[str, str] = {
    "window_bg": "#f5f6f8",
    "surface": "#ffffff",
    "surface_alt": "#f3f5f8",
    "border": "#d0d4da",
    "border_strong": "#b6bcc6",
    "text": "#1f2430",
    "text_muted": "#5b6472",
    "accent": "#2563eb",
    "accent_hover": "#1d4ed8",
    "accent_pressed": "#1e40af",
    "accent_soft": "#dbeafe",
    "header_bg": "#e8ebf0",
    "hover_bg": "#eef2f7",
    "pressed_bg": "#e2e8f0",
    "disabled_bg": "#e9ecf0",
    "disabled_text": "#9aa2ae",
    "figure_bg": "#ffffff",
    "axes_bg": "#ffffff",
    "plot_note": "#5b6472",
    "plot_grid_line": "#777777",
}

# Stability band fills keep their semantic colors (red/yellow/green/blue bands).
STABILITY_BANDS: list[tuple[float, float, str, float]] = [
    (0.0, 0.5, "#f5b7b1", 0.18),
    (0.5, 0.6, "#f9e79f", 0.25),
    (0.6, 0.8, "#abebc6", 0.22),
    (0.8, 1.0, "#aed6f1", 0.24),
]


def build_stylesheet(theme: dict[str, str] = THEME) -> str:
    t = theme
    return f"""
QMainWindow,
QWidget {{
    background-color: {t["window_bg"]};
    color: {t["text"]};
    font-size: 13px;
}}

QLabel {{
    background: transparent;
}}

QLineEdit,
QComboBox,
QTextEdit {{
    background-color: {t["surface"]};
    color: {t["text"]};
    border: 1px solid {t["border"]};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {t["accent"]};
    selection-color: {t["surface"]};
}}

QLineEdit:focus,
QComboBox:focus,
QTextEdit:focus {{
    border: 1px solid {t["accent"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

QComboBox QAbstractItemView {{
    background-color: {t["surface"]};
    border: 1px solid {t["border"]};
    selection-background-color: {t["accent_soft"]};
    selection-color: {t["text"]};
}}

QPushButton,
QToolButton {{
    background-color: {t["surface"]};
    color: {t["text"]};
    border: 1px solid {t["border"]};
    border-radius: 6px;
    padding: 6px 14px;
}}

QPushButton:hover,
QToolButton:hover {{
    background-color: {t["hover_bg"]};
    border-color: {t["border_strong"]};
}}

QPushButton:pressed,
QToolButton:pressed {{
    background-color: {t["pressed_bg"]};
}}

QPushButton:disabled,
QToolButton:disabled {{
    background-color: {t["disabled_bg"]};
    color: {t["disabled_text"]};
    border-color: {t["border"]};
}}

QPushButton#primaryButton {{
    background-color: {t["accent"]};
    color: {t["surface"]};
    border: 1px solid {t["accent"]};
    font-weight: 600;
}}

QPushButton#primaryButton:hover {{
    background-color: {t["accent_hover"]};
    border-color: {t["accent_hover"]};
}}

QPushButton#primaryButton:pressed {{
    background-color: {t["accent_pressed"]};
}}

QPushButton#primaryButton:disabled {{
    background-color: {t["disabled_bg"]};
    color: {t["disabled_text"]};
    border-color: {t["border"]};
}}

QTabWidget::pane {{
    background-color: {t["surface"]};
    border: 1px solid {t["border"]};
    border-radius: 6px;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {t["text_muted"]};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 7px 14px;
    margin-right: 2px;
}}

QTabBar::tab:hover {{
    color: {t["text"]};
}}

QTabBar::tab:selected {{
    color: {t["accent"]};
    border-bottom: 2px solid {t["accent"]};
    font-weight: 600;
}}

QTableView {{
    background-color: {t["surface"]};
    alternate-background-color: {t["surface_alt"]};
    color: {t["text"]};
    border: 1px solid {t["border"]};
    gridline-color: {t["border"]};
    selection-background-color: {t["accent"]};
    selection-color: {t["surface"]};
}}

QHeaderView::section {{
    background-color: {t["header_bg"]};
    color: {t["text"]};
    border: none;
    border-right: 1px solid {t["border"]};
    border-bottom: 1px solid {t["border"]};
    padding: 5px 8px;
    font-weight: 600;
}}

QTableCornerButton::section {{
    background-color: {t["header_bg"]};
    border: none;
    border-right: 1px solid {t["border"]};
    border-bottom: 1px solid {t["border"]};
}}

QProgressBar {{
    background-color: {t["surface_alt"]};
    border: 1px solid {t["border"]};
    border-radius: 6px;
    text-align: center;
    max-height: 14px;
}}

QProgressBar::chunk {{
    background-color: {t["accent"]};
    border-radius: 5px;
}}

QScrollBar:vertical {{
    background: {t["window_bg"]};
    width: 12px;
    margin: 0;
}}

QScrollBar:horizontal {{
    background: {t["window_bg"]};
    height: 12px;
    margin: 0;
}}

QScrollBar::handle:vertical,
QScrollBar::handle:horizontal {{
    background: {t["border_strong"]};
    border-radius: 5px;
    min-height: 24px;
    min-width: 24px;
}}

QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover {{
    background: {t["text_muted"]};
}}

QScrollBar::add-line,
QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}

QMenu {{
    background-color: {t["surface"]};
    color: {t["text"]};
    border: 1px solid {t["border"]};
    padding: 4px;
}}

QMenu::item {{
    padding: 5px 22px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {t["accent_soft"]};
    color: {t["text"]};
}}
"""

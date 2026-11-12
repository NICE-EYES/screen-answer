"""Цветовые темы, таблица стилей и иконка приложения (рисуется кодом, без файлов)."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QIcon, QLinearGradient, QPainter, QPen, QPixmap

PALETTES = {
    "dark": {
        "bg": "#16181d", "surface": "#22252c", "border": "#343842", "text": "#e6e8eb",
        "muted": "#9aa0a8", "accent": "#4f8cff", "accent_text": "#ffffff", "input": "#101216",
        "hint_bg": "#1d2633",
    },
    "light": {
        "bg": "#f6f7f9", "surface": "#ffffff", "border": "#d0d5dc", "text": "#1f2328",
        "muted": "#5f6670", "accent": "#2f6fed", "accent_text": "#ffffff", "input": "#ffffff",
        "hint_bg": "#e8f0fe",
    },
}

STATUS_COLORS = {
    "idle": "#8a9099", "capture": "#f0b429", "ocr": "#f0b429", "ai": "#f0b429", "ready": "#3fb950",
    "nochange": "#58a6ff", "error": "#f85149", "stopped": "#8a9099", "recording": "#f85149",
}


def build_stylesheet(theme: str, font_size: int = 10) -> str:
    c = PALETTES.get(theme, PALETTES["dark"])
    return f"""
    QWidget {{ color: {c['text']}; font-family: "Segoe UI", "Segoe UI Emoji", sans-serif; font-size: {font_size}pt; }}
    QDialog, QMessageBox {{ background: {c['bg']}; }}
    QFrame#Root {{ background: {c['bg']}; border: 1px solid {c['border']}; border-radius: 10px; }}
    QLabel#Title {{ font-weight: 600; }}
    QLabel#Muted {{ color: {c['muted']}; font-size: {max(7, font_size - 1)}pt; }}
    QLabel#Hint {{ background: {c['hint_bg']}; border-radius: 6px; padding: 4px 6px; }}
    QPushButton, QToolButton {{ background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: 6px; padding: 3px 8px; }}
    QPushButton:hover, QToolButton:hover {{ border-color: {c['accent']}; }}
    QPushButton:pressed, QToolButton:pressed {{ background: {c['border']}; }}
    QPushButton:checked, QToolButton:checked {{ border-color: {c['accent']}; background: {c['hint_bg']}; }}
    QPushButton:disabled {{ color: {c['muted']}; }}
    QPushButton#Primary {{ background: {c['accent']}; color: {c['accent_text']}; border: none; font-weight: 600;
        padding: 4px 12px; }}
    QPushButton#Primary:hover {{ background: {QColor(c['accent']).lighter(115).name()}; }}
    QPushButton#Danger:hover {{ border-color: #f85149; color: #f85149; }}
    QToolButton#TitleButton {{ background: transparent; border: none; padding: 2px 6px; border-radius: 4px; }}
    QToolButton#TitleButton:hover {{ background: {c['surface']}; }}
    QToolButton#Section {{ background: transparent; border: none; color: {c['muted']}; padding: 1px 0; }}
    QToolButton#Section:hover {{ color: {c['text']}; }}
    QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {c['input']}; border: 1px solid {c['border']}; border-radius: 6px; padding: 3px;
        selection-background-color: {c['accent']}; selection-color: {c['accent_text']}; }}
    QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus {{ border-color: {c['accent']}; }}
    QComboBox QAbstractItemView {{ background: {c['surface']}; border: 1px solid {c['border']};
        selection-background-color: {c['accent']}; }}
    QListWidget::item {{ padding: 3px; }}
    QListWidget::item:selected {{ background: {c['accent']}; color: {c['accent_text']}; }}
    QSlider::groove:horizontal {{ height: 4px; background: {c['border']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
        background: {c['accent']}; }}
    QProgressBar {{ background: transparent; border: none; max-height: 3px; min-height: 3px; }}
    QProgressBar::chunk {{ background: {c['accent']}; }}
    QMenu {{ background: {c['surface']}; border: 1px solid {c['border']}; padding: 3px; }}
    QMenu::item {{ padding: 4px 22px 4px 18px; border-radius: 4px; }}
    QMenu::item:selected {{ background: {c['accent']}; color: {c['accent_text']}; }}
    QMenu::separator {{ height: 1px; background: {c['border']}; margin: 3px 6px; }}
    QScrollBar:vertical {{ width: 8px; background: transparent; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {c['border']}; border-radius: 4px; min-height: 24px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{ height: 8px; background: transparent; }}
    QScrollBar::handle:horizontal {{ background: {c['border']}; border-radius: 4px; min-width: 24px; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QTabWidget::pane {{ border: 1px solid {c['border']}; border-radius: 6px; top: -1px; background: {c['bg']}; }}
    QTabBar::tab {{ background: {c['surface']}; border: 1px solid {c['border']}; padding: 5px 10px;
        border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }}
    QTabBar::tab:selected {{ background: {c['bg']}; border-bottom-color: {c['bg']}; color: {c['accent']}; }}
    QGroupBox {{ border: 1px solid {c['border']}; border-radius: 6px; margin-top: 10px; padding-top: 6px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {c['muted']}; }}
    QToolTip {{ background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border']}; }}
    """


def make_app_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 20, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        grad = QLinearGradient(0, 0, size, size)
        grad.setColorAt(0, QColor("#4f8cff"))
        grad.setColorAt(1, QColor("#7b5cff"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(QRectF(0.5, 0.5, size - 1, size - 1), size * 0.22, size * 0.22)
        pen = QPen(QColor("white"))
        pen.setWidthF(max(1.0, size / 18))
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        screen = QRectF(size * 0.17, size * 0.2, size * 0.66, size * 0.46)
        p.drawRoundedRect(screen, size * 0.06, size * 0.06)
        p.drawLine(QPointF(size * 0.5, size * 0.66), QPointF(size * 0.5, size * 0.77))
        p.drawLine(QPointF(size * 0.34, size * 0.79), QPointF(size * 0.66, size * 0.79))
        font = QFont("Segoe UI")
        font.setBold(True)
        font.setPixelSize(max(6, int(size * 0.34)))
        p.setFont(font)
        p.drawText(screen, Qt.AlignmentFlag.AlignCenter, "?")
        p.end()
        icon.addPixmap(pm)
    return icon

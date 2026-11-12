"""Захват экрана (mss), определение изменений и выбор области мышью.

Модуль только ЧИТАЕТ экран и геометрию окон. Он не кликает, не переключает
окна и не меняет их состояние.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
from dataclasses import dataclass

from PIL import Image, ImageChops

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"


class CaptureError(Exception):
    """Понятная пользователю ошибка захвата."""


@dataclass(frozen=True)
class Rect:
    """Прямоугольник в физических пикселях виртуального рабочего стола."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def is_valid(self, min_size: int = 8) -> bool:
        return self.width >= min_size and self.height >= min_size

    def intersection(self, other: "Rect") -> "Rect | None":
        left, top = max(self.left, other.left), max(self.top, other.top)
        right, bottom = min(self.right, other.right), min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return None
        return Rect(left, top, right - left, bottom - top)

    def intersects(self, other: "Rect") -> bool:
        return self.intersection(other) is not None

    def to_mss(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def to_list(self) -> list[int]:
        return [self.left, self.top, self.width, self.height]

    @classmethod
    def from_points(cls, x1: int, y1: int, x2: int, y2: int) -> "Rect":
        return cls(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    def __str__(self) -> str:
        return f"{self.width}×{self.height}"


# ----------------------------------------------------------------------
# Win32: только чтение информации об окнах
# ----------------------------------------------------------------------
if IS_WINDOWS:
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _dwmapi = ctypes.WinDLL("dwmapi")

    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    _user32.IsWindow.argtypes = [wintypes.HWND]
    _user32.IsWindowVisible.argtypes = [wintypes.HWND]
    _user32.IsIconic.argtypes = [wintypes.HWND]
    _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    _dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long

    _DWMWA_EXTENDED_FRAME_BOUNDS = 9


def get_cursor_pos() -> tuple[int, int] | None:
    """Физические координаты курсора (только чтение)."""
    if not IS_WINDOWS:
        return None
    pt = wintypes.POINT()
    if _user32.GetCursorPos(ctypes.byref(pt)):
        return int(pt.x), int(pt.y)
    return None


def get_foreground_window() -> int | None:
    if not IS_WINDOWS:
        return None
    hwnd = _user32.GetForegroundWindow()
    return int(hwnd) if hwnd else None


def is_own_window(hwnd: int) -> bool:
    """Окно принадлежит нашему процессу (оверлей, настройки, меню трея)."""
    if not IS_WINDOWS or not hwnd:
        return False
    pid = wintypes.DWORD(0)
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value) == os.getpid()


def is_capturable_window(hwnd: int | None) -> bool:
    if not IS_WINDOWS or not hwnd:
        return False
    return bool(_user32.IsWindow(hwnd)) and bool(_user32.IsWindowVisible(hwnd)) and not _user32.IsIconic(hwnd)


def get_window_title(hwnd: int | None) -> str:
    if not IS_WINDOWS or not hwnd:
        return ""
    length = _user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def get_window_rect(hwnd: int | None, use_dwm: bool = True) -> Rect | None:
    """Видимые границы окна в физических пикселях (без невидимых рамок Windows 10/11)."""
    if not IS_WINDOWS or not hwnd:
        return None
    rect = wintypes.RECT()
    ok = False
    if use_dwm:
        ok = _dwmapi.DwmGetWindowAttribute(
            hwnd, _DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)) == 0
    if not ok:
        ok = bool(_user32.GetWindowRect(hwnd, ctypes.byref(rect)))
    if not ok:
        return None
    r = Rect.from_points(rect.left, rect.top, rect.right, rect.bottom)
    return r if r.is_valid(4) else None


def list_monitors() -> list[Rect]:
    """[0] — весь виртуальный экран, [1..] — отдельные мониторы (как в mss)."""
    try:
        import mss
    except ImportError as exc:
        raise CaptureError("Модуль mss не установлен: pip install mss") from exc
    try:
        with mss.mss() as sct:
            return [Rect(m["left"], m["top"], m["width"], m["height"]) for m in sct.monitors]
    except Exception as exc:
        raise CaptureError(f"Не удалось получить список мониторов: {exc}") from exc


# ----------------------------------------------------------------------
# Захват
# ----------------------------------------------------------------------
class ScreenGrabber:
    """Обёртка над mss. Создаётся и используется в одном (рабочем) потоке."""

    def __init__(self) -> None:
        self._sct = None

    def _ensure(self):
        if self._sct is None:
            try:
                import mss
            except ImportError as exc:
                raise CaptureError("Модуль mss не установлен: pip install mss") from exc
            self._sct = mss.mss()
        return self._sct

    def grab(self, rect: Rect) -> Image.Image:
        for attempt in (1, 2):
            sct = self._ensure()
            try:
                v = sct.monitors[0]
                virtual = Rect(v["left"], v["top"], v["width"], v["height"])
                clipped = rect.intersection(virtual)
                if clipped is None or not clipped.is_valid(4):
                    raise CaptureError("Выбранная область находится за пределами экрана. Выберите область заново.")
                shot = sct.grab(clipped.to_mss())
                return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            except CaptureError:
                raise
            except Exception as exc:
                # Конфигурация мониторов могла измениться — пересоздаём mss один раз.
                log.warning("Ошибка захвата (попытка %s): %s", attempt, exc)
                self.close()
                if attempt == 2:
                    raise CaptureError(f"Не удалось сделать скриншот: {exc}") from exc
        raise CaptureError("Не удалось сделать скриншот")

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None


class ChangeDetector:
    """Сравнивает скриншоты по уменьшенной копии: доля заметно изменённых пикселей."""

    THUMB_WIDTH = 320
    PIXEL_DELTA = 24  # разница яркости, начиная с которой пиксель считается изменённым

    def __init__(self, threshold_percent: float = 0.1) -> None:
        self.threshold_percent = threshold_percent
        self._reference: Image.Image | None = None

    def reset(self) -> None:
        self._reference = None

    def _thumb(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        scale = min(1.0, self.THUMB_WIDTH / max(1, w))
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.convert("L").resize(size, Image.Resampling.BILINEAR)

    def compare(self, image: Image.Image) -> tuple[bool, float]:
        """Возвращает (изменилось, процент изменённых пикселей).

        Эталон обновляется только при изменении — медленные постепенные
        изменения тоже будут замечены.
        """
        thumb = self._thumb(image)
        ref = self._reference
        if ref is None or ref.size != thumb.size:
            self._reference = thumb
            return True, 100.0
        hist = ImageChops.difference(thumb, ref).histogram()
        changed_px = sum(hist[self.PIXEL_DELTA:])
        ratio = changed_px * 100.0 / max(1, thumb.size[0] * thumb.size[1])
        changed = ratio >= self.threshold_percent and changed_px > 0
        if changed:
            self._reference = thumb
        return changed, ratio


# ----------------------------------------------------------------------
# Выбор области мышью (Qt)
# ----------------------------------------------------------------------
from PyQt6.QtCore import QObject, QPoint, QRect, Qt, pyqtSignal  # noqa: E402
from PyQt6.QtGui import QColor, QCursor, QFont, QGuiApplication, QPainter, QPen  # noqa: E402
from PyQt6.QtWidgets import QWidget  # noqa: E402


class _SelectionWindow(QWidget):
    """Полупрозрачная затемняющая подложка на одном мониторе."""

    def __init__(self, screen, owner: "RegionSelector") -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self._screen = screen
        self._owner = owner
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self._origin_phys: tuple[int, int] | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setGeometry(screen.geometry())

    def _physical(self, event) -> tuple[int, int]:
        pos = get_cursor_pos()  # в Windows — точные физические координаты (как у mss)
        if pos is not None:
            return pos
        gp = event.globalPosition()
        dpr = self._screen.devicePixelRatio() or 1.0
        return int(gp.x() * dpr), int(gp.y() * dpr)

    def _selection(self) -> QRect | None:
        if self._origin is None or self._current is None:
            return None
        return QRect(self._origin, self._current).normalized()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(0, 0, 0, 110))
        sel = self._selection()
        if sel is not None:
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            # почти прозрачная (не полностью — чтобы окно продолжало получать мышь)
            p.fillRect(sel, QColor(255, 255, 255, 12))
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            p.setPen(QPen(QColor("#4f8cff"), 2))
            p.drawRect(sel)
            dpr = self._screen.devicePixelRatio() or 1.0
            label = f"{int(sel.width() * dpr)} × {int(sel.height() * dpr)}"
            p.setFont(QFont("Segoe UI", 9))
            p.setPen(QColor("white"))
            p.drawText(sel.left() + 4, max(14, sel.top() - 6), label)
        p.setFont(QFont("Segoe UI", 11))
        p.setPen(QColor("white"))
        p.drawText(self.rect().adjusted(0, 24, 0, 0), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                   "Выделите область мышью  ·  Esc или правая кнопка — отмена")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._owner.cancel()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self._origin_phys = self._physical(event)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._origin is not None:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._origin_phys is None:
            return
        x2, y2 = self._physical(event)
        x1, y1 = self._origin_phys
        self._owner.finish(Rect.from_points(x1, y1, x2, y2))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._owner.cancel()


class RegionSelector(QObject):
    """Показывает затемнение на всех мониторах и возвращает выбранный Rect."""

    selected = pyqtSignal(object)
    cancelled = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._windows: list[_SelectionWindow] = []
        self._done = True

    @property
    def active(self) -> bool:
        return bool(self._windows)

    def start(self) -> None:
        if self._windows:
            return
        self._done = False
        for screen in QGuiApplication.screens():
            w = _SelectionWindow(screen, self)
            w.show()
            self._windows.append(w)
        target = QGuiApplication.screenAt(QCursor.pos())
        for w in self._windows:
            if w._screen is target or target is None:
                w.raise_()
                w.activateWindow()
                w.setFocus()
                break

    def _close_all(self) -> None:
        windows, self._windows = self._windows, []
        for w in windows:
            w.close()

    def finish(self, rect: Rect) -> None:
        if self._done:
            return
        self._done = True
        self._close_all()
        if rect.is_valid(10):
            self.selected.emit(rect)
        else:
            self.cancelled.emit()

    def cancel(self) -> None:
        if self._done:
            return
        self._done = True
        self._close_all()
        self.cancelled.emit()

"""Глобальные горячие клавиши Windows через RegisterHotKey.

Программа только СЛУШАЕТ свои сочетания клавиш — никаких нажатий
клавиш или эмуляции ввода она не производит.
"""
from __future__ import annotations

import ctypes
import logging
import sys

from PyQt6.QtCore import QAbstractNativeEventFilter, QCoreApplication, QObject, QTimer, pyqtSignal

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000
WM_HOTKEY = 0x0312

_SPECIAL_KEYS = {f"F{i}": 0x6F + i for i in range(1, 25)}
_SPECIAL_KEYS.update({
    "SPACE": 0x20, "ENTER": 0x0D, "RETURN": 0x0D, "TAB": 0x09, "ESC": 0x1B, "ESCAPE": 0x1B,
    "INSERT": 0x2D, "INS": 0x2D, "DELETE": 0x2E, "DEL": 0x2E, "HOME": 0x24, "END": 0x23,
    "PAGEUP": 0x21, "PGUP": 0x21, "PAGEDOWN": 0x22, "PGDN": 0x22, "UP": 0x26, "DOWN": 0x28,
    "LEFT": 0x25, "RIGHT": 0x27, "PAUSE": 0x13, "PRINT": 0x2C, "PRINTSCREEN": 0x2C,
    "BACKSPACE": 0x08,
})
_MOD_NAMES = [(MOD_CONTROL, "Ctrl"), (MOD_ALT, "Alt"), (MOD_SHIFT, "Shift"), (MOD_WIN, "Win")]


class HotkeyParseError(ValueError):
    pass


def parse_hotkey(text: str) -> tuple[int, int, str]:
    """'Ctrl+Alt+S' -> (модификаторы, виртуальный код клавиши, каноническая строка)."""
    parts = [p.strip() for p in (text or "").split("+") if p.strip()]
    if not parts:
        raise HotkeyParseError("пустое сочетание")
    mods, key, key_name = 0, None, ""
    for part in parts:
        upper = part.upper()
        if upper in ("CTRL", "CONTROL", "CTL"):
            mods |= MOD_CONTROL
        elif upper == "ALT":
            mods |= MOD_ALT
        elif upper == "SHIFT":
            mods |= MOD_SHIFT
        elif upper in ("WIN", "META", "SUPER"):
            mods |= MOD_WIN
        else:
            if key is not None:
                raise HotkeyParseError("можно указать только одну основную клавишу")
            if len(upper) == 1 and upper.isascii() and upper.isalnum():
                key, key_name = ord(upper), upper
            elif upper in _SPECIAL_KEYS:
                key, key_name = _SPECIAL_KEYS[upper], part.capitalize() if len(part) > 3 else upper
            else:
                raise HotkeyParseError(f"неизвестная клавиша «{part}» (используйте латинские буквы, цифры, F1–F24)")
    if key is None:
        raise HotkeyParseError("не указана основная клавиша")
    if mods == 0:
        raise HotkeyParseError("нужен хотя бы один модификатор (Ctrl, Alt, Shift, Win)")
    canonical = "+".join(name for flag, name in _MOD_NAMES if mods & flag) + "+" + key_name
    return mods, key, canonical


class _WinHotkeyFilter(QAbstractNativeEventFilter):
    def __init__(self, callback) -> None:
        super().__init__()
        self._callback = callback
        self._last: tuple[int, int] | None = None

    def nativeEventFilter(self, event_type, message):
        try:
            if bytes(event_type) in (b"windows_generic_MSG", b"windows_dispatcher_MSG") and message:
                from ctypes import wintypes

                msg = wintypes.MSG.from_address(int(message))
                if msg.message == WM_HOTKEY:
                    stamp = (int(msg.wParam), int(msg.time))
                    if stamp != self._last:  # одно и то же сообщение может прийти дважды
                        self._last = stamp
                        self._callback(int(msg.wParam))
                    return True, 0
        except Exception:
            log.exception("Ошибка обработки горячей клавиши")
        return False, 0


class HotkeyManager(QObject):
    """Регистрирует сочетания и испускает triggered(<действие>)."""

    triggered = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.supported = IS_WINDOWS
        self._ids: dict[int, str] = {}
        self._next_id = 1
        self._filter = None
        if self.supported:
            from ctypes import wintypes

            self._user32 = ctypes.WinDLL("user32", use_last_error=True)
            self._user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
            self._user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
            self._filter = _WinHotkeyFilter(self._on_hotkey)
            app = QCoreApplication.instance()
            if app is not None:
                app.installNativeEventFilter(self._filter)

    def register_all(self, mapping: dict[str, str]) -> list[str]:
        """Возвращает список ошибок (пустой — всё зарегистрировано)."""
        self.unregister_all()
        if not self.supported:
            return []
        errors: list[str] = []
        for action, text in mapping.items():
            if not (text or "").strip():
                continue
            try:
                mods, key, canonical = parse_hotkey(text)
            except HotkeyParseError as exc:
                errors.append(f"{text}: {exc}")
                continue
            hotkey_id = self._next_id
            self._next_id = self._next_id + 1 if self._next_id < 0xBFFF else 1
            if self._user32.RegisterHotKey(None, hotkey_id, mods | MOD_NOREPEAT, key):
                self._ids[hotkey_id] = action
            else:
                errors.append(f"{canonical}: сочетание уже занято другой программой")
        return errors

    def unregister_all(self) -> None:
        if not self.supported:
            return
        for hotkey_id in list(self._ids):
            self._user32.UnregisterHotKey(None, hotkey_id)
        self._ids.clear()

    def _on_hotkey(self, hotkey_id: int) -> None:
        action = self._ids.get(hotkey_id)
        if action:
            # откладываем, чтобы не открывать диалоги внутри нативного фильтра
            QTimer.singleShot(0, lambda a=action: self.triggered.emit(a))

"""Настройки приложения: загрузка и сохранение в JSON рядом с программой."""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

APP_NAME = "Screen Answer"
APP_VERSION = "1.0.0"


def app_dir() -> Path:
    """Папка программы (рядом с main.py или с .exe, если собрано PyInstaller)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


APP_DIR = app_dir()
CONFIG_PATH = APP_DIR / "config.json"
DB_PATH = APP_DIR / "history.db"
LOG_PATH = APP_DIR / "screen_answer.log"
SCREENSHOT_DIR = APP_DIR / "screenshots"

ANALYSIS_INTERVALS = [5, 10, 15, 30, 60]

# Отображаемое имя языка -> как назвать его в инструкции для модели
ANSWER_LANGUAGES: dict[str, str] = {
    "Русский": "русском языке",
    "English": "английском языке (English)",
    "Українська": "украинском языке",
    "Беларуская": "белорусском языке",
    "Қазақша": "казахском языке",
    "O'zbekcha": "узбекском языке",
    "Deutsch": "немецком языке (Deutsch)",
    "Français": "французском языке (Français)",
    "Español": "испанском языке (Español)",
    "Italiano": "итальянском языке (Italiano)",
    "Português": "португальском языке (Português)",
    "Polski": "польском языке (Polski)",
    "Türkçe": "турецком языке (Türkçe)",
    "中文": "китайском языке (中文)",
    "日本語": "японском языке (日本語)",
    "한국어": "корейском языке (한국어)",
}

OCR_ENGINES: dict[str, str] = {
    "tesseract": "Tesseract OCR (локально, рекомендуется)",
    "easyocr": "EasyOCR (локально, нужен PyTorch)",
    "paddleocr": "PaddleOCR (локально, нужен PaddlePaddle)",
}

IMAGE_DETAILS = ["auto", "low", "high"]
MODEL_PRESETS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-5-mini", "gpt-5"]
THEMES = {"dark": "Тёмная", "light": "Светлая"}
CAPTURE_MODES = ("monitor", "window", "region")


@dataclass
class AppConfig:
    # --- ИИ ---
    api_key: str = ""
    api_base_url: str = ""
    model: str = "gpt-4o-mini"
    transcribe_model: str = "whisper-1"
    image_detail: str = "auto"
    max_image_side: int = 1600
    jpeg_quality: int = 75
    max_answer_tokens: int = 1500
    temperature: float = 0.2
    history_depth: int = 4
    request_timeout: int = 60
    answer_language: str = "Русский"

    # --- Конфиденциальность ---
    cloud_ai_enabled: bool = False
    cloud_warning_accepted: bool = False
    no_cloud_screenshots: bool = False
    save_screenshots: bool = False
    save_history: bool = True
    hide_overlay_during_capture: bool = True

    # --- Захват ---
    capture_mode: str = "monitor"
    monitor_index: int = 1
    region: list = field(default_factory=list)
    fresh_capture_on_ask: bool = True
    change_threshold: float = 0.1

    # --- Автоанализ ---
    auto_analysis: bool = False
    analysis_interval: int = 10
    auto_ai_summary: bool = False
    min_ai_interval: int = 30

    # --- OCR ---
    ocr_enabled: bool = True
    ocr_engine: str = "tesseract"
    ocr_languages: str = "rus+eng"
    tesseract_cmd: str = ""
    tessdata_prefix: str = ""
    tesseract_psm: int = 3
    ocr_upscale: bool = True

    # --- Внешний вид ---
    theme: str = "dark"
    opacity: int = 92
    font_size: int = 10
    window_geometry: list = field(default_factory=list)
    collapsed: bool = False
    show_options_panel: bool = True
    show_ocr_panel: bool = False
    show_history_panel: bool = False

    # --- Голос ---
    voice_auto_ask: bool = True

    # --- Горячие клавиши ---
    hotkey_capture_answer: str = "Ctrl+Alt+S"
    hotkey_ask: str = "Ctrl+Alt+Q"
    hotkey_toggle: str = "Ctrl+Alt+H"
    hotkey_stop: str = "Ctrl+Alt+X"

    # ------------------------------------------------------------------
    def effective_api_key(self) -> str:
        return (self.api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()

    def hotkeys(self) -> dict[str, str]:
        return {
            "capture_answer": self.hotkey_capture_answer,
            "ask": self.hotkey_ask,
            "toggle": self.hotkey_toggle,
            "stop": self.hotkey_stop,
        }

    def validate(self) -> None:
        def clamp(value: Any, lo: float, hi: float, cast=int):
            try:
                value = cast(value)
            except (TypeError, ValueError):
                value = cast(lo)
            return max(cast(lo), min(cast(hi), value))

        self.opacity = clamp(self.opacity, 10, 100)
        if self.analysis_interval not in ANALYSIS_INTERVALS:
            iv = clamp(self.analysis_interval, 5, 60)
            self.analysis_interval = min(ANALYSIS_INTERVALS, key=lambda x: abs(x - iv))
        self.history_depth = clamp(self.history_depth, 0, 10)
        self.max_image_side = clamp(self.max_image_side, 512, 4096)
        self.jpeg_quality = clamp(self.jpeg_quality, 30, 95)
        self.max_answer_tokens = clamp(self.max_answer_tokens, 100, 16000)
        self.temperature = clamp(self.temperature, 0.0, 2.0, float)
        self.request_timeout = clamp(self.request_timeout, 10, 300)
        self.change_threshold = clamp(self.change_threshold, 0.0, 50.0, float)
        self.min_ai_interval = clamp(self.min_ai_interval, 5, 3600)
        self.font_size = clamp(self.font_size, 8, 20)
        self.tesseract_psm = clamp(self.tesseract_psm, 0, 13)
        self.monitor_index = clamp(self.monitor_index, 0, 16)
        if self.theme not in THEMES:
            self.theme = "dark"
        if self.ocr_engine not in OCR_ENGINES:
            self.ocr_engine = "tesseract"
        if self.image_detail not in IMAGE_DETAILS:
            self.image_detail = "auto"
        if self.capture_mode not in CAPTURE_MODES:
            self.capture_mode = "monitor"
        if self.answer_language not in ANSWER_LANGUAGES:
            self.answer_language = "Русский"
        if not self.model.strip():
            self.model = "gpt-4o-mini"
        if not self.ocr_languages.strip():
            self.ocr_languages = "rus+eng"
        self.region = _int_list(self.region, need_positive_size=True)
        self.window_geometry = _int_list(self.window_geometry, need_positive_size=True)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "AppConfig":
        cfg = cls()
        if not path.exists():
            cfg.validate()
            return cfg
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("корень JSON не является объектом")
        except Exception as exc:  # повреждённый файл не должен ронять программу
            log.error("Не удалось прочитать %s: %s — используются настройки по умолчанию", path, exc)
            try:
                path.replace(path.with_name(path.name + ".broken"))
            except OSError:
                pass
            cfg.validate()
            return cfg

        for f in fields(cls):
            if f.name not in data:
                continue
            default = getattr(cfg, f.name)
            try:
                setattr(cfg, f.name, _coerce(data[f.name], default))
            except (TypeError, ValueError) as exc:
                log.warning("Параметр %s проигнорирован: %s", f.name, exc)
        cfg.validate()
        return cfg

    def save(self, path: Path = CONFIG_PATH) -> None:
        self.validate()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def copy(self) -> "AppConfig":
        return AppConfig(**copy.deepcopy(asdict(self)))

    def snapshot(self) -> dict:
        """Неизменяемая копия для передачи в рабочие потоки."""
        return copy.deepcopy(asdict(self))

    def update_from(self, other: "AppConfig") -> None:
        for f in fields(self):
            setattr(self, f.name, copy.deepcopy(getattr(other, f.name)))
        self.validate()


def _coerce(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "да", "on")
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return "" if value is None else str(value)
    if isinstance(default, list):
        if not isinstance(value, list):
            raise TypeError("ожидался список")
        return list(value)
    return value


def _int_list(value: Any, need_positive_size: bool = False) -> list:
    try:
        items = [int(v) for v in value]
    except (TypeError, ValueError):
        return []
    if len(items) != 4:
        return []
    if need_positive_size and (items[2] <= 0 or items[3] <= 0):
        return []
    return items

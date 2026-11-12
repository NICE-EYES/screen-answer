"""Координатор: связывает окно, потоки захвата/OCR/ИИ, трей, горячие клавиши и историю.

Поток обработки одного задания:
    снимок (CaptureWorker) -> сравнение с предыдущим -> OCR (OcrWorker)
    -> локальное определение контекста -> [ИИ (AiWorker)] -> окно + SQLite
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from ai.context_detector import ScreenContext, detect_context
from ai.vision_analyzer import DEFAULT_QUESTION, AnswerResult
from ai.voice_input import VoiceError, VoiceRecorder
from capture.screen_capture import (
    IS_WINDOWS, CaptureError, Rect, RegionSelector, get_foreground_window, get_window_rect, get_window_title,
    is_capturable_window, is_own_window, list_monitors,
)
from gui.overlay_window import OverlayWindow
from gui.settings_window import SettingsDialog, confirm_cloud_warning
from gui.theme import make_app_icon
from ocr.text_reader import OcrResult
from storage.history_db import HistoryDB, HistoryEntry
from utils.config import APP_NAME, DB_PATH, SCREENSHOT_DIR, AppConfig
from utils.hotkeys import HotkeyManager
from utils.workers import AiWorker, CaptureWorker, OcrWorker

log = logging.getLogger(__name__)

MAX_SESSION_DIALOG = 10
VOICE_MAX_SECONDS = 60


@dataclass
class Job:
    id: int
    purpose: str                  # auto | manual | ask
    question: str = ""
    use_cloud: bool = False
    rect: Rect | None = None
    hidden_overlay: bool = False
    image: Any = None
    ocr: OcrResult | None = None
    context: ScreenContext | None = None
    started: float = field(default_factory=time.monotonic)


class AppController(QObject):
    sig_capture = pyqtSignal(int, object, float)
    sig_reset_detector = pyqtSignal()
    sig_ocr = pyqtSignal(int, object, object)
    sig_ai = pyqtSignal(int, object)
    sig_transcribe = pyqtSignal(object, object)
    error_reported = pyqtSignal(str)

    def __init__(self, app: QApplication, cfg: AppConfig) -> None:
        super().__init__()
        self.app = app
        self.cfg = cfg
        self._jobs: dict[int, Job] = {}
        self._next_job_id = 1
        self._last_image = None
        self._last_ocr: OcrResult | None = None
        self._last_context: ScreenContext | None = None
        self._session_dialog: list[tuple[str, str]] = []
        self._last_ai_ts = 0.0
        self._last_external_hwnd: int | None = None
        self._monitors: list[Rect] = []
        self._local_only_chosen = False
        self._overlay_was_visible = True
        self._tray_hint_shown = False
        self._shutting_down = False
        self.error_reported.connect(self.report_error)

        self.db: HistoryDB | None = None
        db_error = ""
        try:
            self.db = HistoryDB(DB_PATH)
        except Exception as exc:
            log.exception("История недоступна")
            db_error = f"История недоступна: {exc}"

        self._setup_workers()

        self.overlay = OverlayWindow(cfg)
        self.overlay.monitor_provider = lambda: self.monitors(refresh=True)
        self._connect_overlay()
        self._setup_tray()

        self.region_selector = RegionSelector(self)
        self.region_selector.selected.connect(self._on_region_selected)
        self.region_selector.cancelled.connect(self._on_region_cancelled)

        self.voice = VoiceRecorder()
        self._voice_timer = QTimer(self)
        self._voice_timer.setSingleShot(True)
        self._voice_timer.timeout.connect(self.toggle_voice)

        self.hotkeys = HotkeyManager(self)
        self.hotkeys.triggered.connect(self._on_hotkey)

        self.auto_timer = QTimer(self)
        self.auto_timer.timeout.connect(self._auto_tick)
        self.fg_timer = QTimer(self)
        self.fg_timer.setInterval(400)
        self.fg_timer.timeout.connect(self._track_foreground)
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(800)
        self.save_timer.timeout.connect(self._save_config)

        self.app.screenAdded.connect(lambda _s: self.monitors(refresh=True))
        self.app.screenRemoved.connect(lambda _s: self.monitors(refresh=True))

        if db_error:
            self.overlay.set_status("error", db_error)

    # ================================================================ запуск
    def start(self) -> None:
        self.overlay.show()
        self._load_history()
        errors = self.hotkeys.register_all(self.cfg.hotkeys())
        if errors:
            self.overlay.set_hint("⚠ Горячие клавиши: " + "; ".join(errors))
        elif not self.hotkeys.supported:
            self.overlay.set_hint("Глобальные горячие клавиши работают только в Windows")
        if IS_WINDOWS:
            self.fg_timer.start()
        self.monitors(refresh=True)
        self._update_target_label()
        if self.cfg.auto_analysis:
            self.set_auto(True)
        else:
            self.overlay.set_status("idle")

    def _setup_workers(self) -> None:
        self._threads: list[QThread] = []

        def make_thread(name: str, worker: QObject) -> QThread:
            thread = QThread()
            thread.setObjectName(name)
            worker.moveToThread(thread)
            thread.finished.connect(worker.deleteLater)
            self._threads.append(thread)
            return thread

        self.capture_worker = CaptureWorker()
        self.capture_thread = make_thread("CaptureThread", self.capture_worker)
        self.sig_capture.connect(self.capture_worker.capture)
        self.sig_reset_detector.connect(self.capture_worker.reset_detector)
        self.capture_worker.captured.connect(self._on_captured)
        self.capture_worker.failed.connect(self._on_capture_failed)

        self.ocr_worker = OcrWorker()
        self.ocr_thread = make_thread("OcrThread", self.ocr_worker)
        self.sig_ocr.connect(self.ocr_worker.recognize)
        self.ocr_worker.done.connect(self._on_ocr_done)

        self.ai_worker = AiWorker()
        self.ai_thread = make_thread("AiThread", self.ai_worker)
        self.sig_ai.connect(self.ai_worker.run_job)
        self.sig_transcribe.connect(self.ai_worker.transcribe)
        self.ai_worker.chunk.connect(self._on_ai_chunk)
        self.ai_worker.finished.connect(self._on_ai_finished)
        self.ai_worker.failed.connect(self._on_ai_failed)
        self.ai_worker.transcribed.connect(self._on_transcribed)
        self.ai_worker.transcribe_failed.connect(self._on_transcribe_failed)

        for thread in self._threads:
            thread.start()

    def _connect_overlay(self) -> None:
        o = self.overlay
        o.ask_requested.connect(self.ask)
        o.capture_now_requested.connect(self.capture_now)
        o.capture_mode_selected.connect(self.set_capture_mode)
        o.select_region_requested.connect(self.start_region_selection)
        o.auto_toggled.connect(self.set_auto)
        o.interval_changed.connect(self.set_interval)
        o.opacity_changed.connect(lambda _v: self._schedule_save())
        o.language_changed.connect(self.set_language)
        o.theme_toggle_requested.connect(self.toggle_theme)
        o.clear_history_requested.connect(self.clear_session)
        o.erase_history_requested.connect(self.erase_history)
        o.settings_requested.connect(self.open_settings)
        o.hide_requested.connect(self.hide_overlay)
        o.stop_requested.connect(self.stop_all)
        o.voice_requested.connect(self.toggle_voice)
        o.history_item_selected.connect(self._show_history_entry)
        o.ui_state_changed.connect(self._schedule_save)

    def _setup_tray(self) -> None:
        self.tray = QSystemTrayIcon(make_app_icon(), self)
        self.tray.setToolTip(APP_NAME)
        menu = QMenu()
        menu.addAction("Показать / скрыть окно").triggered.connect(self.toggle_overlay)
        menu.addAction("Снимок и ответ").triggered.connect(lambda: self.ask(""))
        self.tray_auto = menu.addAction("Автоанализ")
        self.tray_auto.setCheckable(True)
        self.tray_auto.setChecked(self.cfg.auto_analysis)
        self.tray_auto.triggered.connect(self.set_auto)
        menu.addAction("Остановить анализ").triggered.connect(self.stop_all)
        menu.addSeparator()
        menu.addAction("Настройки…").triggered.connect(self.open_settings)
        menu.addSeparator()
        menu.addAction("Выход").triggered.connect(self.quit)
        self._tray_menu = menu
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()

    # ============================================================== мониторы
    def monitors(self, refresh: bool = False) -> list[Rect]:
        if refresh or not self._monitors:
            try:
                self._monitors = list_monitors()
            except CaptureError as exc:
                self.overlay.set_status("error", str(exc))
        return self._monitors

    def _resolve_rect(self) -> tuple[Rect | None, str]:
        mode = self.cfg.capture_mode
        if mode == "region":
            if len(self.cfg.region) == 4:
                rect = Rect(*self.cfg.region)
                if rect.is_valid():
                    return rect, ""
            return None, "Область не выбрана. Нажмите «Область» → «Выделить область мышью»."
        if mode == "window":
            if not IS_WINDOWS:
                return None, "Режим «Активное окно» доступен только в Windows."
            hwnd = self._last_external_hwnd
            if not is_capturable_window(hwnd):
                return None, "Активное окно не найдено (закрыто или свёрнуто). Щёлкните по нужному окну и повторите."
            rect = get_window_rect(hwnd)
            if rect is None:
                return None, "Не удалось определить границы активного окна."
            return rect, ""
        monitors = self.monitors()
        if not monitors:
            monitors = self.monitors(refresh=True)
        if not monitors:
            return None, "Не удалось получить список мониторов."
        idx = self.cfg.monitor_index
        if not 0 <= idx < len(monitors):
            idx = 1 if len(monitors) > 1 else 0
        return monitors[idx], ""

    def _update_target_label(self) -> None:
        mode = self.cfg.capture_mode
        if mode == "region" and len(self.cfg.region) == 4:
            text = f"выделенная область {self.cfg.region[2]}×{self.cfg.region[3]}"
        elif mode == "window":
            title = get_window_title(self._last_external_hwnd)
            text = f"активное окно — {title[:45]}" if title else "активное окно (щёлкните по нужному окну)"
        else:
            monitors = self._monitors
            idx = self.cfg.monitor_index
            if idx == 0:
                text = f"все мониторы ({monitors[0]})" if monitors else "все мониторы"
            else:
                size = f" ({monitors[idx]})" if 0 < idx < len(monitors) else ""
                text = f"монитор {idx}{size}"
        self.overlay.set_target_label(text)

    def _track_foreground(self) -> None:
        """Запоминает последнее «чужое» активное окно. Только чтение — окна не переключаются."""
        try:
            hwnd = get_foreground_window()
            if hwnd and hwnd != self._last_external_hwnd and not is_own_window(hwnd):
                self._last_external_hwnd = hwnd
                if self.cfg.capture_mode == "window":
                    self._update_target_label()
                    self.sig_reset_detector.emit()
        except Exception as exc:
            log.debug("foreground tracking: %s", exc)

    # =============================================================== задания
    def _new_job(self, purpose: str, question: str = "") -> Job:
        job = Job(id=self._next_job_id, purpose=purpose, question=question)
        self._next_job_id += 1
        self._jobs[job.id] = job
        self._update_busy()
        return job

    def _finish_job(self, job: Job) -> None:
        if job.hidden_overlay:
            job.hidden_overlay = False
            self.overlay.apply_opacity()
        self._jobs.pop(job.id, None)
        self._update_busy()

    def _fail_job(self, job: Job, message: str) -> None:
        self.overlay.set_status("error", message)
        if job.purpose == "ask":
            self.overlay.set_answer_error(message)
        else:
            self.overlay.set_hint(f"⚠ {message}")
        self._finish_job(job)

    def _update_busy(self) -> None:
        self.overlay.set_busy(bool(self._jobs) and not self.overlay.collapsed)

    def _start_capture(self, job: Job) -> None:
        rect, error = self._resolve_rect()
        if rect is None:
            self._fail_job(job, error)
            return
        job.rect = rect
        self.overlay.set_status("capture")
        hide = (self.cfg.hide_overlay_during_capture and self.overlay.isVisible()
                and self.overlay.windowOpacity() > 0.0 and self.overlay.physical_rect().intersects(rect))
        if hide:
            job.hidden_overlay = True
            self.overlay.setWindowOpacity(0.0)
            QTimer.singleShot(120, lambda jid=job.id, r=rect: self._emit_capture(jid, r))
        else:
            self._emit_capture(job.id, rect)

    def _emit_capture(self, job_id: int, rect: Rect) -> None:
        if job_id not in self._jobs:
            self.overlay.apply_opacity()
            return
        self.sig_capture.emit(job_id, rect, float(self.cfg.change_threshold))

    def _on_captured(self, job_id: int, image, changed: bool, ratio: float) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            self.overlay.apply_opacity()
            return
        if job.hidden_overlay:
            job.hidden_overlay = False
            self.overlay.apply_opacity()
        if job.purpose == "auto" and not changed:
            self.overlay.set_status("nochange", f"{ratio:.2f}% пикс.")
            self._finish_job(job)
            return
        job.image = image
        self._last_image = image
        self._maybe_save_screenshot(image)
        if self.cfg.ocr_enabled:
            self.overlay.set_status("ocr")
            self.sig_ocr.emit(job_id, image, self.cfg.snapshot())
        else:
            self._on_ocr_done(job_id, OcrResult(engine="off"))

    def _on_capture_failed(self, job_id: int, message: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            self.overlay.apply_opacity()
            return
        self._fail_job(job, message)

    def _on_ocr_done(self, job_id: int, result: OcrResult) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.ocr = result
        ctx = detect_context(result.text, result.confidence, self.cfg.answer_language,
                             ocr_enabled=self.cfg.ocr_enabled, ocr_error=result.error)
        job.context = ctx
        self._last_ocr, self._last_context = result, ctx

        info_parts = []
        if result.engine and result.engine != "off":
            info_parts.append(result.engine)
        if result.confidence is not None:
            info_parts.append(f"{result.confidence:.0f}%")
        if result.elapsed:
            info_parts.append(f"{result.elapsed:.1f} с")
        self.overlay.set_ocr_text(result.text if result.ok else f"[{result.error}]", " · ".join(info_parts))
        self.overlay.set_hint(ctx.hint)
        if job.purpose in ("auto", "manual") and self.db is not None and self.cfg.save_history:
            try:
                self.db.add_summary(ctx.kind, ctx.hint, result.text[:300])
            except Exception as exc:
                log.warning("summary save: %s", exc)
        self._after_ocr(job)

    def _after_ocr(self, job: Job) -> None:
        ctx = job.context
        if job.purpose == "ask":
            if not job.use_cloud:
                self._answer_locally(job)
                return
            no_text = not (job.ocr and job.ocr.text.strip())
            if self.cfg.no_cloud_screenshots and no_text:
                self._fail_job(job, "Текст на экране не распознан, а отправка скриншотов отключена — модели нечего "
                                    "анализировать. Выделите область крупнее или разрешите отправку изображения.")
                return
            self._dispatch_ai(job, "ask")
            return
        if job.purpose == "auto" and self._should_auto_summarize():
            self._dispatch_ai(job, "summary")
            return
        detail = ctx.title if ctx else ""
        self.overlay.set_status("ready", detail)
        self._finish_job(job)

    def _should_auto_summarize(self) -> bool:
        c = self.cfg
        if not (c.auto_ai_summary and c.cloud_ai_enabled and c.cloud_warning_accepted and c.effective_api_key()):
            return False
        if any(j.purpose == "ask" for j in self._jobs.values()):
            return False
        return time.monotonic() - self._last_ai_ts >= c.min_ai_interval

    def _dispatch_ai(self, job: Job, kind: str) -> None:
        send_image = not self.cfg.no_cloud_screenshots
        depth = self.cfg.history_depth
        payload = {
            "kind": kind,
            "config": self.cfg.snapshot(),
            "question": job.question or DEFAULT_QUESTION,
            "image": job.image if send_image else None,
            "ocr_text": job.ocr.text if job.ocr else "",
            "ocr_confidence": job.ocr.confidence if job.ocr else None,
            "context": job.context.describe() if job.context else "",
            "history": list(self._session_dialog[-depth:]) if depth > 0 else [],
            "language": self.cfg.answer_language,
        }
        self._last_ai_ts = time.monotonic()
        self.overlay.set_status("ai")
        self.sig_ai.emit(job.id, payload)

    def _answer_locally(self, job: Job) -> None:
        ctx = job.context
        ocr = job.ocr
        lines = ["**Облачный ИИ выключен** — показан локальный анализ (OCR).", ""]
        if ctx:
            lines.append(f"**Что на экране:** {ctx.hint}")
            if ctx.question:
                lines.append(f"**Вопрос на экране:** {ctx.question}")
            if ctx.options:
                lines.append("**Варианты:**")
                lines.extend(f"- {o}" for o in ctx.options)
        if ocr and ocr.text.strip():
            lines += ["", "**Распознанный текст:**", "```", ocr.text.strip()[:3000], "```"]
        lines += ["", "_Чтобы получить ответ модели, включите облачный ИИ в Настройках → Конфиденциальность._"]
        answer = "\n".join(lines)
        self.overlay.set_answer(job.question, answer, "локальный режим")
        self.overlay.set_status("ready", "локально")
        self._finish_job(job)

    # ------------------------------------------------------------- ответы ИИ
    def _on_ai_chunk(self, job_id: int, text: str) -> None:
        job = self._jobs.get(job_id)
        if job is not None and job.purpose == "ask":
            self.overlay.set_answer_stream(text)

    def _on_ai_finished(self, job_id: int, result: AnswerResult) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        if result.kind == "summary":
            self.overlay.set_hint(f"🤖 {result.text}")
            self.overlay.set_status("ready", "сводка")
            self._finish_job(job)
            return
        meta = f"{result.model} · {result.elapsed:.1f} с" + (" · со скриншотом" if result.image_sent else " · только текст")
        self.overlay.set_answer(job.question, result.text, meta)
        self._session_dialog.append((job.question, result.text))
        del self._session_dialog[:-MAX_SESSION_DIALOG]
        entry = HistoryEntry(0, datetime.now().isoformat(timespec="seconds"), job.question, result.text,
                             job.context.kind if job.context else "", result.model, result.image_sent)
        if self.db is not None and self.cfg.save_history:
            try:
                entry = self.db.add_entry(job.question, result.text, entry.context_kind,
                                          job.ocr.text if job.ocr else "", result.model, result.image_sent)
            except Exception as exc:
                log.warning("history save: %s", exc)
                self.overlay.set_hint(f"⚠ Не удалось сохранить историю: {exc}")
        self.overlay.add_history_item(entry)
        self.overlay.set_status("ready")
        self._finish_job(job)

    def _on_ai_failed(self, job_id: int, message: str, cancelled: bool) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        if cancelled:
            self.overlay.set_status("stopped", message)
            self._finish_job(job)
            return
        self._fail_job(job, message)

    # ============================================================ действия
    def _ensure_cloud_ready(self) -> bool | None:
        """True — можно в облако, False — только локально, None — отмена."""
        c = self.cfg
        if not c.cloud_ai_enabled:
            if self._local_only_chosen:
                return False
            box = QMessageBox(self.overlay)
            box.setWindowTitle(APP_NAME)
            box.setIcon(QMessageBox.Icon.Question)
            box.setText("Облачный ИИ выключен — сейчас доступен только локальный анализ (распознанный текст).\n\n"
                        "Включить облачный ИИ, чтобы модель анализировала экран и отвечала на вопросы?")
            enable = box.addButton("Включить", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Только локально", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is not enable:
                self._local_only_chosen = True
                return False
            if not c.cloud_warning_accepted and not confirm_cloud_warning(self.overlay):
                self._local_only_chosen = True
                return False
            c.cloud_warning_accepted = True
            c.cloud_ai_enabled = True
            self._schedule_save()
        elif not c.cloud_warning_accepted:
            if not confirm_cloud_warning(self.overlay):
                return False
            c.cloud_warning_accepted = True
            self._schedule_save()
        if not c.effective_api_key():
            QMessageBox.information(self.overlay, APP_NAME,
                                    "API-ключ не указан. Вставьте ключ в Настройках → ИИ (см. README).")
            self.open_settings("ai")
            return None if not self.cfg.effective_api_key() else True
        return True

    def ask(self, question: str = "") -> None:
        if self._shutting_down:
            return
        question = (question or "").strip() or DEFAULT_QUESTION
        use_cloud = self._ensure_cloud_ready()
        if use_cloud is None:
            return
        # новый вопрос отменяет предыдущий
        for old in [j for j in self._jobs.values() if j.purpose == "ask"]:
            self._finish_job(old)
        self.ai_worker.cancel_current()

        job = self._new_job("ask", question)
        job.use_cloud = bool(use_cloud)
        self.overlay.show_question(question)
        if self.cfg.fresh_capture_on_ask or self._last_image is None:
            self._start_capture(job)
        else:
            job.image = self._last_image
            job.ocr = self._last_ocr or OcrResult()
            job.context = self._last_context or detect_context(job.ocr.text, job.ocr.confidence,
                                                               self.cfg.answer_language)
            self._after_ocr(job)

    def capture_now(self) -> None:
        if any(j.purpose in ("manual", "auto") and j.image is None for j in self._jobs.values()):
            return
        self._start_capture(self._new_job("manual"))

    def set_auto(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.cfg.auto_analysis = enabled
        self.overlay.set_auto_checked(enabled)
        self.tray_auto.setChecked(enabled)
        if enabled:
            self.sig_reset_detector.emit()
            self.auto_timer.start(self.cfg.analysis_interval * 1000)
            self.overlay.set_status("idle", f"авто: каждые {self.cfg.analysis_interval} с")
            QTimer.singleShot(300, self._auto_tick)
        else:
            self.auto_timer.stop()
            self.overlay.set_status("idle", "автоанализ выключен")
        self._schedule_save()

    def set_interval(self, seconds: int) -> None:
        self.cfg.analysis_interval = int(seconds)
        if self.auto_timer.isActive():
            self.auto_timer.setInterval(self.cfg.analysis_interval * 1000)
        self._schedule_save()

    def _auto_tick(self) -> None:
        if not self.cfg.auto_analysis or self._shutting_down or self.region_selector.active:
            return
        if self._jobs:  # предыдущий анализ ещё идёт — пропускаем такт
            return
        self._start_capture(self._new_job("auto"))

    def set_language(self, language: str) -> None:
        self.cfg.answer_language = language
        self._schedule_save()

    def toggle_theme(self) -> None:
        self.cfg.theme = "light" if self.cfg.theme == "dark" else "dark"
        self.overlay.apply_config(self.cfg)
        self._schedule_save()

    def set_capture_mode(self, mode: str, monitor_index: int = 0) -> None:
        self.cfg.capture_mode = mode
        if mode == "monitor":
            self.cfg.monitor_index = int(monitor_index)
        self.sig_reset_detector.emit()
        self._update_target_label()
        self._schedule_save()

    def start_region_selection(self) -> None:
        if self.region_selector.active:
            return
        self._overlay_was_visible = self.overlay.isVisible()
        self.overlay.hide()
        QTimer.singleShot(150, self.region_selector.start)

    def _restore_after_selection(self) -> None:
        if self._overlay_was_visible:
            self.overlay.show()

    def _on_region_selected(self, rect: Rect) -> None:
        self._restore_after_selection()
        self.cfg.capture_mode = "region"
        self.cfg.region = rect.to_list()
        self.sig_reset_detector.emit()
        self._update_target_label()
        self._schedule_save()
        self.overlay.set_status("ready", f"область {rect}")
        self.capture_now()

    def _on_region_cancelled(self) -> None:
        self._restore_after_selection()
        self.overlay.set_status("idle", "выбор области отменён")

    def stop_all(self) -> None:
        self.set_auto(False)
        self.ai_worker.cancel_current()
        self.ai_worker.min_job_id = self._next_job_id
        for job in list(self._jobs.values()):
            self._finish_job(job)
        self._jobs.clear()
        self.overlay.apply_opacity()
        if self.voice.recording:
            self.voice.cancel()
            self._voice_timer.stop()
            self.overlay.set_voice_recording(False)
        self._update_busy()
        self.overlay.set_status("stopped", "анализ остановлен")

    # ---------------------------------------------------------------- голос
    def toggle_voice(self) -> None:
        if self.voice.recording:
            self._voice_timer.stop()
            self.overlay.set_voice_recording(False)
            try:
                wav = self.voice.stop()
            except VoiceError as exc:
                self.overlay.set_status("error", str(exc))
                return
            self.overlay.set_status("ai", "распознаю речь…")
            self.sig_transcribe.emit(wav, self.cfg.snapshot())
            return
        if self._ensure_cloud_ready() is not True:
            self.overlay.set_voice_recording(False)
            self.overlay.set_status("error", "голосовой ввод требует облачного ИИ и API-ключа")
            return
        try:
            self.voice.start()
        except VoiceError as exc:
            self.overlay.set_voice_recording(False)
            self.overlay.set_status("error", str(exc))
            return
        self.overlay.set_voice_recording(True)
        self.overlay.set_status("recording", "говорите… нажмите 🎤 ещё раз")
        self._voice_timer.start(VOICE_MAX_SECONDS * 1000)

    def _on_transcribed(self, text: str) -> None:
        self.overlay.set_question_text(text)
        if self.cfg.voice_auto_ask:
            self.overlay.take_question()
            self.ask(text)
        else:
            self.overlay.set_status("ready", "речь распознана")
            self.overlay.focus_question()

    def _on_transcribe_failed(self, message: str) -> None:
        self.overlay.set_status("error", message)

    # ------------------------------------------------------------- история
    def _load_history(self) -> None:
        if self.db is None:
            return
        try:
            self.overlay.set_history(self.db.recent(30))
        except Exception as exc:
            self.overlay.set_status("error", f"История: {exc}")

    def _show_history_entry(self, entry: HistoryEntry) -> None:
        if entry is not None:
            self.overlay.set_answer(entry.question, entry.answer, f"из истории · {entry.time_label}")

    def clear_session(self) -> None:
        self._session_dialog.clear()
        self.overlay.history_list.clear()
        self.overlay.clear_answer()
        self.overlay.set_status("ready", "история диалога очищена")

    def erase_history(self) -> None:
        answer = QMessageBox.question(
            self.overlay, APP_NAME,
            "Безвозвратно стереть всю историю вопросов и ответов из локальной базы"
            + (" и сохранённые скриншоты" if SCREENSHOT_DIR.exists() else "") + "?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            if self.db is not None:
                self.db.clear()
            removed = 0
            if SCREENSHOT_DIR.exists():
                for path in SCREENSHOT_DIR.glob("*.png"):
                    try:
                        path.unlink()
                        removed += 1
                    except OSError:
                        pass
        except Exception as exc:
            self.overlay.set_status("error", f"Не удалось стереть историю: {exc}")
            return
        self._session_dialog.clear()
        self.overlay.history_list.clear()
        self.overlay.clear_answer()
        self.overlay.set_status("ready", "история стёрта" + (f", скриншотов: {removed}" if removed else ""))

    def _maybe_save_screenshot(self, image) -> None:
        if not self.cfg.save_screenshots:
            return
        try:
            SCREENSHOT_DIR.mkdir(exist_ok=True)
            image.save(SCREENSHOT_DIR / f"{datetime.now():%Y%m%d_%H%M%S_%f}.png", optimize=False)
        except Exception as exc:
            self.overlay.set_hint(f"⚠ Не удалось сохранить скриншот: {exc}")

    # ------------------------------------------------------------ настройки
    def open_settings(self, tab: str | None = None) -> None:
        if not isinstance(tab, str):
            tab = None
        self.overlay.store_geometry()
        dialog = SettingsDialog(self.cfg, self.overlay, initial_tab=tab, erase_callback=self.erase_history)
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            self.apply_config(dialog.result_config())

    def apply_config(self, new: AppConfig) -> None:
        old_hotkeys = self.cfg.hotkeys()
        old_capture = (self.cfg.change_threshold, self.cfg.capture_mode)
        # геометрию и состояние панелей оставляем текущими
        new.window_geometry = list(self.cfg.window_geometry)
        new.collapsed = self.cfg.collapsed
        new.auto_analysis = self.cfg.auto_analysis
        self.cfg.update_from(new)
        if self.cfg.cloud_ai_enabled:
            self._local_only_chosen = False
        self.overlay.apply_config(self.cfg)
        if self.cfg.hotkeys() != old_hotkeys:
            errors = self.hotkeys.register_all(self.cfg.hotkeys())
            if errors:
                self.overlay.set_hint("⚠ Горячие клавиши: " + "; ".join(errors))
        if (self.cfg.change_threshold, self.cfg.capture_mode) != old_capture:
            self.sig_reset_detector.emit()
        if self.auto_timer.isActive():
            self.auto_timer.setInterval(self.cfg.analysis_interval * 1000)
        self._update_target_label()
        self._save_config()
        self.overlay.set_status("ready", "настройки сохранены")

    def _schedule_save(self) -> None:
        if not self._shutting_down:
            self.save_timer.start()

    def _save_config(self) -> None:
        try:
            self.cfg.save()
        except Exception as exc:
            log.error("config save: %s", exc)
            self.overlay.set_status("error", f"Не удалось сохранить настройки: {exc}")

    # ------------------------------------------------------- окно и трей
    def _on_hotkey(self, action: str) -> None:
        if action == "capture_answer":
            self.ask(self.overlay.take_question())
        elif action == "ask":
            self.overlay.focus_question()
        elif action == "toggle":
            self.toggle_overlay()
        elif action == "stop":
            self.stop_all()

    def _on_tray_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.toggle_overlay()

    def toggle_overlay(self) -> None:
        if self.overlay.isVisible():
            self.hide_overlay()
        else:
            self.overlay.show()
            self.overlay.raise_()

    def hide_overlay(self) -> None:
        if not self.tray.isVisible():
            # без трея окно потеряется — сворачиваем в полоску
            self.overlay.set_collapsed(True)
            return
        self.overlay.hide()
        if not self._tray_hint_shown:
            self._tray_hint_shown = True
            self.tray.showMessage(APP_NAME, "Окно скрыто в трей. Ctrl+Alt+H — показать снова.",
                                  QSystemTrayIcon.MessageIcon.Information, 3000)

    def report_error(self, message: str) -> None:
        try:
            self.overlay.set_status("error", message)
        except Exception:
            pass

    def quit(self) -> None:
        # В Qt 6 quit() сначала закрывает окна; оверлей должен разрешить закрытие.
        self.overlay.allow_close = True
        self.app.quit()

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self.auto_timer.stop()
        self.fg_timer.stop()
        self.save_timer.stop()
        self.ai_worker.cancel_current()
        self.ai_worker.min_job_id = 1 << 62
        self._jobs.clear()
        try:
            self.hotkeys.unregister_all()
        except Exception:
            pass
        if self.voice.recording:
            self.voice.cancel()
        try:
            self.overlay.store_geometry()
            self.cfg.save()
        except Exception as exc:
            log.error("config save on exit: %s", exc)
        for thread in self._threads:
            thread.quit()
        for thread in self._threads:
            if not thread.wait(3000):
                log.warning("Поток %s не завершился вовремя — принудительная остановка", thread.objectName())
                thread.terminate()
                thread.wait(1000)
        if self.db is not None:
            self.db.close()
        self.tray.hide()

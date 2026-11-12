"""Рабочие объекты для отдельных QThread: захват экрана, OCR и запросы к ИИ.

Каждый объект живёт в своём потоке, принимает задания через сигналы и
возвращает результат сигналами — интерфейс никогда не блокируется.
"""
from __future__ import annotations

import logging
import threading
import time

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from ai.vision_analyzer import AiCancelled, AiError, VisionAnalyzer
from capture.screen_capture import CaptureError, ChangeDetector, ScreenGrabber
from ocr.text_reader import OcrResult, TextReader

log = logging.getLogger(__name__)


class CaptureWorker(QObject):
    captured = pyqtSignal(int, object, bool, float)  # job_id, PIL.Image, изменился?, % изменений
    failed = pyqtSignal(int, str)

    def __init__(self) -> None:
        super().__init__()
        self._grabber: ScreenGrabber | None = None
        self._detector = ChangeDetector()

    @pyqtSlot(int, object, float)
    def capture(self, job_id: int, rect, threshold: float) -> None:
        try:
            if self._grabber is None:
                self._grabber = ScreenGrabber()  # mss создаётся в этом же потоке
            image = self._grabber.grab(rect)
            self._detector.threshold_percent = threshold
            changed, ratio = self._detector.compare(image)
            self.captured.emit(job_id, image, changed, ratio)
        except CaptureError as exc:
            self.failed.emit(job_id, str(exc))
        except Exception as exc:
            log.exception("Ошибка захвата")
            self.failed.emit(job_id, f"Ошибка захвата экрана: {exc}")

    @pyqtSlot()
    def reset_detector(self) -> None:
        self._detector.reset()


class OcrWorker(QObject):
    done = pyqtSignal(int, object)  # job_id, OcrResult

    def __init__(self) -> None:
        super().__init__()
        self._reader = TextReader()

    @pyqtSlot(int, object, object)
    def recognize(self, job_id: int, image, cfg) -> None:
        try:
            result = self._reader.read(image, cfg)
        except Exception as exc:  # TextReader сам ловит ошибки, это страховка
            log.exception("Ошибка OCR")
            result = OcrResult.failed(cfg.get("ocr_engine", ""), f"Ошибка OCR: {exc}")
        self.done.emit(job_id, result)


class AiWorker(QObject):
    chunk = pyqtSignal(int, str)            # job_id, накопленный текст
    finished = pyqtSignal(int, object)      # job_id, AnswerResult
    failed = pyqtSignal(int, str, bool)     # job_id, сообщение, отменено?
    transcribed = pyqtSignal(str)
    transcribe_failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._analyzer = VisionAnalyzer()
        self._cancel = threading.Event()
        self.min_job_id = 0  # задания с меньшим id отбрасываются (после «Стоп»)

    def cancel_current(self) -> None:
        """Потокобезопасно: вызывается напрямую из главного потока."""
        self._cancel.set()

    @pyqtSlot(int, object)
    def run_job(self, job_id: int, payload) -> None:
        if job_id < self.min_job_id:
            self.failed.emit(job_id, "Запрос отменён", True)
            return
        self._cancel.clear()
        last_emit = [0.0]

        def on_chunk(text: str) -> None:
            now = time.monotonic()
            if now - last_emit[0] >= 0.08:
                last_emit[0] = now
                self.chunk.emit(job_id, text)

        try:
            if payload.get("kind") == "summary":
                result = self._analyzer.summarize(payload, self._cancel)
            else:
                result = self._analyzer.ask(payload, self._cancel, on_chunk)
            self.finished.emit(job_id, result)
        except AiCancelled:
            self.failed.emit(job_id, "Запрос отменён", True)
        except AiError as exc:
            self.failed.emit(job_id, str(exc), False)
        except Exception as exc:
            log.exception("Ошибка запроса к ИИ")
            self.failed.emit(job_id, f"Непредвиденная ошибка ИИ: {exc}", False)

    @pyqtSlot(object, object)
    def transcribe(self, wav_bytes, cfg) -> None:
        try:
            text = self._analyzer.transcribe(cfg, wav_bytes)
            if text:
                self.transcribed.emit(text)
            else:
                self.transcribe_failed.emit("Речь не распознана — попробуйте ещё раз")
        except AiError as exc:
            self.transcribe_failed.emit(str(exc))
        except Exception as exc:
            log.exception("Ошибка распознавания речи")
            self.transcribe_failed.emit(f"Ошибка распознавания речи: {exc}")

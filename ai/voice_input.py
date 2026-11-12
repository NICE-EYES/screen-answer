"""Запись вопроса с микрофона (необязательная функция, нужен пакет sounddevice)."""
from __future__ import annotations

import io
import logging
import threading
import wave

log = logging.getLogger(__name__)


class VoiceError(Exception):
    """Понятная пользователю ошибка записи."""


class VoiceRecorder:
    SAMPLE_RATE = 16000

    def __init__(self) -> None:
        self._stream = None
        self._frames: list = []
        self._lock = threading.Lock()

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise VoiceError("Для голосового ввода установите пакет: pip install sounddevice") from exc
        except OSError as exc:
            raise VoiceError(f"Библиотека PortAudio недоступна: {exc}") from exc
        with self._lock:
            self._frames = []
        try:
            self._stream = sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1, dtype="int16",
                                          callback=self._callback)
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise VoiceError(f"Не удалось открыть микрофон: {exc}") from exc

    def _callback(self, indata, _frames, _time, status) -> None:
        if status:
            log.debug("audio status: %s", status)
        with self._lock:
            self._frames.append(indata.copy())

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                log.warning("Ошибка закрытия микрофона: %s", exc)

    def stop(self) -> bytes:
        """Останавливает запись и возвращает WAV (16 кГц, моно)."""
        self._close_stream()
        with self._lock:
            frames, self._frames = self._frames, []
        if not frames:
            raise VoiceError("Запись пуста — проверьте микрофон")
        import numpy as np

        audio = np.concatenate(frames, axis=0)
        if len(audio) < self.SAMPLE_RATE * 0.4:
            raise VoiceError("Слишком короткая запись — говорите дольше")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    def cancel(self) -> None:
        self._close_stream()
        with self._lock:
            self._frames = []

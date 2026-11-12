"""Отправка скриншота, OCR-текста и вопроса в OpenAI Vision API.

Скриншот передаётся как Base64 data URL (JPEG, уменьшенный для экономии токенов).
Модуль не выполняет никаких действий на компьютере — только HTTP-запросы к API.
"""
from __future__ import annotations

import base64
import io
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

from PIL import Image

from utils.config import ANSWER_LANGUAGES

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — помощник, который анализирует скриншот экрана пользователя. Тебе даны изображение экрана, \
распознанный текст и вопрос пользователя. Определи, что происходит на экране, найди ключевые данные и ответь кратко, \
точно и по делу. Не выдумывай информацию, которой нет на скриншоте. Если вопрос неоднозначный, объясни, что видно, \
и предложи наиболее вероятный ответ. Если на экране тест или вопрос с вариантами, сначала объясни логику, затем укажи \
наиболее подходящий вариант. Отвечай на русском языке, если пользователь не попросил другой язык. Не предлагай \
действия, которые требуют управления компьютером.

Правила по типам содержимого:
- Тест / вопрос с вариантами: коротко перескажи вопрос, разбери варианты, объясни, почему верный вариант верен \
(и при необходимости — почему остальные неверны). Только после объяснения напиши отдельной строкой: \
«Ответ: <буква или номер> — <текст варианта>». Если верных вариантов несколько — перечисли все. \
Никогда не отвечай одной буквой без объяснения.
- Математика: перепиши выражение в понятном виде, решай по шагам, проверь результат подстановкой или оценкой. \
Формулы пиши простым текстом (x^2, sqrt(x), a/b), без LaTeX.
- Таблицы: опиши структуру (что в строках и столбцах), извлеки нужные значения; при расчётах укажи, какие ячейки \
использованы.
- Графики и диаграммы: назови тип графика, оси и единицы, главные тенденции, максимумы и минимумы. Если точные \
значения не видны — пиши «примерно».
- Код: определи язык и назначение. При поиске ошибок укажи строку или фрагмент, причину и исправленный код \
в блоке ```.
- Сообщения об ошибках: объясни причину простыми словами и дай шаги, которые пользователь выполнит сам.
- Иностранный текст: определи язык; при просьбе перевести — переводи точно, сохраняя структуру.
- Если текст на изображении плохо читается, обрезан, или вопрос/варианты видны не полностью — прямо скажи об этом \
и попроси пользователя сделать скриншот нужной области ещё раз (кнопка «Область» → «Выделить мышью»). \
Не угадывай нечитаемое.
- OCR-текст может содержать ошибки распознавания; при расхождении доверяй изображению.
- Форматирование: короткие абзацы, маркированные списки, **жирным** — главный вывод. Без вступлений и повторов."""

SUMMARY_PROMPT = """Ты кратко описываешь, что сейчас на экране пользователя. Ответь одним-двумя короткими \
предложениями (до 200 символов): что открыто и есть ли там вопрос теста, задача, ошибка, таблица или график. \
Не решай задачу и не давай советов."""

DEFAULT_QUESTION = ("Проанализируй экран. Если на нём вопрос, тест или задача — реши и дай ответ с кратким "
                    "объяснением. Если ошибка — объясни причину и как исправить. Иначе кратко опиши главное.")

MAX_OCR_CHARS = 6000
MAX_HISTORY_ANSWER_CHARS = 1500


class AiError(Exception):
    """Понятная пользователю ошибка ИИ."""


class AiCancelled(Exception):
    """Запрос отменён пользователем."""


@dataclass
class AnswerResult:
    text: str
    model: str
    image_sent: bool
    kind: str = "ask"
    elapsed: float = 0.0


def image_to_data_url(image: Image.Image, max_side: int = 1600, quality: int = 75) -> str:
    """Уменьшает и сжимает скриншот в JPEG и возвращает data URL."""
    img = image.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _friendly_error(exc: Exception) -> str:
    try:
        import openai
    except ImportError:
        return str(exc)
    message = getattr(exc, "message", None) or str(exc)
    message = str(message)[:300]
    if isinstance(exc, openai.AuthenticationError):
        return "Неверный или отозванный API-ключ (401). Проверьте ключ в Настройках → ИИ."
    if isinstance(exc, openai.PermissionDeniedError):
        return "Доступ запрещён (403): у ключа нет доступа к этой модели или регион не поддерживается."
    if isinstance(exc, openai.NotFoundError):
        return "Модель или адрес API не найдены (404). Проверьте название модели и адрес API в настройках."
    if isinstance(exc, openai.RateLimitError):
        if "insufficient_quota" in message or "quota" in message.lower():
            return "На счёте API закончились средства (insufficient_quota). Пополните баланс на platform.openai.com."
        return "Слишком много запросов (429). Подождите немного и увеличьте интервал анализа."
    if isinstance(exc, openai.BadRequestError):
        hint = ""
        if "image" in message.lower():
            hint = " Похоже, модель не принимает изображения — выберите gpt-4o-mini или gpt-4o."
        return f"Запрос отклонён (400): {message}.{hint}"
    if isinstance(exc, openai.APITimeoutError):
        return "Сервер ИИ не ответил вовремя. Проверьте интернет или увеличьте таймаут в настройках."
    if isinstance(exc, openai.APIConnectionError):
        return "Нет соединения с сервером ИИ. Проверьте интернет, прокси или адрес API."
    if isinstance(exc, openai.APIStatusError):
        return f"Ошибка сервера ИИ ({getattr(exc, 'status_code', '?')}): {message}. Повторите позже."
    return f"Ошибка ИИ: {message}"


class VisionAnalyzer:
    """Клиент OpenAI-совместимого API с поддержкой изображений."""

    def __init__(self) -> None:
        self._client = None
        self._client_key: tuple | None = None

    # ------------------------------------------------------------------
    def _get_client(self, cfg: dict):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AiError("Пакет openai не установлен: pip install openai") from exc
        import os

        api_key = (cfg.get("api_key") or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise AiError("API-ключ не указан. Откройте Настройки → ИИ и вставьте ключ.")
        base_url = (cfg.get("api_base_url") or "").strip() or None
        timeout = float(cfg.get("request_timeout", 60))
        key = (api_key, base_url, timeout)
        if self._client is None or self._client_key != key:
            self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=2)
            self._client_key = key
        return self._client

    @staticmethod
    def _language_instruction(language: str) -> str:
        name = ANSWER_LANGUAGES.get(language, "русском языке")
        return (f"\n\nЯзык ответа, выбранный пользователем в настройках: отвечай на {name}, "
                "если в вопросе явно не попросили другой язык.")

    def build_messages(self, payload: dict, image_url: str | None) -> list[dict]:
        messages: list[dict] = [{"role": "system",
                                 "content": SYSTEM_PROMPT + self._language_instruction(payload.get("language", ""))}]
        for question, answer in payload.get("history") or []:
            messages.append({"role": "user", "content": str(question)})
            messages.append({"role": "assistant", "content": str(answer)[:MAX_HISTORY_ANSWER_CHARS]})

        ocr_text = (payload.get("ocr_text") or "").strip()
        if len(ocr_text) > MAX_OCR_CHARS:
            ocr_text = ocr_text[:MAX_OCR_CHARS] + "\n…[текст обрезан]"
        parts = ["[Локальный анализ экрана]", payload.get("context") or "нет данных", ""]
        if ocr_text:
            conf = payload.get("ocr_confidence")
            conf_txt = f", уверенность {conf:.0f}%" if isinstance(conf, (int, float)) else ""
            parts += [f"[Распознанный текст (OCR{conf_txt})]", "<<<", ocr_text, ">>>", ""]
        else:
            parts += ["[Распознанный текст] — отсутствует", ""]
        if image_url is None:
            parts.append("[Изображение экрана НЕ передано по настройкам конфиденциальности — "
                         "отвечай только по распознанному тексту и предупреди, если его недостаточно.]")
        parts += ["[Вопрос пользователя]", payload.get("question") or DEFAULT_QUESTION]

        content: list[dict] = [{"type": "text", "text": "\n".join(parts)}]
        if image_url:
            content.append({"type": "image_url",
                            "image_url": {"url": image_url, "detail": payload["config"].get("image_detail", "auto")}})
        messages.append({"role": "user", "content": content})
        return messages

    def _create(self, client, cfg: dict, messages: list[dict], max_tokens: int, stream: bool):
        """Вызов API с автоматической подстройкой под разные модели (temperature / max_tokens)."""
        import openai

        params: dict = {"model": cfg.get("model", "gpt-4o-mini"), "messages": messages,
                        "max_completion_tokens": int(max_tokens), "temperature": float(cfg.get("temperature", 0.2))}
        if stream:
            params["stream"] = True
        for _ in range(3):
            try:
                return client.chat.completions.create(**params)
            except openai.BadRequestError as exc:
                msg = str(exc).lower()
                if "temperature" in msg and "temperature" in params:
                    params.pop("temperature")
                    continue
                if "max_completion_tokens" in msg and "max_completion_tokens" in params:
                    params["max_tokens"] = params.pop("max_completion_tokens")
                    continue
                raise AiError(_friendly_error(exc)) from exc
            except AiError:
                raise
            except Exception as exc:
                raise AiError(_friendly_error(exc)) from exc
        raise AiError("Модель отклонила параметры запроса. Попробуйте другую модель.")

    # ------------------------------------------------------------------
    def ask(self, payload: dict, cancel: threading.Event,
            on_chunk: Callable[[str], None] | None = None) -> AnswerResult:
        cfg = payload["config"]
        client = self._get_client(cfg)
        started = time.monotonic()
        image = payload.get("image")
        image_url = None
        if image is not None:
            image_url = image_to_data_url(image, int(cfg.get("max_image_side", 1600)), int(cfg.get("jpeg_quality", 75)))
        messages = self.build_messages(payload, image_url)
        if cancel.is_set():
            raise AiCancelled()

        stream = self._create(client, cfg, messages, int(cfg.get("max_answer_tokens", 1500)), stream=True)
        parts: list[str] = []
        try:
            for event in stream:
                if cancel.is_set():
                    raise AiCancelled()
                if not getattr(event, "choices", None):
                    continue
                piece = getattr(event.choices[0].delta, "content", None)
                if piece:
                    parts.append(piece)
                    if on_chunk:
                        on_chunk("".join(parts))
        except AiCancelled:
            raise
        except Exception as exc:
            raise AiError(_friendly_error(exc)) from exc
        finally:
            try:
                stream.close()
            except Exception:
                pass

        text = "".join(parts).strip()
        if not text:
            raise AiError("Модель вернула пустой ответ. Повторите вопрос или увеличьте лимит токенов в настройках.")
        return AnswerResult(text=text, model=cfg.get("model", ""), image_sent=image_url is not None,
                            kind="ask", elapsed=time.monotonic() - started)

    def summarize(self, payload: dict, cancel: threading.Event) -> AnswerResult:
        cfg = payload["config"]
        client = self._get_client(cfg)
        started = time.monotonic()
        image = payload.get("image")
        image_url = None
        if image is not None:
            # для сводки достаточно маленького изображения
            image_url = image_to_data_url(image, min(1024, int(cfg.get("max_image_side", 1600))), 60)
        text_part = (f"{payload.get('context', '')}\n\nOCR:\n{(payload.get('ocr_text') or '')[:2000]}"
                     f"{self._language_instruction(payload.get('language', ''))}")
        content: list[dict] = [{"type": "text", "text": text_part}]
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url, "detail": "low"}})
        messages = [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": content}]
        if cancel.is_set():
            raise AiCancelled()
        response = self._create(client, cfg, messages, 200, stream=False)
        if cancel.is_set():
            raise AiCancelled()
        try:
            text = (response.choices[0].message.content or "").strip()
        except (AttributeError, IndexError) as exc:
            raise AiError("Не удалось разобрать ответ модели") from exc
        if not text:
            raise AiError("Пустая сводка от модели")
        return AnswerResult(text=text, model=cfg.get("model", ""), image_sent=image_url is not None,
                            kind="summary", elapsed=time.monotonic() - started)

    def transcribe(self, cfg: dict, wav_bytes: bytes) -> str:
        client = self._get_client(cfg)
        try:
            result = client.audio.transcriptions.create(
                model=cfg.get("transcribe_model") or "whisper-1",
                file=("speech.wav", wav_bytes, "audio/wav"),
            )
        except Exception as exc:
            raise AiError(_friendly_error(exc)) from exc
        return (getattr(result, "text", "") or "").strip()

    def test_connection(self, cfg: dict) -> str:
        client = self._get_client(cfg)
        model = cfg.get("model", "gpt-4o-mini")
        try:
            client.models.retrieve(model)
        except Exception as exc:
            raise AiError(_friendly_error(exc)) from exc
        return f"Подключение работает, модель «{model}» доступна."

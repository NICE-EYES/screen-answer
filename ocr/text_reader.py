"""Локальное распознавание текста: Tesseract (по умолчанию), EasyOCR, PaddleOCR.

Все движки работают на компьютере пользователя, ничего не отправляют в сеть.
"""
from __future__ import annotations

import logging
import os
import shutil
import statistics
import time
from dataclasses import dataclass

from PIL import Image, ImageOps, ImageStat

log = logging.getLogger(__name__)


class OcrError(Exception):
    """Понятная пользователю ошибка OCR."""


@dataclass
class OcrResult:
    text: str = ""
    confidence: float | None = None  # 0–100, если движок её возвращает
    engine: str = ""
    error: str = ""
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error

    @classmethod
    def failed(cls, engine: str, error: str) -> "OcrResult":
        return cls(engine=engine, error=error)


# Коды языков Tesseract -> EasyOCR / PaddleOCR
_TESS_TO_EASY = {
    "rus": "ru", "eng": "en", "ukr": "uk", "bel": "be", "deu": "de", "fra": "fr", "spa": "es",
    "ita": "it", "por": "pt", "pol": "pl", "tur": "tr", "kaz": "kk", "uzb": "uz",
    "chi_sim": "ch_sim", "chi_tra": "ch_tra", "jpn": "ja", "kor": "ko",
}
_TESS_TO_PADDLE = {
    "rus": "ru", "ukr": "uk", "bel": "be", "eng": "en", "deu": "german", "fra": "french",
    "spa": "es", "ita": "it", "por": "pt", "pol": "pl", "tur": "tr", "kaz": "kk",
    "chi_sim": "ch", "chi_tra": "chinese_cht", "jpn": "japan", "kor": "korean",
}

_WINDOWS_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
]


def find_tesseract(configured: str = "") -> str:
    if configured and os.path.isfile(configured):
        return configured
    found = shutil.which("tesseract")
    if found:
        return found
    for path in _WINDOWS_TESSERACT_PATHS:
        if os.path.isfile(path):
            return path
    return ""


def preprocess_for_tesseract(image: Image.Image, upscale: bool) -> Image.Image:
    gray = image.convert("L")
    if upscale:
        longest = max(gray.size)
        factor = 2.0 if longest < 1000 else (1.5 if longest < 1800 else 1.0)
        if factor > 1.0:
            gray = gray.resize((int(gray.width * factor), int(gray.height * factor)), Image.Resampling.LANCZOS)
    # Тёмная тема (IDE, консоль) -> инвертируем: Tesseract лучше читает тёмный текст на светлом.
    if ImageStat.Stat(gray).mean[0] < 110:
        gray = ImageOps.invert(gray)
    return ImageOps.autocontrast(gray, cutoff=1)


def _maybe_upscale(image: Image.Image, upscale: bool) -> Image.Image:
    img = image.convert("RGB")
    if upscale and max(img.size) < 1000:
        img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
    return img


def _join_words(words: list[tuple[float, float, str]]) -> str:
    """words: (left, right, text). Большие промежутки -> несколько пробелов (столбцы таблиц)."""
    words = sorted(words, key=lambda w: w[0])
    char_widths = [(r - l) / max(1, len(t)) for l, r, t in words if r > l]
    avg_char = statistics.median(char_widths) if char_widths else 8.0
    parts = [words[0][2]]
    for prev, cur in zip(words, words[1:]):
        gap = cur[0] - prev[1]
        parts.append("    " if gap > avg_char * 3.5 else " ")
        parts.append(cur[2])
    return "".join(parts)


def _group_into_lines(items: list[tuple[float, float, float, float, str]]) -> str:
    """items: (x0, y0, x1, y1, text) -> текст по строкам сверху вниз."""
    items = [it for it in items if it[4].strip()]
    if not items:
        return ""
    items.sort(key=lambda it: (it[1] + it[3]) / 2)
    lines: list[dict] = []
    for x0, y0, x1, y1, text in items:
        cy, h = (y0 + y1) / 2, max(1.0, y1 - y0)
        if lines and abs(cy - lines[-1]["cy"]) < max(h, lines[-1]["h"]) * 0.5:
            line = lines[-1]
            line["words"].append((x0, x1, text))
            line["cy"] = (line["cy"] + cy) / 2
        else:
            lines.append({"cy": cy, "h": h, "words": [(x0, x1, text)]})
    return "\n".join(_join_words(line["words"]) for line in lines)


class _TesseractEngine:
    def __init__(self, cmd: str, tessdata: str) -> None:
        import pytesseract

        self.pt = pytesseract
        path = find_tesseract(cmd)
        if not path:
            raise OcrError(
                "Tesseract не найден. Установите Tesseract OCR (см. README) "
                "или укажите путь к tesseract.exe в Настройках → OCR.")
        pytesseract.pytesseract.tesseract_cmd = path
        if tessdata:
            os.environ["TESSDATA_PREFIX"] = tessdata
        elif not os.environ.get("TESSDATA_PREFIX"):
            candidate = os.path.join(os.path.dirname(path), "tessdata")
            if os.path.isdir(candidate):
                os.environ["TESSDATA_PREFIX"] = candidate

    def read(self, image: Image.Image, langs: str, psm: int, upscale: bool) -> tuple[str, float | None]:
        img = preprocess_for_tesseract(image, upscale)
        pt = self.pt
        try:
            data = pt.image_to_data(img, lang=langs, config=f"--oem 3 --psm {psm}",
                                    output_type=pt.Output.DICT, timeout=40)
        except pt.TesseractNotFoundError as exc:
            raise OcrError("Tesseract не найден. Проверьте путь к tesseract.exe в настройках.") from exc
        except pt.TesseractError as exc:
            msg = str(exc)
            if "Failed loading language" in msg or "Error opening data file" in msg:
                raise OcrError(
                    f"Нет языковых данных Tesseract для «{langs}». Установите нужные языки "
                    "(rus.traineddata, eng.traineddata) и проверьте TESSDATA_PREFIX.") from exc
            raise OcrError(f"Ошибка Tesseract: {msg[:300]}") from exc
        except RuntimeError as exc:  # timeout
            raise OcrError("Tesseract работал слишком долго. Выберите область поменьше.") from exc

        lines: dict[tuple, list] = {}
        blocks: dict[tuple, tuple] = {}
        confs: list[float] = []
        for i, raw in enumerate(data.get("text", [])):
            word = (raw or "").strip()
            if not word:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf >= 0:
                confs.append(conf)
            key = (data["page_num"][i], data["block_num"][i], data["par_num"][i], data["line_num"][i])
            left, width = float(data["left"][i]), float(data["width"][i])
            lines.setdefault(key, []).append((left, left + width, word))
            blocks[key] = key[:3]

        out: list[str] = []
        prev_par = None
        for key, words in lines.items():  # порядок вставки = порядок чтения Tesseract
            par = blocks[key]
            if prev_par is not None and par != prev_par:
                out.append("")
            out.append(_join_words(words))
            prev_par = par
        confidence = statistics.mean(confs) if confs else None
        return "\n".join(out).strip(), confidence


class _EasyOcrEngine:
    def __init__(self, langs: str) -> None:
        import easyocr

        codes = []
        for code in langs.split("+"):
            mapped = _TESS_TO_EASY.get(code.strip())
            if mapped and mapped not in codes:
                codes.append(mapped)
        self.reader = easyocr.Reader(codes or ["ru", "en"], verbose=False)

    def read(self, image: Image.Image, upscale: bool) -> tuple[str, float | None]:
        import numpy as np

        arr = np.array(_maybe_upscale(image, upscale))
        results = self.reader.readtext(arr, detail=1, paragraph=False)
        items, confs = [], []
        for box, text, conf in results:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            items.append((min(xs), min(ys), max(xs), max(ys), str(text)))
            confs.append(float(conf) * 100)
        return _group_into_lines(items), (statistics.mean(confs) if confs else None)


class _PaddleEngine:
    def __init__(self, langs: str) -> None:
        from paddleocr import PaddleOCR

        lang = "en"
        for code in langs.split("+"):
            code = code.strip()
            if code != "eng" and code in _TESS_TO_PADDLE:
                lang = _TESS_TO_PADDLE[code]
                break
        try:  # PaddleOCR 3.x
            self.ocr = PaddleOCR(lang=lang, use_doc_orientation_classify=False,
                                 use_doc_unwarping=False, use_textline_orientation=False)
        except (TypeError, ValueError):  # PaddleOCR 2.x
            self.ocr = PaddleOCR(lang=lang, use_angle_cls=False, show_log=False)

    def read(self, image: Image.Image, upscale: bool) -> tuple[str, float | None]:
        import numpy as np

        arr = np.array(_maybe_upscale(image, upscale))[:, :, ::-1].copy()  # RGB -> BGR
        items, confs = [], []
        if hasattr(self.ocr, "predict"):
            for res in self.ocr.predict(arr) or []:
                texts = list(res["rec_texts"])
                scores = list(res.get("rec_scores", [])) if hasattr(res, "get") else []
                boxes = res.get("rec_boxes") if hasattr(res, "get") else None
                for idx, text in enumerate(texts):
                    if boxes is not None and len(boxes) > idx:
                        x0, y0, x1, y1 = (float(v) for v in boxes[idx][:4])
                    else:
                        x0, y0, x1, y1 = 0.0, float(idx * 20), 100.0, float(idx * 20 + 15)
                    items.append((x0, y0, x1, y1, str(text)))
                    if idx < len(scores):
                        confs.append(float(scores[idx]) * 100)
        else:
            pages = self.ocr.ocr(arr, cls=False) or []
            for page in pages:
                for box, (text, score) in page or []:
                    xs = [float(p[0]) for p in box]
                    ys = [float(p[1]) for p in box]
                    items.append((min(xs), min(ys), max(xs), max(ys), str(text)))
                    confs.append(float(score) * 100)
        return _group_into_lines(items), (statistics.mean(confs) if confs else None)


_INSTALL_HINTS = {
    "tesseract": "pip install pytesseract (и установите сам Tesseract OCR)",
    "easyocr": "pip install easyocr",
    "paddleocr": "pip install paddleocr paddlepaddle",
}


class TextReader:
    """Кэширует движки (EasyOCR/Paddle грузятся долго) и распознаёт изображение."""

    def __init__(self) -> None:
        self._engines: dict[tuple, object] = {}

    def _engine(self, cfg: dict):
        name = cfg.get("ocr_engine", "tesseract")
        langs = cfg.get("ocr_languages", "rus+eng")
        if name == "tesseract":
            key = (name, cfg.get("tesseract_cmd", ""), cfg.get("tessdata_prefix", ""))
        else:
            key = (name, langs)
        engine = self._engines.get(key)
        if engine is None:
            if name == "easyocr":
                engine = _EasyOcrEngine(langs)
            elif name == "paddleocr":
                engine = _PaddleEngine(langs)
            else:
                engine = _TesseractEngine(cfg.get("tesseract_cmd", ""), cfg.get("tessdata_prefix", ""))
            self._engines[key] = engine
        return engine

    def read(self, image: Image.Image, cfg: dict) -> OcrResult:
        name = cfg.get("ocr_engine", "tesseract")
        started = time.monotonic()
        try:
            engine = self._engine(cfg)
            upscale = bool(cfg.get("ocr_upscale", True))
            if isinstance(engine, _TesseractEngine):
                text, conf = engine.read(image, cfg.get("ocr_languages", "rus+eng"),
                                         int(cfg.get("tesseract_psm", 3)), upscale)
            else:
                text, conf = engine.read(image, upscale)
            return OcrResult(text=text, confidence=conf, engine=name, elapsed=time.monotonic() - started)
        except OcrError as exc:
            return OcrResult.failed(name, str(exc))
        except ImportError as exc:
            log.warning("OCR import error: %s", exc)
            return OcrResult.failed(name, f"Модуль для {name} не установлен: {_INSTALL_HINTS.get(name, '')}")
        except MemoryError:
            return OcrResult.failed(name, "Недостаточно памяти для OCR. Выберите область поменьше.")
        except Exception as exc:
            log.exception("OCR failure")
            return OcrResult.failed(name, f"Ошибка OCR ({name}): {exc}")

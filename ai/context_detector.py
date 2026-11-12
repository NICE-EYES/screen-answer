"""Быстрое локальное определение типа содержимого экрана по OCR-тексту.

Работает без сети и без API: используется в автоматическом режиме для
подсказок («Открыт тест с 4 вариантами») и как подсказка для модели.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

# ---------------------------------------------------------------- шаблоны
_OPTION_RE = re.compile(
    r"""^\s*(?:
        [(\[]?\s*(?:[A-HА-ЕСНa-hа-ес]|[0-9])\s*[)\].:]\s*(?=\S)    # A) a. 1) (B) [в]; OCR путает A/А, C/С, D/0
      | (?:[○◯●◉◎☐☑☒□■▢◻◼◇◆⚪⚫]|\(\s?\)|\[\s?\]|\(\s?[xх•]\s?\))\s*(?=\S)   # радио/чекбоксы
    )""",
    re.X,
)
_RADIO_LIKE_RE = re.compile(r"^\s*[Oo0©®@]\s+(?=[A-Za-zА-Яа-яЁё0-9])")
_QUESTION_CUES = re.compile(
    r"вопрос|выберите|укажите|отметьте|правильн|верн(ый|ое|ая|ые)|неверн|тест|задани|ответ|"
    r"question|choose|select|which|correct|true or false|pick|answer|quiz",
    re.I,
)
_ERROR_STRONG = re.compile(
    r"Traceback \(most recent call last\)|\b\w*(Error|Exception)\b\s*:|\bошибка\b|\bfailed\b|"
    r"не удалось|Uncaught|Segmentation fault|exit code [1-9]|errno|stack trace|"
    r"\bat .+\(.+:\d+(:\d+)?\)|line \d+, in|\berror\s+[A-Z]{1,4}\d{2,5}|fatal:|panic:|ERR!",
    re.I,
)
_CODE_TOKENS = re.compile(
    r"^\s*(def |class |import |from \w+ import|function\b|return\b|const |let |var |public |private |"
    r"protected |static |#include|using |namespace |package |fn |func |if\s*\(|for\s*\(|while\s*\(|"
    r"elif |else\s*[:{]|try\s*[:{]|catch\s*\(|SELECT |INSERT |UPDATE |CREATE TABLE|<\w+[ >]|</\w+>)"
    r"|[;{}]\s*$|=>|::|->|\+\+|!=|==|\bself\.|console\.|System\.out|printf?\(|print\(",
    re.I,
)
_MATH_TOKENS = re.compile(
    r"[=≠≈≤≥±×÷√∑∏∫∞πΔ∂^²³⁴½¼¾]|\b(sin|cos|tg|tan|ctg|cot|log|ln|lg|lim|arcsin|arccos|arctg|det)\b|"
    r"\d\s*[+\-*/:]\s*\d|\b[xyzn]\s*[\^²³]|\bf\s*\(\s*x\s*\)|\bdx\b|\d+\s*%|\bуравнени|\bвычисл|\bреши|"
    r"\bнайдите\b|\bsolve\b|\bequation\b|\bcalculate\b|\bинтеграл|\bпроизводн",
    re.I,
)
_GRAPH_WORDS = re.compile(r"график|диаграмм|гистограмм|chart|graph|plot|axis|\bось\b|легенда|legend|тренд|trend",
                          re.I)
_VIDEO_WORDS = re.compile(
    r"\bzoom\b|microsoft teams|\bteams\b|google meet|skype|discord|телемост|микрофон|демонстрац|"
    r"\bmute\b|unmute|участник|participants|leave meeting|покинуть|выключить камеру|stop video|"
    r"запись встречи|chat\s*\(\d+\)",
    re.I,
)
_GAME_WORDS = re.compile(
    r"\bhp\b|\bmp\b|\blvl\b|\blevel\b|уровень|\bscore\b|очки|health|здоровье|инвентарь|inventory|"
    r"\bquest\b|квест|mana|мана|респаун|respawn|\bxp\b|опыт",
    re.I,
)
_WEB_WORDS = re.compile(
    r"https?://|www\.|\.(com|ru|org|net|io|рф)\b|cookie|войти|log ?in|sign in|регистрац|главная|"
    r"подписаться|subscribe|поиск|search|корзина|cart|меню",
    re.I,
)
_CELL_SPLIT = re.compile(r"\s{3,}|\t|\s*\|\s*")

_SCRIPTS = {
    "cyrillic": re.compile(r"[а-яёіїєґўқғүұһөә]", re.I),
    "latin": re.compile(r"[a-zäöüßàâçéèêëîïôûùÿñáíóúãõąćęłńśźżşğı]", re.I),
    "cjk": re.compile(r"[一-鿿぀-ヿ]"),
    "hangul": re.compile(r"[가-힯]"),
    "arabic": re.compile(r"[؀-ۿ]"),
    "greek": re.compile(r"[Ͱ-Ͽ]"),
    "hebrew": re.compile(r"[֐-׿]"),
    "devanagari": re.compile(r"[ऀ-ॿ]"),
}
_SCRIPT_NAMES = {
    "cyrillic": "кириллица", "latin": "латиница", "cjk": "китайский/японский", "hangul": "корейский",
    "arabic": "арабский", "greek": "греческий", "hebrew": "иврит", "devanagari": "хинди/деванагари",
}
_LANG_SCRIPT = {
    "Русский": "cyrillic", "Українська": "cyrillic", "Беларуская": "cyrillic", "Қазақша": "cyrillic",
    "中文": "cjk", "日本語": "cjk", "한국어": "hangul",
}
_LATIN_LANG_HINTS = [
    ("английский", re.compile(r"\b(the|and|of|to|is|are|with|that|this|for|you)\b", re.I)),
    ("немецкий", re.compile(r"\b(der|die|das|und|ist|nicht|mit|ein|eine|ich)\b", re.I)),
    ("французский", re.compile(r"\b(le|la|les|et|est|des|une|pour|avec|vous)\b", re.I)),
    ("испанский", re.compile(r"\b(el|los|las|que|es|una|para|con|por|del)\b", re.I)),
    ("итальянский", re.compile(r"\b(il|che|non|una|per|sono|della|con|gli)\b", re.I)),
]
_CODE_LANGS = [
    ("Python", re.compile(r"^\s*(def |import |from \w+ import|elif |class \w+.*:)|self\.|print\(|Traceback", re.M)),
    ("JavaScript/TypeScript", re.compile(r"\bconst |\blet |=>|console\.|function\s*\w*\(|\binterface\b")),
    ("Java/C#", re.compile(r"public (static )?(void|class)|System\.out|namespace |using System")),
    ("C/C++", re.compile(r"#include|std::|printf\(|int main\(")),
    ("SQL", re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE)\b.+\b(FROM|INTO|SET|WHERE)\b", re.I)),
    ("HTML/CSS", re.compile(r"</?(div|span|html|body|head|a|p)\b|\{\s*[\w-]+\s*:\s*[^;]+;")),
    ("PHP", re.compile(r"<\?php|\$\w+\s*=|->\w+\(")),
]


@dataclass
class ScreenContext:
    kind: str = "unknown"
    title: str = "Не определено"
    hint: str = ""
    tags: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    question: str = ""
    readable: bool = True
    confidence: float | None = None
    foreign: str = ""
    code_language: str = ""

    def describe(self) -> str:
        """Текст для промпта модели."""
        lines = [f"Тип экрана (локальная эвристика, может ошибаться): {self.title}"]
        if self.tags:
            lines.append("Признаки: " + ", ".join(self.tags))
        if self.question:
            lines.append(f"Предполагаемый вопрос: {self.question}")
        if self.options:
            lines.append(f"Найдено вариантов ответа: {len(self.options)}")
            lines.extend(f"  {opt}" for opt in self.options)
        if self.code_language:
            lines.append(f"Вероятный язык кода: {self.code_language}")
        if self.foreign:
            lines.append(f"Иностранный текст: {self.foreign}")
        if self.confidence is not None:
            quality = "хорошее" if self.confidence >= 75 else "среднее" if self.confidence >= 55 else "низкое"
            lines.append(f"Качество OCR: {quality} ({self.confidence:.0f}%)")
        if not self.readable:
            lines.append("ВНИМАНИЕ: текст распознан плохо — опирайся на изображение; если и там не читается, "
                         "попроси пользователя выделить область крупнее.")
        return "\n".join(lines)


def _dominant_script(text: str) -> tuple[str, float]:
    counts = {name: len(rx.findall(text)) for name, rx in _SCRIPTS.items()}
    total = sum(counts.values())
    if total == 0:
        return "", 0.0
    name = max(counts, key=counts.get)
    return name, counts[name] / total


def _guess_latin_language(text: str) -> str:
    best, best_hits = "", 0
    for name, rx in _LATIN_LANG_HINTS:
        hits = len(rx.findall(text))
        if hits > best_hits:
            best, best_hits = name, hits
    return best if best_hits >= 2 else ""


def _code_language(text: str) -> str:
    for name, rx in _CODE_LANGS:
        if rx.search(text):
            return name
    return ""


def detect_context(text: str, confidence: float | None = None, answer_language: str = "Русский",
                   ocr_enabled: bool = True, ocr_error: str = "") -> ScreenContext:
    ctx = ScreenContext(confidence=confidence)
    if not ocr_enabled:
        ctx.kind, ctx.title = "unknown", "OCR выключен"
        ctx.hint = "OCR выключен — анализ возможен только по изображению"
        return ctx
    if ocr_error:
        ctx.kind, ctx.title = "unknown", "Ошибка OCR"
        ctx.hint = f"⚠ OCR: {ocr_error}"
        return ctx

    clean = (text or "").strip()
    letters = sum(ch.isalpha() for ch in clean)
    if letters < 3:
        ctx.kind, ctx.title, ctx.readable = "empty", "Текст не найден", False
        ctx.hint = "Текст на экране не найден (возможно, изображение, график или видео)"
        return ctx

    lines = [ln for ln in clean.splitlines() if ln.strip()]
    n_lines = len(lines)

    # --- читаемость ---
    tokens = clean.split()
    junk = sum(1 for t in tokens if sum(c.isalnum() for c in t) < len(t) / 2)
    junk_ratio = junk / max(1, len(tokens))
    ctx.readable = (confidence is None or confidence >= 55) and junk_ratio < 0.45

    # --- сигналы ---
    option_lines = [ln.strip() for ln in lines if _OPTION_RE.match(ln)]
    radio_like = [ln.strip() for ln in lines if _RADIO_LIKE_RE.match(ln)]
    has_question_cue = bool(_QUESTION_CUES.search(clean)) or "?" in clean
    error_hits = len(_ERROR_STRONG.findall(clean))
    code_hits = sum(1 for ln in lines if _CODE_TOKENS.search(ln))
    code_ratio = code_hits / max(1, n_lines)
    math_hits = len(_MATH_TOKENS.findall(clean))
    table_rows = [len(_CELL_SPLIT.split(ln.strip())) for ln in lines if len(_CELL_SPLIT.split(ln.strip())) >= 3]
    numeric_lines = sum(1 for ln in lines if re.fullmatch(r"[\s\d.,%\-–+$€₽]+", ln))
    graph_hits = len(_GRAPH_WORDS.findall(clean))
    video_hits = len(set(m.group(0).lower() for m in _VIDEO_WORDS.finditer(clean)))
    game_hits = len(set(m.group(0).lower() for m in _GAME_WORDS.finditer(clean)))
    web_hits = len(_WEB_WORDS.findall(clean))
    avg_len = statistics.mean(len(ln) for ln in lines) if lines else 0

    is_code = code_hits >= 3 and code_ratio >= 0.25
    is_math = math_hits >= 3 or (math_hits >= 2 and "=" in clean)
    options = option_lines if len(option_lines) >= 2 else (radio_like if len(radio_like) >= 3 else [])
    is_test = (len(options) >= 2 and has_question_cue) or \
        sum(1 for ln in option_lines if re.match(r"^\s*([○◯●◉☐☑□■▢◇⚪]|\(\s?\)|\[\s?\])", ln)) >= 2
    is_table = len(table_rows) >= 3
    is_graph = (graph_hits >= 1 and numeric_lines >= 3) or (n_lines >= 6 and numeric_lines / n_lines > 0.5)

    if is_math:
        ctx.tags.append("математика")
    if is_code:
        ctx.code_language = _code_language(clean)
        ctx.tags.append("код" + (f" ({ctx.code_language})" if ctx.code_language else ""))
    if error_hits:
        ctx.tags.append("сообщение об ошибке")
    if is_table:
        ctx.tags.append("таблица")
    if is_graph:
        ctx.tags.append("график/диаграмма")

    # --- иностранный текст ---
    script, share = _dominant_script(clean)
    expected = _LANG_SCRIPT.get(answer_language, "latin")
    if script and share >= 0.6 and script != expected and not is_code and letters >= 25:
        if script == "latin":
            lang = _guess_latin_language(clean)
            ctx.foreign = lang or "латиница"
        else:
            ctx.foreign = _SCRIPT_NAMES.get(script, script)
        ctx.tags.append(f"иностранный текст: {ctx.foreign}")

    # --- вопрос ---
    if options:
        first_opt = next((i for i, ln in enumerate(lines) if ln.strip() in options), 0)
        before = [ln.strip() for ln in lines[max(0, first_opt - 3):first_opt]]
        ctx.question = " ".join(before)[:300]
        ctx.options = [o[:120] for o in options[:8]]
    else:
        q = next((ln.strip() for ln in lines if ln.strip().endswith("?")), "")
        ctx.question = q[:300]

    # --- итоговый тип (по приоритету) ---
    if is_test:
        n = len(options)
        ctx.kind = "test"
        if is_math:
            ctx.title = "Тест по математике"
            ctx.hint = f"На экране вопрос по математике с вариантами ответа ({n})"
        else:
            ctx.title = "Тест / вопрос с вариантами"
            ctx.hint = f"Открыт тест с {n} вариант{'ами' if n != 1 else 'ом'} ответа"
    elif error_hits and (is_code or error_hits >= 2 or n_lines <= 15):
        ctx.kind = "error"
        ctx.title = "Ошибка" + (" в коде" if is_code else "")
        ctx.hint = "Обнаружена ошибка в коде" if is_code else "На экране сообщение об ошибке"
    elif is_code:
        ctx.kind, ctx.title = "code", "Код"
        ctx.hint = "На экране код" + (f" ({ctx.code_language})" if ctx.code_language else "")
    elif is_math:
        ctx.kind, ctx.title = "math", "Математика"
        ctx.hint = "На экране вопрос по математике" if has_question_cue else "На экране математические выражения"
    elif is_table:
        cols = int(statistics.median(table_rows))
        ctx.kind, ctx.title = "table", "Таблица"
        ctx.hint = f"На экране таблица (~{len(table_rows)} строк × {cols} столбцов)"
    elif is_graph:
        ctx.kind, ctx.title = "graph", "График"
        ctx.hint = "Похоже на график или диаграмму"
    elif video_hits >= 2:
        ctx.kind, ctx.title = "video_call", "Видеозвонок"
        ctx.hint = "Похоже, идёт видеозвонок"
    elif game_hits >= 3:
        ctx.kind, ctx.title = "game", "Игра"
        ctx.hint = "Похоже на игру"
    elif ctx.foreign:
        ctx.kind, ctx.title = "foreign", "Иностранный текст"
        ctx.hint = f"Текст на иностранном языке ({ctx.foreign}) — можно спросить «Переведи»"
    elif web_hits >= 2:
        ctx.kind, ctx.title = "website", "Сайт"
        ctx.hint = "Открыт сайт" + (" с вопросом" if "?" in clean else "")
    elif n_lines >= 6 and avg_len >= 40:
        ctx.kind, ctx.title = "document", "Документ"
        ctx.hint = f"Открыт документ / длинный текст (~{len(tokens)} слов)"
    else:
        ctx.kind, ctx.title = "text", "Текст"
        ctx.hint = f"На экране текст ({n_lines} строк)" + (" с вопросом" if "?" in clean else "")

    if ctx.foreign and ctx.kind in ("test", "math", "text", "document", "website"):
        ctx.hint += f" · язык: {ctx.foreign}"
    if not ctx.readable:
        ctx.hint += " · ⚠ текст читается плохо — выделите область крупнее"
    return ctx

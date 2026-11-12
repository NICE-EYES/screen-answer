"""Окно настроек."""
from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from ai.vision_analyzer import AiError, VisionAnalyzer
from gui.theme import build_stylesheet
from ocr.text_reader import find_tesseract
from utils.config import (
    ANALYSIS_INTERVALS, ANSWER_LANGUAGES, APP_NAME, CONFIG_PATH, IMAGE_DETAILS, MODEL_PRESETS, OCR_ENGINES, THEMES,
    AppConfig,
)
from utils.hotkeys import HotkeyParseError, parse_hotkey

_ORPHAN_THREADS: set = set()

CLOUD_WARNING = (
    "Включая облачный ИИ, вы соглашаетесь, что изображение выбранной области экрана и распознанный текст "
    "будут отправляться на сервер ИИ (OpenAI или указанный вами совместимый сервис).\n\n"
    "⚠ На экране могут быть личные данные: пароли, переписка, документы, банковские сведения, "
    "чужие персональные данные.\n\n"
    "Рекомендации:\n"
    "• выделяйте только нужную область (кнопка «Область»);\n"
    "• включите «Не отправлять скриншоты в облако», если достаточно текста;\n"
    "• не задавайте вопросы, когда на экране конфиденциальная информация.\n\n"
    "Включить облачный ИИ?"
)


def confirm_cloud_warning(parent) -> bool:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(f"{APP_NAME} — конфиденциальность")
    box.setText(CLOUD_WARNING)
    yes = box.addButton("Понимаю, включить", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
    box.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    box.exec()
    return box.clickedButton() is yes


class _TestThread(QThread):
    done = pyqtSignal(bool, str)

    def __init__(self, cfg: dict, parent=None) -> None:
        super().__init__(parent)
        self._cfg = cfg

    def run(self) -> None:
        try:
            self.done.emit(True, VisionAnalyzer().test_connection(self._cfg))
        except AiError as exc:
            self.done.emit(False, str(exc))
        except Exception as exc:
            self.done.emit(False, f"Ошибка: {exc}")


class SettingsDialog(QDialog):
    TABS = ("ai", "privacy", "capture", "ocr", "appearance", "hotkeys")

    def __init__(self, cfg: AppConfig, parent=None, initial_tab: str | None = None,
                 erase_callback: Callable[[], None] | None = None) -> None:
        super().__init__(parent)
        self.cfg = cfg.copy()
        self._erase_callback = erase_callback
        self._test_thread: _TestThread | None = None
        self.setWindowTitle(f"{APP_NAME} — настройки")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(540)
        self.setStyleSheet(build_stylesheet(cfg.theme, cfg.font_size))

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_ai(), "ИИ")
        self.tabs.addTab(self._tab_privacy(), "Конфиденциальность")
        self.tabs.addTab(self._tab_capture(), "Захват")
        self.tabs.addTab(self._tab_ocr(), "OCR")
        self.tabs.addTab(self._tab_appearance(), "Вид")
        self.tabs.addTab(self._tab_hotkeys(), "Клавиши")
        layout.addWidget(self.tabs)
        if initial_tab in self.TABS:
            self.tabs.setCurrentIndex(self.TABS.index(initial_tab))

        note = QLabel(f"Настройки сохраняются в {CONFIG_PATH}")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Сохранить")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ---------------------------------------------------------------- ИИ
    def _tab_ai(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        key_row = QHBoxLayout()
        self.ed_key = QLineEdit(self.cfg.api_key)
        self.ed_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.ed_key.setPlaceholderText("sk-…  (пусто — берётся из переменной OPENAI_API_KEY)")
        show = QCheckBox("показать")
        show.toggled.connect(lambda on: self.ed_key.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        key_row.addWidget(self.ed_key, 1)
        key_row.addWidget(show)
        f.addRow("API-ключ:", key_row)
        self.ed_base = QLineEdit(self.cfg.api_base_url)
        self.ed_base.setPlaceholderText("https://api.openai.com/v1 (по умолчанию)")
        f.addRow("Адрес API:", self.ed_base)
        self.cb_model = QComboBox()
        self.cb_model.setEditable(True)
        self.cb_model.addItems(MODEL_PRESETS)
        self.cb_model.setCurrentText(self.cfg.model)
        f.addRow("Модель (с поддержкой изображений):", self.cb_model)
        self.ed_tmodel = QLineEdit(self.cfg.transcribe_model)
        f.addRow("Модель распознавания речи:", self.ed_tmodel)
        self.cb_detail = QComboBox()
        self.cb_detail.addItems(IMAGE_DETAILS)
        self.cb_detail.setCurrentText(self.cfg.image_detail)
        self.cb_detail.setToolTip("low — дешевле, но мелкий текст хуже; high — точнее и дороже")
        f.addRow("Детализация изображения:", self.cb_detail)
        self.sp_side = self._spin(512, 4096, self.cfg.max_image_side, 128, " px")
        f.addRow("Макс. сторона скриншота:", self.sp_side)
        self.sp_quality = self._spin(30, 95, self.cfg.jpeg_quality, 5)
        f.addRow("Качество JPEG:", self.sp_quality)
        self.sp_tokens = self._spin(100, 16000, self.cfg.max_answer_tokens, 100)
        f.addRow("Лимит токенов ответа:", self.sp_tokens)
        self.sp_temp = QDoubleSpinBox()
        self.sp_temp.setRange(0.0, 2.0)
        self.sp_temp.setSingleStep(0.1)
        self.sp_temp.setValue(self.cfg.temperature)
        f.addRow("Температура:", self.sp_temp)
        self.sp_hist = self._spin(0, 10, self.cfg.history_depth)
        self.sp_hist.setToolTip("Сколько последних вопросов/ответов отправлять как контекст (рекомендуется 3–5)")
        f.addRow("Глубина истории диалога:", self.sp_hist)
        self.sp_timeout = self._spin(10, 300, self.cfg.request_timeout, 5, " с")
        f.addRow("Таймаут запроса:", self.sp_timeout)
        test_row = QHBoxLayout()
        self.btn_test = QPushButton("Проверить подключение")
        self.btn_test.clicked.connect(self._test_connection)
        self.lbl_test = QLabel("")
        self.lbl_test.setWordWrap(True)
        test_row.addWidget(self.btn_test)
        test_row.addWidget(self.lbl_test, 1)
        f.addRow(test_row)
        return w

    def _tab_privacy(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.chk_cloud = QCheckBox("Использовать облачный ИИ (OpenAI Vision API)")
        self.chk_cloud.setChecked(self.cfg.cloud_ai_enabled)
        self.chk_cloud.toggled.connect(self._on_cloud_toggled)
        v.addWidget(self.chk_cloud)
        self.chk_no_img = QCheckBox("Не отправлять скриншоты в облако (только распознанный текст)")
        self.chk_no_img.setChecked(self.cfg.no_cloud_screenshots)
        v.addWidget(self.chk_no_img)
        self.chk_save_shots = QCheckBox("Сохранять скриншоты на диск (папка screenshots рядом с программой)")
        self.chk_save_shots.setChecked(self.cfg.save_screenshots)
        v.addWidget(self.chk_save_shots)
        self.chk_save_hist = QCheckBox("Сохранять историю вопросов в локальную базу SQLite")
        self.chk_save_hist.setChecked(self.cfg.save_history)
        v.addWidget(self.chk_save_hist)
        self.chk_hide = QCheckBox("Прятать окно программы на долю секунды во время снимка (чтобы оно не попало "
                                  "на скриншот)")
        self.chk_hide.setChecked(self.cfg.hide_overlay_during_capture)
        v.addWidget(self.chk_hide)
        info = QLabel(
            "• OCR выполняется локально, на вашем компьютере.\n"
            "• Скриншоты не сохраняются на диск, если не включено выше.\n"
            "• История хранится только локально (history.db).\n"
            "• В облако данные уходят только при включённом облачном ИИ и только при запросе к модели.")
        info.setObjectName("Muted")
        info.setWordWrap(True)
        v.addWidget(info)
        btn_erase = QPushButton("Стереть историю сейчас")
        btn_erase.setObjectName("Danger")
        btn_erase.setEnabled(self._erase_callback is not None)
        btn_erase.clicked.connect(lambda: self._erase_callback and self._erase_callback())
        v.addWidget(btn_erase, 0, Qt.AlignmentFlag.AlignLeft)
        v.addStretch(1)
        return w

    def _tab_capture(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.cb_interval = QComboBox()
        for s in ANALYSIS_INTERVALS:
            self.cb_interval.addItem(f"{s} секунд", s)
        self.cb_interval.setCurrentIndex(max(0, self.cb_interval.findData(self.cfg.analysis_interval)))
        f.addRow("Частота автоанализа:", self.cb_interval)
        self.sp_threshold = QDoubleSpinBox()
        self.sp_threshold.setRange(0.0, 50.0)
        self.sp_threshold.setDecimals(2)
        self.sp_threshold.setSingleStep(0.05)
        self.sp_threshold.setSuffix(" %")
        self.sp_threshold.setValue(self.cfg.change_threshold)
        self.sp_threshold.setToolTip("Доля изменённых пикселей, при которой экран считается изменившимся. "
                                     "Меньше — чувствительнее.")
        f.addRow("Порог изменения экрана:", self.sp_threshold)
        self.chk_fresh = QCheckBox("Делать новый снимок при каждом вопросе")
        self.chk_fresh.setChecked(self.cfg.fresh_capture_on_ask)
        f.addRow(self.chk_fresh)
        self.chk_auto_ai = QCheckBox("ИИ-сводка в автоанализе (расходует API; без неё — только локальные подсказки)")
        self.chk_auto_ai.setChecked(self.cfg.auto_ai_summary)
        f.addRow(self.chk_auto_ai)
        self.sp_min_ai = self._spin(5, 3600, self.cfg.min_ai_interval, 5, " с")
        f.addRow("Мин. интервал ИИ-сводок:", self.sp_min_ai)
        self.chk_voice_ask = QCheckBox("Сразу спрашивать после голосового ввода")
        self.chk_voice_ask.setChecked(self.cfg.voice_auto_ask)
        f.addRow(self.chk_voice_ask)
        return w

    def _tab_ocr(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.chk_ocr = QCheckBox("Распознавать текст на скриншотах (локально)")
        self.chk_ocr.setChecked(self.cfg.ocr_enabled)
        f.addRow(self.chk_ocr)
        self.cb_engine = QComboBox()
        for key, title in OCR_ENGINES.items():
            self.cb_engine.addItem(title, key)
        self.cb_engine.setCurrentIndex(max(0, self.cb_engine.findData(self.cfg.ocr_engine)))
        f.addRow("Движок:", self.cb_engine)
        self.ed_langs = QLineEdit(self.cfg.ocr_languages)
        self.ed_langs.setToolTip("Коды Tesseract через +, например rus+eng, eng+deu, ukr+eng")
        f.addRow("Языки распознавания:", self.ed_langs)
        self.ed_tess = QLineEdit(self.cfg.tesseract_cmd)
        self.ed_tess.setPlaceholderText(find_tesseract() or r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        f.addRow("Путь к tesseract.exe:", self._with_browse(self.ed_tess, file=True))
        self.ed_tessdata = QLineEdit(self.cfg.tessdata_prefix)
        self.ed_tessdata.setPlaceholderText(r"C:\Program Files\Tesseract-OCR\tessdata (необязательно)")
        f.addRow("TESSDATA_PREFIX:", self._with_browse(self.ed_tessdata, file=False))
        self.cb_psm = QComboBox()
        for psm, title in ((3, "3 — автоматически (по умолчанию)"), (4, "4 — одна колонка текста"),
                           (6, "6 — единый блок текста"), (11, "11 — разреженный текст (интерфейсы)")):
            self.cb_psm.addItem(title, psm)
        self.cb_psm.setCurrentIndex(max(0, self.cb_psm.findData(self.cfg.tesseract_psm)))
        f.addRow("Режим Tesseract (PSM):", self.cb_psm)
        self.chk_upscale = QCheckBox("Увеличивать мелкий текст перед распознаванием")
        self.chk_upscale.setChecked(self.cfg.ocr_upscale)
        f.addRow(self.chk_upscale)
        row = QHBoxLayout()
        btn = QPushButton("Проверить Tesseract")
        btn.clicked.connect(self._check_tesseract)
        self.lbl_tess = QLabel("")
        self.lbl_tess.setWordWrap(True)
        row.addWidget(btn)
        row.addWidget(self.lbl_tess, 1)
        f.addRow(row)
        return w

    def _tab_appearance(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.cb_theme = QComboBox()
        for key, title in THEMES.items():
            self.cb_theme.addItem(title, key)
        self.cb_theme.setCurrentIndex(max(0, self.cb_theme.findData(self.cfg.theme)))
        f.addRow("Тема:", self.cb_theme)
        self.sp_opacity = self._spin(10, 100, self.cfg.opacity, 5, " %")
        f.addRow("Непрозрачность окна:", self.sp_opacity)
        self.sp_font = self._spin(8, 20, self.cfg.font_size, 1, " pt")
        f.addRow("Размер шрифта:", self.sp_font)
        self.cb_lang = QComboBox()
        self.cb_lang.addItems(list(ANSWER_LANGUAGES))
        self.cb_lang.setCurrentText(self.cfg.answer_language)
        f.addRow("Язык ответа:", self.cb_lang)
        return w

    def _tab_hotkeys(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.ed_hk_capture = QLineEdit(self.cfg.hotkey_capture_answer)
        self.ed_hk_ask = QLineEdit(self.cfg.hotkey_ask)
        self.ed_hk_toggle = QLineEdit(self.cfg.hotkey_toggle)
        self.ed_hk_stop = QLineEdit(self.cfg.hotkey_stop)
        f.addRow("Сделать скриншот и ответить:", self.ed_hk_capture)
        f.addRow("Открыть окно вопроса:", self.ed_hk_ask)
        f.addRow("Скрыть / показать окно:", self.ed_hk_toggle)
        f.addRow("Полностью остановить анализ:", self.ed_hk_stop)
        note = QLabel("Формат: Ctrl+Alt+S, Ctrl+Shift+F2, Win+Alt+Q. Пустое поле — сочетание отключено.\n"
                      "Горячие клавиши глобальные и работают в любом приложении (Windows).")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        f.addRow(note)
        return w

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _spin(lo: int, hi: int, value: int, step: int = 1, suffix: str = "") -> QSpinBox:
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setValue(int(value))
        if suffix:
            sp.setSuffix(suffix)
        return sp

    def _with_browse(self, edit: QLineEdit, file: bool) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        btn = QPushButton("Обзор…")

        def browse() -> None:
            if file:
                path, _ = QFileDialog.getOpenFileName(self, "tesseract.exe", "", "tesseract (tesseract*);;Все (*)")
            else:
                path = QFileDialog.getExistingDirectory(self, "Папка tessdata")
            if path:
                edit.setText(path)

        btn.clicked.connect(browse)
        h.addWidget(btn)
        return box

    def _on_cloud_toggled(self, on: bool) -> None:
        if on and not self.cfg.cloud_warning_accepted:
            if confirm_cloud_warning(self):
                self.cfg.cloud_warning_accepted = True
            else:
                self.chk_cloud.blockSignals(True)
                self.chk_cloud.setChecked(False)
                self.chk_cloud.blockSignals(False)

    def _test_connection(self) -> None:
        if self._test_thread is not None and self._test_thread.isRunning():
            return
        cfg = self.cfg.snapshot()
        cfg.update(api_key=self.ed_key.text().strip(), api_base_url=self.ed_base.text().strip(),
                   model=self.cb_model.currentText().strip(), request_timeout=self.sp_timeout.value())
        self.lbl_test.setText("Проверяю…")
        self.btn_test.setEnabled(False)
        self._test_thread = _TestThread(cfg, self)
        self._test_thread.done.connect(self._on_test_done)
        self._test_thread.start()

    def _on_test_done(self, ok: bool, message: str) -> None:
        self.btn_test.setEnabled(True)
        self.lbl_test.setText(("✅ " if ok else "❌ ") + message)

    def _check_tesseract(self) -> None:
        try:
            import pytesseract
        except ImportError:
            self.lbl_tess.setText("❌ pytesseract не установлен: pip install pytesseract")
            return
        path = find_tesseract(self.ed_tess.text().strip())
        if not path:
            self.lbl_tess.setText("❌ tesseract.exe не найден — установите Tesseract или укажите путь")
            return
        pytesseract.pytesseract.tesseract_cmd = path
        try:
            version = pytesseract.get_tesseract_version()
            langs = pytesseract.get_languages(config="")
        except Exception as exc:
            self.lbl_tess.setText(f"❌ {exc}")
            return
        need = [x for x in self.ed_langs.text().split("+") if x.strip()]
        missing = [x for x in need if x.strip() not in langs]
        msg = f"✅ Tesseract {version}. Языки: {', '.join(sorted(langs))[:200]}"
        if missing:
            msg += f"\n⚠ Нет языков: {', '.join(missing)}"
        self.lbl_tess.setText(msg)

    def _on_accept(self) -> None:
        hotkeys = {"Сделать скриншот и ответить": self.ed_hk_capture, "Открыть окно вопроса": self.ed_hk_ask,
                   "Скрыть/показать": self.ed_hk_toggle, "Остановить анализ": self.ed_hk_stop}
        seen: dict[str, str] = {}
        for title, edit in hotkeys.items():
            text = edit.text().strip()
            if not text:
                continue
            try:
                _, _, canonical = parse_hotkey(text)
            except HotkeyParseError as exc:
                self.tabs.setCurrentIndex(self.TABS.index("hotkeys"))
                QMessageBox.warning(self, APP_NAME, f"«{title}»: {exc}")
                return
            if canonical in seen:
                QMessageBox.warning(self, APP_NAME, f"Сочетание {canonical} назначено дважды: «{seen[canonical]}» "
                                                    f"и «{title}»")
                return
            seen[canonical] = title
            edit.setText(canonical)

        c = self.cfg
        c.api_key = self.ed_key.text().strip()
        c.api_base_url = self.ed_base.text().strip()
        c.model = self.cb_model.currentText().strip() or "gpt-4o-mini"
        c.transcribe_model = self.ed_tmodel.text().strip() or "whisper-1"
        c.image_detail = self.cb_detail.currentText()
        c.max_image_side = self.sp_side.value()
        c.jpeg_quality = self.sp_quality.value()
        c.max_answer_tokens = self.sp_tokens.value()
        c.temperature = self.sp_temp.value()
        c.history_depth = self.sp_hist.value()
        c.request_timeout = self.sp_timeout.value()
        c.cloud_ai_enabled = self.chk_cloud.isChecked()
        c.no_cloud_screenshots = self.chk_no_img.isChecked()
        c.save_screenshots = self.chk_save_shots.isChecked()
        c.save_history = self.chk_save_hist.isChecked()
        c.hide_overlay_during_capture = self.chk_hide.isChecked()
        c.analysis_interval = int(self.cb_interval.currentData())
        c.change_threshold = self.sp_threshold.value()
        c.fresh_capture_on_ask = self.chk_fresh.isChecked()
        c.auto_ai_summary = self.chk_auto_ai.isChecked()
        c.min_ai_interval = self.sp_min_ai.value()
        c.voice_auto_ask = self.chk_voice_ask.isChecked()
        c.ocr_enabled = self.chk_ocr.isChecked()
        c.ocr_engine = self.cb_engine.currentData()
        c.ocr_languages = self.ed_langs.text().strip().replace(" ", "") or "rus+eng"
        c.tesseract_cmd = self.ed_tess.text().strip()
        c.tessdata_prefix = self.ed_tessdata.text().strip()
        c.tesseract_psm = int(self.cb_psm.currentData())
        c.ocr_upscale = self.chk_upscale.isChecked()
        c.theme = self.cb_theme.currentData()
        c.opacity = self.sp_opacity.value()
        c.font_size = self.sp_font.value()
        c.answer_language = self.cb_lang.currentText()
        c.hotkey_capture_answer = self.ed_hk_capture.text().strip()
        c.hotkey_ask = self.ed_hk_ask.text().strip()
        c.hotkey_toggle = self.ed_hk_toggle.text().strip()
        c.hotkey_stop = self.ed_hk_stop.text().strip()
        c.validate()
        self.accept()

    def result_config(self) -> AppConfig:
        return self.cfg

    def done(self, result: int) -> None:
        if self._test_thread is not None and self._test_thread.isRunning():
            self._test_thread.wait(200)
            if self._test_thread.isRunning():
                # поток сам завершится по таймауту запроса; отцепляем его от диалога
                thread = self._test_thread
                thread.done.disconnect()
                thread.setParent(None)
                _ORPHAN_THREADS.add(thread)  # держим ссылку, пока поток не завершится
                thread.finished.connect(lambda t=thread: _ORPHAN_THREADS.discard(t))
        super().done(result)

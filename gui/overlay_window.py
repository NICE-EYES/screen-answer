"""Компактное плавающее окно поверх всех приложений.

Окно никогда не забирает фокус само по себе и не трогает другие окна —
оно только показывает результаты анализа и принимает вопросы.
"""
from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QActionGroup, QGuiApplication, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMenu, QPlainTextEdit, QProgressBar, QPushButton, QSizeGrip, QSlider,
    QTextBrowser, QToolButton, QVBoxLayout, QWidget,
)

from capture.screen_capture import IS_WINDOWS, Rect, get_window_rect
from gui.theme import STATUS_COLORS, build_stylesheet, make_app_icon
from storage.history_db import HistoryEntry
from utils.config import ANALYSIS_INTERVALS, ANSWER_LANGUAGES, APP_NAME, AppConfig

STATUS_TEXT = {
    "idle": "ожидание", "capture": "снимок…", "ocr": "распознаю текст…", "ai": "анализирую…",
    "ready": "готово", "nochange": "нет изменений", "error": "ошибка", "stopped": "остановлено",
    "recording": "запись голоса…",
}

PLACEHOLDER_ANSWER = (
    "### Как пользоваться\n"
    "1. **Область** — выберите монитор, активное окно или выделите участок мышью.\n"
    "2. Введите вопрос и нажмите **Спросить** (или Ctrl+Alt+S).\n"
    "3. Включите **Авто**, чтобы окно само подсказывало, что на экране.\n\n"
    "Горячие клавиши: Ctrl+Alt+S — снимок и ответ, Ctrl+Alt+Q — вопрос, "
    "Ctrl+Alt+H — скрыть/показать, Ctrl+Alt+X — стоп."
)


class _TitleBar(QWidget):
    """Заголовок: перетаскивание окна и двойной щелчок — свернуть в полоску."""

    def __init__(self, window: "OverlayWindow") -> None:
        super().__init__(window)
        self._window = window
        self._drag_offset: QPoint | None = None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self._window.windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
            self._drag_offset = event.globalPosition().toPoint() - self._window.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._window.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, _event) -> None:
        self._drag_offset = None

    def mouseDoubleClickEvent(self, _event) -> None:
        self._window.set_collapsed(not self._window.collapsed)


class OverlayWindow(QWidget):
    ask_requested = pyqtSignal(str)
    capture_now_requested = pyqtSignal()
    capture_mode_selected = pyqtSignal(str, int)
    select_region_requested = pyqtSignal()
    auto_toggled = pyqtSignal(bool)
    interval_changed = pyqtSignal(int)
    opacity_changed = pyqtSignal(int)
    language_changed = pyqtSignal(str)
    theme_toggle_requested = pyqtSignal()
    clear_history_requested = pyqtSignal()
    erase_history_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    hide_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    voice_requested = pyqtSignal()
    history_item_selected = pyqtSignal(object)
    ui_state_changed = pyqtSignal()

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(None)
        self.cfg = cfg
        self.monitor_provider: Callable[[], list[Rect]] = lambda: []
        self.collapsed = False
        self.allow_close = False  # True только при выходе из программы
        self._expanded_height = 600
        self._answer_header = ""
        self._pending_markdown: str | None = None

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(make_app_icon())
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMinimumWidth(320)

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(60)
        self._render_timer.timeout.connect(self._flush_markdown)

        self._build_ui()
        self._restore_geometry()
        self.apply_config(cfg)
        self.set_answer_markdown(PLACEHOLDER_ANSWER)
        if cfg.collapsed:
            QTimer.singleShot(0, lambda: self.set_collapsed(True))

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.root = QFrame(self)
        self.root.setObjectName("Root")
        outer.addWidget(self.root)
        root_layout = QVBoxLayout(self.root)
        root_layout.setContentsMargins(8, 4, 8, 6)
        root_layout.setSpacing(4)

        # --- заголовок
        self.title_bar = _TitleBar(self)
        tb = QHBoxLayout(self.title_bar)
        tb.setContentsMargins(0, 0, 0, 0)
        tb.setSpacing(4)
        self.status_dot = QLabel("●")
        self.status_dot.setFixedWidth(14)
        title = QLabel(APP_NAME)
        title.setObjectName("Title")
        self.status_label = QLabel("ожидание")
        self.status_label.setObjectName("Muted")
        self.status_label.setMinimumWidth(40)
        tb.addWidget(self.status_dot)
        tb.addWidget(title)
        tb.addWidget(self.status_label, 1)
        self.btn_settings = self._title_button("⚙", "Настройки")
        self.btn_theme = self._title_button("◐", "Сменить тему (тёмная/светлая)")
        self.btn_collapse = self._title_button("▁", "Свернуть в полоску (двойной щелчок по заголовку)")
        self.btn_hide = self._title_button("✕", "Скрыть в системный трей (Ctrl+Alt+H — показать)")
        for b in (self.btn_settings, self.btn_theme, self.btn_collapse, self.btn_hide):
            tb.addWidget(b)
        self.btn_settings.clicked.connect(self.settings_requested)
        self.btn_theme.clicked.connect(self.theme_toggle_requested)
        self.btn_collapse.clicked.connect(lambda: self.set_collapsed(not self.collapsed))
        self.btn_hide.clicked.connect(self.hide_requested)
        root_layout.addWidget(self.title_bar)

        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setVisible(False)
        root_layout.addWidget(self.busy_bar)

        # --- тело
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(5)
        root_layout.addWidget(self.body, 1)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.btn_capture = QPushButton("📸 Снимок")
        self.btn_capture.setToolTip("Сделать скриншот сейчас и распознать текст (без запроса к ИИ)")
        self.btn_capture.clicked.connect(self.capture_now_requested)
        self.btn_region = QToolButton()
        self.btn_region.setText("🔲 Область ▾")
        self.btn_region.setToolTip("Весь экран, активное окно или область, выделенная мышью")
        self.btn_region.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.region_menu = QMenu(self)
        self.region_menu.aboutToShow.connect(self._rebuild_region_menu)
        self.btn_region.setMenu(self.region_menu)
        self.chk_auto = QCheckBox("Авто")
        self.chk_auto.setToolTip("Автоматический анализ экрана с заданной частотой")
        self.chk_auto.toggled.connect(self.auto_toggled)
        self.btn_stop = QPushButton("■")
        self.btn_stop.setObjectName("Danger")
        self.btn_stop.setFixedWidth(30)
        self.btn_stop.setToolTip("Полностью остановить анализ (Ctrl+Alt+X)")
        self.btn_stop.clicked.connect(self.stop_requested)
        row.addWidget(self.btn_capture)
        row.addWidget(self.btn_region)
        row.addStretch(1)
        row.addWidget(self.chk_auto)
        row.addWidget(self.btn_stop)
        body.addLayout(row)

        self.target_label = QLabel("")
        self.target_label.setObjectName("Muted")
        body.addWidget(self.target_label)

        # --- параметры (сворачиваемые)
        self.sec_options = self._section_button("Параметры")
        body.addWidget(self.sec_options)
        self.options_panel = QWidget()
        op = QVBoxLayout(self.options_panel)
        op.setContentsMargins(0, 0, 0, 0)
        op.setSpacing(3)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Частота:"))
        self.slider_interval = QSlider(Qt.Orientation.Horizontal)
        self.slider_interval.setRange(0, len(ANALYSIS_INTERVALS) - 1)
        self.slider_interval.setPageStep(1)
        self.slider_interval.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.lbl_interval = QLabel()
        self.lbl_interval.setFixedWidth(40)
        self.slider_interval.valueChanged.connect(self._on_interval_slider)
        r1.addWidget(self.slider_interval, 1)
        r1.addWidget(self.lbl_interval)
        op.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Прозрачн.:"))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(10, 100)
        self.lbl_opacity = QLabel()
        self.lbl_opacity.setFixedWidth(40)
        self.slider_opacity.valueChanged.connect(self._on_opacity_slider)
        r2.addWidget(self.slider_opacity, 1)
        r2.addWidget(self.lbl_opacity)
        op.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Язык ответа:"))
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(list(ANSWER_LANGUAGES))
        self.combo_lang.currentTextChanged.connect(self.language_changed)
        r3.addWidget(self.combo_lang, 1)
        op.addLayout(r3)
        body.addWidget(self.options_panel)
        self.sec_options.toggled.connect(lambda on: self._toggle_section("show_options_panel", self.sec_options,
                                                                         self.options_panel, "Параметры", on))

        # --- подсказка автоанализа
        self.hint_label = QLabel("Подсказки автоанализа появятся здесь")
        self.hint_label.setObjectName("Hint")
        self.hint_label.setWordWrap(True)
        body.addWidget(self.hint_label)

        # --- ответ
        self.answer_view = QTextBrowser()
        self.answer_view.setOpenLinks(False)  # не открываем браузер
        self.answer_view.setMinimumHeight(120)
        body.addWidget(self.answer_view, 3)
        ar = QHBoxLayout()
        ar.setSpacing(4)
        self.answer_meta = QLabel("")
        self.answer_meta.setObjectName("Muted")
        self.btn_copy = QPushButton("Копировать")
        self.btn_copy.setToolTip("Скопировать ответ в буфер обмена")
        self.btn_copy.clicked.connect(self._copy_answer)
        ar.addWidget(self.answer_meta, 1)
        ar.addWidget(self.btn_copy)
        body.addLayout(ar)

        # --- распознанный текст
        self.sec_ocr = self._section_button("Распознанный текст")
        body.addWidget(self.sec_ocr)
        self.ocr_view = QPlainTextEdit()
        self.ocr_view.setReadOnly(True)
        self.ocr_view.setPlaceholderText("Здесь появится текст, распознанный на экране")
        self.ocr_view.setMinimumHeight(70)
        body.addWidget(self.ocr_view, 1)
        self.sec_ocr.toggled.connect(lambda on: self._toggle_section("show_ocr_panel", self.sec_ocr, self.ocr_view,
                                                                     "Распознанный текст", on))

        # --- история
        self.sec_history = self._section_button("История ответов")
        body.addWidget(self.sec_history)
        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(60)
        self.history_list.itemClicked.connect(
            lambda item: self.history_item_selected.emit(item.data(Qt.ItemDataRole.UserRole)))
        body.addWidget(self.history_list, 1)
        self.sec_history.toggled.connect(lambda on: self._toggle_section(
            "show_history_panel", self.sec_history, self.history_list, "История ответов", on))

        # --- ввод вопроса
        ir = QHBoxLayout()
        ir.setSpacing(4)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Вопрос по экрану… (Enter — спросить)")
        self.input.setClearButtonEnabled(True)
        self.input.returnPressed.connect(self._submit)
        self.btn_voice = QPushButton("🎤")
        self.btn_voice.setFixedWidth(34)
        self.btn_voice.setCheckable(True)
        self.btn_voice.setToolTip("Голосовой вопрос: нажмите, говорите, нажмите ещё раз")
        self.btn_voice.clicked.connect(lambda: self.voice_requested.emit())
        self.btn_ask = QPushButton("Спросить")
        self.btn_ask.setObjectName("Primary")
        self.btn_ask.setToolTip("Сделать снимок и задать вопрос (Ctrl+Alt+S)")
        self.btn_ask.clicked.connect(self._submit)
        ir.addWidget(self.input, 1)
        ir.addWidget(self.btn_voice)
        ir.addWidget(self.btn_ask)
        body.addLayout(ir)

        # --- нижняя панель
        br = QHBoxLayout()
        br.setSpacing(4)
        self.btn_clear = QPushButton("Очистить историю")
        self.btn_clear.setToolTip("Очистить окно и контекст диалога (база не затрагивается)")
        self.btn_clear.clicked.connect(self.clear_history_requested)
        self.btn_erase = QPushButton("Стереть историю")
        self.btn_erase.setObjectName("Danger")
        self.btn_erase.setToolTip("Безвозвратно удалить историю из локальной базы SQLite")
        self.btn_erase.clicked.connect(self.erase_history_requested)
        br.addWidget(self.btn_clear)
        br.addWidget(self.btn_erase)
        br.addStretch(1)
        br.addWidget(QSizeGrip(self), 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        body.addLayout(br)

    def _title_button(self, text: str, tip: str) -> QToolButton:
        b = QToolButton()
        b.setObjectName("TitleButton")
        b.setText(text)
        b.setToolTip(tip)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        return b

    def _section_button(self, text: str) -> QToolButton:
        b = QToolButton()
        b.setObjectName("Section")
        b.setCheckable(True)
        b.setText(f"▸ {text}")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        return b

    def _toggle_section(self, cfg_field: str, button: QToolButton, panel: QWidget, title: str, on: bool) -> None:
        button.setText(f"{'▾' if on else '▸'} {title}")
        panel.setVisible(on)
        setattr(self.cfg, cfg_field, on)
        self.ui_state_changed.emit()

    # --------------------------------------------------------- конфигурация
    def apply_config(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        widgets = (self.slider_interval, self.slider_opacity, self.combo_lang, self.chk_auto,
                   self.sec_options, self.sec_ocr, self.sec_history)
        for w in widgets:
            w.blockSignals(True)
        try:
            idx = ANALYSIS_INTERVALS.index(cfg.analysis_interval) if cfg.analysis_interval in ANALYSIS_INTERVALS else 1
            self.slider_interval.setValue(idx)
            self.lbl_interval.setText(f"{ANALYSIS_INTERVALS[idx]} с")
            self.slider_opacity.setValue(cfg.opacity)
            self.lbl_opacity.setText(f"{cfg.opacity}%")
            self.combo_lang.setCurrentText(cfg.answer_language)
            self.chk_auto.setChecked(cfg.auto_analysis)
            self.sec_options.setChecked(cfg.show_options_panel)
            self.sec_ocr.setChecked(cfg.show_ocr_panel)
            self.sec_history.setChecked(cfg.show_history_panel)
        finally:
            for w in widgets:
                w.blockSignals(False)
        for btn, panel, title, on in ((self.sec_options, self.options_panel, "Параметры", cfg.show_options_panel),
                                      (self.sec_ocr, self.ocr_view, "Распознанный текст", cfg.show_ocr_panel),
                                      (self.sec_history, self.history_list, "История ответов",
                                       cfg.show_history_panel)):
            btn.setText(f"{'▾' if on else '▸'} {title}")
            panel.setVisible(on)
        self.setStyleSheet(build_stylesheet(cfg.theme, cfg.font_size))
        self.apply_opacity()

    def apply_opacity(self) -> None:
        self.setWindowOpacity(max(0.1, min(1.0, self.cfg.opacity / 100.0)))

    def set_auto_checked(self, on: bool) -> None:
        self.chk_auto.blockSignals(True)
        self.chk_auto.setChecked(on)
        self.chk_auto.blockSignals(False)

    def _on_interval_slider(self, idx: int) -> None:
        seconds = ANALYSIS_INTERVALS[max(0, min(idx, len(ANALYSIS_INTERVALS) - 1))]
        self.lbl_interval.setText(f"{seconds} с")
        self.interval_changed.emit(seconds)

    def _on_opacity_slider(self, value: int) -> None:
        self.lbl_opacity.setText(f"{value}%")
        self.cfg.opacity = value
        self.apply_opacity()
        self.opacity_changed.emit(value)

    def _rebuild_region_menu(self) -> None:
        menu = self.region_menu
        menu.clear()
        group = QActionGroup(menu)
        group.setExclusive(True)
        try:
            monitors = self.monitor_provider() or []
        except Exception:
            monitors = []
        mode, idx = self.cfg.capture_mode, self.cfg.monitor_index
        if len(monitors) > 2:
            m = monitors[0]
            act = menu.addAction(f"Все мониторы — {m}")
            act.setCheckable(True)
            act.setChecked(mode == "monitor" and idx == 0)
            act.triggered.connect(lambda: self.capture_mode_selected.emit("monitor", 0))
            group.addAction(act)
        for i, m in enumerate(monitors[1:], start=1):
            act = menu.addAction(f"Весь экран: монитор {i} — {m}")
            act.setCheckable(True)
            act.setChecked(mode == "monitor" and idx == i)
            act.triggered.connect(lambda _=False, n=i: self.capture_mode_selected.emit("monitor", n))
            group.addAction(act)
        act = menu.addAction("Активное окно" + ("" if IS_WINDOWS else " (только Windows)"))
        act.setCheckable(True)
        act.setChecked(mode == "window")
        act.setEnabled(IS_WINDOWS)
        act.triggered.connect(lambda: self.capture_mode_selected.emit("window", 0))
        group.addAction(act)
        if len(self.cfg.region) == 4:
            act = menu.addAction(f"Выделенная область — {self.cfg.region[2]}×{self.cfg.region[3]}")
            act.setCheckable(True)
            act.setChecked(mode == "region")
            act.triggered.connect(lambda: self.capture_mode_selected.emit("region", 0))
            group.addAction(act)
        menu.addSeparator()
        menu.addAction("✂ Выделить область мышью…").triggered.connect(self.select_region_requested)

    # ---------------------------------------------------------- состояние
    def set_status(self, state: str, detail: str = "") -> None:
        color = STATUS_COLORS.get(state, STATUS_COLORS["idle"])
        self.status_dot.setStyleSheet(f"color: {color};")
        text = STATUS_TEXT.get(state, state)
        if detail:
            short = detail if len(detail) <= 48 else detail[:47] + "…"
            text = f"{text}: {short}" if state in ("error", "nochange", "ready", "stopped") else short
        self.status_label.setText(text)
        self.status_label.setToolTip(detail or text)

    def set_busy(self, busy: bool) -> None:
        self.busy_bar.setVisible(busy)

    def set_target_label(self, text: str) -> None:
        self.target_label.setText(f"Цель: {text}")

    def set_hint(self, text: str) -> None:
        self.hint_label.setText(text)
        self.hint_label.setToolTip(text)

    def set_ocr_text(self, text: str, info: str = "") -> None:
        self.ocr_view.setPlainText(text)
        title = "Распознанный текст" + (f" · {info}" if info else "")
        on = self.sec_ocr.isChecked()
        self.sec_ocr.setText(f"{'▾' if on else '▸'} {title}")

    # -------------------------------------------------------------- ответ
    def show_question(self, question: str) -> None:
        q = question.replace("\n", " ").strip()
        self._answer_header = f"> **❓ {q}**\n\n"
        self.answer_meta.setText("")
        self.set_answer_markdown(self._answer_header + "_Анализирую экран…_")

    def set_answer_stream(self, text: str) -> None:
        self._pending_markdown = self._answer_header + text
        if not self._render_timer.isActive():
            self._render_timer.start()

    def set_answer(self, question: str, answer: str, meta: str = "") -> None:
        q = question.replace("\n", " ").strip()
        self._answer_header = f"> **❓ {q}**\n\n" if q else ""
        self._render_timer.stop()
        self._pending_markdown = None
        self.set_answer_markdown(self._answer_header + answer)
        self.answer_meta.setText(meta)

    def set_answer_error(self, message: str) -> None:
        self._render_timer.stop()
        self._pending_markdown = None
        self.set_answer_markdown(self._answer_header + f"**⚠ Ошибка:** {message}")

    def set_answer_markdown(self, markdown: str) -> None:
        bar = self.answer_view.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        self.answer_view.setMarkdown(markdown)
        if at_bottom:
            self.answer_view.moveCursor(QTextCursor.MoveOperation.End)
            bar.setValue(bar.maximum())

    def _flush_markdown(self) -> None:
        if self._pending_markdown is not None:
            self.set_answer_markdown(self._pending_markdown)
            self._pending_markdown = None

    def _copy_answer(self) -> None:
        text = self.answer_view.toPlainText().strip()
        if text:
            QApplication.clipboard().setText(text)
            self.answer_meta.setText("Скопировано")

    def clear_answer(self) -> None:
        self._answer_header = ""
        self.answer_meta.setText("")
        self.set_answer_markdown(PLACEHOLDER_ANSWER)

    # ------------------------------------------------------------ история
    def _make_item(self, entry: HistoryEntry) -> QListWidgetItem:
        q = entry.question.replace("\n", " ")
        item = QListWidgetItem(f"{entry.time_label}  {q[:70]}")
        item.setToolTip(entry.answer[:500])
        item.setData(Qt.ItemDataRole.UserRole, entry)
        return item

    def set_history(self, entries: list[HistoryEntry]) -> None:
        self.history_list.clear()
        for entry in entries:
            self.history_list.addItem(self._make_item(entry))

    def add_history_item(self, entry: HistoryEntry) -> None:
        self.history_list.insertItem(0, self._make_item(entry))
        while self.history_list.count() > 50:
            self.history_list.takeItem(self.history_list.count() - 1)

    # --------------------------------------------------------------- ввод
    def _submit(self) -> None:
        text = self.input.text().strip()
        self.input.clear()
        self.ask_requested.emit(text)

    def current_question(self) -> str:
        return self.input.text().strip()

    def take_question(self) -> str:
        text = self.input.text().strip()
        self.input.clear()
        return text

    def set_question_text(self, text: str) -> None:
        self.input.setText(text)

    def set_voice_recording(self, on: bool) -> None:
        self.btn_voice.setChecked(on)
        self.btn_voice.setText("⏺" if on else "🎤")

    def focus_question(self) -> None:
        if self.collapsed:
            self.set_collapsed(False)
        self.show()
        self.raise_()
        self.activateWindow()
        self.input.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.input.selectAll()

    # ------------------------------------------- свернуть / геометрия окна
    def set_collapsed(self, collapsed: bool) -> None:
        if collapsed == self.collapsed:
            return
        if collapsed:
            self._expanded_height = max(self.height(), 300)
            self.body.hide()
            self.busy_bar.setVisible(False)
            self.setMinimumHeight(0)
            self.layout().activate()
            self.resize(self.width(), self.title_bar.sizeHint().height() + 14)
            self.btn_collapse.setText("▢")
            self.btn_collapse.setToolTip("Развернуть")
        else:
            self.body.show()
            self.layout().activate()
            self.resize(self.width(), self._expanded_height)
            self.btn_collapse.setText("▁")
            self.btn_collapse.setToolTip("Свернуть в полоску")
        self.collapsed = collapsed
        self.cfg.collapsed = collapsed
        self.ui_state_changed.emit()

    def _restore_geometry(self) -> None:
        g = self.cfg.window_geometry
        if len(g) == 4 and QGuiApplication.screenAt(QPoint(g[0] + 30, g[1] + 10)) is not None:
            self.setGeometry(g[0], g[1], max(320, g[2]), max(200, g[3]))
            self._expanded_height = max(300, g[3])
            return
        screen = QGuiApplication.primaryScreen()
        w, h = 420, 600
        if screen is not None:
            area = screen.availableGeometry()
            h = min(h, area.height() - 40)
            self.setGeometry(area.right() - w - 20, area.top() + 20, w, h)
        else:
            self.resize(w, h)

    def store_geometry(self) -> None:
        g = self.geometry()
        height = self._expanded_height if self.collapsed else g.height()
        self.cfg.window_geometry = [g.x(), g.y(), g.width(), height]

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        self.store_geometry()
        self.ui_state_changed.emit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.store_geometry()
        self.ui_state_changed.emit()

    def physical_rect(self) -> Rect:
        """Где окно находится в физических пикселях (для скрытия на время снимка)."""
        if IS_WINDOWS:
            r = get_window_rect(int(self.winId()), use_dwm=False)
            if r is not None:
                return r
        g = self.frameGeometry()
        dpr = self.devicePixelRatioF() or 1.0
        return Rect(int(g.x() * dpr), int(g.y() * dpr), int(g.width() * dpr), int(g.height() * dpr))

    def closeEvent(self, event) -> None:
        if self.allow_close:
            event.accept()
            return
        # Alt+F4 и т. п. — не закрываем программу, а прячем в трей
        event.ignore()
        self.hide_requested.emit()

"""Screen Answer — помощник, который смотрит на экран и отвечает на вопросы.

Запуск:  python main.py
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import APP_NAME, APP_VERSION, LOG_PATH, AppConfig  # noqa: E402


def setup_logging() -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)] if sys.stderr else []
    try:
        handlers.append(RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    setup_logging()
    log = logging.getLogger("main")
    log.info("%s %s запускается", APP_NAME, APP_VERSION)

    try:
        from PyQt6.QtCore import QDir, QLockFile, Qt
        from PyQt6.QtGui import QGuiApplication
        from PyQt6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        print("PyQt6 не установлен. Выполните: pip install -r requirements.txt")
        return 1

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setQuitOnLastWindowClosed(False)  # окно прячется в трей, программа продолжает работать

    lock = QLockFile(os.path.join(QDir.tempPath(), "screen_answer.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(100):
        QMessageBox.information(None, APP_NAME, "Screen Answer уже запущен — найдите значок в системном трее.")
        return 0

    try:
        from core.controller import AppController

        cfg = AppConfig.load()
        controller = AppController(app, cfg)
    except Exception as exc:
        log.exception("Ошибка запуска")
        QMessageBox.critical(None, APP_NAME, f"Не удалось запустить программу:\n{exc}\n\nПодробности: {LOG_PATH}")
        return 1

    def excepthook(exc_type, exc_value, exc_tb) -> None:
        # Любая непойманная ошибка логируется и показывается в окне, а не роняет программу.
        log.error("Необработанная ошибка:\n%s", "".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        try:
            controller.error_reported.emit(f"Внутренняя ошибка: {exc_value}")  # сигнал безопасен из любого потока
        except Exception:
            pass

    sys.excepthook = excepthook
    threading.excepthook = lambda args: excepthook(args.exc_type, args.exc_value, args.exc_traceback)

    app.aboutToQuit.connect(controller.shutdown)
    controller.start()
    code = app.exec()
    lock.unlock()
    log.info("Завершение работы (код %s)", code)
    return code


if __name__ == "__main__":
    sys.exit(main())

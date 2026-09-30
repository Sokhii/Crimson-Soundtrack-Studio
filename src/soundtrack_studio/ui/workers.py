"""Runs service calls off the GUI thread with progress and cancellation."""

from __future__ import annotations

import threading
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    progress = Signal(str, int, int)
    done = Signal(object)
    error = Signal(object)


class Job(QRunnable):
    """``fn(report, cancelled)`` runs in the pool; ``report(text, done, total)`` updates the UI."""

    def __init__(self, fn: Callable[[Callable[[str, int, int], None], Callable[[], bool]], Any]) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.fn = fn
        self.signals = _Signals()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def report(self, text: str, done: int = 0, total: int = 0) -> None:
        self.signals.progress.emit(text, int(done), int(total))

    def run(self) -> None:
        try:
            result = self.fn(self.report, self.cancelled)
        except BaseException as exc:  # delivered to the GUI thread, which shows a friendly message
            self.signals.error.emit(exc)
        else:
            self.signals.done.emit(result)


def start(job: Job) -> Job:
    QThreadPool.globalInstance().start(job)
    return job

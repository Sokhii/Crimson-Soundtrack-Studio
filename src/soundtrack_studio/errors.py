"""User-facing error types.

Every error that can reach the user carries a short title, a plain-language
message, an optional hint on what to do next, and technical details that go
to the log (and an expandable section in the error dialog), never only a
stack trace.
"""

from __future__ import annotations


class StudioError(Exception):
    """Base class for errors that should be shown to the user."""

    title = "Something went wrong"

    def __init__(self, message: str, *, title: str | None = None, hint: str = "", details: str = "") -> None:
        super().__init__(message)
        self.message = message
        if title:
            self.title = title
        self.hint = hint
        self.details = details

    def user_text(self) -> str:
        text = self.message
        if self.hint:
            text += "\n\n" + self.hint
        return text


class AnalyzerDbError(StudioError):
    title = "Analyzer database problem"


class ProjectError(StudioError):
    title = "Project problem"


class GameInstallError(StudioError):
    title = "Crimson Desert installation problem"


class LibraryError(StudioError):
    title = "Music library problem"


class OperationCancelled(Exception):
    """Raised inside workers when the user cancels a long-running operation."""

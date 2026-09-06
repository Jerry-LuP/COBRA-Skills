from __future__ import annotations

import subprocess
from typing import Iterator, NoReturn


class EvaluationInfrastructureError(RuntimeError):
    """A runtime/setup failure that must not be recorded as task reward zero."""


_TIMEOUT_EXCEPTION_NAMES = {
    "APITimeoutError",
    "ConnectTimeout",
    "ReadTimeout",
    "TimeoutException",
}
_TIMEOUT_TEXT_MARKERS = (
    "timed out",
    "timeout after",
    "deadline exceeded",
    "exit code 124",
    "task-timeout",
    "task timeout",
    "request timeout",
    "read timeout",
    "connect timeout",
    "operation timeout",
)


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_timeout_error(exc: BaseException) -> bool:
    """Return true only when an exception chain represents an explicit timeout."""
    for current in _exception_chain(exc):
        if isinstance(current, (TimeoutError, subprocess.TimeoutExpired)):
            return True
        if type(current).__name__ in _TIMEOUT_EXCEPTION_NAMES:
            return True
        detail = f"{type(current).__name__}: {current}".lower()
        if any(marker in detail for marker in _TIMEOUT_TEXT_MARKERS):
            return True
    return False


def is_valid_result_row(row: dict) -> bool:
    """Return whether a persisted row represents a scoreable model outcome."""
    phase = str(row.get("phase") or "").strip().lower()
    if phase == "timeout" or bool(row.get("task_timed_out", False)):
        return True

    reason = str(row.get("fail_reason") or "").strip().lower()
    if phase == "error" or reason.startswith(
        ("unexpected:", "agent-error:", "llm-call-failed:")
    ):
        return False

    if row.get("agent_ok") is False:
        terminal_reason = str(row.get("terminal_failure_reason") or "").lower()
        return bool(row.get("accepted_zero", False)) and "trajectory retries" in terminal_reason
    return True


def raise_evaluation_error(
    dataset: str,
    item_id: object,
    exc: BaseException,
) -> NoReturn:
    """Raise a contextual fatal error while retaining the original exception chain."""
    if isinstance(exc, EvaluationInfrastructureError):
        raise exc
    raise EvaluationInfrastructureError(
        f"{dataset} evaluation failed for item {item_id}: "
        f"{type(exc).__name__}: {exc}. No reward was recorded."
    ) from exc

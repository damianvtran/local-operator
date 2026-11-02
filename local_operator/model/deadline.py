"""One foreground metadata budget across executor queue, cache and HTTP legs.

The deadline is captured before executor submission and installed explicitly
inside that worker. Ordinary discovery callers and background revalidations
keep their existing independent budgets; no process-global timeout is changed.
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable, TypeVar

_active: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "model_resolution_deadline", default=None
)
_T = TypeVar("_T")


def current() -> float | None:
    return _active.get()


def remaining(timeout: float) -> float:
    deadline = _active.get()
    return timeout if deadline is None else min(timeout, max(0.0, deadline - time.monotonic()))


def check() -> None:
    if remaining(float("inf")) <= 0:
        raise TimeoutError


@contextmanager
def scope(deadline: float) -> Iterator[None]:
    previous = current()
    token = _active.set(deadline if previous is None else min(previous, deadline))
    try:
        yield
    finally:
        _active.reset(token)


def run_before(deadline: float, call: Callable[..., _T], *args: object) -> _T:
    with scope(deadline):
        check()
        return call(*args)

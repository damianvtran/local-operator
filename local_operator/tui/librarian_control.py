"""The optional bridge between ``/librarian`` and the Librarian package.

WHY A SEPARATE MODULE. The Librarian work package OWNS
:mod:`local_operator.librarian`, and this branch carries the Library backend
without it. The TUI still has to answer the command — "not available in this
build" is a designed state, not an import error at module load — and the app
must not hard-code a private API the other package has not agreed to. So the
contract this module calls is explicit, small and documented here:

* ``local_operator.librarian.status() -> Mapping | str`` — the current state
  (``paused``, ``last_run``, counts); rendered by the caller.
* ``pause()`` / ``resume()`` — flip the cadence gate.
* ``run_once()`` — start one organizing pass.
* ``ensure_session() -> str | None`` — the pinned conversation (an awaitable
  is accepted), created on first use exactly as ``aida.ensure_session`` is.

All four are OPTIONAL. A package that ships only some of them still lands the
other commands, and a missing hook is reported by name rather than treated as
an outage. Import is lazy on every call: the module is absent on this branch
and present on the merged one, and a cached import result would be a process
that can never notice it arriving.

NO WRITES AND NO CONFIG PARSING HERE. ``pause``/``resume`` are the module's
own verbs; this bridge never edits a config key or a wake row itself. That
keeps one writer for the Librarian's state, which is the same rule the Library
page follows for asset metadata.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Mapping

MODULE_NAME = "local_operator.librarian"

#: The page's and the command's vocabulary, in the order ``/librarian`` lists
#: it. ``open`` is handled by the app (it is a session transition), the rest
#: map one-to-one onto hooks above.
VERBS = ("status", "pause", "resume", "run", "open")

_HOOKS = {
    "status": "status",
    "pause": "pause",
    "resume": "resume",
    "run": "run_once",
}


@dataclass(frozen=True)
class LibrarianResult:
    """What one hook call produced, in terms the caller can translate.

    ``available`` is False when the package (or the hook) is absent; that is a
    build fact, not an error. ``error`` carries the module's own exception
    text. ``state`` is the normalized state word for a receipt — the hook's
    mapping is not quoted raw because a footer built from ``repr(dict)`` is
    not a sentence a reader can use.
    """

    available: bool
    hook: str = ""
    state: str = ""
    error: str = ""
    #: Set when the package exists but does not publish this hook yet: the
    #: receipt names the missing function instead of claiming the whole
    #: Librarian is absent.
    missing_hook: str = ""


def module() -> Any | None:
    """The Librarian package, or ``None`` where this build does not carry it."""
    try:
        return importlib.import_module(MODULE_NAME)
    except ImportError:
        return None
    except Exception:  # noqa: BLE001 — an import-time refusal is still absence
        return None


def _state_word(value: Any) -> str:
    """A short state word from whatever a hook returned.

    A mapping is the documented shape; its ``state``/``status`` wins, then a
    boolean ``paused``, then an ``organized`` count. A plain string is used
    verbatim; ``None`` means the hook did its work and had nothing to report,
    which reads as ``ok`` rather than as an empty receipt.
    """
    if isinstance(value, Mapping):
        for key in ("state", "status"):
            word = value.get(key)
            if isinstance(word, str) and word:
                return word
        if value.get("paused") is True:
            return "paused"
        if value.get("paused") is False:
            return "running"
        organized = value.get("organized")
        if isinstance(organized, int):
            return str(organized)
        return "ok"
    if isinstance(value, str) and value:
        return value
    return "ok"


def call(verb: str) -> LibrarianResult:
    """Run one control verb; never raises.

    ``status`` is the bare command's default. An unknown verb is a caller bug,
    reported as an unavailable hook rather than silently mapped to status.
    """
    hook_name = _HOOKS.get(verb)
    if hook_name is None:
        return LibrarianResult(available=False, hook=verb)
    package = module()
    if package is None:
        return LibrarianResult(available=False, hook=hook_name)
    hook = getattr(package, hook_name, None)
    if not callable(hook):
        return LibrarianResult(available=True, hook=hook_name, missing_hook=hook_name)
    try:
        answer = hook()
        if inspect.isawaitable(answer):
            # The TUI calls this from a worker; a coroutine hook is awaited by
            # the module's own loop only if we run it. `call` is synchronous by
            # contract and the app uses `call_async` for those hooks.
            if inspect.iscoroutine(answer):
                answer.close()
            return LibrarianResult(
                available=False, hook=hook_name, error="async hook requires call_async"
            )
    except Exception as error:  # noqa: BLE001 — the command reports the refusal
        return LibrarianResult(available=True, hook=hook_name, error=str(error))
    return LibrarianResult(available=True, hook=hook_name, state=_state_word(answer))


async def call_async(verb: str) -> LibrarianResult:
    """``call`` for the TUI: a blocking hook runs off the event loop.

    A coroutine hook is awaited on the loop (it owns its own awaits); a plain
    function is hopped to a worker thread, because ``run_once`` is an
    organizing pass and the others touch disk. The branch is on the CALLABLE,
    not on what it returns, so a coroutine function is never called twice.
    """
    hook_name = _HOOKS.get(verb)
    if hook_name is None:
        return LibrarianResult(available=False, hook=verb)
    package = module()
    if package is None:
        return LibrarianResult(available=False, hook=hook_name)
    hook = getattr(package, hook_name, None)
    if not callable(hook):
        return LibrarianResult(available=True, hook=hook_name, missing_hook=hook_name)
    try:
        if inspect.iscoroutinefunction(hook):
            answer = await hook()
        else:
            answer = await asyncio.to_thread(hook)
            if inspect.isawaitable(answer):
                answer = await answer
    except Exception as error:  # noqa: BLE001 — the command reports the refusal
        return LibrarianResult(available=True, hook=hook_name, error=str(error))
    return LibrarianResult(available=True, hook=hook_name, state=_state_word(answer))


async def ensure_session() -> tuple[str | None, str]:
    """The pinned conversation id, or ``(None, reason)``.

    ``reason`` is empty on success. A hook that is absent, refuses, or answers
    something that is not a session id is reported here; the app turns it into
    the one "not available in this build" sentence or the hook's own error.
    """
    package = module()
    if package is None:
        return None, ""
    hook = getattr(package, "ensure_session", None)
    if not callable(hook):
        return None, "ensure_session"
    try:
        if inspect.iscoroutinefunction(hook):
            answer = await hook()
        else:
            answer = await asyncio.to_thread(hook)
            if inspect.isawaitable(answer):
                answer = await answer
    except Exception as error:  # noqa: BLE001 — the app reports the refusal
        return None, str(error)
    if isinstance(answer, str) and answer:
        return answer, ""
    return None, "ensure_session"

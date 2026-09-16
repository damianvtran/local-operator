"""The saved-preview directory must be the PREVIEWED conversation's own.

``_lease_sidebar_source`` hands ``AttachedSession.saved_preview`` the working
directory of the conversation it is about to preview, and that conversation is
a DIFFERENT session id from the one this terminal is attached to. The preview
band renders the value as the previewed session's identity field, and the
connect leg binds a runtime in it, so a plausible-looking value that belongs to
another conversation is a claim about someone else's project.

Three wrong answers preceded this one, each fixed by the tests below:

* ``getattr(self._session, "cwd", "")`` — ``cwd`` exists on neither session
  class, so the read was ALWAYS ``""`` and a conversation with no recorded
  directory opened at the process default.
* this session's ``frontend_state.cwd`` — a real directory, but the WRONG one:
  a preview of conversation A advertised the directory of conversation B (the
  one on screen), and nothing ever restored the honest value because
  ``read_saved_preview``'s ``cwd`` half scans for a ``session`` transcript entry
  that no writer in the tree emits (design round 1, D1; UX round 1, U3).
* the fix here — the previewed session's own record, then its wake-index entry,
  and ``""`` as the honest last resort.

Resolution is exercised through the app's real seam (``_lease_sidebar_source``
into ``_preview_cwd``), not by calling the resolver on a private path: the point
of D1 was that the value reaching ``saved_preview`` was wrong, so that is what
the assertions read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.mobile import attach_client
from local_operator.session.attached import AttachedSession
from local_operator.tui.app import OperatorApp
from local_operator.wakes import store as wake_store


class _StopLease(Exception):
    """Stop ``_lease_sidebar_source`` at the call under test.

    Raised from the ``saved_preview`` spy, which runs BEFORE the method's own
    ``try``/``except BaseException`` — so nothing downstream of the call under
    test has to be stood up for the assertion to be about the real code path.
    """


class _Record:
    """The two fields ``find_runtime_record``'s caller reads off a record."""

    def __init__(self, session_id: str, cwd: str) -> None:
        self.session_id = session_id
        self.cwd = cwd


async def _no_takeover() -> Any:
    raise RuntimeError("a sidebar viewer never takes over a session")


def _app_over(session: Any) -> OperatorApp:
    """An app holding ``session``, constructed without ``run_test``.

    ``_lease_sidebar_source`` touches only ``_sidebar_sources`` and ``_session``
    before it calls ``saved_preview``, and both are set in ``__init__``, so the
    pilot would add a mounted widget tree without changing what is exercised.
    """

    async def factory() -> Any:
        return session

    app = OperatorApp(factory)
    app._session = session
    return app


def _record_lookup(monkeypatch: pytest.MonkeyPatch, record: _Record | None) -> None:
    """Make ``find_runtime_record`` answer with ``record`` (or no owner at all).

    ``_preview_cwd`` imports the name inside the function, so patching it on the
    module is what the call site resolves; the call itself goes through
    ``asyncio.to_thread``, which is why the stub is an ordinary function.
    """
    monkeypatch.setattr(
        attach_client,
        "find_runtime_record",
        lambda _config_dir, _session_id: (record, 4321 if record is not None else None),
    )


def _wake_entry(monkeypatch: pytest.MonkeyPatch, entry: dict[str, Any] | None) -> None:
    monkeypatch.setattr(wake_store, "read_entry", lambda _config_dir, _session_id: entry)


async def _cwd_handed_to_saved_preview(
    app: OperatorApp, monkeypatch: pytest.MonkeyPatch, session_id: str
) -> str:
    """Run the real seam and return the ``cwd`` the launcher was given."""

    async def spy(cls: Any, session_id: str, *, config_dir: Path, cwd: str, takeover_factory: Any):
        seen["cwd"] = cwd
        raise _StopLease

    seen: dict[str, Any] = {}
    monkeypatch.setattr(AttachedSession, "saved_preview", classmethod(spy))
    with pytest.raises(_StopLease):
        await app._lease_sidebar_source(session_id, speculative=False)
    return seen["cwd"]


@pytest.mark.asyncio
async def test_preview_cwd_is_the_previewed_sessions_record_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live runtime's own record is the first source, and it is exact."""
    _record_lookup(monkeypatch, _Record("previewed-session", "/work/previewed-project"))
    _wake_entry(monkeypatch, {"cwd": "/work/stale-index-entry"})

    app = _app_over(_Owner())
    cwd = await _cwd_handed_to_saved_preview(app, monkeypatch, "previewed-session")

    # The record wins over the index: a running session's record is current,
    # the wake index is a projection that need not be.
    assert cwd == "/work/previewed-project"


@pytest.mark.asyncio
async def test_preview_cwd_falls_back_to_the_previewed_sessions_wake_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session nobody has open has no record, but may still have a wake entry.

    The same ladder ``resume_click._session_cwd`` uses, and the reason the
    resolver is not simply ``find_runtime_record``: a saved conversation that
    carries wakes keeps a ``cwd`` in the index.
    """
    _record_lookup(monkeypatch, None)
    _wake_entry(monkeypatch, {"cwd": "/work/cold-project"})

    app = _app_over(_Owner())
    cwd = await _cwd_handed_to_saved_preview(app, monkeypatch, "previewed-session")

    assert cwd == "/work/cold-project"


@pytest.mark.asyncio
async def test_preview_cwd_is_empty_when_nothing_records_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``""`` is the honest last resort: the band's rung simply goes absent.

    Deliberately NOT the process default or the home directory — those read as a
    recorded fact in the band, which is the exact confusion D1 is about.
    """
    _record_lookup(monkeypatch, None)
    _wake_entry(monkeypatch, None)

    app = _app_over(_Owner())
    cwd = await _cwd_handed_to_saved_preview(app, monkeypatch, "previewed-session")

    assert cwd == ""


@pytest.mark.asyncio
async def test_preview_cwd_never_borrows_this_sessions_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression guard for D1: this terminal's own cwd must not leak in.

    The viewing session HAS a real directory and the previewed one has none —
    the state in which the old fallback handed over ``known_cwd``. The answer
    must be the honest ``""``, not the directory of the conversation that
    happens to be on screen.
    """
    known_cwd = tmp_path / "the-session-on-screen"
    known_cwd.mkdir()
    current = await AttachedSession.cold(
        "current-session",
        config_dir=tmp_path,
        cwd=str(known_cwd),
        takeover_factory=_no_takeover,
    )
    _record_lookup(monkeypatch, None)
    _wake_entry(monkeypatch, None)

    app = _app_over(current)
    cwd = await _cwd_handed_to_saved_preview(app, monkeypatch, "previewed-session")

    assert cwd == ""
    assert cwd != str(known_cwd)


@pytest.mark.asyncio
async def test_preview_cwd_is_the_record_directory_for_a_non_viewer_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source does not depend on what KIND of session this host holds.

    A non-viewer host has no attached frontend state at all, which is why the
    old fallback offered it ``""`` unconditionally. The previewed session's own
    record is just as available, so it is still the answer.
    """
    _record_lookup(monkeypatch, _Record("previewed-session", "/work/previewed-project"))
    _wake_entry(monkeypatch, None)

    app = _app_over(_Owner())
    cwd = await _cwd_handed_to_saved_preview(app, monkeypatch, "previewed-session")

    assert cwd == "/work/previewed-project"


class _Owner:
    """A minimal non-viewer session: enough for the seam, no frontend state."""

    owns_runtime = True
    session_id = "owner-session"

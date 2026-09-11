"""The sidebar's saved-preview fallback directory must be the session's REAL one.

``_lease_sidebar_source`` hands ``AttachedSession.saved_preview`` a fallback
``cwd`` for a conversation whose journal records none. It read that value as
``getattr(self._session, "cwd", "")``, and ``cwd`` exists on neither session
class — so the fallback was ALWAYS the empty string, and a conversation with no
recorded directory opened its preview at the process default instead of the
session's own working directory. The read now goes through the declared
``frontend_state`` accessor; these tests pin both halves of that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.session.attached import AttachedSession
from local_operator.tui.app import OperatorApp


class _StopLease(Exception):
    """Stop ``_lease_sidebar_source`` at the call under test.

    Raised from the ``saved_preview`` spy, which runs BEFORE the method's own
    ``try``/``except BaseException`` — so nothing downstream of the call under
    test has to be stood up for the assertion to be about the real code path.
    """


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


@pytest.mark.asyncio
async def test_saved_preview_fallback_cwd_is_the_sessions_real_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    known_cwd = tmp_path / "known-session-dir"
    known_cwd.mkdir()

    current = await AttachedSession.cold(
        "current-session",
        config_dir=tmp_path,
        cwd=str(known_cwd),
        takeover_factory=_no_takeover,
    )
    seen: dict[str, Any] = {}

    async def spy(cls: Any, session_id: str, *, config_dir: Path, cwd: str, takeover_factory: Any):
        seen["session_id"] = session_id
        seen["cwd"] = cwd
        raise _StopLease

    monkeypatch.setattr(AttachedSession, "saved_preview", classmethod(spy))

    app = _app_over(current)
    with pytest.raises(_StopLease):
        await app._lease_sidebar_source("other-session", speculative=False)

    assert seen["session_id"] == "other-session"
    # Non-empty, and specifically the SESSION's directory — the two facts the
    # dead probe could never supply.
    assert seen["cwd"]
    assert seen["cwd"] == str(known_cwd)


@pytest.mark.asyncio
async def test_saved_preview_fallback_cwd_is_empty_for_a_non_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-viewer host has no attached frontend state, so the fallback stands.

    The old probe answered ``""`` for EVERY session, so this half looked right
    for the wrong reason; asserting it separately is what keeps the fix from
    being reverted into "read a member nobody declares" if a non-viewer host
    ever mounts this seam.
    """

    class _Owner:
        owns_runtime = True
        session_id = "owner-session"

    seen: dict[str, Any] = {}

    async def spy(cls: Any, session_id: str, *, config_dir: Path, cwd: str, takeover_factory: Any):
        seen["cwd"] = cwd
        raise _StopLease

    monkeypatch.setattr(AttachedSession, "saved_preview", classmethod(spy))

    app = _app_over(_Owner())
    with pytest.raises(_StopLease):
        await app._lease_sidebar_source("other-session", speculative=False)

    assert seen["cwd"] == ""

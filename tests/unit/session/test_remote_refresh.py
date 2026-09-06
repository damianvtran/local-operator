"""A ``retiring`` owner is a refresh, not a death and not a stop.

The runtime retires itself when ``lop-update`` puts a newer build on disk
and it is idle (design-runtime-autorefresh §3.2). The viewer must read the
announced disconnect as "a fresh runtime is owed" — go cold at once, fire
the refresh callback so the app re-engages, and synthesise NO turn end
(nothing was interrupted; the runtime was idle by contract). Both older
outcomes keep their behaviour: ``STOPPED_REASON`` still parks the viewer in
the stopped state, and a bare owner exit still runs recovery.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import local_operator.session.remote as remote_module
from local_operator.mobile.attach_client import RETIRING_REASON, STOPPED_REASON
from local_operator.session.remote import RemoteSession


async def _never_take_over() -> Any:
    raise AssertionError("a refresh must never take over the session")


@pytest.mark.asyncio
async def test_retiring_goes_cold_now_and_fires_the_refresh_callback(tmp_path, monkeypatch):
    """The whole contract of the frame, on the viewer side.

    No recovery task: chasing the record for 8 s would end in "runtime
    exited" for a planned housekeeping event, and the successor does not
    exist yet — the app spawns it. No ``_end_turn_locally``: that is what
    would paint ``interrupted`` for a turn that never existed.
    """
    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over)
    ended: list[str] = []
    monkeypatch.setattr(
        remote, "_end_turn_locally", lambda *a, **k: ended.append("end"), raising=True
    )
    refreshed: list[str] = []
    cold: list[str] = []
    remote.set_refresh_callback(lambda: refreshed.append("refresh"))
    remote.set_went_cold_callback(lambda: cold.append("cold"))

    remote._on_disconnected(RETIRING_REASON)
    await asyncio.sleep(0)

    assert remote.is_cold
    assert remote._recovery_task is None, "a refresh has nothing to recover"
    assert refreshed == ["refresh"]
    assert cold == [], "the went-cold callback is the death path; a refresh is not one"
    assert ended == [], "no turn end may be synthesised for an idle runtime that retired"
    assert remote._deliberate_stop is False, "a refresh is not a stop; the next prompt engages"
    assert remote._owner_ready.is_set(), "a cold viewer is READY: the next prompt engages"


@pytest.mark.asyncio
async def test_stopped_reason_still_parks_the_viewer(tmp_path, monkeypatch):
    """Regression guard: the new branch must not shadow the stop path."""
    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over)
    refreshed: list[str] = []
    remote.set_refresh_callback(lambda: refreshed.append("refresh"))

    remote._on_disconnected(STOPPED_REASON)
    if remote._recovery_task is not None:
        await asyncio.wait_for(remote._recovery_task, timeout=5)

    assert remote._deliberate_stop is True
    assert refreshed == []


@pytest.mark.asyncio
async def test_owner_death_still_recovers(tmp_path, monkeypatch):
    """Regression guard: an unannounced exit still runs ``_recover_owner``."""
    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over)
    refreshed: list[str] = []
    remote.set_refresh_callback(lambda: refreshed.append("refresh"))

    remote._on_disconnected("owner exited")
    task = remote._recovery_task
    assert task is not None, "an unannounced exit must still start recovery"
    assert remote._recovering is True
    assert refreshed == []
    # The recovery loop's own outcomes (takeover, cold after 8 s) are covered
    # by test_remote_takeover; here only its START is the property.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_request_refresh_never_raises(tmp_path, monkeypatch):
    """Every failure is "kept": the reaper's own check is the fallback."""
    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over)
    assert (await remote.request_refresh()).startswith("kept:")

    class OldOwner:
        connected = True

        async def request_refresh(self) -> str:
            raise RuntimeError("unknown op: refresh_if_idle")

    remote._client = OldOwner()  # type: ignore[assignment]
    assert (await remote.request_refresh()).startswith("kept:")

    class Owner:
        connected = True

        async def request_refresh(self) -> str:
            return "retiring"

    remote._client = Owner()  # type: ignore[assignment]
    assert await remote.request_refresh() == "retiring"


def test_owner_idle_reads_the_snapshot(tmp_path, monkeypatch):
    """Cold → not idle (nothing to refresh); streaming, a running job or a
    parked gate → busy; otherwise idle."""
    from types import SimpleNamespace

    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over)
    assert remote.owner_idle() is False, "a cold viewer has no owner to refresh"

    class Client:
        connected = True

    remote._client = Client()  # type: ignore[assignment]
    remote._ready_for_events = True
    assert not remote.is_cold

    def store(**fields: Any) -> Any:
        base = {"streaming": False, "pending_gate": None, "jobs": []}
        base.update(fields)
        return SimpleNamespace(state=SimpleNamespace(**base))

    def with_state(**fields: Any) -> None:
        remote._frontend_store = store(**fields)  # type: ignore[assignment]

    with_state()
    assert remote.owner_idle() is True
    with_state(streaming=True)
    assert remote.owner_idle() is False
    with_state(pending_gate=object())
    assert remote.owner_idle() is False
    with_state(jobs=[SimpleNamespace(status="running")])
    assert remote.owner_idle() is False
    with_state(jobs=[SimpleNamespace(status="completed")])
    assert remote.owner_idle() is True
    remote._streaming = True
    assert remote.owner_idle() is False

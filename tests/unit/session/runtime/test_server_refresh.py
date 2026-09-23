"""``refresh_if_idle`` and ``announce_retiring`` on the RuntimeServer.

The viewer-side belt for the runtime's self-refresh (design-runtime-autorefresh
§3.3): a resume in the seconds after ``lop-update`` binds to a stale idle
owner before its reaper has noticed, and asks it to retire now. Every
uncertain answer is ``kept``: a wrong "retire" costs a cold start nobody
asked for, a wrong "keep" costs the reaper's next check.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from local_operator import update as update_mod
from local_operator.session.runtime.server import RuntimeServer, _ClientConn
from local_operator.update import BuildStamp
from tests.unit.session.runtime.test_server import FakeHandle

OLD = BuildStamp(version="0.49.8", source_ref="46a4e9b1234567")
NEW = BuildStamp(version="0.49.9", source_ref="f4a70b991234567")


class RefreshableHandle(FakeHandle):
    def __init__(self, *, reason: str = "") -> None:
        super().__init__()
        self.reason = reason
        self.stopped = False
        self.probes = 0

    def may_refresh(self) -> str:
        self.probes += 1
        return self.reason

    def request_stop(self) -> None:
        self.stopped = True


def _conn(kind: str) -> _ClientConn:
    return _ClientConn(writer=cast(Any, object()), kind=cast(Any, kind))


@pytest.fixture
def stale(monkeypatch):
    """The server booted on OLD; the disk now carries NEW, settled."""
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: NEW)
    monkeypatch.setattr(update_mod, "disk_build", lambda *_a, **_k: NEW)
    monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: 999.0)
    monkeypatch.delenv("LOP_BUILD_PREFIX", raising=False)


def _rig(handle: Any, *, boot: BuildStamp = OLD) -> tuple[RuntimeServer, list[dict[str, Any]]]:
    """A server booted on ``boot`` whose socket writes are captured, tagged
    with the recipient's kind so the daemon/attach split is assertable."""
    server = RuntimeServer(handle, kind="tui")
    server._boot_build = boot
    sent: list[dict[str, Any]] = []

    async def capture(target, frame):  # noqa: ANN001
        sent.append({"_recipient": target.kind, **frame})

    server._send_to = capture  # type: ignore[assignment]
    return server, sent


async def _ask(server: RuntimeServer, sent: list[dict[str, Any]], conn: _ClientConn) -> str:
    await server._on_request({"op": "refresh_if_idle", "req": 1}, conn)
    acks = [f for f in sent if f.get("op") in ("ack", "error")]
    assert acks, "the op never replied"
    reply = acks[-1]
    assert reply.get("op") == "ack", f"unexpected reply: {reply}"
    return str(reply.get("detail", ""))


@pytest.mark.asyncio
async def test_an_idle_stale_runtime_announces_and_retires(stale) -> None:
    handle = RefreshableHandle(reason="")
    server, sent = _rig(handle)
    viewer = _conn("attach")
    daemon = _conn("daemon")
    server._clients[id(viewer.writer)] = viewer
    server._clients[id(daemon.writer)] = daemon

    detail = await _ask(server, sent, viewer)

    # The answer NAMES the build the runtime is leaving for (NIT, PR #1141), and
    # it does so by prefix: an attach client reads ``retiring`` and the TUI's
    # bind-path refresh uses ``startswith``, while ``retire_now`` — whose callers
    # compare the whole string — still answers a bare ``retiring``.
    assert detail == f"retiring to {NEW.label()}", detail
    assert detail.removeprefix("retiring to ") == NEW.label()
    assert handle.stopped is True
    retiring = [f for f in sent if f.get("op") == "retiring"]
    assert [f["_recipient"] for f in retiring] == ["attach"], "the phone daemon never sees it"
    assert retiring[0]["reason"] == "stale-build"
    assert retiring[0]["from"] == OLD.label() and retiring[0]["to"] == NEW.label()
    # The WIRE field the viewer's notice is gated on, on the rung that refuses
    # NOTHING: an idle refresh must not claim admissions are closing (QA round 3,
    # Q-1 — the notice painted here for a handover that refused nothing).
    assert retiring[0]["draining"] is False, retiring[0]
    assert handle.probes == 2, "re-asked after the announce (the one await before the stop)"


@pytest.mark.asyncio
async def test_a_busy_stale_runtime_is_kept(stale) -> None:
    handle = RefreshableHandle(reason="busy")
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    assert await _ask(server, sent, viewer) == "kept: busy"
    assert handle.stopped is False
    assert not [f for f in sent if f.get("op") == "retiring"]


@pytest.mark.asyncio
async def test_a_matching_build_is_kept(stale, monkeypatch) -> None:
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: OLD)
    monkeypatch.setattr(update_mod, "disk_build", lambda *_a, **_k: OLD)
    handle = RefreshableHandle(reason="")
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    assert (await _ask(server, sent, viewer)).startswith("kept: build on disk matches")
    assert handle.stopped is False


@pytest.mark.asyncio
async def test_an_unsettled_install_is_kept(stale, monkeypatch) -> None:
    """An install that MOVED but has not settled is its own answer, not "current".

    ``lop refresh``'s own documentation says its first run is ``lop-update``, so
    it lands inside ``BUILD_SETTLE_S`` for every session on the machine — and
    the runtime used to answer ``kept: build on disk matches (or has not
    settled)`` for both shapes, which the CLI rendered as "already runs the build
    on disk" and counted as settled (exit 0). That told a rotating script the
    fleet was complete about sessions that were all about to retire (D1/M2, PR
    #1141). The two answers are now distinct, and this cell pins the boundary
    between them: same marker, same server, one variable changed.
    """
    monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: 1.0)
    handle = RefreshableHandle(reason="")
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    assert await _ask(server, sent, viewer) == "kept: the install on disk has not settled yet"
    assert handle.stopped is False
    assert not [f for f in sent if f.get("op") == "retiring"]

    # And the genuinely-matching shape still answers as it always did, from the
    # same code path: the distinction is the settle window, not a new default.
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: OLD)
    monkeypatch.setattr(update_mod, "disk_build", lambda *_a, **_k: OLD)
    assert await _ask(server, sent, viewer) == "kept: build on disk matches"
    assert handle.stopped is False


@pytest.mark.asyncio
async def test_a_draining_runtime_says_so_rather_than_busy(stale) -> None:
    """A runtime already leaving a signal answers with THAT, not with ``busy``.

    The two are different facts — "still working, moves when its turn ends"
    versus "leaving whatever you do next" — and the drain is invisible without
    the second one: a signalled runtime works on, looking ordinary, for up to
    ``SIGNAL_DRAIN_S`` (U2, PR #1141). The record carries the same sentence (see
    ``SessionRecord.leaving``), which is what ``lop sessions`` prints.
    """
    from local_operator.session.runtime.types import LEAVING_ON_SIGNAL

    handle = RefreshableHandle(reason="busy")
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer
    server.note_leaving(LEAVING_ON_SIGNAL)

    assert await _ask(server, sent, viewer) == "kept: already leaving"
    # The drain's own state outranks the stale-build judgement: the process is
    # leaving regardless, so "would you like to move" is not the honest answer.
    assert handle.probes == 0, "the leaving state is reported without asking the handle"
    assert handle.stopped is False
    assert not [f for f in sent if f.get("op") == "retiring"]


@pytest.mark.asyncio
async def test_note_leaving_publishes_on_the_record_immediately(stale) -> None:
    """``note_leaving`` reaches the RECORD before any heartbeat could carry it.

    The field's whole value is the window between the signal and the exit — up
    to two minutes — so a marker that waited for the ordinary 15 s heartbeat
    would leave most of that window invisible. Written through to the record and
    republished in one synchronous step, like ``set_record_started``.
    """
    from local_operator.session.runtime.types import LEAVING_ON_SIGNAL

    handle = RefreshableHandle(reason="")
    server, _sent = _rig(handle)
    published: list[dict[str, Any]] = []

    class Publisher:
        def heartbeat(self, **updates: Any) -> None:
            published.append(updates)

    server._publisher = Publisher()  # type: ignore[assignment]
    assert server._record.leaving == ""

    server.note_leaving(LEAVING_ON_SIGNAL)

    assert server._record.leaving == LEAVING_ON_SIGNAL, "the record is the readable surface"
    assert published and published[-1]["leaving"] == LEAVING_ON_SIGNAL
    # Idempotent: a repeat signal must not put another staged write and rename
    # on the far side of a signal.
    server.note_leaving(LEAVING_ON_SIGNAL)
    assert len(published) == 1


@pytest.mark.asyncio
async def test_a_new_departure_supersedes_the_last_failure(stale) -> None:
    """A record must not describe an abandoned handover while a NEW one is running.

    ``note_updating``'s own rule (agent review round 1, NIT 4) one rung over, and here
    it is load-bearing rather than tidy: ``SessionRecord.update_failed`` is read as a
    PAIR with ``leaving`` by every surface that has to tell a handover still waiting
    from one that was given up, and an abandon KEEPS the ordinary build phrase by design
    (``process._abandon_move``). Left beside a freshly latched drain, a stale failure
    makes that pair report the new attempt as the abandoned one — and the app paints
    from exactly this pair (``tui.app.drain_notice_for``).

    Keyed on the DEPARTURE rather than on a change of phrase, which is the case that
    forces it: the second attempt at the same build announces the same words, so a
    clear-on-change would leave this pair wrong in the one state the pair exists for.
    """
    from local_operator.session.runtime.types import LEAVING_FOR_BUILD

    handle = RefreshableHandle(reason="busy")
    server, _sent = _rig(handle)
    # The state an abandon leaves: the phrase kept, the failed pair published.
    server.note_leaving(LEAVING_FOR_BUILD)
    server._record.update_failed = "0.62.9 -> 0.62.12"

    server.note_leaving(LEAVING_FOR_BUILD)

    assert (
        server._record.leaving == LEAVING_FOR_BUILD
    ), "the phrase still describes the departure in force"
    assert server._record.update_failed == "", (
        "a new departure supersedes the last failure: otherwise this record reads as "
        "an abandoned handover while a fresh one is refusing work"
    )


@pytest.mark.asyncio
async def test_announcing_a_drain_publishes_the_record_in_the_same_call(stale) -> None:
    """The frame flag and the record phrase are ONE commit, not two writers.

    PR #1108 landed the runtime's own drain state and made the ``retiring``
    frame's ``draining`` flag the word the APP paints its notice from at frame
    receipt; this branch had added ``SessionRecord.leaving`` for the FLEET
    surfaces. Two renderings of one fact with two writers is a disagreement
    waiting to happen — a drop that shows the app a notice while ``lop
    sessions`` still says ``live``, or the reverse — so
    :meth:`RuntimeServer.announce_retiring` is the single writer: passing
    ``draining=True`` with a phrase publishes the record before the frame goes
    out, and a caller that announces no drain publishes nothing.
    """
    from local_operator.session.runtime.types import LEAVING_ON_SIGNAL

    handle = RefreshableHandle(reason="busy")
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    await server.announce_retiring("shutdown-drain", draining=True, leaving=LEAVING_ON_SIGNAL)

    frames = [f for f in sent if f.get("op") == "retiring"]
    assert frames and frames[0]["draining"] is True
    assert server._record.leaving == LEAVING_ON_SIGNAL, "the fleet surface is the same commit"

    # The idle handover is the negative control: it announces no drain, so there
    # is no pending exit for a fleet surface to report.
    other, _sent = _rig(RefreshableHandle(reason=""), boot=NEW)
    await other.announce_retiring("stale-build", to=NEW.label(), draining=False)
    assert other._record.leaving == ""


@pytest.mark.asyncio
async def test_work_arriving_after_the_announce_is_kept(stale) -> None:
    class Flips(RefreshableHandle):
        def may_refresh(self) -> str:
            self.probes += 1
            return "" if self.probes == 1 else "busy"

    handle = Flips()
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    detail = await _ask(server, sent, viewer)
    assert detail.startswith("kept: busy")
    assert handle.stopped is False
    assert [f for f in sent if f.get("op") == "retiring"], "announced, then refused: safe"


@pytest.mark.asyncio
async def test_a_runtime_that_cannot_judge_itself_is_kept(stale) -> None:
    server, sent = _rig(FakeHandle())  # no may_refresh
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer
    assert (await _ask(server, sent, viewer)).startswith("kept: this runtime cannot judge")

"""A BACKGROUND session finishing, as the real app announces it.

The reported bug this file guards: eleven concurrent sessions in one terminal
window, a background session completes, the sidebar paints its checkmark, and no
notification of any kind is delivered. The completion fact crossed the process
boundary correctly and was rendered correctly — nothing converted it into a
toast.

These tests drive the real ``OperatorApp`` against a real ``AttentionStore`` and
intercept at ``spawn_detached``, the actual process boundary a delivery crosses.
That is deliberate: a test that called the delivery helper directly would pass
with the observer leg wired to nothing, which is precisely the defect.

The distinction that makes this feature correct is asserted throughout: DELIVERY
and ACKNOWLEDGEMENT are separate watermarks. Announcing a session must never
mark it read (``docs/SESSION_SIDEBAR.md``), or the checkmark the operator uses
to find unread work would be cleared by the toast telling them about it.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.attention import AttentionStore, conversation_identity
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class AttachedSession(FakeSession):
    """A session pinned to the id the operator is looking at."""

    @property
    def session_id(self) -> str:
        return "current"


#: Monotonic birth clock for fixture sessions. Module-level rather than a
#: function attribute so the type checker can see it.
_birth_clock: float = 1_700_000_000.0


def _make_session(root: Path, session_id: str, name: str) -> Path:
    """A session directory the catalog can actually scan and name.

    Uses ``write_session_title`` rather than hand-writing a sidecar, so the name
    resolves through exactly the path the picker and sidebar use.
    """
    from local_operator.resume import write_session_title

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    # Pin the birth date the catalog ranks by. Without it these directories
    # fall through to the filesystem birthtime, which macOS has and Linux does
    # not — so every row would tie at 0 on CI and rank by session id instead of
    # by creation, silently changing which rows land under the announce cap.
    # Monotonic in creation order, matching what a real store looks like.
    global _birth_clock
    _birth_clock += 1.0
    (directory / "created_at.json").write_text(str(_birth_clock))
    (directory / "transcript.jsonl").write_text(
        '{"id":"e1","ts":1,"type":"message",'
        '"payload":{"kind":"message","role":"user","content":[{"text":"go"}]}}\n'
    )
    write_session_title(directory, name, user_set=False, past_names=[])
    return directory


def _journal_selection(directory: Path, selector: str) -> None:
    """Append the selection row ``Session._persist_selected_model`` writes.

    Written here rather than driven through a real ``Session`` because the
    subject is the OBSERVER's read of a stored journal: what matters is the row
    shape the reader keys on (``version == 2``, newest wins), not the live path
    that produced it. Keep it in step with that method if the payload changes —
    ``tests/unit/test_notification_isolation.py`` covers the reader itself
    against the same shape.
    """
    row = {
        "id": "selection-1",
        "ts": 1.0,
        "type": "custom",
        "payload": {
            "custom_type": "selected_model",
            "details": {"version": 2, "selector": selector, "effort": None, "boot": None},
        },
    }
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


@pytest.fixture
def store_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated config root, with cmux scrubbed from the environment.

    Every ``CMUX_*`` variable is removed because this suite runs inside cmux on
    the maintainer's machine: an inherited ``CMUX_SURFACE_ID`` would route every
    delivery down the cmux backend and silently stop testing the bare-terminal
    path, which is the configuration the bug was reported in.
    """
    root = tmp_path / "config"
    (root / "sessions").mkdir(parents=True)
    for key in ("CMUX_SURFACE_ID", "CMUX_WORKSPACE_ID", "CMUX_SOCKET_PATH", "CMUX_PANE_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: root)
    return root


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Every notification the app decides to send, captured at ITS boundary.

    The visible, deliberate opt-in out of ``tests/conftest.py``'s suite-wide
    ``LOCAL_OPERATOR_NO_NOTIFICATIONS`` gate, which exists so no test can put a
    real banner in the maintainer's Notification Centre. Opting in is safe here
    because nothing below is ever launched.

    INTERCEPTED AT ``detached_notify``, NOT AT ``spawn_detached``, and the
    distinction is a CI failure this file already paid for. What these tests
    assert is the app's ROUTING DECISION — which session is announced, how
    often, with what title, and whether the watermark moved. Which argv that
    decision finally becomes is a property of the HOST: ``detached_notify``
    resolves a signed bundle, then ``osascript``, then ``notify-send``, and on a
    Linux CI runner with none of them installed it correctly delivers nothing
    and returns False. Capturing spawns therefore made a green macOS run and a
    red Linux run of identical, correct code (run 34091362532, shard 2).

    The wire those argvs travel is not left untested — ``tui/test_notify.py``
    already pins every backend's exact command as a pure function.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    calls: list[list[str]] = []

    def deliver(title: str, body: str, *, session_id: str = "", subtitle: str = "") -> bool:
        # Recorded in the same flat shape the argv assertions used, so a test
        # reads as "what did the user get told", not as a call signature.
        calls.append([title, body, session_id, subtitle])
        return True

    monkeypatch.setattr("local_operator.tui.notify.detached_notify", deliver)
    # The cmux route is a genuinely different backend, so it is captured too:
    # this suite runs INSIDE cmux on the maintainer's machine, and the
    # ``store_root`` fixture scrubs ``CMUX_*`` precisely so the bare-terminal
    # path — the configuration the bug was reported in — is what is exercised.
    # If that scrub ever regresses, this records the fact instead of launching.
    monkeypatch.setattr(
        "local_operator.proc.spawn_detached", lambda argv, **kwargs: calls.append(list(argv)) or 1
    )
    monkeypatch.setattr(
        "local_operator.tui.notify.spawn_detached",
        lambda argv, **kwargs: calls.append(list(argv)) or 1,
    )
    return calls


#: The worker group `OperatorApp._notify_background_completions` runs its scan
#: under (``run_worker(run(), group="background-notify")``). Named once here so
#: the wait in `_await_completion_scan` cannot drift onto another group's workers.
_SCAN_WORKER_GROUP = "background-notify"


async def _await_completion_scan(app: OperatorApp) -> None:
    """Block until the app's background-completion scan has RAN TO ITS END.

    `_notify_background_completions` does its work on a worker thread
    (`asyncio.to_thread(collect)`), so every effect this file asserts — the row
    banners, the digest, the claims — lands after the poll that dispatched it has
    already returned. A fixed number of `pilot.pause()` rounds is not the same
    wait, because the two are not measured in the same unit: the budget is spent
    in LOOP TURNS, while the scan costs WALL TIME (a catalogue scan plus eleven
    SQLite transactions), and a contended runner stretches only the second.

    That is exactly what happened in run 34764919312 (shard `3.12, 0`, gw2): the
    digest test saw three row banners — precisely
    `_BACKGROUND_NOTIFY_MAX_PER_TICK` — and an EMPTY `digests`, because the digest
    is the last thing `collect` does and so the first thing a closed window loses.
    Nothing about that run was a wrong answer: the scan had the correct ten
    sessions in hand and was still claiming the seven the cap held back. The test
    read `spawned` while the scan was mid-flight.

    Waiting on the worker is waiting on the app's own completion signal, so the
    wait lasts exactly as long as the scan does — and it is the pattern the rest
    of the TUI suite already uses (`tests/unit/tui/test_info_panel.py` and its
    neighbours call `app.workers.wait_for_complete()`). It is scoped to the scan's
    group rather than taking that whole-manager form because an unrelated
    long-lived worker would turn this wait into a hang, and this suite carries no
    `pytest-timeout` to reclaim one.
    """
    scans = [worker for worker in app.workers if worker.group == _SCAN_WORKER_GROUP]
    if scans:
        await asyncio.gather(*(worker.wait() for worker in scans))


async def _settle(app: OperatorApp, pilot: Any, rounds: int = 6) -> None:
    """Run the app's own attention poll to completion, several times.

    Each round WAITS for the scan that round dispatches before the next round
    begins, so a round means "one attention poll and the work it started", not
    "one poll and however much of the work this machine happened to reach" —
    see `_await_completion_scan` for the CI failure that distinction cost.
    """
    for _ in range(rounds):
        await app._poll_completion_attention()
        await _await_completion_scan(app)
        for _ in range(4):
            await pilot.pause()


async def _booted(app: OperatorApp, pilot: Any) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return
    raise AssertionError("the session worker never attached a session")


@pytest.mark.asyncio
async def test_a_background_session_completing_is_announced(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """The operator's exact report: attached to one session, another finishes."""
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Article-search-svc schema review")
    store = AttentionStore(store_root / "attention.db")
    identity = conversation_identity(background)
    # An ESTABLISHED store: a prior, already-read completion, so the baseline is
    # installed and this is the steady state rather than a fresh machine.
    first = str(uuid.uuid4())
    store.publish(identity, first, "old", "complete")
    store.acknowledge(identity, first)

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        spawned.clear()

        store.publish(identity, str(uuid.uuid4()), "fresh", "complete")
        await _settle(app, pilot)

        assert len(spawned) == 1, spawned
        argv = " ".join(spawned[0])
        # Names the session, because with eleven open the name is the only
        # thing that says WHICH one finished.
        assert "Article-search-svc schema review" in argv
        # Carries the session id, which is what makes the banner clickable
        # back into the session (`lop resume-click`).
        assert "bg0000000001" in argv


@pytest.mark.asyncio
async def test_a_test_hosted_session_is_never_announced(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """The mock hosting is not news on THIS surface either (review round 1, R1-1).

    Reproduced as the reviewer found it — a real ``OperatorApp`` over a real
    ``AttentionStore`` — where a ``test/test-model`` session produced the same
    mock-bodied banner as an ``openai/gpt-5`` control, and a ``detached``
    delivery row spent the mock completion's watermark on the way. The process
    switch alone could never cover this surface: it silences the process that
    ADOPTED the mock, and the observer here is the operator's own TUI reading a
    store some rig filled.

    BOTH ARMS IN ONE CELL, so the control is the same run and the same scan:
    the real session is announced, the mock one is not.
    """
    _make_session(store_root, "current", "Current conversation")
    mock_dir = _make_session(store_root, "m00000000001", "Mock provider smoke")
    _journal_selection(mock_dir, "test/test-model")
    real_dir = _make_session(store_root, "r00000000001", "Article-search-svc schema review")
    _journal_selection(real_dir, "openai/gpt-5")

    store = AttentionStore(store_root / "attention.db")
    mock_identity = conversation_identity(mock_dir)
    real_identity = conversation_identity(real_dir)
    # An ESTABLISHED store: both sessions carry a prior, read completion, so the
    # baseline is installed and the next publish is news.
    for identity in (mock_identity, real_identity):
        seed = str(uuid.uuid4())
        store.publish(identity, seed, "old", "complete")
        store.acknowledge(identity, seed)

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        spawned.clear()

        mock_token = str(uuid.uuid4())
        store.publish(mock_identity, mock_token, "mock-completion", "complete")
        store.publish(real_identity, str(uuid.uuid4()), "real-completion", "complete")
        await _settle(app, pilot, rounds=10)

        assert len(spawned) == 1, spawned
        assert "r00000000001" in " ".join(spawned[0]), spawned
        assert "m00000000001" not in " ".join(spawned[0]), spawned

    # The skip is a FILTER ABOVE THE CLAIM, and this is the assertion that
    # distinguishes the two: the mock row is still claimable, so a surface that
    # may legitimately announce it (a desktop app on a host where notifications
    # are on) still can. Had the observer claimed and then suppressed it, this
    # would be False and the completion would be announced by nobody.
    assert store.claim_delivery(mock_identity, mock_token, "probe") is True


@pytest.mark.asyncio
async def test_the_announcement_does_not_repeat_on_later_polls(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """`unseen` is a LEVEL: without the watermark this fires once per second."""
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Background work")
    store = AttentionStore(store_root / "attention.db")
    identity = conversation_identity(background)
    store.publish(identity, str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot, rounds=10)
        assert len(spawned) == 1, spawned
        # Still unread — the level is unchanged — and still silent.
        assert store.state(identity)["unseen"] is True
        await _settle(app, pilot, rounds=10)
        assert len(spawned) == 1, spawned


@pytest.mark.asyncio
async def test_announcing_never_acknowledges_the_session(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """The invariant: a toast must not clear the sidebar's unread mark."""
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Background work")
    store = AttentionStore(store_root / "attention.db")
    identity = conversation_identity(background)
    store.publish(identity, str(uuid.uuid4()), "fresh", "complete")
    before = store.state(identity)["revision"]

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert spawned
        after = store.state(identity)
        assert after["unseen"] is True
        assert after["revision"] == before


@pytest.mark.asyncio
async def test_the_attached_session_is_not_announced_by_the_observer(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """The session on screen keeps its existing owner; two toasts would be wrong.

    This is also the focus gate, evaluated: an event is suppressed only when the
    user is demonstrably looking at the session that produced it, and for the
    attached row the in-app ``Notifier`` on ``TurnEnded`` already answers.

    BOTH ARMS IN ONE CELL, for two reasons that are both about this cell being
    able to fail (QA round 2, Q4). The first is NON-VACUITY: ``spawned == []``
    is also what a scan that never ran produces, so a background session that
    IS announced is what proves the scan happened at all. The second is
    DETERMINISM: the app dispatches its first scan at boot, and a scan whose
    worker read ``self._session`` before adoption finished sees no attached id
    and may announce the very row this cell is about — so the boot-time scans
    are settled and discarded BEFORE the completions under test are published,
    and what is asserted is the steady state rather than a race with boot.
    """
    current = _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Background work")
    store = AttentionStore(store_root / "attention.db")
    attached_identity = conversation_identity(current)
    background_identity = conversation_identity(background)
    # An ESTABLISHED store for BOTH rows: a prior, already-read completion, so
    # the baseline is installed and the publishes below are news rather than a
    # first reading.
    for identity in (attached_identity, background_identity):
        seed = str(uuid.uuid4())
        store.publish(identity, seed, "old", "complete")
        store.acknowledge(identity, seed)

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot, rounds=8)
        spawned.clear()

        store.publish(attached_identity, str(uuid.uuid4()), "attached-completion", "complete")
        store.publish(background_identity, str(uuid.uuid4()), "background-completion", "complete")
        await _settle(app, pilot, rounds=10)

        assert len(spawned) == 1, spawned
        argv = " ".join(spawned[0])
        # The control: a row nobody is looking at IS announced, so the scan ran.
        assert "bg0000000001" in argv, spawned
        # The subject: the attached row is not, and nothing named it.
        assert "Current conversation" not in argv, spawned


@pytest.mark.asyncio
async def test_a_parked_gate_is_left_to_the_runtime(
    store_root: Path, spawned: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row waiting for a person is announced by its own runtime, not here."""
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Waiting on you")
    store = AttentionStore(store_root / "attention.db")
    identity = conversation_identity(background)
    store.publish(identity, str(uuid.uuid4()), "fresh", "complete")

    from local_operator.tui import session_catalog

    real = session_catalog.load_catalog

    def pending(directory: Path) -> Any:
        # The pending mark is published by a live runtime record, which this
        # test has no process to produce; the field is what the leg reads.
        return [
            entry if entry.id != "bg0000000001" else _with_pending(entry)
            for entry in real(directory)
        ]

    def _with_pending(entry: Any) -> Any:
        import dataclasses

        return dataclasses.replace(entry, row=entry.row._replace(pending="approval"))

    monkeypatch.setattr(session_catalog, "load_catalog", pending)

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot, rounds=8)
        assert spawned == []


@pytest.mark.asyncio
async def test_an_upgrading_store_announces_nothing_on_the_first_tick(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """No-flood on upgrade, through the APP rather than the store alone.

    A database written before this feature carries a backlog of unread
    completions. The first app to open it must announce none of them: on the
    maintainer's real store that backlog is 121 conversations, i.e. 121 banners
    in one second.
    """
    import sqlite3

    _make_session(store_root, "current", "Current conversation")
    store = AttentionStore(store_root / "attention.db")
    for index in range(8):
        directory = _make_session(store_root, f"bg000000000{index}", f"Old work {index}")
        store.publish(conversation_identity(directory), str(uuid.uuid4()), "old", "complete")
    # Reproduce a pre-feature database exactly: the table is simply absent.
    with sqlite3.connect(store_root / "attention.db") as conn:
        conn.execute("DROP TABLE deliveries")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot, rounds=8)
        assert spawned == [], spawned
        # The backlog is still UNREAD; only its notifications are spent.
        assert all(
            AttentionStore(store_root / "attention.db").state(
                conversation_identity(store_root / "sessions" / f"bg000000000{i}")
            )["unseen"]
            for i in range(8)
        )


@pytest.mark.asyncio
async def test_notifications_disabled_stays_silent(
    store_root: Path, spawned: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The existing kill switch governs this path too."""
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Background work")
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot, rounds=6)
        assert spawned == []


@pytest.mark.asyncio
async def test_the_session_name_can_be_kept_off_the_banner(
    store_root: Path, spawned: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`display.notification_session_name` off titles with the brand instead.

    The observer path widens where a model-written session name appears,
    including on a lock screen; this is the opt-out.
    """
    monkeypatch.setattr("local_operator.tui.notify.session_names_in_notifications", lambda: False)
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Secret client migration")
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert spawned
        argv = " ".join(spawned[0])
        assert "Secret client migration" not in argv
        assert "Local Operator" in argv
        # Still identifies the session for the click-through.
        assert "bg0000000001" in argv


@pytest.mark.asyncio
async def test_a_failed_delivery_hands_the_claim_back(
    store_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend that delivered nothing must not leave the watermark lying.

    Otherwise a spawn refused once silences that completion permanently — the
    exact class of silent hole this feature exists to close.

    Does not take the ``spawned`` fixture (it patches delivery itself), so it
    opts out of the suite-wide notification gate on its own.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Background work")
    store = AttentionStore(store_root / "attention.db")
    identity = conversation_identity(background)
    token = str(uuid.uuid4())
    store.publish(identity, token, "fresh", "complete")
    monkeypatch.setattr("local_operator.tui.notify.detached_notify", lambda *a, **k: False)

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        # The claim was released, so the event is still available to whoever
        # can actually deliver it.
        assert AttentionStore(store_root / "attention.db").claim_delivery(identity, token, "test")


def _overlay(monkeypatch: pytest.MonkeyPatch, session_id: str, **fields: Any) -> None:
    """Rewrite one catalog row's live fields, leaving every other row real.

    The live state a row carries is published by a running runtime record, and
    these tests have no second process to produce one. Patching the catalog
    rather than faking the whole entry keeps the rest of the row — its name,
    its completion token, its unseen level — coming from the real store.
    """
    import dataclasses

    from local_operator.tui import session_catalog

    real = session_catalog.load_catalog

    def overlaid(directory: Path) -> Any:
        return [
            (
                dataclasses.replace(entry, row=entry.row._replace(**fields))
                if entry.id == session_id
                else entry
            )
            for entry in real(directory)
        ]

    monkeypatch.setattr(session_catalog, "load_catalog", overlaid)


# The round-1 `…attached_in_another_window…` test is subsumed here rather than
# kept beside this one: it asserted exactly this, for exactly one of the three
# states, and two tests spelling the same rule differently is how one of them
# later gets updated alone. Its rationale is preserved in the docstring below.
@pytest.mark.parametrize("live_state", ["attached", "busy", "wedged"])
@pytest.mark.asyncio
async def test_a_session_resident_in_another_window_is_not_announced(
    store_root: Path,
    spawned: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    live_state: str,
) -> None:
    """EVERY state meaning "another window owns this row" suppresses (M3).

    B2 shipped as ``live_state == "attached"``, but ``live_state`` is a single
    slot whose branches are mutually exclusive and ordered ``wedged`` > ``busy``
    > ``attached`` > ``idle``: an attached window RUNNING A TURN reports
    ``busy``, and one whose heartbeat went stale reports ``wedged``. Both still
    have that session open, and both still fire their own ``Notifier`` — so the
    ``attached``-only filter matched the narrowest of the three and let the
    other two double-notify. Reproduced at 1 banner each before the fix.

    ``busy`` is the reachable one post-#720: that PR re-pointed ``busy`` at
    conversational activity, so a session whose last turn finished unread enters
    it the moment the user sends a new prompt in another window.

    Parametrised rather than written three times so a state added to the
    suppressed set is one line here, and so the failure names which state broke.
    """
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Owned by another window")
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "complete")
    _overlay(monkeypatch, "bg0000000001", live_state=live_state)

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot, rounds=8)
        assert spawned == [], f"{live_state} is resident elsewhere; its own window toasts it"


@pytest.mark.asyncio
async def test_an_idle_background_session_is_still_announced(
    store_root: Path, spawned: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The B2 filter must key on ``attached`` alone, not on liveness.

    A resident-but-unattached session is exactly the case this feature exists
    for — a background runtime finishing with nobody watching it — so a filter
    written as "skip anything live" would suppress the reported bug's own
    scenario. Guards the fix from being over-applied.
    """
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Idle background work")
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "complete")
    _overlay(monkeypatch, "bg0000000001", live_state="idle")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot, rounds=8)
        assert len(spawned) == 1, spawned
        assert "Idle background work" in " ".join(spawned[0])


@pytest.mark.asyncio
async def test_a_backlog_is_capped_and_the_remainder_is_summarised(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """A store AT REST with a backlog must not fire one banner per row (B1).

    The baseline only covers the tick that CREATES the table. The overnight
    case — an already-migrated store, no observer running while background
    sessions finish — was measured at 30 banners on one tick. The cap bounds
    the spawn; the digest is what stops the remainder being silently dropped,
    which would be its own defect.

    Every backlogged completion is still CLAIMED, so the flood cannot simply
    return on the next revision change; the assertion below checks that too.
    """
    from local_operator.tui.app import _BACKGROUND_NOTIFY_MAX_PER_TICK

    _make_session(store_root, "current", "Current conversation")
    store = AttentionStore(store_root / "attention.db")
    # Establish the table and its baseline FIRST, so the backlog below is
    # published into a store that already has deliveries — the case the
    # migration test cannot reach.
    seed = _make_session(store_root, "bg0000000000", "Seed")
    seed_token = str(uuid.uuid4())
    store.publish(conversation_identity(seed), seed_token, "old", "complete")
    store.acknowledge(conversation_identity(seed), seed_token)

    backlog = 12
    for index in range(backlog):
        directory = _make_session(store_root, f"bg00000001{index:02d}", f"Overnight {index}")
        store.publish(conversation_identity(directory), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot, rounds=8)

    # Partitioned on the STRUCTURAL fact, not on the banner's prose: the digest
    # is the only delivery that carries no `session_id`, by design and because
    # it covers several sessions (there is no single transcript to reopen). A
    # substring test on "more session" would sort a session literally named
    # "…3 more sessions…" into the wrong bucket, and it breaks the moment a
    # design round rewords the copy — which one already has (review round 2,
    # nit). The `spawned` fixture records `[title, body, session_id, subtitle]`
    # for every delivery, so slot 2 is the discriminator; a 4-slot shape on
    # every call is also what proves no cmux argv leaked in to be mis-sorted.
    assert all(len(call) == 4 for call in spawned), spawned
    named = [call for call in spawned if call[2]]
    digests = [call for call in spawned if not call[2]]
    assert len(named) == _BACKGROUND_NOTIFY_MAX_PER_TICK, spawned
    # The count is reported rather than lost: 12 backlogged, 3 named, and the
    # digest carries the ABSOLUTE total of 12 rather than the remainder of 9 —
    # "9 more" is only meaningful to a user who noticed the three banners it
    # counts from, which a lock screen does not guarantee (design round 2, D11).
    assert len(digests) == 1, spawned
    assert f"{backlog} sessions finished" in " ".join(digests[0])
    # Claimed, not merely skipped — otherwise the backlog re-floods next tick.
    fresh = AttentionStore(store_root / "attention.db")
    assert all(
        not fresh.claim_delivery(
            conversation_identity(store_root / "sessions" / f"bg00000001{index:02d}"),
            str(
                fresh.state(
                    conversation_identity(store_root / "sessions" / f"bg00000001{index:02d}")
                )["completion_token"]
            ),
            "test",
        )
        for index in range(backlog)
    )


@pytest.mark.parametrize("mode", ["returns_false", "raises"])
@pytest.mark.asyncio
async def test_a_failed_digest_does_not_consume_the_claims_it_stood_for(
    store_root: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """A digest that fails hands back every claim it spoke for (review round 2, M1).

    The cap CLAIMS the rows it holds back — it must, or the backlog re-floods —
    and the digest is then the only thing that will ever speak for them. So a
    digest that silently fails takes N completions with it permanently, where a
    single failed row loses one: measured at 9 of 12 the user is never told
    about. That is strictly worse than the flood B1 replaced, because a flood is
    noisy and self-correcting while this is silent and the claim asserts the
    banner went out.

    Neither failure mode is exotic. ``detached_notify`` returns False by
    contract on any host with no notifier on PATH — a Linux box without
    ``notify-send`` — and on macOS before the signed bundle finishes building,
    which is precisely the cold-start moment a backlog exists. The ``raises``
    mode is what the helper's own ``except Exception`` swallows.

    Asserts that NO COMPLETION IS PERMANENTLY LOST — every backlogged session is
    eventually named to the user — rather than asserting the release call. That
    is the property the user has, it is what the reviewer measured (9 of 12
    never announced), and it cannot be satisfied by a release that rolled the
    watermark somewhere useless. It is also why the released rows are followed
    all the way to a banner: the digest fails on every tick here, so the only
    way to reach 12 is for the released claims to keep coming back under the cap
    until each has had a real one.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    _make_session(store_root, "current", "Current conversation")
    store = AttentionStore(store_root / "attention.db")
    seed = _make_session(store_root, "bg0000000000", "Seed")
    seed_token = str(uuid.uuid4())
    store.publish(conversation_identity(seed), seed_token, "old", "complete")
    store.acknowledge(conversation_identity(seed), seed_token)

    backlog = 12
    for index in range(backlog):
        directory = _make_session(store_root, f"bg00000001{index:02d}", f"Overnight {index}")
        store.publish(conversation_identity(directory), str(uuid.uuid4()), "fresh", "complete")

    named: list[list[str]] = []

    def deliver(title: str, body: str, *, session_id: str = "", subtitle: str = "") -> bool:
        # ONLY the digest fails. A digest is the delivery with no `session_id`
        # (it covers several sessions, so there is no transcript to reopen), so
        # this splits on the same structural fact the cap test partitions on.
        if not session_id:
            if mode == "raises":
                raise RuntimeError("notifier backend exploded")
            return False
        named.append([title, body, session_id, subtitle])
        return True

    monkeypatch.setattr("local_operator.tui.notify.detached_notify", deliver)
    monkeypatch.setattr(
        "local_operator.proc.spawn_detached", lambda argv, **kwargs: named.append(list(argv)) or 1
    )
    monkeypatch.setattr(
        "local_operator.tui.notify.spawn_detached",
        lambda argv, **kwargs: named.append(list(argv)) or 1,
    )

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        # The cap lets `_BACKGROUND_NOTIFY_MAX_PER_TICK` through per scan, so
        # draining 12 takes several scans however fast the machine is. Bounded
        # by SCANS, not by a clock: `_settle` drives the poll directly.
        await _settle(app, pilot, rounds=4 * backlog)

    announced = {call[2] for call in named}
    missing = {f"bg00000001{index:02d}" for index in range(backlog)} - announced
    assert not missing, (
        f"{len(missing)} completions were claimed by a digest that never reached the user and "
        f"were never re-announced: {sorted(missing)}"
    )
    # And nothing is left holding a claim with nothing delivered.
    fresh = AttentionStore(store_root / "attention.db")
    stranded = [
        index
        for index in range(backlog)
        if fresh.claim_delivery(
            conversation_identity(store_root / "sessions" / f"bg00000001{index:02d}"),
            str(
                fresh.state(
                    conversation_identity(store_root / "sessions" / f"bg00000001{index:02d}")
                )["completion_token"]
            ),
            "test",
        )
    ]
    assert not stranded, f"rows still unclaimed after every one was announced: {stranded}"


@pytest.mark.asyncio
async def test_a_backlog_does_not_reflood_on_the_next_revision_change(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """The cap holds back a banner, not a claim (B1's other half).

    A cap that skipped rows without claiming them would re-announce the whole
    backlog the moment any unrelated completion moved the revision — turning a
    one-tick flood into a recurring one.
    """
    _make_session(store_root, "current", "Current conversation")
    store = AttentionStore(store_root / "attention.db")
    seed = _make_session(store_root, "bg0000000000", "Seed")
    seed_token = str(uuid.uuid4())
    store.publish(conversation_identity(seed), seed_token, "old", "complete")
    store.acknowledge(conversation_identity(seed), seed_token)
    for index in range(10):
        directory = _make_session(store_root, f"bg00000002{index:02d}", f"Overnight {index}")
        store.publish(conversation_identity(directory), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot, rounds=8)
        spawned.clear()
        # An unrelated completion moves the revision, which is what re-opens
        # the gate the observer holds.
        latecomer = _make_session(store_root, "bg0000000999", "Latecomer")
        store.publish(conversation_identity(latecomer), str(uuid.uuid4()), "fresh", "complete")
        await _settle(app, pilot, rounds=8)

    # Only the new one, never the backlog again.
    assert len(spawned) == 1, spawned
    assert "Latecomer" in " ".join(spawned[0])


@pytest.mark.asyncio
async def test_a_released_claim_is_retried_without_an_unrelated_completion(
    store_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`release_delivery`'s promised retry actually happens (review M1 / QA Q1).

    ``revision()`` reads ``completions`` and ``receipts`` only, so a claim taken
    and handed back leaves it byte-identical and the observer's own gate
    short-circuits forever. Measured at 0 retries across 60 ticks with the
    backend healthy again — on a single-session machine, silence, which is the
    exact hole ``release_delivery`` exists to close.

    Asserts the RETRY, not the release: the pre-existing test already covers
    the claim going back, and it passed throughout the defect.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Background work")
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "complete")

    healthy = False
    calls: list[list[str]] = []

    def deliver(title: str, body: str, *, session_id: str = "", subtitle: str = "") -> bool:
        # Fails while the backend is "down", succeeds once it recovers. Nothing
        # else about the store changes in between, so the only thing that can
        # produce a second attempt is the observer re-scanning on its own.
        if not healthy:
            return False
        calls.append([title, body, session_id, subtitle])
        return True

    monkeypatch.setattr("local_operator.tui.notify.detached_notify", deliver)
    monkeypatch.setattr(
        "local_operator.proc.spawn_detached", lambda argv, **kwargs: calls.append(list(argv)) or 1
    )
    monkeypatch.setattr(
        "local_operator.tui.notify.spawn_detached",
        lambda argv, **kwargs: calls.append(list(argv)) or 1,
    )

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot, rounds=6)
        assert calls == [], "the backend was down; nothing should have been delivered"
        healthy = True
        # NOTHING is published here. Without the fix the revision is unchanged,
        # the gate short-circuits, and this stays empty forever.
        await _settle(app, pilot, rounds=6)
        assert len(calls) == 1, calls
        assert "Background work" in " ".join(calls[0])


@pytest.mark.asyncio
async def test_a_host_that_can_never_deliver_stops_rescanning(
    store_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry is BOUNDED: a dead backend must not rescan forever (round 2, M2).

    The M1/Q1 fix forgets the revision gate whenever a claim was released, so
    the released row is re-examined without waiting for an unrelated completion.
    Unbounded, that degenerates on a host where delivery ALWAYS fails — no
    ``notify-send`` on Linux, no ``osascript``, the macOS bundle not yet built —
    into a permanent 1 Hz loop: scan, claim, fail, release, forget, rescan.
    Measured at 23 catalog scans and 138 SQLite writes over 20 ticks against a
    store where nothing changed, on the same ``attention.db`` all of the
    operator's sessions contend for.

    STRUCTURAL, NOT TIMED. The assertion is that work per tick DECAYS — a later,
    equally long run of idle ticks costs strictly fewer scans and writes than an
    earlier one — rather than a bound on how long anything took. A per-tick
    retry fails it by construction on any machine (every window costs the same),
    and there is no clock reference anywhere in it. Ticks are counted by driving
    `_poll_completion_attention` directly, so machine speed cannot enter.

    Zero is deliberately NOT asserted: the retry must never stop entirely, which
    is the invariant the third assertion below pins. A cap that stopped retrying
    would make a backend outage longer than the cap a permanent silence — the
    round-1 M1/Q1 hole, re-created. That is not hypothetical: capping was
    implemented first here and `…a_released_claim_is_retried_without_an_
    unrelated_completion` went red, which is why this is backoff.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    _make_session(store_root, "current", "Current conversation")
    for index in range(3):
        directory = _make_session(store_root, f"bg00000003{index:02d}", f"Background {index}")
        store = AttentionStore(store_root / "attention.db")
        store.publish(conversation_identity(directory), str(uuid.uuid4()), "fresh", "complete")

    from local_operator.session import attention as attention_module
    from local_operator.tui import session_catalog

    scans = 0
    real_load = session_catalog.load_catalog

    def counting_load(directory: Path) -> Any:
        nonlocal scans
        scans += 1
        return real_load(directory)

    writes = 0
    real_claim = attention_module.AttentionStore.claim_delivery
    real_release = attention_module.AttentionStore.release_delivery

    def counting_claim(self: Any, conversation: str, token: str, backend: str) -> bool:
        nonlocal writes
        writes += 1
        return real_claim(self, conversation, token, backend)

    def counting_release(self: Any, conversation: str, token: str) -> bool:
        nonlocal writes
        writes += 1
        return real_release(self, conversation, token)

    monkeypatch.setattr(session_catalog, "load_catalog", counting_load)
    monkeypatch.setattr(attention_module.AttentionStore, "claim_delivery", counting_claim)
    monkeypatch.setattr(attention_module.AttentionStore, "release_delivery", counting_release)

    delivered: list[str] = []
    healthy = False

    def deliver(title: str, body: str, *, session_id: str = "", subtitle: str = "") -> bool:
        # The host with no notifier backend: `detached_notify`'s documented
        # False, on every call, for as long as the backend is absent.
        if not healthy:
            return False
        delivered.append(title)
        return True

    monkeypatch.setattr("local_operator.tui.notify.detached_notify", deliver)
    monkeypatch.setattr(
        "local_operator.proc.spawn_detached", lambda argv, **kwargs: delivered.append(argv[0]) or 1
    )
    monkeypatch.setattr(
        "local_operator.tui.notify.spawn_detached",
        lambda argv, **kwargs: delivered.append(argv[0]) or 1,
    )

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        # Three EQUAL windows of idle ticks. Nothing is published in any of
        # them, so every scan and every write is pure waste; with the retry
        # rate-limited the later windows must cost strictly less than the first.
        window = 12
        await _settle(app, pilot, rounds=window)
        assert delivered == [], "the backend is down; nothing should have gone out"
        first_scans, first_writes = scans, writes

        await _settle(app, pilot, rounds=window)
        second_scans, second_writes = scans - first_scans, writes - first_writes

        await _settle(app, pilot, rounds=window)
        third_scans, third_writes = (
            scans - first_scans - second_scans,
            writes - first_writes - second_writes,
        )

        assert third_scans < first_scans, (
            f"idle scans per {window}-tick window went {first_scans} -> {second_scans} -> "
            f"{third_scans}: the retry is not backing off and this host rescans forever"
        )
        assert third_writes < first_writes, (
            f"idle SQLite writes per {window}-tick window went {first_writes} -> "
            f"{second_writes} -> {third_writes} against a store where nothing changed"
        )
        # A per-tick retry would scan on every one of them; the gate must hold
        # for the large majority of an idle window.
        assert (
            third_scans * 2 < window
        ), f"{third_scans} scans in {window} idle ticks — the revision gate is barely holding"

        # ...and the budget is per STORE CHANGE, not per process: a real
        # completion re-arms it, so the M1/Q1 retry it bounds still works.
        healthy = True
        latecomer = _make_session(store_root, "bg0000000999", "Latecomer")
        AttentionStore(store_root / "attention.db").publish(
            conversation_identity(latecomer), str(uuid.uuid4()), "fresh", "complete"
        )
        await _settle(app, pilot, rounds=8)
        assert "Latecomer" in " ".join(delivered), delivered


@pytest.mark.asyncio
async def test_one_unreadable_row_does_not_mute_the_others(
    store_root: Path, spawned: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising row loses itself, never its siblings (QA Q2).

    ``claim_delivery`` sits outside the delivery helper's own try/except and
    touches SQLite, so a corrupt store (``DatabaseError``) or a read-only one
    (``OperationalError``) raised straight out of the first row and dropped
    every remaining session that tick — measured at 0 of 5 toasts. Reachable
    without simulation, which is why the guard is per row.
    """
    import sqlite3

    _make_session(store_root, "current", "Current conversation")
    for index in range(4):
        directory = _make_session(store_root, f"bg000000030{index}", f"Sibling {index}")
        AttentionStore(store_root / "attention.db").publish(
            conversation_identity(directory), str(uuid.uuid4()), "fresh", "complete"
        )

    # The catalog ranks by recency, so which row is scanned FIRST depends on
    # directory mtimes the test does not control — and this test is only
    # meaningful when the raising row precedes its siblings. Pin the order by
    # id so the poisoned row is provably first; the ranking itself is asserted
    # elsewhere and is not the property under test here.
    from local_operator.tui import session_catalog

    real_catalog = session_catalog.load_catalog
    monkeypatch.setattr(
        session_catalog,
        "load_catalog",
        lambda directory: sorted(real_catalog(directory), key=lambda entry: entry.id),
    )

    real_claim = AttentionStore.claim_delivery
    poisoned = {"bg0000000300"}

    def claim(self: Any, conversation: str, token: str, backend: str) -> bool:
        if any(bad in conversation for bad in poisoned):
            raise sqlite3.DatabaseError("file is not a database")
        return real_claim(self, conversation, token, backend)

    monkeypatch.setattr(AttentionStore, "claim_delivery", claim)

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot, rounds=8)

    delivered = " ".join(" ".join(call) for call in spawned)
    # The poisoned row is lost, and only it.
    assert "Sibling 0" not in delivered, spawned
    for index in range(1, 4):
        assert f"Sibling {index}" in delivered, spawned


@pytest.mark.asyncio
async def test_an_interrupted_session_is_not_announced_as_an_error(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """`interrupted` gets its own sentence (design D3).

    The store admits exactly complete/error/interrupted, and the maintainer's
    real store holds 318 complete, 14 interrupted and 0 error — so folding
    interrupted into error made every non-success banner the feature can raise
    today assert a failure that did not happen.
    """
    from local_operator.tui.notify import (
        BODY_ERROR,
        CONTEXT_ATTENTION,
        CONTEXT_INTERRUPTED,
    )

    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Halted midway")
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "interrupted")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        argv = " ".join(spawned[0])
        assert CONTEXT_INTERRUPTED in argv
        assert CONTEXT_ATTENTION not in argv
        assert BODY_ERROR not in argv


@pytest.mark.asyncio
async def test_an_errored_session_still_says_it_errored(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """D3's other half: splitting the states must not lose the error wording."""
    from local_operator.tui.notify import CONTEXT_ATTENTION

    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Broken run")
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "error")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        assert CONTEXT_ATTENTION in " ".join(spawned[0])


@pytest.mark.asyncio
async def test_a_session_with_no_stored_title_is_not_titled_with_its_prompt(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """Role boilerplate must not eat the banner's identifying words (design D1/D2).

    ``session_name`` falls back to the opening user message, and an
    agent-spawned session's opener begins with its role preamble — so a banner
    clipped at macOS's ~43 characters showed ``[team: lopdev] You are reviewer
    on this te…`` with every discriminating word past the cut. 15 of 67 rows on
    the maintainer's store have no stored title, and they are disproportionately
    the background sessions this feature exists to announce.

    The neutral sentence is also NOT the brand: ``Local Operator`` was
    pixel-identical to the opt-out banner, two meanings in one frame (D2).
    """
    from local_operator.tui.notify import APP_NAME, BACKGROUND_FALLBACK_TITLE

    _make_session(store_root, "current", "Current conversation")
    # A session directory with a transcript but NO title.json — the shape
    # `_make_session` produces before `write_session_title` runs.
    directory = store_root / "sessions" / "bg0000000001"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id":"e1","ts":1,"type":"message","payload":{"kind":"message","role":"user",'
        '"content":[{"text":"[team: lopdev] You are reviewer on this team. '
        'The manager is manager."}]}}\n'
    )
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(directory), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        argv = " ".join(spawned[0])
        assert BACKGROUND_FALLBACK_TITLE in argv
        assert "team: lopdev" not in argv
        assert "You are reviewer" not in argv
        # Distinct from the opt-out banner, which keeps the brand.
        assert APP_NAME not in argv
        # Still clickable back into the session.
        assert "bg0000000001" in argv


@pytest.mark.asyncio
async def test_the_opt_out_banner_is_distinguishable_from_the_nameless_one(
    store_root: Path, spawned: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two different meanings must not render as one frame (design D2)."""
    from local_operator.tui.notify import APP_NAME, BACKGROUND_FALLBACK_TITLE

    monkeypatch.setattr("local_operator.tui.notify.session_names_in_notifications", lambda: False)
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Secret client migration")
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        argv = " ".join(spawned[0])
        assert APP_NAME in argv
        assert BACKGROUND_FALLBACK_TITLE not in argv
        assert "Secret client migration" not in argv


@pytest.mark.parametrize(
    ("kinds", "expected_subtitle_key"),
    [
        # ALL-ERROR remainder. The catalog ranks by `(tier, -mtime, id)`, not by
        # kind, so this is not a contrived shape — it is what the cap produces
        # whenever a batch of failures finishes together.
        (["error"] * 8, "error"),
        (["interrupted"] * 8, "interrupted"),
        # MIXED remainder, and mixed NO MATTER WHICH ROWS THE CAP ANNOUNCES.
        # Both kinds exceed the cap, so at most one of them can be fully
        # consumed by the three banners and the remainder always holds both.
        # An earlier draft used 5 complete + 3 error and PASSED AGAINST THE
        # DEFECT: the catalog announced the three errors, leaving an all-
        # complete remainder that `Complete` described correctly. That is QA
        # round 1's Q2 vacuity repeating, and this composition is what removes
        # the dependency on an ordering the test does not control.
        (["complete"] * 5 + ["error"] * 5, None),
    ],
    ids=["all-error", "all-interrupted", "mixed"],
)
@pytest.mark.asyncio
async def test_the_digest_never_asserts_an_outcome_the_remainder_did_not_have(
    store_root: Path,
    spawned: list[list[str]],
    kinds: list[str],
    expected_subtitle_key: str | None,
) -> None:
    """The digest hardcoded `Complete` over whatever the cap held back (D8).

    This is design round 1's D3 — `interrupted` folded into a state it is not —
    reappearing at the scale of an arbitrary number of sessions rather than one.
    A user reading "Complete · 5 more sessions finished" on a lock screen has
    been told five things succeeded when all five failed, and the sidebar's `✗`
    is the only thing that contradicts it.

    Driven through the real observer against a real store, so what is asserted
    is the banner the user would actually receive rather than a helper's return
    value. Round 1 shipped a vacuous guard here once already (QA Q2, which
    passed WITH the defect because catalog recency put the poisoned row last),
    so every row in the over-cap remainder carries the kind under test: whatever
    ordering the catalog chooses, the remainder's composition is the same.
    """
    from local_operator.tui.app import _BACKGROUND_NOTIFY_MAX_PER_TICK
    from local_operator.tui.notify import CONTEXT_MIXED, CONTEXTS

    _make_session(store_root, "current", "Current conversation")
    store = AttentionStore(store_root / "attention.db")
    # An ESTABLISHED store, so this is the steady state rather than the tick
    # that creates the deliveries table.
    seed = _make_session(store_root, "bg0000000000", "Seed")
    seed_token = str(uuid.uuid4())
    store.publish(conversation_identity(seed), seed_token, "old", "complete")
    store.acknowledge(conversation_identity(seed), seed_token)

    for index, kind in enumerate(kinds):
        directory = _make_session(store_root, f"bg00000002{index:02d}", f"Overnight {index}")
        store.publish(conversation_identity(directory), str(uuid.uuid4()), "fresh", kind)

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot, rounds=8)

    # Structural discriminator, not prose: the digest is the only delivery with
    # no `session_id`, because it covers several sessions.
    assert all(len(call) == 4 for call in spawned), spawned
    digests = [call for call in spawned if not call[2]]
    assert len(digests) == 1, spawned
    title, body, _, subtitle = digests[0]

    # THE GUARD IS ONLY MEANINGFUL IF THE REMAINDER IS THE SHAPE UNDER TEST, so
    # the remainder's composition is DERIVED from what actually went out rather
    # than assumed from the parametrisation. Each named banner's subtitle is its
    # own kind's category, so subtracting those from the published set leaves
    # the kinds the digest really spoke for — which is the check the vacuous
    # draft above was missing.
    named = [call for call in spawned if call[2]]
    remainder = list(kinds)
    for shown in (call[3] for call in named):
        for kind, context in CONTEXTS.items():
            if context == shown and kind in remainder:
                remainder.remove(kind)
                break
    assert len(remainder) == len(kinds) - _BACKGROUND_NOTIFY_MAX_PER_TICK, (spawned, remainder)

    if expected_subtitle_key is None:
        # Not vacuous: the set the digest stands for genuinely holds both kinds.
        assert len(set(remainder)) > 1, remainder
        assert subtitle == CONTEXT_MIXED, digests
        # The exact failure: neither side's category may be asserted over a set
        # that is only partly in it.
        assert subtitle != CONTEXTS["complete"], digests
        assert subtitle != CONTEXTS["error"], digests
    else:
        assert set(remainder) == {expected_subtitle_key}, remainder
        assert subtitle == CONTEXTS[expected_subtitle_key], digests
    # The defect in one line, for every case: a remainder that is not uniformly
    # complete must never be announced as complete.
    assert subtitle != CONTEXTS["complete"], digests

    # The title carries the tick's ABSOLUTE total — banners shown plus the ones
    # held back — so it means something to a user who never saw the cap (D11).
    assert str(len(kinds)) in title, digests
    assert f"{len(kinds) - _BACKGROUND_NOTIFY_MAX_PER_TICK} more" not in title, digests
    # …and it is not the row banners' title, so the summary is a different kind
    # of object at a glance rather than a fourth identical sibling (D9).
    assert named, spawned
    assert all(title != call[0] for call in named), spawned
    # The one inert banner in the stack says where to go instead (D10).
    assert "sidebar" in body.lower(), digests


@pytest.mark.parametrize(
    ("announced_kind", "held_kind"),
    [
        # Keep announced and held outcomes different under the category ladder:
        # errors precede interruptions. Completed outcomes can no longer be
        # held behind interruptions merely by assigning them older activity.
        ("error", "interrupted"),
        # The same seam in the other direction, where the held-back set is the
        # alarming one: announcing 3 complete over 5 errors read
        # `8 sessions finished` / `Needs attention` under the old subtitle
        # scope and `Complete` under the old title's. Either way the two lines
        # described different sets.
        ("complete", "error"),
    ],
    ids=["error-announced-interrupted-held", "complete-announced-error-held"],
)
@pytest.mark.asyncio
async def test_the_digest_title_and_subtitle_describe_the_same_sessions(
    store_root: Path,
    spawned: list[list[str]],
    announced_kind: str,
    held_kind: str,
) -> None:
    """Title and subtitle must count the SAME set (design round 3, D12).

    D8 moved the subtitle onto the remainder's real kinds; D11 moved the title
    onto the tick's absolute total. Independently each was right, and together
    they left the frame's two lines with different denominators — the title
    naming the whole tick, the subtitle speaking only for the rows the cap held
    back. So `5 complete + 3 interrupted` rendered `8 sessions finished` over
    `Complete`: D8's exact sentence, one line up, reachable with ordinary data
    because the cap is 3 and the catalog ranks by `(tier, -mtime, id)` rather
    than by kind.

    Asserted as a RELATION between the two lines rather than as an expected
    string: the guard is that whatever set the title counts is the set the
    subtitle describes, so it fails for a title scoped to the remainder just as
    it fails for a subtitle scoped to it. A test pinning only `Mixed outcomes`
    would pass a build that fixed the subtitle by shrinking the title, which
    would silently reopen D11.

    THE PRECONDITION IS ASSERTED, NOT ASSUMED. This PR has now shipped three
    guards that passed against the defect they named because the catalog put
    the interesting rows where the test did not expect (QA round 1's Q2; the
    round-3 mixed case, where the cap announced all three errors and left an
    all-complete remainder that `Complete` described correctly). A D12 case
    whose announced and held-back sets happen to share a kind proves nothing at
    all — the two scopes agree by accident and the old code passes. So the two
    sets are derived from what actually went out and are required to differ in
    kind before any assertion about the digest is trusted.
    """
    from local_operator.tui.app import _BACKGROUND_NOTIFY_MAX_PER_TICK
    from local_operator.tui.notify import (
        CONTEXT_MIXED,
        CONTEXTS,
        background_digest_title,
        digest_subtitle,
    )

    _make_session(store_root, "current", "Current conversation")
    store = AttentionStore(store_root / "attention.db")
    # An ESTABLISHED store, so this is the steady state rather than the tick
    # that creates the deliveries table.
    seed = _make_session(store_root, "bg0000000000", "Seed")
    seed_token = str(uuid.uuid4())
    store.publish(conversation_identity(seed), seed_token, "old", "complete")
    store.acknowledge(conversation_identity(seed), seed_token)

    # Each announced category precedes its held category. Pin creation dates
    # within the groups as well, so filesystem timing cannot affect selection.
    # The precondition below still proves the two scopes differ in kind.
    held = [held_kind] * 5
    announced = [announced_kind] * _BACKGROUND_NOTIFY_MAX_PER_TICK
    for index, kind in enumerate(held + announced):
        directory = _make_session(store_root, f"bg00000002{index:02d}", f"Overnight {index}")
        (directory / "created_at.json").write_text(str(1_700_000_000 + index))
        store.publish(conversation_identity(directory), str(uuid.uuid4()), "fresh", kind)

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot, rounds=8)

    # Structural discriminator, not prose: the digest is the only delivery with
    # no `session_id`, because it covers several sessions.
    assert all(len(call) == 4 for call in spawned), spawned
    digests = [call for call in spawned if not call[2]]
    named = [call for call in spawned if call[2]]
    assert len(digests) == 1, spawned
    title, _, _, subtitle = digests[0]

    # WHAT WAS ACTUALLY ANNOUNCED, read back off the banners rather than
    # assumed from the parametrisation: each named banner's subtitle is its own
    # kind's category, so inverting `CONTEXTS` recovers the announced set, and
    # the rest of the published set is what the digest stood for.
    by_context = {context: kind for kind, context in CONTEXTS.items()}
    shown_kinds = [by_context[call[3]] for call in named]
    remainder = list(held + announced)
    for kind in shown_kinds:
        remainder.remove(kind)

    # THE PRECONDITION. If these two sets share a kind the case is vacuous: the
    # title's scope and the subtitle's scope agree by accident and the defect
    # passes. Everything below is only evidence because this holds.
    assert len(shown_kinds) == _BACKGROUND_NOTIFY_MAX_PER_TICK, spawned
    assert set(shown_kinds) == {announced_kind}, shown_kinds
    assert set(remainder) == {held_kind}, remainder
    assert set(shown_kinds).isdisjoint(remainder), (shown_kinds, remainder)

    # THE DEFECT, as a relation between the two lines. The title counts the
    # whole tick, so the subtitle must speak for the whole tick — a set holding
    # two kinds, which is mixed. Under the defect this read `Complete` (or the
    # held kind's own category) over a count that included sessions in neither.
    whole_tick = shown_kinds + remainder
    assert title == background_digest_title(len(whole_tick)), (title, whole_tick)
    assert subtitle == digest_subtitle(whole_tick), (title, subtitle, whole_tick)
    assert subtitle == CONTEXT_MIXED, (title, subtitle, whole_tick)
    # Neither side's category may be asserted over a set only partly in it.
    assert subtitle != CONTEXTS[announced_kind], (title, subtitle)
    assert subtitle != CONTEXTS[held_kind], (title, subtitle)
    # …and the title still carries the ABSOLUTE total, so a subtitle fixed by
    # shrinking the title's scope fails here rather than silently reopening D11.
    assert str(len(whole_tick)) in title, (title, whole_tick)
    assert str(len(remainder)) not in title.split()[0], (title, remainder)


def _with_assistant_reply(directory: Path, text: str) -> None:
    """Append one assistant turn, so the session has something it last SAID.

    Written through the same one-line-per-entry shape ``_make_session`` uses
    rather than through a writer helper, because what is under test is exactly
    the tail read ``session_preview`` performs over the bytes on disk.
    """
    entry = json.dumps(
        {
            "id": "a1",
            "ts": 2,
            "type": "message",
            "payload": {"kind": "message", "role": "assistant", "content": [{"text": text}]},
        }
    )
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(entry + "\n")


@pytest.mark.asyncio
async def test_the_banner_body_carries_the_final_assistant_line(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """The operator's request: say WHAT the session concluded, not where they were.

    The body was the fixed sentence ``BODY_BACKGROUND`` — a routing fact the
    reader is already the authority on — so eleven completions in one window
    produced eleven identical bodies. The session's last assistant line answers
    the question the banner actually raises, and it is the same fact the
    sidebar row already shows.
    """
    from local_operator.tui.notify import BODY_BACKGROUND

    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Article-search-svc schema review")
    _with_assistant_reply(background, "The schema review is done: two indexes are redundant.")
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        title, body, session_id, _subtitle = spawned[0]
        assert body == "The schema review is done: two indexes are redundant."
        # The routing sentence is GONE from the default frame, not merely
        # joined: it is the fallback now, and a body carrying both would spend
        # the content line on the thing that made this uninformative.
        assert BODY_BACKGROUND not in body
        # The title still identifies WHICH session, which is what carries the
        # routing fact now that the body carries content (D5 is not reopened by
        # restating it beside the snippet).
        assert title == "Article-search-svc schema review"
        assert session_id == "bg0000000001"


@pytest.mark.asyncio
async def test_the_last_reply_wins_over_earlier_ones(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """A snippet of the FINAL message, not the first — the preview reads the tail."""
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Long conversation")
    _with_assistant_reply(background, "First pass: I am still investigating.")
    _with_assistant_reply(background, "Final answer: the leak was the unclosed cursor.")
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        body = spawned[0][1]
        assert body == "Final answer: the leak was the unclosed cursor."
        assert "still investigating" not in body


@pytest.mark.asyncio
async def test_the_snippet_is_kept_off_the_banner_by_the_privacy_flag(
    store_root: Path, spawned: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`display.notification_session_name` off means NO session text at all.

    The flag exists to keep model-written session text off a screen other
    people can see, and a transcript snippet is strictly more session-derived
    than the name it was written for: a name is a topic, a snippet is content.
    A gate that covered only the title would leak the conversation itself into
    the frame directly beneath the brand the opt-out substitutes — the same
    review round 1 M2 shape, where a flag governing only some banners makes its
    own settings copy false.
    """
    from local_operator.tui.notify import APP_NAME, BODY_BACKGROUND

    monkeypatch.setattr("local_operator.tui.notify.session_names_in_notifications", lambda: False)
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Secret client migration")
    _with_assistant_reply(background, "Merged the acquisition due-diligence data room export.")
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        title, body, _session_id, _subtitle = spawned[0]
        assert body == BODY_BACKGROUND
        assert title == APP_NAME
        # Not one word of the conversation reaches the frame.
        argv = " ".join(spawned[0])
        assert "acquisition" not in argv
        assert "due-diligence" not in argv
        assert "Secret client migration" not in argv


@pytest.mark.asyncio
async def test_a_tool_only_final_turn_falls_back_to_the_neutral_sentence(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """A turn that only made tool calls SAID nothing, so there is nothing to quote.

    ``session_preview`` skips a text-free assistant entry by contract; what is
    pinned here is that the observer path renders the neutral sentence for it
    rather than a banner with an empty content line.
    """
    from local_operator.tui.notify import BODY_BACKGROUND

    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Tool only run")
    with (background / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "a1",
                    "ts": 2,
                    "type": "message",
                    "payload": {"kind": "message", "role": "assistant", "content": []},
                }
            )
            + "\n"
        )
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        assert spawned[0][1] == BODY_BACKGROUND


@pytest.mark.asyncio
async def test_an_unreadable_transcript_still_delivers_the_banner(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """A snippet read must never cost the notification (it runs in the 1 s poll).

    The body is chrome; delivery is not. An unreadable transcript degrades to
    the neutral sentence and the toast still goes out with its title, subtitle
    and click-through intact.
    """
    from local_operator.tui.notify import BODY_BACKGROUND

    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Unreadable session")
    _with_assistant_reply(background, "should never be read")
    (background / "transcript.jsonl").chmod(0o000)
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await _booted(app, pilot)
            await _settle(app, pilot)
            assert len(spawned) == 1, spawned
            title, body, session_id, _subtitle = spawned[0]
            assert body == BODY_BACKGROUND
            assert title == "Unreadable session"
            assert session_id == "bg0000000001"
    finally:
        # Restored so the tmp tree can be cleaned up on every platform.
        (background / "transcript.jsonl").chmod(0o644)


@pytest.mark.asyncio
async def test_a_reply_carrying_escape_sequences_is_sanitised(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """The snippet crosses the same wires as the title, so it gets the same scrub.

    The text is MODEL-WRITTEN and reaches an argv (cmux, the signed bundle,
    ``notify-send``) and, on the ``osascript`` leg, an AppleScript string
    literal — the wires ``sanitize_text`` exists for on this path (D16). It
    does NOT reach an OSC escape: the only OSC emitter passes a fixed
    ``BODIES`` constant, never this text (review round 1, m2).
    """
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Escape session")
    _with_assistant_reply(background, "done\x1b]0;pwned\x07 and\nwrapped\ttoo")
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        body = spawned[0][1]
        assert "\x1b" not in body
        assert "\x07" not in body
        # Collapsed to one line as well: a banner body is a line, not a block.
        assert "\n" not in body and "\t" not in body
        assert body == "done ]0;pwned and wrapped too"


@pytest.mark.asyncio
async def test_a_long_reply_is_cut_to_the_banner_budget(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """A banner is read in about a second; an unbounded paragraph is its own defect.

    Nothing on the wire forces this — Notification Centre WRAPS a body rather
    than clipping it, and both argv legs take it as one element — so the bound
    is chosen for the reader and has to be asserted, or it silently stops
    applying the day someone passes the text through unbudgeted.
    """
    from local_operator.tui.notify import BACKGROUND_SNIPPET_MAX_CHARS

    reply = (
        "I finished the migration audit across all seventeen tenant schemas and found "
        "three tables still carrying the legacy tenant_id column, which the backfill "
        "job skipped because their names do not match the prefix filter it uses."
    )
    # Comfortably past the budget, so the case cannot go vacuous if the number
    # is nudged: a reply that only just exceeds it would pass against a bound
    # that had been quietly raised.
    assert len(reply) > BACKGROUND_SNIPPET_MAX_CHARS + 80, "the case must exceed the budget"

    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Long reply session")
    _with_assistant_reply(background, reply)
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        body = spawned[0][1]
        assert len(body) <= BACKGROUND_SNIPPET_MAX_CHARS
        assert body.startswith("I finished the migration audit")
        # Ellipsised on a word boundary, so the cut reads as a truncation
        # rather than as the model having stopped mid-word.
        assert body.endswith("…")
        assert not body[:-1].endswith(" ")


#: A last assistant line that SOUNDS like success, which is what the snippet
#: path returns for a session that failed after it: `session_preview` filters to
#: `role == "assistant"`, and a runtime failure is never an assistant message.
#: Deliberately a sentence a reader would act on, so a regression that reopens
#: M1/D1 fails here as the contradiction a user would actually see rather than
#: as an abstract string mismatch.
_SUCCESS_SOUNDING_REPLY = "All 412 tests pass. The migration is complete."


@pytest.mark.parametrize(
    ("kind", "expected_body"),
    [
        ("error", "Stopped with an error"),
        ("interrupted", "Stopped before finishing"),
    ],
)
@pytest.mark.asyncio
async def test_a_non_complete_session_says_its_state_rather_than_its_last_reply(
    store_root: Path, spawned: list[list[str]], kind: str, expected_body: str
) -> None:
    """The snippet is only true of a session that COMPLETED (review M1, design D1).

    The failure that produced `error`, and the abort that produced
    `interrupted`, are runtime facts that write no assistant message — so the
    last assistant line is always PRE-failure text, and a session whose previous
    turn went well hands the banner a success sentence to print under a
    "Needs attention" subtitle. The two content lines of the frame would assert
    opposite things.

    `interrupted` is gated alongside `error` even though design round 1 would
    have accepted restricting it to `error`: whether a mid-work last line reads
    honestly is a property of the TEXT, and the fixture below is an interrupted
    session whose last line closes a sub-task. Nothing in the kind separates
    that from the coherent case, so the rule is decided by the kind alone.

    Asserts the COMPOSED body, not that some substring is absent: what shipped
    without this coverage was a body nobody had looked at for these kinds.
    """
    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Nightly index rebuild")
    _with_assistant_reply(background, _SUCCESS_SOUNDING_REPLY)
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", kind)

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        title, body, _session_id, subtitle = spawned[0]
        assert body == expected_body
        # The transcript is not merely absent from the body — no word of it
        # reaches ANY field, so the assertion cannot be satisfied by the snippet
        # having moved to the subtitle or the title.
        assert "migration" not in " ".join(spawned[0]).lower()
        # The state vocabulary agrees across the frame: the subtitle is this
        # kind's category and the body is this kind's sentence, which is what
        # "no line contradicts another" means here.
        from local_operator.tui.notify import CONTEXTS

        assert subtitle == CONTEXTS[kind]
        # The title still identifies WHICH session; gating the body does not
        # cost the routing cue (D2/D5).
        assert title == "Nightly index rebuild"


@pytest.mark.asyncio
async def test_a_completed_session_still_carries_its_last_reply(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """The other side of the gate: `complete` is unaffected by it.

    Same transcript as the non-complete cases above, so the pair isolates the
    KIND as the only variable — the snippet reaching the body on `complete` and
    not on the others is then a property of the gate rather than of the fixture.
    """
    from local_operator.tui.notify import BODIES, CONTEXT_COMPLETE

    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Nightly index rebuild")
    _with_assistant_reply(background, _SUCCESS_SOUNDING_REPLY)
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        _title, body, _session_id, subtitle = spawned[0]
        assert body == _SUCCESS_SOUNDING_REPLY
        assert subtitle == CONTEXT_COMPLETE
        # `complete` takes the snippet, never the house sentence — that constant
        # is what the OTHER kinds say, and swapping them would make every
        # completed banner say "Task complete" again (D5's tautology).
        assert body != BODIES["complete"]


@pytest.mark.asyncio
async def test_a_failed_session_with_no_transcript_still_says_it_failed(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """The state sentence does not depend on a readable transcript.

    `error` is the kind most likely to have nothing to read — a session that
    died early may never have written an assistant turn at all — so the gate
    must resolve before the tail read rather than after it. If the body were
    computed first and corrected afterwards, this case would degrade to the
    neutral routing sentence and lose the one fact worth stating.
    """
    from local_operator.tui.notify import BODY_BACKGROUND, BODY_ERROR

    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Died on startup")
    # No assistant turn was ever written: only the seeded user message.
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(background), str(uuid.uuid4()), "fresh", "error")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        body = spawned[0][1]
        assert body == BODY_ERROR
        assert body != BODY_BACKGROUND


@pytest.mark.asyncio
async def test_the_privacy_opt_out_covers_every_completion_kind(
    store_root: Path, spawned: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state sentence is house prose, so the opt-out neither adds nor removes it.

    `BODIES[kind]` is a fixed constant carrying nothing session-derived, so a
    user who opted out of session text is entitled to see it — the flag's
    promise is about MODEL-WRITTEN text, and gating the state sentence too would
    make the opt-out a strictly worse banner for no privacy gain. The completed
    case in the same run is what shows the flag is still doing its job.
    """
    from local_operator.tui.notify import APP_NAME, BODY_BACKGROUND, BODY_INTERRUPTED

    monkeypatch.setattr("local_operator.tui.notify.session_names_in_notifications", lambda: False)
    _make_session(store_root, "current", "Current conversation")
    halted = _make_session(store_root, "bg0000000001", "Halted rebuild")
    _with_assistant_reply(halted, _SUCCESS_SOUNDING_REPLY)
    finished = _make_session(store_root, "bg0000000002", "Finished rebuild")
    _with_assistant_reply(finished, _SUCCESS_SOUNDING_REPLY)
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(halted), str(uuid.uuid4()), "fresh", "interrupted")
    store.publish(conversation_identity(finished), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 2, spawned
        bodies = {call[3]: call[1] for call in spawned}
        assert bodies["Interrupted"] == BODY_INTERRUPTED
        # The completed one loses its snippet to the flag, which is the gate
        # still working: no word of the transcript survives in any field.
        assert bodies["Complete"] == BODY_BACKGROUND
        assert all(call[0] == APP_NAME for call in spawned), spawned
        assert "migration" not in " ".join(" ".join(call) for call in spawned).lower()


def _desktop_presence(root: Path, *, kinds: list[str] | None = None):
    """A live desktop delivery lease in ``root``, through the production writer.

    Real rather than doubled, because the thing under test is that the TUI's
    announcer reads the SAME machine-wide artifact every other surface does —
    a monkeypatched predicate would pass with the read wired to nothing.
    """
    from local_operator.server.utils.desktop_presence import DesktopDeliveryPublisher
    from local_operator.session.runtime.presence import reset_cache

    publisher = DesktopDeliveryPublisher(root)
    publisher.update(
        "sub-1",
        can_notify=True,
        can_notify_kinds=["complete", "error"] if kinds is None else kinds,
        window={"exists": True, "focused": True, "visible": True, "minimized": False},
    )
    reset_cache()
    return publisher


def _established(store_root: Path, session_id: str, name: str) -> Any:
    """A background session with a prior, already-read completion.

    The baseline matters: an upgrading store announces nothing on its first
    tick, so a test that skipped this would prove nothing about the rung.
    """
    directory = _make_session(store_root, session_id, name)
    store = AttentionStore(store_root / "attention.db")
    identity = conversation_identity(directory)
    first = str(uuid.uuid4())
    store.publish(identity, first, "old", "complete")
    store.acknowledge(identity, first)
    return store, identity


@pytest.mark.asyncio
async def test_a_notify_capable_desktop_defers_the_tui_announcer(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """RUNG 2 ABOVE RUNG 3.

    The desktop composes this completion from the machine-wide feed, so the TUI
    announcing it as well is the duplicate one rung down — and on a machine with
    both apps the winner would otherwise be whichever polls faster.
    """
    _make_session(store_root, "current", "Current conversation")
    store, identity = _established(store_root, "bg0000000002", "Background review")

    from local_operator.session.runtime.presence import reset_cache

    app = OperatorApp(lambda: _factory(AttachedSession()))
    publisher = _desktop_presence(store_root)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await _booted(app, pilot)
            await _settle(app, pilot)
            spawned.clear()

            store.publish(identity, str(uuid.uuid4()), "fresh", "complete")
            await _settle(app, pilot)

            assert spawned == [], spawned
    finally:
        publisher.close()
        reset_cache()


@pytest.mark.asyncio
async def test_a_kind_the_desktop_cannot_deliver_is_still_announced_here(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """The presence is narrowed by KIND, and this is why.

    The machine-wide feed carries completions only, so a lease claiming
    ``complete``/``error`` must not silence an ``interrupted`` row — the desktop
    would never banner it, and the completion would reach nobody.
    """
    _make_session(store_root, "current", "Current conversation")
    store, identity = _established(store_root, "bg0000000003", "Interrupted review")

    from local_operator.session.runtime.presence import reset_cache

    app = OperatorApp(lambda: _factory(AttachedSession()))
    publisher = _desktop_presence(store_root, kinds=["complete", "error"])
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await _booted(app, pilot)
            await _settle(app, pilot)
            spawned.clear()

            store.publish(identity, str(uuid.uuid4()), "fresh", "interrupted")
            await _settle(app, pilot)

            assert len(spawned) == 1, spawned
    finally:
        publisher.close()
        reset_cache()


def test_the_tui_cap_and_the_feed_cap_are_the_same_promise() -> None:
    """Two transports, one ceiling — asserted rather than asserted-in-prose.

    The TUI caps per tick and digests the remainder locally; the feed does the
    same for the frames the desktop renders. Two hand-maintained 3s are one edit
    away from disagreeing about the promise the user actually experiences, so
    the agreement is a test rather than a convention.
    """
    from local_operator.server.utils.desktop_feed import BURST_LIMIT
    from local_operator.tui import app as app_module

    assert app_module._BACKGROUND_NOTIFY_MAX_PER_TICK == BURST_LIMIT


def _stop_reason(rung: str, *, command: str) -> str:
    """The durable sentence a deliberate stop of ``rung`` carries, from the code."""
    from local_operator.incidents import (
        DELIBERATE_CUT_OFF_CAUSE,
        render_cut_off_reason,
        render_stop_attribution,
    )

    return render_cut_off_reason(
        DELIBERATE_CUT_OFF_CAUSE,
        detail=render_stop_attribution(rung=rung, command=command, killer_pid=40609),
    )


@pytest.mark.asyncio
async def test_an_escalated_stop_names_its_rung_on_the_banner(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """Design round 1, D1 on the lock screen: the banner said only "stopped early".

    The fixed body was kind-gated, so the banner for a rung-3 kill and the banner
    for a rung-1 request were the same ten words — on the surface the operator
    reads while looking at something else entirely. Only the ESCALATED rung
    appends the attributed phrase, because ``Interrupted`` already means the user
    stopped it and the interesting fact is that the ladder had to signal.
    """
    from local_operator.tui.notify import BODY_INTERRUPTED

    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Halted midway")
    store = AttentionStore(store_root / "attention.db")
    store.publish(
        conversation_identity(background),
        str(uuid.uuid4()),
        "fresh",
        "interrupted",
        reason=_stop_reason("sigkill", command="/stop --all"),
        cause="user-stop",
    )

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        assert spawned[0][1] == f"{BODY_INTERRUPTED} — killed by /stop --all", spawned[0]


@pytest.mark.asyncio
async def test_a_plain_request_keeps_the_banners_established_sentence(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """The other half of D1's fix: rung 1 must render byte-identically to today.

    A stop that exited on request is what the sentence already describes, so
    appending its own attribution would spend the banner's one content line
    saying ``Stopped before finishing — stopped on request by /stop``.
    """
    from local_operator.tui.notify import BODY_INTERRUPTED

    _make_session(store_root, "current", "Current conversation")
    background = _make_session(store_root, "bg0000000001", "Halted midway")
    store = AttentionStore(store_root / "attention.db")
    store.publish(
        conversation_identity(background),
        str(uuid.uuid4()),
        "fresh",
        "interrupted",
        reason=_stop_reason("socket", command="/stop"),
        cause="user-stop",
    )

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        assert len(spawned) == 1, spawned
        assert spawned[0][1] == BODY_INTERRUPTED, spawned[0]


@pytest.mark.asyncio
async def test_a_machine_started_run_is_never_announced_and_a_workstream_is(
    store_root: Path, spawned: list[list[str]]
) -> None:
    """This surface already filtered by ORIGIN — pinned, and pinned in BOTH directions.

    ``_notify_background_completions`` builds its candidate set from
    ``load_catalog``, which scans through ``resume._scan_sessions`` and keeps only
    the user's own origins, so an ``agent-shell`` run has never reached this rung
    and must not start now. What the workstream origin changes is the OTHER row:
    it is a user origin, so the operator's own TUI announces it exactly as it
    announces a conversation he opened — which is what makes a fan-out the
    operator asked for visible on the surface he is actually looking at.

    BOTH ARMS IN ONE CELL, so the control is the same run and the same scan: the
    two sessions differ only by their marker. The hidden arm's watermark is
    asserted too, because "no banner" for the wrong reason (a claim someone else
    spent) is not the property under test.
    """
    from local_operator.resume import (
        ORIGIN_AGENT_SHELL,
        ORIGIN_AGENT_WORKSTREAM,
        mark_session_origin,
    )

    _make_session(store_root, "current", "Current conversation")
    delegated = _make_session(store_root, "d00000000001", "Delegated review run")
    mark_session_origin(delegated, ORIGIN_AGENT_SHELL)
    workstream = _make_session(store_root, "w00000000001", "Fan-out audit")
    mark_session_origin(workstream, ORIGIN_AGENT_WORKSTREAM, opened_by={"session": "manager00001"})

    store = AttentionStore(store_root / "attention.db")
    delegated_identity = conversation_identity(delegated)
    workstream_identity = conversation_identity(workstream)
    # An ESTABLISHED store: both carry a prior, already-read completion, so the
    # baseline is installed and the next publish is news on both identities.
    for identity in (delegated_identity, workstream_identity):
        seed = str(uuid.uuid4())
        store.publish(identity, seed, "old", "complete")
        store.acknowledge(identity, seed)

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot)
        spawned.clear()

        delegated_token = str(uuid.uuid4())
        store.publish(delegated_identity, delegated_token, "delegated-completion", "complete")
        store.publish(workstream_identity, str(uuid.uuid4()), "workstream-completion", "complete")
        await _settle(app, pilot, rounds=10)

        assert len(spawned) == 1, spawned
        argv = " ".join(spawned[0])
        assert "w00000000001" in argv
        assert "Fan-out audit" in argv
        assert "Delegated review run" not in argv

    # The skip is a FILTER ABOVE THE CLAIM, exactly as it is for a test-hosted
    # row above: the hidden completion is still claimable, so a surface that may
    # legitimately announce it — the desktop app on a host where notifications
    # are on — still can. A spent watermark here would read as "somebody was
    # told" about a banner nobody ever received.
    assert store.claim_delivery(delegated_identity, delegated_token, "probe") is True

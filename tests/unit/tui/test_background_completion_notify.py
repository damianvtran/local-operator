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


def _make_session(root: Path, session_id: str, name: str) -> Path:
    """A session directory the catalog can actually scan and name.

    Uses ``write_session_title`` rather than hand-writing a sidecar, so the name
    resolves through exactly the path the picker and sidebar use.
    """
    from local_operator.resume import write_session_title

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id":"e1","ts":1,"type":"message",'
        '"payload":{"kind":"message","role":"user","content":[{"text":"go"}]}}\n'
    )
    write_session_title(directory, name, user_set=False, past_names=[])
    return directory


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


async def _settle(app: OperatorApp, pilot: Any, rounds: int = 6) -> None:
    """Run the app's own attention poll to completion, several times."""
    for _ in range(rounds):
        await app._poll_completion_attention()
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
    """
    current = _make_session(store_root, "current", "Current conversation")
    store = AttentionStore(store_root / "attention.db")
    store.publish(conversation_identity(current), str(uuid.uuid4()), "fresh", "complete")

    app = OperatorApp(lambda: _factory(AttachedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot)
        await _settle(app, pilot, rounds=8)
        assert spawned == []


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

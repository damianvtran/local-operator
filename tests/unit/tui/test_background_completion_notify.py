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

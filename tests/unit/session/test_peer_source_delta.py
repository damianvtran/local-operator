"""Commit B: the facade's retained source domain and the recovery delta's family.

THE CONTRACT THIS PINS. The recovery delta (``HistoryDeltaEvent``) is where a
reconnecting viewer learns the rows it missed. Commit B lets such a delta carry
the complete-source family (``journal_generation`` + ``journal_ordinals`` +
``entry_times``) — but ONLY when the facts are true of EVERY row: the attachment
declared the complete peer-source contract, and every row was owner-attested in
the retained domain observed on THIS attachment. There is no per-row binding
beyond the wake's own attestations, so a partial map under a packet-level claim
would over-release rows the owner never attested; the all-or-nothing rule is the
safety property, not a nicety.

WHAT IS REAL HERE. The socket cases drive a REAL owner (in-process
``RuntimeServer``) emitting REAL ``journal_order_ready`` wake bytes over a REAL
attach socket; the wake's shape and its retention are asserted from those bytes.
The delta cases drive the real ``_load_frontend_history`` construction (the same
entry point ``bind_runtime`` uses, with the cold-gap trigger the bind sets), so
the family's attachment point is exercised where it actually runs. The one
synthetic piece is LABELED at its definition: the declaring client — the real
remote client's flag, set on a real local client, standing in for the peer dial
that production writes it from (pinned in
``tests/unit/network/test_peer_journal_order.py::test_remote_attach_declares_complete_order_only_when_negotiated``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import HistoryDeltaEvent, Message
from local_operator.session.attached import (
    SOURCE_ORDINALS_CAP,
    AttachedSession,
    deserialize_event,
)
from local_operator.session.owner import LocalOwner
from tests.unit.session.test_attached_entry_times import _frontend, _window
from tests.unit.session.test_cold_bind_gap import _never_take_over, _serve_owner

pytestmark = pytest.mark.asyncio


#: The wake shape the REAL owner emits (captured live in
#: ``test_real_owner_wakes_seed_only_a_declaring_attachment`` below and asserted
#: key-exact there): ``{type, journal_generation, rows:[{id, journal_seq, ...}]}``
#: with a ``{type, reconcile: True}`` variant. Kept byte-shaped like the wire
#: dict a facade really decodes, never a model-and-re-dump.
def _wake(
    generation: str, rows: list[tuple[str, int]], *, reconcile: bool = False
) -> dict[str, Any]:
    if reconcile:
        return {"type": "journal_order_ready", "reconcile": True}
    return {
        "type": "journal_order_ready",
        "journal_generation": generation,
        "rows": [{"id": id_, "journal_seq": seq} for id_, seq in rows],
    }


class _ClientStub:
    """The one flag and the one teardown call the facade makes on a client."""

    def close(self) -> None:
        """The facade's teardown calls ``close``; a no-op keeps go-cold driveable."""


class _DeclaredClient(_ClientStub):
    """A client that declared the complete contract (the flag the facade reads)."""

    peer_journal_source_declared = True


class _LegacyClient(_ClientStub):
    """A client that never declared it — the plain local build's shape."""

    peer_journal_source_declared = False


def _viewer(tmp_path: Path, *, declared: bool = True) -> AttachedSession:
    viewer = AttachedSession(
        config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over
    )
    viewer._client = _DeclaredClient() if declared else _LegacyClient()  # type: ignore[assignment]
    return viewer


def _extra(event: HistoryDeltaEvent) -> dict[str, Any]:
    """The event's extra fields, never None (pydantic types the map Optional)."""
    return dict(event.model_extra or {})


# ---------------------------------------------------------------------------
# Retention: what the wake seeds, and what it must never do
# ---------------------------------------------------------------------------


async def test_retention_merges_within_a_generation_and_replaces_across(tmp_path: Path) -> None:
    viewer = _viewer(tmp_path)
    viewer._note_source_wake(deserialize_event(_wake("g1", [("a", 1)])))
    viewer._note_source_wake(deserialize_event(_wake("g1", [("b", 2)])))
    assert viewer._source_ordinals == {"a": 1, "b": 2}
    assert viewer._source_domain is not None and viewer._source_domain[0] == "g1"
    # A DIFFERENT generation is the owner having rotated: the old ids' lineage
    # is disowned atomically (never the old map under the new name).
    viewer._note_source_wake(deserialize_event(_wake("g2", [("c", 3)])))
    assert viewer._source_ordinals == {"c": 3}
    assert viewer._source_domain is not None and viewer._source_domain[0] == "g2"


async def test_reconcile_invalidates_and_only_the_family_type_is_retained(tmp_path: Path) -> None:
    viewer = _viewer(tmp_path)
    viewer._note_source_wake(deserialize_event(_wake("g1", [("a", 1)])))
    viewer._note_source_wake(deserialize_event(_wake("g1", [], reconcile=True)))
    assert viewer._source_domain is None and viewer._source_ordinals == {}
    # A non-wake event is not retention material, whatever its extras carry.
    other = deserialize_event({"type": "agent_start", "journal_generation": "g9"})
    viewer._note_source_wake(other)
    assert viewer._source_domain is None and viewer._source_ordinals == {}


async def test_retention_requires_the_complete_declaration(tmp_path: Path) -> None:
    legacy = _viewer(tmp_path, declared=False)
    legacy._note_source_wake(deserialize_event(_wake("g1", [("a", 1)])))
    assert legacy._source_domain is None and legacy._source_ordinals == {}
    # And a facade with no client at all (cold, never dialed) reads False too.
    unattached = AttachedSession(
        config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over
    )
    unattached._note_source_wake(deserialize_event(_wake("g1", [("a", 1)])))
    assert unattached._source_domain is None


async def test_overflow_invalidates_rather_than_growing(tmp_path: Path) -> None:
    viewer = _viewer(tmp_path)
    viewer._note_source_wake(
        deserialize_event(_wake("g1", [(f"id-{i}", i) for i in range(SOURCE_ORDINALS_CAP + 1)]))
    )
    # A silently PARTIAL map under a packet-level claim would over-release
    # rows, so the bound invalidates the whole domain instead.
    assert viewer._source_domain is None and viewer._source_ordinals == {}


async def test_a_wake_is_read_never_rewritten_and_never_emits(tmp_path: Path) -> None:
    viewer = _viewer(tmp_path)
    events: list[Any] = []
    viewer.subscribe(events.append)
    event = deserialize_event(_wake("g1", [("a", 1)]))
    before = event.model_dump()
    viewer._note_source_wake(event)
    assert event.model_dump() == before, "retention must not mutate the passed event"
    assert viewer._source_ordinals == {"a": 1}
    assert events == [], "retention is passive: it must not publish anything"
    # Through the real wire path the wake still rides the ordinary buffer,
    # exactly as it did before commit B — retention adds nothing to delivery.
    viewer._on_wire_event(_wake("g1", [("b", 2)]))
    assert events == []
    assert [e.type for e in viewer._buffered_events] == ["journal_order_ready"]


# ---------------------------------------------------------------------------
# The delta's family: all-or-nothing, entry-times subset, byte-legacy omission
# ---------------------------------------------------------------------------


async def test_history_delta_attaches_only_when_every_row_is_attested(tmp_path: Path) -> None:
    viewer = _viewer(tmp_path)
    first, second, third = Message.user("one"), Message.assistant("two"), Message.user("three")
    viewer._note_source_wake(deserialize_event(_wake("g1", [(first.id, 5), (second.id, 6)])))
    viewer._entry_times = {first.id: 10.5}  # second unattested stays absent

    event = viewer._history_delta([first, second])
    assert _extra(event).get("journal_generation") == "g1"
    assert _extra(event).get("journal_ordinals") == {first.id: 5, second.id: 6}
    # The entry-times SUBSET: owner-stamped rows only; absent stays absent.
    assert _extra(event).get("entry_times") == {first.id: 10.5}

    # ONE uncovered row omits the WHOLE family — never a partial claim.
    event = viewer._history_delta([first, second, third])
    assert not _extra(event), "an unattested row must omit every family key"

    # An empty delta and an id-less row omit too (nothing to claim).
    assert not _extra(viewer._history_delta([]))
    idless = Message.assistant("no id")
    idless.id = ""
    assert not _extra(viewer._history_delta([idless]))


async def test_history_delta_without_entry_stamps_still_carries_the_domain(
    tmp_path: Path,
) -> None:
    viewer = _viewer(tmp_path)
    first = Message.user("one")
    viewer._note_source_wake(deserialize_event(_wake("g1", [(first.id, 5)])))
    event = viewer._history_delta([first])
    assert _extra(event).get("journal_generation") == "g1"
    assert "entry_times" not in _extra(event), "no stamp may be synthesized"


async def test_history_delta_omission_is_byte_legacy(tmp_path: Path) -> None:
    """The omitted delta is EXACTLY the pre-commit-B event, key for key."""
    viewer = _viewer(tmp_path, declared=False)
    first = Message.user("one")
    omitted = viewer._history_delta([first], reset=True)
    legacy = HistoryDeltaEvent(messages=[first], reset=True)
    assert omitted.model_dump() == legacy.model_dump()


async def test_history_delta_omits_after_a_client_swap(tmp_path: Path) -> None:
    """The identity guard: a swapped client can never inherit the domain."""
    viewer = _viewer(tmp_path)
    first = Message.user("one")
    viewer._note_source_wake(deserialize_event(_wake("g1", [(first.id, 5)])))
    assert _extra(viewer._history_delta([first])).get("journal_generation") == "g1"
    viewer._client = _DeclaredClient()  # type: ignore[assignment] — a NEW client
    assert not _extra(viewer._history_delta([first]))


async def test_generation_change_between_retention_and_delta_omits_or_retells(
    tmp_path: Path,
) -> None:
    viewer = _viewer(tmp_path)
    first, second = Message.user("one"), Message.assistant("two")
    viewer._note_source_wake(deserialize_event(_wake("g1", [(first.id, 1)])))
    viewer._note_source_wake(deserialize_event(_wake("g2", [(second.id, 2)])))
    # The old id's lineage was disowned with g1: its row can never ride a g2 claim.
    assert not _extra(viewer._history_delta([first]))
    # A row attested in the CURRENT domain rides it.
    event = viewer._history_delta([second])
    assert _extra(event).get("journal_generation") == "g2"
    assert _extra(event).get("journal_ordinals") == {second.id: 2}


async def test_go_cold_disposes_the_retained_domain(tmp_path: Path) -> None:
    viewer = _viewer(tmp_path, declared=False)
    viewer._client = _DeclaredClient()  # type: ignore[assignment]
    viewer._note_source_wake(deserialize_event(_wake("g1", [("a", 1)])))
    assert viewer._source_domain is not None
    viewer._go_cold()
    assert viewer._source_domain is None and viewer._source_ordinals == {}


# ---------------------------------------------------------------------------
# The real socket: real wake bytes, real decode, real gate
# ---------------------------------------------------------------------------


class _Turns:
    """One scripted turn per call, then nothing (later commits starve safely)."""

    def __init__(self, turns: list[list[Any]]) -> None:
        self.turns = turns
        self.calls = 0

    def __call__(self, request: Any, signal: Any = None) -> Any:
        if self.calls >= len(self.turns):
            return None
        turn = self.turns[self.calls]
        self.calls += 1

        async def gen() -> Any:
            for event in turn:
                yield event

        return gen()


class _DeclaringLocalOwner(LocalOwner):
    """LABELED SYNTHETIC: the local owner, with the peer client's declaration.

    Everything here is the production local path over a REAL socket; the ONE
    difference is the flag the real ``RemoteSessionClient`` writes in its own
    connect gate. It exists so the facade half of the gate (this commit) can be
    exercised against REAL wake bytes without standing a second device up.
    """

    def make_client(self, on_projection: Any, on_disconnected: Any, **kwargs: Any) -> Any:
        client = super().make_client(on_projection, on_disconnected, **kwargs)
        setattr(client, "peer_journal_source_declared", True)
        return client


async def _wait_until(predicate: Any, timeout: float) -> bool:
    """Bounded poll; False on timeout (the caller decides what that means)."""

    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.02)

    try:
        await asyncio.wait_for(poll(), timeout=timeout)
        return True
    except TimeoutError:
        return False


async def _commit_until_a_rows_wake(session: Any, wakes: list[dict[str, Any]]) -> bool:
    """Drive real commit waves until the owner emits a real ``rows`` wake.

    THE OWNER'S EMISSION IS NOT DETERMINISTIC PER COMMIT, by its own design: a
    wave whose single-flight refresh read a snapshot older than the append
    emits ``reconcile`` and drops that id (``serving.py::emit_order``). So the
    test commits in a bounded loop until real rows bytes arrive — the bytes are
    real; the NUMBER of waves it takes is the owner's business, not the test's.
    """
    for attempt in range(3):
        await session.prompt(f"commit wave {attempt}")
        if await _wait_until(lambda: any(w.get("rows") for w in wakes), 10.0):
            return True
    return False


async def _wire_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, declared: bool
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """A real owner behind a real socket, and a viewer capturing its wake bytes."""
    from tests.e2e.harness import build_session, seed_transcript, text_turn

    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    directory = config / "sessions" / "peersrc"
    await seed_transcript(directory, [Message.user("seed one"), Message.assistant("seed two")])
    stream = _Turns([text_turn(f"committed wave {i}") for i in range(3)])
    session = build_session(directory, stream, cwd=tmp_path)
    _handle, _server = await _serve_owner(directory, session, tmp_path, monkeypatch)

    owner = _DeclaringLocalOwner(config, directory.name) if declared else None
    viewer = await AttachedSession.cold(
        directory.name,
        config_dir=config,
        cwd=str(tmp_path),
        takeover_factory=_never_take_over,
        owner=owner,
    )
    wakes: list[dict[str, Any]] = []
    original = viewer._on_wire_event

    def capture(data: Any) -> Any:
        if isinstance(data, dict) and data.get("type") == "journal_order_ready":
            wakes.append(data)
        return original(data)

    viewer._on_wire_event = capture
    return viewer, session, wakes


async def test_real_owner_wakes_seed_only_a_declaring_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real wire bytes: shape pinned, retention gated, the local path inert."""
    viewer, session, wakes = await _wire_viewer(tmp_path, monkeypatch, declared=True)
    try:
        await viewer.bind_runtime()
        assert await _commit_until_a_rows_wake(
            session, wakes
        ), "the real owner never emitted a rows wake in three waves"
        row_wake = next(w for w in wakes if w.get("rows"))
        # The REAL byte shape commit B reads (docs/DESKTOP_API.md invariant).
        assert set(row_wake) == {"type", "journal_generation", "rows"}
        assert isinstance(row_wake["journal_generation"], str) and row_wake["journal_generation"]
        assert set(row_wake["rows"][0]) >= {"id", "journal_seq"}

        # The live declaring attachment observes the real bytes; a later
        # reconcile wake legitimately disposes the domain (the owner disowned
        # the lineage), so the retained-state claim is conditional on the tail.
        row_wakes = [w for w in wakes if w.get("rows")]
        assert row_wakes
        tail = wakes[wakes.index(row_wakes[-1]) :]
        if not any(w.get("reconcile") for w in tail):
            retained = dict(viewer._source_ordinals)
            assert retained, "a declaring attachment retains observed wake facts"
            assert (
                retained.items()
                <= {r["id"]: r["journal_seq"] for w in row_wakes for r in w["rows"]}.items()
            )
            assert viewer._source_domain is not None
            assert viewer._source_domain[0] == row_wakes[-1]["journal_generation"]

        # Controlled re-injection of the REAL bytes through the same receive
        # path a live frame takes: exactly the pairs retained, and a delta over
        # them rides the domain (the helper reads ids alone — id-tagged
        # stand-ins are the honest floor for the row objects).
        declared = _viewer(tmp_path)
        declared._on_wire_event(row_wake)
        pairs = {r["id"]: r["journal_seq"] for r in row_wake["rows"]}
        assert declared._source_ordinals == pairs
        rows = [Message.user("").model_copy(update={"id": id_}) for id_ in sorted(pairs)]
        delta = declared._history_delta(rows)
        assert _extra(delta).get("journal_generation") == row_wake["journal_generation"]
        assert _extra(delta).get("journal_ordinals") == pairs
        # And the same real bytes retain NOTHING on an undeclared attachment.
        undeclared = _viewer(tmp_path, declared=False)
        undeclared._on_wire_event(row_wake)
        assert undeclared._source_domain is None and undeclared._source_ordinals == {}

        # Disposal on the real teardown path.
        await viewer.dispose()
        assert viewer._source_domain is None and viewer._source_ordinals == {}
    finally:
        if not viewer._disposed:
            await viewer.dispose()


async def test_the_real_local_path_stays_inert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production local build (no declaration): wakes arrive, NOTHING is retained."""
    viewer, session, wakes = await _wire_viewer(tmp_path, monkeypatch, declared=False)
    try:
        await viewer.bind_runtime()
        assert await _commit_until_a_rows_wake(
            session, wakes
        ), "the owner still emits wakes; the FACADE decides to retain"
        assert viewer._source_domain is None and viewer._source_ordinals == {}
        attested = {r["id"] for w in wakes if w.get("rows") for r in w["rows"]}
        rows = [Message.user("").model_copy(update={"id": id_}) for id_ in sorted(attested)]
        assert not _extra(viewer._history_delta(rows))
    finally:
        if not viewer._disposed:
            await viewer.dispose()


async def test_the_bootstrap_delta_carries_the_exact_ordinals(tmp_path: Path) -> None:
    """The bootstrap ordering, driven through the real delta construction.

    The cold-gap trigger is what ``bind_runtime`` sets for a paint-first attach
    (``_cold_painted_ids``), driven directly here for determinism; the wake
    bytes were captured from the REAL owner in the socket tests above, and the
    delta itself is built by the real ``_load_frontend_history``. Ordering (i):
    the wake was observed before the delta — the family is present, ordinals
    exactly the attested rows. Ordering (ii): no wake ever arrived — the delta
    omits, with IDENTICAL row delivery.
    """
    first, second = Message.user("one"), Message.assistant("two")

    async def emit(viewer: AttachedSession) -> HistoryDeltaEvent:
        viewer._display_window_requested = True
        viewer._cold_painted_ids = set()  # the whole window is the "gap"
        await viewer._load_frontend_history(
            _frontend(_window(first, second, entry_times={first.id: 10.5}))
        )
        deltas = [e for e in viewer._buffered_events if isinstance(e, HistoryDeltaEvent)]
        assert len(deltas) == 1, viewer._buffered_events
        return deltas[0]

    before = _viewer(tmp_path)
    before._note_source_wake(deserialize_event(_wake("g1", [(first.id, 5), (second.id, 6)])))
    wake_first = await emit(before)
    assert _extra(wake_first).get("journal_generation") == "g1"
    assert _extra(wake_first).get("journal_ordinals") == {first.id: 5, second.id: 6}
    assert _extra(wake_first).get("entry_times") == {first.id: 10.5}
    assert wake_first.reset is False

    after = _viewer(tmp_path)
    wake_last = await emit(after)
    assert not _extra(wake_last), "a wake arriving later can never retro-attach"

    # Delivery identical either way: same rows, same reset, same order — the
    # family is a fast path, never a delivery difference.
    assert wake_first.reset is wake_last.reset is False
    assert wake_first.messages == wake_last.messages

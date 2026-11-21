"""Several admitted bodies served as ONE turn of K frames (``PendingTurnBatch``).

Driven against the REAL store, like ``test_pending_turn``: the contract is
about durable row transitions per member (claim, release, complete, deferred
never flipped to pending), which a double could not prove.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Any

from local_operator.harness.message_types import (
    BRIDGE_CHANNEL_PARTICIPATION_RULE,
    BRIDGE_RECEIVER_RULES,
)
from local_operator.team_bridge.pending_turn import FetchedBody, PendingTurnBatch
from local_operator.team_bridge.request_body import BodyBudget
from local_operator.team_bridge.settings import TeamBridgeSettings
from local_operator.team_bridge.store import (
    OwnerReceipt,
    OwnerSnapshot,
    Store,
    state_db_path,
)

from .test_pending_turn import _asst_calls, _tool_reply, _wire_pairing_failures

CONVERSATION_ID = "bcv_abcdefgh"
OTHER_CONVERSATION = "bcv_zzzzzzzz"
SESSION = "sess_cos"
FINGERPRINT = "aa11" * 16


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id="acc_1",
        tenant_id="3f2a1111-2222-4333-8444-555566667777",
        endpoint_id="ep_1",
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    )


def _store(tmp_path: Path) -> tuple[Store, int]:
    store = Store(tmp_path)
    store.commit_enrollment(
        _owner(), OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id=SESSION
    )
    epoch = store.publish_screening_policy(
        _owner(),
        version="v1",
        template_sha256="ab" * 32,
        default_model="qa-default",
        revision="rev-1",
    )
    return store, epoch


def _body(message_id: str) -> str:
    return f"body of {message_id}"


def _sha(message_id: str) -> str:
    return hashlib.sha256(_body(message_id).encode()).hexdigest()


def _mid(n: int) -> str:
    return f"bcmsg_burst{n:04d}"


def _reserve(
    store: Store,
    epoch: int,
    n: int,
    *,
    conversation_id: str = CONVERSATION_ID,
    wake_class: str | None = None,
    created_at: float | None = None,
) -> str:
    message_id = _mid(n)
    assert store.record_body_reservation(
        _owner(),
        conversation_id=conversation_id,
        message_id=message_id,
        conversation_seq=n,
        mention=False,
        body_sha256=_sha(message_id),
        screening_fingerprint=FINGERPRINT,
        policy_epoch=epoch,
        consent_version="bridge_consent_v1",
        designated_session=SESSION,
        now=created_at if created_at is not None else time.time() + n * 0.001,
        wake_class=wake_class,
    )
    return message_id


def _fetcher(*, bad: frozenset[str] = frozenset(), kind: str | None = "channel"):
    fetched: list[str] = []

    async def fetch(conversation_id: str, message_id: str) -> FetchedBody | None:
        fetched.append(message_id)
        sha = "00" * 32 if message_id in bad else _sha(message_id)
        return FetchedBody(
            body=_body(message_id),
            body_sha256=sha,
            conversation_label="#qa",
            sender_label="Ada",
            sender_kind="agent",
            conversation_kind=kind,
        )

    fetch.fetched = fetched  # type: ignore[attr-defined]
    return fetch


def _budget(size: int = 200_000) -> BodyBudget:
    return BodyBudget(available_bytes=size, measured_from="test budget")


def _state(tmp_path: Path, message_id: str) -> dict[str, Any]:
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM body_reservations WHERE message_id = ?", (message_id,)
        ).fetchone()
    assert row is not None
    return dict(row)


def _batch(tmp_path: Path, **kwargs: Any) -> PendingTurnBatch:
    return PendingTurnBatch(_owner(), SESSION, config_dir=tmp_path, **kwargs)


def _open(batch: PendingTurnBatch, fetch=None, budget: BodyBudget | None = None) -> bool:
    return asyncio.run(batch.open(budget=budget or _budget(), fetch=fetch or _fetcher()))


def _frames(shaped: list[Any]) -> list[tuple[int, str]]:
    return [
        (i, m.details["text"])
        for i, m in enumerate(shaped)
        if getattr(m, "custom_type", None) == "team_bridge_message"
    ]


def _message_ids(frames: list[tuple[int, str]]) -> list[str]:
    out = []
    for _, text in frames:
        line = next(x for x in text.splitlines() if x.startswith("message_id: "))
        out.append(line.removeprefix("message_id: "))
    return out


def test_a_burst_rides_one_request_as_contiguous_frames_in_seq_order(tmp_path) -> None:
    from local_operator.harness.types import Message

    store, epoch = _store(tmp_path)
    ids = [_reserve(store, epoch, n) for n in range(1, 6)]
    batch = _batch(tmp_path)
    assert _open(batch)
    shaped = batch.transform([Message.user("prior"), Message.user("wake card")])
    frames = _frames(shaped)
    assert [i for i, _ in frames] == [2, 3, 4, 5, 6]
    assert _message_ids(frames) == ids
    # Rules ONCE, on the first frame only; the rest start at VERIFIED SOURCE.
    texts = [t for _, t in frames]
    assert texts[0].startswith(BRIDGE_RECEIVER_RULES)
    assert BRIDGE_CHANNEL_PARTICIPATION_RULE in texts[0]
    assert all(BRIDGE_RECEIVER_RULES not in t for t in texts[1:])
    assert all(t.startswith("VERIFIED SOURCE") for t in texts[1:])
    assert sorted(batch.identities) == sorted((CONVERSATION_ID, i) for i in ids)
    assert all(_state(tmp_path, i)["state"] == "claimed" for i in ids)


def test_each_frame_carries_its_own_presented_ref(tmp_path) -> None:
    from local_operator.harness.types import Message, TextContent
    from local_operator.team_bridge.card_projection import (
        carry_presented_body,
        presented_bridge_ref,
    )

    store, epoch = _store(tmp_path)
    ids = [_reserve(store, epoch, n) for n in range(1, 4)]
    batch = _batch(tmp_path)
    assert _open(batch)
    shaped = batch.transform([Message.user("wake card")])
    refs = []
    for _, frame in [(i, shaped[i]) for i, _ in _frames(shaped)]:
        rendered = Message(role="user", content=[TextContent(text=frame.details["text"])])
        carry_presented_body(frame, rendered)
        ref = presented_bridge_ref([rendered])
        assert ref is not None
        refs.append(ref.message_id)
    assert refs == ids


def test_one_sha_mismatch_out_of_k_still_serves_the_rest(tmp_path) -> None:
    from local_operator.harness.types import Message

    store, epoch = _store(tmp_path)
    ids = [_reserve(store, epoch, n) for n in range(1, 5)]
    batch = _batch(tmp_path)
    assert _open(batch, _fetcher(bad=frozenset({ids[2]})))
    served = _message_ids(_frames(batch.transform([Message.user("wake")])))
    assert served == [ids[0], ids[1], ids[3]]
    bad = _state(tmp_path, ids[2])
    assert bad["state"] == "pending" and bad["last_error"] == "binding_drift"
    # Tried, so this pass does not offer it again as a primary.
    assert (CONVERSATION_ID, ids[2]) in batch.identities
    assert batch.settle(entry_id="e1", error=None, stop_reason="stop", aborted=False) == "completed"
    assert [_state(tmp_path, i)["state"] for i in served] == ["completed"] * 3


def test_a_companion_over_capacity_is_released_for_the_next_pass(tmp_path) -> None:
    from local_operator.harness.types import Message

    store, epoch = _store(tmp_path)
    ids = [_reserve(store, epoch, n) for n in range(1, 4)]
    # Measure: room for the rules + two frames, not three.
    probe = _batch(tmp_path / "probe")
    probe_store, probe_epoch = _store(tmp_path / "probe")
    for n in range(1, 4):
        _reserve(probe_store, probe_epoch, n)
    assert _open(probe)
    from local_operator.team_bridge.body_frame import request_size

    # The bytes one request carries: three frames, the rules head on the first.
    sizes = [request_size(t) for _, t in _frames(probe.transform([Message.user("wake")]))]
    probe.settle(entry_id=None, error="x", stop_reason=None, aborted=True)
    budget = _budget(sum(sizes) - 10)

    batch = _batch(tmp_path)
    assert _open(batch, budget=budget)
    served = _message_ids(_frames(batch.transform([Message.user("wake")])))
    assert served == ids[:2]
    assert batch.deferred_for_capacity == [(CONVERSATION_ID, ids[2])]
    assert (CONVERSATION_ID, ids[2]) not in batch.identities
    row = _state(tmp_path, ids[2])
    assert row["state"] == "pending" and row["last_error"] == "body_not_fitted"


def test_a_primary_that_cannot_fit_serves_nothing_and_returns_every_claim(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    ids = [_reserve(store, epoch, n) for n in range(1, 3)]
    _reserve(store, epoch, 50, wake_class="ambient")
    batch = _batch(tmp_path)
    assert not _open(batch, budget=_budget(100))
    assert _state(tmp_path, ids[0])["last_error"] == "body_not_fitted"
    assert [_state(tmp_path, i)["state"] for i in ids] == ["pending", "pending"]
    digest = _state(tmp_path, _mid(50))
    assert digest["state"] == "deferred" and digest["attempts"] == 0


def test_a_failed_or_cut_turn_releases_every_member(tmp_path) -> None:
    from local_operator.harness.types import Message

    store, epoch = _store(tmp_path)
    ids = [_reserve(store, epoch, n) for n in range(1, 4)]
    _reserve(store, epoch, 60, wake_class="ambient")
    batch = _batch(tmp_path)
    assert _open(batch)
    batch.transform([Message.user("wake")])
    assert batch.settle(entry_id="e1", error=None, stop_reason=None, aborted=True) == "unverified"
    assert [_state(tmp_path, i)["state"] for i in ids] == ["pending"] * 3
    # The digest member goes back to ``deferred`` -- never ``pending``.
    assert _state(tmp_path, _mid(60))["state"] == "deferred"


def test_a_partial_claim_loss_drops_only_that_member(tmp_path) -> None:
    from local_operator.harness.types import Message

    store, epoch = _store(tmp_path)
    ids = [_reserve(store, epoch, n) for n in range(1, 4)]
    batch = _batch(tmp_path)
    assert _open(batch)
    batch.transform([Message.user("wake")])
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute(
            "UPDATE body_reservations SET claim_token = 'pt:stolen' WHERE message_id = ?",
            (ids[1],),
        )
    served = _message_ids(_frames(batch.transform([Message.user("wake")])))
    assert served == [ids[0], ids[2]]
    assert batch.settle(entry_id="e1", error=None, stop_reason="stop", aborted=False) == "completed"
    outcomes = dict(batch.last_outcomes)
    assert outcomes[(CONVERSATION_ID, ids[1])] == "claim_lost"
    assert _state(tmp_path, ids[1])["claim_token"] == "pt:stolen"
    assert _state(tmp_path, ids[0])["state"] == "completed"


def test_digest_rows_lead_the_block_and_are_never_a_primary(tmp_path) -> None:
    from local_operator.harness.types import Message

    store, epoch = _store(tmp_path)
    digest = [_reserve(store, epoch, n, wake_class="ambient") for n in (10, 11)]
    staggered = _reserve(store, epoch, 12, wake_class="staggered")
    # Only deferred rows: nothing may open (a digest row is never a primary).
    assert not _open(_batch(tmp_path))
    assert all(_state(tmp_path, d)["state"] == "deferred" for d in digest)

    woken = _reserve(store, epoch, 20)
    batch = _batch(tmp_path)
    assert _open(batch)
    frames = _frames(batch.transform([Message.user("wake")]))
    assert _message_ids(frames) == [*digest, woken]
    assert all("delivery: ambient_digest" in t for _, t in frames[:2])
    assert "delivery:" not in frames[2][1]
    assert frames[0][1].startswith(BRIDGE_RECEIVER_RULES)
    # A staggered row still owns its urgent-failover wake: never in a digest.
    assert _state(tmp_path, staggered)["state"] == "deferred"
    assert batch.settle(entry_id="e1", error=None, stop_reason="stop", aborted=False) == "completed"
    assert [_state(tmp_path, d)["state"] for d in digest] == ["completed"] * 2


def test_a_failing_digest_row_retires_after_two_attempts_never_pending(tmp_path) -> None:
    from local_operator.harness.types import Message

    store, epoch = _store(tmp_path)
    ambient = _reserve(store, epoch, 10, wake_class="ambient")
    for attempt, n in enumerate((20, 21), start=1):
        _reserve(store, epoch, n)
        batch = _batch(tmp_path)
        assert _open(batch)
        batch.transform([Message.user("wake")])
        batch.settle(entry_id=None, error="boom", stop_reason=None, aborted=False)
        row = _state(tmp_path, ambient)
        assert row["state"] == ("deferred" if attempt == 1 else "retired")
        assert row["state"] != "pending"
        with sqlite3.connect(state_db_path(tmp_path)) as conn:
            conn.execute(
                "UPDATE body_reservations SET state = 'retired' WHERE message_id = ?", (_mid(n),)
            )
    assert _state(tmp_path, ambient)["terminal_code"] == "ambient_not_shown"


def test_digest_bounds_drop_old_rows_uncounted_and_say_how_many(tmp_path) -> None:
    from local_operator.harness.types import Message

    store, epoch = _store(tmp_path)
    digest = [_reserve(store, epoch, n, wake_class="ambient") for n in range(10, 14)]
    _reserve(store, epoch, 20)
    batch = _batch(tmp_path, digest_max=2)
    assert _open(batch)
    frames = _frames(batch.transform([Message.user("wake")]))
    # Newest two shown, oldest-first on the wire; the first frame counts the rest.
    assert _message_ids(frames)[:2] == digest[2:]
    assert "2 more ambient messages not shown; channel_history reads them." in frames[0][1]
    assert [_state(tmp_path, d)["state"] for d in digest[:2]] == ["deferred"] * 2

    # Over the byte cap: claimed, then dropped UN-COUNTED back to deferred.
    tight = tmp_path / "tight"
    store2, epoch2 = _store(tight)
    dropped = _reserve(store2, epoch2, 10, wake_class="ambient")
    _reserve(store2, epoch2, 20)
    batch2 = _batch(tight, digest_max_bytes=10)
    assert _open(batch2)
    row = _state(tight, dropped)
    assert row["state"] == "deferred" and row["attempts"] == 0


def test_the_block_stays_at_its_anchor_across_tool_steps_and_compaction(tmp_path) -> None:
    from local_operator.harness.types import Message

    store, epoch = _store(tmp_path)
    for n in range(1, 4):
        _reserve(store, epoch, n)
    batch = _batch(tmp_path)
    assert _open(batch)
    prior, wake = Message.user("prior"), Message.user("wake card")
    call1 = batch.transform([prior, wake])
    assert [i for i, _ in _frames(call1)] == [2, 3, 4]
    step = [prior, wake, _asst_calls("c1"), _tool_reply("c1")]
    call2 = batch.transform(step)
    assert [i for i, _ in _frames(call2)] == [2, 3, 4]
    # Byte-identical prefix through the block on Call 2.
    assert [t for _, t in _frames(call2)] == [t for _, t in _frames(call1)]
    assert _wire_pairing_failures(call2) == []
    # Mid-turn compaction dropped the anchor: the whole block goes last, valid
    # on both wires.
    compacted = [Message.user("[summary]"), _asst_calls("c5"), _tool_reply("c5")]
    call3 = batch.transform(compacted)
    assert [i for i, _ in _frames(call3)] == [3, 4, 5]
    assert _wire_pairing_failures(call3) == []


def test_companions_are_same_conversation_horizon_bounded_and_capped(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    now = time.time()
    first = _reserve(store, epoch, 1, created_at=now)
    _reserve(store, epoch, 2, conversation_id=OTHER_CONVERSATION, created_at=now + 0.1)
    for n in range(3, 13):
        _reserve(store, epoch, n, created_at=now + n * 0.01)
    late = _reserve(store, epoch, 99, created_at=now + 50)
    batch = _batch(tmp_path, admitted_before=now + 10, digest_max=0)
    assert _open(batch)
    served = [m.identity[1] for m in batch.members]  # type: ignore[index]
    assert served[0] == first and len(served) == 1 + 7
    assert _mid(2) not in served and late not in served


def test_an_inadmissible_companion_is_left_untouched(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    ids = [_reserve(store, epoch, n) for n in range(1, 4)]
    batch = _batch(tmp_path, admissible=lambda row: row["message_id"] != ids[1])
    assert _open(batch)
    assert [m.identity[1] for m in batch.members] == [ids[0], ids[2]]  # type: ignore[index]
    row = _state(tmp_path, ids[1])
    assert row["state"] == "pending" and row["attempts"] == 0


def test_multi_frame_turns_setting_defaults_on_and_reads_off() -> None:
    def parse(value: Any) -> bool:
        return TeamBridgeSettings.from_values(
            {"team_bridge": {"multi_frame_turns": value}}
        ).multi_frame_turns

    assert TeamBridgeSettings.from_values({}).multi_frame_turns is True
    assert parse(False) is False and parse("off") is False
    assert parse(True) is True and parse("junk") is True


def test_store_reads_several_woken_rows_and_the_newest_digest_rows(tmp_path) -> None:
    store, epoch = _store(tmp_path)
    woken = [_reserve(store, epoch, n) for n in range(1, 4)]
    digest = [_reserve(store, epoch, n, wake_class="passive") for n in range(10, 13)]
    rows = store.pending_body_reservations_for(_owner(), SESSION, limit=5)
    assert [r["message_id"] for r in rows] == woken
    pending = store.pending_body_reservation_for(_owner(), SESSION)
    assert pending is not None
    assert pending["message_id"] == woken[0]
    newest = store.deferred_body_reservations_for(_owner(), SESSION, limit=2)
    assert [r["message_id"] for r in newest] == digest[::-1][:2]
    # The ordinary claim CAS refuses a deferred row; the digest CAS keeps it classed.
    assert not store.claim_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=digest[0],
        claim_token="pt:aaaa",
        turn_token="tt:aaaa",
        now=time.time(),
        ttl=60,
    )
    assert store.claim_digest_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=digest[0],
        claim_token="pt:bbbb",
        turn_token="tt:bbbb",
        now=time.time(),
        ttl=60,
    )
    assert _state(tmp_path, digest[0])["wake_class"] == "passive"
    assert store.pending_body_reservations_for(_owner(), SESSION, limit=10, now=time.time() + 120)
    assert digest[0] not in [
        r["message_id"]
        for r in store.pending_body_reservations_for(
            _owner(), SESSION, limit=10, now=time.time() + 120
        )
    ]


# -- round-1 remediation (PR #4382: Q1, R1, R2, R3) ----------------------------


def test_an_exclusion_never_scopes_the_woken_reader_to_its_conversation(tmp_path) -> None:
    """Q1: the ``exclude`` loop once rebound the ``conversation_id`` parameter,
    so excluding ONE row silently filtered the reader to that row's
    conversation and the drain skipped every other conversation's rows."""

    store, epoch = _store(tmp_path)
    now = time.time()
    first = _reserve(store, epoch, 1, created_at=now)
    other = _reserve(store, epoch, 2, conversation_id=OTHER_CONVERSATION, created_at=now + 0.1)
    excluded = frozenset({(CONVERSATION_ID, first)})
    row = store.pending_body_reservation_for(_owner(), SESSION, exclude=excluded)
    assert row is not None and row["message_id"] == other
    # An explicit conversation scope still applies, exclusion or not.
    assert (
        store.pending_body_reservations_for(
            _owner(), SESSION, exclude=excluded, conversation_id=CONVERSATION_ID, limit=5
        )
        == []
    )


def _flaky_renew(member: Any, failures: int = 1) -> None:
    """Make ``member``'s NEXT lease renewal(s) refuse as a busy store would."""

    from local_operator.team_bridge.store import StoreRefused

    original = member._store.renew_body_claim
    left = {"n": failures}

    def renew(owner: Any, **kw: Any) -> Any:
        if kw["message_id"] == member.identity[1] and left["n"] > 0:
            left["n"] -= 1
            raise StoreRefused("store_busy", "busy")
        return original(owner, **kw)

    member._store.renew_body_claim = renew


def test_the_receiver_rules_move_to_the_first_live_frame_when_the_carrier_drops(
    tmp_path,
) -> None:
    """R1: one transient refusal on the rule carrier's renewal drops it for the
    rest of the turn; the remaining untrusted bodies must still be preceded by
    the receiver rules (config-request boundary included) on every call."""

    from local_operator.harness.message_types import BRIDGE_CONFIG_REQUEST_RULE
    from local_operator.harness.types import Message

    store, epoch = _store(tmp_path)
    ids = [_reserve(store, epoch, n) for n in range(1, 4)]
    _reserve(store, epoch, 10, wake_class="ambient")
    batch = _batch(tmp_path, digest_max=2)
    assert _open(batch)
    call1 = _frames(batch.transform([Message.user("wake card")]))
    assert call1[0][1].startswith(BRIDGE_RECEIVER_RULES)
    carrier_id = _message_ids(call1)[0]
    for carrier in batch.members:
        assert carrier.identity is not None
        if carrier.identity[1] == carrier_id:
            break
    else:
        raise AssertionError("the first live frame must have a carrier")
    _flaky_renew(carrier)
    for _ in range(2):  # calls 2 and 3
        frames = _frames(batch.transform([Message.user("wake card")]))
        assert _message_ids(frames) == ids
        texts = [t for _, t in frames]
        assert texts[0].startswith(BRIDGE_RECEIVER_RULES)
        assert BRIDGE_CONFIG_REQUEST_RULE in texts[0]
        assert BRIDGE_CHANNEL_PARTICIPATION_RULE in texts[0]
        assert "ambient messages not shown" not in texts[0]  # nothing omitted
        # Still exactly ONE rules head per request.
        assert sum(BRIDGE_RECEIVER_RULES in t for t in texts) == 1
        assert all(t.startswith("VERIFIED SOURCE") for t in texts[1:])


def test_digest_rows_never_crowd_out_a_woken_body_that_fits_alone(tmp_path) -> None:
    """R2: participation rule + "not shown" line belong to the digest's cost;
    a DM body whose one-body frame fits exactly must still be served."""

    from local_operator.harness.types import Message
    from local_operator.team_bridge.pending_turn import PendingTurn

    async def fetch(conversation_id: str, message_id: str) -> FetchedBody | None:
        return FetchedBody(
            body=_body(message_id),
            body_sha256=_sha(message_id),
            conversation_label="x",
            sender_label="Ada",
            sender_kind="agent",
            conversation_kind="channel" if conversation_id == OTHER_CONVERSATION else "dm",
        )

    alone = tmp_path / "alone"
    store1, epoch1 = _store(alone)
    _reserve(store1, epoch1, 1)
    one = PendingTurn(_owner(), SESSION, config_dir=alone)
    assert asyncio.run(one.open(budget=_budget(), fetch=fetch))
    size = one.frame_size

    store, epoch = _store(tmp_path)
    digest = [
        _reserve(store, epoch, n, conversation_id=OTHER_CONVERSATION, wake_class="ambient")
        for n in range(50, 53)
    ]
    primary = _reserve(store, epoch, 1)
    batch = _batch(tmp_path)
    assert _open(batch, fetch=fetch, budget=_budget(size))
    assert _state(tmp_path, primary)["state"] == "claimed"
    frames = _frames(batch.transform([Message.user("wake")]))
    assert _message_ids(frames) == [primary]
    # The digest rows that did not fit went back un-counted, and no
    # participation rule rode a DM-only request.
    assert BRIDGE_CHANNEL_PARTICIPATION_RULE not in frames[0][1]
    for d in digest:
        row = _state(tmp_path, d)
        assert row["state"] == "deferred" and row["attempts"] == 0


def test_a_refused_optional_read_degrades_to_fewer_members(tmp_path) -> None:
    """R3: a transient refusal reading companions or the digest must not cancel
    the turn; the primary is served as the one-body path would serve it."""

    from local_operator.team_bridge.store import StoreRefused

    for reader in ("deferred_body_reservations_for", "pending_body_reservations_for"):
        root = tmp_path / reader
        store, epoch = _store(root)
        primary = _reserve(store, epoch, 1)
        companion = _reserve(store, epoch, 2)
        ambient = _reserve(store, epoch, 10, wake_class="ambient")
        batch = _batch(root)
        original = getattr(batch._store, reader)

        def boom(*args: Any, _original: Any = original, _r: str = reader, **kwargs: Any) -> Any:
            # Only the OPTIONAL reads refuse: the primary's own read (the
            # same woken reader, no conversation scope) must succeed.
            if _r.startswith("deferred") or kwargs.get("conversation_id") is not None:
                raise StoreRefused("state_unreadable", "database is locked")
            return _original(*args, **kwargs)

        setattr(batch._store, reader, boom)
        assert _open(batch), reader
        served = [m.identity[1] for m in batch.members]  # type: ignore[index]
        assert _state(root, primary)["state"] == "claimed"
        if reader == "pending_body_reservations_for":
            # Companions unread: left pending, untouched, for the next pass;
            # the digest (read fine) still leads the block.
            assert served == [ambient, primary]
            row = _state(root, companion)
            assert row["state"] == "pending" and row["attempts"] == 0
        else:
            assert served == [primary, companion]
            assert _state(root, ambient)["state"] == "deferred"

"""S4 cells: the wipe — ends reachable copies, by provenance, with a confirmation.

WHAT THIS FILE PINS (design §4.3, §5.5c):

- a revoke wipes the member's copy and CLOSES the ledger — the member deletes
  and confirms, and the owner's row reads ``wiped``;
- an OFFLINE member is not exempt: nothing is queued at the member, and the
  same notice completes on its NEXT contact (catch-up, never invalidate);
- MEMBER REMOVAL carries the ending itself, through the real verb: the relay's
  ``member rm`` runs the bounded ending exchange BEFORE the tombstone (after it
  the dial is refused by design — an inactive member is never contacted again),
  so the copies are deleted while the member is still contactable and the
  receipt says so; unreachable at that instant, the ledger row stays OPEN and
  the receipt names the count plus the ceiling sentence (§2.3's discipline:
  an ending that cannot complete says so);
- the confirmation rides the REPLY to the owner's dialled announce — an
  un-approved member cannot open a frame (its ``broker_credential`` capability
  left with the grant; measured, the transport refuses ``net_broker`` from a
  deactivated row), so a fresh-request ack would be refused at the owner's door;
- a wipe that cannot confirm blocks ITS OWN key only: a member that answers
  receipt-only (a pre-S4 build) still receives every OTHER key's updates on
  the same pass (review round 1, M2 — the old wipes-only return starved them);
- a ``local-only`` mark ends existing copies the same way;
- a wipe spares rows another owner or another key wrote — deletion is bounded
  by the provenance marker, not by the notice;
- a RE-SHARE after a wipe delivers again (a wiped ledger row must not match
  the old generation+digest, or the re-shared copy would stay undelivered).

The delivery runs two real relays on loopback roots, exactly as
``test_secret_copies`` does.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import store
from local_operator.network.credentials import sync
from local_operator.network.credentials.messages import COPY_CEILING_LINES
from local_operator.network.credentials.sync import SyncState
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_real_link import (  # noqa: F401
    _lop_network,
    _pull,
)
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixtures by import
from tests.unit.network.test_secret_copies import (  # noqa: F401 — fixtures by import
    KEY,
    SECRET_NAME,
    VALUE_1,
    VALUE_2,
    _member_has,
    _member_value,
    _secret_store,
    secret_mesh,
)


@pytest.fixture()
def copied_mesh(request: pytest.FixtureRequest) -> Any:
    """``secret_mesh`` with A's copy ALREADY delivered to B (one real exchange)."""
    mesh = request.getfixturevalue("secret_mesh")
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(lambda: _member_has(mesh.b.root, SECRET_NAME))
    return mesh


def _ledger_wiped(mesh: Any) -> bool:
    row = SyncState.load(mesh.network_id, mesh.a.root).ack_for(mesh.member, KEY) or {}
    return bool(row.get("wiped"))


def test_a_revoke_wipes_the_member_copy_and_the_ledger_closes(copied_mesh: Any) -> None:
    mesh = copied_mesh
    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(
        lambda: not _member_has(mesh.b.root, SECRET_NAME)
    ), "the member's copy was not wiped"
    assert net_fixtures.wait_for(
        lambda: _ledger_wiped(mesh)
    ), "the ending was not confirmed to the owner"
    # NOTHING WAS LEFT BEHIND: neither the value nor a record under the name.
    assert not _member_has(mesh.b.root, SECRET_NAME)


def test_an_offline_member_is_wiped_on_reconnect(copied_mesh: Any) -> None:
    """The unreachable member: nothing is queued at either end, and the SAME
    notice recomputes and completes on the next contact (§5.5c, (a)'s catch-up).

    Reachability is simulated at the endpoint (a dead port), not by stopping a
    relay: ``stop``/``start`` is not a supported cycle in this build, and the
    product property under test is the DIAL, not the process.

    THE SURVIVING LIVE LINK IS CLOSED FIRST (review round 3, F3): a live link
    carries the wipe however dead the endpoint reads — measured while writing
    the sibling cell, and here as an intermittent red (1 of 8 full-file runs)
    where the async exchange won the race against the immediate store read.
    Closing what the idle reaper eventually would (``LINK_IDLE_S`` is 120 s, far
    past a cell) makes the dial the only path, deterministically.
    """
    mesh = copied_mesh
    for link in list(mesh.a.links.values()):
        if getattr(link, "device_id", "") == mesh.member:
            link.close("test: the member becomes unreachable")
    with store.mutate(mesh.network_id, mesh.a.root) as record:
        member_row = record.member(mesh.member)
        assert member_row is not None
        member_row.endpoints = ["127.0.0.1:1"]  # a dead endpoint: unreachable now
        store.save(record, mesh.a.root)
    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert _member_has(mesh.b.root, SECRET_NAME), "no contact, no wipe"
    assert not _ledger_wiped(mesh)

    with store.mutate(mesh.network_id, mesh.a.root) as record:
        member_row = record.member(mesh.member)
        assert member_row is not None
        member_row.endpoints = [f"127.0.0.1:{mesh.b.settings.port}"]
        store.save(record, mesh.a.root)

    def _landed() -> bool:
        # The dead-endpoint dial may still hold the per-member in-flight guard,
        # so the cell RETRIES the tick rather than assuming one step lands it —
        # which is also how the definitions syncer drives it in the product.
        code = sync.credentials_sync_step(mesh.a, mesh.member)
        assert code in ("scheduled", "in_sync", "in_flight"), code
        return not _member_has(mesh.b.root, SECRET_NAME)

    assert net_fixtures.wait_for(_landed), "the notice did not complete on reconnect"
    assert net_fixtures.wait_for(lambda: _ledger_wiped(mesh))


def test_member_rm_delivers_the_ending_while_the_member_is_contactable(
    copied_mesh: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un-approve through the REAL verb: the removal path carries the ending.

    Q1 of review round 1: the definitions tick only visits ACTIVE members, so
    the old cell — deactivate, then drive ``credentials_sync_step`` directly —
    proved a path no product call ever takes, and the product's ``member rm``
    left every copy usable on the removed device with no wipe attempt and no
    sentence. This drives the verb: the relay's handler runs the bounded ending
    exchange BEFORE the tombstone, the member deletes and the ledger closes,
    and the receipt says the ending was confirmed.

    THE MEMBER'S STORE OPEN IS DELAYED PAST THE OLD BOUND, deterministically
    (review round 2, F1): a loaded member measured 5.02 s in
    ``access.open_store`` ALONE, and against the original 3 s frame bound the
    request returned ``None`` at exactly 3.00 s, the member finished its delete
    ~2 s later, and the owner's ledger row stayed open forever — the cell
    failed 6/6 under load. The simulated delay below is 5 s: >= the old bound
    (so this cell fails on the old code) and inside the new one, the tick's own
    10 s (so the new code confirms). It discriminates whether or not the host
    happens to be loaded.
    """
    from local_operator.secrets import access as access_mod

    mesh = copied_mesh
    real_open = access_mod.open_store

    def slow_open(root: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(str(root)) == Path(str(mesh.b.root)):
            time.sleep(5.0)
        return real_open(root, *args, **kwargs)

    monkeypatch.setattr(access_mod, "open_store", slow_open)
    try:
        record = store.load(mesh.network_id, mesh.a.root)
        assert _lop_network("member", "rm", record.name, mesh.b.identity.name) == 0
    finally:
        # The member's delete ran INLINE inside the relay op, so the delay has
        # done its work; un-patching keeps the poll below fast.
        monkeypatch.setattr(access_mod, "open_store", real_open)
    out = capsys.readouterr().out
    assert net_fixtures.wait_for(
        lambda: not _member_has(mesh.b.root, SECRET_NAME)
    ), "the removal did not deliver the ending"
    assert net_fixtures.wait_for(
        lambda: _ledger_wiped(mesh)
    ), "the wipe was not confirmed to the owner"
    assert "1 copy on " in out and "was deleted (the ending is confirmed)" in out, out
    # The confirmed arm used to be the one removal state WITHOUT the ceiling (design
    # review round 1, D8: the state that claims the most withheld the sentence about what
    # it did not do), so the ceiling rides it too.
    for ceiling_line in COPY_CEILING_LINES:
        assert ceiling_line in out, out
    # The tombstone is still the verb's observable side effect, and the removed
    # member is never ticked again — no later contact can re-deliver anything.
    removed = store.load(mesh.network_id, mesh.a.root).member(mesh.member)
    assert removed is not None and not removed.active


def test_a_removal_timeout_renders_distinctly_from_unreachable(
    copied_mesh: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1's second half: a give-up that may still complete renders as its own class.

    The owner's frame bound is lowered below the member's simulated store open
    (1 s vs 2 s), so the exchange outruns the bound exactly as F1 measured under
    load — deterministically, and in ~2 s rather than a loaded 5 s. The receipt
    must then say ``timed out — the member may still complete the deletion``,
    NOT ``could NOT be confirmed deleted`` (which asserts nothing was
    contacted). The member DOES complete ~a second after the receipt, and the
    cell waits for the copy to be gone to prove the wording describes the real
    half-life of this failure. The ledger row stays open — the one honest
    value nobody can recompute, because a removed member is never contacted
    again.
    """
    from local_operator.network.credentials import sync as sync_mod
    from local_operator.secrets import access as access_mod

    mesh = copied_mesh
    real_open = access_mod.open_store

    def slow_open(root: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(str(root)) == Path(str(mesh.b.root)):
            time.sleep(2.0)
        return real_open(root, *args, **kwargs)

    monkeypatch.setattr(access_mod, "open_store", slow_open)
    monkeypatch.setattr(sync_mod, "REMOVAL_FRAME_TIMEOUT_S", 1.0)
    record = store.load(mesh.network_id, mesh.a.root)
    assert _lop_network("member", "rm", record.name, mesh.b.identity.name) == 0
    out = capsys.readouterr().out
    assert "timed out — the member may still complete the deletion" in out, out
    assert "could NOT be confirmed deleted" not in out, out
    # The wording is a claim about the real half-life: the member finishes the
    # delete it started, ~1 s after the owner gave up on the answer.
    monkeypatch.setattr(access_mod, "open_store", real_open)
    assert net_fixtures.wait_for(
        lambda: not _member_has(mesh.b.root, SECRET_NAME)
    ), "the timed-out member did NOT complete; the receipt wording would be a lie"
    assert not _ledger_wiped(mesh), "no confirmation arrived, so the row must stay open"


def test_the_removal_wait_covers_the_exchange_envelope() -> None:
    """F2 (review round 3): the caller's wait must exceed the exchange's envelope.

    The loop clamps each frame's wait by the remaining budget, so its true
    envelope is probe + budget — and BOTH callers (the CLI's ``net_member_rm``
    and the desktop route's call of the same op) wait ``REMOVAL_CLI_TIMEOUT_S``.
    If a future bound change makes the wait fit INSIDE the envelope, the caller
    falls back to the local write mid-exchange; this cell fails first, before
    that window can exist. The behavioral half — that the clamp is what keeps
    the envelope at budget rather than budget + frame — is pinned by
    ``test_the_frame_wait_is_clamped_to_the_remaining_budget`` below.
    """
    from local_operator.network.credentials import sync as sync_mod

    envelope = sync_mod.REMOVAL_PROBE_TIMEOUT_S + sync_mod.REMOVAL_TOTAL_BUDGET_S
    assert sync_mod.REMOVAL_CLI_TIMEOUT_S > envelope, (
        "the caller's wait must cover probe + budget (the loop's clamped envelope): "
        f"{sync_mod.REMOVAL_CLI_TIMEOUT_S}s vs {envelope}s"
    )


def test_the_frame_wait_is_clamped_to_the_remaining_budget(
    copied_mesh: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2's clamp, behaviorally: a frame cannot outrun the loop's budget.

    The budget is lowered below the member's simulated open while the frame
    bound stays long: an UNCLAMPED wait would let the member answer inside its
    full bound (the exchange would confirm at the member's pace, past the
    budget), while the clamped wait gives up when the budget expires — which is
    what keeps the caller's wait safe by construction. The receipt carries the
    timed-out class, and the member still completes afterwards; the wall check
    is secondary to that class assertion, which no load can flip.
    """
    from local_operator.network.credentials import sync as sync_mod
    from local_operator.secrets import access as access_mod

    mesh = copied_mesh
    real_open = access_mod.open_store

    def slow_open(root: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(str(root)) == Path(str(mesh.b.root)):
            time.sleep(3.5)
        return real_open(root, *args, **kwargs)

    monkeypatch.setattr(access_mod, "open_store", slow_open)
    monkeypatch.setattr(sync_mod, "REMOVAL_FRAME_TIMEOUT_S", 5.0)
    monkeypatch.setattr(sync_mod, "REMOVAL_TOTAL_BUDGET_S", 1.5)
    started = time.monotonic()
    record = store.load(mesh.network_id, mesh.a.root)
    assert _lop_network("member", "rm", record.name, mesh.b.identity.name) == 0
    elapsed = time.monotonic() - started
    out = capsys.readouterr().out
    # The class assertion is the discriminator: an unclamped wait would let the
    # member's 3.5 s open answer inside the 5 s bound and CONFIRM the wipe; the
    # clamped wait expires with the budget, so the owner reads the give-up class.
    assert "timed out" in out, out
    # Sanity bound only (the class above is the discriminator): a clamped loop
    # answers in ~budget + overhead; even a loaded host stays well inside this,
    # while an unclamped wait on a slower member would sit at its full bound.
    assert elapsed < 4.5, f"the loop outran its 1.5 s budget: {elapsed:.1f}s"
    monkeypatch.setattr(access_mod, "open_store", real_open)
    assert net_fixtures.wait_for(
        lambda: not _member_has(mesh.b.root, SECRET_NAME)
    ), "the member still completes after the owner's budget expired"


def test_a_lock_refused_removal_does_not_run_its_ending_exchange(
    copied_mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """QA r3 observation: the ending exchange and the removal are one decision.

    A ``member rm`` refused by the rotation lock used to run its ending
    exchange FIRST — measured by QA: G's copies were wiped while the refusal
    said nothing about them, and the member stayed an active holder a tick
    could re-deliver the copy to. The lock is checked before the exchange now,
    with the same sentence ``rotate_epoch`` raises; the copies are untouched.
    Clearing the lock, the retried removal delivers the ending normally.
    """
    mesh = copied_mesh
    with store.mutate(mesh.network_id, mesh.a.root) as record:
        # The state a just-run rotation leaves behind (the QA repro armed it
        # with a real second removal; this is that record value, directly).
        record.rotation_lock_until = time.time() + 27.0
        store.save(record, mesh.a.root)
    record = store.load(mesh.network_id, mesh.a.root)
    assert _lop_network("member", "rm", record.name, mesh.b.identity.name) == 1
    captured = capsys.readouterr()
    assert "already in progress" in captured.err, captured
    assert _member_has(mesh.b.root, SECRET_NAME), "a refused removal must not wipe"
    assert not _ledger_wiped(mesh)
    # The lock clears; the same verb now runs the whole unit of decision.
    with store.mutate(mesh.network_id, mesh.a.root) as record:
        record.rotation_lock_until = 0.0
        store.save(record, mesh.a.root)
    record = store.load(mesh.network_id, mesh.a.root)
    assert _lop_network("member", "rm", record.name, mesh.b.identity.name) == 0
    out = capsys.readouterr().out
    assert net_fixtures.wait_for(lambda: not _member_has(mesh.b.root, SECRET_NAME))
    assert net_fixtures.wait_for(lambda: _ledger_wiped(mesh))
    assert "was deleted (the ending is confirmed)" in out, out


def test_member_rm_without_contact_records_the_open_ending(
    copied_mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unreachable at removal: the receipt and the ledger carry the OPEN ending.

    A removed member is never contacted again — an inactive row is refused the
    dial by design — so there is no next-contact catch-up to defer to. §2.3's
    discipline applies instead: the exchange runs, fails, and SAYS SO, so a
    removal with live copies never reads as a clean sweep.

    The copy exchange leaves a live link (``LINK_IDLE_S`` is 120 s, far past a
    cell), and a live link would carry the ending however dead the endpoint
    reads — measured while writing this cell. A genuinely unreachable peer has
    none, so the cell closes what the idle reaper eventually would.
    """
    mesh = copied_mesh
    for link in list(mesh.a.links.values()):
        if getattr(link, "device_id", "") == mesh.member:
            link.close("test: the member becomes unreachable")
    with store.mutate(mesh.network_id, mesh.a.root) as record:
        member_row = record.member(mesh.member)
        assert member_row is not None
        member_row.endpoints = ["127.0.0.1:1"]  # a dead endpoint: unreachable now
        store.save(record, mesh.a.root)
    record = store.load(mesh.network_id, mesh.a.root)
    assert _lop_network("member", "rm", record.name, mesh.b.identity.name) == 0
    out = capsys.readouterr().out
    assert _member_has(mesh.b.root, SECRET_NAME), "no contact, no wipe"
    assert not _ledger_wiped(mesh), "nothing confirmed it, so it must stay open"
    assert "could NOT be confirmed deleted" in out, out
    # Two lines now (one fact each), printed after the state line.
    for ceiling_line in COPY_CEILING_LINES:
        assert ceiling_line in out, out


def test_an_unconfirmable_wipe_does_not_starve_other_keys(
    copied_mesh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M2 (review round 1): a wipe that never confirms blocks ITS key only.

    A pre-S4 member answers receipt-only, so the owner's ledger row for the
    revoked key stays un-``wiped`` forever. The old ``_pending_announces``
    returned wipes-ONLY whenever any wipe was owed, so every other key's future
    value updates were withheld from that member silently and unboundedly.
    The fix gates per key: the ending rides every exchange AND the positives
    flow in the same pass, each kind under its own cap. Both arms are pinned:
    the unconfirmable wipe's key stays withheld (the copy is gone and the row
    is open), and the other key's update reaches the member anyway.
    """
    mesh = copied_mesh
    second_name = "STORE_COPY_TOKEN_TWO"
    second_key = f"secret:{second_name}"
    mesh.store_a.set(second_name, VALUE_1, description="M2's second key")
    assert _lop_network("credential", "share", second_key, "--with", mesh.b.identity.name) == 0
    pulled = _pull(SimpleNamespace(b=mesh.b, borrower=mesh.member))
    assert second_key in pulled.get("changed", []), pulled
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(lambda: _member_has(mesh.b.root, second_name))

    # The member becomes one that CANNOT confirm a wipe: the reply is answered
    # (the copy is deleted) but the owner never records it — the pre-S4 shape.
    assert sync.sync_for_relay(mesh.a) is mesh.engine_a
    monkeypatch.setattr(mesh.engine_a, "_record_wipe_reply", lambda *a, **k: None)
    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(lambda: not _member_has(mesh.b.root, SECRET_NAME))
    assert not _ledger_wiped(mesh), "the unconfirmable wipe must stay open"

    # The OTHER key changes at the source: the same pass must still carry it.
    mesh.store_a.update(second_name, VALUE_2)
    _net, frames = mesh.engine_a._pending_announces(mesh.member)
    kinds = [(f.get("key"), f.get("value_state")) for f in frames]
    assert (KEY, sync.VALUE_STATE_ABSENT) in kinds, kinds
    assert (second_key, sync.VALUE_STATE_PRESENT) in kinds, kinds
    assert kinds[0][0] == KEY, f"the ending must lead the frames, got {kinds}"

    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(
        lambda: _member_value(mesh.b.root, second_name) == VALUE_2
    ), "the other key's update was starved behind the unconfirmable wipe"
    assert not _ledger_wiped(mesh), "and the open row is still tracked, not dropped"


def test_marking_local_only_ends_existing_copies(copied_mesh: Any) -> None:
    """§4.2's kill switch, second half: the mark ends a copy that already exists."""
    mesh = copied_mesh
    assert _lop_network("credential", "mark", SECRET_NAME, "local-only") == 0
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(
        lambda: not _member_has(mesh.b.root, SECRET_NAME)
    ), "a local-only key must end copies it already has"
    assert net_fixtures.wait_for(lambda: _ledger_wiped(mesh))


def test_a_wipe_spares_rows_another_owner_or_key_wrote(request: pytest.FixtureRequest) -> None:
    """The provenance bound: deletion is matched on BOTH the sender and the key."""
    mesh = request.getfixturevalue("secret_mesh")
    handle = _secret_store(mesh.b.root, create=True)
    handle.set("LOCAL_TOKEN", b"a local row")
    handle.set(
        "OTHER_OWNERS_ROW",
        b"not from A",
        origin={"owner_device": "d_" + "f" * 32, "key": KEY, "gen": 1},
    )
    handle.set(
        "OTHER_KEY_ROW",
        b"from A, another key",
        origin={"owner_device": mesh.owner, "key": "secret:OTHER_KEY", "gen": 1},
    )

    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(lambda: _member_has(mesh.b.root, SECRET_NAME))

    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    assert sync.credentials_sync_step(mesh.a, mesh.member) in ("scheduled", "in_sync")
    assert net_fixtures.wait_for(lambda: not _member_has(mesh.b.root, SECRET_NAME))

    assert _member_has(mesh.b.root, "LOCAL_TOKEN"), "a local row is not this owner's to delete"
    assert _member_has(mesh.b.root, "OTHER_OWNERS_ROW"), "another owner's row must survive"
    assert _member_has(mesh.b.root, "OTHER_KEY_ROW"), "another key's row must survive"


def test_copy_use_revoke_wipe_in_one_run(request: pytest.FixtureRequest) -> None:
    """The whole sequence in one run: copy lands, the member USES it, revoke wipes.

    "Use" is the member's own read path (``get`` — the ordinary read a session
    performs, which also counts as a use), not a re-read of the frame. The
    closing canary re-scans the member's root: after the wipe, the value's
    plaintext appears nowhere, and neither does a record under the name.
    """
    mesh = request.getfixturevalue("secret_mesh")
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(lambda: _member_has(mesh.b.root, SECRET_NAME))

    handle = _secret_store(mesh.b.root)
    assert handle.get(SECRET_NAME) == VALUE_1  # the member uses its copy

    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    assert sync.credentials_sync_step(mesh.a, mesh.member) in ("scheduled", "in_sync")
    assert net_fixtures.wait_for(lambda: not _member_has(mesh.b.root, SECRET_NAME))
    assert net_fixtures.wait_for(lambda: _ledger_wiped(mesh))

    for path in Path(mesh.b.root).rglob("*"):
        if path.is_file():
            assert VALUE_1 not in path.read_bytes(), path


def test_the_revoke_and_mark_receipts_say_which_ending_applies(
    copied_mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """The receipt sentences the design commits to (§2.3, §4.3) are what prints."""
    mesh = copied_mesh
    assert _lop_network("credential", "mark", SECRET_NAME, "sync") == 0
    out = capsys.readouterr().out
    assert f"'{SECRET_NAME}' is marked sync" in out

    # THE LISTING RENDERS IT ON THE ROW (one document, so the mark cannot
    # disagree with the holders it governs).
    assert _lop_network("credentials") == 0
    out = capsys.readouterr().out
    assert "marked sync" in out

    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    out = capsys.readouterr().out
    # Prose names the secret the way the operator typed it (design review round 1, D5);
    # the placement key is an identifier and stays in --json and the listing.
    assert f"a wipe notice for '{SECRET_NAME}' is queued for" in out
    assert "secret:" not in out.split("a wipe notice for", 1)[1]
    for ceiling_line in COPY_CEILING_LINES:
        assert ceiling_line in out, out


def test_a_re_share_after_a_wipe_delivers_again(copied_mesh: Any) -> None:
    """The wiped ledger row is an ENDING, not a match: a re-share must announce."""
    mesh = copied_mesh
    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    assert sync.credentials_sync_step(mesh.a, mesh.member) in ("scheduled", "in_sync")
    assert net_fixtures.wait_for(lambda: not _member_has(mesh.b.root, SECRET_NAME))
    assert net_fixtures.wait_for(lambda: _ledger_wiped(mesh))

    assert _lop_network("credential", "share", KEY, "--with", mesh.b.identity.name) == 0
    pulled = _pull(SimpleNamespace(b=mesh.b, borrower=mesh.member))
    assert KEY in pulled.get("changed", []), pulled
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(
        lambda: _member_has(mesh.b.root, SECRET_NAME)
    ), "a re-shared copy stayed undelivered"
    row = SyncState.load(mesh.network_id, mesh.a.root).ack_for(mesh.member, KEY) or {}
    assert not row.get("wiped"), "a live copy must not read wiped"

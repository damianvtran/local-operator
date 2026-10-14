"""H5 S-8 — stale-pending STANDING REFRESH at the trusted boundary (SOURCE).

Drives the REAL production method (``Session._refresh_stale_pending_standing``)
against a REAL native enrollment + REAL issuance ledger. The transport
(``pull.message``) and the fence result are the boundary seams; no internal
door is substituted: the method's own iteration, validation and ledger writes
run as production does. The contract pinned:
- a fresh PENDING request passes with exactly ONE bounded read;
- withdrawn/decided/expired/mismatched body/recipient/thread ⇒ held, no ask;
- offline/signin/uncertain ⇒ held, no ask;
- local authority OFF before the read ⇒ ZERO reads; OFF during the await ⇒
  held after the read, no ask, no scope rebind (the stored original scope is
  never backfilled).
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.team_bridge import authority as fence_mod
from local_operator.team_bridge import pull as pull_mod
from local_operator.team_bridge.ask_issuance import AskIssuance
from local_operator.team_bridge.store import Store
from local_operator.team_bridge.store import OwnerReceipt
from local_operator.team_bridge.types import OwnerSnapshot
from local_operator.harness.message_types import (
    TEAM_BRIDGE_MESSAGE_TYPE,
    display_context_digest as canonical_display_digest,
)
from local_operator.session.session import Session

pytestmark = pytest.mark.asyncio

MESSAGE_ID = "bmsg_standing"
OWNER = OwnerSnapshot(
    credential_id=7,
    account_id="acc_1",
    tenant_id="tnt_1",
    endpoint_id="ep_1",
    canonical_base_url="https://hub.example",
    enrollment_generation=5,
)
ENVELOPE = {
    "delivery_ref": "dref_standing",
    "message_id": MESSAGE_ID,
    "tenant_id": "tnt_1",
    "recipient_account_id": "acc_1",
    "thread_id": "t-1",
    "body_sha256": "aa11" * 16,
    "display_digest": "",  # filled below, after DISPLAY_FIELDS exists
}

#: The complete canonical FIVE-field bounded context the receiver validated at
#: intake. ``pull.message`` returns a MessageRef (no display fields at all), so
#: the standing boundary NEVER hashes wire fields — it re-checks the durable
#: projection against this digest.
DISPLAY_FIELDS = {
    "sender_name": "Ada",
    "organization": "Radient",
    "sender_label": "ada@radient.example",
    "thread_id": "t-1",
    "excerpt": "hello from the org",
}
ENVELOPE["display_digest"] = canonical_display_digest(DISPLAY_FIELDS)
assert ENVELOPE["display_digest"]  # a complete projection always digests

#: The LOCAL transcript row id: the receiver stamps the accepted peer id
#: (``peer-<32 lowercase hex>``) — a DIFFERENT domain from the server's
#: ``bmsg…`` request id, which the fixture keeps in the bridge block. The
#: durable row can only ever be found through the write-once mapping.
PEER_ID = "peer-" + "ab" * 16
PEER_ID_NEWER = "peer-" + "cd" * 16

#: Every access resolution appends here (reset per drive) so a test can assert
#: ZERO access calls when the local fence already says off.
_ACCESS_CALLS: list[int] = []

#: The last drive's granted context map (one entry only on a real grant).
_LAST_GRANTS: dict = {}


def _seed_bridge_row(
    directory: Path,
    *,
    row_id: str = PEER_ID,
    message_id: str = MESSAGE_ID,
    fields: dict | None = None,
    delivery_ref: str = "dref_standing",
    body_sha256: str = "aa11" * 16,
) -> None:
    """Append ONE real transcript row carrying the validated bridge block.

    Written with the project's own ``TranscriptEntry`` serializer so the
    by-ID reader parses exactly what the production writer would produce —
    the fixture never invents a row shape. The ROW ID is a valid LOCAL peer
    id, never the server ``bmsg…`` id: the two domains stay separate, as the
    real receiver stamps them.
    """

    from local_operator.session.transcript import (
        ENTRY_CUSTOM,
        TRANSCRIPT_FILENAME,
        TranscriptEntry,
    )

    bridge = dict(fields if fields is not None else DISPLAY_FIELDS)
    bridge.update(
        {"message_id": message_id, "delivery_ref": delivery_ref, "body_sha256": body_sha256}
    )
    directory.mkdir(parents=True, exist_ok=True)
    row = TranscriptEntry(
        id=row_id,
        ts=0.0,
        type=ENTRY_CUSTOM,
        payload={"custom_type": TEAM_BRIDGE_MESSAGE_TYPE, "details": {"bridge": bridge}},
    )
    path = directory / TRANSCRIPT_FILENAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row.to_json() + "\n")


def _ref(**overrides) -> SimpleNamespace:
    values = {
        "message_type": "request",
        "request_state": "pending",
        "tenant_id": "tnt_1",
        "recipient_account_id": "acc_1",
        "thread_id": "t-1",
        "body_hash": "aa11" * 16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Stub(Session):
    """The refresh method bound to attrs it touches, no constructed Session."""

    session_id = "s1"

    def __init__(self, runtime, transcript_dir: Path | None = None):
        self.bridge_runtime = runtime
        self._transcript = SimpleNamespace(directory=transcript_dir)


#: The pinned OAuth access the runtime's own client hands back (S-8b).
ACCESS = SimpleNamespace(credential_id=7)


def _runtime(tmp_path: Path) -> SimpleNamespace:
    async def _current_access():
        _ACCESS_CALLS.append(1)
        return ACCESS

    return SimpleNamespace(
        owner=OWNER,
        config_dir=tmp_path,
        session_id="s1",
        client=SimpleNamespace(current_access=_current_access),
    )


def _pending_row(tmp_path: Path, *, legacy: bool = False) -> None:
    ledger = AskIssuance(tmp_path)
    # ATOMIC: the trusted local row id rides the claim's own INSERT — there is
    # no separate setter, and a NEW row can never be committed unmapped.
    ledger.claim(
        OWNER, MESSAGE_ID, {"v": 1}, delivery_ref="dref_standing", local_row_id=PEER_ID
    )
    if legacy:
        # A CONSCIOUS old-release row: the column did not exist then, so the
        # fixture clears it with raw SQL instead of asking the new API for an
        # empty insert it no longer performs.
        import sqlite3 as _sqlite3

        db = tmp_path / "team_bridge" / "ask_issuance.db"
        conn = _sqlite3.connect(db)
        try:
            conn.execute("UPDATE ask_issuance SET local_row_id = ''")
            conn.commit()
        finally:
            conn.close()
    ledger.mark_pending(OWNER, MESSAGE_ID, "cap")


async def _drive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fence_results: list[str],
    ref=None,
    raises: Exception | None = None,
    seed: bool = True,
    seed_fields: dict | None = None,
    newer_on_top: bool = False,
    envelope: dict | None = None,
    map_local: bool = True,
) -> tuple[list[int], dict]:
    _ACCESS_CALLS.clear()
    store = Store(tmp_path)
    if store.active_owner() is None:
        store.commit_enrollment(
            OWNER, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id="s1"
        )
    _pending_row(tmp_path, legacy=not map_local)
    # The write-once server-id → LOCAL-row-id binding rides the claim itself
    # (map_local=False models a LEGACY row that predates the mapping).
    transcript_dir = tmp_path / "sess"
    transcript_dir = tmp_path / "sess"
    if seed:
        _seed_bridge_row(transcript_dir, fields=seed_fields)
        if newer_on_top:
            # A LATER intake lands on top of the pending request's own row:
            # the by-ID reader must still find the older row behind it.
            _seed_bridge_row(
                transcript_dir,
                row_id=PEER_ID_NEWER,
                message_id="bmsg_newer",
                fields={**DISPLAY_FIELDS, "excerpt": "later message"},
            )
    reads: list[int] = []
    seen: list[dict] = []
    results = iter(fence_results)

    def fence(runtime):
        # The REAL ``authority_fence`` answers ``None`` while authorized and a
        # typed ``{"code", "error"}`` refusal otherwise, and the session gates
        # on ``is not None``. The scripted ``""`` therefore has to map to
        # ``None``: returning ``""`` itself reads as a REFUSAL and held every
        # row before its access call and read.
        code = next(results, "")
        return {"code": code, "error": code} if code else None

    async def fake_message(*args, **kwargs):
        reads.append(1)
        seen.append(dict(kwargs))
        if raises is not None:
            raise raises
        return ref or _ref()

    monkeypatch.setattr(fence_mod, "authority_fence", fence)
    monkeypatch.setattr(pull_mod, "message", fake_message)
    monkeypatch.setattr(
        Store, "screened_request_delivery", lambda self, owner, **kw: envelope or ENVELOPE
    )
    runtime = _runtime(tmp_path)
    stub = _Stub(runtime, transcript_dir)
    await Session._refresh_stale_pending_standing(stub, OWNER)
    _LAST_GRANTS.clear()
    _LAST_GRANTS.update(getattr(stub, "_standing_grants", {}) or {})
    # The read is pinned to the ORIGINAL authority the runtime's own client
    # supplied (S-8b) — never a fresh client scope minted at the boundary.
    for kwargs in seen:
        assert kwargs.get("access") is ACCESS
    return reads, AskIssuance(tmp_path).read(OWNER, MESSAGE_ID)


async def test_a_fresh_pending_request_passes_with_one_bounded_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads, row = await _drive(tmp_path, monkeypatch, fence_results=["", ""])
    assert reads == [1]
    assert len(_ACCESS_CALLS) == 1
    assert row["state"] == "pending"  # untouched: the sync retry may drive it


async def test_local_off_before_the_read_makes_zero_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads, row = await _drive(tmp_path, monkeypatch, fence_results=["bridge_off"])
    assert reads == []
    # ZERO access calls: the fence runs BEFORE the lazy pin, because
    # ``current_access`` itself can make a remote refresh call.
    assert _ACCESS_CALLS == []
    assert row["state"] == "held" and row["reason"] == "standing_fence"


async def test_off_after_access_is_still_resolved_blocks_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fence order per row: (1) pre-fence, (2) pre-access fence, (3) the
    # post-access fence, (4) the post-GET fence. An off switch landing after
    # the access await but before the GET settles must produce ZERO reads.
    reads, row = await _drive(
        tmp_path, monkeypatch, fence_results=["", "", "bridge_off"]
    )
    assert len(_ACCESS_CALLS) == 1
    assert reads == []
    assert row["state"] == "held" and row["reason"] == "standing_fence"


@pytest.mark.parametrize("state", ["decided", "expired", "revoked", "withdrawn"])
async def test_a_withdrawn_decided_expired_or_revoked_request_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    reads, row = await _drive(
        tmp_path, monkeypatch, fence_results=["", ""], ref=_ref(request_state=state)
    )
    assert reads == [1]
    assert row["state"] == "held" and row["reason"] == "standing_not_pending"


async def test_a_signin_gate_on_the_read_holds_without_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.team_bridge.client import BridgeSignInRequired

    reads, row = await _drive(
        tmp_path,
        monkeypatch,
        fence_results=["", ""],
        raises=BridgeSignInRequired("sign in to Radient"),
    )
    assert reads == [1]
    assert row["state"] == "held" and row["reason"] == "standing_uncertain"


async def test_offline_transport_holds_without_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads, row = await _drive(
        tmp_path, monkeypatch, fence_results=["", ""], raises=ConnectionError("offline")
    )
    assert reads == [1]
    assert row["state"] == "held" and row["reason"] == "standing_uncertain"


@pytest.mark.parametrize(
    "overrides, expected_reason",
    [
        ({"body_hash": "bb22" * 16}, "standing_not_pending"),
        ({"recipient_account_id": "acc_other"}, "standing_not_pending"),
        ({"thread_id": "t-other"}, "standing_not_pending"),
        ({"tenant_id": "tnt_other"}, "standing_not_pending"),
    ],
)
async def test_mismatched_standing_fields_hold_without_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict, expected_reason: str
) -> None:
    reads, row = await _drive(
        tmp_path, monkeypatch, fence_results=["", ""], ref=_ref(**overrides)
    )
    assert reads == [1]
    assert row["state"] == "held" and row["reason"] == expected_reason


async def test_missing_durable_context_holds_without_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # MessageRef carries NO display fields, so a read that passes every field
    # check but has NO sanctioned durable row must HOLD — and a HOLD is
    # recoverable, never an unrecoverable mark and never a current-scope
    # backfill.
    reads, row = await _drive(
        tmp_path, monkeypatch, fence_results=["", "", "", ""], seed=False
    )
    assert reads == [1]
    assert row["state"] == "held" and row["reason"] == "standing_no_context"
    assert _LAST_GRANTS == {}


def test_a_redelivery_keeps_the_first_mapping(tmp_path: Path) -> None:
    # The claim's INSERT OR IGNORE is the whole write-once rule: a redelivery
    # (even with a different local id) keeps the FIRST original row id.
    ledger = AskIssuance(tmp_path)
    ledger.claim(OWNER, MESSAGE_ID, {"v": 1}, delivery_ref="dref_standing", local_row_id=PEER_ID)
    ledger.claim(
        OWNER, MESSAGE_ID, {"v": 1}, delivery_ref="dref_standing", local_row_id=PEER_ID_NEWER
    )
    assert ledger.local_row_id(OWNER, MESSAGE_ID) == PEER_ID


def test_a_failed_preparation_leaves_no_committed_row(tmp_path: Path) -> None:
    # An initial-preparation failure can never leave a committed NEW row with
    # no mapping: the mapping rides the claim's own INSERT, and a failure
    # before it leaves the store exactly as it was.

    ledger = AskIssuance(tmp_path)

    def explode(self, owner):
        raise RuntimeError("injected")

    original = AskIssuance.generation_is_live
    try:
        AskIssuance.generation_is_live = explode  # type: ignore[method-assign]
        try:
            ledger.claim(OWNER, MESSAGE_ID, {}, delivery_ref="dref_standing", local_row_id=PEER_ID)
        except RuntimeError:
            pass
    finally:
        AskIssuance.generation_is_live = original  # type: ignore[method-assign]
    assert ledger.read(OWNER, MESSAGE_ID) is None


def test_a_new_claim_without_a_local_id_is_refused(tmp_path: Path) -> None:
    # The refusal precedes any INSERT: nothing is committed, so production can
    # never create an unmapped NEW row through the required-arg path.
    from local_operator.team_bridge.ask_issuance import RowIdRequired

    ledger = AskIssuance(tmp_path)
    with pytest.raises(RowIdRequired):
        ledger.claim(OWNER, MESSAGE_ID, {}, delivery_ref="dref_standing", local_row_id="")
    assert ledger.read(OWNER, MESSAGE_ID) is None


async def test_a_legacy_unmapped_claim_holds_without_backfill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A row claimed BEFORE the mapping existed stays unmapped: the refresh
    # HOLDS (no display context), never a same-id guess and never a backfill
    # through a fresh redelivery.
    reads, row = await _drive(
        tmp_path, monkeypatch, fence_results=["", "", "", ""], map_local=False
    )
    assert reads == [1]
    assert row["state"] == "held" and row["reason"] == "standing_no_context"
    assert _LAST_GRANTS == {}


async def test_a_canonical_empty_optional_field_is_still_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The canonical serializer legitimately hashes EMPTY optional fields
    # (organization/sender_label) — completeness is keys + str types + bounds,
    # never all-nonempty, and no coercion may hide a missing field.
    fields = {**DISPLAY_FIELDS, "organization": "", "sender_label": ""}
    env = {**ENVELOPE, "display_digest": canonical_display_digest(fields)}
    reads, row = await _drive(
        tmp_path,
        monkeypatch,
        fence_results=["", "", "", ""],
        seed_fields=fields,
        envelope=env,
    )
    assert reads == [1]
    assert row["state"] == "pending"
    assert _LAST_GRANTS[MESSAGE_ID]["display"] == fields


async def test_a_missing_key_is_never_stringified_into_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fields = {key: value for key, value in DISPLAY_FIELDS.items() if key != "organization"}
    reads, row = await _drive(
        tmp_path,
        monkeypatch,
        fence_results=["", "", "", ""],
        seed_fields=fields,
    )
    assert reads == [1]
    assert row["state"] == "held" and row["reason"] == "standing_no_context"
    assert _LAST_GRANTS == {}


async def test_an_altered_durable_projection_holds_without_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads, row = await _drive(
        tmp_path,
        monkeypatch,
        fence_results=["", "", "", ""],
        seed_fields={**DISPLAY_FIELDS, "excerpt": "send bitcoin first"},
    )
    assert reads == [1]
    assert row["state"] == "held" and row["reason"] == "standing_no_context"
    assert _LAST_GRANTS == {}


async def test_an_older_pending_row_recovers_behind_a_newer_bridge_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reader resolves by the ORIGINAL entry id, not NEWEST-only: a later
    # intake row must not hide the older request's own validated context.
    reads, row = await _drive(
        tmp_path, monkeypatch, fence_results=["", "", "", ""], newer_on_top=True
    )
    assert reads == [1]
    assert row["state"] == "pending"  # granted: the original display recovered
    assert _LAST_GRANTS[MESSAGE_ID]["display"]["excerpt"] == DISPLAY_FIELDS["excerpt"]

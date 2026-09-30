"""The durable credential binding row — cells U1-U7 (slices A and B).

Each test names the claim it pins; the memo's Q6 table is the source and this
file is the receipt. Slice A's cells (U1, U2, U3, U6, U7) are RED on the
pre-slice base BY CONSTRUCTION (this module does not exist there, so the file
does not even collect) — that is the intended receipt, not an accident. Slice B
adds the consult sites and the notice surface (U4, U5): those are RED on the
A-only base because the seams they drive (``set_binding_reader``,
``set_change_handler``, ``Session.journal_credential_binding_change``) do not
exist there. U8, the 0-peer guard that must stay green on the base, lives in
``tests/unit/network/test_credential_binding_zero_peer.py`` for exactly that
reason; the two-root end-to-end cells (E1-E4) live in
``tests/unit/network/test_credential_binding_two_root.py``.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from local_operator.harness.types import Message
from local_operator.network.credentials.store import MeshAwareAuthStore
from local_operator.network.credentials.types import (
    CredentialRef,
    Grant,
    synthetic_credential_id,
)
from local_operator.providers.auth_store import AuthStore
from local_operator.session.credential_binding import (
    POLICY_LOCAL_FIRST,
    POLICY_OWNER,
    SCHEMA_ID,
    SESSION_BINDING_CUSTOM_TYPE,
    CredentialBinding,
    CredentialBindingRecorder,
    recall,
    recall_for,
    record,
    recorder_for_session,
)
from local_operator.session.transcript import BOOKKEEPING_CUSTOM_TYPES, Transcript

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.model.configure import SessionStreamFn
    from local_operator.session.session import Session

SELF = "d_00000000000000000000000000000011"
OWNER = "d_00000000000000000000000000000022"
PROVIDER = "openai"
SESSION = "sess-binding"
#: A label that must never surface anywhere the model can see.
LABEL = "bound-account@example.test"


def _binding(**overrides: Any) -> CredentialBinding:
    fields: dict[str, Any] = {
        "provider": PROVIDER,
        "owner_device": OWNER,
        "owner_device_name": "owner-laptop",
        "credential_id": 42,
        "identity_label": LABEL,
        "writer": "1000:1000",
    }
    fields.update(overrides)
    return CredentialBinding(**fields)


def _grant(*, credential_id: int, owner: str = OWNER, email: str = LABEL) -> Grant:
    return Grant(
        access_token="borrowed-key",
        kind="api_key",
        token_expires_at_ms=0,
        grant_expires_at_ms=int(time.time() * 1000) + 60_000,
        credential_ref=CredentialRef(
            owner_device=owner,
            owner_device_name="owner-laptop",
            provider=PROVIDER,
            kind="api_key",
            credential_id=credential_id,
        ),
        served_by=owner,
        identity={"email": email},
    )


class _StubMesh:
    """The borrower-side seam, with a call counter.

    ``grant_async`` walks ``grants`` so a sibling pick is a real second grant;
    the last grant repeats afterwards, so a repeated identical resolve is
    identical.
    """

    #: Protocol members the exercised paths never read; present so the stub is a
    #: structural ``CredentialSource``.
    self_device: str = SELF
    grants: Any = None
    placement: Any = None

    def __init__(self, grants: list[Grant]) -> None:
        self._grants = list(grants)
        self.grant_calls = 0

    def should_borrow(self, key: str) -> bool:
        return True

    def owner_of(self, key: str) -> str:
        return OWNER

    def owner_label(self, key: str) -> str:
        return "owner-laptop"

    def owner_last_seen_s(self, device: str) -> float | None:
        return 1.0

    async def grant_async(
        self,
        key: str,
        *,
        session_id: str = "",
        model_id: str = "",
        force_refresh: bool = False,
        provider: str = "",
    ) -> Grant:
        self.grant_calls += 1
        return self._grants[min(self.grant_calls - 1, len(self._grants) - 1)]

    def report_sync(
        self,
        key: str,
        *,
        kind: str,
        session_id: str = "",
        model_id: str = "",
        retry_after_ms: int = 0,
        block_scope: str = "",
    ) -> None:
        return None

    def close(self) -> None:
        return None


def _rows(transcript: Transcript) -> list[dict[str, Any]]:
    """Every binding row in the journal, oldest first (file order)."""
    rows: list[dict[str, Any]] = []
    for line in transcript.path.read_text().splitlines():
        raw = json.loads(line)
        payload = raw.get("payload") or {}
        if payload.get("custom_type") == SESSION_BINDING_CUSTOM_TYPE:
            rows.append(raw)
    return rows


def _serve_store(
    root: Path, mesh: _StubMesh, recorder: CredentialBindingRecorder
) -> MeshAwareAuthStore:
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    store = MeshAwareAuthStore(auth, mesh=mesh, config_dir=root)
    store.set_serve_sink(recorder.observe_serve)
    return store


def _recorder(transcript: Transcript, **kwargs: Any) -> CredentialBindingRecorder:
    return CredentialBindingRecorder(
        transcript,
        session_id=SESSION,
        device_id=SELF,
        device_name="my-laptop",
        **kwargs,
    )


# -- U1: the module ----------------------------------------------------------


@pytest.mark.asyncio
async def test_u1_roundtrip_version_gate_and_synthetic_refusal(tmp_path: Path) -> None:
    """U1: round-trip; unknown version → None; string numerics → None; synthetics refused."""
    transcript = Transcript(tmp_path / "sess")
    bound = _binding()
    await record(transcript, bound)

    details = transcript.latest_custom(SESSION_BINDING_CUSTOM_TYPE)
    assert details is not None
    assert details["schema"] == SCHEMA_ID
    assert details["version"] == 1
    assert details["provider"] == PROVIDER
    assert details["owner_device"] == OWNER
    assert details["credential_id"] == 42
    assert details["policy"] == POLICY_LOCAL_FIRST
    assert details["writer"] == "1000:1000"
    assert recall(transcript) == bound

    # An empty label is OMITTED, never fabricated; the reader reads it back as "".
    assert "identity_label" not in _binding(identity_label="").to_details()

    # Unknown/malformed versions and shapes are "no row" — never a defaulted read.
    await transcript.append_custom(SESSION_BINDING_CUSTOM_TYPE, {**details, "version": 2})
    assert recall(transcript) is None
    await transcript.append_custom(SESSION_BINDING_CUSTOM_TYPE, {"version": 1, "provider": ""})
    assert recall(transcript) is None

    # Numeric fields are read STRICTLY, exactly as the spend ledger's reader
    # reads its own (round-1 M1): ``json`` writes a number as a number, so a
    # quoted one is a malformed row — never coerced into a confident binding.
    await transcript.append_custom(
        SESSION_BINDING_CUSTOM_TYPE, {**bound.to_details(), "version": "1"}
    )
    assert recall(transcript) is None
    await transcript.append_custom(
        SESSION_BINDING_CUSTOM_TYPE, {**bound.to_details(), "credential_id": "42"}
    )
    assert recall(transcript) is None

    # Synthetic ids are refused at BOTH gates: construction raises, and a row
    # carrying one reads as absent.
    synthetic = synthetic_credential_id(PROVIDER, OWNER)
    with pytest.raises(ValueError):
        CredentialBinding(provider=PROVIDER, owner_device=OWNER, credential_id=synthetic)
    await transcript.append_custom(
        SESSION_BINDING_CUSTOM_TYPE, {**bound.to_details(), "credential_id": synthetic}
    )
    assert recall(transcript) is None


def test_u1_write_path_is_bookkeeping_and_keeps_the_activity_clock(tmp_path: Path) -> None:
    """U1 (write contract): the append is exempt from the activity clock.

    Driven through ``record()`` — the writer the recorder itself uses — and not
    through ``append_custom`` alone, so deleting ``preserve_mtime=True`` there
    turns this red. The exemption is honoured only when the whole batch is a
    ``BOOKKEEPING_CUSTOM_TYPES`` member, which is why the membership and the
    flag are asserted together.
    """
    from local_operator.session.transcript import _is_bookkeeping_batch

    transcript = Transcript(tmp_path / "sess")

    async def write() -> None:
        await transcript.append_message(Message.user("hello"))
        before = transcript.path.stat().st_mtime
        await asyncio.sleep(0.02)
        await record(transcript, _binding())
        assert transcript.path.stat().st_mtime == pytest.approx(before, abs=1e-6)
        # The row IS there: the exemption must not be achieved by not writing.
        assert transcript.latest_custom(SESSION_BINDING_CUSTOM_TYPE) is not None
        entry = await transcript.append_custom(SESSION_BINDING_CUSTOM_TYPE, {"version": 1})
        assert _is_bookkeeping_batch([entry]) is True

    asyncio.run(write())

    from local_operator.session.session import _PERSISTABLE_CUSTOM_TYPES

    assert SESSION_BINDING_CUSTOM_TYPE in BOOKKEEPING_CUSTOM_TYPES
    assert SESSION_BINDING_CUSTOM_TYPE in _PERSISTABLE_CUSTOM_TYPES


# -- U2: the recorder, first serve and idempotence ---------------------------


@pytest.mark.asyncio
async def test_u2_first_borrow_writes_one_row_and_repeats_are_idempotent(
    tmp_path: Path,
) -> None:
    """U2: one row after the first borrow; none after an identical re-resolve.

    The sink reads the serving row via ``session_credential_id`` and the two
    serves are the two borrow requests that were always made — the recorder
    adds no round trips of its own.
    """
    root = tmp_path / "root"
    root.mkdir()
    transcript = Transcript(tmp_path / "sess")
    recorder = _recorder(transcript)
    mesh = _StubMesh([_grant(credential_id=42)])
    store = _serve_store(root, mesh, recorder)
    try:
        first = await store.get_api_key(PROVIDER, SESSION)
        await recorder.drain()
        assert first == "borrowed-key"

        assert len(_rows(transcript)) == 1
        binding = recall(transcript)
        assert binding is not None
        assert binding.provider == PROVIDER
        assert binding.owner_device == OWNER
        assert binding.owner_device_name == "owner-laptop"
        assert binding.credential_id == 42
        assert binding.identity_label == LABEL
        assert binding.policy == POLICY_LOCAL_FIRST
        assert binding.writer  # stamped by the write path

        await store.get_api_key(PROVIDER, SESSION)
        await recorder.drain()
        assert len(_rows(transcript)) == 1, "an identical re-resolve appended a row"
        assert mesh.grant_calls == 2, "the recorder made a borrow of its own"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_u2_local_branch_records_this_device_and_the_serving_row(
    tmp_path: Path,
) -> None:
    """U2 (local branch): the sink's sticky read names the row that just served.

    The ``[I]`` the memo flags — "the read after a serve returns the serving
    row" — pinned per branch. Local-first wins over the mesh, and the row names
    THIS device with the local row id, never a synthetic.
    """
    root = tmp_path / "root"
    root.mkdir()
    transcript = Transcript(tmp_path / "sess")
    recorder = _recorder(transcript)
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    row = auth.upsert_credential(PROVIDER, {"type": "api_key", "key": "sk-local"})
    mesh = _StubMesh([_grant(credential_id=42)])
    store = MeshAwareAuthStore(auth, mesh=mesh, config_dir=root)
    store.set_serve_sink(recorder.observe_serve)
    try:
        key = await store.get_api_key(PROVIDER, SESSION)
        await recorder.drain()
        assert key == "sk-local"
        binding = recall_for(transcript, PROVIDER)
        assert binding is not None
        assert binding.owner_device == SELF
        assert binding.owner_device_name == "my-laptop"
        assert binding.credential_id == row.id
        assert binding.credential_id > 0
        # The local label is resolved at the Slice B notice site; a borrow
        # already carries one on the grant.
        assert binding.identity_label == ""
        assert mesh.grant_calls == 0
    finally:
        store.close()


# -- U3: the sibling pick ----------------------------------------------------


@pytest.mark.asyncio
async def test_u3_sibling_pick_is_one_change_row_signalled_once(tmp_path: Path) -> None:
    """U3: same owner, new credential_id → exactly one more row, one signal.

    The §4.9 sentence itself is the follow-up slice's copy; this pins the
    CHANGE SIGNAL it will ride — exactly one append whose previous row exists,
    so "emitted once" cannot decay into "emitted on every serve".
    """
    root = tmp_path / "root"
    root.mkdir()
    transcript = Transcript(tmp_path / "sess")
    changes: list[tuple[CredentialBinding, CredentialBinding | None]] = []
    recorder = _recorder(
        transcript,
        on_change=lambda new, previous: changes.append((new, previous)),
    )
    mesh = _StubMesh([_grant(credential_id=42), _grant(credential_id=43)])
    store = _serve_store(root, mesh, recorder)
    try:
        await store.get_api_key(PROVIDER, SESSION)
        await recorder.drain()
        await store.get_api_key(PROVIDER, SESSION)
        await recorder.drain()

        rows = _rows(transcript)
        assert [row["payload"]["details"]["credential_id"] for row in rows] == [42, 43]
        newest = recall(transcript)
        assert newest is not None and newest.credential_id == 43
        assert newest.owner_device == OWNER
        # Two appends; exactly ONE is a change (a previous row exists). The
        # first serve's append has nothing to compare and must not signal.
        assert len(changes) == 2
        assert changes[0][1] is None
        assert changes[1][1] is not None and changes[1][1].credential_id == 42
    finally:
        store.close()


# -- U6: back-compat read ----------------------------------------------------


@pytest.mark.asyncio
async def test_u6_unknown_version_reads_as_absent_and_is_never_rewritten(
    tmp_path: Path,
) -> None:
    """U6: a version-2 row is "no row" and its bytes are never rewritten.

    "Treated as absent" is the whole write rule's vocabulary — no row for the
    provider — so the next serve appends its own v1 snapshot. What must never
    happen is a rewrite/normalisation of the v2 row: sync digests compare
    transcript bytes, so an edit there would corrupt a move.
    """
    transcript = Transcript(tmp_path / "sess")
    future = {
        "schema": SCHEMA_ID,
        "version": 2,
        "provider": PROVIDER,
        "owner_device": "d_future",
        "credential_id": 7,
        "policy": POLICY_LOCAL_FIRST,
    }
    await transcript.append_custom(SESSION_BINDING_CUSTOM_TYPE, future)
    lines = transcript.path.read_text().splitlines()
    v2_line = next(
        line for line in lines if json.loads(line)["payload"]["details"].get("version") == 2
    )
    assert recall(transcript) is None
    assert recall_for(transcript, PROVIDER) is None

    recorder = _recorder(transcript)
    recorder.observe_local(provider=PROVIDER, credential_id=11)
    await recorder.drain()

    text = transcript.path.read_text()
    assert text.count(v2_line) == 1  # byte-identical, exactly once, never rewritten
    rows = _rows(transcript)
    assert [row["payload"]["details"]["credential_id"] for row in rows] == [7, 11]
    newest = recall(transcript)
    assert newest is not None and newest.credential_id == 11


# -- U7: compaction, resume, and the model -----------------------------------


@pytest.mark.asyncio
async def test_u7_row_survives_compaction_and_resume_and_never_reaches_the_model(
    tmp_path: Path,
) -> None:
    """U7: byte-identical through a real compaction and a resume; invisible.

    ``compact_file`` folds superseded collapsible copies and prune rows, and it
    must leave the binding row byte-for-byte — and neither replay path may
    hand the row (or its label) to the model.
    """
    transcript = Transcript(tmp_path / "sess")
    await transcript.append_message(Message.user("hello"))
    await record(transcript, _binding())
    await transcript.append_message(Message.user("more work"))
    # A prune pair gives the compaction something real to fold, so the rewrite
    # below is a rewrite and not a no-op the assertion would pass through.
    await transcript.append_prune("dropped-turn", "trimmed")
    before = [line for line in transcript.path.read_text().splitlines() if LABEL in line]
    assert len(before) == 1

    reclaimed = await transcript.compact_file(min_reclaim_bytes=1)
    assert reclaimed > 0, "the compaction did nothing; this cell would not witness it"
    after = [line for line in transcript.path.read_text().splitlines() if LABEL in line]
    assert after == before  # byte-identical through the rewrite

    resumed = Transcript(transcript.directory)
    binding = recall(resumed)
    assert binding is not None and binding.credential_id == 42
    history = resumed.build_llm_history()
    assert all(
        getattr(message, "custom_type", None) != SESSION_BINDING_CUSTOM_TYPE for message in history
    )
    assert all(LABEL not in str(message) for message in history)


# -- the recorder factory's gate (U8's contrapositive, module present) -------


def test_recorder_for_session_gate_requires_document_and_identity(tmp_path: Path) -> None:
    """The gate: placement document AND identity, else ``None``.

    The gate is deliberately NOT ``build_auth_store``'s borrow predicate (see
    the module docstring/§5.5 owner→borrower direction); this pins the factory
    level of the same guard U8 pins at the config-root level.
    """
    from local_operator.network.credentials import placement as placement_mod
    from local_operator.network.identity import mint

    root = tmp_path / "cfg"
    root.mkdir()
    transcript = Transcript(tmp_path / "sess")

    # A ``None`` root is "nothing to gate on", never the ambient default
    # (round-1 N1): the recorder must read the root it was built for.
    assert recorder_for_session(transcript, config_dir=None, session_id=SESSION) is None
    assert recorder_for_session(transcript, config_dir=root, session_id=SESSION) is None

    document = placement_mod.PlacementDocument("n_binding", root=root, written_by=OWNER)
    document.declare(
        PROVIDER,
        owner_device=OWNER,
        owner_device_name="owner-laptop",
        provider=PROVIDER,
        by=OWNER,
    )
    document.save()
    assert recorder_for_session(transcript, config_dir=root, session_id=SESSION) is None

    identity = mint(root, name="my-laptop")
    recorder = recorder_for_session(transcript, config_dir=root, session_id=SESSION)
    assert recorder is not None
    assert recorder.device_id == identity.device_id
    assert recorder.device_name == "my-laptop"


# -- the wiring seams (the serve-point feeds) --------------------------------


class _FakeSession:
    """Records ``add_dispose_hook`` registrations and binding-change notices."""

    def __init__(self) -> None:
        self.hooks: list[Any] = []
        self.changes: list[tuple[Any, Any, str]] = []

    def add_dispose_hook(self, hook: Any, *, last: bool = False) -> None:
        self.hooks.append(hook)

    def _on_credential_binding_change(self, binding: Any, previous: Any, *, device_id: str) -> None:
        self.changes.append((binding, previous, device_id))


class _FakeStream:
    """Records the recorder install the factory attaches."""

    def __init__(self) -> None:
        self.installed: Any = None

    def set_credential_binding(self, recorder: Any) -> None:
        self.installed = recorder


@pytest.mark.asyncio
async def test_attach_binds_both_feeds_and_folds_drain_into_dispose(tmp_path: Path) -> None:
    """The factory attach: both feeds, both readers, the notice seam, dispose.

    Thin pass-throughs are exactly what breaks silently when a getattr spelling
    drifts, so the wiring is exercised end to end — and with a ``None`` recorder
    it must wire nothing at all (the 0-peer path). Slice B adds two seams to
    this same attach: the store's consult reader and the recorder's notice
    handler (device-bound — the recorder is the only object that knows which
    device "this device" is).
    """
    from local_operator.session_factory import attach_credential_binding

    root = tmp_path / "root"
    root.mkdir()
    transcript = Transcript(tmp_path / "sess")
    recorder = _recorder(transcript)
    mesh = _StubMesh([_grant(credential_id=42)])
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    store = MeshAwareAuthStore(auth, mesh=mesh, config_dir=root)
    stream = _FakeStream()
    session = _FakeSession()
    try:
        attach_credential_binding(
            cast("Session", session), recorder, store, cast("SessionStreamFn", stream)
        )
        assert stream.installed is recorder
        assert session.hooks == [recorder.drain]
        # Slice B: the consult reader is installed on the store...
        assert store._binding_reader == recorder.recall_for  # noqa: SLF001 — the seam under test
        # ...and the notice seam reaches the session, device-bound.
        notice_handler = recorder._on_change  # noqa: SLF001 — the seam under test
        assert notice_handler is not None, "the notice seam was not installed"
        notice_handler(_binding(credential_id=43), _binding())
        assert session.changes, "the notice seam did not reach the session"
        assert session.changes[-1][2] == SELF
        # The sink is live end to end: one borrow serve records one row.
        assert await store.get_api_key(PROVIDER, SESSION) == "borrowed-key"
        await recorder.drain()
        binding = recall(transcript)
        assert binding is not None and binding.credential_id == 42

        empty_session = _FakeSession()
        empty_stream = _FakeStream()
        attach_credential_binding(
            cast("Session", empty_session), None, store, cast("SessionStreamFn", empty_stream)
        )
        assert empty_stream.installed is None
        assert empty_session.hooks == []
    finally:
        store.close()


@pytest.mark.asyncio
async def test_the_boundary_feed_reports_the_serving_row(tmp_path: Path) -> None:
    """The plain-store feed: the boundary note reads the sticky and records.

    A stub store whose ONLY method is ``session_credential_id`` — the same read
    ``preflight_usage`` takes — makes "no added round trips" structural: the
    recorder has nothing else it could call. A second note with the same row is
    one more read and no extra row.
    """
    from local_operator.harness.types import ModelSpec
    from local_operator.model.configure import create_stream_fn

    class _StickyStore:
        def __init__(self, credential_id: int) -> None:
            self.credential_id = credential_id
            self.reads = 0

        def session_credential_id(self, provider: str, session_id: str | None) -> int:
            assert provider == PROVIDER and session_id == SESSION
            self.reads += 1
            return self.credential_id

    transcript = Transcript(tmp_path / "sess")
    recorder = _recorder(transcript)
    store = _StickyStore(7)
    stream = create_stream_fn(cast("AuthStore", store), {}, session_id=SESSION)
    try:
        stream.set_credential_binding(recorder)
        model = ModelSpec(provider="openai", model_id="test-model")
        stream._note_serving_binding(model)  # noqa: SLF001 — the seam under test
        await recorder.drain()
        binding = recall(transcript)
        assert binding is not None
        assert binding.owner_device == SELF
        assert binding.owner_device_name == "my-laptop"
        assert binding.credential_id == 7

        stream._note_serving_binding(model)  # noqa: SLF001
        await recorder.drain()
        assert len(_rows(transcript)) == 1
        assert store.reads == 2
    finally:
        await stream._http.aclose()


# -- U4: the consult sites (the policy carve-out) ----------------------------


def _serve_read_store(
    root: Path, mesh: _StubMesh, recorder: CredentialBindingRecorder
) -> MeshAwareAuthStore:
    """``_serve_store`` plus the consult reader, as ``attach_credential_binding`` sets it.

    The reader is the recorder's fresh transcript read — the same object the
    factory installs — so a cell that passes here exercises the real seam.
    """
    store = _serve_store(root, mesh, recorder)
    store.set_binding_reader(recorder.recall_for)
    return store


class _RefusingMesh(_StubMesh):
    """Not a holder for the key: the borrow rung refuses before it ever dials."""

    def should_borrow(self, key: str) -> bool:
        return False


@pytest.mark.asyncio
async def test_u4_owner_policy_brokers_even_though_local_would_answer(tmp_path: Path) -> None:
    """U4: row ``policy: owner`` + a local key → the broker serves, local does not.

    THE ONE DELIBERATE CARVE-OUT of "local-first is not negotiable" (design
    §2.4): the row records a choice made once, and the resolve honours it
    instead of re-deciding. Same account still serving means no new row — the
    carve-out changes WHICH rung answers, not what the row then says.
    """
    root = tmp_path / "root"
    root.mkdir()
    transcript = Transcript(tmp_path / "sess")
    recorder = _recorder(transcript)
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    auth.upsert_credential(PROVIDER, {"type": "api_key", "key": "sk-local"})
    grant = _grant(credential_id=42)
    mesh = _StubMesh([grant])
    store = _serve_read_store(root, mesh, recorder)
    try:
        await record(transcript, _binding(policy=POLICY_OWNER))
        key = await store.get_api_key(PROVIDER, SESSION)
        await recorder.drain()
        assert key == grant.access_token, "the carve-out must not return the local key"
        assert mesh.grant_calls == 1
        # The served row is unchanged: nothing to append (replacement state).
        assert len(_rows(transcript)) == 1
    finally:
        store.close()


@pytest.mark.asyncio
async def test_u4_owner_policy_has_no_local_fallback(tmp_path: Path) -> None:
    """U4 (failure half): ``owner`` + an unusable broker → ``None``, never local.

    M5 states it: on failure the carve-out does not fall back, because falling
    back would re-make per call the decision the row recorded.
    """
    root = tmp_path / "root"
    root.mkdir()
    transcript = Transcript(tmp_path / "sess")
    recorder = _recorder(transcript)
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    auth.upsert_credential(PROVIDER, {"type": "api_key", "key": "sk-local"})
    mesh = _RefusingMesh([_grant(credential_id=42)])
    store = _serve_read_store(root, mesh, recorder)
    try:
        await record(transcript, _binding(policy=POLICY_OWNER))
        assert await store.get_api_key(PROVIDER, SESSION) is None
        assert mesh.grant_calls == 0, "not a holder: the rung refuses before dialling"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_u4_local_first_still_wins(tmp_path: Path) -> None:
    """U4 (control): the default policy keeps local-first exactly as shipped.

    Same shape as the carve-out cell with one field changed, so a carve-out
    that widened past ``policy: owner`` fails here rather than only in review.
    """
    root = tmp_path / "root"
    root.mkdir()
    transcript = Transcript(tmp_path / "sess")
    recorder = _recorder(transcript)
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    auth.upsert_credential(PROVIDER, {"type": "api_key", "key": "sk-local"})
    mesh = _StubMesh([_grant(credential_id=42)])
    store = _serve_read_store(root, mesh, recorder)
    try:
        await record(transcript, _binding())
        assert await store.get_api_key(PROVIDER, SESSION) == "sk-local"
        assert mesh.grant_calls == 0
    finally:
        store.close()


# -- U5: the account-change notice -------------------------------------------


@pytest.mark.asyncio
async def test_u5_local_capture_writes_the_new_row_and_one_notice_signal(
    tmp_path: Path,
) -> None:
    """U5: row owner=A → local login appears → new row owner=self + ONE signal.

    The capture itself is allowed (D1: record + notify, never block). The
    recorded row is the durable half; the change callback — whose production
    handler journals the notice — is the visible half, and it must fire once.
    """
    from local_operator.network.credentials.messages import render_binding_change_notice

    root = tmp_path / "root"
    root.mkdir()
    transcript = Transcript(tmp_path / "sess")
    changes: list[tuple[CredentialBinding, CredentialBinding | None]] = []
    recorder = _recorder(
        transcript, on_change=lambda new, previous: changes.append((new, previous))
    )
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    grant = _grant(credential_id=42)
    mesh = _StubMesh([grant])
    store = _serve_read_store(root, mesh, recorder)
    try:
        first = await store.get_api_key(PROVIDER, SESSION)
        await recorder.drain()
        assert first == grant.access_token
        bound = recall_for(transcript, PROVIDER)
        assert bound is not None and bound.owner_device == OWNER

        # The device gains its own login mid-session; the next serve is local.
        auth.upsert_credential(PROVIDER, {"type": "api_key", "key": "sk-local"})
        assert await store.get_api_key(PROVIDER, SESSION) == "sk-local"
        await recorder.drain()
        rows = _rows(transcript)
        assert [row["payload"]["details"]["owner_device"] for row in rows] == [OWNER, SELF]
        assert len(changes) == 2, "one first-serve signal, one CHANGE signal"
        new, previous = changes[-1]
        assert new is not None and previous is not None
        assert previous.owner_device == OWNER and new.owner_device == SELF
        notice = render_binding_change_notice(new, previous, self_device=SELF)
        assert "your login on this device" in notice
        assert LABEL in notice, "the account it came FROM is the operator-usable half"

        # A third identical resolve is not news: no row, no signal.
        assert await store.get_api_key(PROVIDER, SESSION) == "sk-local"
        await recorder.drain()
        assert len(_rows(transcript)) == 2
        assert len(changes) == 2
    finally:
        store.close()


@pytest.mark.asyncio
async def test_u5b_the_notice_row_persists_operator_only_and_journals_once(
    tmp_path: Path,
) -> None:
    """U5 (surface half): ONE notice row; operator-visible, never model context.

    The row is written by the Session method the recorder's production handler
    calls. It must persist as a MESSAGE row (that is what the TUI/phone/replay
    folds paint — it is replayed by ``build_llm_history`` like its MCP and
    redaction siblings), and it must be DROPPED by the model-context
    conversion, which is the exclusion that matters. A change with nothing to
    say (a label-only refinement) must journal nothing.
    """
    from local_operator.harness.message_types import SESSION_BINDING_NOTICE_MESSAGE_TYPE
    from local_operator.harness.render import _default_convert_to_llm
    from local_operator.harness.types import StreamEndEvent
    from tests.unit.session.test_session import ScriptedStream, make_session

    session = make_session(tmp_path, ScriptedStream([[StreamEndEvent(stop_reason="stop")]]))
    try:
        new = _binding(
            owner_device=SELF,
            owner_device_name="my-laptop",
            credential_id=7,
            identity_label="",
        )
        previous = _binding()
        await session.journal_credential_binding_change(new, previous, device_id=SELF)
        rows = [json.loads(line) for line in session.transcript.path.read_text().splitlines()]
        notices = [
            row
            for row in rows
            if row.get("payload", {}).get("custom_type") == SESSION_BINDING_NOTICE_MESSAGE_TYPE
        ]
        assert len(notices) == 1
        details = notices[0]["payload"]["details"]
        assert "your login on this device" in details["text"]
        assert details["owner_device"] == SELF
        assert details["credential_id"] == 7

        # Operator surfaces replay the row (a message row, so the fold paints
        # it); the model-context conversion drops it — unlisted in the
        # renderer's allow-list, exactly like the credential-shape notice.
        history = session.transcript.build_llm_history()
        notice_messages = [
            message
            for message in history
            if getattr(message, "custom_type", None) == SESSION_BINDING_NOTICE_MESSAGE_TYPE
        ]
        assert len(notice_messages) == 1
        rendered = _default_convert_to_llm(history)
        assert all(
            getattr(message, "custom_type", None) != SESSION_BINDING_NOTICE_MESSAGE_TYPE
            for message in rendered
        )
        assert all(LABEL not in str(message) for message in rendered)

        # A label-only refinement renders "" — the row still records it, but
        # there is nothing an operator must be told.
        await session.journal_credential_binding_change(
            _binding(identity_label="refined@example.test"), previous, device_id=SELF
        )
        after = session.transcript.path.read_text().splitlines()
        assert len(after) == len(rows)
    finally:
        await session.dispose()


# -- the collapsibility call (#1807 review M2) --------------------------------


@pytest.mark.asyncio
async def test_u7b_two_providers_rows_survive_compaction_uncollapsed(tmp_path: Path) -> None:
    """The superseded-row call: ``mesh_credential_binding.v1`` does NOT join
    ``_COLLAPSIBLE_CUSTOM_TYPES``, because that set keeps ONE newest row per
    TYPE while binding rows are per (type, provider) facts — a naive join would
    drop the other provider's newest row and ``recall_for`` would answer
    ``None`` after the next compaction. Pin the granularity: two providers'
    rows (one superseded) all survive a real rewrite and both still resolve.
    """
    transcript = Transcript(tmp_path / "sess")
    await transcript.append_message(Message.user("hello"))
    await record(transcript, _binding())
    await record(transcript, _binding(credential_id=43))
    await record(transcript, _binding(provider="[redacted]", credential_id=7))
    await transcript.append_prune("dropped-turn", "trimmed")
    reclaimed = await transcript.compact_file(min_reclaim_bytes=1)
    assert reclaimed > 0, "the compaction did nothing; this cell would not witness it"
    openai = recall_for(transcript, PROVIDER)
    anthropic = recall_for(transcript, "[redacted]")
    assert openai is not None and openai.credential_id == 43
    assert anthropic is not None and anthropic.credential_id == 7
    assert len(_rows(transcript)) == 3

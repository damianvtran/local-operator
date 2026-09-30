"""The durable credential binding row — cells U1, U2, U3, U6, U7 (slice A).

Each test names the claim it pins; the memo's Q6 table is the source and this
file is the receipt. Every cell here is RED on the pre-slice base BY
CONSTRUCTION (this module does not exist there, so the file does not even
collect) — that is the intended receipt, not an accident. U8, the 0-peer guard
that must stay green on the base, lives in
``tests/unit/network/test_credential_binding_zero_peer.py`` for exactly that
reason.
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
    """Records ``add_dispose_hook`` registrations."""

    def __init__(self) -> None:
        self.hooks: list[Any] = []

    def add_dispose_hook(self, hook: Any, *, last: bool = False) -> None:
        self.hooks.append(hook)


class _FakeStream:
    """Records the recorder install the factory attaches."""

    def __init__(self) -> None:
        self.installed: Any = None

    def set_credential_binding(self, recorder: Any) -> None:
        self.installed = recorder


@pytest.mark.asyncio
async def test_attach_binds_both_feeds_and_folds_drain_into_dispose(tmp_path: Path) -> None:
    """The factory attach: store sink + stream recorder + drain-on-dispose.

    Thin pass-throughs are exactly what breaks silently when a getattr spelling
    drifts, so the wiring is exercised end to end — and with a ``None`` recorder
    it must wire nothing at all (the 0-peer path).
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

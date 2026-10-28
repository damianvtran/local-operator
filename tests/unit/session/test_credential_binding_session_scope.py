"""A subagent's account must never read as its parent's account change.

THE DEFECT (live session 17cd3e74c66f, 2026-10-28): a subagent's stream is
``SessionStreamFn.fork(child_id)``, which REUSES the parent's
``MeshAwareAuthStore``; that store has ONE serve-sink slot, filled with the
PARENT's ``recorder.observe_serve``; and the sink reported the sticky of
whichever session resolved without saying which session that was. Sessions in
one process legitimately sit on different accounts (ties spread by
``crc32(session_id)``), so every child resolve wrote the child's account into
the parent's binding and the parent's next resolve wrote its own back — 3,876
alternating binding rows and a notice per flip, while the parent's own receipt
never left its account.

The first two cells are RED on the pre-fix base: the parent's recorder sees
``A, B, A, B, …`` and journals a change for every flip after the first. The
ENOSPC cell is RED there too: the failed write was swallowed at DEBUG and
retried on every resolve, so the disk-full state was never surfaced and was
hammered instead.
"""

from __future__ import annotations

import errno
import json
import logging
import time
import zlib
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.credentials.messages import render_binding_change_notice
from local_operator.network.credentials.store import MeshAwareAuthStore
from local_operator.network.credentials.types import CredentialRef, Grant
from local_operator.providers.auth_store import AuthStore
from local_operator.session.credential_binding import (
    SESSION_BINDING_CUSTOM_TYPE,
    CredentialBinding,
    CredentialBindingRecorder,
)
from local_operator.session.transcript import Transcript

SELF = "d_" + "1" * 32
OWNER = "d_" + "2" * 32
PROVIDER = "anthropic"
MODEL = "claude-opus-4-1"
PARENT = "parent"
#: A child whose ``crc32`` tie-spread lands on the OTHER of two rows than the
#: parent's, so the two sessions genuinely sit on different accounts — the live
#: shape, without pinning anything by hand.
CHILD = next(
    candidate
    for candidate in (f"child{i}" for i in range(64))
    if zlib.crc32(candidate.encode()) % 2 != zlib.crc32(PARENT.encode()) % 2
)


def _recorder(transcript: Transcript, changes: list[Any]) -> CredentialBindingRecorder:
    recorder = CredentialBindingRecorder(
        transcript, session_id=PARENT, device_id=SELF, device_name="my-laptop"
    )
    recorder.set_change_handler(lambda new, old: changes.append((new, old)))
    return recorder


def _binding_rows(transcript: Transcript) -> list[dict[str, Any]]:
    if not transcript.path.exists():
        return []
    rows = []
    for line in transcript.path.read_text().splitlines():
        payload = json.loads(line).get("payload") or {}
        if payload.get("custom_type") == SESSION_BINDING_CUSTOM_TYPE:
            rows.append(payload["details"])
    return rows


def _two_oauth_rows(root: Path) -> tuple[AuthStore, list[int]]:
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    expires = int(time.time() * 1000) + 3_600_000
    ids = [
        auth.upsert_credential(
            PROVIDER,
            {
                "type": "oauth",
                "access": f"access-{i}",
                "refresh": f"refresh-{i}",
                "expires": expires,
                "email": f"account-{i}@example.test",
            },
        ).id
        for i in range(2)
    ]
    return auth, ids


@pytest.mark.asyncio
async def test_a_child_resolve_on_the_shared_store_never_reaches_the_parents_binding(
    tmp_path: Path,
) -> None:
    """Parent on A, child on B, interleaved: ONE row (A), ZERO change signals."""
    root = tmp_path / "root"
    root.mkdir()
    auth, _ids = _two_oauth_rows(root)
    store = MeshAwareAuthStore(auth, mesh=None, config_dir=root)
    transcript = Transcript(tmp_path / "sess")
    changes: list[Any] = []
    recorder = _recorder(transcript, changes)
    store.set_serve_sink(recorder.observe_serve)
    try:
        for _ in range(4):
            assert await store.get_oauth_access(PROVIDER, CHILD, model_id=MODEL) is not None
            await recorder.drain()
            assert await store.get_oauth_access(PROVIDER, PARENT, model_id=MODEL) is not None
            await recorder.drain()
        parent_row = auth.session_credential_id(PROVIDER, PARENT)
        child_row = auth.session_credential_id(PROVIDER, CHILD)
        assert parent_row != child_row, "the fixture must put the two sessions on two accounts"
        rows = _binding_rows(transcript)
        assert [row["credential_id"] for row in rows] == [parent_row], rows
        # Only the FIRST serve signals (previous is None, which journals nothing);
        # no account change is ever signalled for the parent.
        assert [old for _new, old in changes] == [None]
    finally:
        store.close()


def _grant(credential_id: int, email: str) -> Grant:
    return Grant(
        access_token=f"borrowed-{credential_id}",
        kind="api_key",
        token_expires_at_ms=0,
        grant_expires_at_ms=int(time.time() * 1000) + 60_000,
        credential_ref=CredentialRef(
            owner_device=OWNER,
            owner_device_name="owner-laptop",
            provider=PROVIDER,
            kind="api_key",
            credential_id=credential_id,
        ),
        served_by=OWNER,
        identity={"email": email},
    )


class _PerSessionMesh:
    """A borrower seam whose owner lends each session its OWN row (A or B)."""

    self_device: str = SELF
    grants: Any = None
    placement: Any = None

    def __init__(self, by_session: dict[str, Grant]) -> None:
        self._by_session = by_session

    def should_borrow(self, key: str) -> bool:
        return True

    def owner_of(self, key: str) -> str:
        return OWNER

    def owner_label(self, key: str) -> str:
        return "owner-laptop"

    def owner_last_seen_s(self, device: str) -> float | None:
        return 1.0

    async def grant_async(self, key: str, *, session_id: str = "", **_: Any) -> Grant:
        return self._by_session[session_id]

    def report_sync(self, key: str, **_: Any) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_a_child_borrow_on_the_shared_store_never_reaches_the_parents_binding(
    tmp_path: Path,
) -> None:
    """The borrow rung keys the same way: a child's lent row is not the parent's."""
    root = tmp_path / "root"
    root.mkdir()
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)  # no local login
    mesh = _PerSessionMesh(
        {PARENT: _grant(104, "parent@example.test"), CHILD: _grant(125, "child@example.test")}
    )
    store = MeshAwareAuthStore(auth, mesh=mesh, config_dir=root)  # type: ignore[arg-type]
    transcript = Transcript(tmp_path / "sess")
    changes: list[Any] = []
    recorder = _recorder(transcript, changes)
    store.set_serve_sink(recorder.observe_serve)
    try:
        for _ in range(4):
            assert await store.get_api_key(PROVIDER, CHILD) == "borrowed-125"
            await recorder.drain()
            assert await store.get_api_key(PROVIDER, PARENT) == "borrowed-104"
            await recorder.drain()
        rows = _binding_rows(transcript)
        assert [(row["owner_device"], row["credential_id"]) for row in rows] == [(OWNER, 104)]
        assert [old for _new, old in changes] == [None]
    finally:
        store.close()


def test_a_report_without_a_session_cannot_be_attributed_and_is_dropped(tmp_path: Path) -> None:
    """An unattributed report (an older sink caller) must not land on the parent."""
    transcript = Transcript(tmp_path / "sess")
    recorder = _recorder(transcript, [])
    recorder.observe_serve(provider=PROVIDER, owner_device="", credential_id=7)
    recorder.observe_serve(provider=PROVIDER, owner_device="", credential_id=7, session_id=CHILD)
    assert recorder._observed == {}  # noqa: SLF001 — nothing was queued for a write
    recorder.observe_serve(provider=PROVIDER, owner_device="", credential_id=7, session_id=PARENT)
    assert set(recorder._observed) == {PROVIDER}  # noqa: SLF001


@pytest.mark.asyncio
async def test_a_full_disk_is_warned_once_and_never_retried_per_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """ENOSPC on the row write: ONE warning, no per-resolve retry, no notice."""
    transcript = Transcript(tmp_path / "sess")
    changes: list[Any] = []
    recorder = _recorder(transcript, changes)
    attempts: list[int] = []

    def _full(*_args: Any, **_kwargs: Any) -> Any:
        attempts.append(1)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(transcript, "_write_entries", _full)
    caplog.set_level(logging.DEBUG, logger="local_operator.session.credential_binding")
    for _ in range(5):
        recorder.observe_serve(
            provider=PROVIDER, owner_device="", credential_id=104, session_id=PARENT
        )
        await recorder.drain()
    # A different account is news and does try the disk once more — still no
    # second warning, and nothing announced for a change that was not recorded.
    recorder.observe_serve(provider=PROVIDER, owner_device="", credential_id=125, session_id=PARENT)
    await recorder.drain()
    warnings = [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING and "credential binding" in record.getMessage()
    ]
    assert len(warnings) == 1, [record.getMessage() for record in caplog.records]
    assert len(attempts) == 2, "the same account must not re-hit a full disk on every resolve"
    assert changes == []


def test_the_notice_names_the_account_so_a_real_switch_is_distinguishable() -> None:
    """Two logins on one device: the sentences differ by the sign-in label."""

    def local(row: int, label: str) -> CredentialBinding:
        return CredentialBinding(
            provider=PROVIDER,
            owner_device=SELF,
            owner_device_name="my-laptop",
            credential_id=row,
            identity_label=label,
        )

    to_b = render_binding_change_notice(
        local(125, "b@example.test"), local(104, "a@example.test"), self_device=SELF
    )
    to_a = render_binding_change_notice(
        local(104, "a@example.test"), local(125, "b@example.test"), self_device=SELF
    )
    assert to_b == ("[session credential] This session is now using b@example.test on my-laptop.")
    assert to_a != to_b and "a@example.test" in to_a
    # A peer-chosen label is rendered as ONE printable line: no escape sequence
    # or newline may repaint the operator's terminal.
    hostile = render_binding_change_notice(
        local(125, "evil\x1b[2J\nfake line"), local(104, ""), self_device=SELF
    )
    assert "\x1b" not in hostile and "\n" not in hostile
    assert "evil fake line on my-laptop" in hostile


@pytest.mark.asyncio
async def test_the_journaled_notice_names_the_local_account_from_its_row(tmp_path: Path) -> None:
    """A local row carries no label; the notice resolves it from the store row."""
    transcript = Transcript(tmp_path / "sess")
    changes: list[Any] = []
    recorder = _recorder(transcript, changes)
    labels = {104: "a@example.test", 125: "b@example.test"}
    recorder.set_label_resolver(labels.__getitem__)
    for row in (104, 125):
        recorder.observe_serve(
            provider=PROVIDER, owner_device="", credential_id=row, session_id=PARENT
        )
        await recorder.drain()
    new, old = changes[-1]
    assert (old.identity_label, new.identity_label) == ("a@example.test", "b@example.test")
    # The stored row stays exactly as before: the label rides the notice only.
    assert all("identity_label" not in row for row in _binding_rows(transcript))
    notice = render_binding_change_notice(new, old, self_device=SELF)
    assert "b@example.test" in notice


def test_two_long_labels_still_render_distinguishable_accounts() -> None:
    """Labels past the display bound stay distinguishable (round 1, D1).

    THE REPRO: two accounts whose labels share their first 64 cells rendered the
    IDENTICAL sentence, so a real second switch read as no change at all. The
    elision now drops the MIDDLE and keeps both ends, and for an email it cuts
    inside the local part — the domain is what two logins on one provider share,
    so a tail cut spent its budget on the part that cannot differ.

    NO DIGEST IS APPENDED, and its absence is pinned: an unkeyed digest of an
    email is dictionary-reversible, and recording it in the transcript row would
    put a linkable pseudonym for the operator's account on every peer replica.
    Two accounts still producing two notices is the row's ``credential_id``'s
    job (see the TUI fold cells); the rendering only has to be good enough for a
    reader, never injective.
    """
    from local_operator.network.credentials.messages import _account_label

    head = "a-very-long-local-part-that-runs-well-past-the-sixty-four-cell-bound-"
    label_a = head + "alpha@example.test"
    label_b = head + "bravo@example.test"

    shown_a = _account_label(label_a)
    shown_b = _account_label(label_b)
    assert shown_a != label_a, "a label past the bound must be shortened"
    assert len(shown_a) <= 64 and len(shown_b) <= 64
    assert shown_a != shown_b, "two accounts must not render one string"
    # Both ENDS survive: the shared prefix and each distinct local-part tail.
    assert shown_a.startswith(head[:10]) and shown_b.startswith(head[:10])
    assert "alpha" in shown_a and "bravo" in shown_b
    assert "@example.test" in shown_a
    # A short label is byte-identical to before: no churn for the common case.
    assert _account_label("a@example.test") == "a@example.test"

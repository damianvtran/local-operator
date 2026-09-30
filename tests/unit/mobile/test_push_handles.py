"""The conversation handle (push/ack-sync S2): mint, resolve, rotate.

WHAT THIS FILE PINS. ADR 0006 §4/§3.1 (@22e2cce2) freezes a machine-local
handle -- ``base64url(HMAC-SHA256(key, conversation_identity))[:22]`` under
``<config root>/push-handle.key`` -- on the aggregate's ``conversations[]``
rows, plus ``GET /api/push/conversation/{handle}`` so a cold tap (a
conversation that is no longer unread) can name its session again. The
failure modes this file exists to make impossible:

* **The handle moves.** Notification ``thread-id``/collapse on the phone
  require one conversation to mint one handle forever: across builds, a
  daemon restart, and later completions. The key persists and the mint is
  deterministic -- and the KEY FILE's bytes are compared too, not just the
  handle, because a re-mint that happened to agree would still be the wrong
  mechanism.
* **An unknown handle is a crash, or an oracle.** Stale after rotation, a
  deleted conversation, or gibberish: a clean 404, never a 500, with the
  same refusal shape the other unknown-id routes use [ADR §3.1].
* **Handles leak beyond the rows that need them.** They ride the aggregate's
  rows ONLY -- never the listing's frame, per-row or nested [ADR §4 records
  the frame-size rejection]; the aggregate's own ``unread`` block is the
  route minus its rows.
* **A read mints.** The key appears on the first MINT, not on a GET;
  deleting the key file is the documented rotation [ADR §4], and no request
  that merely reads can undo it.

Isolated config dir per test (the suite's autouse HOME isolation, plus the
config-dir monkeypatch below -- the same split the S1 suite documents: the
daemon's lazy ``config_dir`` reads follow the monkeypatch, while
``AttentionStore()`` follows the HOME-isolated root). Nothing here touches a
live daemon or store.
"""

from __future__ import annotations

import os
import shutil
import stat
import uuid
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from local_operator.mobile import push_handles
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.session.attention import AttentionStore
from tests.unit.session.test_catalog_read_failures import _store


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *session_ids: str
) -> tuple[Path, MobileDaemon]:
    """An isolated config root with the named listable sessions, and a daemon.

    The same shape as the S1 suite's fixture (and the same ``_store`` import):
    sessions are built through the catalogue suite's own helper so the two
    files agree about what a listable conversation IS.
    """
    cfg = tmp_path / "config"
    if session_ids:
        _store(cfg, *session_ids)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    return cfg, MobileDaemon(port=0, password="pw123")


def _logged_in(daemon: MobileDaemon) -> TestClient:
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    return client


def _publish(session_id: str, *, kind: str = "complete") -> str:
    """One completion for ``session/<id>``; returns its token."""
    token = str(uuid.uuid4())
    AttentionStore().publish(f"session/{session_id}", token, "result", kind)
    return token


def _refresh(daemon: MobileDaemon) -> None:
    """Force the next read to rebuild, as a structural change does.

    The aggregate is served from the listing's snapshot, so anything that
    changes the store or the key file must invalidate it or the next read
    serves the cached build for up to its TTL.
    """
    daemon.table.invalidate_summaries_cache()


def _key_path(cfg: Path) -> Path:
    return cfg / push_handles.PUSH_HANDLE_KEY_NAME


def _handle_for(client: TestClient, session_id: str) -> str:
    """The handle the aggregate serves for ``session_id`` right now."""
    body = client.get("/api/attention/unread").json()
    assert body["degraded"] == [], "fixture sanity: the store read is healthy"
    rows = {row["session_id"]: row for row in body["conversations"]}
    assert session_id in rows, f"fixture sanity: {session_id} is an unread row"
    handle = rows[session_id]["push_handle"]
    assert isinstance(handle, str)
    return handle


# ---------------------------------------------------------------------------
# The mint: a pinned recipe, a persisted private key, per-machine scope
# ---------------------------------------------------------------------------


def test_the_mint_recipe_is_pinned_to_its_exact_bytes() -> None:
    """``base64url(HMAC-SHA256(key, identity))[:22]`` -- pinned literally.

    The handle is a wire and on-disk contract (``thread-id``/collapse, deep
    links), so the vectors below are literals, cross-checked against
    ``openssl dgst -sha256 -mac HMAC``: a refactor that changes the encoding,
    the truncation, or the identity spelling fails HERE rather than silently
    on a phone whose pending pushes stop resolving. Determinism is pinned
    beside them: the same input must never produce two answers.
    """
    key = bytes(range(32))
    handle = push_handles.handle_for(key, "session/aaaaaaaaaaaa")
    assert handle == "uLjm7vldyGosNZLlv19bjo"
    assert push_handles.handle_for(key, "session/bbbbbbbbbbbb") == "Mbi9p4Td5VXVnkh0m-wYyZ"
    # The identity's namespace is part of the input, not decoration.
    assert push_handles.handle_for(key, "agent/aaaaaaaaaaaa") != handle
    # Same input, same output: the mint holds no state.
    assert push_handles.handle_for(key, "session/aaaaaaaaaaaa") == handle
    assert len(handle) == 22
    assert set(handle) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def test_the_key_is_private_32_bytes_and_written_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key file: created on first mint, 0600, 32 bytes, never rewritten.

    "Minted on first use" is pinned as laziness in both directions: no key
    before the first build that serves a row, and the next build neither
    changes the handles nor the file's BYTES. A resting daemon that rewrote
    its key would invalidate every pending push on the way past.
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)
    _publish("aaaaaaaaaaaa")
    _refresh(daemon)

    assert not _key_path(cfg).exists(), "the key appeared before any handle was needed"
    first = _handle_for(client, "aaaaaaaaaaaa")
    raw = _key_path(cfg).read_bytes()
    assert len(raw) == 32
    assert stat.S_IMODE(_key_path(cfg).stat().st_mode) == 0o600

    _refresh(daemon)
    assert _handle_for(client, "aaaaaaaaaaaa") == first
    assert _key_path(cfg).read_bytes() == raw


def test_two_machines_mint_different_handles_for_the_same_conversation(
    tmp_path: Path,
) -> None:
    """Per machine, not per account: two config roots, one identity, two answers.

    Both the key and the handle differ -- deterministic per root (mint twice
    over the same root agrees), independent across roots [ADR §4: two
    machines' "same" conversation IS a different conversation].
    """
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    for root in (root_a, root_b):
        (root / "sessions" / "aaaaaaaaaaaa").mkdir(parents=True)

    first_a = push_handles.conversation_handles(root_a, ["aaaaaaaaaaaa"])[0]
    assert push_handles.conversation_handles(root_a, ["aaaaaaaaaaaa"])[0] == first_a
    handle_b = push_handles.conversation_handles(root_b, ["aaaaaaaaaaaa"])[0]
    assert handle_b != first_a
    assert _key_path(root_a).read_bytes() != _key_path(root_b).read_bytes()


def test_a_daemon_restart_serves_the_same_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stability across a daemon restart, and across a later completion.

    A fresh ``MobileDaemon`` over the same config root is the restart the
    ADR names: it must serve the identical handle, because the handle lives
    with the KEY on disk, not in the process. A later completion on the same
    conversation must not move it either.
    """
    _cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)
    _publish("aaaaaaaaaaaa")
    _refresh(daemon)
    first = _handle_for(client, "aaaaaaaaaaaa")

    restarted = MobileDaemon(port=0, password="pw123")
    client2 = _logged_in(restarted)
    assert _handle_for(client2, "aaaaaaaaaaaa") == first

    _publish("aaaaaaaaaaaa")
    _refresh(restarted)
    assert _handle_for(client2, "aaaaaaaaaaaa") == first


# ---------------------------------------------------------------------------
# The resolve route: round trip, cold tap, refusals, the gate
# ---------------------------------------------------------------------------


def test_resolve_round_trips_an_unread_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aggregate row -> handle -> session id, and the row shape stays frozen."""
    _cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    client = _logged_in(daemon)
    _publish("aaaaaaaaaaaa")
    _refresh(daemon)

    handle = _handle_for(client, "aaaaaaaaaaaa")
    resolved = client.get(f"/api/push/conversation/{handle}")
    assert resolved.status_code == 200
    assert resolved.json() == {"session_id": "aaaaaaaaaaaa"}
    # The row is the ADR's shape: the handle rides it, nothing else changed.
    row = client.get("/api/attention/unread").json()["conversations"][0]
    assert set(row) == {"session_id", "push_handle", "completion_token", "kind", "revision"}


def test_resolve_serves_an_acknowledged_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cold tap: after the ack, the conversation is off the unread set.

    Resolving exactly that state is the route's reason to exist -- a tap on
    a notification whose conversation was already acknowledged must still
    land on the right session, not 404 [ADR §3.1].
    """
    _cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)
    token = _publish("aaaaaaaaaaaa")
    _refresh(daemon)
    handle = _handle_for(client, "aaaaaaaaaaaa")

    AttentionStore().acknowledge("session/aaaaaaaaaaaa", token)
    _refresh(daemon)
    assert client.get("/api/attention/unread").json()["conversations"] == []
    assert client.get(f"/api/push/conversation/{handle}").json() == {"session_id": "aaaaaaaaaaaa"}


def test_handles_ride_the_aggregate_rows_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The field lives on ``/api/attention/unread``'s ``conversations[]`` ONLY.

    The per-row alternative on the listing was rejected on frame size
    [ADR §4], and the listing's ``unread`` block is the aggregate minus its
    rows -- so the field must appear NOWHERE in the listing frame, asserted
    on the raw body so a future refactor cannot smuggle it in nested.
    """
    _cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)
    _publish("aaaaaaaaaaaa")
    _refresh(daemon)

    listing = client.get("/api/sessions")
    assert "push_handle" not in listing.text
    unread = client.get("/api/attention/unread")
    assert "push_handle" in unread.text
    assert len(unread.json()["conversations"][0]["push_handle"]) == 22


def test_unknown_handles_are_a_clean_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never a 500, and never an oracle: every unresolvable shape 404s alike.

    A well-shaped handle nothing mints, a wrong-length one, and a string
    that is no handle at all get the same refusal -- the refusal must not
    become a way to probe which handles exist, and a 500 would turn every
    stale push into a crash in the app instead of its list fallback.
    """
    _cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)
    _publish("aaaaaaaaaaaa")
    _refresh(daemon)

    for candidate in ("A" * 22, "short", "This!is-not-a-handle"):
        response = client.get(f"/api/push/conversation/{candidate}")
        assert response.status_code == 404, candidate
        assert response.json() == {"error": "unknown conversation handle"}


def test_a_deleted_conversation_404s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A conversation that no longer exists is unknown, not an error."""
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)
    _publish("aaaaaaaaaaaa")
    _refresh(daemon)
    handle = _handle_for(client, "aaaaaaaaaaaa")

    shutil.rmtree(cfg / "sessions" / "aaaaaaaaaaaa")
    assert client.get(f"/api/push/conversation/{handle}").status_code == 404


def test_a_mail_spool_conversation_without_a_transcript_yet_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is the origin predicate -- not the transcript detail.

    S1's remediation dropped the per-row transcript check from the aggregate
    because a mail-spool conversation with no transcript yet (activity =
    ``inbox.jsonl``) is still a row the user can be shown -- and a row the
    aggregate mints a handle for. A resolve gate that re-derived the
    transcript detail would 404 exactly the handle this cell just read off
    the aggregate [S1 remediation, review round 1 MAJOR-1].
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch)
    client = _logged_in(daemon)
    spool = cfg / "sessions" / "abcdef123456"
    spool.mkdir(parents=True)
    (spool / "inbox.jsonl").write_text('{"type": "message", "content": "queued"}\n')
    _publish("abcdef123456")
    _refresh(daemon)

    handle = _handle_for(client, "abcdef123456")
    assert client.get(f"/api/push/conversation/{handle}").json() == {"session_id": "abcdef123456"}


def test_a_never_minted_conversation_404s_even_though_it_is_mintable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mint is population-blind; the ROUTE carries the population.

    A subagent transcript's handle can be computed by anyone who can read
    the key -- and it must still refuse, because the aggregate never serves
    that conversation, so no legitimate push can carry it. The gate is the
    same user-facing predicate the listing applies
    (``is_user_session_origin``, via ``_live_generation_is_user_facing``).
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)
    subagent = cfg / "sessions" / "dddddddddddd"
    subagent.mkdir(parents=True)
    (subagent / "transcript.jsonl").write_text("{}\n")
    (subagent / "origin.json").write_text('{"origin": "subagent"}')

    mintable = push_handles.conversation_handles(cfg, ["dddddddddddd"])[0]
    assert client.get(f"/api/push/conversation/{mintable}").status_code == 404


def test_the_resolver_is_gated_exactly_like_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No cookie, no resolve -- the same refusal the listing and badge give."""
    _cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = TestClient(build_app(daemon), follow_redirects=False)

    listing = client.get("/api/sessions")
    resolve = client.get("/api/push/conversation/" + "A" * 22)
    assert listing.status_code == resolve.status_code == 401
    assert resolve.json() == listing.json() == {"error": "authentication required"}


# ---------------------------------------------------------------------------
# Rotation: an operation, not a side effect
# ---------------------------------------------------------------------------


def test_a_stale_key_404s_old_handles_and_the_next_build_mints_new_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replace the key file and every old handle stops resolving.

    The documented recovery path [ADR §4]: pending pushes 404 and the app
    falls back to its list. The next build picks the new key up
    deterministically (the mint reads the key per build) and the same
    conversation gets a NEW handle.
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)
    _publish("aaaaaaaaaaaa")
    _refresh(daemon)
    old = _handle_for(client, "aaaaaaaaaaaa")

    _key_path(cfg).write_bytes(os.urandom(32))
    os.chmod(_key_path(cfg), 0o600)  # the write above bypasses the mint path

    assert client.get(f"/api/push/conversation/{old}").status_code == 404
    _refresh(daemon)
    new = _handle_for(client, "aaaaaaaaaaaa")
    assert new != old
    assert client.get(f"/api/push/conversation/{new}").json() == {"session_id": "aaaaaaaaaaaa"}
    assert client.get(f"/api/push/conversation/{old}").status_code == 404


def test_deleting_the_key_is_the_documented_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete the file; a resolve does not bring it back; the next mint does.

    Two rules in one cell: (a) rotation is DELETING the key -- old handles
    404 from then on; (b) no GET ever mints, so a request that merely reads
    cannot resurrect (or silently rotate) the handles it only looks up.
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)
    _publish("aaaaaaaaaaaa")
    _refresh(daemon)
    old = _handle_for(client, "aaaaaaaaaaaa")

    _key_path(cfg).unlink()
    assert client.get(f"/api/push/conversation/{old}").status_code == 404
    assert not _key_path(cfg).exists(), "a resolve minted the key it only looked up"

    _refresh(daemon)
    fresh = _handle_for(client, "aaaaaaaaaaaa")
    assert _key_path(cfg).exists(), "the next mint did not create a key"
    assert fresh != old
    assert client.get(f"/api/push/conversation/{old}").status_code == 404
    assert client.get(f"/api/push/conversation/{fresh}").status_code == 200

"""The move's working-set consent gate (docs/design/move-working-set.md §4 P1/P3, §11.1).

The operator's complaint (2026-11-18): "people could use the move function and then
realize all the work is not available". Each cell drives the move the way the CLI
does — ``mobility.request_move`` through this device's relay control socket, over the
in-process two-relay rig from ``test_mobility`` — with a REAL git worktree as the
session's cwd, and pins one half of the contract:

* a non-empty working set without consent is refused ``working_set_unconfirmed``,
  ``changed: False``, with the manifest, and the source is untouched (no journal);
* the dry run returns the manifest and its scope digest and moves nothing;
* a confirmed move proceeds and its receipt states, per worktree, that nothing was
  transferred (``transferred: false``, ``not_transferred``) — PR A's honest arrival;
* a scope change between preflight and move (a new withheld secret) is refused
  ``working_set_changed`` with the fresh manifest; a byte-only change passes;
* a digest confirmed for ANOTHER device id does not authorise this one;
* ``--without-working-set`` is the explicit "leave it here";
* an empty working set needs no consent and its receipt has no ``working_set`` key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.network import mobility, workset
from local_operator.session.placement import read_handoff_journal
from tests.unit.network.test_mobility import (  # noqa: F401 — `devices` is a fixture
    SESSION,
    Devices,
    _owned_session,
    devices,
)
from tests.unit.network.test_relay_e2e import _pair_settled
from tests.unit.network.workset_rig import Rig, make_rig, point_session_at


@pytest.fixture()
def pair(request: pytest.FixtureRequest) -> Devices:
    """Both relays serving a control socket (the recall is issued by B)."""
    value: Devices = request.getfixturevalue("devices")
    _server_a, server_b, _host, _port = value
    server_b.bind_control()
    server_b.start()
    return value


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    """The working set, with HOME pointed at its home so paths are ``~/``-relative."""
    built = make_rig(tmp_path / "ws")
    monkeypatch.setenv("HOME", str(built.home))
    return built


def _move(server: Any, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> dict[str, Any]:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server.root))
    return dict(mobility.request_move(SESSION, root=server.root, **kwargs))


def _session_with_worktree(pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig) -> Path:
    server_a, server_b, _host, _port = pair
    # B's own settings make it dialable from A's member row, so the offload cells
    # (A invites B) work over the same pair the recall cells use.
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    source = _owned_session(server_a)
    point_session_at(source, rig.feat)
    return source


def test_an_unconfirmed_working_set_is_refused_and_nothing_changes(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig
) -> None:
    server_a, server_b, _host, _port = pair
    source = _session_with_worktree(pair, monkeypatch, rig)

    refused = _move(server_b, monkeypatch, to="local")

    assert refused["ok"] is False, refused
    assert refused["code"] == "working_set_unconfirmed", refused
    assert refused["changed"] is False
    block = refused["working_set"]
    assert block["destination"]["device_id"] == server_b.identity.device_id
    assert [row["path"] for row in block["worktrees"]] == ["~/proj/.worktrees/feat"]
    assert block["worktrees"][0]["state"] == "planned"
    assert block["carries_files"] is False
    # THE SOURCE IS UNTOUCHED: the directory is still here and nothing journaled.
    assert source.is_dir()
    assert read_handoff_journal(server_a.root) == {}
    assert not (server_b.root / "sessions" / SESSION).exists()


def test_the_dry_run_returns_the_manifest_and_moves_nothing(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig
) -> None:
    server_a, server_b, _host, _port = pair
    source = _session_with_worktree(pair, monkeypatch, rig)

    preflight = _move(server_b, monkeypatch, to="local", dry_run=True)

    assert preflight["ok"] is True and preflight["dry_run"] is True, preflight
    assert preflight["confirmation_required"] is True
    block = preflight["working_set"]
    assert block["scope_digest"].startswith("sha256:")
    assert block["totals"]["worktrees"] == 1
    assert source.is_dir()
    assert read_handoff_journal(server_a.root) == {}


def test_a_confirmed_move_proceeds_and_says_the_work_was_not_transferred(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig
) -> None:
    _server_a, server_b, _host, _port = pair
    source = _session_with_worktree(pair, monkeypatch, rig)
    digest = _move(server_b, monkeypatch, to="local", dry_run=True)["working_set"]["scope_digest"]
    status_before = (rig.feat / "b.txt").read_bytes()

    moved = _move(server_b, monkeypatch, to="local", confirm_working_set=digest)

    assert moved["ok"] is True, moved
    assert moved["state"] == "completed"
    assert not source.exists()
    block = moved["working_set"]
    assert block["transferred"] is False
    assert block["consent"] == "confirmed"
    assert block["scope_digest"] == digest
    assert [(row["state"], row["reason"]) for row in block["worktrees"]] == [
        ("left_behind", "not_transferred")
    ]
    # Never deletes: the worktree on the source is exactly as it was.
    assert (rig.feat / "b.txt").read_bytes() == status_before
    assert (rig.feat / "notes.txt").is_file()


def test_a_scope_change_after_the_preflight_is_refused_with_the_fresh_manifest(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig
) -> None:
    server_a, server_b, _host, _port = pair
    source = _session_with_worktree(pair, monkeypatch, rig)
    digest = _move(server_b, monkeypatch, to="local", dry_run=True)["working_set"]["scope_digest"]
    # A NEW withheld secret appears in the confirmed worktree.
    (rig.feat / "id_rsa").write_text("not really a key\n", encoding="utf-8")

    refused = _move(server_b, monkeypatch, to="local", confirm_working_set=digest)

    assert refused["code"] == "working_set_changed", refused
    assert refused["changed"] is False
    fresh = refused["working_set"]
    assert fresh["scope_digest"] != digest
    assert "~/proj/.worktrees/feat/id_rsa" in [item["path"] for item in fresh["excluded"]]
    assert source.is_dir()
    assert read_handoff_journal(server_a.root) == {}


def test_a_byte_only_change_in_a_confirmed_worktree_still_moves(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig
) -> None:
    _server_a, server_b, _host, _port = pair
    _session_with_worktree(pair, monkeypatch, rig)
    digest = _move(server_b, monkeypatch, to="local", dry_run=True)["working_set"]["scope_digest"]
    # Same file, a few more bytes: the scope (which trees, what is withheld) is equal.
    with (rig.feat / "b.txt").open("a", encoding="utf-8") as handle:
        handle.write("one more line\n")

    moved = _move(server_b, monkeypatch, to="local", confirm_working_set=digest)

    assert moved["ok"] is True, moved


def test_a_digest_confirmed_for_another_device_does_not_authorise_this_one(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig
) -> None:
    server_a, server_b, _host, _port = pair
    source = _session_with_worktree(pair, monkeypatch, rig)
    elsewhere = workset.compute_manifest(
        server_a.root,
        SESSION,
        to_device="d_someone_else",
        home=rig.home,
        cwd_hint=str(rig.feat),
    )

    refused = _move(
        server_b, monkeypatch, to="local", confirm_working_set=workset.scope_digest(elsewhere)
    )

    assert refused["code"] == "working_set_changed", refused
    assert source.is_dir()


def test_without_working_set_is_the_explicit_leave_it_here(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig
) -> None:
    _server_a, server_b, _host, _port = pair
    _session_with_worktree(pair, monkeypatch, rig)

    moved = _move(server_b, monkeypatch, to="local", without_working_set=True)

    assert moved["ok"] is True, moved
    assert moved["working_set"]["consent"] == "without"
    assert moved["working_set"]["transferred"] is False


def test_an_offload_is_gated_on_the_owner_before_the_peer_is_asked(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig
) -> None:
    server_a, server_b, _host, _port = pair
    source = _session_with_worktree(pair, monkeypatch, rig)

    refused = _move(server_a, monkeypatch, to=server_b.identity.device_id)

    assert refused["code"] == "working_set_unconfirmed", refused
    assert refused["working_set"]["destination"]["device_id"] == server_b.identity.device_id
    assert source.is_dir()
    assert read_handoff_journal(server_a.root) == {}
    # The confirmed offload goes through and reports the same honest arrival.
    digest = refused["working_set"]["scope_digest"]
    moved = _move(server_a, monkeypatch, to=server_b.identity.device_id, confirm_working_set=digest)
    assert moved["ok"] is True, moved
    assert moved["working_set"]["transferred"] is False


def test_an_empty_working_set_needs_no_consent_and_adds_no_receipt_key(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig, tmp_path: Path
) -> None:
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    plain = tmp_path / "ws" / "home" / "not-a-repo"
    plain.mkdir()
    point_session_at(source, plain)

    preflight = _move(server_b, monkeypatch, to="local", dry_run=True)
    moved = _move(server_b, monkeypatch, to="local")

    assert preflight["confirmation_required"] is False
    assert moved["ok"] is True, moved
    # FROZEN-CONTRACT PARITY: a chat-only move's receipt is unchanged.
    assert "working_set" not in moved


def test_an_unreadable_working_set_fails_closed_with_working_set_unknown(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig
) -> None:
    """Manager ruling 2026-11-18: a manifest that cannot be computed is the silent-loss
    case itself, so it refuses (naming the step) instead of reading as "clean"."""
    server_a, server_b, _host, _port = pair
    source = _session_with_worktree(pair, monkeypatch, rig)
    # A REAL failure, not a stub: a corrupt index makes ``git status`` exit 128.
    admin = Path(rig.feat.joinpath(".git").read_text(encoding="utf-8").split(":", 1)[1].strip())
    admin.joinpath("index").write_bytes(b"DIRC-corrupt")

    preflight = _move(server_b, monkeypatch, to="local", dry_run=True)
    refused = _move(server_b, monkeypatch, to="local")
    journal_after_refusal = read_handoff_journal(server_a.root)
    escaped = _move(server_b, monkeypatch, to="local", without_working_set=True)

    assert preflight["confirmation_required"] is True
    assert preflight["working_set"]["state"] == "unknown"
    assert refused["code"] == "working_set_unknown", refused
    assert refused["changed"] is False
    assert refused["working_set"]["failure"] == {"step": "status", "repo": "~/proj/.worktrees/feat"}
    assert journal_after_refusal == {}
    # The explicit escape still moves the conversation, and says nothing was carried.
    assert escaped["ok"] is True, escaped
    assert escaped["working_set"]["consent"] == "without"
    assert escaped["working_set"]["transferred"] is False
    assert not source.exists()


def test_a_keep_copy_is_not_gated_but_says_it_arrives_without_the_work(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig
) -> None:
    _server_a, server_b, _host, _port = pair
    source = _session_with_worktree(pair, monkeypatch, rig)

    copied = _move(server_b, monkeypatch, to="local", keep=True)

    assert copied["ok"] is True, copied
    block = copied["working_set"]
    assert block["consent"] == "not_applicable_keep"
    assert block["transferred"] is False
    assert [(row["path"], row["state"]) for row in block["worktrees"]] == [
        ("~/proj/.worktrees/feat", "left_behind")
    ]
    assert source.is_dir()  # a keep copy never retires the source

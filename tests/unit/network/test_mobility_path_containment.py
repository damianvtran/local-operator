"""Peer-supplied move ids and on-disk handoff paths stay inside their stores.

The receiving handler is exercised separately from the filesystem guards: a frame
must be refused before a journal or worker exists, and a corrupted internal record
must not turn the later cleanup or promotion into an arbitrary filesystem action.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.network import mobility, sync
from local_operator.network.types import MeshRefusal
from local_operator.session.placement import read_handoff_journal
from tests.unit.network.test_mobility import (  # noqa: F401
    Devices,
    _move,
    _owned_session,
    pair,
)
from tests.unit.network.test_relay_e2e import _pair_settled, devices  # noqa: F401


def _server(root: Path) -> Any:
    return SimpleNamespace(root=root)


def _victim(root: Path) -> tuple[Path, Path]:
    directory = root / "victim"
    directory.mkdir()
    sentinel = directory / "keep.txt"
    sentinel.write_text("do not remove", encoding="utf-8")
    return directory, sentinel


@pytest.mark.parametrize(
    "session_id",
    [
        "../escaped-invite",
        "/absolute/invite",
        "",
        ".",
        "..",
        "nested/id",
        r"C:\victim",
        r"\\host\share",
    ],
)
def test_receiving_handler_refuses_an_unsafe_invite_before_writes(
    pair: Devices,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    session_id: str,
) -> None:
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    workers: list[object] = []

    class NoWorker:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            workers.append(self)

        def start(self) -> None:
            raise AssertionError("an unsafe invite started a move worker")

    monkeypatch.setattr(mobility.threading, "Thread", NoWorker)
    handler = server_b._handlers["net_session_move"]  # noqa: SLF001
    link = cast(Any, SimpleNamespace(device_id=server_a.identity.device_id))

    with pytest.raises(MeshRefusal, match="one safe conversation id") as refused:
        handler(link, {"phase": "invite", "session_id": session_id})

    assert refused.value.code == "bad_request"
    assert workers == []
    assert read_handoff_journal(server_b.root) == {}


def test_existing_opaque_session_id_charset_remains_valid() -> None:
    for session_id in ("legacy-session.v1", "conversation with spaces", "session~2026"):
        assert mobility._validated_session_id(session_id) == session_id


def test_recall_refuses_an_unsafe_id_before_ownership_or_disk_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _server(tmp_path)

    def unexpected_access(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("ownership was checked before the id was validated")

    monkeypatch.setattr(mobility, "_owned_here", unexpected_access)

    refused = mobility._recall(server, "../outside", keep=False, wait_s=0.0)

    assert refused["ok"] is False
    assert refused["code"] == "bad_request"


def test_absolute_victim_survives_a_corrupted_rollback_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "config"
    root.mkdir()
    victim, sentinel = _victim(tmp_path)
    monkeypatch.setattr(mobility, "_audit", lambda *args, **kwargs: None)

    mobility._roll_back_destination(_server(root), str(victim), victim)

    assert sentinel.read_text(encoding="utf-8") == "do not remove"
    assert victim.is_dir()


def test_rollback_refuses_a_staging_path_outside_the_real_staging_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "config"
    root.mkdir()
    victim, sentinel = _victim(tmp_path)
    monkeypatch.setattr(mobility, "_audit", lambda *args, **kwargs: None)

    mobility._roll_back_destination(_server(root), "opaque-session", victim)

    assert sentinel.read_text(encoding="utf-8") == "do not remove"


def test_rollback_refuses_a_symlinked_staging_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "config"
    staging_root = root / "network" / "staging"
    staging_root.mkdir(parents=True)
    victim, sentinel = _victim(tmp_path)
    staging = staging_root / "opaque-session"
    staging.symlink_to(victim, target_is_directory=True)
    monkeypatch.setattr(mobility, "_audit", lambda *args, **kwargs: None)

    mobility._roll_back_destination(_server(root), "opaque-session", staging)

    assert sentinel.read_text(encoding="utf-8") == "do not remove"
    assert victim.is_dir()
    assert staging.is_symlink()


def test_rollback_refuses_a_symlinked_staging_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "config"
    root.mkdir()
    victim, sentinel = _victim(tmp_path)
    staging_root = root / "network" / "staging"
    staging_root.parent.mkdir()
    staging_root.symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setattr(mobility, "_audit", lambda *args, **kwargs: None)

    mobility._roll_back_destination(_server(root), "victim", staging_root / "victim")

    assert sentinel.read_text(encoding="utf-8") == "do not remove"
    assert victim.is_dir()


@pytest.mark.parametrize("target_id", ["../escaped-promoted", "ABSOLUTE"])
def test_promote_refuses_a_target_outside_sessions(tmp_path: Path, target_id: str) -> None:
    root = tmp_path / "config"
    staged = root / "network" / "staging" / "staged"
    staged.mkdir(parents=True)
    if target_id == "ABSOLUTE":
        target_id = str(tmp_path / "absolute-promoted")
        escaped = Path(target_id)
    else:
        escaped = root / "sessions" / target_id

    outcome = mobility._promote(_server(root), staged, target_id)

    assert outcome.ok is False
    assert not escaped.exists()
    assert staged.is_dir()


def test_promote_refuses_a_symlinked_sessions_root(tmp_path: Path) -> None:
    root = tmp_path / "config"
    staged = root / "network" / "staging" / "staged"
    staged.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    sessions = root / "sessions"
    sessions.symlink_to(outside, target_is_directory=True)

    outcome = mobility._promote(_server(root), staged, "staged")

    assert outcome.ok is False
    assert not (outside / "staged").exists()
    assert staged.is_dir()


def test_promote_refuses_a_symlinked_staging_source(tmp_path: Path) -> None:
    root = tmp_path / "config"
    outside = tmp_path / "outside"
    outside.mkdir()
    staging_root = root / "network" / "staging"
    staging_root.mkdir(parents=True)
    staged = staging_root / "staged"
    staged.symlink_to(outside, target_is_directory=True)

    outcome = mobility._promote(_server(root), staged, "staged")

    assert outcome.ok is False
    assert not (root / "sessions" / "staged").exists()
    assert outside.is_dir()


def test_legitimate_move_still_uses_the_contained_paths(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)

    moved = _move(server_b, "9f3ac1e0b7d2", monkeypatch=monkeypatch)

    assert moved["ok"] is True, moved
    assert not source.exists()
    assert (server_b.root / "sessions" / "9f3ac1e0b7d2").is_dir()
    assert not sync.staging_dir(server_b.root, "9f3ac1e0b7d2").exists()

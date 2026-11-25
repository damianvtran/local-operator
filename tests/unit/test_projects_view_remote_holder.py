"""A linked session another device holds is shown with its holder, not as missing.

THE DEFECT (remote-parity lane, 2026-11-25). ``build_project_view`` answered
``exists`` from ``<config>/sessions/<id>`` alone, so a session minted on a peer
(placement "peer", pinned to node2) had no directory on the Mac and the
``project`` tool printed ``[missing] — no session directory`` for it — the
sentence the projects guide tells agents to unlink. These tests pin the
cache-only resolution that replaced it: existing authorized metadata
(tombstones + membership, the peer-listing cache) names the holder; nothing on
the view's path dials, reads a listing or wakes a runtime; an unresolved or
unreachable peer is said as such, never as "deleted", and leaks no metadata.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from local_operator.network import store as net_store
from local_operator.network import types as net_types
from local_operator.network.projection import write_tombstone
from local_operator.projects import ProjectEdit, ProjectRegistry, build_project_view
from local_operator.resume import SessionRow
from local_operator.session import peer_rows
from local_operator.tools.project_tool import _session_lines

LOCAL = "4e92693767fa"
REMOTE = "d5188ee174d4"
FILER = "abcdef012345"
SELF = "d_" + "a" * 32
NODE2 = "d_" + "b" * 32
STRANGER = "d_" + "c" * 32
NETWORK = "n_" + "f" * 24


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Never let the developer's real install answer: the default config dir is
    # pointed at a directory no test seeds unless it means to.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "default-home"))
    peer_rows.clear_cache()
    yield
    peer_rows.clear_cache()


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    path = tmp_path / "config"
    path.mkdir()
    return path


def _project(root: Path, *, work: list[str], filed: list[str] | None = None):
    registry = ProjectRegistry(root)
    project = registry.create_project(ProjectEdit(name="parity"), sessions=work)
    if filed:
        project = project.model_copy(update={"coordination_sessions": list(filed)})
    return project


def _seed_local(root: Path, session_id: str) -> None:
    session_dir = root / "sessions" / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "created_at.json").write_text("1700000000.0")
    (session_dir / "title.json").write_text(json.dumps({"text": "Local work", "user_set": True}))


def _seed_network(root: Path, *, members: tuple[tuple[str, str], ...] = ((NODE2, "node2"),)):
    record = net_types.NetworkRecord(
        network_id=NETWORK,
        name="home",
        self_device_id=SELF,
        self_role="admin",
        members=[
            net_types.MemberRecord(device_id=SELF, name="mac"),
            *(net_types.MemberRecord(device_id=dev, name=name) for dev, name in members),
        ],
    )
    net_store.save(record, root)


def _peer_row(session_id: str, *, device: str = NODE2, name: str = "node2", **fields) -> SessionRow:
    return SessionRow(
        session_id,
        time.time(),
        fields.pop("title", "Peer work"),
        locality="remote",
        owner_device=device,
        owner_device_name=name,
        **fields,
    )


def _seed_listing(root: Path | None, rows=(), unanswered=()) -> None:
    key = "" if root is None else str(root)
    peer_rows._CACHE[key] = (time.monotonic(), tuple(rows), tuple(unanswered), 0)  # noqa: SLF001


def _rows(view):
    return {row["session_id"]: row for row in view["sessions"]}


# -- unchanged surfaces ------------------------------------------------------


def test_a_local_row_is_byte_identical_with_a_network_present(root: Path) -> None:
    _seed_local(root, LOCAL)
    project = _project(root, work=[LOCAL])
    before = _rows(build_project_view(project, config_dir=root))[LOCAL]
    _seed_network(root)
    _seed_listing(root, rows=[_peer_row(LOCAL, title="a peer's claim on the same id")])
    after = _rows(build_project_view(project, config_dir=root))[LOCAL]
    assert after == before
    assert "holder" not in after
    assert after["exists"] is True and after["title"] == "Local work"


def test_zero_network_keeps_the_missing_row_and_its_text(root: Path) -> None:
    project = _project(root, work=[REMOTE], filed=[FILER])
    rows = _rows(build_project_view(project, config_dir=root))
    assert "holder" not in rows[REMOTE] and "holder" not in rows[FILER]
    assert rows[REMOTE]["exists"] is False
    lines = _session_lines(build_project_view(project, config_dir=root))
    assert f"  - {REMOTE} [missing] — no session directory" in lines
    assert f"  - {FILER} [filed] — no session directory" in lines


def test_coordination_rows_never_gain_liveness_keys(root: Path) -> None:
    _seed_network(root)
    _seed_listing(root, rows=[_peer_row(FILER, live_state="busy")])
    project = _project(root, work=[], filed=[FILER])
    (row,) = build_project_view(project, config_dir=root)["sessions"]
    assert row["role"] == "coordination"
    assert row["holder"]["kind"] == "remote"
    for key in ("runtime", "subagents", "todos"):
        assert key not in row
    assert row["holder"]["state"] is None and row["holder"]["busy"] is None


# -- the resolution ----------------------------------------------------------


def test_a_listed_peer_row_names_its_holder_and_state(root: Path) -> None:
    _seed_network(root)
    _seed_listing(root, rows=[_peer_row(REMOTE, live_state="busy", title="Mesh work")])
    project = _project(root, work=[REMOTE])
    row = _rows(build_project_view(project, config_dir=root))[REMOTE]
    assert row["exists"] is False  # "a directory HERE" — unchanged meaning
    assert row["holder"] == {
        "kind": "remote",
        "device_id": NODE2,
        "device_name": "node2",
        "source": "listing",
        "title": "Mesh work",
        "state": "live",
        "busy": True,
        "listing_age_s": row["holder"]["listing_age_s"],
    }
    assert row["holder"]["listing_age_s"] is not None
    # The top-level runtime stays today's default so no surface that reads it
    # (the TUI's live count) starts counting a row it renders as not-here.
    assert row["runtime"]["state"] == "stopped" and row["title"] is None
    (line,) = [line for line in _session_lines({"sessions": [row]}) if REMOTE in line]
    assert line.startswith(f"  - {REMOTE} [remote · node2 · live, busy")
    assert line.endswith("] Mesh work")


@pytest.mark.parametrize(
    ("live_state", "state", "busy"),
    [
        ("idle", "live", False),
        ("attached", "live", False),
        ("wedged", "wedged", None),
        ("", "stopped", None),
    ],
)
def test_the_peer_state_maps_to_this_views_vocabulary(
    root: Path, live_state: str, state: str, busy: bool | None
) -> None:
    _seed_network(root)
    _seed_listing(root, rows=[_peer_row(REMOTE, live_state=live_state)])
    row = _rows(build_project_view(_project(root, work=[REMOTE]), config_dir=root))[REMOTE]
    assert (row["holder"]["state"], row["holder"]["busy"]) == (state, busy)


def test_a_seeded_row_without_a_listing_has_no_claimed_age(root: Path) -> None:
    _seed_network(root)
    peer_rows.seed_peer_row(root, _peer_row(REMOTE, live_state="idle"))
    row = _rows(build_project_view(_project(root, work=[REMOTE]), config_dir=root))[REMOTE]
    assert row["holder"]["kind"] == "remote"
    assert row["holder"]["listing_age_s"] is None


def test_a_tombstone_naming_a_member_is_remote_with_unknown_state(root: Path) -> None:
    _seed_network(root)
    write_tombstone(REMOTE, device_id=NODE2, device_name="node2", config_dir=root)
    row = _rows(build_project_view(_project(root, work=[REMOTE]), config_dir=root))[REMOTE]
    assert row["holder"] == {
        "kind": "remote",
        "device_id": NODE2,
        "device_name": "node2",
        "source": "tombstone",
        "title": None,
        "state": "unknown",
        "busy": None,
        "listing_age_s": None,
    }
    assert row["runtime"]["state"] == "stopped"  # unchanged default, never a claim
    lines = _session_lines({"sessions": [row]})
    assert f"  - {REMOTE} [remote · node2 · state unknown] (untitled)" in lines


def test_a_tombstone_naming_a_non_member_leaks_nothing(root: Path) -> None:
    _seed_network(root)
    write_tombstone(REMOTE, device_id=STRANGER, device_name="old-laptop", config_dir=root)
    row = _rows(build_project_view(_project(root, work=[REMOTE]), config_dir=root))[REMOTE]
    assert row["holder"] == {
        "kind": "unresolved",
        "device_id": None,
        "device_name": None,
        "source": None,
        "title": None,
        "state": None,
        "busy": None,
        "listing_age_s": None,
    }
    text = "\n".join(_session_lines({"sessions": [row]}))
    assert STRANGER not in text and "old-laptop" not in text
    assert f"  - {REMOTE} [unresolved] — not on this device; no peer listing names it" in text
    assert "not verified deleted" in text


def test_a_listed_row_from_a_removed_device_leaks_nothing(root: Path) -> None:
    _seed_network(root)
    _seed_listing(root, rows=[_peer_row(REMOTE, device=STRANGER, name="old-laptop")])
    row = _rows(build_project_view(_project(root, work=[REMOTE]), config_dir=root))[REMOTE]
    assert row["holder"]["kind"] == "unresolved"
    assert "old-laptop" not in json.dumps(row) and STRANGER not in json.dumps(row)


def test_an_unanswered_holder_is_unreachable_without_row_metadata(root: Path) -> None:
    _seed_network(root)
    write_tombstone(REMOTE, device_id=NODE2, device_name="node2", config_dir=root)
    _seed_listing(
        root,
        unanswered=[peer_rows.UnansweredPeer(NODE2, "node2", "did not answer within 3s")],
    )
    row = _rows(build_project_view(_project(root, work=[REMOTE]), config_dir=root))[REMOTE]
    assert row["holder"]["kind"] == "unreachable"
    assert row["holder"]["device_name"] == "node2"
    assert row["holder"]["title"] is None and row["holder"]["state"] is None
    lines = _session_lines({"sessions": [row]})
    assert (
        f"  - {REMOTE} [unreachable · node2] — peer not answering; "
        "last listing could not confirm it"
    ) in lines


def test_a_cached_unreachable_row_drops_its_title(root: Path) -> None:
    _seed_network(root)
    _seed_listing(
        root, rows=[_peer_row(REMOTE, title="Secret cwd work", reachable=False, live_state="busy")]
    )
    row = _rows(build_project_view(_project(root, work=[REMOTE]), config_dir=root))[REMOTE]
    assert row["holder"]["kind"] == "unreachable"
    assert "Secret cwd work" not in json.dumps(row)


def test_nothing_found_in_a_mesh_is_unresolved_not_deleted(root: Path) -> None:
    _seed_network(root)
    row = _rows(build_project_view(_project(root, work=[REMOTE]), config_dir=root))[REMOTE]
    assert row["holder"]["kind"] == "unresolved"
    assert "[missing]" not in "\n".join(_session_lines({"sessions": [row]}))


def test_filed_rows_render_their_holder(root: Path) -> None:
    _seed_network(root)
    _seed_listing(root, rows=[_peer_row(FILER, title="Manager")])
    project = _project(root, work=[], filed=[FILER])
    lines = _session_lines(build_project_view(project, config_dir=root))
    assert f"  - {FILER} [filed · remote · node2] Manager" in lines


# -- which cache entry answers ------------------------------------------------


def test_the_default_dir_also_reads_the_rootless_cache_key(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _seed_network(root)
    _seed_listing(None, rows=[_peer_row(REMOTE, live_state="idle")])
    row = _rows(build_project_view(_project(root, work=[REMOTE]), config_dir=root))[REMOTE]
    assert row["holder"]["kind"] == "remote"


def test_an_isolated_root_never_reads_the_rootless_key(root: Path) -> None:
    _seed_network(root)
    _seed_listing(None, rows=[_peer_row(REMOTE, live_state="idle")])
    row = _rows(build_project_view(_project(root, work=[REMOTE]), config_dir=root))[REMOTE]
    assert row["holder"]["kind"] == "unresolved"


# -- the cost invariant --------------------------------------------------------


def test_the_view_never_reads_a_listing_or_dials(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("build_project_view must stay cache-only")

    for name in (
        "read_listing",
        "peer_session_rows",
        "unanswered_peers",
        "_read_all",
        "_open_catalog",
    ):
        monkeypatch.setattr(peer_rows, name, refuse)
    from local_operator.network import projection

    monkeypatch.setattr(projection, "locate_owner", refuse, raising=False)
    _seed_network(root)
    _seed_listing(root, rows=[_peer_row(REMOTE, live_state="idle")])
    row = _rows(build_project_view(_project(root, work=[REMOTE]), config_dir=root))[REMOTE]
    assert row["holder"]["kind"] == "remote"


def test_membership_is_read_once_per_view_and_not_at_all_when_every_row_is_local(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_network(root)
    calls: list[int] = []
    real = net_store.list_networks

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(net_store, "list_networks", counting)
    _seed_local(root, LOCAL)
    build_project_view(_project(root, work=[LOCAL]), config_dir=root)
    assert calls == []
    other = "0123456789ab"
    project = ProjectRegistry(root).create_project(
        ProjectEdit(name="many"), sessions=[REMOTE, other, LOCAL]
    )
    build_project_view(project, config_dir=root)
    assert len(calls) == 1

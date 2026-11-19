"""The destination AUTHORISES the import, and decides where and how it writes (#4245 round 2).

The operator's request (2026-11-18): "we should be able to move the session and its
working data, worktrees, etc. all to a remote device ... Otherwise people could use the
move function and then realize all the work is not available."

Round 2 of the security review, and the security lead's ruling on it: "require explicit
destination authorization for importing the working set, separate from session
mobility (a receiver-controlled capability ...). Bind that policy check to the
authenticated peer, destination, lease and approved manifest; absent grant should
leave the session-only move available without file import." Each cell is the
regression test for one ledger item, red on ``f51ac3bfb`` and green after:

* G   — ``workset_import``: granted on the receiver, implied by no role, checked before
        any clone/plan/fetch, re-checked on the way and again at the restore, bound to
        the transfer binding;
* S9  — no repository at/inside/above a PATH directory or under ``node_modules``, and
        an absent repository is cloned only beside the user's existing ones by default;
* S10 — no git filter driver runs on arrival content (clone and restore);
* S11 — the clone honours the host policy verdict it pins;
* M1  — a failed ``worktree repair`` after the publish does not land a second copy;
* N2  — the recall's fast-forward matches the repository, not just the branch name;
        an offer past ``MAX_OFFER_ROWS`` declines the extra rows by name.

The cross-device halves (the probe's ``import_granted``, the prepare's ``import`` flag,
the session-only move without a grant) are in ``test_workset_move_e2e``.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import (
    types,
    workset,
    workset_move,
    workset_probe,
    workset_text,
)
from local_operator.network import workset_transfer as wt
from tests.unit.network.test_workset_receive_policy import (
    EPOCH,
    GRANT,
    SID,
    _dest,
    _pending,
    _PlanOnly,
)
from tests.unit.network.workset_git_fixtures import (  # noqa: F401
    GIT_ENV,
    TEST_REMOTE,
    git,
    git_env,
    make_rig,
    serve_origin_as_https,
    transfer,
    write,
)

pytestmark = pytest.mark.usefixtures("git_env")


# ------------------------------------------------------------- G: the capability


def test_g_workset_import_is_receiver_granted_and_implied_by_no_role() -> None:
    """``move`` (and ``admin``) never imply it; the receiving device records it itself."""
    assert "workset_import" in types.CAPABILITIES
    assert "workset_import" in types.GRANTABLE_CAPABILITIES
    assert "workset_import" in types.SELF_DECIDED_SCOPES
    for role in types.ROLES:
        assert "workset_import" not in types.ROLE_CAPABILITIES[role], role
    assert "workset_import" in types.CAPABILITY_WORDS


def test_g_the_remedy_names_ids_shell_quoted() -> None:
    assert (
        workset_move.import_remedy("n1", "d_src")
        == "lop network member grant n1 d_src workset_import"
    )
    # Both come from a peer's frame: a hostile id cannot become a second command.
    assert workset_move.import_remedy("n1", "x; rm -rf ~") == (
        "lop network member grant n1 'x; rm -rf ~' workset_import"
    )


def test_g_without_an_import_authority_nothing_is_cloned_planned_or_fetched(
    tmp_path: Path,
) -> None:
    server, home = _dest(tmp_path)
    transport = _PlanOnly()
    offer = {"worktrees": [{"n": 0, "path": "~/proj", "repo": "~/proj", "tips": []}]}

    with pytest.raises(workset_move.WorksetTransferRefused) as info:
        workset_move.destination_transfer(
            server, transport, SID, lease_epoch=EPOCH, offer=offer, owner_device="d_src"
        )

    assert info.value.code == "working_set_import_not_granted"
    assert "lop network member grant" in info.value.message
    assert transport.frames == []
    assert sorted(os.listdir(home)) == []


def test_g_a_revoke_during_the_transfer_stops_it_before_the_plan(tmp_path: Path) -> None:
    server, _home = _dest(tmp_path)
    transport = _PlanOnly()
    answers = iter([True, False])
    offer = {"worktrees": [{"n": 0, "path": "~/.x", "repo": "~/.x", "tips": []}]}

    with pytest.raises(workset_move.WorksetTransferRefused) as info:
        workset_move.destination_transfer(
            server,
            transport,
            SID,
            lease_epoch=EPOCH,
            offer=offer,
            owner_device="d_src",
            authority=workset_move.IMPORT_BY_GRANT,
            recheck=lambda: next(answers, False),
        )

    assert info.value.code == "working_set_import_not_granted"
    assert transport.frames == []


def test_g_an_offer_not_bound_to_the_approved_manifest_is_refused(tmp_path: Path) -> None:
    """The binding is recomputed HERE from the confirmed scope, this device, this lease
    and the offered snapshot ids; an offer whose binding does not match is not taken."""
    server, _home = _dest(tmp_path)
    rows = [{"n": 0, "path": "~/.x", "repo": "~/.x", "head": "a", "index": "b", "worktree": "c"}]
    good = workset.transfer_binding(
        "sha256:scope", lease_epoch=EPOCH, to_device="d_me", snapshots=rows
    )
    bound = {
        "approved_scope": "sha256:scope",
        "self_device": "d_me",
        **GRANT,
    }

    transport = _PlanOnly()
    with pytest.raises(workset_move.WorksetTransferRefused):
        workset_move.destination_transfer(
            server,
            transport,
            SID,
            lease_epoch=EPOCH,
            offer={"worktrees": rows, "binding": good.replace("sha256:", "sha256:0")},
            owner_device="d_src",
            **bound,
        )
    assert transport.frames == []

    workset_move.destination_transfer(
        server,
        transport,
        SID,
        lease_epoch=EPOCH,
        offer={"worktrees": rows, "binding": good},
        owner_device="d_src",
        **bound,
    )
    assert _pending(server)["import"] == {
        "authority": "grant",
        "from_device": "d_src",
        "to_device": "d_me",
        "network_id": "",
        "lease_epoch": EPOCH,
        "binding": good,
    }


# S13/S16 (#4245 security round 3): the destination checks the offer against what
# ITS operator confirmed, and a named destination never skips the binding.
#
# The block the operator read. Its digest is the REAL one for this device
# (``d_me``): since round 4 (M1) ``record_confirmable`` recomputes it from the rows
# and refuses a block whose label does not match, so a made-up ``sha256:111…`` no
# longer records anything.
_SHOWN_ROWS: list[dict[str, Any]] = [
    {"path": "~/src/app", "repo": "~/src/app", "state": "planned"},
    {"path": "~/src/kept", "repo": "~/src/kept", "state": "left_behind", "reason": "contended"},
]
_CONFIRMED = workset.block_scope_digest({"worktrees": _SHOWN_ROWS}, to_device="d_me")


def _shown_block(**extra: Any) -> dict[str, Any]:
    return {"scope_digest": _CONFIRMED, "worktrees": [dict(r) for r in _SHOWN_ROWS], **extra}


def _snapshot_rows(path: str) -> list[dict[str, Any]]:
    return [
        {"n": 0, "path": path, "repo": path, "tips": [], "head": "h", "index": "i", "worktree": "w"}
    ]


def _bound_offer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    snapshots = [{k: r.get(k) for k in ("path", "head", "index", "worktree")} for r in rows]
    binding = workset.transfer_binding(
        _CONFIRMED, lease_epoch=EPOCH, to_device="d_me", snapshots=snapshots
    )
    return {"worktrees": rows, "binding": binding}


def _recall(server: Any, offer: dict[str, Any], transport: Any) -> None:
    workset_move.destination_transfer(
        server,
        transport,
        SID,
        lease_epoch=EPOCH,
        offer=offer,
        owner_device="d_src",
        authority=workset_move.IMPORT_BY_RECALL,
        approved_scope=_CONFIRMED,
        self_device="d_me",
        confirmed=workset_move.confirmed_rows(server.root, SID, _CONFIRMED),
    )


def test_s13_a_recall_offer_outside_the_confirmed_rows_is_refused(tmp_path: Path) -> None:
    """The probe_g2 shape: the operator confirmed a recall listing only ``~/src/app``;
    the owner offers ``~/src/other`` with a binding it computed itself (it knows the
    digest and the lease). Refused before any plan, and nothing is written."""
    server, home = _dest(tmp_path)
    assert workset_move.record_confirmable(server.root, SID, _shown_block(), self_device="d_me")
    for path in ("~/src/other", "~/src/kept"):
        transport = _PlanOnly()
        with pytest.raises(workset_move.WorksetTransferRefused) as info:
            _recall(server, _bound_offer(_snapshot_rows(path)), transport)
        assert info.value.code == "working_set_transfer_failed", path
        assert transport.frames == [], path
    assert not (home / "src" / "other").exists()

    # Control: the confirmed row itself is taken (planned against this device).
    transport = _PlanOnly()
    _recall(server, _bound_offer(_snapshot_rows("~/src/app")), transport)
    assert [f.get("phase") for f in transport.frames] == ["workset_plan"]


def test_s13_a_recall_with_no_preflight_record_carries_no_files(tmp_path: Path) -> None:
    server, _home = _dest(tmp_path)
    transport = _PlanOnly()

    with pytest.raises(workset_move.WorksetTransferRefused):
        _recall(server, _bound_offer(_snapshot_rows("~/src/app")), transport)
    assert transport.frames == []


class _OwnerAnswers:
    """A stand-in link to a recall's OWNER: each preflight gets the next block."""

    def __init__(self, root: Path, *blocks: dict[str, Any]) -> None:
        class _Identity:
            device_id = "d_me"

        class _Relay:
            identity = _Identity()

        _Relay.root = root  # type: ignore[attr-defined]
        self.server = _Relay()
        self.blocks = list(blocks)

    def ask(self, frame: dict[str, Any], **_: Any) -> dict[str, Any]:
        assert frame["phase"] == "workset_preflight", frame
        return {"result": "workset", "working_set": self.blocks.pop(0)}


def test_s13_the_recall_preflight_records_the_rows_the_operator_reads(tmp_path: Path) -> None:
    from local_operator.network import mobility

    transport: Any = _OwnerAnswers(tmp_path, _shown_block())
    mobility._ask_workset_preflight(transport, SID, "d_src", "owner")  # noqa: SLF001

    assert workset_move.confirmed_rows(tmp_path, SID, _CONFIRMED) == {("~/src/app", "~/src/app")}
    assert workset_move.confirmed_rows(tmp_path, SID, "sha256:" + "2" * 64) is None


def _widened() -> dict[str, Any]:
    """The hostile owner's answer: the CONFIRMED label over one more planned row."""
    block = _shown_block()
    block["worktrees"].append({"path": "~/src/other", "repo": "~/src/other", "state": "planned"})
    return block


def test_s13_the_confirmed_runs_hidden_preflight_cannot_widen_the_record(tmp_path: Path) -> None:
    """probe_s13_r4 (#4245 security round 4, S13/M1), ported. The operator dry-ran a
    recall listing only ``~/src/app`` and confirmed its digest D; on the confirmed run
    ``_recall``'s compatibility preflight (nobody reads its answer) got D over
    ``~/src/app`` AND ``~/src/other``. Before: that answer replaced the record, and an
    owner-bound offer for ``~/src/other`` reached ``workset_plan``. Now: the hidden
    preflight records nothing, the record still says ``~/src/app``, and the offer for
    ``~/src/other`` is refused before any frame is sent."""
    from local_operator.network import mobility

    server, home = _dest(tmp_path)
    transport: Any = _OwnerAnswers(server.root, _shown_block(), _widened())
    mobility._ask_workset_preflight(transport, SID, "d_src", "owner")  # noqa: SLF001
    # ``_recall``'s own call shape for the probe before ``_destination_move``.
    mobility._ask_workset_preflight(transport, SID, "d_src", "owner", record=False)  # noqa: SLF001

    assert workset_move.confirmed_rows(server.root, SID, _CONFIRMED) == {("~/src/app", "~/src/app")}
    plan = _PlanOnly()
    with pytest.raises(workset_move.WorksetTransferRefused) as info:
        _recall(server, _bound_offer(_snapshot_rows("~/src/other")), plan)
    assert info.value.code == "working_set_transfer_failed"
    assert plan.frames == []
    assert not (home / "src" / "other").exists()


def test_s13_a_block_whose_rows_do_not_hash_to_its_digest_is_not_recorded(
    tmp_path: Path,
) -> None:
    """Even on a path the operator DOES read, the owner's label is checked: D over
    widened rows, or D minted for another device, records nothing (fail closed)."""
    from local_operator.network import mobility

    assert not workset_move.record_confirmable(tmp_path, SID, _widened(), self_device="d_me")
    assert not workset_move.record_confirmable(tmp_path, SID, _shown_block(), self_device="d_x")
    assert not workset_move.record_confirmable(tmp_path, SID, _shown_block(), self_device="")
    transport: Any = _OwnerAnswers(tmp_path, _widened())
    block = mobility._ask_workset_preflight(transport, SID, "d_src", "owner")  # noqa: SLF001
    assert block["scope_digest"] == _CONFIRMED  # still shown, never recorded
    assert workset_move.confirmed_rows(tmp_path, SID, _CONFIRMED) is None


def test_s13_a_recorded_digest_is_write_once(tmp_path: Path) -> None:
    """First write wins: an equal re-record is a no-op; a different row set under a
    recorded digest (a tampered file stands in for a collision) keeps the first."""
    assert workset_move.record_confirmable(tmp_path, SID, _shown_block(), self_device="d_me")
    assert workset_move.record_confirmable(tmp_path, SID, _shown_block(), self_device="d_me")
    path = workset_move._confirmable_path(tmp_path, SID)  # noqa: SLF001
    stored = json.loads(path.read_text())
    stored["scopes"][_CONFIRMED] = [["~/src/first", "~/src/first"]]
    path.write_text(json.dumps(stored))

    assert not workset_move.record_confirmable(tmp_path, SID, _shown_block(), self_device="d_me")
    assert workset_move.confirmed_rows(tmp_path, SID, _CONFIRMED) == {
        ("~/src/first", "~/src/first")
    }


def test_s13_the_block_recompute_matches_the_digest_the_owner_mints() -> None:
    """The drift guard: ``block_scope_digest`` over ``Manifest.to_dict()`` equals the
    ``scope_digest`` that dict carries, for every digest input (branch, reason,
    withheld paths, ``carries_files``) — or an honest recall would record nothing."""
    facts = workset.WorktreeFacts(
        path="~/src/app",
        repo="~/src/app",
        abs_path="/h/src/app",
        abs_repo="/h/src/app",
        branch="feat",
        excluded=(workset.Excluded(path="~/src/app/.env", why="secret_path", rule="dotenv"),),
    )
    kept = workset.WorktreeFacts(
        path="~/src/kept",
        repo="~/src/kept",
        abs_path="/h/src/kept",
        abs_repo="/h/src/kept",
        state="left_behind",
        reason="contended",
    )
    for carries in (False, True):
        block = workset.Manifest(
            to_device="d_me", to_name="me", worktrees=(facts, kept), carries_files=carries
        ).to_dict()
        assert workset.block_scope_digest(block, to_device="d_me") == block["scope_digest"]
        assert workset.block_scope_digest(block, to_device="d_x") != block["scope_digest"]


def test_s16_a_granted_owner_cannot_skip_the_binding(tmp_path: Path) -> None:
    """probe_g2 G3: ``authority=grant`` with no scope digest and no binding used to
    skip the binding check entirely; with this device named it now refuses."""
    server, home = _dest(tmp_path)
    for offer, scope in (
        ({"worktrees": _snapshot_rows("~/src/third")}, ""),
        ({"worktrees": _snapshot_rows("~/src/third")}, _CONFIRMED),
    ):
        transport = _PlanOnly()
        with pytest.raises(workset_move.WorksetTransferRefused) as info:
            workset_move.destination_transfer(
                server,
                transport,
                SID,
                lease_epoch=EPOCH,
                offer=offer,
                owner_device="d_src",
                approved_scope=scope,
                self_device="d_me",
                **GRANT,
            )
        assert info.value.code == "working_set_transfer_failed"
        assert transport.frames == []
    assert not (home / "src").exists()


def _arrived_pending(
    server: Any, authority: str, paths: tuple[str, ...] = ("~/proj/.worktrees/feat",)
) -> None:
    plans = [_plan(n, path) for n, path in enumerate(paths)]
    directory = workset_move.arrived_dir(server.root, SID)
    directory.mkdir(parents=True)
    (directory / "pending.json").write_text(
        json.dumps(
            {
                "session_id": SID,
                "from_device": "d_src",
                "lease_epoch": EPOCH,
                "plans": [asdict(plan) for plan in plans],
                "left_behind": [],
                "restored": {},
                "import": {
                    "authority": authority,
                    "from_device": "d_src",
                    "to_device": "d_me",
                    "network_id": "",
                    "lease_epoch": EPOCH,
                    "binding": "",
                },
            }
        )
    )


def _plan(n: int, path: str) -> wt.RestorePlan:
    return wt.RestorePlan(
        session_id=SID,
        n=n,
        bundle_digest="sha256:" + str(n) * 64,
        path=path,
        primary="~/proj",
        remote_url="",
        head="a" * 40,
        index="b" * 40,
        worktree="c" * 40,
        branch="feat",
        upstream_merge=None,
        src_device="d_src",
        is_primary=False,
        fresh_clone=False,
    )


def test_g_the_restore_rechecks_the_grant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A grant revoked between the fetch and the restore stops the writes; the bundle
    stays staged (``pending_restore``), so a re-grant restores it on the next resume.
    A recall stands on its own authority."""
    server, _home = _dest(tmp_path)
    called: list[str] = []

    def fake_restore(_dir: Path, plan: wt.RestorePlan, **_kw: Any) -> wt.RestoreReport:
        called.append(plan.path)
        return wt.RestoreReport("pending_restore", plan.path, reason="restore_failed")

    monkeypatch.setattr(wt, "restore", fake_restore)
    _arrived_pending(server, workset_move.IMPORT_BY_GRANT)  # no store row grants d_src

    out = workset_move.restore_arrived(server, SID)

    assert out is not None
    assert [r["reason"] for r in out["worktrees"]] == ["import_not_granted"]
    assert called == []
    assert "allowed" in workset_text.reason_text("import_not_granted")

    monkeypatch.setattr(workset_move, "import_granted", lambda *_a, **_k: True)
    workset_move.restore_arrived(server, SID)
    assert called == ["~/proj/.worktrees/feat"]


@pytest.mark.parametrize("revoke_after_first", [True, False])
def test_g_a_revoke_mid_restore_stops_the_rows_not_yet_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, revoke_after_first: bool
) -> None:
    """The grant is re-read before EACH row, not once per restore: a revoke landing
    after row 1 restored keeps row 1 and leaves row 2 staged (``import_not_granted``),
    never published; a re-grant and the next resume finish it. Control: no revoke
    restores both."""
    server, _home = _dest(tmp_path)
    granted = [True]
    called: list[str] = []
    monkeypatch.setattr(workset_move, "import_granted", lambda *_a, **_k: granted[0])

    def fake_restore(_dir: Path, plan: wt.RestorePlan, **_kw: Any) -> wt.RestoreReport:
        called.append(plan.path)
        if revoke_after_first:
            granted[0] = False  # the operator revokes `workset_import` mid-restore
        return wt.RestoreReport("restored", plan.path)

    monkeypatch.setattr(wt, "restore", fake_restore)
    first, second = "~/proj/.worktrees/one", "~/proj/.worktrees/two"
    _arrived_pending(server, workset_move.IMPORT_BY_GRANT, paths=(first, second))

    out = workset_move.restore_arrived(server, SID)

    assert out is not None
    rows = [(r["state"], r["reason"]) for r in out["worktrees"]]
    if not revoke_after_first:
        assert called == [first, second]
        assert rows == [("restored", ""), ("restored", "")]
        assert out["state"] == "restored"
        return
    assert called == [first]
    assert rows == [("restored", ""), ("pending_restore", "import_not_granted")]
    assert out["state"] == "pending_restore"
    # Row 2's bundle stays staged for the re-grant (only a fully restored set drops it).
    assert (workset_move.arrived_dir(server.root, SID) / "pending.json").is_file()

    granted[0] = True
    revoke_after_first = False  # the re-grant: fake_restore no longer revokes
    called.clear()
    again = workset_move.restore_arrived(server, SID)
    assert called == [second]  # row 1 is not redone
    assert again is not None and again["state"] == "restored"


def test_g_a_pending_journal_without_its_import_record_restores_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _home = _dest(tmp_path)
    monkeypatch.setattr(wt, "restore", lambda *_a, **_k: pytest.fail("restore ran"))
    _arrived_pending(server, workset_move.IMPORT_BY_RECALL)
    path = workset_move.arrived_dir(server.root, SID) / "pending.json"
    record = json.loads(path.read_text())
    record["import"]["lease_epoch"] = "another-move"
    path.write_text(json.dumps(record))

    out = workset_move.restore_arrived(server, SID)

    assert out is not None and out["worktrees"][0]["reason"] == "import_not_granted"


def test_g_the_summary_states_the_missing_grant_and_its_command_first() -> None:
    block = {
        "worktrees": [],
        "destination_probe": {"state": "answered", "import_granted": False},
        "import": {
            "granted": False,
            "remedy": "lop network member grant n1 d_src workset_import",
        },
    }
    lines = workset_text.probe_block_lines(block, device="node2")
    assert "lop network member grant n1 d_src workset_import" in lines[0]
    assert "node2" in lines[0]


def test_g_apply_probe_keeps_only_a_boolean_import_answer() -> None:
    block: dict[str, Any] = {"worktrees": [{"remote_url": "https://x/y.git"}]}
    workset_probe.apply_probe(block, {"worktrees": [], "import_granted": True})
    assert block["destination_probe"]["import_granted"] is True
    block = {"worktrees": [{"remote_url": "https://x/y.git"}]}
    workset_probe.apply_probe(block, {"worktrees": [], "import_granted": "yes"})
    assert "import_granted" not in block["destination_probe"]


# ----------------------------------------------------------- S9: PATH and clones


@pytest.fixture
def evil_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """repro_s1_bin's attacker: a public repository whose only file is ``ls``."""
    src = tmp_path / "attacker"
    marker = tmp_path / "PWNED"
    write(src / "ls", f'#!/bin/sh\ntouch {marker}\nexec /bin/ls "$@"\n', mode=0o755)
    git(src.parent, "init", "-q", "-b", "main", str(src))
    git(src, "add", "ls")
    git(src, "commit", "-qm", "x")
    evil = tmp_path / "evil.git"
    git(tmp_path, "clone", "-q", "--bare", str(src), str(evil))
    monkeypatch.setattr(workset_probe, "_ALLOWED_HOSTS", ("git.test",))
    monkeypatch.setattr(
        wt,
        "_CLONE_TEST_CONFIG",
        ("protocol.file.allow=always", f"url.{evil}.insteadOf={TEST_REMOTE}"),
    )
    return marker


def test_s9_a_source_cannot_plant_a_repository_in_bin(
    tmp_path: Path, evil_tools: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """repro_s1_bin: before, ``~/bin`` was cloned and the victim's next ``ls`` ran the
    attacker's. Now ``~/bin`` (a stock login-shell PATH entry) is refused before any
    clone, even though ``~`` holds the user's repositories."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # lop's own PATH can lack ~/bin
    server, home = _dest(tmp_path)
    git(home, "init", "-q", str(home / "mine"))
    transport = _PlanOnly()
    offer = {"worktrees": [{"n": 0, "path": "~/bin", "repo": "~/bin", "remote_url": TEST_REMOTE}]}

    workset_move.destination_transfer(
        server, transport, SID, lease_epoch=EPOCH, offer=offer, owner_device="d_src", **GRANT
    )

    assert not (home / "bin").exists()
    assert transport.frames[0]["decline"] == {"0": "restore_path_refused"}
    rc = subprocess.run(
        ["bash", "-c", 'PATH="$HOME/.local/bin:$HOME/bin:$PATH"; ls >/dev/null'],
        env={**os.environ, "HOME": str(home), "PATH": "/usr/bin:/bin"},
        check=False,
    ).returncode
    assert rc == 0 and not evil_tools.exists()


@pytest.mark.parametrize(
    ("rel", "refused"),
    [
        ("~/tools", True),  # contains a PATH directory
        ("~/tools/bin", True),  # is one
        ("~/tools/bin/x", True),  # inside one
        ("~/go", True),  # contains ~/go/bin, a common rc entry even when PATH lacks it
        ("~/.local/bin", True),
        ("~/proj/node_modules/x", True),
        ("~/node_modules", True),
        ("~/proj", False),
        ("~/src/tools", False),
    ],
)
def test_s9_command_and_dependency_directories_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rel: str, refused: bool
) -> None:
    monkeypatch.setenv("PATH", f"{tmp_path / 'tools' / 'bin'}:/usr/bin:{tmp_path}")
    got = wt.primary_path_refusal(tmp_path, rel, roots=("~",))
    assert bool(got) is refused, got


def test_s9_an_absent_repository_is_cloned_only_beside_existing_ones(tmp_path: Path) -> None:
    home = tmp_path
    assert wt.clone_parent_refusal(home, "~/new/proj") == "the parent directory does not exist"
    assert wt.clone_parent_refusal(home, "~/proj") == "the parent holds no repository"
    (home / "notes").mkdir()
    assert wt.clone_parent_refusal(home, "~/notes/proj") == "the parent holds no repository"
    git(home, "init", "-q", str(home / "src" / "mine"))
    assert wt.clone_parent_refusal(home, "~/src/proj") == ""
    assert wt.clone_parent_refusal(home, "~/proj") == "the parent holds no repository"
    git(home, "init", "-q", str(home / "other"))
    assert wt.clone_parent_refusal(home, "~/proj") == ""


def test_s9_the_default_policy_declines_the_clone_and_explicit_roots_widen_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, home = _dest(tmp_path)
    monkeypatch.setattr(workset_probe, "_ALLOWED_HOSTS", ("git.test",))
    tried: list[dict[str, Any]] = []

    def fake_clone(url: str, rel: str, **kw: Any) -> wt.PrimaryResolution:
        tried.append({"rel": rel, "make_parents": kw.get("make_parents")})
        raise wt.MoveWorksetError("clone_failed", "test")

    monkeypatch.setattr(wt, "clone_primary", fake_clone)
    offer = {
        "worktrees": [
            {"n": 0, "path": "~/code/x", "repo": "~/code/x", "remote_url": "https://git.test/x.git"}
        ]
    }
    transport = _PlanOnly()

    workset_move.destination_transfer(
        server, transport, SID, lease_epoch=EPOCH, offer=offer, owner_device="d_src", **GRANT
    )

    assert tried == []
    assert transport.frames[0]["decline"] == {"0": "destination_declined"}  # S8: coarse
    assert _pending(server)["left_behind"][0]["reason"] == "clone_parent_refused"
    assert not (home / "code").exists()

    (server.root / "config.yml").write_text(
        "values:\n  mesh:\n    move:\n      working_set_roots: ['~/code']\n"
    )
    workset_move.discard_incoming(server.root, SID)
    workset_move.destination_transfer(
        server, _PlanOnly(), SID, lease_epoch=EPOCH, offer=offer, owner_device="d_src", **GRANT
    )
    assert tried == [{"rel": "~/code/x", "make_parents": True}]


# ------------------------------------------------------------ S10: no filters


def _standin_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, *, repo: Path | None = None
) -> Path:
    """What ``git lfs install`` writes, with a stand-in that logs instead of fetching.
    ``kind`` is ``smudge`` or ``process``. Into the user's global config, or into
    ``repo``'s own config (a destination primary whose filters are its own): in the
    single-process restore rig the source's snapshot shares the environment, and a
    ``process`` stand-in there would break the SOURCE's clean step, not the arrival."""
    log = tmp_path / "DRIVER.log"
    fake = tmp_path / "lfs-standin"
    fake.write_text(f'#!/bin/sh\necho "$* cwd=$PWD" >> {log}\ncat\n')
    fake.chmod(0o755)
    settings = {
        "filter.lfs.clean": "cat",
        f"filter.lfs.{kind}": f"{fake} {kind} -- %f",
        "filter.lfs.required": "true",
    }
    if repo is not None:
        for key, value in settings.items():
            git(repo, "config", key, value)
        return log
    gcfg = tmp_path / "dst-gitconfig"
    gcfg.write_text(
        "".join(f"[filter \"lfs\"]\n\t{k.rpartition('.')[2]} = {v}\n" for k, v in settings.items())
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gcfg))
    return log


def _attacker_lfs(tree: Path) -> None:
    write(tree / ".gitattributes", "* filter=lfs\n")
    write(tree / ".lfsconfig", "[lfs]\n\turl = http://169.254.169.254/latest/\n")
    pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 1\n"
    write(tree / "x.bin", pointer)


@pytest.mark.parametrize("kind", ["smudge", "process"])
def test_s10_the_restore_runs_no_filter_driver_on_the_sources_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """probe_filter: a dirty worktree carrying ``* filter=lfs`` + an ``.lfsconfig``. Before,
    the destination's driver ran on every path with the peer's ``.lfsconfig`` in its cwd."""
    rig = make_rig(tmp_path)
    _attacker_lfs(rig.worktree)
    log = _standin_driver(tmp_path, monkeypatch, kind, repo=rig.dst_primary)

    _snap, _plan, report = transfer(
        src_worktree=rig.worktree,
        src_primary=rig.src_primary,
        src_home=rig.src_home,
        dst_home=rig.dst_home,
        work=tmp_path / "w",
        session_id=SID,
    )

    assert report.state == "restored", report
    assert not log.exists(), log.read_text()
    landed = rig.dst_home / "proj" / ".worktrees" / "feat"
    assert (landed / "x.bin").read_text().startswith("version https://git-lfs")
    recipes = [r.kind for r in wt.recipes_for(landed, rig.dst_primary, home=rig.dst_home)]
    assert "git_lfs" in recipes


@pytest.mark.parametrize("kind", ["smudge", "process"])
def test_s10_the_w1_clone_runs_no_filter_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """probe_clone A3: the clone's checkout called the driver on every path."""
    src = tmp_path / "evil-src"
    _attacker_lfs(src)
    write(src / "README", "hi\n")
    git(tmp_path, "init", "-q", "-b", "main", str(src))
    git(src, "add", "-A")
    git(src, "commit", "-qm", "x")
    origin = tmp_path / "evil.git"
    git(tmp_path, "clone", "-q", "--bare", str(src), str(origin))
    monkeypatch.setattr(workset_probe, "_ALLOWED_HOSTS", ("git.test",))
    monkeypatch.setattr(
        wt,
        "_CLONE_TEST_CONFIG",
        ("protocol.file.allow=always", f"url.{origin}.insteadOf={TEST_REMOTE}"),
    )
    log = _standin_driver(tmp_path, monkeypatch, kind)
    home = tmp_path / "home"
    home.mkdir()

    res = wt.clone_primary(TEST_REMOTE, "~/proj", home=home, pack_bytes=0)

    assert res.state == "present" and res.cloned
    assert not log.exists(), log.read_text()
    assert (home / "proj" / "README").read_text() == "hi\n"  # still checked out
    assert git(home / "proj", "status", "--porcelain") == ""


# ------------------------------------------------------------- S11: the pin


def test_s11_the_clone_honours_the_verdict_it_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """probe_pin: the first lookup public, the second (the pin's) a rebinding answer.
    Before, git ran with no pin and resolved a third time on its own."""
    calls: list[str] = []

    def rebinding(url: str) -> tuple[bool, list[str]]:
        calls.append(url)
        if len(calls) == 1:
            return True, ["-c", "http.curloptResolve=evil.example:443:93.184.216.34"]
        return False, []

    monkeypatch.setattr(workset_probe, "host_policy", rebinding)
    monkeypatch.setattr(workset_probe, "classify_url", lambda _u: "https")
    monkeypatch.setattr(wt, "_run_clone", lambda *_a, **_k: pytest.fail("git ran"))

    with pytest.raises(wt.MoveWorksetError) as info:
        wt.clone_primary("https://evil.example/x.git", "~/proj", home=tmp_path, pack_bytes=0)

    assert info.value.reason == "url_refused"


# ---------------------------------------------------- M1: repair after publish


def test_m1_a_failed_repair_after_the_publish_does_not_land_a_second_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review M1: the first ``worktree repair`` fails after the rename. Before, the
    restore reported failure, and the retry landed ``feat.moved-*`` beside it."""
    rig = make_rig(tmp_path)
    real_git = wt._git
    failed: list[tuple[str, ...]] = []

    def flaky(cwd: Path | str, *args: str, **kw: Any) -> subprocess.CompletedProcess[str]:
        if args[:2] == ("worktree", "repair") and not failed:
            failed.append(args)
            if kw.get("check", True):
                raise wt.MoveWorksetError("git_failed", "repair: injected")
            return subprocess.CompletedProcess(["git"], 1, "", "injected")
        return real_git(cwd, *args, **kw)

    monkeypatch.setattr(wt, "_git", flaky)
    kwargs: dict[str, Any] = {
        "src_worktree": rig.worktree,
        "src_primary": rig.src_primary,
        "src_home": rig.src_home,
        "dst_home": rig.dst_home,
        "work": tmp_path / "w",
        "session_id": SID,
    }

    _snap, plan, first = transfer(**kwargs)
    second = wt.restore(tmp_path / "w" / "in" / f"{SID}-0", plan, home=rig.dst_home, config=None)

    assert failed, "the injected repair failure never happened"
    assert first.state == "restored", first
    assert second.dest_path == first.dest_path, (first, second)
    names = sorted(p.name for p in (rig.dst_home / "proj" / ".worktrees").iterdir())
    assert [n for n in names if n.startswith("feat")] == ["feat"]
    assert "prunable" not in git(rig.dst_primary, "worktree", "list", "--porcelain")


# ------------------------------------------------- N2: journal + offer bounds


def test_n2_the_recall_fast_forward_matches_the_repository_not_just_the_branch(
    tmp_path: Path,
) -> None:
    repo_a = tmp_path / "a"
    repo_b = tmp_path / "b"
    for repo in (repo_a, repo_b):
        git(tmp_path, "init", "-q", "-b", "main", str(repo))
    config = tmp_path / "cfg"
    journal = config / "network" / "workset-departed" / f"{SID}.json"
    journal.parent.mkdir(parents=True)
    journal.write_text(
        json.dumps(
            {
                "session_id": SID,
                "worktrees": [
                    {"branch": "main", "head": "f" * 40, "common_dir": wt._common_dir(repo_a)}
                ],
            }
        )
    )

    assert wt._departed_branch_tip(config, SID, "main", repo_a) == "f" * 40
    assert wt._departed_branch_tip(config, SID, "main", repo_b) is None
    assert wt._departed_branch_tip(config, SID, "main") is None


def test_n2_an_offer_past_the_row_bound_declines_the_rest_by_name(tmp_path: Path) -> None:
    server, _home = _dest(tmp_path)
    transport = _PlanOnly()
    rows = [
        {"n": n, "path": f"~/.x{n}", "repo": f"~/.x{n}", "tips": []}
        for n in range(workset_move.MAX_OFFER_ROWS + 1)
    ]

    workset_move.destination_transfer(
        server,
        transport,
        SID,
        lease_epoch=EPOCH,
        offer={"worktrees": rows},
        owner_device="d",
        **GRANT,
    )

    extra = str(workset_move.MAX_OFFER_ROWS)
    assert transport.frames[0]["decline"][extra] == "too_many_worktrees"
    reasons = {str(r["n"]): r["reason"] for r in _pending(server)["left_behind"]}
    assert reasons[extra] == "too_many_worktrees"


def test_n1_the_path_refusal_names_every_kind_of_folder_it_covers() -> None:
    text = workset_text.reason_text("restore_path_refused")
    assert "hidden" in text and "mesh.move.working_set_roots" in text


# --------------------------------------------- QA round 1 (Q1, Q4, Q5, Q6, Q7)


def test_q1_a_beside_restore_names_where_the_files_are() -> None:
    """QA Q1: the arrival note and the receipt named the source path for a tree that
    was restored BESIDE an occupied one, so an agent would edit the wrong tree."""
    block = {"worktrees": [{"path": "~/proj/.worktrees/occ", "branch": "occ", "state": "planned"}]}
    dest = "~/proj/.worktrees/occ.moved-d_a-202611182118"
    restored = {
        "state": "restored",
        "worktrees": [
            {
                "state": "restored",
                "path": "~/proj/.worktrees/occ",
                "dest_path": dest,
                "mode": "beside",
            }
        ],
    }

    receipt = workset_move.arrival_receipt(block, restored)

    assert receipt is not None
    (row,) = receipt["worktrees"]
    assert row["remapped"] == ["~/proj/.worktrees/occ", dest]
    note = workset_probe.arrival_note(receipt, session_id=SID, source="laptop", destination="node2")
    assert f"restored on this device at {dest}" in note
    assert "~/proj/.worktrees/occ (branch occ): restored on this device." not in note


def test_q4_a_secret_only_in_the_index_is_listed_withheld(tmp_path: Path) -> None:
    """QA Q4: a key staged and then removed from the working file rode the index commit;
    the outgoing scan then refused the whole worktree as a "history" secret. The
    manifest now lists the path withheld, and the snapshot reverts that index entry."""
    repo = tmp_path / "repo"
    git(tmp_path, "init", "-q", "-b", "main", str(repo))
    write(repo / "config.txt", "plain\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")
    write(repo / "config.txt", "key = " + "AKIA" + "Q3VZ7XK2M4N8P6RT" + "\n")
    git(repo, "add", "config.txt")
    write(repo / "config.txt", "plain again\n")  # the working file no longer has it

    found = workset.classify_exclusions(repo, ["config.txt"], home=tmp_path)

    assert [(e.why, e.path.endswith("config.txt")) for e in found] == [("secret_content", True)]


def test_q5_the_unconfirmed_sentence_counts_the_planned_worktrees() -> None:
    from local_operator.network import mobility

    block = {
        "totals": {"worktrees": 2, "unpushed_commits": 1, "dirty_files": 1},
        "worktrees": [{"state": "planned"}, {"state": "planned"}, {"state": "left_behind"}],
    }
    text = mobility._workset_refusal(SID, "working_set_unconfirmed", block, to_label="b")["message"]
    assert "in 2 git worktrees" in text
    assert "1 unpushed commit," in text and "1 uncommitted file)" in text


def test_q6_a_policy_refused_host_is_not_a_missing_credential() -> None:
    block: dict[str, Any] = {"worktrees": [{"remote_url": "https://127.0.0.1/x.git"}]}
    answer = {
        "worktrees": [
            {
                "remote_url": "https://127.0.0.1/x.git",
                "clone_check": "destination_cannot_fetch",
                "detail": "host_refused",
            }
        ]
    }
    workset_probe.apply_probe(block, answer, device_id="d_b")
    (row,) = block["worktrees"]
    assert "remedy" not in row
    (line,) = workset_text.probe_row_lines(row, device="devb")
    assert "policy" in line and "own access" not in line


def test_q7_one_preflight_asks_the_destination_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """QA Q7: the carry recompute re-asked the destination, so a refused (rate-limited)
    probe was asked, and audited, twice."""
    from local_operator.network import mobility

    asks: list[dict[str, Any]] = []

    class _Transport:
        def __init__(self, *_a: Any) -> None:
            pass

        def ask(self, frame: dict[str, Any], **_k: Any) -> dict[str, Any]:
            asks.append(frame)
            return {"result": "refused", "code": "rate_limited"}

    class _Identity:
        device_id = "d_me"

    class _Srv:
        identity = _Identity()
        root = Path("/nonexistent")

    monkeypatch.setattr(mobility, "LinkTransport", _Transport)
    monkeypatch.setattr(mobility, "_link_for_move", lambda *_a: object())
    block = {"worktrees": [{"remote_url": "https://git.test/x.git", "state": "planned"}]}
    asked: dict[str, Any] = {}
    for _ in range(2):
        out = mobility._probe_destination(
            _Srv(),  # type: ignore[arg-type]
            dict(block, worktrees=[dict(r) for r in block["worktrees"]]),
            SID,
            target_device="d_b",
            target_name="b",
            here=False,
            asked=asked,
        )
        assert out["destination_probe"]["state"] == "unknown_destination"
    assert len(asks) == 1

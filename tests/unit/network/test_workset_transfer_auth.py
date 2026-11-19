"""``workset_plan`` / ``workset_fetch`` authorisation, audit-before-bytes and resume.

Binding rules (mesh-lane owner and security lead, 2026-11-18; design §8.2, §8.3):

* every fetch is authorised against the CURRENT move's journaled artifact set — the
  in-flight ``prepared`` source entry, the AUTHENTICATED link peer equal to its
  ``to_device``, the frame's lease epoch equal to the entry's, the journal's binding
  equal to the entry's — never a directory listing (the staged directory also holds
  ``<hex>.json`` sidecars and the journal);
* a durable ``session.workset.serve`` row is on disk BEFORE the first byte of each
  artifact AND before any resumed (non-contiguous) serve; an audit that cannot be
  proven written refuses with ZERO bytes;
* the destination resumes a ``.part`` only after the owner confirms its prefix digest;
  a corrupted prefix is discarded and the pull restarts at 0.

The owner half runs for real (real git snapshot + bundle, real ``AuditLog`` file);
only the relay link is a stub, because the cells pin these functions' own checks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import audit as audit_mod
from local_operator.network import sync as sync_mod
from local_operator.network import workset, workset_move
from local_operator.network.audit import AuditLog
from local_operator.session.placement import write_handoff_entry
from tests.unit.network.workset_rig import GIT_ENV, Rig, make_rig

SID = "5e55a0c0ffee"
EPOCH = "epoch-1"
DEST = "d_destination"


class _Identity:
    device_id = "d_owner"


class _Server:
    def __init__(self, root: Path, home: Path) -> None:
        self.root = root
        self.workset_home = home
        self.identity = _Identity()
        self.audit = AuditLog(root)

    def _member_name(self, device_id: str) -> str:
        return "dest"


class _Link:
    def __init__(self, device_id: str) -> None:
        self.device_id = device_id


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    for key, value in GIT_ENV.items():
        monkeypatch.setenv(key, value)
    built = make_rig(tmp_path / "ws")
    monkeypatch.setenv("HOME", str(built.home))
    workset_move._SERVE_CURSOR.clear()
    workset_move._PREFIX_QUERIES.clear()
    return built


@pytest.fixture()
def staged(rig: Rig, tmp_path: Path) -> tuple[_Server, str]:
    """An owner with a prepared move to ``DEST`` and one planned artifact."""
    root = tmp_path / "owner"
    root.mkdir()
    server = _Server(root, rig.home)
    facts = workset.worktree_facts(rig.feat, home=rig.home, primary=rig.primary)
    manifest = workset.Manifest(
        to_device=DEST, to_name="dest", worktrees=(facts,), carries_files=True
    )
    bound, _offer = workset_move.source_prepare(
        server, SID, manifest, lease_epoch=EPOCH, to_device=DEST  # type: ignore[arg-type]
    )
    write_handoff_entry(
        root,
        SID,
        {
            "role": "source",
            "phase": "prepared",
            "mode": "move",
            "to_device": DEST,
            "lease_epoch": EPOCH,
            "instance_id": "i",
            "at": time.time(),
            "working_set": bound,
        },
    )
    frame = {"session_id": SID, "lease_epoch": EPOCH, "have": {}}
    planned = workset_move.serve_plan(server, _Link(DEST), frame)  # type: ignore[arg-type]
    assert planned["result"] == "planned", planned
    (artifact,) = planned["artifacts"]
    return server, str(artifact["digest"])


def _fetch(server: _Server, device: str = DEST, **frame: Any) -> dict[str, Any]:
    return workset_move.serve_fetch(
        server,  # type: ignore[arg-type]
        _Link(device),  # type: ignore[arg-type]
        {"session_id": SID, "lease_epoch": EPOCH, **frame},
    )


def _serve_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "network" / "audit.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("event") == "session.workset.serve"]


@pytest.fixture()
def spans(staged: tuple[_Server, str], monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Every span the owner reads, with the number of serve rows on disk AT THAT MOMENT."""
    seen: list[tuple[int, int]] = []
    real = sync_mod.read_open_span
    root = staged[0].root

    def spy(handle: Any, *, offset: int, limit: int, size: int) -> Any:
        seen.append((offset, len(_serve_rows(root))))
        return real(handle, offset=offset, limit=limit, size=size)

    monkeypatch.setattr(sync_mod, "read_open_span", spy)
    return seen


def test_a_wrong_lease_epoch_is_refused_with_zero_bytes(
    staged: tuple[_Server, str], spans: list[tuple[int, int]]
) -> None:
    server, digest = staged
    answer = workset_move.serve_fetch(
        server,  # type: ignore[arg-type]
        _Link(DEST),  # type: ignore[arg-type]
        {"session_id": SID, "lease_epoch": "epoch-0", "digest": digest, "offset": 0},
    )
    assert answer["result"] == "refused" and "data" not in answer, answer
    assert spans == [] and _serve_rows(server.root) == []


def test_a_device_that_is_not_the_destination_is_refused(
    staged: tuple[_Server, str], spans: list[tuple[int, int]]
) -> None:
    server, digest = staged
    answer = _fetch(server, device="d_third_member", digest=digest, offset=0)
    assert answer["result"] == "refused" and answer["code"] == "not_authorised", answer
    assert spans == [] and _serve_rows(server.root) == []
    plan = workset_move.serve_plan(
        server,  # type: ignore[arg-type]
        _Link("d_third_member"),  # type: ignore[arg-type]
        {"session_id": SID, "lease_epoch": EPOCH, "have": {}},
    )
    assert plan["result"] == "refused" and plan["code"] == "not_authorised", plan


@pytest.mark.parametrize(
    "name",
    [
        "sha256:" + "0" * 64,  # well-formed, not staged for this move
        "{hex}.json",  # the sidecar beside the artifact
        "journal.json",
        "sha256:../journal.json",
        "sha256:{hex}/../journal.json",
    ],
)
def test_only_a_journaled_64_hex_artifact_is_served(
    staged: tuple[_Server, str], spans: list[tuple[int, int]], name: str
) -> None:
    server, digest = staged
    hexdigest = digest.removeprefix("sha256:")
    answer = _fetch(server, digest=name.format(hex=hexdigest), offset=0)
    assert answer["result"] == "refused" and answer["code"] == "unknown_item", answer
    assert spans == [] and _serve_rows(server.root) == []


def test_the_serve_audit_is_on_disk_before_the_first_and_every_resumed_chunk(
    staged: tuple[_Server, str], spans: list[tuple[int, int]]
) -> None:
    server, digest = staged
    first = _fetch(server, digest=digest, offset=0, limit=64)
    assert first["result"] == "chunk" and first["offset"] == 0, first
    contiguous = _fetch(server, digest=digest, offset=64, limit=64)
    assert contiguous["result"] == "chunk"
    # A relay restart (or a new pull) loses the contiguity: the next nonzero-offset
    # serve is a RESUME and must be audited again before its first byte.
    workset_move._SERVE_CURSOR.clear()
    resumed = _fetch(server, digest=digest, offset=128, limit=64)
    assert resumed["result"] == "chunk"
    # (offset, serve rows already on disk when that span was read)
    assert spans == [(0, 1), (64, 1), (128, 2)], spans
    rows = _serve_rows(server.root)
    assert [r["detail"]["offset"] for r in rows] == [0, 128]
    for row in rows:
        assert row["detail"]["to_device"] == DEST
        assert row["detail"]["lease_epoch"] == EPOCH
        assert row["detail"]["artifact"] == digest
        assert row["detail"]["path"] == "~/proj/.worktrees/feat"
        assert "data" not in row["detail"]


def test_an_audit_that_cannot_be_proven_written_serves_zero_bytes(
    staged: tuple[_Server, str], spans: list[tuple[int, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    server, digest = staged
    # The fsync fails (``sync`` reports it as ``degraded`` and does not count it).
    monkeypatch.setattr(server.audit, "sync", lambda: None)
    answer = _fetch(server, digest=digest, offset=0)
    assert answer["result"] == "refused" and answer["code"] == "audit_unavailable", answer
    assert "data" not in answer and spans == []


# The serve gate is ``AuditLog.record_required`` (PR #4228). These cells pin its
# fail-closed contract at the serve. Against the pre-#4228 stand-in (which read
# ``_enabled`` and the fsync counter from the writer's privates) the disabled and
# fsync cells were already green, the degraded cell was RED — it served a chunk from
# a log already known to have a hole — and the healthy cell was red because nothing
# went through ``record_required``.


def test_a_disabled_audit_log_serves_zero_bytes(
    staged: tuple[_Server, str], spans: list[tuple[int, int]]
) -> None:
    server, digest = staged
    server.audit = AuditLog(server.root, enabled=False)
    answer = _fetch(server, digest=digest, offset=0)
    assert answer["result"] == "refused" and answer["code"] == "audit_unavailable", answer
    assert "data" not in answer and spans == []
    assert _serve_rows(server.root) == []


def test_a_degraded_audit_log_serves_zero_bytes(
    staged: tuple[_Server, str], spans: list[tuple[int, int]]
) -> None:
    server, digest = staged
    # An earlier failed write left a hole the log cannot see; a row appended after
    # it would sit in a trail already known to be incomplete.
    server.audit.degraded = True
    answer = _fetch(server, digest=digest, offset=0)
    assert answer["result"] == "refused" and answer["code"] == "audit_unavailable", answer
    assert "data" not in answer and spans == []
    assert _serve_rows(server.root) == []


def test_a_failed_audit_fsync_serves_zero_bytes(
    staged: tuple[_Server, str], spans: list[tuple[int, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    server, digest = staged

    def failing_fsync(fd: int) -> None:
        raise OSError("injected serve-row fsync failure")

    monkeypatch.setattr(audit_mod.os, "fsync", failing_fsync)
    answer = _fetch(server, digest=digest, offset=0)
    assert answer["result"] == "refused" and answer["code"] == "audit_unavailable", answer
    assert "data" not in answer and spans == []
    assert server.audit.degraded


def test_a_healthy_log_writes_one_durable_row_before_the_first_byte(
    staged: tuple[_Server, str], spans: list[tuple[int, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    server, digest = staged
    real_required = server.audit.record_required
    answers: list[bool] = []

    def required(event: audit_mod.AuditEvent) -> bool:
        answers.append(real_required(event))
        return answers[-1]

    monkeypatch.setattr(server.audit, "record_required", required)
    syncs = server.audit.syncs
    answer = _fetch(server, digest=digest, offset=0, limit=64)
    assert answer["result"] == "chunk" and answer["offset"] == 0, answer
    assert answers == [True]
    assert server.audit.syncs > syncs, "the serve row was not fsynced"
    assert spans == [(0, 1)], spans
    (row,) = _serve_rows(server.root)
    assert set(row["detail"]) == {
        "peer",
        "to_device",
        "lease_epoch",
        "artifact",
        "bytes",
        "offset",
        "path",
        "kind",
    }
    assert set(row["detail"]) == audit_mod.DETAIL_KEYS["session.workset.serve"]
    assert row["detail"]["kind"] == "span" and row["detail"]["offset"] == 0
    assert row["detail"]["artifact"] == digest and row["detail"]["peer"] == DEST
    assert "session.workset.serve" in AuditLog.DURABLE_EVENTS


class _Transport:
    """The destination's ``ask``, wired straight to the owner's handler."""

    def __init__(self, server: _Server) -> None:
        self.server = server
        self.frames: list[dict[str, Any]] = []

    def ask(self, frame: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        from local_operator.network.mobility import Moved

        self.frames.append(dict(frame))
        answer = workset_move.serve_fetch(self.server, _Link(DEST), frame)  # type: ignore[arg-type]
        if answer.get("result") == "refused":
            raise Moved(str(answer["code"]), str(answer["message"]))
        return answer


def _artifact_bytes(server: _Server, digest: str) -> bytes:
    hexdigest = digest.removeprefix("sha256:")
    return (workset_move.outgoing_dir(server.root, SID) / hexdigest).read_bytes()


def test_a_part_with_a_corrupted_prefix_is_discarded_and_refetched(
    staged: tuple[_Server, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Resume prefixes are span-aligned (security review #4245 S4); a 25-byte span
    # lets this small artifact exercise a 50-byte resume.
    monkeypatch.setattr(workset_move, "_RESUME_ALIGN", 25)
    server, digest = staged
    good = _artifact_bytes(server, digest)
    into = tmp_path / "into"
    into.mkdir()
    part = into / (digest.removeprefix("sha256:") + ".part")
    part.write_bytes(b"\x00" * 50)  # same length as a real prefix, wrong bytes
    transport = _Transport(server)

    landed = workset_move._fetch_artifact(
        transport,
        session_id=SID,
        lease_epoch=EPOCH,
        digest=digest,
        expected_bytes=len(good),
        into=into,
    )

    assert landed.read_bytes() == good
    assert [f.get("prefix") for f in transport.frames if "prefix" in f] == [50]
    assert [f["offset"] for f in transport.frames if "offset" in f][0] == 0
    assert not part.exists()


def test_a_part_with_a_good_prefix_resumes_where_it_stopped(
    staged: tuple[_Server, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workset_move, "_RESUME_ALIGN", 25)
    server, digest = staged
    good = _artifact_bytes(server, digest)
    into = tmp_path / "into"
    into.mkdir()
    (into / (digest.removeprefix("sha256:") + ".part")).write_bytes(good[:50])
    transport = _Transport(server)

    landed = workset_move._fetch_artifact(
        transport,
        session_id=SID,
        lease_epoch=EPOCH,
        digest=digest,
        expected_bytes=len(good),
        into=into,
    )

    assert landed.read_bytes() == good
    assert [f["offset"] for f in transport.frames if "offset" in f][0] == 50
    # The resume was audited before its first byte: the prefix answer itself is the
    # audited serve (S4), and the span that continues it rides that row.
    rows = _serve_rows(server.root)
    assert [(r["detail"]["offset"], r["detail"]["kind"]) for r in rows] == [(50, "prefix")]


# --------------------------------------------------------------------------------------
# Security review #4245, S4 and S5: the resume oracle and repeated plans are bounded.
# --------------------------------------------------------------------------------------


def test_a_resume_prefix_is_aligned_audited_before_answering_and_bounded(
    staged: tuple[_Server, str], spans: list[tuple[int, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """S4: before, ``prefix`` answered the sha256 of ANY leading length with no audit
    row, a byte-at-a-time content oracle. Now: a span-aligned length only, the answer
    is itself an audited serve (row first), and at most two per artifact."""
    server, digest = staged
    monkeypatch.setattr(workset_move, "_RESUME_ALIGN", 25)

    unaligned = _fetch(server, digest=digest, prefix=10)
    assert unaligned["result"] == "refused" and unaligned["code"] == "bad_request", unaligned
    assert _serve_rows(server.root) == []

    first = _fetch(server, digest=digest, prefix=25)
    assert first["result"] == "prefix" and first["prefix"] == 25, first
    rows = _serve_rows(server.root)
    assert [(r["detail"]["offset"], r["detail"]["kind"]) for r in rows] == [(25, "prefix")]

    assert _fetch(server, digest=digest, prefix=50)["result"] == "prefix"
    third = _fetch(server, digest=digest, prefix=25)
    assert third["result"] == "refused" and "prefix_sha256" not in third, third
    assert len(_serve_rows(server.root)) == 2


def test_a_prefix_is_not_answered_when_its_audit_cannot_be_written(
    staged: tuple[_Server, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    server, digest = staged
    monkeypatch.setattr(workset_move, "_RESUME_ALIGN", 25)
    monkeypatch.setattr(server.audit, "sync", lambda: None)
    answer = _fetch(server, digest=digest, prefix=25)
    assert answer["result"] == "refused" and answer["code"] == "audit_unavailable", answer
    assert "prefix_sha256" not in answer


def _plan_again(server: _Server, have: dict[str, list[str]] | None = None) -> dict[str, Any]:
    return workset_move.serve_plan(
        server,  # type: ignore[arg-type]
        _Link(DEST),  # type: ignore[arg-type]
        {"session_id": SID, "lease_epoch": EPOCH, "have": have or {}},
    )


def _staged_files(server: _Server) -> list[str]:
    out = workset_move.outgoing_dir(server.root, SID)
    return sorted(p.name for p in out.iterdir() if p.name != "journal.json")


def test_one_move_answers_one_plan_and_one_replan(staged: tuple[_Server, str]) -> None:
    """S5: every plan stages bundles on the owner's disk; a destination re-planning
    with permutations of ``have`` could fill it. One plan + W5's one re-plan."""
    server, digest = staged
    again = _plan_again(server)
    assert again["result"] == "planned", again
    third = _plan_again(server)
    assert third["result"] == "refused" and third["code"] == "plan_limit", third
    # The re-plan with the same inputs reuses the staged artifact; nothing extra stays.
    hexdigest = digest.removeprefix("sha256:")
    assert _staged_files(server) == [hexdigest, f"{hexdigest}.json"]


def test_an_over_cap_plan_deletes_the_bundles_it_staged(staged: tuple[_Server, str]) -> None:
    server, _digest = staged
    journal_path = workset_move.outgoing_dir(server.root, SID) / "journal.json"
    journal = json.loads(journal_path.read_text())
    journal["cap_bytes"] = 1
    journal_path.write_text(json.dumps(journal))
    answer = _plan_again(server)
    assert answer["result"] == "refused" and answer["code"] == "working_set_over_cap", answer
    assert _staged_files(server) == []

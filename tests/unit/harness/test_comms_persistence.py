"""Persisting and rehydrating the subagent roster across a resume.

``SubagentComms._records`` is the resume basis: it holds the
``job_id -> session_dir`` mapping ``hub op='resume'`` relaunches a child from,
and the recorded outcome the roster shows after a child's job row is swept.
Both die with the process, so a resumed session cannot see — let alone continue
— the children the previous one launched unless the records are rebuilt from
disk.

``snapshot`` writes the durable half and ``restore`` reads it back. These tests
pin that the round trip preserves resumability and the recorded outcome, that a
child which never got a transcript is not persisted (there is nothing to
resume), and that restore never clobbers a live record.
"""

from __future__ import annotations

from pathlib import Path

from local_operator.harness.comms import SubagentComms
from local_operator.session.transcript import TRANSCRIPT_FILENAME
from tests.unit.harness.test_comms import FakeChild, FakeJobs, FakeParent


def _comms_with_settled_child(tmp_path: Path, jobs: FakeJobs) -> SubagentComms:
    """A comms holding one launched-then-settled child with a real transcript."""
    parent = FakeParent(jobs)
    comms = SubagentComms(parent)  # type: ignore[arg-type]
    session_dir = tmp_path / "child"
    session_dir.mkdir()
    (session_dir / TRANSCRIPT_FILENAME).write_text("{}\n", encoding="utf-8")
    jobs.add("job1", status="running")
    comms.record_launch(
        "job1",
        "explore the repo",
        prompt="Inspect it.",
        effective_prompt="Specialist guidance.\n\nInspect it.",
        launch_message_id="launch-row-1",
    )
    comms.attach("job1", FakeChild(), session_dir)  # type: ignore[arg-type]
    comms.record_outcome("job1", "completed")
    comms.detach("job1")
    jobs.jobs["job1"].status = "completed"
    return comms


def test_snapshot_captures_the_resumable_fields(tmp_path) -> None:
    comms = _comms_with_settled_child(tmp_path, FakeJobs())
    rows = comms.snapshot()
    assert len(rows) == 1
    row = rows[0]
    assert row["job_id"] == "job1"
    assert row["label"] == "explore the repo"
    assert row["session_dir"] == str(tmp_path / "child")
    assert row["outcome"] == "completed"
    assert row["prompt"] == "Inspect it."
    assert row["effective_prompt"] == "Specialist guidance.\n\nInspect it."
    assert row["launch_message_id"] == "launch-row-1"


def test_a_never_started_child_is_not_snapshotted(tmp_path) -> None:
    """No transcript directory means nothing to resume, so nothing to persist."""
    parent = FakeParent(FakeJobs())
    comms = SubagentComms(parent)  # type: ignore[arg-type]
    comms.record_launch("parked", "queued behind the gate")  # never attached
    assert comms.snapshot() == []


def test_restore_rebuilds_a_resumable_record(tmp_path) -> None:
    """The round trip: snapshot from one comms, restore into a fresh one, and
    the resumed roster shows the child as resumable with its transcript."""
    jobs_a = FakeJobs()
    rows = _comms_with_settled_child(tmp_path, jobs_a).snapshot()

    # A fresh session: empty jobs manager (the row was swept), empty comms.
    fresh = SubagentComms(FakeParent(FakeJobs()))  # type: ignore[arg-type]
    fresh.restore(rows)

    roster = fresh.roster()
    assert len(roster) == 1
    info = roster[0]
    assert info.job_id == "job1"
    assert info.label == "explore the repo"
    assert info.status == "completed"
    assert info.resumable is True
    assert info.session_id == "child"
    node = fresh.node("job1")
    assert node is not None
    assert node.prompt == "Inspect it."
    assert node.effective_prompt == "Specialist guidance.\n\nInspect it."
    assert node.launch_message_id == "launch-row-1"


def test_restore_preserves_a_failure_outcome(tmp_path) -> None:
    jobs = FakeJobs()
    parent = FakeParent(jobs)
    comms = SubagentComms(parent)  # type: ignore[arg-type]
    session_dir = tmp_path / "c2"
    session_dir.mkdir()
    (session_dir / TRANSCRIPT_FILENAME).write_text("{}\n", encoding="utf-8")
    jobs.add("j2", status="running")
    comms.record_launch("j2", "flaky task")
    comms.attach("j2", FakeChild(), session_dir)  # type: ignore[arg-type]
    comms.record_outcome("j2", "failed", "boom")
    comms.detach("j2")

    fresh = SubagentComms(FakeParent(FakeJobs()))  # type: ignore[arg-type]
    fresh.restore(comms.snapshot())
    info = fresh.roster()[0]
    assert info.status == "failed"
    assert info.detail is not None and "boom" in info.detail


def test_restore_collapses_legacy_resume_records_and_preserves_aliases(tmp_path) -> None:
    session_dir = tmp_path / "child"
    rows = [
        {"job_id": "old", "label": "same", "session_dir": str(session_dir)},
        {"job_id": "new", "label": "same", "session_dir": str(session_dir)},
    ]
    comms = SubagentComms(FakeParent(FakeJobs()))  # type: ignore[arg-type]
    comms.restore(rows)

    assert [item.job_id for item in comms.roster()] == ["new"]
    assert comms.session_dir_of("old") == session_dir
    assert comms.label_of("old") == "same"
    assert comms.snapshot()[0]["attempt_aliases"] == ["old"]


def test_attach_collapse_keeps_prior_attempt_launch_prompts(tmp_path) -> None:
    """A resumed attempt inherits every collapsed launch's concise prompt.

    #314 folds the settled predecessor into the newest record, whose own
    ``prompt``/``launch_message_id`` only describe the continuation. The viewer
    reconciles ALL durable launch rows in the shared transcript, so the prior
    attempt's concise prompt must survive the collapse keyed by ITS deterministic
    launch identity (review round 4 R4-1).
    """
    jobs = FakeJobs()
    parent = FakeParent(jobs)
    comms = SubagentComms(parent)  # type: ignore[arg-type]
    session_dir = tmp_path / "child"
    session_dir.mkdir()
    (session_dir / TRANSCRIPT_FILENAME).write_text("{}\n", encoding="utf-8")

    jobs.add("attempt-1", status="running")
    comms.record_launch(
        "attempt-1",
        "reviewer",
        prompt="Original task.",
        effective_prompt="[role: reviewer]\nPREAMBLE\nOriginal task.",
        launch_message_id="subagent-launch:attempt-1",
    )
    comms.attach("attempt-1", FakeChild(), session_dir)  # type: ignore[arg-type]
    comms.record_outcome("attempt-1", "cancelled")
    comms.detach("attempt-1")
    jobs.jobs["attempt-1"].status = "cancelled"

    jobs.add("attempt-2", status="running")
    comms.record_launch(
        "attempt-2",
        "reviewer",
        prompt="Wrap up.",
        effective_prompt="[role: reviewer]\nPREAMBLE\nWrap up.",
        launch_message_id="subagent-launch:attempt-2",
    )
    comms.attach("attempt-2", FakeChild(), session_dir)  # type: ignore[arg-type]

    node = comms.node("attempt-2")
    assert node is not None
    # Both deterministic launch rows resolve to their own concise instruction.
    assert node.launch_prompts == {
        "subagent-launch:attempt-1": "Original task.",
        "subagent-launch:attempt-2": "Wrap up.",
    }
    # The predecessor id still resolves to the collapsed record.
    assert comms.node("attempt-1") is node or comms.node("attempt-1") == node


def test_fanned_resume_keeps_each_target_launch_identity_and_prompts_distinct(
    tmp_path,
) -> None:
    """#318's `hub op='resume'` fan-out must not entangle launch identities.

    A batch resume loops the SAME single-child ``comms.resume`` per target, each
    on its own session directory, so this PR's per-attempt ``launch_prompts``
    folding stays strictly per-record. Prove two children resumed in one fan-out
    keep distinct deterministic launch ids and each folds only ITS OWN prior
    attempt's concise prompt — no cross-target leakage (round 5 semantic check
    against #318).
    """

    def make_attempt(comms, jobs, session_dir, job_id, prompt) -> None:
        jobs.add(job_id, status="running")
        comms.record_launch(
            job_id,
            "reviewer",
            prompt=prompt,
            launch_message_id=f"subagent-launch:{job_id}",
        )
        comms.attach(job_id, FakeChild(), session_dir)  # type: ignore[arg-type]

    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    dir_a = tmp_path / "child-a"
    dir_a.mkdir()
    (dir_a / TRANSCRIPT_FILENAME).write_text("{}\n", encoding="utf-8")
    dir_b = tmp_path / "child-b"
    dir_b.mkdir()
    (dir_b / TRANSCRIPT_FILENAME).write_text("{}\n", encoding="utf-8")

    # Each target has its own original attempt on its own directory.
    make_attempt(comms, jobs, dir_a, "a1", "Task A.")
    comms.record_outcome("a1", "cancelled")
    comms.detach("a1")
    jobs.jobs["a1"].status = "cancelled"
    make_attempt(comms, jobs, dir_b, "b1", "Task B.")
    comms.record_outcome("b1", "cancelled")
    comms.detach("b1")
    jobs.jobs["b1"].status = "cancelled"

    # The fan-out resumes both, each as its own continuation on its own dir.
    make_attempt(comms, jobs, dir_a, "a2", "Continue A.")
    make_attempt(comms, jobs, dir_b, "b2", "Continue B.")

    node_a = comms.node("a2")
    node_b = comms.node("b2")
    assert node_a is not None and node_b is not None
    # Each target folds ONLY its own predecessor; no cross-directory bleed.
    assert node_a.launch_prompts == {
        "subagent-launch:a1": "Task A.",
        "subagent-launch:a2": "Continue A.",
    }
    assert node_b.launch_prompts == {
        "subagent-launch:b1": "Task B.",
        "subagent-launch:b2": "Continue B.",
    }
    assert node_a.launch_message_id != node_b.launch_message_id


def test_snapshot_restore_round_trips_collapsed_launch_prompts(tmp_path) -> None:
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    session_dir = tmp_path / "child"
    session_dir.mkdir()
    (session_dir / TRANSCRIPT_FILENAME).write_text("{}\n", encoding="utf-8")
    jobs.add("attempt-1", status="running")
    comms.record_launch(
        "attempt-1",
        "reviewer",
        prompt="Original task.",
        launch_message_id="subagent-launch:attempt-1",
    )
    comms.attach("attempt-1", FakeChild(), session_dir)  # type: ignore[arg-type]
    comms.record_outcome("attempt-1", "cancelled")
    comms.detach("attempt-1")
    jobs.jobs["attempt-1"].status = "cancelled"
    jobs.add("attempt-2", status="running")
    comms.record_launch(
        "attempt-2",
        "reviewer",
        prompt="Wrap up.",
        launch_message_id="subagent-launch:attempt-2",
    )
    comms.attach("attempt-2", FakeChild(), session_dir)  # type: ignore[arg-type]

    fresh = SubagentComms(FakeParent(FakeJobs()))  # type: ignore[arg-type]
    fresh.restore(comms.snapshot())
    node = fresh.node("attempt-2")
    assert node is not None
    assert node.launch_prompts == {
        "subagent-launch:attempt-1": "Original task.",
        "subagent-launch:attempt-2": "Wrap up.",
    }


def test_restore_folds_legacy_per_attempt_launch_prompts(tmp_path) -> None:
    """A legacy snapshot with one record per resume yields every launch prompt.

    Older builds stored a separate durable record per attempt. Restore collapses
    them by directory; the loser's own launch prompt must fold under the winner
    so the viewer can still reconcile the earlier durable launch row.
    """
    session_dir = tmp_path / "child"
    comms = SubagentComms(FakeParent(FakeJobs()))  # type: ignore[arg-type]
    comms.restore(
        [
            {
                "job_id": "subagent-launch-old",
                "label": "same",
                "session_dir": str(session_dir),
                "prompt": "Original task.",
                "launch_message_id": "subagent-launch:old",
            },
            {
                "job_id": "subagent-launch-new",
                "label": "same",
                "session_dir": str(session_dir),
                "prompt": "Wrap up.",
                "launch_message_id": "subagent-launch:new",
            },
        ]
    )
    node = comms.node("subagent-launch-new")
    assert node is not None
    assert node.launch_prompts == {
        "subagent-launch:old": "Original task.",
        "subagent-launch:new": "Wrap up.",
    }


def test_identical_labels_with_distinct_transcripts_remain_distinct(tmp_path) -> None:
    comms = SubagentComms(FakeParent(FakeJobs()))  # type: ignore[arg-type]
    comms.restore(
        [
            {"job_id": "one", "label": "same", "session_dir": str(tmp_path / "one")},
            {"job_id": "two", "label": "same", "session_dir": str(tmp_path / "two")},
        ]
    )
    assert [item.job_id for item in comms.roster()] == ["one", "two"]


def test_restore_skips_an_id_that_is_already_live(tmp_path) -> None:
    """A stale snapshot must not overwrite a running child of this session."""
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    jobs.add("dup", status="running")
    comms.record_launch("dup", "live child")
    live_child = FakeChild()
    comms.attach("dup", live_child, tmp_path / "live")  # type: ignore[arg-type]

    comms.restore([{"job_id": "dup", "label": "stale", "session_dir": str(tmp_path / "old")}])
    # The live record's label and transcript survive the stale restore.
    assert comms.label_of("dup") == "live child"
    assert comms.session_dir_of("dup") == tmp_path / "live"

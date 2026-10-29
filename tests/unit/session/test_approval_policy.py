"""The conversation's own ``/approvals`` record: read, write and precedence.

The module exists because an explicit per-session choice was memory-only —
resuming the conversation lost it, and a global default could undo it. These
tests pin the three things that make the record safe to trust:

* the READ fails CLOSED. A malformed document is ``ask`` plus a warning, never
  "no choice" — dropping a protective override and inheriting ``auto`` is the
  unsafe direction, and a truncated write is exactly how a record goes bad;
* the WRITE is atomic, private (0600), survives a failed rename without
  damaging the previous document, and does not move the directory's mtime
  (recency ranks by session activity, and recording a policy is not activity);
* the PRECEDENCE is one function, so both readers (the runtime handle and the
  headless ``session_factory`` gate) cannot disagree about it.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from local_operator.session.approval_policy import (
    OVERRIDE_FILENAME,
    SCHEMA_VERSION,
    read_override,
    resolve_effective,
    write_override,
)


def _payload(**overrides: object) -> str:
    document = {"schema_version": SCHEMA_VERSION, "override": None}
    document.update(overrides)
    return json.dumps(document)


def test_a_missing_file_is_no_choice(tmp_path) -> None:
    """A conversation nobody typed ``/approvals`` into: the config decides."""
    assert read_override(tmp_path) == (None, None)


def test_an_explicit_null_is_no_choice(tmp_path) -> None:
    (tmp_path / OVERRIDE_FILENAME).write_text(_payload(), encoding="utf-8")
    assert read_override(tmp_path) == (None, None)


@pytest.mark.parametrize("mode", ["ask", "auto"])
def test_both_modes_round_trip(tmp_path, mode: str) -> None:
    write_override(tmp_path, mode)
    assert read_override(tmp_path) == (mode, None)


def test_corrupt_json_fails_closed_to_ask(tmp_path) -> None:
    """A half-written document must not read as "the user chose nothing".

    That reading would inherit ``auto`` for a session whose human had hardened
    it to ``ask`` — the unsafe direction, and the one a process killed
    mid-write produces.
    """
    (tmp_path / OVERRIDE_FILENAME).write_text('{"schema_version": 1, "overr', encoding="utf-8")
    override, warning = read_override(tmp_path)
    assert override == "ask"
    assert warning is not None and OVERRIDE_FILENAME in warning
    assert warning.startswith("tool approvals: ask")


def test_an_unknown_schema_fails_closed_to_ask(tmp_path) -> None:
    (tmp_path / OVERRIDE_FILENAME).write_text(_payload(schema_version=99), encoding="utf-8")
    override, warning = read_override(tmp_path)
    assert override == "ask"
    assert warning is not None and "schema_version" in warning


def test_a_missing_schema_fails_closed_to_ask(tmp_path) -> None:
    """No version stamp is an unknown version, not an implicit v1."""
    (tmp_path / OVERRIDE_FILENAME).write_text(json.dumps({"override": "auto"}), encoding="utf-8")
    assert read_override(tmp_path)[0] == "ask"


def test_an_invalid_value_fails_closed_to_ask(tmp_path) -> None:
    (tmp_path / OVERRIDE_FILENAME).write_text(_payload(override="sometimes"), encoding="utf-8")
    override, warning = read_override(tmp_path)
    assert override == "ask"
    assert warning is not None and "sometimes" in warning


def test_a_non_object_document_fails_closed_to_ask(tmp_path) -> None:
    (tmp_path / OVERRIDE_FILENAME).write_text("[]", encoding="utf-8")
    assert read_override(tmp_path)[0] == "ask"


def test_the_written_payload_carries_no_global_snapshot(tmp_path) -> None:
    """The record is a CHOICE, not a copy of the machine's default.

    Storing the global value beside it would freeze a decision nobody made in
    this conversation and, worse, let a later reader mistake the global value
    for a per-session one.
    """
    write_override(tmp_path, "auto")
    document = json.loads((tmp_path / OVERRIDE_FILENAME).read_text(encoding="utf-8"))
    assert document == {"schema_version": SCHEMA_VERSION, "override": "auto"}


def test_the_write_is_private_and_creates_directories(tmp_path) -> None:
    directory = tmp_path / "nested" / "session"
    write_override(directory, "ask")
    path = directory / OVERRIDE_FILENAME
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_the_directory_mtime_is_preserved(tmp_path) -> None:
    """Recording a policy is bookkeeping ABOUT a session, never activity IN it.

    The ``/resume`` picker ranks by session activity; a write that bumped the
    directory's mtime would move the conversation to the top of the list.
    """
    stamp = 1_600_000_000
    os.utime(tmp_path, (stamp, stamp))
    write_override(tmp_path, "auto")
    assert tmp_path.stat().st_mtime == pytest.approx(stamp, abs=1)


def test_a_failed_rename_leaves_the_previous_document_intact(tmp_path, monkeypatch) -> None:
    """Atomicity is the point of the temp-and-replace dance.

    The rename is made to fail AFTER the temp is written, which is the window a
    crash or a full disk lands in: the old record must still be what a reader
    sees, and no ``.tmp`` litter may be left for the next writer to trip on.
    """
    write_override(tmp_path, "ask")
    before = (tmp_path / OVERRIDE_FILENAME).read_text(encoding="utf-8")

    def _refuse(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("local_operator.session.approval_policy.os.replace", _refuse)
    with pytest.raises(OSError):
        write_override(tmp_path, "auto")

    assert (tmp_path / OVERRIDE_FILENAME).read_text(encoding="utf-8") == before
    assert read_override(tmp_path) == ("ask", None)
    assert not [name for name in os.listdir(tmp_path) if name.endswith(".tmp")]


def test_an_invalid_mode_is_refused_before_anything_is_written(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_override(tmp_path, "sometimes")
    assert not (tmp_path / OVERRIDE_FILENAME).exists()


@pytest.mark.parametrize(
    ("override", "global_mode", "pinned", "expected"),
    [
        # The conversation's decision wins BOTH ways.
        ("auto", "ask", None, ("auto", "session")),
        ("ask", "auto", None, ("ask", "session")),
        ("auto", "auto", None, ("auto", "session")),
        ("ask", "ask", None, ("ask", "session")),
        # No decision: the machine default decides.
        (None, "auto", None, ("auto", "global")),
        (None, "ask", None, ("ask", "global")),
        # A flag on this run outranks everything durable.
        ("ask", "ask", True, ("auto", "pin")),
        ("auto", "ask", True, ("auto", "pin")),
        ("auto", "auto", False, ("ask", "pin")),
        # A corrupt/unknown global value degrades to the safe gate.
        (None, "sometimes", None, ("ask", "global")),
        (None, None, None, ("ask", "global")),
        # An unknown override is NOT a session choice the file recorded, but it
        # must not fall through to a global `auto` either — the reader already
        # fails a corrupt record closed to `ask`, and this function agrees.
        ("sometimes", "auto", None, ("ask", "session")),
    ],
)
def test_resolve_effective_precedence(
    override: str | None, global_mode: str | None, pinned: bool | None, expected
) -> None:
    assert resolve_effective(override, global_mode, pinned_auto=pinned) == expected

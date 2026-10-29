"""The device-level promotion index (LOH-9 S4).

The index answers one question — where did this account definition come from —
and these tests pin the three properties that make it usable as an audit trail:
the row shape is fixed and complete, an append is atomic enough that a rolled-back
promotion leaves no id behind, and the file is BOUNDED (an append-only file that
nothing rotates is a leak with a receipt attached).
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from local_operator import role_provenance
from local_operator.role_provenance import (
    RECORD_FIELDS,
    append_promotions,
    provenance_path,
    read_promotions,
    require_writable,
)


def _record(**overrides):
    base = {
        "kind": "agent",
        "id": str(uuid.uuid4()),
        "name": "release-check",
        "source_session": "aaaaaaaa",
        "at": "2026-10-22T00:00:00+00:00",
        "authorized_by": "surface:tui",
        "replaced": None,
    }
    base.update(overrides)
    return base


def test_append_and_read_round_trip(tmp_path: Path) -> None:
    ids = append_promotions(
        tmp_path,
        [
            _record(name="one"),
            _record(kind="team", name="pod", replaced="pod"),
        ],
    )

    rows = read_promotions(tmp_path)

    assert [row["id"] for row in rows] == ids
    assert [row["name"] for row in rows] == ["one", "pod"]
    assert rows[1]["kind"] == "team"
    assert rows[1]["replaced"] == "pod"
    assert set(rows[0]) == set(RECORD_FIELDS)


def test_the_row_carries_only_the_declared_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown provenance field"):
        append_promotions(tmp_path, [_record(secret="x")])

    missing = _record()
    missing.pop("authorized_by")
    with pytest.raises(ValueError, match="'authorized_by' is required"):
        append_promotions(tmp_path, [missing])


def test_a_non_uuid_definition_id_is_recorded(tmp_path: Path) -> None:
    """An id is any safe path segment (``validate_agent_id``), not a UUID.

    A row mirrored from an older peer or imported from an archive can carry one,
    and refusing it here would fail a promotion AFTER its rows were written.
    """

    append_promotions(tmp_path, [_record(id="imported-role-2")])

    assert read_promotions(tmp_path)[0]["id"] == "imported-role-2"


def test_an_empty_batch_writes_nothing(tmp_path: Path) -> None:
    assert append_promotions(tmp_path, []) == []
    assert not provenance_path(tmp_path).exists()


def test_a_torn_trailing_line_is_skipped(tmp_path: Path) -> None:
    append_promotions(tmp_path, [_record(name="kept")])
    with open(provenance_path(tmp_path), "a", encoding="utf-8") as handle:
        handle.write('{"kind": "agent", "id"')

    assert [row["name"] for row in read_promotions(tmp_path)] == ["kept"]


def test_the_index_is_bounded_and_keeps_the_newest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(role_provenance, "MAX_PROVENANCE_ROWS", 3)
    monkeypatch.setattr(role_provenance, "MAX_PROVENANCE_BYTES", 1)

    for index in range(6):
        append_promotions(tmp_path, [_record(name=f"row-{index}")])

    rows = read_promotions(tmp_path)
    assert [row["name"] for row in rows] == ["row-3", "row-4", "row-5"]
    assert provenance_path(tmp_path).stat().st_size <= role_provenance.MAX_PROVENANCE_BYTES + 1024


def test_rows_survive_a_process_boundary(tmp_path: Path) -> None:
    first = _record(name="persisted")
    append_promotions(tmp_path, [first])

    raw = provenance_path(tmp_path).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in raw] == [first["id"]]


def test_require_writable_creates_the_file_and_the_directory(tmp_path: Path) -> None:
    root = tmp_path / "config"
    require_writable(root)

    assert provenance_path(root).exists()


def test_require_writable_refuses_an_unappendable_file(tmp_path: Path) -> None:
    # The probe is a REAL append open rather than a mode check, so a file the
    # process may read but not append to is exactly the case it must catch:
    # discovering that after the rows were created would roll back good work.
    append_promotions(tmp_path, [_record(name="one")])
    provenance_path(tmp_path).chmod(0o400)
    try:
        if os.access(provenance_path(tmp_path), os.W_OK):  # pragma: no cover — root
            pytest.skip("file permissions are not enforced for this user")
        with pytest.raises(OSError):
            require_writable(tmp_path)
    finally:
        provenance_path(tmp_path).chmod(0o600)


def test_a_failed_rotation_records_nothing(tmp_path: Path, monkeypatch) -> None:
    """Rotate BEFORE appending: a rotation failure leaves no row (MINOR-1).

    Appending first meant a rotation that then failed left the batch in the
    index while the caller rolled the promotion back — the index named ids that
    no longer existed, and the user was told nothing had been promoted.
    """

    monkeypatch.setattr(role_provenance, "MAX_PROVENANCE_BYTES", 1)
    append_promotions(tmp_path, [_record(name="old")])

    def boom(*_args, **_kwargs):
        raise OSError("disk full during rotation")

    monkeypatch.setattr(role_provenance, "_atomic_write", boom)
    with pytest.raises(OSError, match="disk full during rotation"):
        append_promotions(tmp_path, [_record(name="new")])

    assert [row["name"] for row in read_promotions(tmp_path)] == ["old"]


def test_a_failed_append_records_nothing(tmp_path: Path, monkeypatch) -> None:
    """The batch is published atomically, so a failed write leaves no row.

    A raw append was durable up to its own write: an EIO after the bytes landed
    left an index row for a promotion whose caller then rolled the definitions
    back, so the index named an id that no longer existed (review round 2). The
    write now goes through ``_atomic_write`` (temp file + one ``os.replace``),
    so a failure leaves the PREVIOUS index intact.
    """

    append_promotions(tmp_path, [_record(name="old")])
    before = provenance_path(tmp_path).read_text(encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError("EIO on append")

    monkeypatch.setattr(role_provenance, "_atomic_write", boom)
    with pytest.raises(OSError, match="EIO on append"):
        append_promotions(tmp_path, [_record(name="new")])

    assert provenance_path(tmp_path).read_text(encoding="utf-8") == before
    assert [row["name"] for row in read_promotions(tmp_path)] == ["old"]

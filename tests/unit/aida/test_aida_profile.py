"""R23's guarded write path: the section, the refusals, and the atomicity.

The write is path-FREE by contract — the caller passes text, never a file — so
these tests pin both halves: what a good note produces, and every refusal the
guard exists for (empty, invalid, oversize, unsafe, failed). "No writes
outside the config root" is pinned with a symlinked file, the one way a
path-free API could still reach somewhere it should not.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_operator.aida import profile


def _text(root: Path) -> str:
    return (root / "system_prompt.md").read_text(encoding="utf-8")


def test_a_note_lands_in_a_marked_section_under_the_config_root(isolated_root: Path) -> None:
    assert profile.record_profile_note("Name: Damian", config_dir=isolated_root) == "recorded"
    text = _text(isolated_root)
    assert text.count(profile.SECTION_START) == 1
    assert text.count(profile.SECTION_END) == 1
    assert profile.SECTION_HEADING in text
    assert "- Name: Damian" in text
    start = text.index(profile.SECTION_START)
    end = text.index(profile.SECTION_END)
    assert start < text.index("- Name: Damian") < end


def test_the_config_root_is_resolved_from_the_env_when_not_passed(isolated_root: Path) -> None:
    """``config_dir()`` is the resolver (its own read-per-call rule), not a constant."""
    assert profile.instruction_file() == isolated_root / "system_prompt.md"
    assert profile.record_profile_note("Prefers to be addressed as Dame") == "recorded"
    assert "- Prefers to be addressed as Dame" in _text(isolated_root)


def test_operator_text_is_never_touched_and_notes_append_inside_one_section(
    isolated_root: Path,
) -> None:
    path = isolated_root / "system_prompt.md"
    path.write_text("My standing rules.\nKeep them.\n", encoding="utf-8")

    assert profile.record_profile_note("Email: op@example.com", config_dir=isolated_root) == (
        "recorded"
    )
    assert profile.record_profile_note("Timezone: UTC", config_dir=isolated_root) == "recorded"

    text = _text(isolated_root)
    assert text.startswith("My standing rules.\nKeep them.\n")
    assert text.count(profile.SECTION_START) == 1
    assert text.index("- Email: op@example.com") < text.index("- Timezone: UTC")
    assert text.index("- Timezone: UTC") < text.index(profile.SECTION_END)


def test_a_repeated_note_is_a_duplicate_and_changes_nothing(isolated_root: Path) -> None:
    assert profile.record_profile_note("Name: Damian", config_dir=isolated_root) == "recorded"
    before = _text(isolated_root)
    assert profile.record_profile_note("Name: Damian", config_dir=isolated_root) == "duplicate"
    assert _text(isolated_root) == before


def test_a_blank_note_is_refused_and_writes_nothing(isolated_root: Path) -> None:
    assert profile.record_profile_note("  \n\t\n", config_dir=isolated_root) == "empty"
    assert not (isolated_root / "system_prompt.md").exists()


def test_section_markers_in_the_note_are_refused(isolated_root: Path) -> None:
    """A note carrying the markers could corrupt the section for later writes."""
    assert (
        profile.record_profile_note(f"x {profile.SECTION_START} y", config_dir=isolated_root)
        == "invalid"
    )
    assert (
        profile.record_profile_note(f"x {profile.SECTION_END} y", config_dir=isolated_root)
        == "invalid"
    )
    assert not (isolated_root / "system_prompt.md").exists()


def test_an_oversize_note_is_refused_and_the_file_is_unchanged(isolated_root: Path) -> None:
    path = isolated_root / "system_prompt.md"
    original = "x" * (profile.MAX_FILE_CHARS - 10)
    path.write_text(original, encoding="utf-8")

    assert profile.record_profile_note("Name: Damian", config_dir=isolated_root) == "oversize"
    assert path.read_text(encoding="utf-8") == original


def test_a_symlink_out_of_the_root_is_refused_and_the_target_untouched(
    isolated_root: Path,
) -> None:
    """The one way the path-free API could still write somewhere else."""
    outside = isolated_root.parent / "outside.md"
    outside.write_text("not lop's file\n", encoding="utf-8")
    (isolated_root / "system_prompt.md").symlink_to(outside)

    assert profile.record_profile_note("Name: sneaky", config_dir=isolated_root) == "unsafe"
    assert outside.read_text(encoding="utf-8") == "not lop's file\n"


def test_a_symlink_to_a_MISSING_out_of_root_target_is_also_refused(
    isolated_root: Path,
) -> None:
    """The dangling arm (review round 1, F2).

    ``path.exists()`` follows the link, so a link whose out-of-root target did
    not exist YET read as "nothing there": the guard was skipped and the write
    then created the target through the link. The existing-target arm above
    was the only one pinned.
    """
    outside = isolated_root.parent / "not-yet.md"
    (isolated_root / "system_prompt.md").symlink_to(outside)

    assert profile.record_profile_note("Name: sneaky", config_dir=isolated_root) == "unsafe"
    assert not outside.exists(), "the refusal must not create the target"


def test_a_symlink_inside_the_root_is_followed(isolated_root: Path) -> None:
    """The guard is about the ROOT boundary, not about refusing symlinks."""
    target = isolated_root / "real_prompt.md"
    target.write_text("", encoding="utf-8")
    (isolated_root / "system_prompt.md").symlink_to(target)

    assert profile.record_profile_note("Name: Damian", config_dir=isolated_root) == "recorded"
    assert "- Name: Damian" in target.read_text(encoding="utf-8")


def test_a_failed_write_reports_failed_and_leaves_no_temp_files(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(src: str, dst: str) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(os, "replace", _boom)

    assert profile.record_profile_note("Name: Damian", config_dir=isolated_root) == "failed"
    leftovers = [p.name for p in isolated_root.iterdir() if p.name.startswith(".system_prompt.md")]
    assert leftovers == []


def test_the_cap_is_the_loaders_own_budget() -> None:
    """The refusal threshold must BE the number the loader bounds the file by.

    Two constants that must agree are one constant plus a drift; the loader's
    is authoritative (``session_factory``), and this pins the pair.
    """
    from local_operator.session_factory import MAX_USER_INSTRUCTIONS_CHARS

    assert profile.MAX_FILE_CHARS == MAX_USER_INSTRUCTIONS_CHARS


# --------------------------------------------------------------------------- #
# The Radient sign-in's identity (audit A5/A6)
# --------------------------------------------------------------------------- #


def test_a_radient_login_writes_name_and_email_once(isolated_root: Path) -> None:
    """Idempotent by replacement: a re-login with the same identity writes
    nothing; a changed name replaces its own line; her notes stay put."""
    assert (
        profile.record_profile_note("Prefers short answers", config_dir=isolated_root) == "recorded"
    )
    credential = {"name": "Jane Doe", "email": "jane@x.com", "type": "oauth"}
    assert profile.record_radient_login(credential, config_dir=isolated_root) == "recorded"
    first = _text(isolated_root)
    assert "- Name: Jane Doe (Radient account)" in first
    assert "- Email: jane@x.com (Radient account)" in first

    assert profile.record_radient_login(credential, config_dir=isolated_root) == "duplicate"
    assert _text(isolated_root) == first

    renamed = {"name": "Jane Q Doe", "email": "jane@x.com"}
    assert profile.record_radient_login(renamed, config_dir=isolated_root) == "recorded"
    text = _text(isolated_root)
    assert text.count("- Name:") == 1 and "Jane Q Doe" in text
    assert "- Prefers short answers" in text
    # The identity sits FIRST under the heading: a stable prefix position.
    body = text.split(profile.SECTION_HEADING, 1)[1].lstrip("\n").splitlines()
    assert body[0].startswith("- Name: Jane Q Doe")


def test_a_login_without_claims_writes_nothing(isolated_root: Path) -> None:
    assert profile.record_radient_login({"type": "oauth"}, config_dir=isolated_root) == "empty"
    assert not (isolated_root / "system_prompt.md").exists()


def test_a_hand_written_name_line_is_never_touched(isolated_root: Path) -> None:
    """Only lines carrying the sign-in's tag are the sign-in's to replace."""
    profile.record_profile_note("Name: Jay (prefers Jay)", config_dir=isolated_root)
    profile.record_radient_login({"name": "Jane Doe", "email": ""}, config_dir=isolated_root)
    text = _text(isolated_root)
    assert "- Name: Jay (prefers Jay)" in text
    assert "- Name: Jane Doe (Radient account)" in text

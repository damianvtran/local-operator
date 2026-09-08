"""``lop config instructions``: provenance for the assembled system prompt.

The command exists because #822 found the packaged guide asserting that lop has
no ``AGENTS.md`` mechanism while lop was reading the reporter's — a divergence
between what the install does and what it says about itself. So the property
these tests pin is not the wording of the box but the CORRESPONDENCE: what the
command reports is what ``resolve_user_instructions`` actually assembles, for
every arrangement an operator can be in. A provenance command that lies is
worse than no provenance command.

The second property is that it never prints the CONTENTS. Those are the
operator's standing rules, routinely thousands of lines, and pasting a report
into an issue must not paste their preferences with it.

``HOME`` is redirected by the autouse ``isolate_environment`` fixture, so these
reach a scratch home rather than the developer's real ``~/.agents`` — which on
a machine that has one would inject their own preferences into every assertion.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from local_operator import ecosystem_instructions as eco
from local_operator.cli import config_instructions_command
from local_operator.session_factory import resolve_user_instructions


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {"agent_name": None}
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch HOME with a redirected config dir, as an isolated run needs.

    Both variables, per AGENTS.md "Isolating a run": ``LOCAL_OPERATOR_CONFIG_DIR``
    alone leaves anything deriving from the home directory pointed at the real
    one, and ``~/.agents/AGENTS.md`` is derived from ``Path.home()``.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / ".local-operator"))
    (tmp_path / ".local-operator").mkdir()
    return tmp_path


def _write_agents_md(home: Path, text: str) -> Path:
    path = home / ".agents" / "AGENTS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_system_prompt(home: Path, text: str) -> Path:
    path = home / ".local-operator" / "system_prompt.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_no_imported_file_reports_none_rather_than_an_empty_box(home: Path, capsys) -> None:
    """The DEFAULT install has no ``~/.agents/AGENTS.md``.

    An empty listing there reads as "the command is broken" rather than as the
    answer it is, which is the whole reason the imported half gets its own box.
    """
    _write_system_prompt(home, "- lop only.\n")

    assert config_instructions_command(_args()) == 0

    out = capsys.readouterr().out
    assert "Files read: none — no imported file exists at those paths" in out
    assert "Source: default (~/.agents/AGENTS.md)" in out


def test_an_imported_file_is_reported_with_its_path_and_size(home: Path, capsys) -> None:
    _write_agents_md(home, "- Shared rule.\n")
    _write_system_prompt(home, "- lop only.\n")

    assert config_instructions_command(_args()) == 0

    out = capsys.readouterr().out
    assert str(home / ".agents" / "AGENTS.md") in out
    # Assembly ORDER is the fact operators get wrong, so it is pinned as an
    # order and not merely as membership.
    assert out.index("1. imported") < out.index("2. system_prompt.md")


def test_a_collapsed_duplicate_is_named_as_collapsed(home: Path, capsys) -> None:
    """The undocumented dedup #822 reports as a silent context cost.

    Reported as read-versus-included rather than as one number: "37 read, 0
    included" is the confirmation an operator is looking for, and a single
    figure cannot express it.
    """
    shared = "- Shared rule one.\n- Shared rule two.\n"
    _write_agents_md(home, shared)
    _write_system_prompt(home, shared)

    assert config_instructions_command(_args()) == 0

    out = capsys.readouterr().out
    assert "Collapsed: identical to instructions already loaded" in out
    assert "(collapsed)" in out


def test_a_superset_native_file_does_not_collapse(home: Path, capsys) -> None:
    """The case the issue reporter had to discover by hand.

    The collapse is keyed on the digest of the WHOLE file, so shared rules plus
    a lop-only overlay pays for both copies. The report must show that as two
    contributing sources, since it is the only place an operator finds out.
    """
    shared = "- Shared rule.\n"
    _write_agents_md(home, shared)
    _write_system_prompt(home, shared + "- lop-only extra.\n")

    assert config_instructions_command(_args()) == 0

    out = capsys.readouterr().out
    assert "Collapsed" not in out
    assert "Included: 14 chars" in out  # the imported file, paid for in full


def test_the_env_override_is_reflected_in_the_report(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    elsewhere = tmp_path / "elsewhere" / "AGENTS.md"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("- Redirected rule.\n", encoding="utf-8")
    _write_agents_md(home, "- Default rule that must NOT be read.\n")
    monkeypatch.setenv(eco.ECOSYSTEM_INSTRUCTIONS_ENV, str(elsewhere))

    assert config_instructions_command(_args()) == 0

    out = capsys.readouterr().out
    assert f"{eco.ECOSYSTEM_INSTRUCTIONS_ENV}={elsewhere}" in out
    assert str(elsewhere) in out
    assert "Default rule" not in out


def test_an_empty_override_reports_the_feature_as_disabled(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """ "Nothing imported" has two causes with opposite fixes.

    An operator told "no file exists" while the feature is switched off goes
    looking in the wrong place entirely.
    """
    _write_agents_md(home, "- Shared rule.\n")
    monkeypatch.setenv(eco.ECOSYSTEM_INSTRUCTIONS_ENV, "")

    assert config_instructions_command(_args()) == 0

    out = capsys.readouterr().out
    assert "DISABLED" in out
    assert "Files read: none — the feature is disabled" in out


def test_a_file_over_the_cap_is_reported_as_truncated(home: Path, capsys) -> None:
    _write_agents_md(home, "z" * (eco.MAX_FILE_BYTES * 3))
    _write_system_prompt(home, "- lop only.\n")

    assert config_instructions_command(_args()) == 0

    out = capsys.readouterr().out
    assert "Truncated: hit the size cap" in out


def test_an_unreadable_file_is_named_as_unreadable_not_as_absent(home: Path, capsys) -> None:
    """A degraded session, not a default one.

    The loader skips an unreadable file so a bad mode never costs a session —
    which means the operator's rules are silently missing. "Empty" or "none"
    here sends them to create a file that already exists.
    """
    path = _write_agents_md(home, "- Shared rule.\n")
    path.chmod(0o000)
    try:
        if os.access(path, os.R_OK):  # pragma: no cover — root ignores the mode
            pytest.skip("running as root; the mode is not enforced")
        assert config_instructions_command(_args()) == 0
        out = capsys.readouterr().out
    finally:
        path.chmod(0o644)

    assert "Unreadable: skipped; the session ran without it" in out
    assert "unreadable; skipped" in out


def test_an_existing_but_empty_file_is_reported_as_empty_not_as_absent(home: Path, capsys) -> None:
    """A zero-byte ``~/.agents/AGENTS.md`` EXISTS.

    Reporting "no imported file exists at those paths" about a path that has a
    file on it is a statement about the filesystem that is simply false, and it
    is the mirror image of the distinction ``InstructionSource.unreadable``
    exists to draw: an operator who truncated their shared file while debugging
    must be sent to the empty file they have, not to a missing one.
    """
    _write_agents_md(home, "")
    _write_system_prompt(home, "- lop only.\n")

    assert config_instructions_command(_args()) == 0

    out = capsys.readouterr().out
    assert "no imported file exists at those paths" not in out
    # The stronger claim for an imported row: the loader lists only paths that
    # resolve to a real file, so "might not exist" would understate what is known.
    assert "Empty: the file is there but holds no instructions" in out
    assert "(empty)" in out
    assert str(home / ".agents" / "AGENTS.md") in out


def test_a_whitespace_only_file_is_reported_as_empty_not_as_absent(home: Path, capsys) -> None:
    """Same rule as the zero-byte case: the bytes are there, the rules are not."""
    _write_agents_md(home, "\n\n   \t\n")
    _write_system_prompt(home, "- lop only.\n")

    assert config_instructions_command(_args()) == 0

    out = capsys.readouterr().out
    assert "no imported file exists at those paths" not in out
    assert "Empty: the file is there but holds no instructions" in out


def test_a_superset_native_file_names_the_overlap_it_pays_for(home: Path, capsys) -> None:
    """The one arrangement the digest collapse cannot catch.

    Without this row the frame is byte-identical to two genuinely distinct
    files, so the guide's "check with ``config instructions``" pointed at a
    diagnosis the output could not give — the gap #822 reported, one level down.
    The overlapping TEXT must never appear: the count is the answer.

    The other source is named by ROW NUMBER rather than label because labels are
    not unique — several override paths all render as ``imported`` — and because
    the label form did not fit 80 columns, wrapping the row out of the box.
    """
    shared = "- SHARED-OVERLAP-MARKER rule." + "x" * 200
    _write_agents_md(home, f"{shared}\n")
    _write_system_prompt(home, f"{shared}\n\n- lop only.\n")

    assert config_instructions_command(_args()) == 0

    out = capsys.readouterr().out
    assert "Overlaps: contains source 1 verbatim (229 chars); both copies are sent" in out
    assert "SHARED-OVERLAP-MARKER" not in out


def test_the_overlap_row_reports_the_migration_case_on_the_subset(home: Path, capsys) -> None:
    """Rules moved into the shared file and grew there, leaving the old
    ``system_prompt.md`` behind as a subset. Both copies still ship, so silence
    here would be the report asserting an all-clear it has not established — and
    the direction has to be on the row, because the remedy differs: delete the
    subset rather than trim the superset."""
    shared = "- SHARED-OVERLAP-MARKER rule." + "y" * 200
    _write_agents_md(home, f"{shared}\n\n- Grown since the move.\n")
    _write_system_prompt(home, f"{shared}\n")

    assert config_instructions_command(_args()) == 0

    out = capsys.readouterr().out
    assert "Overlaps: verbatim inside source 1 (229 chars); both copies are sent" in out
    assert "SHARED-OVERLAP-MARKER" not in out


def test_every_overlap_row_fits_inside_the_box_at_eighty_columns(home: Path, capsys) -> None:
    """The box does no wrapping of its own, so a row past 80 characters
    soft-wraps in a standard terminal and the overflow lands outside the “│”
    gutter — the failure the two-line footer below already avoids. Asserted on
    the widest realistic inputs (a 64,000-character span, a two-digit row
    number) rather than on the default install, because it is the large numbers
    that push the row over."""
    shared = "- Shared rule." + "z" * 200
    _write_agents_md(home, f"{shared}\n")
    _write_system_prompt(home, f"{shared}\n\n- lop only.\n")

    assert config_instructions_command(_args()) == 0
    rendered = [line for line in capsys.readouterr().out.splitlines() if "Overlaps:" in line]
    assert rendered and all(len(line) <= 80 for line in rendered)

    # The widest states the format string can reach, measured directly rather
    # than by constructing a 64,000-character file per case.
    for index in (1, 9, 10, 99):
        for count in (1, 999, 1_000, 63_998, 64_000):
            for direction in (
                f"contains source {index} verbatim",
                f"verbatim inside source {index}",
            ):
                row = f"│    Overlaps: {direction} ({count:,} chars); both copies are sent"
                assert len(row) <= 80, row


def test_the_overlap_row_stays_silent_on_the_collapsed_and_distinct_cases(
    home: Path, capsys
) -> None:
    """``Overlaps:`` is a cost the operator can remove, so it must not fire on
    the two arrangements where nothing is being paid twice: identical files are
    already collapsed and reported as such, and two distinct files are healthy.
    A warning on a healthy install is a warning operators learn to ignore."""
    _write_agents_md(home, "- Same.\n")
    _write_system_prompt(home, "- Same.\n")
    assert config_instructions_command(_args()) == 0
    collapsed = capsys.readouterr().out
    assert "Collapsed:" in collapsed
    assert "Overlaps:" not in collapsed

    _write_agents_md(home, "- Shared rule.\n")
    _write_system_prompt(home, "- Entirely different.\n")
    assert config_instructions_command(_args()) == 0
    assert "Overlaps:" not in capsys.readouterr().out

    # Third silent case: containment too short to be worth an operator's
    # attention. "- " appears in both files, which is literal containment.
    _write_agents_md(home, "- \n")
    _write_system_prompt(home, "- Entirely different.\n")
    assert config_instructions_command(_args()) == 0
    assert "Overlaps:" not in capsys.readouterr().out


def test_the_report_never_prints_the_contents(home: Path, capsys) -> None:
    """Paths and sizes, never the rules themselves."""
    _write_agents_md(home, "- SHARED-SECRET-MARKER rule.\n")
    _write_system_prompt(home, "- NATIVE-SECRET-MARKER rule.\n")

    assert config_instructions_command(_args()) == 0

    out = capsys.readouterr().out
    assert "SHARED-SECRET-MARKER" not in out
    assert "NATIVE-SECRET-MARKER" not in out


def test_the_command_writes_nothing(home: Path, capsys) -> None:
    """Read-only by contract: ``system_prompt.md`` is lop's only write target,
    and a provenance command that touched a file would be the one surface able
    to break that. Asserted against the filesystem, because "we do not call
    write" is exactly the claim a later refactor invalidates silently."""
    imported = _write_agents_md(home, "- Shared rule.\n")
    native = _write_system_prompt(home, "- lop only.\n")
    before = {path: path.stat().st_mtime_ns for path in (imported, native)}

    assert config_instructions_command(_args()) == 0
    capsys.readouterr()

    assert {path: path.stat().st_mtime_ns for path in (imported, native)} == before
    # And no file was created anywhere under the two roots it can see.
    assert sorted(p.name for p in (home / ".agents").iterdir()) == ["AGENTS.md"]


def test_the_command_writes_nothing_under_agent(tmp_path: Path, monkeypatch, capsys) -> None:
    """``--agent`` is the ONE branch that can write, and the writes-nothing test
    above cannot see it: it passes no agent, so it never reaches the registry,
    and it scans only ``~/.agents`` — never the config dir, which is where
    ``AgentRegistry.__init__``'s ``mkdir`` lands. On a machine with no config
    root, asking a read-only provenance command about an agent materialised
    ``<config_dir>/`` and ``<config_dir>/agents/``.

    Deliberately NOT using the ``home`` fixture, which pre-creates the config
    dir: the defect only shows on a home that has none.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / ".local-operator"))

    assert config_instructions_command(_args(agent_name="ghost")) == 1

    err = capsys.readouterr().err
    assert "No agent found with name: ghost" in err
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_an_unknown_agent_name_is_refused_without_creating_it(home: Path, capsys) -> None:
    """The interactive path CREATES a missing named agent. This command must
    not: it is read-only, and reporting on an agent it just invented would
    describe a session nobody asked for."""
    _write_system_prompt(home, "- lop only.\n")

    assert config_instructions_command(_args(agent_name="nonexistent-agent")) == 1

    err = capsys.readouterr().err
    assert "No agent found with name: nonexistent-agent" in err


@pytest.mark.parametrize(
    "shared,native",
    [
        (None, "- lop only.\n"),
        ("- Shared rule.\n", "- lop only.\n"),
        ("- Same.\n", "- Same.\n"),
        ("- Shared.\n", "- Shared.\n- Extra.\n"),
        ("z" * (eco.MAX_FILE_BYTES * 3), "- lop only.\n"),
    ],
    ids=["absent", "distinct", "collapsed", "superset", "over-cap"],
)
def test_the_reported_total_is_the_prompt_that_is_actually_assembled(
    home: Path, capsys, shared: str | None, native: str
) -> None:
    """The property the command exists for, across every arrangement.

    ``resolve_user_instructions`` is the call ``create_session`` makes, so this
    compares the report against the real prompt rather than against a second
    model of it — the divergence being guarded is precisely a report drifting
    from the thing it reports on.
    """
    if shared is not None:
        _write_agents_md(home, shared)
    _write_system_prompt(home, native)

    assert config_instructions_command(_args()) == 0
    out = capsys.readouterr().out

    assembled, _ = resolve_user_instructions()
    line = next(ln for ln in out.splitlines() if "Total assembled" in ln)
    reported = int(line.split(":")[1].split("of")[0].strip().replace(",", ""))
    assert reported == len(assembled)

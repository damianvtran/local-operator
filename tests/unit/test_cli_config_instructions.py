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

"""The executable form of the per-session rules POC spec (SPEC-session-rules.md).

Frozen against base ``origin/main`` f2187d4 with the architect decisions D1-D6
(``session-rules-decisions.md``). This file IS the contract: a contender makes it
pass and does not edit it.

Every import of the feature under test is LAZY (``importlib.import_module``
inside the test or a helper). The feature does not exist on the base, so a
module-scope import would turn the whole file into one collection error and hide
WHICH behaviour is missing; instead each test fails on its own, naming the piece
it needs.

Two seams the spec fixes as a BEHAVIOUR rather than as a parameter name, and how
they are read here:

* **The provider's session directory.** Driven the way existing tests drive it
  (``tests/unit/session/test_memory_wire.py``, ``tests/unit/test_session_factory.py``):
  a real :class:`~local_operator.session.transcript.Transcript` whose
  ``.directory`` IS the session directory, plus an optional
  ``session_dir=``/``rules_dir=``/``rules_directory=``/``session_directory=``
  keyword when the factory has one. Both routes are accepted because the spec
  says "from the session's own directory" and never names the seam.
* **The dev CLI envelope.** The spec pins the exit codes and that every command
  speaks JSON, but not whether ``list`` prints a bare array or
  ``{"rules": [...]}``, nor whether a mutation prints the rule, ``{"rule": ...}``
  or the whole list. Those shapes are read tolerantly; the ids, texts and exit
  codes are asserted exactly.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import local_operator.prompts_api as prompts_api
from local_operator.prompts_api import (
    assemble_system_block_sections,
    build_system_blocks,
    split_tail_sections,
)
from local_operator.session.transcript import Transcript

# ---------------------------------------------------------------------------
# The frozen strings (SPEC "Flag", "Data", "Module API", "Rendering")
# ---------------------------------------------------------------------------

RULES_ENV_NAME = "LOP_SESSION_RULES"
SIDECAR = "rules.json"

#: Appended to block 0's "Session state updates" paragraph whenever the flag is
#: on -- including at zero rules, so adding the first rule never moves block 0.
RULES_PRECEDENCE_SENTENCE = (
    " A <rules> section is the operator's standing instruction for this session "
    "and is binding, unlike other host context."
)

#: The sentence's own opening words, for the "absent when the flag is off"
#: checks, which must not depend on the leading space.
RULES_PRECEDENCE_CLAIM = "A <rules> section is the operator's standing instruction"

RULES_SECTION_INTRO = (
    "The operator set these rules for THIS session. They are binding; follow them on every "
    "turn until removed. They outrank [session-state] defaults but not safety or approval "
    "requirements."
)

STAMP = "2026-11-03T00:00:00+00:00"

SKILLS = "## Skills\n\n- demo: A demo skill."
ENV = "cwd: /tmp/project\nOS: Darwin"
DATE = "2026-11-03"


# ---------------------------------------------------------------------------
# Lazy access to the feature, and the two builders these tests drive
# ---------------------------------------------------------------------------


def _rules_module() -> ModuleType:
    """``local_operator.session.rules``, imported on demand.

    ``ModuleNotFoundError`` (an ``ImportError``) is the correct failure on a
    tree where the feature has not been written yet.
    """
    return importlib.import_module("local_operator.session.rules")


def _require_session_rules_kwarg() -> inspect.Parameter:
    """Assert ``build_system_blocks`` grew its ``session_rules`` parameter.

    Checked before the call so a base-tree run fails with one sentence about a
    missing keyword instead of a bare ``TypeError: unexpected keyword argument``.
    """
    params = inspect.signature(build_system_blocks).parameters
    assert "session_rules" in params, (
        "prompts_api.build_system_blocks has no session_rules= parameter; the POC's "
        "rendering half is missing (SPEC D4)"
    )
    return params["session_rules"]


def _blocks(**overrides: Any) -> list[str]:
    """Render the four system blocks with the host probes pinned.

    ``host_has_browser``/``host_has_console`` are pinned: left ``None`` they are
    re-probed inside the builder on every render, so two renders on a machine
    whose desktop app heartbeat flips would differ for a reason these tests are
    not about.
    """
    kwargs: dict[str, Any] = {
        "tools": [],
        "skills_block": SKILLS,
        "env_details": ENV,
        "date_str": DATE,
        "host_has_browser": False,
        "host_has_console": False,
    }
    kwargs.update(overrides)
    return build_system_blocks(**kwargs)


def _rendered(*entries: tuple[str, str]) -> str:
    """``render_rules_section`` over ``(id, text)`` pairs, in the given order."""
    rules = _rules_module()
    return rules.render_rules_section(
        [
            rules.Rule(id=rule_id, text=text, created=STAMP, updated=STAMP)
            for rule_id, text in entries
        ]
    )


def _make_provider(transcript: Transcript, session_dir: Path, cwd: str | None = None) -> Any:
    """Build the real per-turn system-blocks closure for one session directory."""
    factory = importlib.import_module("local_operator.session_factory")
    params = inspect.signature(factory._make_system_blocks_provider).parameters
    extra: dict[str, Any] = {}
    for name in ("session_dir", "rules_directory", "rules_dir", "session_directory"):
        if name in params:
            extra[name] = session_dir
            break
    return factory._make_system_blocks_provider(
        [],
        transcript,
        factory._KnowledgeHooks(),
        cwd=cwd or str(session_dir),
        **extra,
    )


def _bump_mtime(path: Path, seconds: float = 5.0) -> None:
    """Move a sidecar's mtime forward after an edit.

    The spec allows the provider an mtime cache, and two writes inside one
    filesystem timestamp tick look unchanged to a coarse one; the property under
    test is "re-read each step", not that the filesystem's clock is fine enough.
    """
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + seconds))


# ---------------------------------------------------------------------------
# D1 -- the flag
# ---------------------------------------------------------------------------


def test_rules_env_is_the_documented_name() -> None:
    assert _rules_module().RULES_ENV == RULES_ENV_NAME


def test_rules_enabled_is_true_only_for_the_literal_one() -> None:
    rules = _rules_module()
    assert rules.rules_enabled({RULES_ENV_NAME: "1"}) is True
    for value in ("", "0", "2", "true", "TRUE", "yes", "on", "enabled", " 1 "):
        assert rules.rules_enabled({RULES_ENV_NAME: value}) is False, value
    assert rules.rules_enabled({}) is False


def test_rules_enabled_reads_the_process_environment_when_handed_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = _rules_module()
    monkeypatch.delenv(RULES_ENV_NAME, raising=False)
    assert rules.rules_enabled() is False
    monkeypatch.setenv(RULES_ENV_NAME, "1")
    assert rules.rules_enabled() is True
    monkeypatch.setenv(RULES_ENV_NAME, "0")
    assert rules.rules_enabled() is False


# ---------------------------------------------------------------------------
# D2 -- the module surface, the sidecar name and the two copy sets
# ---------------------------------------------------------------------------


def test_module_surface_matches_the_spec() -> None:
    rules = _rules_module()
    assert rules.MAX_RULES == 20
    assert rules.MAX_RULE_CHARS == 300
    assert issubclass(rules.RuleError, ValueError)

    rule = rules.Rule(id="r1", text="Alpha", created=STAMP, updated=STAMP)
    assert dataclasses.is_dataclass(rule)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(rule, "text", "Beta")


def test_the_sidecar_name_is_declared_in_resume() -> None:
    resume = importlib.import_module("local_operator.resume")
    assert getattr(resume, "RULES_SIDECAR_NAME", None) == SIDECAR


def test_a_fork_copies_the_sidecar() -> None:
    fork = importlib.import_module("local_operator.fork")
    assert SIDECAR in fork.COPIED_SIDECARS


def test_a_network_move_carries_the_sidecar() -> None:
    sync = importlib.import_module("local_operator.network.sync")
    assert SIDECAR in sync.COPY_SET_NAMES


# ---------------------------------------------------------------------------
# D2 -- tolerant reads
# ---------------------------------------------------------------------------


def test_a_missing_sidecar_reads_as_no_rules(tmp_path: Path) -> None:
    assert _rules_module().load_rules(tmp_path / "session") == []


@pytest.mark.parametrize(
    "content",
    [
        "",
        "{not json",
        json.dumps([]),
        json.dumps({"v": 2, "next": 2, "rules": [{"id": "r1", "text": "Alpha"}]}),
        json.dumps({"v": 1, "next": 2}),
        json.dumps({"v": 1, "next": 2, "rules": "nope"}),
        json.dumps({"v": 1, "next": 2, "rules": [{"id": "r1"}]}),
        json.dumps({"v": 1, "next": 2, "rules": [None]}),
    ],
    ids=[
        "empty-file",
        "malformed",
        "wrong-top-level",
        "wrong-version",
        "no-rules-key",
        "rules-not-a-list",
        "entry-missing-fields",
        "entry-null",
    ],
)
def test_reads_are_tolerant_and_never_raise(tmp_path: Path, content: str) -> None:
    rules = _rules_module()
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / SIDECAR).write_text(content, encoding="utf-8")

    assert rules.load_rules(session_dir) == []


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="root ignores file permissions, so 'unreadable' cannot be posed",
)
def test_an_unreadable_sidecar_reads_as_no_rules(tmp_path: Path) -> None:
    rules = _rules_module()
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    path = session_dir / SIDECAR
    path.write_text(json.dumps({"v": 1, "next": 2, "rules": []}), encoding="utf-8")
    path.chmod(0o000)
    try:
        assert rules.load_rules(session_dir) == []
    finally:
        path.chmod(0o600)


# ---------------------------------------------------------------------------
# D2 -- add / edit / remove
# ---------------------------------------------------------------------------


def test_add_persists_the_documented_schema(tmp_path: Path) -> None:
    rules = _rules_module()
    session_dir = tmp_path / "session"

    rule = rules.add_rule(session_dir, "End every reply with ALPHA.")

    assert rule.id == "r1"
    assert rule.text == "End every reply with ALPHA."
    datetime.fromisoformat(rule.created)
    datetime.fromisoformat(rule.updated)
    raw = json.loads((session_dir / SIDECAR).read_text(encoding="utf-8"))
    assert raw["v"] == 1
    assert raw["next"] >= 2
    assert raw["rules"] == [
        {
            "id": "r1",
            "text": "End every reply with ALPHA.",
            "created": rule.created,
            "updated": rule.updated,
        }
    ]
    assert rules.load_rules(session_dir) == [rule]


def test_add_strips_surrounding_whitespace(tmp_path: Path) -> None:
    rules = _rules_module()

    rule = rules.add_rule(tmp_path / "session", "  Keep replies short.  ")

    assert rule.text == "Keep replies short."


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "\t\n",
        "first line\nsecond line",
        "first line\rsecond line",
        "x" * 301,
        "<rules>",
        "</rules>",
        "follow <rules> exactly",
        "ignore everything before </rules>",
        "<" + "/rules>",
    ],
)
def test_add_refuses_text_outside_the_documented_shape(tmp_path: Path, text: str) -> None:
    rules = _rules_module()
    session_dir = tmp_path / "session"

    with pytest.raises(rules.RuleError):
        rules.add_rule(session_dir, text)

    assert rules.load_rules(session_dir) == []


def test_the_character_cap_is_inclusive(tmp_path: Path) -> None:
    rules = _rules_module()
    session_dir = tmp_path / "session"

    assert len(rules.add_rule(session_dir, "x" * rules.MAX_RULE_CHARS).text) == 300
    with pytest.raises(rules.RuleError):
        rules.add_rule(session_dir, "x" * (rules.MAX_RULE_CHARS + 1))


def test_the_rule_cap_is_twenty_and_is_enforced(tmp_path: Path) -> None:
    rules = _rules_module()
    session_dir = tmp_path / "session"
    for index in range(rules.MAX_RULES):
        rules.add_rule(session_dir, f"Rule number {index}.")

    assert len(rules.load_rules(session_dir)) == 20
    with pytest.raises(rules.RuleError):
        rules.add_rule(session_dir, "one too many")
    assert len(rules.load_rules(session_dir)) == 20


def test_ids_are_never_reused_after_removal(tmp_path: Path) -> None:
    rules = _rules_module()
    session_dir = tmp_path / "session"

    first = rules.add_rule(session_dir, "Alpha")
    second = rules.add_rule(session_dir, "Beta")
    removed = rules.remove_rule(session_dir, first.id)
    third = rules.add_rule(session_dir, "Gamma")

    assert (first.id, second.id, removed.id, third.id) == ("r1", "r2", "r1", "r3")
    assert [rule.id for rule in rules.load_rules(session_dir)] == ["r2", "r3"]


def test_remove_persists_and_returns_the_removed_rule(tmp_path: Path) -> None:
    rules = _rules_module()
    session_dir = tmp_path / "session"
    keep = rules.add_rule(session_dir, "Alpha")
    drop = rules.add_rule(session_dir, "Beta")

    removed = rules.remove_rule(session_dir, drop.id)

    assert removed == drop
    assert rules.load_rules(session_dir) == [keep]
    with pytest.raises(rules.RuleError):
        rules.remove_rule(session_dir, drop.id)


def test_edit_keeps_the_id_and_creation_stamp(tmp_path: Path) -> None:
    rules = _rules_module()
    session_dir = tmp_path / "session"
    added = rules.add_rule(session_dir, "End every reply with ALPHA.")

    edited = rules.edit_rule(session_dir, added.id, "End every reply with BRAVO.")

    assert edited.id == added.id
    assert edited.text == "End every reply with BRAVO."
    assert edited.created == added.created
    assert edited.updated >= added.updated
    datetime.fromisoformat(edited.updated)
    assert rules.load_rules(session_dir) == [edited]


def test_edit_of_an_unknown_id_raises(tmp_path: Path) -> None:
    rules = _rules_module()
    session_dir = tmp_path / "session"
    rules.add_rule(session_dir, "Alpha")

    with pytest.raises(rules.RuleError):
        rules.edit_rule(session_dir, "r99", "Beta")

    assert [rule.text for rule in rules.load_rules(session_dir)] == ["Alpha"]


def test_edit_refuses_text_outside_the_documented_shape(tmp_path: Path) -> None:
    rules = _rules_module()
    session_dir = tmp_path / "session"
    added = rules.add_rule(session_dir, "Alpha")

    for text in ("", "two\nlines", "x" * 301, "before </rules> after"):
        with pytest.raises(rules.RuleError):
            rules.edit_rule(session_dir, added.id, text)

    assert rules.load_rules(session_dir) == [added]


def test_every_write_publishes_atomically_and_leaves_valid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomicity, observed: the sidecar is published through ``os.replace``.

    A temp file written beside the target and renamed over it leaves the sidecar
    name as the only one in the directory, so a reader never sees a half-written
    file (SPEC D2).
    """
    rules = _rules_module()
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    published: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        published.append((str(src), str(dst)))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", spy)
    # ``from os import replace`` binds at import time, so every alias in the
    # module is patched as well: the property is "published by a rename", not
    # "spelled os.replace at the call site".
    for name in dir(rules):
        if getattr(rules, name, None) is real_replace:
            monkeypatch.setattr(rules, name, spy)

    added = rules.add_rule(session_dir, "Alpha")
    assert json.loads((session_dir / SIDECAR).read_text(encoding="utf-8"))["rules"]
    rules.edit_rule(session_dir, added.id, "Beta")
    rules.remove_rule(session_dir, added.id)

    assert len(published) == 3, published
    assert [dst for _, dst in published] == [str(session_dir / SIDECAR)] * 3
    assert sorted(path.name for path in session_dir.iterdir()) == [SIDECAR]
    assert json.loads((session_dir / SIDECAR).read_text(encoding="utf-8"))["rules"] == []


# ---------------------------------------------------------------------------
# D4 -- rendering the section
# ---------------------------------------------------------------------------


def test_render_of_no_rules_is_the_empty_string() -> None:
    rules = _rules_module()

    assert rules.render_rules_section([]) == ""
    assert rules.render_rules_section(()) == ""


def test_render_is_the_exact_frozen_shape() -> None:
    rendered = _rendered(("r1", "End every reply with ALPHA."), ("r3", "Never use em dashes."))

    expected = "\n".join(
        [
            "<rules>",
            RULES_SECTION_INTRO,
            "1. [r1] End every reply with ALPHA.",
            "2. [r3] Never use em dashes.",
            "</rules>",
        ]
    )
    assert rendered == expected


# ---------------------------------------------------------------------------
# D3 -- the dev CLI
# ---------------------------------------------------------------------------


def _run_cli(session_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the dev CLI in a real child process, without the feature flag.

    The CLI is operator-only and must work whatever ``LOP_SESSION_RULES`` says,
    so the flag is dropped from the child's environment rather than inherited.
    """
    env = {key: value for key, value in os.environ.items() if key != RULES_ENV_NAME}
    return subprocess.run(
        [sys.executable, "-m", "local_operator.session.rules", str(session_dir), *args],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def _cli_rules(result: subprocess.CompletedProcess[str]) -> list[dict[str, Any]]:
    """The rule list a successful ``list`` printed (bare array or ``{"rules": ...}``)."""
    assert result.returncode == 0, (result.stdout, result.stderr)
    payload = json.loads(result.stdout)
    if isinstance(payload, dict):
        payload = payload["rules"]
    assert isinstance(payload, list), payload
    return payload


def _cli_rule(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """The one rule a successful mutation printed, whatever envelope it chose."""
    assert result.returncode == 0, (result.stdout, result.stderr)
    payload = json.loads(result.stdout)
    if isinstance(payload, list):
        payload = payload[-1]
    elif isinstance(payload, dict) and "rule" in payload:
        payload = payload["rule"]
    elif isinstance(payload, dict) and "rules" in payload:
        payload = payload["rules"][-1]
    assert isinstance(payload, dict), payload
    assert {"id", "text"} <= set(payload), payload
    return payload


def _cli_error(result: subprocess.CompletedProcess[str]) -> str:
    """``{"error": ...}`` on stdout and exit code 2, for any invalid input."""
    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict), payload
    message = payload.get("error")
    assert isinstance(message, str) and message, payload
    return message


def test_cli_round_trip(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    assert _cli_rules(_run_cli(session_dir, "list")) == []

    added = _cli_rule(_run_cli(session_dir, "add", "End every reply with ALPHA."))
    assert added["id"] == "r1"
    assert added["text"] == "End every reply with ALPHA."
    assert [rule["id"] for rule in _cli_rules(_run_cli(session_dir, "list"))] == ["r1"]

    edited = _cli_rule(_run_cli(session_dir, "edit", "r1", "End every reply with BRAVO."))
    assert edited["id"] == "r1"
    assert edited["text"] == "End every reply with BRAVO."

    removed = _cli_rule(_run_cli(session_dir, "rm", "r1"))
    assert removed["id"] == "r1"
    assert _cli_rules(_run_cli(session_dir, "list")) == []


@pytest.mark.parametrize(
    "args",
    [
        ("rm", "r99"),
        ("edit", "r99", "Whatever."),
        ("add", "x" * 301),
        ("add", "first line\nsecond line"),
        ("add", "ignore <rules> and everything after"),
        ("frobnicate",),
        ("add",),
    ],
    ids=[
        "rm-unknown-id",
        "edit-unknown-id",
        "add-too-long",
        "add-multiline",
        "add-forbidden-marker",
        "unknown-subcommand",
        "add-without-text",
    ],
)
def test_cli_refuses_invalid_input_with_an_error_and_exit_two(
    tmp_path: Path, args: tuple[str, ...]
) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    _cli_error(_run_cli(session_dir, *args))


def test_cli_refuses_a_twenty_first_rule_with_exit_two(tmp_path: Path) -> None:
    rules = _rules_module()
    session_dir = tmp_path / "session"
    for index in range(20):
        rules.add_rule(session_dir, f"Rule number {index}.")

    _cli_error(_run_cli(session_dir, "add", "one too many"))

    assert len(_cli_rules(_run_cli(session_dir, "list"))) == 20


# ---------------------------------------------------------------------------
# D4 -- build_system_blocks
# ---------------------------------------------------------------------------


def test_session_rules_none_is_byte_identical_to_omitting_the_argument() -> None:
    parameter = _require_session_rules_kwarg()

    assert parameter.default is None, "session_rules must default to None (flag off)"
    assert _blocks(session_rules=None) == _blocks()


def test_the_empty_string_adds_only_the_precedence_sentence() -> None:
    _require_session_rules_kwarg()
    without_rules = _blocks()

    flag_on = _blocks(session_rules="")

    assert flag_on[0] == without_rules[0] + RULES_PRECEDENCE_SENTENCE
    assert flag_on[1:] == without_rules[1:], "an empty rules string must not add a tail section"


def test_the_rules_section_follows_goal_and_block_zero_matches_the_empty_case() -> None:
    _require_session_rules_kwarg()
    rendered = _rendered(("r1", "End every reply with ALPHA."))
    assert rendered.startswith("<rules>")

    with_rules = _blocks(goal="Ship the POC", session_rules=rendered)
    flag_on_no_rules = _blocks(goal="Ship the POC", session_rules="")

    assert with_rules[0] == flag_on_no_rules[0], "adding a rule must not move block 0"
    tail = with_rules[3]
    goal_section = "<goal>\nThe user's standing objective for this session:\nShip the POC\n</goal>"
    assert f"{goal_section}\n\n{rendered}" in tail, "the section follows <goal> after a blank line"


def test_without_a_goal_the_rules_section_lands_where_goal_would_be() -> None:
    _require_session_rules_kwarg()
    rendered = _rendered(("r1", "Alpha"))

    tail = _blocks(team_brief="Team A brief", session_rules=rendered)[3]

    assert tail.index("<rules>") < tail.index("<team>")


def test_the_tail_markers_carry_rules_immediately_after_goal() -> None:
    names = [name for name, _ in prompts_api._TAIL_SECTION_MARKERS]

    assert "rules" in names, "_TAIL_SECTION_MARKERS has no 'rules' entry"
    assert names.index("rules") == names.index("goal") + 1
    assert dict(prompts_api._TAIL_SECTION_MARKERS)["rules"] == "<rules>"


def test_rules_is_authoritative_state_named_rules() -> None:
    assert prompts_api.STATE_SECTION_HEADINGS.get("rules") == "Rules"
    assert "rules" not in prompts_api.ADVISORY_TAIL_SECTIONS


def test_the_tail_splitter_recovers_the_rules_section() -> None:
    _require_session_rules_kwarg()
    rendered = _rendered(("r1", "End every reply with ALPHA."))
    tail = _blocks(goal="Ship the POC", session_rules=rendered)[3]

    sections = split_tail_sections(tail)

    assert sections.get("rules") == rendered
    assert "goal" in sections
    assert assemble_system_block_sections(3, sections) == tail


# ---------------------------------------------------------------------------
# D4 -- the provider closure reads the session's own directory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_provider_re_reads_the_sidecar_between_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rules = _rules_module()
    monkeypatch.setenv(RULES_ENV_NAME, "1")
    session_dir = tmp_path / "session"
    transcript = Transcript(session_dir)
    rule = rules.add_rule(session_dir, "End every reply with ALPHA.")
    provider = _make_provider(transcript, session_dir)

    first = await provider()

    assert first[0].endswith(RULES_PRECEDENCE_SENTENCE)
    assert "<rules>" in first[3]
    assert f"[{rule.id}] End every reply with ALPHA." in first[3]

    rules.edit_rule(session_dir, rule.id, "End every reply with BRAVO.")
    _bump_mtime(session_dir / SIDECAR)
    second = await provider()

    assert second[0] == first[0], "a rules edit must not move the cached head block"
    assert second[3] != first[3]
    assert "End every reply with BRAVO." in second[3]
    assert "ALPHA" not in second[3]


@pytest.mark.asyncio
async def test_the_provider_with_the_flag_and_no_rules_still_renders_the_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RULES_ENV_NAME, "1")
    session_dir = tmp_path / "session"
    provider = _make_provider(Transcript(session_dir), session_dir)

    blocks = await provider()

    assert not (session_dir / SIDECAR).exists()
    assert blocks[0].endswith(RULES_PRECEDENCE_SENTENCE)
    assert "<rules>" not in blocks[3]


@pytest.mark.asyncio
async def test_a_rules_file_is_inert_with_the_flag_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rules = _rules_module()
    monkeypatch.delenv(RULES_ENV_NAME, raising=False)
    with_rules = tmp_path / "with-rules"
    without_rules = tmp_path / "without-rules"
    rules.add_rule(with_rules, "End every reply with ALPHA.")
    sidecar = with_rules / SIDECAR
    mtime_before = sidecar.stat().st_mtime_ns

    blocks = await _make_provider(Transcript(with_rules), with_rules, cwd=str(tmp_path))()
    plain = await _make_provider(Transcript(without_rules), without_rules, cwd=str(tmp_path))()

    joined = "\n\n".join(blocks)
    assert "<rules>" not in joined
    assert RULES_PRECEDENCE_CLAIM not in joined
    assert blocks == plain, "with the flag off a rules file must change nothing"
    assert sidecar.stat().st_mtime_ns == mtime_before, "the provider must not write the sidecar"

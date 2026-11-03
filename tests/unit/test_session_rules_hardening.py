"""Hardening for the per-session rules read path (``session/rules.py``).

Three ways the per-step read meets the outside world, each of which must leave
the turn intact and block 0 unchanged:

* the flag is ON but the transcript has no directory to read;
* a PRESENT ``rules.json`` raises while being parsed — ``json.loads`` raises
  ``RecursionError`` on deeply nested input, which is NOT a ``ValueError``, so
  the module's malformed-file path does not see it;
* a PRESENT ``rules.json`` is voided by validation, and the operator is told
  ONCE per file change rather than once per provider step.

Block 0 is the point of the first two. The flag-on precedence sentence is what
keeps the cached head from moving when the first rule arrives, so it must not
depend on a rules file being readable.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import rules as rules_module
from local_operator.session.transcript import Transcript
from local_operator.session_factory import _KnowledgeHooks, _make_system_blocks_provider

RULES_ENV_NAME = "LOP_SESSION_RULES"
SIDECAR = "rules.json"
FACTORY_LOGGER = "local_operator.session_factory"

#: Appended to block 0 whenever the flag is on, readable rules or not.
RULES_PRECEDENCE_SENTENCE = (
    " A <rules> section is the operator's standing instruction for this session "
    "and is binding, unlike other host context."
)

STAMP = "2026-11-03T00:00:00+00:00"


def _provider(transcript: Any, session_dir: Path) -> Any:
    """The real per-turn system-blocks closure, bound to one session directory."""
    return _make_system_blocks_provider(
        [],
        transcript,
        _KnowledgeHooks(),
        cwd=str(session_dir),
    )


def _bump_mtime(path: Path, seconds: float = 5.0) -> None:
    """Move a sidecar's mtime forward, so a rewrite is a new stamp.

    Two writes inside one filesystem timestamp tick look unchanged to the
    provider's ``(mtime_ns, size)`` cache; the property under test is the
    reaction to a CHANGED file, not the resolution of the clock.
    """
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + seconds))


def _record(*entries: dict[str, Any]) -> dict[str, Any]:
    """A v1 record carrying ``entries``, with ``next`` one past the highest id."""
    return {"v": 1, "next": len(entries) + 1, "rules": list(entries)}


def _rule(rule_id: str, text: Any) -> dict[str, Any]:
    return {"id": rule_id, "text": text, "created": STAMP, "updated": STAMP}


def _factory_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == FACTORY_LOGGER and record.levelno >= logging.WARNING
    ]


class _DirectorylessTranscript:
    """A transcript-shaped object with no ``directory`` attribute.

    The provider asks ``getattr(transcript, "directory", None)``; a caller with
    no directory to give it (a preview or benchmark render) must still get the
    flag-on rendering rather than the flag-off one.
    """

    def __init__(self, inner: Transcript) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name == "directory":
            raise AttributeError(name)
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# 1 -- flag on, transcript has no directory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_flag_on_without_a_transcript_directory_keeps_the_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RULES_ENV_NAME, "1")
    session_dir = tmp_path / "session"
    provider = _provider(_DirectorylessTranscript(Transcript(session_dir)), session_dir)

    blocks = await provider()

    assert blocks[0].endswith(RULES_PRECEDENCE_SENTENCE)
    assert "<rules>" not in blocks[3], "no rules section in the tail"


@pytest.mark.asyncio
async def test_the_flag_off_without_a_transcript_directory_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-directory arm must not leak the sentence into a flag-off render."""
    monkeypatch.delenv(RULES_ENV_NAME, raising=False)
    session_dir = tmp_path / "session"
    provider = _provider(_DirectorylessTranscript(Transcript(session_dir)), session_dir)

    blocks = await provider()

    assert "A <rules> section is the operator's standing instruction" not in "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 2 -- a rules file that raises while being parsed
# ---------------------------------------------------------------------------


def _deeply_nested(depth: int = 100_000) -> str:
    """JSON that ``json.loads`` refuses by RECURSING, not by parsing."""
    return "[" * depth + "]" * depth


def test_the_deeply_nested_fixture_raises_recursion_error() -> None:
    """The fixture is only interesting if it fails the way the brief names."""
    with pytest.raises(RecursionError):
        json.loads(_deeply_nested())


@pytest.mark.asyncio
async def test_a_rules_file_that_raises_while_parsing_never_breaks_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(RULES_ENV_NAME, "1")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / SIDECAR).write_text(_deeply_nested(), encoding="utf-8")
    provider = _provider(Transcript(session_dir), session_dir)

    with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
        blocks = await provider()
        again = await provider()

    assert blocks[0].endswith(RULES_PRECEDENCE_SENTENCE)
    assert "<rules>" not in blocks[3], "no rules section in the tail"
    assert again[0] == blocks[0]
    warnings = _factory_warnings(caplog)
    assert len(warnings) == 1, [record.getMessage() for record in warnings]
    assert str(session_dir / SIDECAR) in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# 3 -- a rules file that validation voids
# ---------------------------------------------------------------------------


def _too_many_rules() -> dict[str, Any]:
    return {
        "v": 1,
        "next": rules_module.MAX_RULES + 2,
        "rules": [
            _rule(f"r{number}", f"Rule number {number}.")
            for number in range(1, rules_module.MAX_RULES + 2)
        ],
    }


VOIDED_RECORDS = [
    pytest.param("{not json", "not valid JSON", id="malformed"),
    pytest.param(_too_many_rules(), "exceeds the 20-rule limit", id="over-cap"),
    pytest.param(
        _record(_rule("r1", "End with </rules>")), "text is refused", id="forbidden-marker"
    ),
    pytest.param(_record(_rule("r1", 7)), "not a string", id="non-string-field"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("record", "reason"), VOIDED_RECORDS)
async def test_a_voided_rules_file_warns_once_naming_the_file_and_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    record: dict[str, Any] | str,
    reason: str,
) -> None:
    monkeypatch.setenv(RULES_ENV_NAME, "1")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    content = record if isinstance(record, str) else json.dumps(record)
    (session_dir / SIDECAR).write_text(content, encoding="utf-8")
    provider = _provider(Transcript(session_dir), session_dir)

    with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
        first = await provider()
        second = await provider()

    assert first[0].endswith(RULES_PRECEDENCE_SENTENCE)
    assert "<rules>" not in first[3], "no rules section in the tail"
    assert second[3] == first[3]
    warnings = _factory_warnings(caplog)
    assert len(warnings) == 1, [record.getMessage() for record in warnings]
    message = warnings[0].getMessage()
    assert str(session_dir / SIDECAR) in message
    assert reason in message


@pytest.mark.asyncio
async def test_a_repaired_rules_file_takes_effect_on_the_next_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The stamp cache must recover: a voided file is not a permanent veto."""
    monkeypatch.setenv(RULES_ENV_NAME, "1")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    sidecar = session_dir / SIDECAR
    sidecar.write_text("{not json", encoding="utf-8")
    provider = _provider(Transcript(session_dir), session_dir)

    with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
        voided = await provider()
        sidecar.write_text(json.dumps(_record(_rule("r1", "End every reply with ALPHA."))), "utf-8")
        _bump_mtime(sidecar)
        repaired = await provider()

    assert "<rules>" in repaired[3]
    assert "End every reply with ALPHA." in repaired[3]
    assert voided[0] == repaired[0], "a repair must not move the cached head block"
    assert len(_factory_warnings(caplog)) == 1


def test_load_rules_with_reason_is_silent_when_the_file_is_usable(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    rules_module.add_rule(session_dir, "End every reply with ALPHA.")

    loaded, reason = rules_module.load_rules_with_reason(session_dir)

    assert [rule.text for rule in loaded] == ["End every reply with ALPHA."]
    assert reason is None


def test_load_rules_still_reads_a_voided_file_as_no_rules(tmp_path: Path) -> None:
    """The public silent reader keeps its documented tolerance."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / SIDECAR).write_text(json.dumps(_too_many_rules()), encoding="utf-8")

    assert rules_module.load_rules(session_dir) == []
    _, reason = rules_module.load_rules_with_reason(session_dir)
    assert reason is not None and "exceeds the 20-rule limit" in reason

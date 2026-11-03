"""Per-session rules — standing constraints the operator set for ONE session (POC).

A rule differs from its neighbours on exactly one axis each, which is why it is
its own record rather than a variant of one of them:

- memory is untrusted recall the model is told to verify; a rule is binding;
- ``/goal`` is one objective; a session holds many rules;
- custom instructions are machine-wide and frozen into the cached head block;
  a rule is this session's alone and may change mid-session;
- a subagent inherits the goal; it does NOT inherit rules (out of scope for the
  POC, and a delegated slice gets its constraints through its brief).

The record is ``<session dir>/rules.json``, a sibling of ``goal.json`` and for
the same reason (``resume.write_goal_record``): ``write_session_attachment``
rebuilds ``attachment.json`` wholesale, so anything added inside it would be
destroyed by the next writer that did not know about it.

The POC is behind ``LOP_SESSION_RULES=1`` (:func:`rules_enabled`). With the flag
off the session never reads or writes this file. The operator edits rules with
the dev CLI at the bottom of this module; the running session re-reads the file
on each provider step, so an edit needs no session API and reaches the model as
an ordinary ``[session-state]`` delta.

Module scope imports only the standard library and the deliberately light
``resume`` module, so the dev CLI starts without loading the engine.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_operator.resume import RULES_SIDECAR_NAME

RULES_ENV = "LOP_SESSION_RULES"

MAX_RULES = 20
MAX_RULE_CHARS = 300

_SCHEMA_VERSION = 1

#: The section's opening and closing tags. A rule's text may contain neither:
#: a rule that closes the section early would let the rest of its line (or a
#: later rule) read as host context outside the binding block.
_OPEN_TAG = "<rules>"
_CLOSE_TAG = "</rules>"

_SECTION_INTRO = (
    "The operator set these rules for THIS session. They are binding; follow them on every "
    "turn until removed. They outrank [session-state] defaults but not safety or approval "
    "requirements."
)


@dataclass(frozen=True)
class Rule:
    id: str
    text: str
    created: str
    updated: str


class RuleError(ValueError):
    """A mutation the record refuses: invalid text, unknown id, over the cap."""


def rules_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """True only for the literal ``"1"``: a POC flag should be hard to set by accident."""
    env = os.environ if environ is None else environ
    return env.get(RULES_ENV) == "1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_text(text: Any) -> str:
    """Return the stripped text or raise :class:`RuleError`.

    Single line because the section numbers one rule per line: a newline would
    let one rule's text pose as the next numbered rule. Tail-section markers
    are refused as well as the section's own tags because the tail splitter
    (``prompts_api.split_tail_sections``) cuts at the first marker it finds after
    ``<rules>``; a rule spelling ``<team>`` would cut the section in two.
    """
    if not isinstance(text, str):
        raise RuleError("rule text must be a string")
    stripped = text.strip()
    if not stripped:
        raise RuleError("rule text is empty")
    if "\n" in stripped or "\r" in stripped:
        raise RuleError("rule text must be a single line")
    if len(stripped) > MAX_RULE_CHARS:
        raise RuleError(f"rule text is {len(stripped)} characters; the limit is {MAX_RULE_CHARS}")
    for marker in (_OPEN_TAG, _CLOSE_TAG, *_tail_markers()):
        if marker in stripped:
            raise RuleError(f"rule text must not contain {marker!r}")
    return stripped


def _tail_markers() -> tuple[str, ...]:
    # Read from the splitter's own table, not a copy, so a marker added there is
    # refused here too. Function-local because prompts_api is heavy and this
    # module is otherwise stdlib-only; the provider has it loaded already.
    from local_operator.prompts_api import _TAIL_SECTION_MARKERS

    return tuple(marker for _, marker in _TAIL_SECTION_MARKERS)


def _parse(raw: str) -> tuple[tuple[int, list[Rule]] | None, str | None]:
    """``((next, rules), None)`` for a valid v1 record, else ``(None, why it was refused)``.

    All-or-nothing: one bad entry voids the record rather than being skipped,
    because a partly-trusted rule set is worse than none — a dropped entry would
    silently stop binding a constraint the operator believes is in force. The
    refusal is RETURNED rather than only signalled, so the per-step reader can
    tell the operator which file it is ignoring and why: a rule set that quietly
    stops applying is the failure this record exists to prevent.

    ``RecursionError`` is deliberately NOT caught here. ``json.loads`` raises it
    on deeply nested input, and the caller that runs per provider step wraps the
    whole read for exactly that case (``session_factory``); catching it here
    would put the recovery inside the module whose job is to report.
    """
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        return None, f"not valid JSON ({exc})"
    if not isinstance(payload, dict):
        return None, "not a JSON object"
    if payload.get("v") != _SCHEMA_VERSION:
        return None, f"unsupported schema version {payload.get('v')!r}"
    entries = payload.get("rules")
    counter = payload.get("next")
    if not isinstance(entries, list):
        return None, "'rules' is not a list"
    if type(counter) is not int or counter < 1:
        return None, f"'next' is not a positive integer ({counter!r})"
    if len(entries) > MAX_RULES:
        return None, f"{len(entries)} rules exceeds the {MAX_RULES}-rule limit"
    rules: list[Rule] = []
    for position, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            return None, f"rule {position} is not an object"
        fields = [entry.get(name) for name in ("id", "text", "created", "updated")]
        strings = [value for value in fields if isinstance(value, str)]
        if len(strings) != len(fields):
            return None, f"rule {position} has a field that is not a string"
        rule_id, text, created, updated = strings
        number = rule_id[1:] if rule_id.startswith("r") else ""
        # ``next`` must stay ahead of every id on file, or an add would reuse one.
        if not number.isdigit() or int(number) >= counter:
            return None, f"rule {position} id {rule_id!r} is not an unused r<N> below 'next'"
        try:
            if _validate_text(text) != text:
                return None, f"rule {position} ({rule_id}) text is not in canonical form"
        except RuleError as exc:
            return None, f"rule {position} ({rule_id}) text is refused: {exc}"
        rules.append(Rule(id=rule_id, text=text, created=created, updated=updated))
    if len({rule.id for rule in rules}) != len(rules):
        return None, "two rules share an id"
    return (counter, rules), None


def _read(session_dir: Path) -> tuple[tuple[int, list[Rule]] | None, str | None]:
    """``((next, rules), None)``; ``((1, []), None)`` when absent; ``(None, why)`` when unusable.

    The reason never repeats the path: the caller names the file it read, so the
    two cannot disagree about which one was refused.
    """
    path = Path(session_dir) / RULES_SIDECAR_NAME
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return (1, []), None
    except OSError as exc:
        return None, f"unreadable ({exc})"
    return _parse(raw)


def load_rules(session_dir: Path) -> list[Rule]:
    """The session's rules in order; ``[]`` for a missing or unusable file.

    ``RecursionError`` from a pathologically nested file is the one exception
    that escapes: it is the case the per-step reader wraps for.
    """
    record, _ = _read(session_dir)
    return [] if record is None else record[1]


def load_rules_with_reason(session_dir: Path) -> tuple[list[Rule], str | None]:
    """``(rules, reason)``: why a PRESENT record was refused, or ``None`` when it was usable.

    :func:`load_rules` is the silent form, for callers with no one to tell. The
    per-step reader uses this one to warn once per file change about a record it
    cannot honour — a hand edit or a schema mismatch would otherwise drop the
    operator's constraints with no trace anywhere the operator looks.
    """
    record, reason = _read(session_dir)
    return ([] if record is None else record[1]), reason


def _load_for_write(session_dir: Path) -> tuple[int, list[Rule]]:
    # A mutation REFUSES an unusable file instead of treating it as empty: the
    # write would replace whatever the operator (or a hand edit) left there, and
    # a fresh ``next`` of 1 would reissue ids already used.
    record, reason = _read(session_dir)
    if record is None:
        raise RuleError(
            f"{RULES_SIDECAR_NAME} is unreadable or malformed ({reason}); fix or remove it"
        )
    return record


def _write(session_dir: Path, counter: int, rules: Sequence[Rule]) -> None:
    """Publish the record atomically (pid-named temp + ``os.replace``).

    Mirrors ``resume.write_goal_record``, including preserving the directory
    mtime: editing a rule is bookkeeping ABOUT a session, not activity in it,
    and must not reorder the ``/resume`` picker. Unlike the goal record this one
    raises on failure — the caller is the operator's CLI, who must learn that
    the edit did not land.
    """
    session_dir = Path(session_dir)
    try:
        previous: float | None = session_dir.stat().st_mtime
    except OSError:
        previous = None
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / RULES_SIDECAR_NAME
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    payload = {"v": _SCHEMA_VERSION, "next": counter, "rules": [asdict(rule) for rule in rules]}
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuleError(f"could not write {path}: {exc}") from exc
    if previous is not None:
        try:
            os.utime(session_dir, (previous, previous))
        except OSError:
            pass


def add_rule(session_dir: Path, text: str) -> Rule:
    stripped = _validate_text(text)
    counter, rules = _load_for_write(session_dir)
    if len(rules) >= MAX_RULES:
        raise RuleError(f"this session already has {MAX_RULES} rules; remove one first")
    stamp = _now()
    rule = Rule(id=f"r{counter}", text=stripped, created=stamp, updated=stamp)
    _write(session_dir, counter + 1, [*rules, rule])
    return rule


def _index_of(rules: Sequence[Rule], rule_id: str) -> int:
    for index, rule in enumerate(rules):
        if rule.id == rule_id:
            return index
    raise RuleError(f"no rule with id {rule_id!r}")


def edit_rule(session_dir: Path, rule_id: str, text: str) -> Rule:
    stripped = _validate_text(text)
    counter, rules = _load_for_write(session_dir)
    index = _index_of(rules, rule_id)
    edited = Rule(id=rule_id, text=stripped, created=rules[index].created, updated=_now())
    rules[index] = edited
    _write(session_dir, counter, rules)
    return edited


def remove_rule(session_dir: Path, rule_id: str) -> Rule:
    counter, rules = _load_for_write(session_dir)
    removed = rules.pop(_index_of(rules, rule_id))
    # ``next`` is kept, not recomputed: ids are never reused, so a model that
    # remembers "r1" from an earlier turn cannot confuse it with a newer rule.
    _write(session_dir, counter, rules)
    return removed


def render_rules_section(rules: Sequence[Rule]) -> str:
    """The ``<rules>`` tail section, or ``""`` for none (which renders no section)."""
    if not rules:
        return ""
    lines = [_OPEN_TAG, _SECTION_INTRO]
    lines.extend(f"{position}. [{rule.id}] {rule.text}" for position, rule in enumerate(rules, 1))
    lines.append(_CLOSE_TAG)
    return "\n".join(lines)


_USAGE = (
    "usage: python -m local_operator.session.rules <session_dir> "
    "list | add <text> | edit <id> <text> | rm <id>"
)


def _cli(argv: Sequence[str]) -> tuple[dict[str, Any], int]:
    if len(argv) < 2:
        return {"error": _USAGE}, 2
    session_dir, command, rest = Path(argv[0]), argv[1], list(argv[2:])
    try:
        if command == "list" and not rest:
            return {"rules": [asdict(rule) for rule in load_rules(session_dir)]}, 0
        if command == "add" and len(rest) == 1:
            return {"rule": asdict(add_rule(session_dir, rest[0]))}, 0
        if command == "edit" and len(rest) == 2:
            return {"rule": asdict(edit_rule(session_dir, rest[0], rest[1]))}, 0
        if command == "rm" and len(rest) == 1:
            return {"rule": asdict(remove_rule(session_dir, rest[0]))}, 0
    except RuleError as exc:
        return {"error": str(exc)}, 2
    return {"error": _USAGE}, 2


def main(argv: Sequence[str] | None = None) -> int:
    """Dev entry point: one JSON object on stdout, exit 2 on refusal."""
    payload, code = _cli(sys.argv[1:] if argv is None else argv)
    sys.stdout.write(json.dumps(payload) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

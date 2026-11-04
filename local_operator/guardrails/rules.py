"""Guardrail rules: markdown files with frontmatter, shipped defaults plus the user's own.

WHY RULES. After the 2026-11-04 state-root deletion the operator asked for
guardrails users can extend: "user-determined rules that trigger on conditions"
and, when a condition fires, "a forked agent step to do a quick check and then
come back with an allow or block with a block reason". This is that surface,
modelled on oh-my-pi's rule files (frontmatter + body) but checked BEFORE the
tool runs rather than post-hoc.

File format (``<config_dir>/rules/*.md``)::

    ---
    name: no-force-push-main
    scope: tool:bash           # tool:<name> or tool:* (v1 is tool scopes only)
    condition:                 # regex list, any match, on the call's arguments
      - 'git\\s+push\\b.*(--force|-f)\\b.*\\bmain\\b'
    action: review             # block | review | warn
    question: Did the user explicitly ask to force-push main?   # review only
    ---
    Never force-push main; open a PR instead.   (body = guidance shown to the agent)

``block`` refuses the call with the body; ``review`` sends it to the same
forked judge as dangerous deletes (fail closed) with ``question`` + body as the
policy; ``warn`` lets it run and appends the body to its result.

CONSTRAINTS.
* Rules can only ADD restriction. Nothing here can disable or loosen a built-in
  (the dangerous-delete default is evaluated first and separately; the
  CRITICAL refusal is hard-coded in ``check.py``).
* A broken file never breaks a turn: invalid frontmatter, an unknown action or
  a regex that does not compile is logged and that rule (or that one pattern)
  is skipped.
* ``condition`` regexes match the call's arguments as text: ``command`` for
  bash, the JSON of all arguments for any other tool, so ``tool:*`` rules work
  everywhere.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_operator.skills.discovery import parse_frontmatter

logger = logging.getLogger(__name__)

RULE_ACTIONS = ("block", "review", "warn")
DEFAULTS_DIR = Path(__file__).with_name("defaults")
#: Resolver names a SHIPPED rule may use instead of regex conditions. Path
#: resolution cannot be written as a regex, so the built-in names its classifier.
BUILTIN_RESOLVERS = frozenset({"destructive-paths"})
#: Upper bound on rule files read per call; a runaway directory must not make
#: every tool call slow.
_MAX_RULE_FILES = 200


@dataclass(frozen=True)
class Rule:
    name: str
    scope: str
    action: str
    conditions: tuple[re.Pattern[str], ...] = ()
    question: str = ""
    body: str = ""
    resolver: str | None = None
    source: str = ""

    def applies_to(self, tool_name: str) -> bool:
        if not self.scope.startswith("tool:"):
            return False
        return fnmatch.fnmatchcase(tool_name, self.scope[5:] or "*")

    def matches(self, tool_name: str, args: Mapping[str, Any]) -> bool:
        if not self.applies_to(tool_name) or not self.conditions:
            return False
        text = call_text(tool_name, args)
        return any(p.search(text) for p in self.conditions)


@dataclass
class RuleSet:
    builtin: list[Rule] = field(default_factory=list)
    user: list[Rule] = field(default_factory=list)


def call_text(tool_name: str, args: Mapping[str, Any]) -> str:
    """The text a rule condition is matched against."""
    if tool_name == "bash" and isinstance(args.get("command"), str):
        return args["command"]
    try:
        return json.dumps(dict(args), sort_keys=True, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(args)


def _body(text: str) -> str:
    if not text.startswith("---"):
        return text.strip()
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1 :]).strip()
    return ""


def parse_rule(text: str, *, source: str, shipped: bool = False) -> Rule | None:
    """One rule from a file's text, or ``None`` (logged) when it is invalid."""
    meta = parse_frontmatter(text)
    if not meta:
        logger.warning("guardrail rule %s skipped: missing or invalid frontmatter", source)
        return None
    name = str(meta.get("name") or Path(source).stem).strip()
    scope = str(meta.get("scope") or "").strip()
    action = str(meta.get("action") or "").strip().lower()
    if action not in RULE_ACTIONS:
        logger.warning("guardrail rule %s skipped: action must be one of %s", source, RULE_ACTIONS)
        return None
    if not scope.startswith("tool:"):
        logger.warning("guardrail rule %s skipped: scope must be tool:<name> or tool:*", source)
        return None
    resolver = meta.get("resolver")
    if resolver is not None:
        # Only shipped files may name the built-in classifier: a user file that
        # did would just duplicate it, and a typo would read as coverage.
        if not shipped or resolver not in BUILTIN_RESOLVERS:
            logger.warning("guardrail rule %s skipped: unknown resolver %r", source, resolver)
            return None
    raw = meta.get("condition")
    if isinstance(raw, str):
        raw = [raw]
    patterns: list[re.Pattern[str]] = []
    if isinstance(raw, list):
        for item in raw:
            try:
                patterns.append(re.compile(str(item)))
            except re.error as exc:
                logger.warning("guardrail rule %s: condition %r skipped: %s", source, item, exc)
    if not patterns and resolver is None:
        logger.warning("guardrail rule %s skipped: no usable condition", source)
        return None
    return Rule(
        name=name,
        scope=scope,
        action=action,
        conditions=tuple(patterns),
        question=str(meta.get("question") or "").strip(),
        body=_body(text),
        resolver=str(resolver) if resolver else None,
        source=source,
    )


#: Parsed rules keyed by (path, mtime_ns, size, shipped) (Q4). The checkpoint
#: runs on every tool call; re-parsing every file each time re-logged a broken
#: file's warning per call. A file edit changes the key, so it is re-read — and
#: warned about — once per version.
_PARSE_CACHE: dict[tuple[str, int, int, bool], Rule | None] = {}
_PARSE_CACHE_MAX = 512


def _load_dir(directory: Path, *, shipped: bool) -> list[Rule]:
    rules: list[Rule] = []
    try:
        files = sorted(p for p in directory.glob("*.md") if p.is_file())[:_MAX_RULE_FILES]
    except OSError:
        return rules
    for path in files:
        try:
            st = path.stat()
        except OSError:
            continue
        key = (str(path), st.st_mtime_ns, st.st_size, shipped)
        if key in _PARSE_CACHE:
            rule = _PARSE_CACHE[key]
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("guardrail rule %s unreadable: %s", path, exc)
                continue
            rule = parse_rule(text, source=str(path), shipped=shipped)
            if len(_PARSE_CACHE) >= _PARSE_CACHE_MAX:
                _PARSE_CACHE.clear()
            _PARSE_CACHE[key] = rule
        if rule is not None:
            rules.append(rule)
    return rules


def load_rules(rules_dir: Path | None) -> RuleSet:
    """Shipped defaults plus the user's rules directory (absent = none)."""
    rules = RuleSet(builtin=_load_dir(DEFAULTS_DIR, shipped=True))
    if rules_dir is not None and rules_dir.is_dir():
        rules.user = _load_dir(rules_dir, shipped=False)
    return rules

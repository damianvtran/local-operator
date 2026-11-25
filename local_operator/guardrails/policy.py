"""The guard's own policy files, and which tool calls would change them.

WHY. A user rule (``<config_dir>/rules/*.md``), the native hooks file
(``hooks.json``, whose commands run as shell commands outside the guard) and
``config.yml`` (the switches for hooks and the guard) ARE the guard's policy.
Bash already ranks every write to them; the file tools did not, so one ``write``
call could empty a rule or add a hook command. The operator's intent (Ben,
2026-11-23) is that these edits go to the judge, not to a hard refusal:
``guide://guardrails`` tells agents to write rules when the user asks for one.

This module is the ONE place that names the protected set and decides, from the
call's arguments alone, whether a call touches it. It makes no model call and
reads no conversation, so the checkpoint, the preview and the agent loop's
fail-closed prelude all share the same answer.

Not covered, said plainly (also in ``guide://guardrails``): a path composed in
eval code (``Path.home() / ".local-operator"``, env vars, globs), other
interpreters, the ``skill``/``agent``/``team`` stores, forwarded Claude/Codex hook
files, the installed package, and a symlink planted by a concurrent sibling call
between this check and the write.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Setting keys the ``setting`` tool may not change without the judge: the switches
#: for hook forwarding and the guard itself (``hooks.native`` runs shell commands).
_SETTING_PREFIXES = ("hooks.", "guardrails.")

#: Mirrors ``rules._MAX_RULE_FILES``: the loader reads no more than this many.
_MAX_RULE_FILES = 200

#: Bytes of a current file read for the judge's facts.
_FACT_READ_BYTES = 65536

LABELS = {
    "rules": "guardrail rule file",
    "hooks": "native hooks file (its commands run as shell commands)",
    "config": "config.yml (holds the hooks, guardrail and command settings)",
    "setting": "setting that controls hooks or the guard",
    "eval": "protected path named in eval code",
}


@dataclass(frozen=True)
class PolicyHit:
    kind: str  # "rules" | "hooks" | "config" | "setting" | "eval"
    #: The resolved physical path, the setting key, or the literal found in the cell.
    shown: str
    label: str
    #: ``None`` for a setting or an eval literal.
    exists: bool | None = None
    size: int | None = None


def _state_files() -> tuple[tuple[str, Path], ...]:
    """The two single-file members of the protected set, in the CURRENT config dir.

    ``paths.config_dir`` reads ``LOCAL_OPERATOR_CONFIG_DIR`` per call and is the
    function the hook loader and ``ConfigManager`` use, so an override cannot put
    the guard's idea of the files somewhere else than the loader's.
    """
    from local_operator.paths import config_dir

    root = config_dir()
    return (("hooks", root / "hooks.json"), ("config", root / "config.yml"))


def _identity(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    return (st.st_dev, st.st_ino)


def _fold_equal(left: str, right: str, fold: bool) -> bool:
    return left == right or (fold and left.casefold() == right.casefold())


def _protected_identities(rules_dirs: Iterable[Path]) -> dict[tuple[int, int], str]:
    """Inode identity -> kind for every EXISTING protected file.

    A path match alone misses a hardlink (``ln rules/x.md ~/y.md``) and a rule
    that is a symlink to a file elsewhere: ``stat`` follows links, the tool's
    resolved target is the real file, and the inode is what the loader reads.
    """
    out: dict[tuple[int, int], str] = {}
    for kind, path in _state_files():
        ident = _identity(str(path))
        if ident is not None:
            out[ident] = kind
    for directory in rules_dirs:
        try:
            files = sorted(p for p in directory.glob("*.md") if p.is_file())[:_MAX_RULE_FILES]
        except (OSError, ValueError):
            continue
        for path in files:
            ident = _identity(str(path))
            if ident is not None:
                out[ident] = "rules"
    return out


def _file_kind(target: str, rules_dirs: Sequence[Path]) -> str | None:
    """Which protected kind the resolved path ``target`` is, or ``None``."""
    from local_operator.guardrails.classify import _case_insensitive, _spellings

    spellings = _spellings(target)
    for kind, path in _state_files():
        fold = _case_insensitive(str(path))
        for protected in _spellings(str(path)):
            if any(_fold_equal(s, protected, fold) for s in spellings):
                return kind
    for directory in rules_dirs:
        fold = _case_insensitive(str(directory))
        dir_spellings = _spellings(str(directory))
        for spelled in spellings:
            # The loader globs ``*.md`` non-recursively, so only a direct child
            # with that suffix is a rule; anything else in the folder is inert.
            if not os.path.basename(spelled).casefold().endswith(".md"):
                continue
            parent = os.path.dirname(spelled)
            if any(_fold_equal(parent, d, fold) for d in dir_spellings):
                return "rules"
    ident = _identity(target)
    if ident is not None:
        return _protected_identities(rules_dirs).get(ident)
    return None


def _resolve_target(raw: Any, cwd: str | None, scratchpad: str | None) -> str | None:
    """Where a file tool would write ``raw``, using the tools' own resolvers.

    One resolver: ``harness/guard_area.py`` imports the same two functions for
    the same reason. ``None`` means no protected file can be reached (the tool
    would refuse the spelling, or it is not a path).
    """
    from local_operator.scratchpad import ScratchpadPathError, parse_scratchpad_url
    from local_operator.tools.builtin import (
        _has_scratchpad_scheme,
        _resolve_workspace_path,
    )

    if not isinstance(raw, str) or not raw.strip():
        return None
    if raw.strip().lower().startswith("memory:"):
        return None
    if _has_scratchpad_scheme(raw):
        if not scratchpad:
            return None
        try:
            # Resolves THROUGH a pad that was swapped for a symlink.
            return str(parse_scratchpad_url(raw.strip(), Path(scratchpad)).path)
        except (ScratchpadPathError, ValueError, OSError):
            return None
    if "://" in raw:
        return None
    # ``cwd or "."`` is what ``builtin._safe_cwd`` does; ``raw`` is unstripped,
    # as ``execute_write``/``execute_edit`` pass it.
    try:
        path, _inside, _resolvable = _resolve_workspace_path(raw, cwd or ".")
    except (OSError, ValueError, RuntimeError):
        return None
    return str(path)


def _file_hit(
    raw: Any, cwd: str | None, rules_dirs: Sequence[Path], scratchpad: str | None
) -> PolicyHit | None:
    target = _resolve_target(raw, cwd, scratchpad)
    if target is None:
        return None
    kind = _file_kind(target, rules_dirs)
    if kind is None:
        return None
    exists = os.path.lexists(target)
    size: int | None = None
    if exists:
        try:
            size = os.stat(target).st_size
        except OSError:
            size = None
    return PolicyHit(kind, target, LABELS[kind], exists, size)


def _eval_literals(rules_dirs: Sequence[Path]) -> tuple[str, ...]:
    """Spellings of the protected set that eval code could contain literally.

    Deliberately NOT bare ``.local-operator`` or ``config.yml``: agents write
    ``~/.local-operator/sessions/...`` and project ``config.yml`` files constantly.
    """
    from local_operator.paths import config_dir

    root = config_dir()
    home = os.path.expanduser("~")
    bases = [str(root / "hooks.json"), str(root / "config.yml"), str(root / "rules")]
    bases += [str(d) for d in rules_dirs]
    literals: list[str] = []
    for base in bases:
        literals.append(base)
        if home and base.startswith(home + os.sep):
            literals.append("~" + base[len(home) :])
    literals += [f"{root.name}/hooks.json", f"{root.name}/config.yml", f"{root.name}/rules"]
    return tuple(dict.fromkeys(literals))


def _eval_hit(code: Any, rules_dirs: Sequence[Path]) -> PolicyHit | None:
    """Best effort: a protected literal in a cell that is not provably read-only."""
    if not isinstance(code, str) or not code:
        return None
    found = next((lit for lit in _eval_literals(rules_dirs) if lit in code), None)
    if found is None:
        return None
    from local_operator.guardrails.resolve import python_code_read_only

    if python_code_read_only(code):
        return None
    return PolicyHit("eval", found, LABELS["eval"])


def policy_hits(
    tool_name: str,
    args: Mapping[str, Any],
    *,
    cwd: str | None,
    rules_dirs: Sequence[Path],
    scratchpad: str | None,
) -> tuple[PolicyHit, ...]:
    """The protected policy this call would change; empty when it touches none.

    Pure: no model, no conversation, no network.
    """
    from local_operator.guardrails.check import FILE_DESTINATION_TOOLS

    key = FILE_DESTINATION_TOOLS.get(tool_name)
    if key is not None:
        hit = _file_hit(args.get(key), cwd, rules_dirs, scratchpad)
        return (hit,) if hit is not None else ()
    if tool_name == "setting":
        setting = args.get("key")
        if (
            args.get("op") == "set"
            and isinstance(setting, str)
            and setting.strip().casefold().startswith(_SETTING_PREFIXES)
        ):
            return (PolicyHit("setting", setting.strip(), LABELS["setting"]),)
        return ()
    if tool_name == "eval":
        found = _eval_hit(args.get("code"), rules_dirs)
        return (found,) if found is not None else ()
    return ()


def hits_text(hits: Sequence[PolicyHit]) -> str:
    """The hits as one line for a refusal, the journal and the preview."""
    parts = [f"{h.shown} ({h.label})" for h in hits[:6]]
    if len(hits) > 6:
        parts.append(f"… and {len(hits) - 6} more")
    return "; ".join(parts)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _rule_summary(text: str, source: str) -> str:
    """How the loader would read ``text``: the one fact a judge cannot see in a diff."""
    from local_operator.guardrails.rules import parse_rule

    if not text.strip():
        return "EMPTY content"
    rule = parse_rule(text, source=source)
    if rule is None:
        return "does NOT parse: the loader would skip it, so it protects nothing"
    return f"parses as rule {rule.name!r} (action {rule.action}, scope {rule.scope})"


def _hook_commands(text: str) -> str:
    from local_operator.hook_forwarding import _commands_from

    try:
        data = json.loads(text)
    except ValueError:
        return "is not valid JSON, so no hook would load"
    if not isinstance(data, Mapping):
        return "has no hooks object"
    commands = _commands_from(data, "proposed")
    if not commands:
        return "defines no hook commands"
    shown = "; ".join(f"{c.event}: {_clip(c.command, 120)}" for c in commands[:5])
    more = f" (+{len(commands) - 5} more)" if len(commands) > 5 else ""
    return f"defines {len(commands)} hook command(s): {shown}{more}"


def _current_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read(_FACT_READ_BYTES)
    except (OSError, ValueError):
        return None


def _describe(hit: PolicyHit, text: str) -> str:
    return _rule_summary(text, hit.shown) if hit.kind == "rules" else _hook_commands(text)


def hit_facts(
    tool_name: str, args: Mapping[str, Any], hits: Sequence[PolicyHit]
) -> tuple[str, ...]:
    """Deterministic fact lines for the judge's ``<targets>`` block.

    The judge is told to trust these over the call text: they are computed from
    the proposed content, not from the agent's description of it.
    """
    facts: list[str] = []
    for hit in hits:
        if hit.kind in ("setting", "eval"):
            if hit.kind == "setting":
                facts.append(f"{hit.label}: {hit.shown} = {_clip(repr(args.get('value')), 200)}")
            else:
                facts.append(f"{hit.label}: {hit.shown}")
            continue
        state = f"exists, {hit.size} bytes" if hit.exists else "does not exist yet"
        facts.append(f"{hit.label}: {hit.shown} ({state})")
        if hit.kind not in ("rules", "hooks"):
            continue
        if tool_name == "write" and isinstance(args.get("content"), str):
            proposed = _describe(hit, args["content"])
            facts.append(f"  proposed content {proposed}")
        elif tool_name == "edit":
            edits = args.get("edits")
            count = len(edits) if isinstance(edits, list) else 1
            facts.append(
                f"  edit with {count} hunk(s): the resulting file is not computed here, "
                "judge the hunks in the call"
            )
        current = _current_text(hit.shown) if hit.exists else None
        if current is not None:
            facts.append(f"  current content {_describe(hit, current)}")
    return tuple(facts)


__all__ = ["LABELS", "PolicyHit", "hit_facts", "hits_text", "policy_hits"]

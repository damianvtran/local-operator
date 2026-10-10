"""Run ``PostToolUse`` hooks: lop's own (``hooks.json``) and forwarded
Claude Code / Codex hooks.

lop's own hook file is ``config_dir()/hooks.json`` — "native" hooks, enabled by
``hooks.native`` — and it uses Claude's schema. lop also runs the hooks the
operator already configured for Claude Code (``~/.claude/settings.json``, the
project's ``.claude/settings{,.local}.json``, and the ``hooks/hooks.json`` of
every enabled Claude plugin) and for Codex (``~/.codex/hooks.json``). All
sources share ONE contract — the same stdin payload and the same output
handling — so a script written for either tool behaves the same when lop runs
it.

Scope, deliberately: ``PostToolUse`` and ``PostToolUseFailure`` only. The other
events (``SessionStart``, ``UserPromptSubmit``, ``PreToolUse``, ``Stop``, the
subagent events) are not run yet; ``docs/HOOKS.md`` lists them.

All sources are OFF by default (``hooks.native`` / ``hooks.forward_claude`` /
``hooks.forward_codex``) and read at CALL time, the way ``bash.shell`` is, so
toggling them in ``/settings`` reaches the very next tool call.

Execution is platform-conditional: every hook spawns in its own session
(``start_new_session``, POSIX-only — ignored on Windows), so a timeout or a
cancelled turn stops the hook's whole tree — the process group on POSIX,
``taskkill /T /F`` on Windows — through
``local_operator.procstate.terminate_process_tree``, the repository's one kill
path, which never raises.

A hook can never break a turn: every failure (unreadable config, a hook that
crashes, times out or prints garbage) is logged and the tool result goes back
to the model unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.procstate import terminate_process_tree

logger = logging.getLogger(__name__)

FORWARD_CLAUDE_PATH: tuple[str, ...] = ("hooks", "forward_claude")
FORWARD_CLAUDE_DEFAULT = False
FORWARD_CODEX_PATH: tuple[str, ...] = ("hooks", "forward_codex")
FORWARD_CODEX_DEFAULT = False
NATIVE_PATH: tuple[str, ...] = ("hooks", "native")
NATIVE_DEFAULT = False

#: Claude Code's default for a command hook on the tool events, in seconds.
DEFAULT_TIMEOUT_S = 600.0
#: Claude Code caps ``additionalContext``/``reason``/plain stdout at this many
#: characters per string.
CONTEXT_CAP = 10_000

POST_TOOL_USE = "PostToolUse"
POST_TOOL_USE_FAILURE = "PostToolUseFailure"


@dataclass(frozen=True)
class TaggedNote:
    """A note that carries its OWN event tag rather than the caller's event.

    The post-tool seam wraps every note as ``<hook-context event="…">``, where
    the event is the CALLER's (``PostToolUse``/``PostToolUseFailure``). The
    session's own code-request note is neither — it is a harness fact about the
    call, not a forwarded hook event — so it carries its own tag
    (``code-requests``) and :func:`format_notes` honours it. A bare ``str``
    keeps the pre-existing behaviour exactly, which is why the list type widened
    instead of the wrapper replacing it.
    """

    event: str
    text: str


#: lop tool name -> the Claude Code tool name matchers are written against.
#: Unmapped tools (MCP tools, lop-only tools) pass through under their own name.
TOOL_NAME_MAP: dict[str, str] = {
    "bash": "Bash",
    "write": "Write",
    "edit": "Edit",
    "read": "Read",
    "grep": "Grep",
    "glob": "Glob",
    "web_fetch": "WebFetch",
    "web_search": "WebSearch",
    # Claude Code's subagent tool is ``Agent``; ``Task`` survives as a matcher
    # alias (``TOOL_MATCHER_ALIASES``) for hooks written against the old name.
    "task": "Agent",
}

#: Extra names a matcher may use for the same call. Claude Code's subagent tool
#: is ``Agent`` and its file-edit tools are ``Write``/``Edit``, while Codex
#: reports edits as ``apply_patch`` — a hook written for either tool must fire
#: under lop, so matching tries the canonical name and these aliases.
TOOL_MATCHER_ALIASES: dict[str, tuple[str, ...]] = {
    "write": ("apply_patch",),
    "edit": ("apply_patch",),
    "task": ("Task",),
}

_EXACT_MATCHER = re.compile(r"^[A-Za-z0-9_\-\s,|]*$")


@dataclass(frozen=True)
class HookCommand:
    """One ``type: command`` entry, with where it came from."""

    event: str
    matcher: str | None
    command: str
    timeout_s: float
    source: str
    plugin_root: str | None = None


@dataclass(frozen=True)
class HookIdentity:
    """Who is running the tool: the fields the payload carries about the caller."""

    session_id: str
    cwd: str
    transcript_path: str | None = None
    #: Set only inside a subagent, exactly like Claude Code's ``agent_id``.
    agent_id: str | None = None
    agent_type: str | None = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def forwarding_enabled() -> tuple[bool, bool]:
    """``(claude, codex)`` read from config.yml at call time; any failure is off."""
    try:
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir

        manager = ConfigManager(config_dir())
        claude = manager.get_nested_value(FORWARD_CLAUDE_PATH, FORWARD_CLAUDE_DEFAULT)
        codex = manager.get_nested_value(FORWARD_CODEX_PATH, FORWARD_CODEX_DEFAULT)
    except Exception:  # noqa: BLE001 - config trouble must never block a tool
        return (False, False)
    return (claude is True, codex is True)


def native_hooks_enabled() -> bool:
    """Whether ``config_dir()/hooks.json`` is on; read at call time, any failure off."""
    try:
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir

        manager = ConfigManager(config_dir())
        native = manager.get_nested_value(NATIVE_PATH, NATIVE_DEFAULT)
    except Exception:  # noqa: BLE001 - config trouble must never block a tool
        return False
    return native is True


def _read_json_value(path: Path) -> tuple[bool, Any]:
    """``(readable, value)``: the file's JSON value as written.

    ``_read_json`` projects this onto ``dict | None``; callers that must
    distinguish "no usable shape" from "no file" (the native loader) need the
    present-but-unusable shapes that projection erases — an array, ``{}``, a
    scalar. An unreadable file is logged once, here.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return (False, None)
    except (OSError, ValueError):
        logger.warning("hooks: could not read %s", path, exc_info=True)
        return (False, None)
    return (True, value)


def _read_json(path: Path) -> dict[str, Any] | None:
    readable, value = _read_json_value(path)
    return value if readable and isinstance(value, dict) else None


def _project_root(cwd: str) -> Path:
    """The nearest ancestor holding ``.claude`` or ``.git``; else ``cwd`` itself."""
    start = Path(cwd).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".claude").is_dir() or (candidate / ".git").exists():
            return candidate
    return start


def _commands_from(
    data: Mapping[str, Any] | None, source: str, plugin_root: str | None = None
) -> list[HookCommand]:
    if not data:
        return []
    events = data.get("hooks")
    if not isinstance(events, Mapping):
        return []
    out: list[HookCommand] = []
    for event, groups in events.items():
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, Mapping):
                continue
            matcher = group.get("matcher")
            for hook in group.get("hooks") or []:
                if not isinstance(hook, Mapping) or hook.get("type", "command") != "command":
                    continue
                command = hook.get("command")
                if not isinstance(command, str) or not command.strip():
                    continue
                timeout = hook.get("timeout")
                timeout_s = (
                    float(timeout)
                    if isinstance(timeout, int | float) and timeout > 0
                    else DEFAULT_TIMEOUT_S
                )
                out.append(
                    HookCommand(
                        event=str(event),
                        matcher=matcher if isinstance(matcher, str) else None,
                        command=command,
                        timeout_s=timeout_s,
                        source=source,
                        plugin_root=plugin_root,
                    )
                )
    return out


def _enabled_plugin_roots(
    claude_dir: Path, enabled: Mapping[str, Any], project_root: Path
) -> list[str]:
    """Install paths of the enabled plugins, resolved through ``installed_plugins.json``."""
    installed = _read_json(claude_dir / "plugins" / "installed_plugins.json") or {}
    plugins = installed.get("plugins")
    if not isinstance(plugins, Mapping):
        return []
    roots: list[str] = []
    for key, on in enabled.items():
        if on is not True:
            continue
        for entry in plugins.get(key) or []:
            if not isinstance(entry, Mapping):
                continue
            scope = entry.get("scope", "user")
            if scope != "user":
                project = entry.get("projectPath")
                if not isinstance(project, str) or Path(project).resolve() != project_root:
                    continue
            path = entry.get("installPath")
            if isinstance(path, str):
                roots.append(path)
                break
    return roots


def load_hook_commands(
    cwd: str, *, claude: bool, codex: bool, home: Path | None = None
) -> list[HookCommand]:
    """Every configured command hook from the enabled sources, merged in Claude's order."""
    home = home or Path.home()
    commands: list[HookCommand] = []
    if claude:
        claude_dir = home / ".claude"
        root = _project_root(cwd)
        # ``~/.claude`` exists for every Claude Code user, so the ancestor walk
        # resolves any cwd under $HOME to the home directory itself. $HOME is
        # not a project: reading ``root/.claude/settings.json`` there would run
        # the user layer's hooks a second time (and the walk could just as
        # easily load an ANCESTOR's settings that Claude Code — cwd or git repo
        # root only — never reads). Skip the project/local layers in that case.
        candidates: list[tuple[str, Path]] = [("claude:user", claude_dir / "settings.json")]
        if root != home.resolve():
            candidates += [
                ("claude:project", root / ".claude" / "settings.json"),
                ("claude:local", root / ".claude" / "settings.local.json"),
            ]
        # Dedupe by RESOLVED path: a file reached through more than one slot (a
        # symlink, or $HOME resolved as its own project) is loaded once.
        layers: list[tuple[str, dict[str, Any] | None]] = []
        seen: set[Path] = set()
        for source, path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            layers.append((source, _read_json(path)))
        # ``disableAllHooks``: the last layer that states it wins, as in Claude.
        disabled = False
        enabled_plugins: dict[str, Any] = {}
        for _source, data in layers:
            if data is None:
                continue
            if isinstance(data.get("disableAllHooks"), bool):
                disabled = data["disableAllHooks"]
            plugins = data.get("enabledPlugins")
            if isinstance(plugins, Mapping):
                enabled_plugins.update(plugins)
        if not disabled:
            for source, data in layers:
                commands.extend(_commands_from(data, source))
            for plugin_root in _enabled_plugin_roots(claude_dir, enabled_plugins, root):
                data = _read_json(Path(plugin_root) / "hooks" / "hooks.json")
                commands.extend(_commands_from(data, f"claude:plugin:{plugin_root}", plugin_root))
    if codex:
        commands.extend(_commands_from(_read_json(home / ".codex" / "hooks.json"), "codex"))
    return commands


def load_native_commands() -> list[HookCommand]:
    """Every ``type: command`` hook in ``config_dir()/hooks.json`` (Claude schema).

    Read at CALL time like the forwarded sources, so an edit lands on the next
    tool call. An absent file is a no-op; unreadable JSON keeps its ``could not
    read`` warning; every other present-but-unusable shape (no usable ``hooks``
    mapping, an empty or non-object document) warns rather than loading nothing
    in silence. ``disableAllHooks: true`` disables its hooks.
    """
    from local_operator.paths import config_dir

    path = config_dir() / "hooks.json"
    readable, value = _read_json_value(path)
    if not readable:
        return []  # absent, or unreadable (already logged)
    if isinstance(value, dict) and value.get("disableAllHooks") is True:
        return []
    if isinstance(value, dict) and isinstance(value.get("hooks"), Mapping):
        return _commands_from(value, "native")
    # Present and valid JSON, but no usable ``hooks`` mapping: the natural
    # hand-written mistake (``{"PostToolUse": [...]}`` at the top level), an
    # empty document, or a non-object — say so rather than loading nothing in
    # silence.
    logger.warning("hooks: %s has no top-level 'hooks' mapping; nothing will run", path)
    return []


# ---------------------------------------------------------------------------
# Matching and payload
# ---------------------------------------------------------------------------


def matcher_matches(matcher: str | None, value: str) -> bool:
    """Claude Code's matcher rules: blank/``*`` = all, name list = exact, else regex."""
    if matcher is None or matcher.strip() in ("", "*"):
        return True
    if _EXACT_MATCHER.match(matcher):
        names = {part.strip() for part in re.split(r"[|,]", matcher) if part.strip()}
        return value in names
    try:
        return re.search(matcher, value) is not None
    except re.error:
        logger.warning("hooks: invalid matcher %r", matcher)
        return False


def matcher_names(tool_name: str, mapped: str) -> tuple[str, ...]:
    """Every name this call may be matched against: canonical first, then aliases.

    The candidate set lives at the call site rather than inside
    ``matcher_matches`` because it depends on which lop tool ran; that function
    keeps comparing one name at a time.
    """
    return (mapped, *TOOL_MATCHER_ALIASES.get(tool_name, ()))


def _absolute(path: Any, cwd: str) -> Any:
    if not isinstance(path, str) or not path:
        return path
    return str((Path(cwd) / os.path.expanduser(path)).resolve())


def claude_tool(tool_name: str, args: Mapping[str, Any], cwd: str) -> tuple[str, dict[str, Any]]:
    """The Claude-shaped ``(tool_name, tool_input)`` for one lop call."""
    if tool_name == "bash":
        tool_input: dict[str, Any] = {"command": args.get("command", "")}
        if isinstance(args.get("timeout"), int | float):
            tool_input["timeout"] = int(args["timeout"] * 1000)
        tool_input["run_in_background"] = bool(args.get("background", False))
        return "Bash", tool_input
    if tool_name == "write":
        return "Write", {
            "file_path": _absolute(args.get("path"), cwd),
            "content": args.get("content", ""),
        }
    if tool_name == "edit":
        file_path = _absolute(args.get("path"), cwd)
        hunks = args.get("edits")
        if isinstance(hunks, list) and len(hunks) != 1:
            # Claude Code has no ``MultiEdit`` (its ``Edit`` edits one string at
            # a time), so a multi-hunk edit is reported as ``Edit`` — the
            # standard ``Edit``/``Edit|Write`` matchers must fire — carrying the
            # first hunk's fields plus the whole list.
            first = hunks[0] if hunks and isinstance(hunks[0], Mapping) else {}
            return "Edit", {
                "file_path": file_path,
                "old_string": first.get("old_text", ""),
                "new_string": first.get("new_text", ""),
                "replace_all": bool(first.get("replace_all", False)),
                "edits": [
                    {
                        "old_string": h.get("old_text", ""),
                        "new_string": h.get("new_text", ""),
                        "replace_all": bool(h.get("replace_all", False)),
                    }
                    for h in hunks
                    if isinstance(h, Mapping)
                ],
            }
        hunk = hunks[0] if isinstance(hunks, list) and isinstance(hunks[0], Mapping) else args
        return "Edit", {
            "file_path": file_path,
            "old_string": hunk.get("old_text", ""),
            "new_string": hunk.get("new_text", ""),
            "replace_all": bool(hunk.get("replace_all", False)),
        }
    if tool_name == "read":
        return "Read", {"file_path": _absolute(args.get("path"), cwd)}
    if tool_name == "web_fetch":
        return "WebFetch", {"url": args.get("url", "")}
    mapped = TOOL_NAME_MAP.get(tool_name, tool_name)
    return mapped, dict(args)


def build_payload(
    event: str,
    identity: HookIdentity,
    tool_name: str,
    tool_input: Mapping[str, Any],
    *,
    tool_use_id: str,
    output: str,
    duration_s: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": identity.session_id,
        "cwd": identity.cwd,
        "permission_mode": "default",
        "hook_event_name": event,
        "tool_name": tool_name,
        "tool_input": dict(tool_input),
        "tool_use_id": tool_use_id,
    }
    if identity.transcript_path:
        payload["transcript_path"] = identity.transcript_path
    if identity.agent_id:
        payload["agent_id"] = identity.agent_id
    if identity.agent_type:
        payload["agent_type"] = identity.agent_type
    if duration_s is not None:
        payload["duration_ms"] = int(duration_s * 1000)
    if event == POST_TOOL_USE:
        if tool_name == "Bash":
            payload["tool_response"] = {
                "stdout": output,
                "stderr": "",
                "interrupted": False,
            }
        else:
            payload["tool_response"] = {"output": output}
    else:
        payload["error"] = output
        payload["is_interrupt"] = False
    return payload


# ---------------------------------------------------------------------------
# Running and interpreting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HookRun:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    """Stop the hook and everything it spawned; never raises.

    ``terminate_process_tree`` owns the platform split: a leader-checked
    ``killpg`` of the group a ``start_new_session`` child leads on POSIX, a
    ``taskkill /T /F`` tree kill on Windows — so this module never touches
    the POSIX-only ``os.killpg``/``signal.SIGKILL`` attributes itself.
    """
    terminate_process_tree(proc.pid, force=True)


async def run_hook(hook: HookCommand, payload: Mapping[str, Any], cwd: str) -> HookRun:
    """Run one hook in its own process group; reap the whole tree on timeout or cancel."""
    env = dict(os.environ)
    env["CLAUDE_PROJECT_DIR"] = str(_project_root(cwd))
    if hook.plugin_root:
        env["CLAUDE_PLUGIN_ROOT"] = hook.plugin_root
    proc = await asyncio.create_subprocess_shell(
        hook.command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        # POSIX-only: makes the child a process-group leader so a timeout or a
        # cancelled turn can reap its whole tree (ignored on Windows, where
        # ``_kill_group`` falls back to a ``taskkill /T`` tree kill).
        start_new_session=True,
    )
    data = json.dumps(payload).encode()
    try:
        out, err = await asyncio.wait_for(proc.communicate(data), timeout=hook.timeout_s)
    except TimeoutError:
        _kill_group(proc)
        with contextlib.suppress(Exception):
            await proc.wait()
        return HookRun(None, "", "", timed_out=True)
    except BaseException:
        _kill_group(proc)
        raise
    # A hook that exits normally keeps whatever it deliberately backgrounded
    # (a notifier, a status ping): Claude Code does not reap those either.
    return HookRun(
        proc.returncode,
        out.decode("utf-8", "replace"),
        err.decode("utf-8", "replace"),
    )


def _cap(text: str) -> str:
    text = text.strip()
    return text if len(text) <= CONTEXT_CAP else text[:CONTEXT_CAP] + "\n[... truncated]"


def interpret(run: HookRun, event: str) -> list[str]:
    """What the model should see from one hook run, per Claude Code's contract."""
    if run.timed_out or run.exit_code is None:
        return []
    notes: list[str] = []
    stdout = run.stdout.strip()
    parsed: dict[str, Any] | None = None
    if stdout.startswith("{") and stdout.endswith("}"):
        try:
            candidate = json.loads(stdout)
        except ValueError:
            logger.warning("hooks: %s hook printed unparseable JSON", event)
        else:
            parsed = candidate if isinstance(candidate, dict) else None
    if parsed is not None:
        specific = parsed.get("hookSpecificOutput")
        if isinstance(specific, Mapping):
            context = specific.get("additionalContext")
            if isinstance(context, str) and context.strip():
                notes.append(_cap(context))
        if parsed.get("decision") == "block":
            reason = parsed.get("reason")
            if isinstance(reason, str) and reason.strip():
                notes.append(_cap(reason))
    if run.exit_code == 2 and run.stderr.strip() and not (parsed and parsed.get("reason")):
        notes.append(_cap(run.stderr))
    return notes


async def run_post_tool_hooks(
    identity: HookIdentity,
    *,
    tool_name: str,
    args: Mapping[str, Any],
    tool_use_id: str,
    output: str,
    is_error: bool,
    duration_s: float | None,
    home: Path | None = None,
) -> list[str]:
    """Run every matching Post hook for one finished call; return context notes.

    Native (``hooks.json``) entries run before forwarded ones; within a source
    the file order holds. Never raises for a hook's own failure; cancellation
    still propagates (and reaps the hook's process tree) so an aborted turn
    stops promptly.
    """
    claude, codex = forwarding_enabled()
    native = native_hooks_enabled()
    if not (claude or codex or native):
        return []
    event = POST_TOOL_USE_FAILURE if is_error else POST_TOOL_USE
    try:
        loaded: list[HookCommand] = []
        if native:
            loaded.extend(load_native_commands())
        if claude or codex:
            loaded.extend(load_hook_commands(identity.cwd, claude=claude, codex=codex, home=home))
        hooks = [h for h in loaded if h.event == event]
        if not hooks:
            return []
        mapped, tool_input = claude_tool(tool_name, args, identity.cwd)
        names = matcher_names(tool_name, mapped)
        hooks = [h for h in hooks if any(matcher_matches(h.matcher, n) for n in names)]
        if not hooks:
            return []
        payload = build_payload(
            event,
            identity,
            mapped,
            tool_input,
            tool_use_id=tool_use_id,
            output=output,
            duration_s=duration_s,
        )
    except Exception:
        logger.warning("hooks: could not prepare %s hooks", event, exc_info=True)
        return []

    async def one(hook: HookCommand) -> list[str]:
        try:
            run = await run_hook(hook, payload, identity.cwd)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "hooks: %s hook failed: %s",
                event,
                hook.command,
                exc_info=True,
            )
            return []
        if run.timed_out:
            logger.warning("hooks: %s hook timed out: %s", event, hook.command)
        elif run.exit_code not in (0, 2):
            logger.warning(
                "hooks: %s hook exited %s: %s",
                event,
                run.exit_code,
                run.stderr.strip(),
            )
        return interpret(run, event)

    # Claude runs matching hooks in parallel; so does this.
    results = await asyncio.gather(*(one(h) for h in hooks))
    return [note for notes in results for note in notes]


def format_notes(event_notes: Sequence[str | TaggedNote], event: str) -> str:
    """The block appended to the tool result, one per note.

    ``event`` is the event the notes came from: the caller knows whether the
    tool call failed, and a failure note labelled ``PostToolUse`` would report
    the wrong event to whatever reads the tag. A :class:`TaggedNote` supplies
    its own tag instead. ``Sequence`` (not ``list``) because the loop's hook is
    typed to return ``list[str]``: a list is invariant, so a list-typed
    parameter would reject the plain-strings case at the call site.
    """
    rendered: list[str] = []
    for item in event_notes:
        if isinstance(item, TaggedNote):
            rendered.append(f'<hook-context event="{item.event}">\n{item.text}\n</hook-context>')
        else:
            rendered.append(f'<hook-context event="{event}">\n{item}\n</hook-context>')
    return "\n\n".join(rendered)

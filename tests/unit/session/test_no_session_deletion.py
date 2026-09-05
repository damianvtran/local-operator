"""No code outside ``session/cleanup.py`` may remove, rename or replace a
session directory — enforced over the source tree, by AST, naming file:line.

WHY THIS TEST EXISTS. An automatic "unused session" reaper (#576) and an
exit-path ``rmdir`` (#622) both shipped as "safe" and both fired on the
operator's real store; every surviving session directory afterwards had a
birth time hours after its transcript's first row, so directories were also
being recreated — a rename or replace of a session directory is a deletion
of the original by another name. Reading the module docstring that promised
"nothing is ever deleted" did not stop either. A test that fails the build
on the *shape* of the code does.

HOW IT WORKS. Every ``.py`` under ``local_operator/`` is parsed and each call
named in :data:`_NAMES` is checked against :data:`_ALLOWED`, an explicit
allow-list keyed by ``relative/path.py::function::call-label`` — the
FUNCTION and the SPECIFIC CALL SHAPE in it (``os.replace``, ``<path>.unlink``,
``shutil.rmtree``…) — carrying the reason that call cannot reach a session
directory. Keyed by call, not by function, because an excused function is
otherwise a blind surface: review round 2 (R2-1) dropped ``shutil.rmtree``
into an allow-listed ``wakes/store.py::remove_entry`` and the function-keyed
list waved it through. Anything not in the list fails with its
``file:line``. Adding a call site therefore means adding a row HERE with a
reason a reviewer can check — the point is not that the list is short, it
is that every entry was argued for.

BIASED TOWARD FALSE POSITIVES, on purpose. Import aliases are resolved
(``from shutil import rmtree``, ``import shutil as sh``), a call on ANY
receiver counts (``target.rmdir()``, ``Path(...).unlink()``, ``self.path.
rename(x)``), and ``unlink``/``remove`` are in the set because a transcript is
a file. The only exclusions are shapes that are provably not the filesystem:
``.replace(a, b)`` with two positionals is ``str.replace``; ``.remove(x)`` on a
literal container. A ``widget.remove()`` or ``listeners.remove(cb)`` therefore
lands in the allow-list with a one-word reason — that cost is the point, and
review round 1 measured what the cheaper guard missed (R1-2: 5 of 8 mutants).
Two tests at the bottom pin the mutants and the exclusions.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "local_operator"

#: The one module permitted to ``rmtree`` a session directory.
CLEANUP_MODULE = "local_operator/session/cleanup.py"

#: Call names that remove or displace something on disk. ``rmtree``/``rmdir``/
#: ``removedirs`` take directories; ``rename``/``replace``/``renames``/``move``
#: displace either (a moved session directory is a deletion of the original
#: by another name); ``unlink``/``remove`` take files, and a transcript is a
#: file — deleting it empties a session as surely as removing the directory.
_NAMES = frozenset(
    {"rmtree", "rmdir", "removedirs", "rename", "renames", "replace", "move", "unlink", "remove"}
)

#: Modules whose functions of those names hit the filesystem.
_FS_MODULES = frozenset({"os", "shutil", "pathlib"})

#: ``(path::function, call-label, reason)`` — why this call can never touch a
#: directory under ``sessions/``. ``<module>`` is module scope. Keep the
#: reasons honest: a reviewer reading a new row should be able to open the
#: line and agree.
_ALLOWED_ROWS: tuple[tuple[str, str, str], ...] = (
    # -- the one legitimate remover -----------------------------------------
    (
        "local_operator/session/cleanup.py::remove_session_dir",
        "shutil.rmtree",
        "THE session remover: guarded by the store marker, the config dir, "
        "the hard guards and the cleanup log",
    ),
    # -- agent / team storage (agents/<id>/, teams/<name>/), never sessions/ --
    ("local_operator/agents.py::AgentRegistry.delete_agent", "shutil.rmtree", "agents/<id>"),
    (
        "local_operator/agents.py::AgentRegistry.save_agent",
        "shutil.rmtree",
        "agents/<id> rollback of a failed save",
    ),
    (
        "local_operator/agents.py::AgentRegistry.import_agent",
        "shutil.rmtree",
        "agents/<id> rollback, failed import",
    ),
    (
        "local_operator/agents.py::AgentRegistry.migrate_agents_dir",
        "shutil.rmtree",
        "legacy agents/ layout",
    ),
    (
        "local_operator/agents.py::AgentRegistry.export_agent_archive",
        "shutil.rmtree",
        "mkdtemp staging dir",
    ),
    (
        "local_operator/agents.py::AgentRegistry.exported_agent_archive",
        "shutil.rmtree",
        "mkdtemp staging dir",
    ),
    ("local_operator/teams.py::_atomic_write_text", "os.replace", "temp FILE -> teams/ json"),
    ("local_operator/teams.py::_atomic_write_text", "<path>.unlink", "temp FILE -> teams/ json"),
    (
        "local_operator/teams.py::TeamRegistry._save_team_locked",
        "os.replace",
        "teams/<name> staging swap",
    ),
    (
        "local_operator/teams.py::TeamRegistry._save_team_locked",
        "shutil.rmtree",
        "teams/<name> staging swap",
    ),
    (
        "local_operator/teams.py::TeamRegistry._swap_row_directory_locked",
        "<path>.rmdir",
        "teams/<name> staging swap",
    ),
    (
        "local_operator/teams.py::TeamRegistry._swap_row_directory_locked",
        "shutil.rmtree",
        "teams/<name> staging swap",
    ),
    (
        "local_operator/teams.py::TeamRegistry._swap_row_directory_locked",
        "os.replace",
        "teams/<name> staging swap",
    ),
    (
        "local_operator/teams.py::TeamRegistry._recover_interrupted_swap_locked",
        "shutil.rmtree",
        "teams/<name> staging swap recovery",
    ),
    (
        "local_operator/teams.py::TeamRegistry._recover_interrupted_swap_locked",
        "os.replace",
        "teams/<name> staging swap recovery",
    ),
    ("local_operator/teams.py::TeamRegistry.delete_team", "shutil.rmtree", "teams/<name>"),
    (
        "local_operator/server/routes/transcription.py::create_transcription_endpoint",
        "shutil.rmtree",
        "mkdtemp upload dir",
    ),
    # -- file-level atomic writes: temp FILE -> its final FILE name ---------
    # These write a file that may live INSIDE a session directory (transcript,
    # roster sidecar, inbox, lease, origin/title sidecars) but never move or
    # remove the directory itself, and os.replace of a file onto a directory
    # fails with EISDIR/ENOTDIR by construction.
    (
        "local_operator/config.py::ConfigManager._handle_bad_config",
        "<path>.replace",
        "config.yml -> .bad backup FILE",
    ),
    (
        "local_operator/config.py::ConfigManager._write_config",
        "os.replace",
        "temp FILE -> config.yml",
    ),
    (
        "local_operator/config.py::ConfigManager._write_config",
        "os.unlink",
        "temp FILE -> config.yml",
    ),
    (
        "local_operator/credentials.py::CredentialManager.write_to_file",
        "os.replace",
        "temp FILE -> .env",
    ),
    (
        "local_operator/credentials.py::CredentialManager.write_to_file",
        "os.unlink",
        "temp FILE -> .env",
    ),
    ("local_operator/browser_bridge/daemon.py::_private_write", "os.replace", "temp FILE"),
    ("local_operator/browser_bridge/state.py::publish", "os.replace", "temp FILE"),
    ("local_operator/browser_bridge/state.py::publish", "os.unlink", "temp FILE"),
    ("local_operator/evaluation/adapters/supervisor.py::persist_rescue", "os.replace", "temp FILE"),
    ("local_operator/evaluation/adapters/supervisor.py::persist_rescue", "os.unlink", "temp FILE"),
    (
        "local_operator/evaluation/evidence/store.py::EvidenceWriter._write_state",
        "os.rename",
        "temp FILE -> state, dir_fd-bound to an evidence root",
    ),
    ("local_operator/mcp/config.py::_write_json_atomic", "os.replace", "temp FILE -> mcp json"),
    ("local_operator/mcp/config.py::_write_json_atomic", "os.unlink", "temp FILE -> mcp json"),
    ("local_operator/mobile/seen.py::SeenStore._persist_locked", "os.replace", "temp FILE"),
    ("local_operator/mobile/seen.py::SeenStore._persist_locked", "os.unlink", "temp FILE"),
    (
        "local_operator/multiplexer/markers.py::_FileBackend.publish",
        "os.replace",
        "temp FILE -> pane marker",
    ),
    (
        "local_operator/skills/index.py::SkillIndex._persist_cache",
        "os.replace",
        "temp FILEs -> index cache",
    ),
    ("local_operator/tools/spill.py::_atomic_write_bytes", "os.replace", "temp FILE -> spill"),
    ("local_operator/tools/spill.py::_atomic_write_bytes", "<path>.unlink", "temp FILE -> spill"),
    (
        "local_operator/tunnels/config.py::private_write",
        "<path>.unlink",
        "temp FILE -> tunnels/ config",
    ),
    (
        "local_operator/tunnels/config.py::private_write",
        "os.replace",
        "temp FILE -> tunnels/ config",
    ),
    ("local_operator/wakes/store.py::write_entry", "os.replace", "temp FILE -> wakes/<id>.json"),
    ("local_operator/wakes/store.py::write_entry", "os.unlink", "temp FILE -> wakes/<id>.json"),
    (
        "local_operator/session/runtime/registry.py::publish",
        "os.replace",
        "temp FILE -> runtime/<pid>.json",
    ),
    (
        "local_operator/session/runtime/registry.py::publish",
        "os.unlink",
        "temp FILE -> runtime/<pid>.json",
    ),
    (
        "local_operator/session/runtime/inbox.py::_replace_remainder",
        "os.replace",
        "temp FILE -> inbox.jsonl inside the same session directory",
    ),
    (
        "local_operator/session/runtime/inbox.py::_replace_remainder",
        "<path>.unlink",
        "temp FILE -> inbox.jsonl inside the same session directory",
    ),
    (
        "local_operator/session/session.py::_write_roster_sidecar",
        "os.replace",
        "temp FILE -> roster sidecar inside the same session directory",
    ),
    (
        "local_operator/session/session.py::_write_roster_sidecar",
        "<path>.unlink",
        "temp FILE -> roster sidecar inside the same session directory",
    ),
    (
        "local_operator/session/transcript.py::Transcript._replace_file",
        "os.replace",
        "compaction temp FILE -> transcript.jsonl inside the same session directory",
    ),
    (
        "local_operator/session_lease.py::acquire_session_lease",
        "os.replace",
        "stale lease FILE -> tombstone FILE inside the same session directory",
    ),
    (
        "local_operator/session_lease.py::acquire_session_lease",
        "<path>.unlink",
        "stale lease FILE -> tombstone FILE inside the same session directory",
    ),
    # -- in-memory .replace(), not the filesystem ---------------------------
    (
        "local_operator/session/remote.py::RemoteSession._install_frontend",
        "<path>.replace",
        "store facade .replace()",
    ),
    (
        "local_operator/session/remote.py::RemoteSession._apply_frontend_facades",
        "<path>.replace",
        "facade .replace() on in-memory state",
    ),
    (
        "local_operator/evaluation/runner/provider_client.py::ProviderModelClient._maybe_compact",
        "<path>.replace",
        "in-memory context .replace()",
    ),
    (
        "local_operator/evaluation/runner/provider_client.py"
        "::ProviderModelClient._shed_stale_turns",
        "<path>.replace",
        "in-memory context .replace()",
    ),
    # -- unlink/remove of FILES the same function owns (locks, caches, sidecars,
    #    temp files, install artefacts). A session directory is never the arg.
    (
        "local_operator/browser_bridge/daemon.py::BridgeService._try_pair",
        "<path>.unlink",
        "pending-pair FILE",
    ),
    ("local_operator/browser_bridge/daemon.py::reset_pairing", "<path>.unlink", "pairing FILE"),
    (
        "local_operator/browser_bridge/install.py::uninstall",
        "<path>.unlink",
        "plist/unit FILEs; state_store.remove()",
    ),
    (
        "local_operator/browser_bridge/install.py::uninstall",
        "<path>.remove",
        "plist/unit FILEs; state_store.remove()",
    ),
    ("local_operator/browser_bridge/state.py::remove", "<path>.unlink", "bridge state FILE"),
    (
        "local_operator/evaluation/adapters/supervisor.py::discard_rescue",
        "os.unlink",
        "rescue FILE",
    ),
    (
        "local_operator/evaluation/evidence/store.py::_OSCalls.unlink",
        "os.unlink",
        "dir_fd-bound FILE unlink",
    ),
    (
        "local_operator/fork.py::consume_boot_prompt",
        "<path>.unlink",
        "one-shot boot-prompt sidecar FILE",
    ),
    (
        "local_operator/fork.py::consume_fork_boundary",
        "<path>.unlink",
        "one-shot fork-boundary sidecar FILE",
    ),
    ("local_operator/mobile/install.py::uninstall", "<path>.unlink", "plist FILE"),
    (
        "local_operator/model/catalogue.py::_ListingFetchLease.acquire",
        "<path>.unlink",
        "lease FILE under the cache",
    ),
    (
        "local_operator/model/catalogue.py::_ListingFetchLease.release",
        "<path>.unlink",
        "lease FILE under the cache",
    ),
    (
        "local_operator/model/catalogue.py::_write_cache",
        "<path>.replace",
        "temp FILE -> cache FILE",
    ),
    ("local_operator/model/catalogue.py::_write_cache", "<path>.unlink", "temp FILE -> cache FILE"),
    ("local_operator/model/catalogue.py::invalidate", "<path>.unlink", "catalogue cache FILE"),
    (
        "local_operator/model/catalogue.py::invalidate_documents",
        "<path>.unlink",
        "catalogue cache FILEs",
    ),
    (
        "local_operator/model/catalogue.py::purge_legacy_documents",
        "<path>.unlink",
        "catalogue cache FILEs",
    ),
    (
        "local_operator/model/catalogue.py::purge_stranded_temp_files",
        "<path>.unlink",
        "catalogue temp FILEs",
    ),
    (
        "local_operator/multiplexer/markers.py::_FileBackend.retire",
        "<path>.unlink",
        "pane marker FILE",
    ),
    (
        "local_operator/resume.py::_save_origin_cache",
        "<path>.replace",
        "temp FILE -> origin-verdicts.json",
    ),
    (
        "local_operator/resume.py::_write_origin_scan_sentinel",
        "<path>.replace",
        "temp FILE -> sentinel in a session",
    ),
    (
        "local_operator/resume.py::_write_title_scan_sentinel",
        "<path>.replace",
        "temp FILE -> sentinel in a session",
    ),
    (
        "local_operator/resume.py::write_session_attachment",
        "<path>.replace",
        "temp FILE -> attachment.json",
    ),
    (
        "local_operator/resume.py::write_session_title",
        "<path>.replace",
        "temp FILE -> title.json in a session",
    ),
    (
        "local_operator/session/retention.py::release_session",
        "<path>.unlink",
        ".session.pid marker FILE",
    ),
    (
        "local_operator/session/runtime/registry.py::scan",
        "<path>.unlink",
        "stale runtime/<pid>.json FILE",
    ),
    (
        "local_operator/session/runtime/registry.py::unpublish",
        "<path>.unlink",
        "own runtime/<pid>.json FILE",
    ),
    (
        "local_operator/session/search_index.py::_save",
        "<path>.replace",
        "temp FILE -> search index FILE",
    ),
    (
        "local_operator/session/transcript.py::Transcript._write_entries",
        "<path>.unlink",
        "rollback of a half-written rebuild of the transcript FILE it just opened",
    ),
    (
        "local_operator/session_lease.py::SessionLease.release",
        "<path>.unlink",
        "own lease + mirror FILEs",
    ),
    (
        "local_operator/session_lease.py::reap_proven_dead_session_claim",
        "<path>.unlink",
        "a dead owner's lease + mirror FILEs; the directory is kept (QA N1)",
    ),
    (
        "local_operator/tools/group_reaper.py::_rewrite_without_pgid_locked",
        "<path>.unlink",
        "pgid ledger FILE",
    ),
    ("local_operator/tools/group_reaper.py::_safe_unlink", "<path>.unlink", "pgid ledger FILE"),
    ("local_operator/tools/group_reaper.py::kill_own_groups", "<path>.unlink", "pgid ledger FILE"),
    ("local_operator/tools/spill.py::SpillStore._remove", "<path>.unlink", "spill FILE"),
    (
        "local_operator/tui/notifier_app/__init__.py::_build_in_background",
        "<path>.unlink",
        "build marker FILE",
    ),
    (
        "local_operator/tui/notifier_app/__init__.py::_build_in_background._run",
        "<path>.unlink",
        "build marker FILE",
    ),
    ("local_operator/tunnels/cli.py::dispatch", "<path>.unlink", "tunnel pid/state FILEs"),
    ("local_operator/tunnels/install.py::uninstall", "<path>.unlink", "plist FILE"),
    ("local_operator/tunnels/service.py::run", "<path>.unlink", "tunnel pid/state FILEs"),
    ("local_operator/update.py::_write_cache", "<path>.replace", "temp FILE -> update cache FILE"),
    ("local_operator/update.py::_write_cache", "<path>.unlink", "temp FILE -> update cache FILE"),
    ("local_operator/wakes/install.py::uninstall", "<path>.unlink", "plist FILE"),
    ("local_operator/wakes/store.py::remove_entry", "<path>.unlink", "wakes/<id>.json FILE"),
    ("local_operator/web_fetch/service.py::_prune_cache", "<path>.unlink", "fetch cache FILEs"),
    # -- container/in-memory .remove()/.replace(), not the filesystem --------
    (
        "local_operator/browser_bridge/daemon.py::BridgeService.shutdown",
        "<path>.remove",
        "state_store.remove() FILE",
    ),
    (
        "local_operator/config_watch.py::ConfigWatcher.subscribe.unsubscribe",
        "<path>.remove",
        "list.remove(listener)",
    ),
    (
        "local_operator/evaluation/adapters/discovery.py::_verified_imports",
        "<path>.remove",
        "sys.meta_path.remove",
    ),
    (
        "local_operator/session/frontend_state.py::FrontendStateStore.checkpoint",
        "<path>.replace",
        "in-memory .replace",
    ),
    (
        "local_operator/session/frontend_state.py::FrontendStateStore.subscribe.unsubscribe",
        "<path>.remove",
        "list.remove(listener)",
    ),
    (
        "local_operator/session/frontend_state.py::SnapshotJobs.__init__",
        "<path>.replace",
        "in-memory .replace",
    ),
    (
        "local_operator/session/frontend_state.py::SnapshotMcpManager.__init__",
        "<path>.replace",
        "in-memory .replace",
    ),
    (
        "local_operator/session/frontend_state.py::SnapshotSubagentComms.__init__",
        "<path>.replace",
        "in-memory .replace",
    ),
    (
        "local_operator/session/frontend_state.py::SnapshotWakeScheduler.__init__",
        "<path>.replace",
        "in-memory .replace",
    ),
    (
        "local_operator/session/remote.py::RemoteSession.subscribe.unsubscribe",
        "<path>.remove",
        "list.remove",
    ),
    (
        "local_operator/session/session.py::Session.subscribe._unsubscribe",
        "<path>.remove",
        "list.remove",
    ),
    (
        "local_operator/session/session.py::Session.subscribe_presentation.unsubscribe",
        "<path>.remove",
        "list.remove",
    ),
    (
        "local_operator/session/session.py::Session.subscribe_rejected_steering.unsubscribe",
        "<path>.remove",
        "list.remove",
    ),
    (
        "local_operator/session/session.py::_paired_prefix",
        "<path>.remove",
        "set.remove(tool_call_id)",
    ),
    (
        "local_operator/session/transcript.py::Transcript.subscribe_admitted_commands.unsubscribe",
        "<path>.remove",
        "list.remove",
    ),
    (
        "local_operator/tui/app.py::OperatorApp._close_org_chart_view",
        "<path>.remove",
        "widget.remove()",
    ),
    (
        "local_operator/tui/app.py::OperatorApp._close_settings_view",
        "<path>.remove",
        "widget.remove()",
    ),
    (
        "local_operator/tui/app.py::OperatorApp._close_subagent_view",
        "<path>.remove",
        "widget.remove()",
    ),
    (
        "local_operator/tui/app.py::OperatorApp._recall_queued_steers",
        "<path>.remove",
        "widget.remove() / list.remove",
    ),
    ("local_operator/tui/app.py::OperatorApp._unmount_prompt", "<path>.remove", "widget.remove()"),
    (
        "local_operator/tui/widgets/subagent_panel.py::SubagentPanel._sync_rows",
        "<path>.remove",
        "widget.remove()",
    ),
    (
        "local_operator/tui/widgets/transcript.py::TranscriptView.clear_blocks",
        "<path>.remove",
        "widget.remove()",
    ),
    (
        "local_operator/tui/widgets/transcript.py::TranscriptView.remove_block",
        "<path>.remove",
        "widget.remove()",
    ),
)

_ALLOWED: dict[str, str] = {f"{fn}::{label}": reason for fn, label, reason in _ALLOWED_ROWS}


def _qualname_index(tree: ast.Module) -> dict[int, str]:
    """Line → enclosing ``Class.method`` / ``function`` / ``<module>``."""
    owner: dict[int, str] = {}

    def visit(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = [*stack, child.name]
                for line in range(child.lineno, (child.end_lineno or child.lineno) + 1):
                    owner[line] = ".".join(inner)
                visit(child, inner)
            else:
                visit(child, stack)

    visit(tree, [])
    return owner


def _import_aliases(tree: ast.Module) -> dict[str, tuple[str, str | None]]:
    """Local name → ``(module, attribute or None)`` for every import.

    ``import shutil as sh`` → ``sh: ("shutil", None)``;
    ``from shutil import rmtree as rm`` → ``rm: ("shutil", "rmtree")``;
    ``from os import path`` → ``path: ("os", "path")``. Resolving these is
    what makes the guard see ``rm(d)`` and ``sh.rmtree(d)`` as
    ``shutil.rmtree`` — round 1 of review mutation-tested the guard and both
    spellings walked straight through (R1-2, M2/M3).
    """
    aliases: dict[str, tuple[str, str | None]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                aliases[local] = (alias.name.split(".")[0], None)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            for alias in node.names:
                aliases[alias.asname or alias.name] = (top, alias.name)
    return aliases


def _classify(node: ast.Call, aliases: dict[str, tuple[str, str | None]]) -> str | None:
    """The label for a call that could remove or displace a filesystem entry,
    or ``None``.

    BIAS TOWARD FALSE POSITIVES. Any call whose name is in :data:`_NAMES`
    counts unless it is provably the string method: a ``.replace(a, b)`` with
    two positional arguments and no keywords is ``str.replace`` (``Path.replace``
    takes exactly one), and a ``.remove(x)`` on a receiver that is a literal
    list/set/dict is a container method. Everything else — a variable
    receiver (``target.rmdir()``), a call receiver (``Path(...).unlink()``),
    an attribute chain (``self.path.rename(x)``), a bare name resolved through
    an import alias — is reported and resolved through :data:`_ALLOWED` with
    a reason. A guard that skipped the commonest shape in this codebase
    (``session_dir = config_dir / "sessions" / id; session_dir.rmdir()``) was
    the round-1 finding (R1-2, M4/M5/M7).
    """
    func = node.func
    positional = len(node.args)
    keywords = {kw.arg for kw in node.keywords}
    if isinstance(func, ast.Name):
        # Bare name: only meaningful through an import alias.
        module, attr = aliases.get(func.id, (None, None))
        if module in _FS_MODULES and attr in _NAMES:
            return f"{module}.{attr}"
        return None
    if not isinstance(func, ast.Attribute) or func.attr not in _NAMES:
        return None
    name = func.attr
    receiver = func.value
    if isinstance(receiver, ast.Name):
        module, attr = aliases.get(receiver.id, (None, None))
        if module in _FS_MODULES and attr is None:
            return f"{module}.{name}"  # os.replace / shutil.rmtree / sh.rmtree
        if receiver.id in _FS_MODULES:
            return f"{receiver.id}.{name}"
    # str.replace / str.removeprefix shapes: two positionals, or the
    # ``count`` keyword, are never Path.replace(target).
    if name == "replace" and (positional != 1 or keywords - {"target"}):
        return None
    if name == "rename" and (positional != 1 or keywords - {"target"}):
        return None
    if name == "move" and positional != 2 and "dst" not in keywords:
        return None
    if name == "remove" and isinstance(receiver, (ast.List, ast.Set, ast.Dict, ast.ListComp)):
        return None
    if name in ("rmdir", "unlink") and positional > 0:
        return None  # Path.rmdir()/unlink() take no positional argument
    if name in ("removedirs", "renames") and not isinstance(receiver, ast.Name):
        return None  # os-only functions
    return f"<path>.{name}"


_SHELL_REMOVERS = ("rm ", "rm\t", "rmdir ", "rm -", "unlink ")


def _shell_remover(node: ast.Call) -> str | None:
    """A shell-out whose command text starts with ``rm``/``rmdir``: the M8
    shape (``os.system("rm -rf " + str(dir))``). Only the literal prefix of
    the command is inspected, so ``["git", "rm"]`` is not caught here — the
    point is naming the obvious spelling, not parsing shell."""
    func = node.func
    dotted = None
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        dotted = (func.value.id, func.attr)
    elif isinstance(func, ast.Name):
        dotted = (None, func.id)
    if dotted is None or dotted[1] not in ("system", "run", "call", "check_call", "Popen"):
        return None
    if not node.args:
        return None
    first = node.args[0]
    texts: list[str] = []
    for leaf in ast.walk(first):
        if isinstance(leaf, ast.Constant) and isinstance(leaf.value, str):
            texts.append(leaf.value)
        elif isinstance(leaf, ast.List) and leaf.elts:
            head = leaf.elts[0]
            if isinstance(head, ast.Constant) and isinstance(head.value, str):
                texts.append(head.value + " ")
    if any(t.lstrip().startswith(_SHELL_REMOVERS) or t.strip() in ("rm", "rmdir") for t in texts):
        return f"shell:{dotted[1]}(rm ...)"
    return None


def _call_sites() -> list[tuple[str, int, str, str]]:
    """``(relative path, line, call label, owner)`` for EVERY classified call."""
    found: list[tuple[str, int, str, str]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        owners = _qualname_index(tree)
        aliases = _import_aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            label = _classify(node, aliases) or _shell_remover(node)
            if label is None:
                continue
            found.append((rel, node.lineno, label, owners.get(node.lineno, "<module>")))
    return found


def _offenders() -> list[tuple[str, int, str, str]]:
    """Classified calls NOT allow-listed."""
    return [site for site in _call_sites() if f"{site[0]}::{site[3]}::{site[2]}" not in _ALLOWED]


def test_no_session_directory_removal_outside_cleanup() -> None:
    offenders = _offenders()
    assert not offenders, "\n".join(
        [
            "Calls that can remove/rename/replace a directory, not allow-listed in "
            "tests/unit/session/test_no_session_deletion.py. If the call provably "
            "cannot reach a directory under sessions/, add it to _ALLOWED with the "
            "reason; if it can, it belongs in session/cleanup.py behind the guards.",
            *(
                f"  {rel}:{line}: {call} in {owner}  (key: {rel}::{owner}::{call})"
                for rel, line, call, owner in offenders
            ),
        ]
    )


def test_cleanup_module_is_the_only_directory_remover_near_sessions() -> None:
    """Belt for the allow-list itself: a DIRECTORY remover (``rmtree``/
    ``rmdir``/``removedirs``) under ``local_operator/session/``,
    ``session_factory.py``, ``resume.py`` and ``session_lease.py`` may appear
    in exactly ONE function — the cleanup remover — allow-listed or not."""
    near = {
        *sorted(p.relative_to(ROOT).as_posix() for p in (PACKAGE / "session").rglob("*.py")),
        "local_operator/session_factory.py",
        "local_operator/resume.py",
        "local_operator/session_lease.py",
    }
    hits = sorted(
        f"{rel}::{owner}"
        for rel, _line, label, owner in _call_sites()
        if rel in near and label.rsplit(".", 1)[-1] in ("rmtree", "rmdir", "removedirs")
    )
    assert hits == [f"{CLEANUP_MODULE}::remove_session_dir"], hits


def test_allow_list_is_not_stale() -> None:
    """Every allow-list entry must still name a real call site, so a removed
    call cannot leave a dangling permission behind for the next one."""
    live = {f"{rel}::{owner}::{label}" for rel, _line, label, owner in _call_sites()}
    stale = sorted(set(_ALLOWED) - live)
    assert not stale, f"allow-list entries with no call site any more: {stale}"


_MUTANTS = [
    # (source appended to a real module, label the guard must report)
    ("import shutil\n\ndef _m(d):\n    shutil.rmtree(d)\n", "shutil.rmtree"),
    ("from shutil import rmtree\n\ndef _m(d):\n    rmtree(d)\n", "shutil.rmtree"),
    ("import shutil as sh\n\ndef _m(d):\n    sh.rmtree(d)\n", "shutil.rmtree"),
    ("def _m(cfg):\n    target = cfg / 'sessions' / 'x'\n    target.rmdir()\n", "<path>.rmdir"),
    ("def _m(cfg, t):\n    t.rename(cfg / 'sessions' / 'y')\n", "<path>.rename"),
    (
        "from pathlib import Path\n\ndef _m(cfg):\n    Path(cfg, 'sessions', 'x').rmdir()\n",
        "<path>.rmdir",
    ),
    ("def _m(d):\n    (d / 'transcript.jsonl').unlink()\n", "<path>.unlink"),
    ("import os\n\ndef _m(d):\n    os.remove(d / 'transcript.jsonl')\n", "os.remove"),
    ("from os import replace as swap\n\ndef _m(a, b):\n    swap(a, b)\n", "os.replace"),
    ("import shutil\n\ndef _m(a, b):\n    shutil.move(a, b)\n", "shutil.move"),
    ("import os\n\ndef _m(d):\n    os.system('rm -rf ' + str(d))\n", "shell:system(rm ...)"),
    (
        "import subprocess\n\ndef _m(d):\n    subprocess.run(['rm', '-rf', str(d)])\n",
        "shell:run(rm ...)",
    ),
]


@pytest.mark.parametrize(
    "source,label", _MUTANTS, ids=[m[1] + str(i) for i, m in enumerate(_MUTANTS)]
)
def test_the_guard_names_every_remover_shape(
    source: str, label: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove the guard can fail, per shape — the review round-1 mutants
    (R1-2) plus ``unlink``/``os.remove``/aliased ``os.replace``/``shutil.move``.
    A copy of the package is NOT made; instead the classifier is run over a
    synthetic module the way ``_call_sites`` runs it over a real one."""
    tree = ast.parse(source, filename="mutant.py")
    aliases = _import_aliases(tree)
    labels = [
        _classify(node, aliases) or _shell_remover(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    assert label in labels, (source, labels)


@pytest.mark.parametrize(
    "source",
    [
        "x = 'a-b'.replace('-', '_')\n",
        "def f(s, a, b):\n    return s.replace(a, b)\n",
        "def f(s):\n    return s.replace('a', 'b', 1)\n",
        "[1, 2].remove(1)\n",
        "def f(lst, v):\n    {1, 2}.remove(v)\n",
    ],
)
def test_the_guard_ignores_string_and_container_methods(source: str) -> None:
    tree = ast.parse(source)
    aliases = _import_aliases(tree)
    labels = [_classify(n, aliases) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert all(label is None for label in labels), labels


@pytest.mark.parametrize("key", sorted(_ALLOWED))
def test_every_allowed_entry_has_a_reason(key: str) -> None:
    assert _ALLOWED[key].strip(), key

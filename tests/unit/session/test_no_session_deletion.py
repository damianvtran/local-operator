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
to one of :data:`_REMOVERS` or :data:`_MOVERS` is checked against
:data:`_ALLOWED`, an explicit allow-list keyed by ``relative/path.py::
function`` carrying the reason the call cannot reach a session directory.
Anything not in the list fails with its ``file:line``. Adding a call site
therefore means adding a line HERE with a reason a reviewer can check — the
point is not that the list is short, it is that every entry was argued for.

Two heuristics keep ``str.replace`` and editor ``replace`` out of the way:
a bare ``.replace(...)``/``.rename(...)`` method call counts only with exactly
ONE positional argument and no keywords (``Path.replace(target)``), which is
never the string method's shape. ``os.*`` and ``shutil.*`` forms always count.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "local_operator"

#: The one module permitted to ``rmtree`` a session directory.
CLEANUP_MODULE = "local_operator/session/cleanup.py"

_REMOVERS = {("shutil", "rmtree"), ("os", "rmdir"), ("os", "removedirs")}
_MOVERS = {("shutil", "move"), ("os", "rename"), ("os", "replace"), ("os", "renames")}
_METHODS = {"rmdir", "rename", "replace"}

#: ``path::function`` → why this call can never touch a directory under
#: ``sessions/``. ``<module>`` is module scope. Keep the reasons honest: a
#: reviewer reading a new entry should be able to open the line and agree.
_ALLOWED: dict[str, str] = {
    # -- the one legitimate remover -----------------------------------------
    f"{CLEANUP_MODULE}::remove_session_dir": (
        "THE session remover: guarded by the store marker, the config dir, "
        "the hard guards and the cleanup log"
    ),
    # -- agent / team storage (agents/<id>/, teams/<name>/), never sessions/ --
    "local_operator/agents.py::AgentRegistry.delete_agent": "agents/<id>",
    "local_operator/agents.py::AgentRegistry.save_agent": "agents/<id> rollback of a failed save",
    "local_operator/agents.py::AgentRegistry.import_agent": "agents/<id> rollback, failed import",
    "local_operator/agents.py::AgentRegistry.migrate_agents_dir": "legacy agents/ layout",
    "local_operator/agents.py::AgentRegistry.export_agent_archive": "mkdtemp staging dir",
    "local_operator/agents.py::AgentRegistry.exported_agent_archive": "mkdtemp staging dir",
    "local_operator/teams.py::_atomic_write_text": "temp FILE -> teams/ json",
    "local_operator/teams.py::TeamRegistry._save_team_locked": "teams/<name> staging swap",
    "local_operator/teams.py::TeamRegistry._swap_row_directory_locked": "teams/<name> staging swap",
    "local_operator/teams.py::TeamRegistry._recover_interrupted_swap_locked": (
        "teams/<name> staging swap recovery"
    ),
    "local_operator/teams.py::TeamRegistry.delete_team": "teams/<name>",
    "local_operator/server/routes/transcription.py::create_transcription_endpoint": (
        "mkdtemp upload dir"
    ),
    # -- file-level atomic writes: temp FILE -> its final FILE name ---------
    # These write a file that may live INSIDE a session directory (transcript,
    # roster sidecar, inbox, lease, origin/title sidecars) but never move or
    # remove the directory itself, and os.replace of a file onto a directory
    # fails with EISDIR/ENOTDIR by construction.
    "local_operator/config.py::ConfigManager._handle_bad_config": "config.yml -> .bad backup FILE",
    "local_operator/config.py::ConfigManager._write_config": "temp FILE -> config.yml",
    "local_operator/credentials.py::CredentialManager.write_to_file": "temp FILE -> .env",
    "local_operator/browser_bridge/daemon.py::_private_write": "temp FILE",
    "local_operator/browser_bridge/state.py::publish": "temp FILE",
    "local_operator/evaluation/adapters/supervisor.py::persist_rescue": "temp FILE",
    "local_operator/evaluation/evidence/store.py::EvidenceWriter._write_state": (
        "temp FILE -> state, dir_fd-bound to an evidence root"
    ),
    "local_operator/mcp/config.py::_write_json_atomic": "temp FILE -> mcp json",
    "local_operator/mobile/seen.py::SeenStore._persist_locked": "temp FILE",
    "local_operator/multiplexer/markers.py::_FileBackend.publish": "temp FILE -> pane marker",
    "local_operator/skills/index.py::SkillIndex._persist_cache": "temp FILEs -> index cache",
    "local_operator/tools/spill.py::_atomic_write_bytes": "temp FILE -> spill",
    "local_operator/wakes/store.py::write_entry": "temp FILE -> wakes/<id>.json",
    "local_operator/session/runtime/registry.py::publish": "temp FILE -> runtime/<pid>.json",
    "local_operator/session/runtime/inbox.py::_replace_remainder": (
        "temp FILE -> inbox.jsonl inside the same session directory"
    ),
    "local_operator/session/session.py::_write_roster_sidecar": (
        "temp FILE -> roster sidecar inside the same session directory"
    ),
    "local_operator/session/transcript.py::Transcript._replace_file": (
        "compaction temp FILE -> transcript.jsonl inside the same session directory"
    ),
    "local_operator/session_lease.py::acquire_session_lease": (
        "stale lease FILE -> tombstone FILE inside the same session directory"
    ),
    # -- in-memory .replace(), not the filesystem ---------------------------
    "local_operator/session/remote.py::RemoteSession._install_frontend": "store facade .replace()",
    "local_operator/session/remote.py::RemoteSession._apply_frontend_facades": (
        "facade .replace() on in-memory state"
    ),
    "local_operator/evaluation/runner/provider_client.py::ProviderModelClient._maybe_compact": (
        "in-memory context .replace()"
    ),
    "local_operator/evaluation/runner/provider_client.py::ProviderModelClient._shed_stale_turns": (
        "in-memory context .replace()"
    ),
}


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


def _dotted(node: ast.AST) -> tuple[str, str] | None:
    """``os.replace`` → ``("os", "replace")`` for a two-part attribute call."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id, node.attr
    return None


def _offenders() -> list[tuple[str, int, str, str]]:
    """``(relative path, line, call, owner)`` for every call NOT allow-listed."""
    found: list[tuple[str, int, str, str]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        owners = _qualname_index(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            dotted = _dotted(func)
            label: str | None = None
            if dotted in _REMOVERS or dotted in _MOVERS:
                label = ".".join(dotted)  # type: ignore[arg-type]
            elif (
                isinstance(func, ast.Attribute)
                and func.attr in _METHODS
                and dotted is None
                and not node.keywords
                and (func.attr == "rmdir" and not node.args or len(node.args) == 1)
            ):
                label = f"<path>.{func.attr}"
            if label is None:
                continue
            owner = owners.get(node.lineno, "<module>")
            key = f"{rel}::{owner}"
            if key in _ALLOWED:
                continue
            found.append((rel, node.lineno, label, owner))
    return found


def test_no_session_directory_removal_outside_cleanup() -> None:
    offenders = _offenders()
    assert not offenders, "\n".join(
        [
            "Calls that can remove/rename/replace a directory, not allow-listed in "
            "tests/unit/session/test_no_session_deletion.py. If the call provably "
            "cannot reach a directory under sessions/, add it to _ALLOWED with the "
            "reason; if it can, it belongs in session/cleanup.py behind the guards.",
            *(f"  {rel}:{line}: {call} in {owner}" for rel, line, call, owner in offenders),
        ]
    )


def test_cleanup_module_is_the_only_rmtree_on_sessions() -> None:
    """Belt for the allow-list itself: ``rmtree`` under ``local_operator/session/``,
    ``session_factory.py``, ``resume.py`` and ``session_lease.py`` must appear in
    exactly ONE function — the cleanup remover."""
    hits: list[str] = []
    for rel in [
        *sorted(p.relative_to(ROOT).as_posix() for p in (PACKAGE / "session").rglob("*.py")),
        "local_operator/session_factory.py",
        "local_operator/resume.py",
        "local_operator/session_lease.py",
    ]:
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)
        owners = _qualname_index(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _dotted(node.func) in (_REMOVERS | {("os", "rmdir")}):
                hits.append(f"{rel}::{owners.get(node.lineno, '<module>')}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "rmdir"
            ):
                hits.append(f"{rel}::{owners.get(node.lineno, '<module>')}")
    assert hits == [f"{CLEANUP_MODULE}::remove_session_dir"], hits


def test_allow_list_is_not_stale() -> None:
    """Every allow-list entry must still name a real call site, so a removed
    call cannot leave a dangling permission behind for the next one."""
    live: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        owners = _qualname_index(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                dotted = _dotted(node.func)
                if (
                    dotted in _REMOVERS
                    or dotted in _MOVERS
                    or (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr in _METHODS
                        and dotted is None
                    )
                ):
                    live.add(f"{rel}::{owners.get(node.lineno, '<module>')}")
    stale = sorted(set(_ALLOWED) - live)
    assert not stale, f"allow-list entries with no call site any more: {stale}"


@pytest.mark.parametrize("key", sorted(_ALLOWED))
def test_every_allowed_entry_has_a_reason(key: str) -> None:
    assert _ALLOWED[key].strip(), key

"""Every environment variable production code reads is either scrubbed by the
autouse isolation fixture or allow-listed here with a reason.

WHY. Two incidents in one week had the same shape — a test reaching the real
machine through a variable inherited from the developer's shell despite an
isolated ``HOME``: a QA driver inherited ``LOP_MOBILE_CHILD_RESUME`` from the
operator's own runtime and created that id in a store, and a headless fork
e2e test inherited ``CMUX_WORKSPACE_ID``/``CMUX_SURFACE_ID`` and renamed the
operator's LIVE cmux window (#648). ``tests/conftest.py`` scrubs a list,
``_AMBIENT_VARS``; this test is what keeps that list complete as the code
grows a new ``os.environ.get("...")``.

HOW. Walk ``local_operator/`` by AST for every read of an environment
variable: ``os.environ[...]``/``.get``/``.pop``/``in os.environ``,
``os.getenv``, and ``<mapping>.get(<NAME>)`` where ``<NAME>`` is a module
constant whose value looks like an environment variable — the shape the cmux
helpers use (``env.get(WORKSPACE_ENV)`` with ``env`` defaulting to
``os.environ``). Every name found must be in ``_AMBIENT_VARS`` or in
:data:`_HARMLESS` with a reason a reviewer can check. A name in BOTH is a
defect too (the reason is stale), and so is an allow-list entry nothing reads.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.conftest import _AMBIENT_VARS

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "local_operator"

#: Variables production code reads that CANNOT name a real-machine resource a
#: test could damage or a credential that could change what a test does, with
#: the reason. Terminal identity and platform plumbing are here; anything that
#: names a session, a config dir, a window, a socket, a token or a provider is
#: NOT, and belongs in ``_AMBIENT_VARS``.
_HARMLESS: dict[str, str] = {
    # -- platform / process plumbing ---------------------------------------
    "PATH": "binary lookup; scrubbing it would break every subprocess",
    "SHELL": "which shell to spawn for bash; a name, not a resource",
    "COMSPEC": "Windows shell name",
    "SystemRoot": "Windows system dir",
    "LOCALAPPDATA": "Windows app-data root (HOME analogue; tests redirect HOME)",
    "TMPDIR": "scratch root; pytest's tmp_path already lives under it",
    "USER": "display only (banner / peer messaging labels)",
    "EDITOR": "which editor /edit spawns; tests never spawn one",
    "VISUAL": "which editor /edit spawns; tests never spawn one",
    "PIPX_HOME": "install-location detection for the update hint; read-only",
    "XDG_STATE_HOME": "log dir root; tests redirect HOME and the log dir under it",
    "LOG_LEVEL": "verbosity only",
    "ANONYMIZED_TELEMETRY": "set to 0 for chromadb; never read back",
    # -- terminal capability probes (read-only, cosmetic) --------------------
    "TERM": "colour/capability probe; tests pin it themselves",
    "TERM_PROGRAM": "terminal identity for tab-title / notification routing",
    "NO_COLOR": "colour probe; tests pin it themselves",
    "TMUX": "presence probe for terminal-title routing; never used to address a pane",
    "KITTY_WINDOW_ID": "presence probe for terminal-title routing",
    "ITERM_SESSION_ID": "presence probe for terminal detection; never used to address",
    "WEZTERM_PANE": "presence probe for terminal detection; never used to address",
    "WEZTERM_EXECUTABLE": "binary path probe for terminal detection",
    "GHOSTTY_BIN": "binary path probe for terminal detection",
    "GHOSTTY_RESOURCES_DIR": "presence probe for terminal detection",
    "SSH_TTY": "presence probe (remote session -> no local terminal spawn)",
    "SSH_CONNECTION": "presence probe (remote session -> no local terminal spawn)",
    # -- lop's own cosmetic/limit switches (no resource named) --------------
    "LOCAL_OPERATOR_NO_NERD_ICONS": "glyph set switch",
    "LOCAL_OPERATOR_NO_SHIMMER": "animation switch; the visual harness sets it",
    "LOCAL_OPERATOR_NO_TERMINAL_TITLE": "title-escape switch",
    "LOCAL_OPERATOR_NO_MULTIPLEXER_RESUME": "kill switch for the pane resume marker; safer ON",
    "LOCAL_OPERATOR_NO_NOTIFICATIONS": "desktop-notification kill switch (safer ON)",
    "LOCAL_OPERATOR_IMAGES": "inline-image protocol switch",
    "LOCAL_OPERATOR_GREP_ENGINE": "rg vs python grep",
    "LOCAL_OPERATOR_MCP_TIMEOUT_MS": "a timeout",
    "LOCAL_OPERATOR_SPILL_MAX_BYTES": "a size limit",
    "LOCAL_OPERATOR_SCHEDULED_TASK_TIMEOUT_SECONDS": "a timeout",
    "LOCAL_OPERATOR_CONTEXT_FILES": "extra context file names, read relative to cwd",
    "LOCAL_OPERATOR_SKILL_EXTRA_ROOTS": "extra skill roots; read-only directories",
    "LOP_SESSION_GRACE_S": "runtime residency grace; a duration",
    "LOP_RUNTIME_DEBUG_STACKS": "debug dump switch",
    "LO_MOBILE_NO_DIAL": "disables the mobile dial-out; safer ON",
    # -- evaluation adapter fds: only meaningful inside a spawned adapter ----
    "LO_ADAPTER_LAUNCH_IDENTITY": "adapter child handshake; unset outside the harness",
    "LO_ADAPTER_OWNER_FD": "adapter child fd; unset outside the harness",
    "LO_ADAPTER_REQUEST_FD": "adapter child fd; unset outside the harness",
    "LO_ADAPTER_RESPONSE_FD": "adapter child fd; unset outside the harness",
    # -- provider endpoints (not credentials) -------------------------------
    "KIMI_OAUTH_HOST": "OAuth host override; a URL, no token",
    "KIMI_CODE_OAUTH_HOST": "OAuth host override; a URL, no token",
    "RADIENT_API_BASE_URL": "base URL override; no token",
    "RADIENT_CLIENT_ID": "public OAuth client id; not a secret",
    "CMUX_BUNDLED_CLI_PATH": "where the cmux binary is; a path, not a window",
}

_ENV_SHAPE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$|^SystemRoot$")


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """``NAME = "SOME_ENV_VAR"`` at module scope → ``{NAME: SOME_ENV_VAR}``."""
    found: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and _ENV_SHAPE.match(node.value.value)
        ):
            found[node.targets[0].id] = node.value.value
    return found


def _is_environ(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "environ"


def env_reads() -> dict[str, set[str]]:
    """Variable name → the ``file:line`` sites that read it."""
    # First pass: every env-shaped module constant across the package, so a
    # constant defined in ``terminals.py`` and read in ``spawn/cmux.py``
    # resolves.
    shared: dict[str, str] = {}
    trees: list[tuple[str, ast.Module, dict[str, str]]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        local = _module_constants(tree)
        trees.append((rel, tree, local))
        shared.update(local)

    reads: dict[str, set[str]] = {}

    for rel, tree, local in trees:
        # The defining module wins: three modules each spell ``_ENV_DISABLE``
        # with a different value, and the cross-module map exists only for
        # constants imported from elsewhere (``terminals.CMUX_SURFACE_ENV``).
        def resolve(key: ast.AST, _local: dict[str, str] = local) -> str | None:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                return key.value if _ENV_SHAPE.match(key.value) else None
            if isinstance(key, ast.Name):
                return _local.get(key.id) or shared.get(key.id)
            if isinstance(key, ast.Attribute):
                return _local.get(key.attr) or shared.get(key.attr)
            return None

        def note(name: str | None, node: ast.AST, _rel: str = rel) -> None:
            if name is not None:
                reads.setdefault(name, set()).add(f"{_rel}:{getattr(node, 'lineno', 0)}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if not node.args:
                    continue
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "getenv"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "os"
                ):
                    note(resolve(node.args[0]), node)
                elif isinstance(func, ast.Attribute) and func.attr in ("get", "pop", "setdefault"):
                    if _is_environ(func.value):
                        note(resolve(node.args[0]), node)
                    else:
                        # ``env.get(WORKSPACE_ENV)``: a mapping parameter that
                        # defaults to os.environ. Only a CONSTANT key counts;
                        # a literal here would be a dict lookup, not the env.
                        if isinstance(node.args[0], (ast.Name, ast.Attribute)):
                            note(resolve(node.args[0]), node)
            elif isinstance(node, ast.Subscript) and _is_environ(node.value):
                note(resolve(node.slice), node)
            elif isinstance(node, ast.Compare) and any(_is_environ(c) for c in node.comparators):
                note(resolve(node.left), node)
    return reads


def test_every_environment_variable_read_is_scrubbed_or_explained() -> None:
    reads = env_reads()
    scrubbed = set(_AMBIENT_VARS)
    unaccounted = sorted(name for name in reads if name not in scrubbed and name not in _HARMLESS)
    assert not unaccounted, "\n".join(
        [
            "Production code reads these environment variables and the test suite "
            "neither scrubs them (tests/conftest.py _AMBIENT_VARS) nor explains why "
            "an inherited value is harmless (_HARMLESS in this file). A variable that "
            "names a real-machine resource — a session, a window, a socket, a config "
            "dir, a credential — goes in _AMBIENT_VARS.",
            *(f"  {name}: {', '.join(sorted(reads[name]))}" for name in unaccounted),
        ]
    )


def test_cmux_identity_is_scrubbed() -> None:
    """The #648 incident, pinned by name: a test must never inherit the
    operator's live cmux window."""
    for name in ("CMUX_WORKSPACE_ID", "CMUX_SURFACE_ID", "CMUX_PANEL_ID", "CMUX_TAB_ID"):
        assert name in _AMBIENT_VARS, name


def test_no_variable_is_both_scrubbed_and_called_harmless() -> None:
    both = sorted(set(_AMBIENT_VARS) & set(_HARMLESS))
    assert not both, f"pick one: {both}"


def test_harmless_list_is_not_stale() -> None:
    reads = env_reads()
    stale = sorted(name for name in _HARMLESS if name not in reads)
    assert not stale, f"_HARMLESS names nothing reads any more: {stale}"


@pytest.mark.parametrize("name", sorted(_HARMLESS))
def test_every_harmless_entry_has_a_reason(name: str) -> None:
    assert _HARMLESS[name].strip(), name

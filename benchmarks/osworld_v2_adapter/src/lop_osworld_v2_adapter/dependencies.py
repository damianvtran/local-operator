"""Static packaging checks, only when an AWS episode names its task.

Never import a task or a dependency to discover dependencies: upstream imports
can validate service settings, initialise clients or write caches. This is a
presence check, not a claim that arbitrary Python or every evaluator will run.
The corpus acceptance check separately exercises module loading without setup.
"""

from __future__ import annotations

import ast
import sys
from importlib.machinery import PathFinder
from pathlib import Path

# These are real upstream file namespaces, unlike third-party compatibility
# aliases such as requests.packages. Check their complete module paths so an
# installed desktop_env alone cannot disguise a missing runtime helper.
_RUNTIME_ROOTS = frozenset({"desktop_env", "evaluation_examples"})
_IMPORT_ERRORS = frozenset({"ImportError", "ModuleNotFoundError", "Exception", "BaseException"})


class MissingTaskDependencies(RuntimeError):
    """The isolated interpreter lacks dependencies, before provider allocation."""


def import_census(source: str | bytes) -> tuple[set[str], set[str]]:
    """Return required and guarded-optional imports, including function bodies.

    TYPE_CHECKING bodies are not runtime dependencies. An import in a try body
    that catches import failure is optional by upstream's own contract; scanning
    its fallback still finds any mandatory replacement dependency.
    """

    required: set[str] = set()
    optional: set[str] = set()

    def walk(node: ast.AST, guarded: bool = False) -> None:
        if isinstance(node, ast.If) and (
            isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
            or isinstance(node.test, ast.Attribute)
            and isinstance(node.test.value, ast.Name)
            and node.test.value.id == "typing"
            and node.test.attr == "TYPE_CHECKING"
        ):
            for child in node.orelse:
                walk(child, guarded)
            return
        if isinstance(node, ast.Try):
            catches_import = any(
                handler.type is None
                or any(
                    isinstance(part, ast.Name) and part.id in _IMPORT_ERRORS
                    for part in ast.walk(handler.type)
                )
                for handler in node.handlers
            )
            for child in node.body:
                walk(child, guarded or catches_import)
            for child in [*node.handlers, *node.orelse, *node.finalbody]:
                walk(child, guarded)
            return
        target = optional if guarded else required
        if isinstance(node, ast.Import):
            target.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            target.add(node.module)
        for child in ast.iter_child_nodes(node):
            walk(child, guarded)

    walk(ast.parse(source))
    return required, optional - required


def module_present(name: str) -> bool:
    """Resolve files without executing parent __init__ modules or import hooks.

    Third-party modules may expose submodules dynamically, so only their root
    is a static packaging contract. Upstream runtime namespaces are file-based
    and checked down to the leaf. Normal isolated imports remain authoritative
    for execution (including the worker's RECORD-verified source loader).
    """

    root = name.split(".", 1)[0]
    if root in sys.stdlib_module_names:
        return True
    parts = name.split(".") if root in _RUNTIME_ROOTS else [root]
    search = None
    for index in range(len(parts)):
        spec = PathFinder.find_spec(".".join(parts[: index + 1]), search)
        if spec is None:
            return False
        if index < len(parts) - 1:
            if spec.submodule_search_locations is None:
                return False
            search = list(spec.submodule_search_locations)
    return True


def validate_task_dependencies(source: str | bytes) -> None:
    """Fail before spending on a VM; never execute task setup or import code."""

    required, _ = import_census(source)
    if any(name.split(".", 1)[0] == "evaluation_examples" for name in required):
        # The wheel owns this entire three-file helper package. Parse its
        # closure as well, not only the task that imports it; never traverse
        # third-party source trees or execute their package initialisers.
        helpers = Path(__file__).parent.parent / "evaluation_examples"
        for path in helpers.rglob("*.py"):
            helper_required, _ = import_census(path.read_bytes())
            required.update(helper_required)
    missing = sorted(name for name in required if not module_present(name))
    if missing:
        raise MissingTaskDependencies(
            "isolated OSWorld environment is missing task dependencies: "
            + ", ".join(missing)
            + "; rebuild/install the pinned adapter wheel and locked OSWorld extra"
        )

"""Static packaging checks, only when an AWS episode names its task.

Never import a task or a dependency to discover dependencies: upstream imports
can validate service settings, initialise clients or write caches. This is a
presence check, not a claim that arbitrary Python or every evaluator will run.
The corpus acceptance check separately exercises module loading without setup.
"""

from __future__ import annotations

import ast
import sys
from importlib.machinery import ModuleSpec, PathFinder
from importlib.util import resolve_name
from pathlib import Path

# These are real upstream file namespaces, unlike third-party compatibility
# aliases such as requests.packages. Check their complete module paths so an
# installed desktop_env alone cannot disguise a missing runtime helper.
_RUNTIME_ROOTS = frozenset({"desktop_env", "evaluation_examples"})
_IMPORT_ERRORS = frozenset({"ImportError", "ModuleNotFoundError", "Exception", "BaseException"})


class MissingTaskDependencies(RuntimeError):
    """The isolated interpreter lacks dependencies, before provider allocation."""


def import_census(
    source: str | bytes, *, resolve_runtime_exports: bool = False
) -> tuple[set[str], set[str]]:
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
            if resolve_runtime_exports and node.module.split(".", 1)[0] in _RUNTIME_ROOTS:
                target.update(_from_import_modules(node.module, node.names))
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
    return _module_spec(name if root in _RUNTIME_ROOTS else root) is not None


def _module_spec(name: str) -> ModuleSpec | None:
    parts = name.split(".")
    search = None
    spec = None
    for index in range(len(parts)):
        spec = PathFinder.find_spec(".".join(parts[: index + 1]), search)
        if spec is None:
            return None
        if index < len(parts) - 1:
            if spec.submodule_search_locations is None:
                return None
            search = list(spec.submodule_search_locations)
    return spec


def _from_import_modules(package: str, names: list[ast.alias]) -> set[str]:
    """Resolve pinned package exports without executing their initializers.

    ``from evaluators import metrics`` needs a child module even when its file
    is missing. Conversely, ``from getters import get_vm_file`` is an explicit
    re-export from ``getters.file``, not a fictional ``getters.get_vm_file``
    module. Only upstream's file-based namespaces use this rule; dynamic
    third-party compatibility exports retain the import-root presence check.
    """

    spec = _module_spec(package)
    if spec is None or spec.submodule_search_locations is None:
        return set()
    exports: dict[str, str | None] = {}
    if spec.origin is not None and spec.origin.endswith(".py"):
        for node in ast.parse(Path(spec.origin).read_bytes()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                exports[node.name] = None
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for part in ast.walk(target):
                        if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store):
                            exports[part.id] = None
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    exports[alias.asname or alias.name.split(".")[0]] = alias.name
            elif isinstance(node, ast.ImportFrom):
                origin = resolve_name("." * node.level + (node.module or ""), package)
                for alias in node.names:
                    exports[alias.asname or alias.name] = (
                        origin if node.module else f"{origin}.{alias.name}"
                    )
    modules: set[str] = set()
    for alias in names:
        if alias.name == "*":
            continue
        dependency = exports.get(alias.name, f"{package}.{alias.name}")
        if dependency is not None:
            modules.add(dependency)
    return modules


def validate_task_dependencies(source: str | bytes) -> None:
    """Fail before spending on a VM; never execute task setup or import code."""

    required, _ = import_census(source, resolve_runtime_exports=True)
    if any(name.split(".", 1)[0] == "evaluation_examples" for name in required):
        # The wheel owns this entire three-file helper package. Parse its
        # closure as well, not only the task that imports it; never traverse
        # third-party source trees or execute their package initialisers.
        helpers = Path(__file__).parent.parent / "evaluation_examples"
        for path in helpers.rglob("*.py"):
            helper_required, _ = import_census(path.read_bytes(), resolve_runtime_exports=True)
            required.update(helper_required)
    missing = sorted(name for name in required if not module_present(name))
    if missing:
        raise MissingTaskDependencies(
            "isolated OSWorld environment is missing task dependencies: "
            + ", ".join(missing)
            + "; rebuild/install the pinned adapter wheel and locked OSWorld extra"
        )

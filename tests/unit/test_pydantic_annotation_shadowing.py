"""Guard Pydantic models against annotations that resolve a shadowed builtin.

Pydantic evaluates a model's field annotations *after* the class body has run,
against the class namespace. On Python 3.14+ that evaluation is deferred
(PEP 649), so a class that defines ``def dict(...)`` and also annotates a field
with ``dict[str, Any]`` resolves ``dict`` to the method and fails at import with
``TypeError: 'function' object is not subscriptable``.

v0.77.0 shipped that combination in ``clients/radient.py`` and ``lop serve``
could not start. PR CI runs Python 3.12, where class-body annotations are
evaluated eagerly and the defective source imports fine, so a runtime import
test cannot catch the regression there. This static guard runs on every
supported version and fails on the pre-fix source.

The rule is deliberately narrow: only class-level field annotations (the ones
Pydantic evaluates) and only the generic builtins that can be subscripted. A
method named ``dict`` is still allowed -- renaming it would be a wider API
compatibility question -- as long as no field annotation references ``dict``.
"""

import ast
from pathlib import Path
from typing import Iterator, Set

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "local_operator"
SHADOWABLE_BUILTINS = frozenset({"dict", "list", "set", "tuple", "type", "frozenset"})


def _is_pydantic_model(node: ast.ClassDef) -> bool:
    """Return whether a class looks like a Pydantic model.

    The package has hundreds of models and importing all of them to inspect
    ``BaseModel.__subclasses__`` would drag the whole application (and its
    optional dependencies) into a source-level guard. ``model_config`` or a
    ``Field(`` call in the class body, or a model-shaped base, identifies the
    classes whose annotations Pydantic evaluates.
    """
    base_names = {getattr(base, "id", None) or getattr(base, "attr", "") for base in node.bases}
    if any("Model" in name or "Settings" in name for name in base_names):
        return True
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign):
            target = statement.target
            if isinstance(target, ast.Name) and target.id == "model_config":
                return True
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "model_config"
            for target in statement.targets
        ):
            return True
    return any(
        isinstance(call, ast.Call)
        and (
            isinstance(call.func, ast.Name)
            and call.func.id == "Field"
            or isinstance(call.func, ast.Attribute)
            and call.func.attr == "Field"
        )
        for call in ast.walk(node)
    )


def _annotation_names(annotation: ast.expr) -> Set[str]:
    """Collect every bare name referenced by an annotation node."""
    names = {node.id for node in ast.walk(annotation) if isinstance(node, ast.Name)}
    for node in ast.walk(annotation):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                names |= _annotation_names(ast.parse(node.value, mode="eval").body)
            except SyntaxError:
                continue
    return names


def _models(path: Path) -> Iterator[ast.ClassDef]:
    """Yield every Pydantic-model class defined in a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _is_pydantic_model(node):
            yield node


def test_no_pydantic_model_annotates_a_builtin_it_shadows() -> None:
    """No model field may reference a builtin the same class body defines."""
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for model in _models(path):
            shadowed = {
                statement.name
                for statement in model.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name in SHADOWABLE_BUILTINS
            }
            if not shadowed:
                continue
            for statement in model.body:
                if not isinstance(statement, ast.AnnAssign):
                    continue
                clashes = _annotation_names(statement.annotation) & shadowed
                if clashes:
                    offenders.append(
                        f"{path.relative_to(PACKAGE_ROOT.parent)}:{model.lineno} "
                        f"{model.name}: annotation on line {statement.lineno} "
                        f"references shadowed {sorted(clashes)}"
                    )

    assert not offenders, (
        "Pydantic evaluates these annotations after the class body, so the "
        "shadowed builtin resolves to the method and import fails on Python "
        "3.14+ (PEP 649). Use typing generics such as List/Dict instead:\n" + "\n".join(offenders)
    )

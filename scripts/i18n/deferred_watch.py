"""The deferred-namespace census: which files hold a wire namespace's copy.

A namespace in `local_operator.i18n.catalogues.DEFERRED_NAMESPACES` is not in
the eager catalogue bundle, so everything that renders it must read the
catalogue in-process (the bundle is not involved). The policy is only safe
while no consumer SENDS A KEY of a deferred namespace across the boundary: a
client resolving `wire.x.some_key` that never arrives renders English forever.

`tests/unit/i18n/test_deferred_references.py` ratchets this census against
`local_operator/i18n/deferred_wire_inventory.json`, and this module is the
ONE counting rule both read — a census written by one implementation and
checked by another drifts silently.

WHAT COUNTS AS A REFERENCE (text, per file — the same shape as the existing
i18n baseline in `i18n/baseline.json`, because two counting rules would
disagree):

* the quoted message-id prefix, `"wire.slash."` / `'wire.slash.'`;
* the quoted namespace itself, `"wire.exec"` — the `tr("wire.exec")` resolver
  form every render-point module binds once;
* the key module identifier (`wire_exec`), so an import counts once;
* for each `from ...i18n.keys import wire_x as alias` in the file, uses of
  that alias (`alias.`) — how most render points call their keys.

The GENERATED key modules (`local_operator/i18n/keys/`) are excluded: they are
mechanically the catalogue (one `Msg("<namespace>.<key>")` per message),
enforced byte-for-byte by `generate.py --check`. Counting them would conflate
"a key was added" with "a screen renders it", and only the second is the event
the guard exists to catch.

BLIND SPOTS, stated here so a future reviewer does not read this census as
more than it is:

* it counts REFERENCES, not what they do — a key called and returned as an
  envelope still counts once. That gap is closed by the separate AST check
  (`unwrapped_key_calls`), which requires every call of a deferred key to sit
  inside a render wrapper (`_t`, `_desc`, `render`, ...);
* it reads TEXT, so a key module reached through a dynamic name
  (`getattr(keys, name)()`) or a name concatenated at runtime is invisible;
* it scans `*.py` under the package directory only — a render point in a
  template, a JS client, a generated file or another repository is out of view.

`--write` refreshes the recorded counts in the inventory (reasons and verified
render points are review data and are left alone); `--check` reports a live
census that differs without writing.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
# Scripts under scripts/ read the tree they live in regardless of which venv
# launched them (a console script's sys.path[0] is .venv/bin, not the cwd).
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_operator.i18n import catalogues  # noqa: E402

PACKAGE = REPO / "local_operator"
INVENTORY = REPO / "local_operator" / "i18n" / "deferred_wire_inventory.json"

#: The generated key modules: mechanically the catalogue, not render points.
GENERATED_KEY_MODULES = Path("i18n") / "keys"

#: The policy module itself: its dict keys name every deferred namespace, and
#: its reasons quote render-point calls. Counting it would make the census
#: partly a record of itself.
POLICY_MODULE = Path("i18n") / "catalogues.py"

_KEYS_IMPORT = re.compile(
    r"^from\s+local_operator\.i18n\.keys\s+import\s+(?:\(([^)]*)\)|([^\n]+))", re.MULTILINE
)


def key_module(namespace: str) -> str:
    """The generated key module's name for one ``wire.*`` namespace."""
    return "wire_" + namespace[len("wire.") :].replace(".", "_")


def _aliases(text: str, module: str) -> set[str]:
    """Local names a file binds to ``module`` (the module name itself or an alias)."""
    return set(_all_aliases(text, {module}))


def _longer_key_modules(known: set[str], module: str) -> tuple[str, ...]:
    """Key modules whose name starts with ``module`` (its children).

    ``wire_settings_account_policy`` contains ``wire_settings`` as a substring,
    and a plain count would charge the parent for a child's render point. The
    caller passes every module it knows — the namespace set it was asked about
    plus whatever the key directory holds — so a scratch tree with no generated
    key directory still gets the exclusion right.
    """
    return tuple(sorted(stem for stem in known if stem.startswith(module) and stem != module))


def _text_count(
    text: str, namespace: str, children: tuple[str, ...], longer: tuple[str, ...]
) -> int:
    """References to ``namespace`` in one file, by the rule in the module docstring."""
    # Quotes are part of every id pattern: a bare mention in prose does not
    # count, and the identifiers are always written quoted.
    if children:
        # A nested deferred namespace owns its keys; do not count them here.
        skip = "|".join(re.escape(child[len(namespace) + 1 :]) for child in children)
        count = len(re.findall(rf"[\"']{re.escape(namespace)}\.(?!({skip})[.\"'])", text))
    else:
        count = text.count(f'"{namespace}.')
        count += text.count(f"'{namespace}.'")
    count += text.count(f'"{namespace}"') + text.count(f"'{namespace}'")
    module = key_module(namespace)
    count += text.count(module) - sum(text.count(other) for other in longer)
    for alias in _aliases(text, module):
        if alias != module:
            count += len(re.findall(rf"\b{re.escape(alias)}\.", text))
    return count


def census(
    root: Path = PACKAGE, namespaces: tuple[str, ...] | None = None
) -> dict[str, dict[str, int]]:
    """Per-namespace ``{path relative to root: reference count}`` over ``*.py``.

    ``root`` is the package directory (``.../local_operator``). It is an
    argument so the guard test can point it at a scratch tree and count there
    without touching the checkout.
    """
    namespaces = namespaces or tuple(catalogues.DEFERRED_NAMESPACES)
    counts: dict[str, dict[str, int]] = {namespace: {} for namespace in namespaces}
    known = {key_module(namespace) for namespace in namespaces}
    known.update(path.stem for path in (root / GENERATED_KEY_MODULES).glob("*.py"))
    longer = {
        namespace: _longer_key_modules(known, key_module(namespace)) for namespace in namespaces
    }
    generated = root / GENERATED_KEY_MODULES
    policy = root / POLICY_MODULE
    for path in sorted(root.rglob("*.py")):
        if generated in path.parents or path == policy:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(root).as_posix()
        for namespace in namespaces:
            # Cheap C-speed prefilter first: the counting regexes are ~10x
            # slower per megabyte and most (file, namespace) pairs have no
            # reference (`tui/app.py` alone is 3.4 MB).
            if (
                key_module(namespace) not in text
                and f'"{namespace}' not in text
                and f"'{namespace}" not in text
            ):
                continue
            children = tuple(
                other
                for other in namespaces
                if other.startswith(namespace + ".") and other != namespace
            )
            found = _text_count(text, namespace, children, longer[namespace])
            if found:
                counts[namespace][relative] = found
    return counts


def compare(
    recorded: dict[str, dict[str, int]], live: dict[str, dict[str, int]]
) -> list[tuple[str, str, int, int]]:
    """Every (namespace, path, recorded count, live count) that differs."""
    rows: list[tuple[str, str, int, int]] = []
    for namespace in sorted(set(recorded) | set(live)):
        baseline = recorded.get(namespace, {})
        current = live.get(namespace, {})
        for path in sorted(set(baseline) | set(current)):
            was, now = baseline.get(path, 0), current.get(path, 0)
            if was != now:
                rows.append((namespace, path, was, now))
    return rows


def new_references(
    recorded: dict[str, dict[str, int]], live: dict[str, dict[str, int]]
) -> list[str]:
    """References the ratchet forbids: a count that rose, or an unlisted file.

    A FALL is allowed — a render point moved out or was deleted. A rise is a
    new render point until a reviewer records it in the inventory as one more
    plain-string consumer.
    """
    return [
        f"{namespace}: {path} ({was} -> {now})"
        for namespace, path, was, now in compare(recorded, live)
        if now > was
    ]


#: Wrappers that turn a ``Msg`` into a rendered string. A deferred key call
#: must sit inside one of these; returning the ``Msg`` for a caller to render
#: is fine too (the reference is then not a call and never reaches this check).
RENDER_WRAPPERS = frozenset({"_t", "_tr", "_desc", "render", "describe", "tr"})


def _all_aliases(text: str, modules: set[str]) -> dict[str, str]:
    """Local name -> key-module name, for imports of any of ``modules``."""
    found: dict[str, str] = {}
    for match in _KEYS_IMPORT.finditer(text):
        listing = (match.group(1) or match.group(2)).replace("(", " ").replace(")", " ")
        for part in listing.split(","):
            fields = part.split()
            if len(fields) >= 1 and fields[0] in modules:
                local = fields[2] if len(fields) == 3 and fields[1] == "as" else fields[0]
                found[local] = fields[0]
    return found


def _call_parent_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """The name of the call that consumes ``node``, or the parent's kind.

    The key call is often the VALUE of an attribute (``_k.key().envelope()``),
    so climb the attribute/subscript wrappers before judging: the failure this
    check exists for is the key's value being carried onward, not wrapped.
    """
    parent = parents.get(node)
    for _ in range(6):
        if isinstance(parent, ast.keyword):
            parent = parents.get(parent)
            continue
        if isinstance(parent, (ast.Attribute, ast.Subscript, ast.Starred, ast.JoinedStr)):
            parent = parents.get(parent)
            continue
        break
    func = getattr(parent, "func", None)
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return type(parent).__name__


def unwrapped_key_calls(
    references: dict[str, dict[str, int]],
    root: Path = PACKAGE,
    namespaces: tuple[str, ...] | None = None,
) -> list[str]:
    """Deferred key calls NOT wrapped in a render resolver, one line each.

    ``references`` is a census (its keys are the files worth parsing), so this
    adds no directory walk. The check is the second half of the policy: the
    census catches a NEW render point, this catches a render point edited to
    hand the ``Msg``/envelope on instead of rendering it.

    A call whose value is used as an attribute or passed through unchanged is
    reported, because that is how an envelope identity reaches a client. A key
    that is merely RETURNED (``_k.verb``) is not a call and is left to the
    caller's own wrapper.
    """
    namespaces = namespaces or tuple(catalogues.DEFERRED_NAMESPACES)
    modules = {key_module(namespace): namespace for namespace in namespaces}
    problems: list[str] = []
    for relative in sorted({path for counts in references.values() for path in counts}):
        text = (root / relative).read_text(encoding="utf-8", errors="replace")
        aliases = _all_aliases(text, set(modules))
        if not aliases:
            continue
        tree = ast.parse(text)
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
            ):
                continue
            module = aliases.get(node.func.value.id)
            if module is None:
                continue
            wrapper = _call_parent_name(node, parents)
            if wrapper not in RENDER_WRAPPERS:
                problems.append(  # i18n: ignore dev-tool diagnostic (English-only)
                    f"{relative}:{node.lineno}: {node.func.value.id}.{node.func.attr}() "
                    f"({modules[module]}) is not inside a render wrapper ({wrapper})"
                )
    return problems


def recorded_references(data: dict[str, Any]) -> dict[str, dict[str, int]]:
    """The census the inventory records, as the comparison inputs it.

    ``Any`` is the JSON document's real shape: the top-level mapping is
    heterogeneous by design — a ``note`` string beside the per-namespace
    entries — and a TypedDict for a dev tool's file would be more ceremony
    than clarity.
    """
    return {namespace: dict(entry["references"]) for namespace, entry in data["namespaces"].items()}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        # i18n: ignore operator tooling help (English-only), like check.py's
        description="Deferred wire namespace census",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="report drift, write nothing",  # i18n: ignore operator tooling help
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="record the live census in the inventory",  # i18n: ignore operator tooling help
    )
    args = parser.parse_args(argv)
    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    if args.check:
        rows = compare(recorded_references(data), census())
        for namespace, path, was, now in rows:
            print(f"{namespace}: {path} {was} -> {now}")
        return 1 if rows else 0
    live = census()
    for namespace, entry in data["namespaces"].items():
        entry["references"] = dict(sorted(live.get(namespace, {}).items()))
    INVENTORY.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {INVENTORY}")  # i18n: ignore operator tooling diagnostic
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

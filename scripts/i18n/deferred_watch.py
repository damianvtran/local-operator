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

`--write` refreshes the recorded counts in the inventory (reasons and verified
render points are review data and are left alone); `--check` reports a live
census that differs without writing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

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
    names: set[str] = set()
    for match in _KEYS_IMPORT.finditer(text):
        listing = (match.group(1) or match.group(2)).replace("(", " ").replace(")", " ")
        for part in listing.split(","):
            fields = part.split()
            if fields and fields[0] == module:
                names.add(fields[2] if len(fields) == 3 and fields[1] == "as" else module)
    return names


def _longer_key_modules(root: Path, module: str) -> tuple[str, ...]:
    """Sibling key modules whose name starts with ``module`` (its children).

    ``wire_settings_account_policy`` contains ``wire_settings`` as a substring,
    and a plain count would charge the parent for a child's render point. The
    key directory is the authority for which modules exist, so a namespace
    added later is covered without editing this rule.
    """
    stems = {path.stem for path in (root / GENERATED_KEY_MODULES).glob("*.py")}
    return tuple(sorted(stem for stem in stems if stem.startswith(module) and stem != module))


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
    longer = {
        namespace: _longer_key_modules(root, key_module(namespace)) for namespace in namespaces
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


def recorded_references(data: dict) -> dict[str, dict[str, int]]:
    """The census the inventory records, as the comparison inputs it."""
    return {
        namespace: dict(entry["references"])
        for namespace, entry in data["namespaces"].items()
    }


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

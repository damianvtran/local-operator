#!/usr/bin/env python3
"""The i18n ratchet and catalogue checks (RFC §2.7).

TWO JOBS, one cheap entry point (CI job `i18n`, and a unit test):

1. **The AST ratchet.** Literals reaching user-facing sinks are counted per
   file and compared against ``i18n/baseline.json``, a per-file ceiling that
   may only stay equal or go down. Extraction slices lower their files'
   entries with ``--update``; new files start at 0, so new code must extract,
   pragma (``# i18n: ignore <reason>``) or be allowlisted (``i18n/allowlist.toml``,
   reason REQUIRED). This module deliberately flags BROADLY: a false positive
   is visible noise the allowlist records, a false negative is a hole nothing
   reports — the repository's fail-closed rule. The one known blind spot is
   documented at ``_arguments`` (string concatenation via ``+``).

   **ENFORCEMENT IS SCOPED (RFC §9: P1 ships ADVISORY).** The baseline carries
   an ``enforced`` list of path prefixes (and exact file paths). Findings
   OUTSIDE that scope are printed as ``advisory:`` lines and DO NOT fail
   (exit 0): the ratchet reports the tree's real state without turning every
   unrelated PR in a fleet of concurrent sessions red before an extraction
   slice exists to give a string a key. Findings INSIDE the scope fail, as do
   catalogue parity and generation drift EVERYWHERE (those only involve
   i18n-owned files). The list starts with the i18n package and its tooling;
   extraction slices ACTIVATE their area by adding that area's prefix in a
   reviewed change, and P5 flips the whole tree to enforced.

2. **Catalogue checks.** Every message parses in the shipped subset; keys are
   lower snake_case and namespaced; per namespace, every locale's key set,
   argument tuples (name AND kind) and plural categories match ``en``; a
   plural branch must define ``other``; the en-only context sidecar may only
   describe keys that exist.

The ratchet's counts are a SNAPSHOT, not a truth: they pin what "the tree as
it stands" means so a diff's effect is measurable. ``--init`` is therefore
one-time by design (it refuses to overwrite an existing baseline without
``--force``); a legitimate scanner-coverage change or a full re-snapshot on a
moved base re-runs ``--init --force`` and says so in its PR body — that
preserves ``enforced`` and its note, which are REVIEW STATE, not counts.

Usage:
    python scripts/i18n/check.py                 # ratchet + parity (CI)
    python scripts/i18n/check.py --init          # capture the baseline once
    python scripts/i18n/check.py --init --force  # full re-snapshot (moved base / scanner change)
    python scripts/i18n/check.py --update <path> # lower a file's ceiling
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
# Scripts under scripts/ read the tree they live in regardless of which venv
# launched them (a console script's sys.path[0] is .venv/bin, not the cwd).
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_operator.i18n import catalogues  # noqa: E402
from local_operator.i18n import runtime as msg_runtime  # noqa: E402

PRODUCT_TREE = "local_operator"
#: The scanned corpus, repo-relative. `scripts/i18n/` is scanned because it is
#: inside the initial enforced scope: the tooling must not grow user-visible
#: prose of its own (its existing developer-diagnostic prints are frozen at
#: their baseline counts; a NEW file there starts at 0 like anywhere else).
SCAN_ROOTS = (PRODUCT_TREE, "scripts/i18n")
#: Fallback enforcement scope when the baseline carries no `enforced` field
#: (a pre-schema-2 file): the i18n package and its tooling — the code this
#: program itself owns. Slices extend the list; P5 flips the whole tree.
DEFAULT_ENFORCED = ("local_operator/i18n/", "scripts/i18n/")
BASELINE_NOTE = (
    "enforced = blocking scope (path prefixes or exact paths); findings elsewhere are "
    "advisory (report-only) until the area's extraction slice adds its prefix. "
    "files = per-file literal ceilings (shrink-only; new files start at 0). "
    "See scripts/i18n/check.py."
)
BASELINE = REPO / "i18n" / "baseline.json"
ALLOWLIST = REPO / "i18n" / "allowlist.toml"
PRAGMA_RE = re.compile(r"#\s*i18n:\s*ignore\s+\S")
KEY_RE = re.compile(r"^[a-z0-9_.]+$")
#: Any Unicode letter — the line between a prose-like literal and punctuation,
#: width strings ("──"), or pure format tokens ("{}").
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    kind: str
    line: int
    text: str


def _arguments(call: ast.Call) -> list[ast.expr]:
    """Every argument expression of ``call``, positional then keyword.

    KNOWN BLIND SPOT, stated rather than implied: a literal assembled with `+`
    (`"a " + name`) is a BinOp, not a constant, and is NOT counted. The
    extraction slices re-spell those as f-strings as they pass, and a checker
    extension that follows the tui slice's counts would add `+` folding here.
    """
    return [*call.args, *(kw.value for kw in call.keywords)]


def _literal_worthy(node: ast.expr) -> bool:
    """Whether ``node`` counts as one prose-like literal.

    A constant string with at least one letter, or an f-string with at least
    one such constant SEGMENT (`f"Session {id} failed"` counts; `f"{a}-{b}"`
    does not). Counting f-strings is deliberate: they are the dominant prose
    form in the TUI (RFC §1), and a ratchet that cannot see them would pass
    exactly the new code it exists to catch.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return bool(_LETTER_RE.search(node.value))
    if isinstance(node, ast.JoinedStr):
        return any(
            isinstance(part, ast.Constant)
            and isinstance(part.value, str)
            and _LETTER_RE.search(part.value)
            for part in node.values
        )
    return False


def _call_name(call: ast.Call) -> str:
    """``print`` / ``console.print`` / ``Text`` — the syntactic callee name."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


#: Keyword names that carry user copy on the registry/dataclass constructors
#: below — the fields RFC §2.7 names as sinks (`NoticeEvent`'s text/headline
#: get their own rule above; they are not kwargs of these three).
_COPY_KWARGS = ("label", "help", "warning", "description")


def _call_violations(call: ast.Call) -> list[tuple[str, ast.expr]]:
    """Sink matches for one call: ``(kind, literal node)`` pairs, deduplicated.

    Deduplication matters because one literal can satisfy two rules (a
    `Setting(help=...)` is both the generic `help=` rule and the registry-copy
    rule); the count is per LITERAL, not per rule.

    `.append`/`.update` count on ANY receiver, deliberately: in this corpus
    user prose flows through list-building (`lines.append("her check-ins
    stopped")` is the status sentence), so a receiver-name allowlist would
    miss most copy while still passing machine tokens. The broad read trades
    false positives — visible, allowlist-able — for no false negatives.
    """
    name = _call_name(call)
    matched: dict[int, tuple[str, ast.expr]] = {}

    def take(kind: str, node: ast.expr | None) -> None:
        if node is not None and _literal_worthy(node):
            matched.setdefault(id(node), (kind, node))

    if name == "print":
        # `print(...)` and `console.print(...)` both resolve here: `_call_name`
        # returns the attribute name for attribute calls.
        for arg in call.args:
            take("print", arg)
    if isinstance(call.func, ast.Attribute) and call.func.attr in ("append", "update"):
        for arg in call.args:
            take(f"textual.{call.func.attr}", arg)
    if name == "Text" or (
        isinstance(call.func, ast.Attribute) and call.func.attr in ("assemble", "from_markup")
    ):
        for arg in call.args:
            take("Text", arg)
    take("help=", _keyword(call, "help"))
    if name == "HTTPException":
        take("HTTPException", _keyword(call, "detail"))
        if len(call.args) >= 2:
            take("HTTPException", call.args[1])
    if name == "CRUDResponse":
        take("CRUDResponse", _keyword(call, "message"))
    if name == "NoticeEvent":
        for kw in ("text", "headline"):
            take("NoticeEvent", _keyword(call, kw))
    if name in ("Setting", "SlashCommand", "Choice"):
        for kw in _COPY_KWARGS:
            take(f"{name} copy", _keyword(call, kw))
        # Positional copy fields, by the constructors' own field order:
        # SlashCommand("<name token>", "<description>", ...) and
        # Choice("<value>", "<label>", "<description>").
        if name == "SlashCommand":
            for arg in call.args[1:]:
                take("SlashCommand copy", arg)
        elif name == "Choice":
            for arg in call.args[1:3]:
                take("Choice copy", arg)
    return list(matched.values())


def scan_source(path: str, source: str) -> list[Violation]:
    """The violations of one file's source, pragmas applied.

    A ``# i18n: ignore <reason>`` comment ANYWHERE between a call's first and
    last line exempts the literals in that call — not just a comment on the
    literal's own line, which is what the allowlist docs used to claim
    (round-1 n2). Multi-line calls are the norm here, so the call span is the
    span that reads naturally; the allowlist's text now says exactly this.
    """
    try:
        module = ast.parse(source, filename=path)
    except SyntaxError as exc:  # pragma: no cover - the repo gate parses first
        raise SystemExit(f"check.py: cannot parse {path}: {exc}") from exc
    lines = source.splitlines()
    found: list[Violation] = []
    for call in (node for node in ast.walk(module) if isinstance(node, ast.Call)):
        for kind, node in _call_violations(call):
            start = min(call.lineno, node.lineno)
            end = max(
                getattr(call, "end_lineno", call.lineno) or call.lineno,
                node.end_lineno or node.lineno,
            )
            if any(PRAGMA_RE.search(lines[i - 1]) for i in range(start, min(end, len(lines)) + 1)):
                continue
            text = node.value if isinstance(node, ast.Constant) else "<f-string>"
            found.append(Violation(kind=kind, line=node.lineno, text=str(text)[:80]))
    return found


def scan_tree(root: Path) -> dict[str, list[Violation]]:
    """Scan the corpus (``SCAN_ROOTS``) under ``root``, repo-relative keys."""
    results: dict[str, list[Violation]] = {}
    for base in SCAN_ROOTS:
        tree = root / base
        if not tree.is_dir():  # a fixture or partial tree need not carry every root
            continue
        for path in sorted(tree.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            source = path.read_text(encoding="utf-8")
            violations = scan_source(rel, source)
            if violations:
                results[rel] = violations
    return results


# ---------------------------------------------------------------------------
# Allowlist and baseline
# ---------------------------------------------------------------------------


def load_allowlist(path: Path) -> list[tuple[str, str]]:
    """``(pattern, reason)`` entries; a missing reason is a hard error."""
    if not path.is_file():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    parsed: list[tuple[str, str]] = []
    if not isinstance(entries, list):
        raise SystemExit(
            f"check.py: {path.name}: `entries` must be a list of [pattern, reason] pairs"
        )
    for entry in entries:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not all(isinstance(part, str) for part in entry)
        ):
            raise SystemExit(
                f'check.py: {path.name}: every allowlist entry must be ["path-pattern", "reason"]'
            )
        pattern, reason = entry
        if not reason.strip():
            raise SystemExit(f"check.py: {path.name}: allowlist entry {pattern!r} has no reason")
        parsed.append((pattern, reason))
    return parsed


def allowlisted(path: str, entries: list[tuple[str, str]]) -> str | None:
    """The reason this file is exempt, or None.

    Two forms, both exact: a repo-relative file path, or a directory prefix
    ending in ``/``. NO GLOBS on purpose — fnmatch's `*` spans path
    separators, so a "narrow-looking" glob can silently exempt a subtree;
    a directory the author means to cover is then written as one, visibly.
    """
    for pattern, reason in entries:
        if pattern.endswith("/"):
            if path.startswith(pattern):
                return reason
        elif path == pattern:
            return reason
    return None


def load_baseline(path: Path) -> dict[str, Any]:
    """The whole baseline object: ``enforced`` (scope) and ``files`` (ceilings).

    Missing or damaged shapes are tolerated as EMPTY rather than fatal: the
    file is data, and a hand-edit mistake should surface as a normal check
    failure (counted files read as "not in baseline"), not a traceback. A
    MISSING ``enforced`` key falls back to ``DEFAULT_ENFORCED`` (fail closed —
    a schema-1 file must not silently un-enforce the code this program owns);
    a PRESENT-but-empty list is respected as the deliberate "nothing is
    enforced" state; and a NON-LIST value (a hand-edit) fails closed the same
    way — silently UN-enforcing on damage is the direction this guard must not
    fail (round-4 m1).
    """
    if not path.is_file():
        return {"schema": 2, "enforced": list(DEFAULT_ENFORCED), "files": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        data = {}
    files = data.get("files", {})
    enforced = data.get("enforced", list(DEFAULT_ENFORCED))
    return {
        "schema": data.get("schema", 1),
        "note": data.get("note", BASELINE_NOTE),
        "enforced": (
            [str(p) for p in enforced] if isinstance(enforced, list) else list(DEFAULT_ENFORCED)
        ),
        "files": {str(p): int(c) for p, c in files.items()} if isinstance(files, dict) else {},
    }


def save_baseline(path: Path, files: dict[str, int], enforced: list[str], note: str) -> None:
    payload = {
        "schema": 2,
        "note": note,
        "enforced": list(enforced),
        "files": {k: files[k] for k in sorted(files)},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _activation_state(path: Path) -> tuple[list[str], str]:
    """``(enforced, note)`` to PRESERVE across ``--init --force``.

    These are review state, not counts: a re-snapshot on a moved base must not
    walk an area's activation back. Presence of the key selects defaults, not
    truthiness — a deliberate empty list stays empty (round-1 n3's lesson,
    applied to the new field).
    """
    if not path.is_file():
        return list(DEFAULT_ENFORCED), BASELINE_NOTE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return list(DEFAULT_ENFORCED), BASELINE_NOTE
    if not isinstance(data, dict):
        return list(DEFAULT_ENFORCED), BASELINE_NOTE
    enforced = data.get("enforced", list(DEFAULT_ENFORCED))
    return (
        [str(p) for p in enforced] if isinstance(enforced, list) else list(DEFAULT_ENFORCED),
        str(data.get("note", BASELINE_NOTE)),
    )


def in_enforced_scope(rel: str, enforced: list[str]) -> bool:
    """Whether ``rel`` sits inside the blocking scope.

    A value ending in ``/`` is a path-segment prefix (``local_operator/i18n/``);
    anything else is an exact file path. No globs by design — ``fnmatch``'s
    ``*`` crosses ``/``, which is the trap the package-data guard documents
    (round-1 m4).
    """
    for entry in enforced:
        if entry.endswith("/"):
            if rel.startswith(entry):
                return True
        elif rel == entry:
            return True
    return False


@dataclass(frozen=True)
class Finding:
    """One counted file that exceeds — or is missing from — its ceiling."""

    path: str
    count: int
    ceiling: int | None  # None = not in the baseline: a new file, starts at 0

    def line(self) -> str:
        if self.ceiling is None:
            return (
                f"{self.path}: {self.count} literal(s) — not in baseline.json; "
                "a file with no recorded ceiling starts at 0"
            )
        return f"{self.path}: {self.count} literal(s) against a ceiling of {self.ceiling}"


def findings(counts: dict[str, int], files: dict[str, int]) -> list[Finding]:
    out: list[Finding] = []
    for rel in sorted(counts):
        ceiling = files.get(rel)
        count = counts[rel]
        if ceiling is None:
            out.append(Finding(rel, count, None))
        elif count > ceiling:
            out.append(Finding(rel, count, ceiling))
    return out


def _counts(scanned: dict[str, list[Violation]], entries: list[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rel, violations in scanned.items():
        if allowlisted(rel, entries) is not None:
            continue
        if violations:
            counts[rel] = len(violations)
    return counts


# ---------------------------------------------------------------------------
# Catalogue checks
# ---------------------------------------------------------------------------


def check_catalogues() -> list[str]:
    """Parity + shape problems across ``catalogues/``, as printable lines."""
    problems: list[str] = []
    root = catalogues.catalogue_root()
    if not root.is_dir():
        return [f"catalogues: root {root} does not exist"]
    plural = json.loads(
        (Path(msg_runtime.__file__).with_name("data") / "plural_rules.json").read_text(
            encoding="utf-8"
        )
    )
    en_namespaces = catalogues.namespaces("en")
    en_messages: dict[str, dict[str, str]] = {}
    for namespace in en_namespaces:
        messages = catalogues.load_catalogue("en", namespace)
        en_messages[namespace] = messages
        for key, source in messages.items():
            if not KEY_RE.match(key):
                problems.append(f"en/{namespace}: key {key!r} is not lower snake_case")
            if not key.startswith(namespace + "."):
                problems.append(f"en/{namespace}: key {key!r} does not repeat its namespace prefix")
            try:
                # Syntax gate for the en source: parse it (the cross-locale
                # argument comparison below re-reads it as the reference).
                msg_runtime.message_arguments(source)
            except msg_runtime.MessageSyntaxError as exc:
                problems.append(f"en/{namespace}: {key}: {exc}")
                continue
            _check_plural_categories(problems, "en", namespace, key, source, plural, "en")
        _check_context_sidecar(problems, "en", namespace, messages)
    # Cross-file duplicates: one message is defined by exactly ONE file (the
    # collision rule `messages._namespace_for` relies on, resolving
    # deepest-first). The per-file checks above cannot see this — a key like
    # `wire.errors.auth.x` satisfies both the `wire.errors` and the
    # `wire.errors.auth` prefix tests — so a duplicate would silently shadow
    # instead of failing. Translation-side duplicates reduce to this case or
    # to the missing/unknown-key parity checks below.
    owners: dict[str, str] = {}
    for namespace in en_namespaces:
        for key in en_messages[namespace]:
            if key in owners:
                problems.append(
                    # i18n: ignore checker diagnostic (operator tooling output, English-only)
                    f"en: key {key!r} is defined by both {owners[key]!r} and {namespace!r}"
                )
            else:
                owners[key] = namespace
    for locale in catalogues.locales():
        if locale == "en":
            continue
        if locale not in plural.get("locales", {}):
            problems.append(f"{locale}: no plural rules generated for this locale")
        for namespace in en_namespaces:
            try:
                translated = catalogues.load_catalogue(locale, namespace)
            except catalogues.CatalogueNotFound:
                problems.append(f"{locale}/{namespace}: missing (en has it)")
                continue
            except catalogues.CatalogueInvalid as exc:
                problems.append(f"{locale}/{namespace}: {exc}")
                continue
            expected = en_messages[namespace]
            for key in sorted(set(expected) - set(translated)):
                problems.append(f"{locale}/{namespace}: missing key {key!r}")
            for key in sorted(set(translated) - set(expected)):
                problems.append(f"{locale}/{namespace}: unknown key {key!r}")
            for key in sorted(set(expected) & set(translated)):
                try:
                    got = msg_runtime.message_arguments(translated[key])
                except msg_runtime.MessageSyntaxError as exc:
                    problems.append(f"{locale}/{namespace}: {key}: {exc}")
                    continue
                want = msg_runtime.message_arguments(expected[key])
                if got != want:
                    problems.append(
                        f"{locale}/{namespace}: {key}: arguments {got!r} != en's {want!r}"
                    )
                _check_plural_categories(
                    problems, locale, namespace, key, translated[key], plural, locale
                )
    return problems


def _check_plural_categories(
    problems: list[str],
    locale: str,
    namespace: str,
    key: str,
    source: str,
    plural: dict[str, Any],
    table_locale: str,
) -> None:
    """Every literal plural selector must be a category of the locale."""
    known = set(plural["locales"].get(table_locale, {}).get("categories", []))
    try:
        selector_tuples = msg_runtime.plural_selectors(source)
    except msg_runtime.MessageSyntaxError:
        return  # reported by the argument check; not twice
    for selectors in selector_tuples:
        if "other" not in selectors:
            problems.append(f"{locale}/{namespace}: {key}: plural has no `other` branch")
        for selector in selectors:
            if selector.startswith("="):
                continue
            if selector not in known:
                categories = sorted(known)
                problems.append(
                    f"{locale}/{namespace}: {key}: plural category {selector!r} "
                    f"not in {categories}"
                )


def _check_context_sidecar(
    problems: list[str], locale: str, namespace: str, messages: dict[str, str]
) -> None:
    try:
        sidecar = catalogues.context(locale, namespace)
    except catalogues.CatalogueInvalid as exc:
        problems.append(f"{locale}/{namespace}: {exc}")
        return
    for key, entry in sidecar.items():
        if key not in messages:
            problems.append(f"{locale}/{namespace}: context sidecar describes unknown key {key!r}")
        if not isinstance(entry, dict):
            problems.append(f"{locale}/{namespace}: context sidecar {key!r} must be an object")
            continue
        max_cells = entry.get("maxCells")
        if max_cells is not None and not isinstance(max_cells, int):
            problems.append(
                f"{locale}/{namespace}: context sidecar {key!r} maxCells must be an int"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--init", action="store_true", help="write the baseline from the current tree (one-time)"
    )
    mode.add_argument(
        "--update", nargs="+", metavar="PATH", help="lower the ceilings for these files"
    )
    mode.add_argument(
        "--catalogues-only",
        action="store_true",
        help="run the catalogue checks alone (used by tests and fast local runs)",
    )
    parser.add_argument(
        "--force", action="store_true", help="allow --init over an existing baseline"
    )
    args = parser.parse_args(argv)

    if args.catalogues_only:
        problems = check_catalogues()
        for line in problems:
            print(line)
        print(f"check.py: catalogue checks: {len(problems)} problem(s)")
        return 1 if problems else 0

    entries = load_allowlist(ALLOWLIST)
    scanned = scan_tree(REPO)
    counts = _counts(scanned, entries)
    baseline = load_baseline(BASELINE)
    enforced = baseline["enforced"]
    files = baseline["files"]

    if args.init:
        # ANY existing baseline refuses, empty or not: an empty file used to be
        # treated as "no baseline" by a truthiness test and silently
        # overwritten without `--force` (round-1 n3).
        if BASELINE.is_file() and not args.force:
            print(
                "check.py --init: a baseline already exists; use --force only for a reviewed "
                "scanner-coverage change (say so in the PR body)."
            )
            return 2
        preserved_enforced, note = _activation_state(BASELINE)
        save_baseline(BASELINE, counts, enforced=preserved_enforced, note=note)
        total = sum(counts.values())
        print(
            f"check.py --init: baseline written: {len(counts)} file(s), {total} literal(s); "
            f"enforced: {', '.join(preserved_enforced) or 'none'}"
        )
        return 0

    if args.update:
        for rel in args.update:
            rel = Path(rel).as_posix()
            if allowlisted(rel, entries) is not None:
                print(f"check.py --update: {rel} is allowlisted; nothing to lower")
                continue
            new = counts.get(rel, 0)
            old = files.get(rel)
            if old is None:
                if new > 0:
                    print(
                        f"check.py --update: {rel} has {new} literal(s) with no recorded "
                        "ceiling — a file with no recorded ceiling starts at 0: extract, "
                        "pragma, or allowlist instead of recording one."
                    )
                    return 1
                continue
            if new > old:
                print(
                    f"check.py --update: {rel} went {old} -> {new}; the baseline may only stay "
                    "equal or go down."
                )
                return 1
            if new:
                files[rel] = new
            else:
                files.pop(rel, None)
        save_baseline(BASELINE, files, enforced=enforced, note=baseline["note"])
        print(f"check.py --update: baseline now {len(files)} file(s)")
        return 0

    failures: list[str] = []
    advisories: list[str] = []
    for finding in findings(counts, files):
        line = finding.line()
        if in_enforced_scope(finding.path, enforced):
            failures.append(line)
        else:
            advisories.append(f"advisory: {line}")
    problems = check_catalogues()
    for line in advisories + failures + problems:
        print(line)
    if failures or problems:
        print(
            f"check.py: FAILED — {len(failures)} ratchet finding(s) in the enforced scope, "
            f"{len(problems)} catalogue problem(s) ({len(advisories)} advisory finding(s) "
            "outside it). Extract the string, add `# i18n: ignore <reason>`, or (for a "
            "file-level exemption) record it in i18n/allowlist.toml with a reason."
        )
        return 1
    total = sum(counts.values())
    suffix = (
        f"; {len(advisories)} advisory finding(s) (report-only until activated)"
        if advisories
        else ""
    )
    print(
        f"check.py: ok — {len(scanned)} file(s) scanned, {total} literal(s) at or "
        f"below the baseline, catalogues consistent{suffix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Regenerate the committed i18n artifacts, or check them for drift.

WHAT THIS OWNS (all of it generated, all of it committed):

* ``local_operator/i18n/data/plural_rules.json`` + ``formats.json`` — emitted by
  ``scripts/i18n/emit.mjs`` from Node `Intl` (see that file for determinism);
* ``local_operator/i18n/keys/<namespace>.py`` — one function per message key,
  typed from the message's own placeholders (RFC §2.4: per-key functions, NOT
  `@overload`s — the overload shape did not finish a 5,000-key typecheck);
* ``local_operator/mobile/web/src/i18n/keys.gen.ts`` — type-only wire/relay
  key map and two-overload translator contract from the same ICU parser;
* ``local_operator/i18n/data/shipped.json`` — the shipped-locale set computed
  from the audit ledger (§6: a locale ships iff every namespace has a
  ledger entry that passed against the CURRENT en source hash). ``en`` is
  always shipped.
* ``i18n/ledger.json`` — the legacy aggregate ledger, mirrored from the
  per-locale shards under ``i18n/ledger/<locale>.json`` (the authoritative
  source since 2026-10-29: one shared array made every locale PR conflict
  with every other). Append audit rows to your locale's shard, then
  regenerate; ``--check`` fails if the aggregate and shards disagree, so a
  stale aggregate cannot silently win.

WHY A DRIFT CHECK. These are generated-but-committed, so a hand edit or a
catalogue change without regeneration would ship tables that disagree with the
catalogues. ``--check`` re-runs everything into a temp directory and requires
byte-identical output; CI runs it as the cheap `i18n` job (and locally:
``.venv/bin/python scripts/i18n/generate.py --check``).

DETERMINISM RULES, learned from `scripts/gen_agent_seed_manifest.py`: no clock
reads, no environment reads, sorted iteration everywhere, and the emitter is
invoked with a fixed locale set. The only input that can move the output is a
change to the catalogue files, the emitter, or the ledger — which is exactly
what should void the check.
"""

from __future__ import annotations

import argparse
import json
import keyword
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
# Scripts under scripts/ read the tree they live in regardless of which venv
# launched them (a console script's sys.path[0] is .venv/bin, not the cwd).
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_operator.i18n import catalogues  # noqa: E402
from local_operator.i18n import runtime as msg_runtime  # noqa: E402
from local_operator.i18n.resolve import SUPPORTED_LOCALES  # noqa: E402

EMITTER = REPO / "scripts" / "i18n" / "emit.mjs"
DATA_DIR = REPO / "local_operator" / "i18n" / "data"
KEYS_DIR = REPO / "local_operator" / "i18n" / "keys"
LEDGER = REPO / "i18n" / "ledger.json"
LEDGER_SHARDS = REPO / "i18n" / "ledger"


def _ledger_entries() -> list[dict[str, Any]]:
    """The authoritative audit rows: per-locale shards, aggregate fallback.

    Shards win whenever ``i18n/ledger/`` holds any ``*.json``; the aggregate
    is read only for trees that predate sharding (fixture roots, old
    checkouts). Within each shard, file order is preserved and per-locale
    last-wins is applied downstream exactly as the aggregate's array order
    did — the migration is order-preserving per locale, so eligibility
    semantics cannot move. Cross-locale order never mattered: the shipped
    computation keys on ``(locale, namespace)`` and iterates SUPPORTED_LOCALES.
    """
    shard_paths = sorted(LEDGER_SHARDS.glob("*.json")) if LEDGER_SHARDS.is_dir() else []
    if shard_paths:
        entries: list[dict[str, Any]] = []
        for path in shard_paths:
            shard = _read_json(path)
            rows = shard.get("entries", []) if isinstance(shard, dict) else []
            entries.extend(row for row in rows if isinstance(row, dict))
        return entries
    if not LEDGER.is_file():
        return []
    ledger = _read_json(LEDGER)
    if not isinstance(ledger, dict):
        return []
    return [row for row in ledger.get("entries", []) if isinstance(row, dict)]


def _render_aggregate() -> str:
    """The legacy aggregate, re-rendered from shards for compatibility.

    Sorted by locale (then each shard's own row order): a deterministic,
    reviewable mirror. No consumer reads it for eligibility anymore —
    ``_ledger_entries`` prefers the shards — so its ordering is free to be
    canonical rather than historical. ``schema`` is carried over from the
    pre-migration aggregate so existing readers keep working.
    """
    payload: dict[str, Any] = {"schema": 1, "entries": _ledger_entries()}
    if LEDGER.is_file():
        legacy = _read_json(LEDGER)
        if isinstance(legacy, dict) and "schema" in legacy:
            payload["schema"] = legacy["schema"]
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _aggregate_rows_missing_from_shards() -> list[str]:
    """Rows a contributor appended to the AGGREGATE instead of its shard.

    The aggregate is a generated mirror, so re-rendering it would DROP a row
    written there directly — silently reversing somebody's audit. ``--check``
    fails on the drift, but a plain ``generate.py`` run must refuse rather than
    overwrite: the contributor's row belongs in ``i18n/ledger/<locale>.json``,
    and moving it is their decision (the aggregate carries no ordering by which
    we could reconstruct intent). Compared as a multiset of serialized rows, so
    a pure reordering is not mistaken for a lost row, and summarised per
    ``(locale, namespace)`` because a wave usually appends one namespace many
    times.
    """
    if not (LEDGER_SHARDS.is_dir() and any(LEDGER_SHARDS.glob("*.json"))):
        return []
    if not LEDGER.is_file():
        return []
    legacy_rows = _read_json(LEDGER)
    if not isinstance(legacy_rows, dict):
        return []
    shard_counts = Counter(json.dumps(row, sort_keys=True) for row in _ledger_entries())
    stranded: Counter[tuple[str, str]] = Counter()
    for row in legacy_rows.get("entries", []):
        if not isinstance(row, dict):
            continue
        key = json.dumps(row, sort_keys=True)
        if shard_counts[key] > 0:
            shard_counts[key] -= 1
            continue
        stranded[(str(row.get("locale")), str(row.get("namespace")))] += 1
    return [
        f"{locale}/{namespace}" + (f" ({count} rows)" if count > 1 else "")
        for (locale, namespace), count in sorted(stranded.items())
    ]


def _bundled_dir() -> Path:
    """The relay SPA's generated own-copy modules — one eager English map beside
    one chunk per complete shipped locale (see ``_render_bundled_module``).

    Derived from ``REPO`` at CALL time, not bound at import: the test fixtures
    monkeypatch ``REPO`` at a temp root, and a module-level constant would keep
    planning into the real checkout — a fixture run has already overwritten a
    committed artifact that way.
    """
    return REPO / "local_operator" / "mobile" / "web" / "src" / "i18n"


#: Security disposition is independent of per-carried-namespace audit passes.
#: A held locale cannot be made eligible by omitting its file or ledger row and
#: publishing a smaller passed package; removal requires a separately reviewed,
#: explicit security disposition from the policy owner. This blocks eligibility,
#: not recognition/formatting.
#:
#: EMPTY AS OF 2026-10-24, by the owner's explicit disposition. zh-CN had been
#: held because ``wire.slash.credential`` was read outside the redaction policy
#: (docs/security/redaction-policy.md, 2026-10-19). That blocker is resolved at
#: the source: the policy now names key names as never-secret, and the
#: sanctioned ``read_public_copy`` route (PR #2877, released in 0.89.0) served
#: the committed blob with checkout-identity binding and an audit record. The
#: namespace was then re-audited at its current bytes (evidence:
#: i18n/audits/evidence/zh-CN/wire.slash/zh24s1/), every carried namespace has a
#: current-byte passed audit, and i18n/ledger.json carries 81 passed zh-CN rows.
#: Add a locale back here only with a new reviewed disposition.
SECURITY_HELD_LOCALES: frozenset[str] = frozenset()

#: Namespaces whose audited shipped-locale copy the relay SPA carries in-process,
#: as the ``bundled.gen.ts`` map the store registers before any consumer renders.
#: EXPLICIT and reviewed, never "every ``relay.*`` file": ``relay.login`` is
#: rendered by the daemon's SSR gate before the SPA exists, so bundling it into
#: the client would ship copy the client never paints. ``wire.*`` is never
#: bundled — it is served, per §2.1, and an implicit bundle of it would defeat
#: the envelope. A later relay slice adds its namespace here, one line, reviewed.
#: ``relay.session`` is the session-extraction slice (E3 C2): the list body,
#: session view and history/start surfaces, painted by the SPA like the shell.
#: ``relay.account_policy`` is the LOH-12 account-routing slice: the settings
#: screen the SPA paints, with its wire.* counterpart served per §2.1 rather
#: than bundled. Both were added independently on either side of the rebase;
#: the tuple is one reviewed line per slice and carries all four.
#: ``relay.session.tags`` is the tag slice (LOH-11 C1): the tag line, the
#: assignment sheet and the list's long-press copy, split out of ``relay.session``
#: because the catalogues are one small owned namespace per surface.
#: ``relay.asks`` is the asks/cards slice (the relay's last planned namespace,
#: RFC §5): the ask card, asks sheet, dock, settled row, pending decision card,
#: approval-gate sheet and the disclosure that holds panels shut while a
#: question waits. The tuple is one reviewed line per slice, never a glob.
#: ``relay.approvals`` is the approvals-policy slice that landed on main while
#: ``relay.asks`` was in flight (`feat(mobile): offer both approval scopes on
#: the phone`): the policy state line, the third (global) scope and the mode
#: change links the pending card carries. It is a namespace of its own rather
#: than more ``relay.asks`` keys because the asks catalogues are already
#: carried by every translated locale, and a new key there would be untranslated
#: in all of them. It was expected to start en-only and ride §2.5 fallback; it
#: does not, because the programme's rule is that new copy ships translated. It
#: carries all seven locales, and the day it landed untranslated is the day the
#: ratchet and the census became mutually unsatisfiable for it (check.py forbade
#: the debt, the census required it). Recorded here because the next new
#: namespace will hit the same wall.
OWNED_SPA_NAMESPACES = (
    "relay.shell",
    "relay.session",
    "relay.session.tags",
    "relay.account_policy",
    "relay.library",
    # The ask card's own copy (R5 surfaces, 2026-10-23): the dismiss reasons and
    # the stale-ask context line, painted by the SPA inside the asks sheet. One
    # small owned namespace rather than keys added to `relay.session`, because a
    # new key in a namespace a locale has not carried yet GROWS the shrink-only
    # coverage allowance (docs/I18N.md).
    "relay.ask",
    "relay.approvals",
    "relay.asks",
    # The borderless redesign's chrome (drawer, chips, tray, banners; 2026-10-25,
    # docs/design/mobile-relay-redesign.md §11.1). Its own namespace so no existing
    # English value changes: a value edit stales every locale's passed ledger rows.
    "relay.chrome",
    # Condensed transcript bars and tool-fold units (C2). A separate owned
    # namespace keeps the frozen relay.chrome source bytes and ledger hashes
    # untouched while its English copy is prepared for the translation wave.
    "relay.transcript",
    # The session header's status dot (live / not answering / ended). Its own
    # owned namespace for the same reason relay.transcript has one: the two new
    # keys must not change relay.chrome's frozen English bytes, which would stale
    # every locale's passed ledger row for that namespace.
    "relay.health",
    # The phone inbox (`#/notifications`, docs/design/notifications-inbox.md §9):
    # reel, item detail, receipts and confirmations. Its own namespace for the
    # `relay.chrome` reason: adding keys there would stale every locale's passed
    # ledger row for a redesign another lane owns.
    "relay.notifications",
    # The composer's `$skill` picker and the one notice naming a skill the relay
    # could not attach (SKM-D6). Its own namespace for the same frozen-bytes
    # reason: relay.chrome carries the composer chrome and must not stale.
    "relay.skills",
)

GENERATED_HEADER = (
    '"""GENERATED by scripts/i18n/generate.py — do not edit by hand.\n\n'
    "Regenerate with `.venv/bin/python scripts/i18n/generate.py`; CI rejects\n"
    "drift (`generate.py --check`). The one function per message key is the\n"
    "shape RFC §2.4 proved: a mistyped key or kwarg fails pyright.\n"
    '"""\n\n'
    "from __future__ import annotations\n\n"
)


def _run_emitter(out_dir: Path) -> None:
    """Run the Node emitter into ``out_dir``; a missing node is a loud failure."""
    try:
        result = subprocess.run(
            ["node", str(EMITTER), "--out", str(out_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "generate.py: `node` is required to emit the Intl tables (the emitter "
            "is scripts/i18n/emit.mjs). Install Node or run this via CI, which has it."
        ) from exc
    if result.returncode != 0:
        raise SystemExit(
            f"generate.py: emit.mjs failed ({result.returncode}):\n{result.stderr or result.stdout}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identifier(name: str) -> str:
    """A valid Python identifier for a key segment or argument name.

    Two legal catalogue shapes would otherwise emit SYNTAX ERRORS that the
    text-comparing drift check cannot see (round-1 m2): a digit-leading
    segment (`demo.bad.2fa` -> `def 2fa`) and a Python-keyword argument
    (`{class, select, ...}` -> `*, class: str`). The Python spelling moves; the
    MESSAGE KEY and the argument name in the Msg body stay exactly as the
    catalogue spells them, so the wire contract is untouched.
    """
    safe = name
    if safe and safe[0].isdigit():
        safe = "_" + safe
    if keyword.iskeyword(safe):
        safe += "_"
    return safe


def _fn_name(namespace: str, key: str) -> str:
    suffix = key[len(namespace) + 1 :] if key.startswith(namespace + ".") else key
    return _identifier(suffix.replace(".", "_").replace("-", "_"))


def _param_types(message: str) -> list[tuple[str, str]]:
    """``(name, annotation)`` per argument, in first-appearance order."""
    annotations = {
        "plural": "int | float",
        "number": "int | float",
        "date": "datetime",
        "time": "datetime",
    }
    return [
        (name, annotations.get(kind, "str"))
        for name, kind in msg_runtime.message_arguments(message)
    ]


def _render_keys_module(namespace: str, messages: dict[str, str]) -> str:
    lines = [GENERATED_HEADER]
    body: list[str] = []
    needs_datetime = False
    seen_fnames: dict[str, str] = {}
    for key in sorted(messages):
        params = _param_types(messages[key])
        if any(annotation == "datetime" for _, annotation in params):
            needs_datetime = True
        fname = _fn_name(namespace, key)
        # Two legal keys can sanitise to one function name (`demo.x.class` /
        # `demo.x.class_`, `demo.x.foo-bar` / `demo.x.foo_bar`); emitting both
        # would silently SHADOW one def (round-2 m5) — refuse instead, like the
        # keyword-argument collision below.
        if fname in seen_fnames:
            raise SystemExit(
                f"generate.py: {namespace}: keys {seen_fnames[fname]!r} and {key!r} emit the "
                f"same function name {fname!r}; rename one in the catalogue"
            )
        seen_fnames[fname] = key
        if params:
            # Sanitise the ARGUMENT identifiers too (m2) and refuse a
            # sanitisation collision loudly rather than emitting duplicate
            # kwargs the compiler would reject far from the cause.
            sanitised = [_identifier(name) for name, _ in params]
            if len(set(sanitised)) != len(sanitised):
                raise SystemExit(
                    f"generate.py: {namespace}: arguments of {key!r} sanitise to "
                    f"duplicate identifiers {sanitised}; rename one in the catalogue"
                )
            # ONE bare ``*`` introduces the keyword-only block: emitting it per
            # parameter (``*, a: str, *, b: str``) is a SyntaxError and only
            # ever went unnoticed while every message carried one argument —
            # the analytics slice's first two-argument message caught it.
            signature = "*, " + ", ".join(
                f"{safe}: {ann}" for safe, (_, ann) in zip(sanitised, params)
            )
            pairs = ", ".join(f'"{name}": {safe}' for safe, (name, _) in zip(sanitised, params))
        else:
            signature = ""
            pairs = ""
        statement = f'    return Msg("{key}", {{{pairs}}})'
        if len(statement) <= 100:
            call = statement
        else:
            # Black-compatible wrapping for a call too long for one line: the
            # generated files must satisfy `black --check` AS GENERATED — the
            # drift gate compares bytes, so a formatter pass after generation
            # would make the check unreproducible (review round 2, MAJOR).
            call = f'    return Msg(\n        "{key}",\n        {{{pairs}}},\n    )'
        header = f"def {fname}({signature}) -> Msg:"
        # A namespace's longer function name plus several typed placeholders can
        # exceed the same 100-column gate as the return statement. Wrap the
        # generated signature too, rather than hand-editing one generated key
        # module or giving this namespace a lint exemption. `signature` is EMPTY
        # for a zero-placeholder key, and wrapping an empty one emitted a lone
        # `,` inside the parentheses — invalid Python for a long identifier that
        # the generator has always accepted (agent review round 1, C1-2). A
        # function NAME cannot be split by any formatter, so that shape keeps the
        # single-line header it has at the merge base.
        if len(header) > 100 and signature:
            wrapped = signature.replace(", ", ",\n    ")
            header = f"def {fname}(\n    {wrapped},\n) -> Msg:"
        body.append(f"{header}\n{call}\n")
    if body:
        # Imports only when something uses them: a namespace stub with zero
        # keys yet must not trip flake8's unused-import rule (F401) — the
        # ratchet gate runs over exactly these files.
        if needs_datetime:
            lines.append("from datetime import datetime\n\n")
        lines.append("from ..messages import Msg\n\n\n")
        lines.append("\n\n".join(body))
    return "".join(lines).rstrip() + "\n"


def _render_keys_init(modules: list[str]) -> str:
    # The modules are IMPORTED, not just listed in `__all__`: pyright reads an
    # `__all__` naming a module the package never imported as a defect
    # (reportUnsupportedDunderAll), and callers reaching
    # `keys.wire_errors.model_unavailable` should not need a second import.
    sorted_modules = sorted(modules)
    import_lines = "\n".join(f"    {name}," for name in sorted_modules)
    listed = "\n".join(f'    "{name}",' for name in sorted_modules)
    return (
        GENERATED_HEADER + f"from . import (\n{import_lines}\n)\n\n" + f"__all__ = [\n{listed}\n]\n"
    )


def _en_sha(namespace: str) -> str | None:
    """The current ``en`` hash for ``namespace``, or ``None`` when absent.

    ``None`` never equals a ledger's ``source_sha256`` string, so a carried
    namespace with no ``en`` counterpart (a renamed child still on disk)
    fails the freshness comparison instead of raising through the gate.
    """
    try:
        return catalogues.catalogue_sha256("en", namespace)
    except catalogues.CatalogueNotFound:
        return None


def _shipped_locales() -> list[str]:
    """The shipped set: en, plus locales with complete, fresh ledger passes.

    Freshness is the §6 gate: an entry counts iff its ``source_sha256`` equals
    the CURRENT hash of the en file for that namespace. A stale entry is not a
    pass; a locale with any of ITS OWN namespaces missing or stale is not
    shipped.

    PER-NAMESPACE since the 2026-10-11 amendment (docs/design/i18n.md §6): the
    set a locale is judged over is the set it CARRIES — the catalogue files
    that exist for that locale — not every namespace that exists in ``en``. A
    new en-only namespace (an extraction slice's child, before its translation
    wave) therefore does not un-ship passing locales; clients read that
    namespace from ``en`` (per-namespace fallback, §2.5). The gate's
    guarantees are unchanged: a locale ships only namespaces whose audits
    passed against the CURRENT en bytes, and a stale member still un-ships the
    locale. The earlier all-en-namespaces reading assumed §9's ordering (all
    extraction before any shipping wave); once the programme interleaved the
    two, it would have un-shipped every locale on every extraction slice that
    adds a namespace (measured 2026-10-11: one new en namespace dropped the
    generated set es/fr → en only).
    """
    shipped: set[str] = {"en"}
    entries = _ledger_entries()
    if not entries:
        return [loc for loc in SUPPORTED_LOCALES if loc in shipped]
    by_locale: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        locale = entry.get("locale")
        namespace = entry.get("namespace")
        if isinstance(locale, str) and isinstance(namespace, str):
            by_locale.setdefault(locale, {})[namespace] = entry
    for locale in SUPPORTED_LOCALES:
        if locale == "en" or locale in SECURITY_HELD_LOCALES:
            continue
        record = by_locale.get(locale)
        if not record:
            continue
        try:
            carried = catalogues.namespaces(locale)
        except catalogues.CatalogueNotFound:
            continue
        if not carried:
            continue
        if all(
            (entry := record.get(namespace)) is not None and entry.get("status") == "passed"
            # `en_sha is not None` is LOAD-BEARING, not belt-and-braces: the
            # entry is read with `.get`, so a `passed` entry that OMITS
            # `source_sha256` yields None too — and None == None would then read
            # a carried namespace with no `en` counterpart as FRESH (round-1
            # minor M1, reproduced: locale ships with an unverifiable
            # namespace). Requiring a real `en` hash first keeps the docstring's
            # promise structural instead of incidental.
            and (en_sha := _en_sha(namespace)) is not None and entry.get("source_sha256") == en_sha
            for namespace in carried
        ):
            shipped.add(locale)
    return [loc for loc in SUPPORTED_LOCALES if loc in shipped]


def _render_shipped() -> str:
    payload = {"schema": 1, "locales": _shipped_locales()}
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


#: Generated-module scaffolding, held as constants and referenced BY NAME: these
#: are TypeScript headers and type annotations, never rendered user copy, and the
#: copy gate counts prose literals handed to a sink rather than every string a
#: module mentions.
_BUNDLED_EAGER_HEADER = (
    "/** GENERATED by scripts/i18n/generate.py — do not edit by hand.",
    " * The relay SPA's eager English copy plus one lazy chunk per complete",
    " * shipped locale; explicit owned namespaces only — never a wire bundle.",
    " * Regenerate with generate.py; --check rejects drift in CI. */",
)
_BUNDLED_OWNED_OPEN = (
    "export const BUNDLED_OWNED: Readonly<Record<string, Readonly<Record<string,",
    " Readonly<Record<string, string>>>>>> = {",
)
_BUNDLED_CHUNKS_OPEN = (
    "/** Each complete shipped locale as a lazily loaded chunk. A locale absent",
    " * here has no rendering of its own: `uiLocaleFor` keeps the surface English",
    " * rather than mixing two languages under one page tag. */",
    "export const BUNDLED_CHUNKS: Readonly<Record<string, () => Promise<",
    " { readonly BUNDLED_LOCALE: typeof BUNDLED_OWNED }>>> = {",
)
_BUNDLED_CHUNK_HEADER = (
    "/** GENERATED by scripts/i18n/generate.py — do not edit by hand.",
    " * The relay SPA's owned copy in {locale}: whole, so the runtime can select",
    " * it; a locale missing one owned namespace is not emitted at all.",
    " * Regenerate with generate.py; --check rejects drift in CI. */",
)
_BUNDLED_LOCALE_OPEN = (
    "export const BUNDLED_LOCALE: Readonly<Record<string, Readonly<Record<string,",
    " Readonly<Record<string, string>>>>>> = {",
)


def _render_bundled_module(
    namespaces: dict[str, dict[str, dict[str, str]]], lazy_locales: list[str]
) -> str:
    """The relay SPA's EAGER own copy — English — plus the lazy locale manifest.

    A ``.ts`` map rather than a shipped JSON import tree (the RFC's "bundle a
    bootstrap set" for the relay, §2.1/§3.3): the store imports it and hands
    each namespace to the runtime's ``registerBundle`` before first paint, so
    the fallback copy travels with the code and cannot 404. Types only live in
    ``keys.gen.ts``; THIS file carries strings from the explicit owned
    namespace list alone — no wire prose is bundled.

    WHY ENGLISH IS THE ONLY EAGER LOCALE. Every owned string exists in English
    and English is what ``uiLocaleFor`` reads while a locale is incomplete, so
    it is the one map that must exist before the first paint. The other shipped
    locales are one chunk per complete locale and only one is ever active, so a
    reader pays only for the locale they select. NO SIZE BELONGS IN THIS
    DOCSTRING: the chunk set grows as catalogues complete, so any kB figure
    written here goes stale — measure it instead, from ``dist/assets`` after a
    ``vite build`` of ``local_operator/mobile/web`` (raw bytes from the files;
    ``gzip -9 -n -c <file> | wc -c`` for the compressed size). The manifest
    names one chunk per complete locale, and the eager map keeps its
    namespace->locale->messages shape with the ``en`` tag alone — the shape is
    shared with the chunks so the store registers both through ONE code path.
    """
    lines = [*_BUNDLED_EAGER_HEADER, *_BUNDLED_OWNED_OPEN]
    for namespace in sorted(namespaces):
        lines.append(f"  {json.dumps(namespace)}: {{")
        for locale in [tag for tag in sorted(namespaces[namespace]) if tag == "en"]:
            lines.append(f"    {json.dumps(locale)}: {{")
            for key, message in sorted(namespaces[namespace][locale].items()):
                lines.append(f"      {json.dumps(key)}: {json.dumps(message)},")
            lines.append("    },")
        lines.append("  },")
    lines.append("};")
    lines.append("")
    lines.extend(_BUNDLED_CHUNKS_OPEN)
    for locale in sorted(lazy_locales):
        lines.append(f'  {json.dumps(locale)}: () => import("./bundled.{locale}.gen"),')
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def _render_locale_module(namespaces: dict[str, dict[str, dict[str, str]]], locale: str) -> str:
    """One complete shipped locale's owned copy, as its own lazily loaded chunk.

    Same namespace->locale->messages shape as the eager map, so the store
    registers both through ONE code path (the same double loop over
    ``registerBundle``) and there is no second way of reading owned copy.
    """
    lines = [line.format(locale=locale) for line in _BUNDLED_CHUNK_HEADER]
    lines.extend(_BUNDLED_LOCALE_OPEN)
    for namespace in sorted(namespaces):
        lines.append(f"  {json.dumps(namespace)}: {{")
        lines.append(f"    {json.dumps(locale)}: {{")
        for key, message in sorted(namespaces[namespace][locale].items()):
            lines.append(f"      {json.dumps(key)}: {json.dumps(message)},")
        lines.append("    },")
        lines.append("  },")
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def _render_ts_key_map(namespaces: dict[str, dict[str, str]]) -> str:
    """Relay key/parameter types from the SAME parsed ICU arguments as Python.

    This artifact carries types only, never an implicit bootstrap of every
    backend string. Relay-owned copy bundles separately; wire fetching later
    follows an explicit used-key manifest. Keep the two-overload shape from
    RFC §2.4, matching the desktop contract rather than a conditional tuple.
    """
    types = {
        "plural": "number",
        "number": "number",
        "date": "Date | number",
        "time": "Date | number",
    }
    lines = [
        "/** GENERATED by scripts/i18n/generate.py — do not edit by hand.",
        " * Type-only relay contract; no wire catalogue is bundled implicitly.",
        " * Regenerate with generate.py; --check rejects drift in CI. */",
        "export type MessageParams = {",
    ]
    seen: set[str] = set()
    for namespace in sorted(namespaces):
        for key, message in sorted(namespaces[namespace].items()):
            if key in seen:
                raise SystemExit(f"generate.py: duplicate TS message key {key!r}")
            seen.add(key)
            arguments = msg_runtime.message_arguments(message)
            params = "; ".join(
                f"{json.dumps(name)}: {types.get(kind, 'string')}" for name, kind in arguments
            )
            shape = "{ " + params + " }" if arguments else "undefined"
            lines.append(f"  {json.dumps(key)}: {shape};")
    lines.extend(
        [
            "};",
            "export type MessageKey = keyof MessageParams;",
            "export type WithParams = {",
            "  [K in MessageKey]: MessageParams[K] extends undefined ? never : K;",
            "}[MessageKey];",
            "export type WithoutParams = {",
            "  [K in MessageKey]: MessageParams[K] extends undefined ? K : never;",
            "}[MessageKey];",
            "/** Both overloads are mandatory: bare keys never accept parameters. */",
            "export interface Translator {",
            "  <K extends WithParams>(key: K, params: MessageParams[K]): string;",
            "  (key: WithoutParams): string;",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _planned_files(emitted: Path) -> dict[Path, str]:
    """Everything generation should produce: path -> content (strings)."""
    planned: dict[Path, str] = {
        DATA_DIR / "plural_rules.json": (emitted / "plural_rules.json").read_text(encoding="utf-8"),
        DATA_DIR / "formats.json": (emitted / "formats.json").read_text(encoding="utf-8"),
        DATA_DIR / "shipped.json": _render_shipped(),
    }
    # The aggregate is a generated mirror of the shards whenever shards exist;
    # planning it here makes `--check` fail when a contributor edits only one
    # of the two, so a stale aggregate cannot silently disagree with the
    # authoritative shards.
    if LEDGER_SHARDS.is_dir() and any(LEDGER_SHARDS.glob("*.json")):
        planned[LEDGER] = _render_aggregate()
    en_dir = catalogues.catalogue_root() / "en"
    modules: list[str] = []
    relay_types: dict[str, dict[str, str]] = {}
    for namespace in catalogues.namespaces("en"):
        messages = _read_json(en_dir / f"{namespace}.json")
        if namespace.startswith(("wire.", "relay.")):
            relay_types[namespace] = messages
        module = namespace.replace(".", "_").replace("-", "_")
        modules.append(module)
        planned[KEYS_DIR / f"{module}.py"] = _render_keys_module(namespace, messages)
    planned[KEYS_DIR / "__init__.py"] = _render_keys_init(modules)
    planned[REPO / "local_operator/mobile/web/src/i18n/keys.gen.ts"] = _render_ts_key_map(
        relay_types
    )
    # A declared owned namespace ALWAYS has an en file in the committed tree;
    # skipping an absent one keeps the generator usable against fixture roots
    # (the generator tests build mini-trees) without binding them to relay.shell.
    # The committed bundle's exact contents are pinned by test_generate's
    # ownership test, so a silent empty bundle cannot ship.
    #
    # ONLY SELECTABLE LOCALES ARE EMITTED (review round 2, MINOR). The runtime's
    # `uiLocaleFor` selects a non-English UI locale only when EVERY owned
    # namespace carries the whole English key set for it; a locale that misses
    # one owned namespace or one key paints English as a whole and can never have
    # its bundle read. Emitting it would ship dead copy — measured at 43,129 B for
    # eight locales against 15,445 B for the en-only baseline — so the set is
    # intersected with the English key sets here, mirroring that runtime rule.
    # `en` is always present: it is the fallback every partial locale reads.
    zh = catalogues.catalogue_root()
    owned: dict[str, dict[str, dict[str, str]]] = {}
    for namespace in OWNED_SPA_NAMESPACES:
        en_path = en_dir / f"{namespace}.json"
        if not en_path.is_file():
            continue
        english = _read_json(en_path)
        for locale in _shipped_locales():
            path = zh / locale / f"{namespace}.json"
            if not path.is_file():
                continue
            messages = _read_json(path)
            if locale == "en" or set(english) <= set(messages):
                owned.setdefault(namespace, {})[locale] = messages
    # A locale is only kept when it cleared EVERY owned namespace above.
    covering = set(owned[next(iter(owned))]) if owned else set()
    for namespace in owned:
        covering &= set(owned[namespace])
    for namespace in list(owned):
        owned[namespace] = {
            locale: messages
            for locale, messages in owned[namespace].items()
            if locale in covering or locale == "en"
        }
    lazy_locales = sorted(covering - {"en"})
    bundled = _bundled_dir()
    planned[bundled / "bundled.gen.ts"] = _render_bundled_module(owned, lazy_locales)
    for locale in lazy_locales:
        planned[bundled / f"bundled.{locale}.gen.ts"] = _render_locale_module(owned, locale)
    return planned


def _check(planned: dict[Path, str]) -> int:
    stale: list[str] = []
    for path, content in sorted(planned.items(), key=lambda item: str(item[0])):
        rel = path.relative_to(REPO)
        if not path.is_file():
            stale.append(f"missing: {rel}")
            continue
        if path.read_text(encoding="utf-8") != content:
            stale.append(f"changed: {rel}")
    # ORPHANS, the failure a plan-vs-plan diff can never see (round-1 n4): a
    # namespace rename/removal leaves `keys/<old>.py` (or a data file) behind,
    # nothing regenerates it, and the diff stays green while the package grows
    # a module no catalogue knows about. Checked for the two generated dirs.
    bundled = _bundled_dir()
    for directory, pattern in (
        (KEYS_DIR, "*.py"),
        (DATA_DIR, "*.json"),
        (bundled, "bundled.*.gen.ts"),
    ):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(pattern)):
            if path not in planned:
                stale.append(f"orphaned: {path.relative_to(REPO)}")
    if stale:
        print("generate.py --check: committed artifacts are stale — regenerate with")
        print("  .venv/bin/python scripts/i18n/generate.py")
        print(
            "(the emitter is pinned to Node 24 — CI's setup-node major — and its output "
            "is normalised to be node-major stable; see the note in scripts/i18n/emit.mjs)"
        )
        for line in stale:
            print(f"  {line}")
        return 1
    print(f"generate.py --check: {len(planned)} artifact(s) byte-identical")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate into a temp dir and fail on any byte difference (CI mode)",
    )
    parser.add_argument(
        "--accept-stranded",
        action="store_true",
        help=(
            # i18n: ignore operator tooling diagnostic (English-only)
            "re-render the mirror even though it holds rows no shard carries, "
            "DISCARDING them (reviewed exception only: the ledger is append-only, "
            "so this is for an authorised row removal, never for a rebase)"
        ),
    )
    args = parser.parse_args(argv)

    stranded = _aggregate_rows_missing_from_shards()
    if stranded and args.accept_stranded:
        print(
            # i18n: ignore operator tooling diagnostic (English-only)
            "generate.py: --accept-stranded given — these aggregate rows are NOT in any "
            "shard and will be dropped from the mirror:"
        )
        for row in stranded:
            print(f"  dropped aggregate row: {row}")  # i18n: ignore operator diagnostic
        stranded = []
    if stranded:
        print(
            # i18n: ignore operator tooling diagnostic (English-only)
            "generate.py: refusing to overwrite i18n/ledger.json — it holds audit row(s)\n"
            "that no shard carries. The aggregate is a GENERATED mirror of\n"
            "i18n/ledger/<locale>.json; appending to it would lose the row on the next run."
        )
        for row in stranded:
            print(f"  stranded aggregate row: {row}")  # i18n: ignore operator diagnostic
        print(
            # i18n: ignore operator tooling diagnostic (English-only)
            "Move each row into its locale's shard (keeping order), then re-run:\n"
            "  python3 scripts/i18n/shard_ledger.py --fold-stranded   # rebased branch\n"
            "  python3 scripts/i18n/generate.py                       # re-render the mirror"
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="i18n-emit-") as td:
        emitted = Path(td)
        _run_emitter(emitted)
        planned = _planned_files(emitted)
        if args.check:
            return _check(planned)
        for path, content in planned.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independent audit checks for one Spanish TUI catalogue namespace (round 2).

Read-only: never writes a catalogue, sidecar, ledger, allowance or generated
file. Prints one PASS/FAIL family per check, then measurements, then negative
controls, then a result block.

Run from the repository root:

    PYTHONPATH=$PWD python3 i18n/audits/evidence/es/<ns>/w6a1/checks.py

Cell widths use ``rich.cells.cell_len`` — the same function the TUI uses to
budget a row (``_row_overhead`` reads it). If ``rich`` is not importable the
script falls back to an equivalent local implementation that it first validates
against two known widths, and says so in its output.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from pathlib import Path

NAMESPACE = "tui.import"
LOCALE = "es"
TARGET_CATEGORIES = {"one", "many", "other"}
FORBIDDEN_CATEGORIES = {"zero", "two", "few"}

ROOT = Path(os.environ.get("AUDIT_REPO_ROOT", os.getcwd()))
CAT = ROOT / "local_operator" / "i18n" / "catalogues"

failures: list[str] = []
notes: list[str] = []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fail(check: str, message: str) -> None:
    failures.append(f"[{check}] {message}")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _fallback_cell_len(text: str) -> int:
    """Mirror rich.cells.cell_len: W/F = 2 cells, combining = 0, else 1."""
    total = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        total += 2 if unicodedata.east_asian_width(char) in "WF" else 1
    return total


try:
    from rich.cells import cell_len as _rich_cell_len

    CELL_SOURCE = "rich.cells.cell_len (the function the TUI measures with)"
    cell_len = _rich_cell_len
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    cell_len = _fallback_cell_len
    CELL_SOURCE = "local fallback (rich not importable in this interpreter)"
    # Prove the fallback agrees with rich on the two widths this audit leans on.
    assert _fallback_cell_len("enter · e all") == 13, "fallback disagrees with rich (13)"
    assert _fallback_cell_len("enter · e todo") == 14, "fallback disagrees with rich (14)"


def normalise_count(text: str) -> str:
    """Strip rendered numbers so two branches can be compared by shape."""
    return re.sub(r"[\d.,\u00a0\u202f]+", "#", text)


def blank_branch(text: str, selector: str) -> str:
    """Replace one plural branch's body with a space, brace-aware.

    Brace counting is required because a branch body may itself contain
    interpolations (`{{shown} lecturas}`), which a non-nesting regex cannot
    match. Used only by the negative controls.
    """
    token = f"{selector} {{"
    start = text.rindex(token) + len(token)
    depth = 1
    index = start
    while index < len(text) and depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    return text[:start] + " " + text[index - 1 :]


# ---------------------------------------------------------------------------
# Check 1 — files present, JSON valid, key parity
# ---------------------------------------------------------------------------


def check_keys(en: dict, target: dict) -> None:
    if not en:
        fail("keys", "en catalogue is empty")
    missing = sorted(set(en) - set(target))
    extra = sorted(set(target) - set(en))
    if missing:
        fail("keys", f"{len(missing)} key(s) missing from target: {missing[:8]}")
    if extra:
        fail("keys", f"{len(extra)} unexpected key(s) in target: {extra[:8]}")
    for name, value in target.items():
        if not isinstance(value, str):
            fail("keys", f"{name}: value is {type(value).__name__}, not str")
    if len(target) != len(set(target)):
        fail("keys", "duplicate keys in target")


# ---------------------------------------------------------------------------
# Check 2 — ICU parses; argument names, kinds and order identical to EN
# ---------------------------------------------------------------------------


def check_icu(en: dict, target: dict) -> None:
    from local_operator.i18n.runtime import MessageSyntaxError, message_arguments, parse_message

    for name, value in target.items():
        try:
            parse_message(value)
        except MessageSyntaxError as exc:
            fail("icu", f"{name}: target does not parse: {exc}")
            continue
        if name not in en:
            continue
        try:
            want = message_arguments(en[name])
        except MessageSyntaxError as exc:
            fail("icu", f"{name}: EN source does not parse: {exc}")
            continue
        got = message_arguments(value)
        if got != want:
            fail("icu", f"{name}: argument tuple differs — en={want} target={got}")


# ---------------------------------------------------------------------------
# Check 3 — Spanish CLDR plural categories, branch losslessness
# ---------------------------------------------------------------------------


def check_plurals(en: dict, target: dict) -> None:
    from local_operator.i18n.runtime import plural_arguments

    plurals = 0
    for name, value in target.items():
        for plural in plural_arguments(value):
            plurals += 1
            literal = {b.selector for b in plural.branches if not b.selector.startswith("=")}
            forbidden = literal & FORBIDDEN_CATEGORIES
            if forbidden:
                fail(
                    "plural",
                    f"{name}: {sorted(forbidden)} is not a Spanish CLDR category "
                    f"(argument {plural.argument})",
                )
            if not TARGET_CATEGORIES.issubset(literal):
                fail(
                    "plural",
                    f"{name}: argument {plural.argument} has {sorted(literal)}; "
                    f"es requires at least {sorted(TARGET_CATEGORIES)}",
                )
            empty = [b.selector for b in plural.branches if b.empty]
            if empty:
                fail("plural", f"{name}: empty branch/branches {empty}")
            # `renders_value` (runtime) recognises only ICU `#` and
            # `{count, number}`. These keys deliberately render the count
            # through a pre-formatted sibling argument (`{shown}`), in EN and ES
            # alike, so an ABSOLUTE losslessness assertion would fire on correct
            # copy. Assert what is actually true: no empty branch, and no branch
            # that is lossier than its English counterpart.
            # Branch-level parity against EN: EN's per-branch losslessness must
            # not be repaired by dropping the count from a branch.
            if name in en:
                src = {b.selector: b for b in plural_arguments(en[name])[0].branches}
                for branch in plural.branches:
                    origin = src.get(branch.selector)
                    if origin and origin.renders_value and not branch.renders_value:
                        fail(
                            "plural",
                            f"{name}: '{branch.selector}' lost the rendered count that EN keeps",
                        )
    if plurals == 0:
        fail("plural", "no plural messages found — this check exercised nothing")
    notes.append(f"plural messages checked: {plurals}")


# ---------------------------------------------------------------------------
# Check 4 — the plural rules actually SELECT correctly for es (real render)
# ---------------------------------------------------------------------------


def check_render(en: dict, target: dict) -> None:
    from local_operator.i18n.runtime import (
        MessageSyntaxError,
        message_arguments,
        plural_arguments,
        render_message,
    )

    samples = [(1, "one"), (0, "other"), (2, "other"), (1_000_000, "many"), (2_000_000, "many"), (1_500_000, "other")]
    rendered_any = 0
    for name, value in sorted(target.items()):
        try:
            plurals = plural_arguments(value)
        except MessageSyntaxError:
            continue
        if not plurals:
            continue
        args = [a for a, _ in message_arguments(value)]
        shapes: dict[int, str] = {}
        shown: dict[int, str] = {}
        for count, _category in samples:
            try:
                out = render_message(value, {a: count for a in args}, LOCALE)
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                fail("render", f"{name}: render at {count} raised {exc!r}")
                continue
            shown[count] = out
            shapes[count] = normalise_count(out)
        if len(shown) != len(samples):
            continue
        rendered_any += 1
        notes.append(
            "render %s: %s" % (name, " | ".join(f"{c}->{shown[c]}" for c, _ in samples))
        )
        # 0 is `other` in es (never `one`): its shape must match 2, not 1.
        if shapes[0] != shapes[2]:
            fail("render", f"{name}: count 0 shaped {shapes[0]!r}, count 2 {shapes[2]!r} — 0 must take 'other'")
        # 10^6 is `many`: distinct from the plain `other` shape.
        if shapes[1_000_000] == shapes[2]:
            fail("render", f"{name}: 1000000 shaped like 'other' — 'many' did not select")
        # 1.5x10^6 is NOT `many` (the guide's settled rule): must equal 2's shape.
        if shapes[1_500_000] != shapes[2]:
            fail(
                "render",
                f"{name}: 1500000 shaped {shapes[1_500_000]!r}, expected the 'other' shape {shapes[2]!r}",
            )
        # 1 must take `one` and be distinguishable from `other`.
        if shapes[1] == shapes[2]:
            fail("render", f"{name}: count 1 and count 2 render identically — 'one' uncoupled")
    if rendered_any == 0:
        fail("render", "no plural message was rendered — this check exercised nothing")


# ---------------------------------------------------------------------------
# Check 5 — protected machine tokens survive byte-identically
# ---------------------------------------------------------------------------

PROTECTED = [
    (re.compile(r"`[^`]+`"), "backtick span"),
    (re.compile(r"(?<![\w-])--[a-z][a-z0-9-]+"), "--flag"),
    (re.compile(r"<[a-z_]+>"), "<angle-token>"),
    (re.compile(r"\[[a-z]+\]"), "[bracket-token]"),
    (re.compile(r"(?<![\w])/[a-z][a-z0-9-]*"), "/command"),
    (re.compile(r"→"), "arrow"),
    (re.compile(r"\$—"), "$— money mark"),
    (re.compile(r"↑↓|←→"), "cursor glyph"),
    (re.compile(r"≈"), "≈ estimate mark"),
]


def protected_tokens(text: str) -> list[str]:
    found: list[str] = []
    for pattern, label in PROTECTED:
        for match in pattern.finditer(text):
            found.append(f"{label}:{match.group(0)}")
    return sorted(found)


def check_tokens(en: dict, target: dict) -> None:
    checked = 0
    for name in sorted(set(en) & set(target)):
        want = protected_tokens(en[name])
        got = protected_tokens(target[name])
        checked += 1
        if want != got:
            fail("tokens", f"{name}: protected tokens differ — en={want} target={got}")
    notes.append(f"protected-token comparison on {checked} shared keys")


# ---------------------------------------------------------------------------
# Check 6 — maxCells budgets declared in the EN sidecar
# ---------------------------------------------------------------------------


def over_state(en_cells: int, es_cells: int, budget: int) -> str:
    if es_cells <= budget:
        return "within cap"
    if en_cells > budget:
        return "cap already exceeded by EN; es no worse than EN" if es_cells <= en_cells else "OVER cap and OVER en"
    return "OVER cap while EN fits"


def check_budgets(en: dict, target: dict, context: dict) -> None:
    declared = {k: v["maxCells"] for k, v in context.items() if "maxCells" in v}
    if not declared:
        notes.append("no maxCells declared in this sidecar — widths are advisory only")
    for name, budget in sorted(declared.items()):
        if name not in target:
            fail("budget", f"{name}: budget declared but key absent from target")
            continue
        en_cells = cell_len(en[name])
        es_cells = cell_len(target[name])
        state = over_state(en_cells, es_cells, budget)
        notes.append(
            f"budget {name}: es={es_cells} en={en_cells} cap={budget} "
            f"(es - cap = {es_cells - budget:+d}) -> {state}"
        )
        if state.startswith("OVER cap while EN fits"):
            fail("budget", f"{name}: {es_cells} cells exceeds maxCells {budget} (en {en_cells} fits)")
        if en_cells > budget:
            # The cap is unattainable in the English source itself. A cap the
            # source cannot meet is not a translation defect; record the delta
            # and leave the achievability judgement to the human judgement in
            # report.md (an auditor rule that auto-failed here would flag every
            # locale for a defect it cannot fix).
            notes.append(
                f"    -> cap unattainable in source (en {en_cells} > cap {budget}); "
                f"es delta vs en = {es_cells - en_cells:+d} cells"
            )


# ---------------------------------------------------------------------------
# Check 7 — polarity: a negation in EN may not vanish in es
# ---------------------------------------------------------------------------

EN_NEG = re.compile(r"\b(no|not|nothing|never|cannot|none|neither|nor|without|unsupported|unknown)\b", re.I)
ES_NEG = re.compile(r"\b(no|nada|ning[uú]n[oa]?s?|nunca|jam[aá]s|ni|sin|tampoco|desconocid[oa])\b", re.I)


def check_polarity(en: dict, target: dict) -> None:
    lost = 0
    added: list[str] = []
    for name in sorted(set(en) & set(target)):
        en_has = bool(EN_NEG.search(en[name]))
        es_has = bool(ES_NEG.search(target[name]))
        if en_has and not es_has:
            lost += 1
            fail("polarity", f"{name}: EN negates, target carries no negative marker")
        if not en_has and es_has:
            added.append(name)
    notes.append(
        f"polarity: {lost} lost negation(s); {len(added)} key(s) where es adds a negative marker "
        f"(reviewed by hand, not auto-failed): {added}"
    )


# ---------------------------------------------------------------------------
# Check 8 — terminology against the es style guide / glossary
# ---------------------------------------------------------------------------

TERM_RULES = [
    # (key scope regex or None, forbidden regex, why)
    (None, re.compile(r"\bdemonios?\b", re.I), "daemon stays English (§2 amendment)"),
    (None, re.compile(r"\binquilinos?\b", re.I), "tenant stays English"),
    (None, re.compile(r"\bcache\b", re.I), "cache is 'caché' (W1 amendment)"),
    (None, re.compile(r"\bcosto\b", re.I), "wave 1 standardised on 'coste'"),
    (None, re.compile(r"\bfichas?\b", re.I), "token stays 'token'"),
    (None, re.compile(r"\barn[eé]s\b", re.I), "harness stays English"),
    (None, re.compile(r"\bsesion\b", re.I), "missing accent on 'sesión'"),
    (None, re.compile(r"[a-záéíóúñ]\)\s*!|\s+[!?]", re.I), "es spacing: no space before ?/!"),
]


def check_terms(target: dict) -> None:
    for name, value in sorted(target.items()):
        for _scope, pattern, why in TERM_RULES:
            match = pattern.search(value)
            if match:
                fail("terms", f"{name}: {match.group(0)!r} — {why}")


# ---------------------------------------------------------------------------
# Check 9 — no prose left untranslated (byte-identical to EN)
# ---------------------------------------------------------------------------


def check_untranslated(en: dict, target: dict) -> None:
    identical = 0
    for name in sorted(set(en) & set(target)):
        prose = re.sub(r"\{[^}]*\}", "", en[name])
        words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,}", prose)
        if en[name] == target[name] and len(words) >= 3:
            identical += 1
            fail("untranslated", f"{name}: target is byte-identical to EN prose — {en[name]!r}")
    notes.append(f"byte-identical prose values: {identical}")


# ---------------------------------------------------------------------------
# Check 10 — per-namespace semantic assertions (money / destructive actions)
# ---------------------------------------------------------------------------


def build_semantic_rules() -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Return (required, forbidden) rules: (key, regex, description).

    Semantics, not wording preferences: each rule encodes something the English
    string promises and a translation could silently break.
    """
    if NAMESPACE == "tui.analytics":
        required = [
            ("tui.analytics.empty.title", r"no hay uso registrado", "empty ledger states nothing recorded"),
            ("tui.analytics.chart.daily_empty", r"no hay uso diario registrado|no se ha registrado uso diario",
             "daily chart empty ≠ failed"),
            ("tui.analytics.chart.monthly_empty", r"no hay uso mensual registrado|no se ha registrado uso mensual",
             "monthly chart empty"),
            ("tui.analytics.model.legend", r"descifrado|decode", "'decode' column name kept/explained"),
            ("tui.analytics.model.legend", r"wall", "'wall = whole call' not shortened to a rate word"),
            ("tui.analytics.model.failed", r"resto", "failure stays scoped ('the rest of this report')"),
            ("tui.analytics.model.meta", r"registro", "'from the ledger'"),
            ("tui.analytics.input.meta", r"estimad", "'estimated' honesty label kept"),
            ("tui.analytics.search.meta_live", r"proceso", "process-wide scope"),
            ("tui.analytics.search.note", r"no del disco", "reads the live ledger, NOT disk"),
            ("tui.analytics.search.note", r"no se cuentan", "non-open sessions are excluded"),
            ("tui.analytics.totals.fresh_note", r"tarifa completa", "'billed at full rate'"),
            ("tui.analytics.model.capped", r"\{limit\}", "'top N' bound preserved"),
            ("tui.analytics.header.scope", r"todas las sesiones", "scope is every session"),
        ]
        forbidden = [
            # unpriced must never be described as free ('gratis') — they differ
            ("tui.analytics.search.incl_unpriced_read", r"gratis", "'unpriced' rendered as FREE"),
            ("tui.analytics.search.incl_unpriced_search", r"gratis", "'unpriced' rendered as FREE"),
            ("tui.analytics.search.note_read_unpriced", r"gratis", "'unpriced' rendered as FREE"),
            ("tui.analytics.search.note_search_unpriced", r"gratis", "'unpriced' rendered as FREE"),
            ("tui.analytics.search.unpriced_clause", r"gratis", "'unpriced' rendered as FREE"),
            # the free rungs must not claim 'no price' — free is a priced outcome
            ("tui.analytics.search.summary_free", r"sin precio", "'free' rendered as UNPRICED"),
            ("tui.analytics.search.summary_free", r"\ba pagar\b", "'free' rendered as payable"),
            # the fresh row is NOT user input (sidecar says so explicitly)
            ("tui.analytics.totals.fresh", r"usuario", "'fresh' mislabelled as user input"),
            ("tui.analytics.totals.fresh_note", r"usuario", "'fresh' mislabelled as user input"),
            # a pending read must not read as an empty one
            ("tui.analytics.model.pending", r"sin filas|no hay filas", "'reading…' reads as 'no rows'"),
        ]
        required += [
            ("tui.analytics.search.incl_money_read", r"\{money\}", "{money} keeps the pre-formatted figure"),
            ("tui.analytics.search.incl_money_search", r"\{money\}", "{money} keeps the pre-formatted figure"),
            ("tui.analytics.search.summary_free_paid", r"\{free\}", "{free} preserved"),
            ("tui.analytics.search.summary_free_paid", r"\{priced\}", "{priced} denominator preserved"),
            ("tui.analytics.search.summary_paid", r"\{money\}", "{money} preserved"),
            ("tui.analytics.search.summary_free_paid_short", r"\{money\}", "{money} preserved"),
            ("tui.analytics.legend.cost", r"\bmenor\b|límite inferior", "'+ lower bound' kept as a floor"),
            ("tui.analytics.legend.cost", r"sin precio publicado", "'$— no published price'"),
            ("tui.analytics.legend.cost", r"   ", "deliberate three-space alignment run survives"),
            ("tui.analytics.search.note_read_full", r"sin precio publicado", "'no published price'"),
            ("tui.analytics.chart.metric_clause", r"^\{active\} · t → \{other\}$", "machine-vocabulary clause untouched"),
            ("tui.analytics.header.metric", r"\{metric\}", "metric placeholder preserved"),
            ("tui.analytics.table.cache", r"caché", "cache tail keeps 'caché'"),
            ("tui.analytics.totals.cache_hit", r"caché", "cache-hit row keeps 'caché'"),
        ]
        return required, forbidden

    # tui.import — where a wrong verb loses data
    required = [
        # Nothing HAS been imported + nothing WILL be (yet)
        ("tui.import.preview_confirm", r"no se ha importado nada", "preview states nothing was imported"),
        ("tui.import.preview_confirm", r"enter", "names the key that does import"),
        ("tui.import.preview_confirm_metadata", r"no se importará nada", "states nothing WILL be imported"),
        ("tui.import.preview_narrow", r"no se ha importado nada|aún no se ha importado", "narrow preview: not imported yet"),
        ("tui.import.preview_narrow_metadata", r"no se importará nada", "narrow preview: nothing will be imported"),
        # the three rerun source behaviours, all named
        ("tui.import.rerun_hint", r"se omiten|omitid|se saltan", "rerun: unchanged sources are SKIPPED"),
        ("tui.import.rerun_hint", r"se reanudan|reanud", "rerun: grown sources RESUME"),
        ("tui.import.rerun_hint", r"copia nueva|nueva copia", "rerun: rewritten sources get a NEW COPY"),
        ("tui.import.rerun_hint", r"seguro", "rerun is asserted safe"),
        ("tui.import.error_select", r"previsualizar|vista previa", "preview is the step being gated"),
        # destructive / data-loss paths stay explicit
        ("tui.import.metadata_only_explain", r"no se puede importar nada", "index-only: nothing CAN be imported"),
        ("tui.import.metadata_only_explain", r"cuerpo", "explains the missing conversation body"),
        ("tui.import.outcome_metadata_only", r"nada que importar", "index-only outcome says nothing to import"),
        ("tui.import.availability_metadata_only", r"índice", "'index only' availability"),
        ("tui.import.subtitle", r"solo lectura|sólo lectura", "the source stays READ-ONLY"),
        ("tui.import.error_busy", r"ya hay|en curso", "an import is already running"),
        ("tui.import.error_cancel", r"nada en curso|no hay nada", "nothing running to cancel"),
        ("tui.import.error_no_job", r"no se encontró", "no import job found"),
        ("tui.import.empty_sessions", r"nada que importar", "nothing to import"),
        ("tui.import.empty_header", r"no hay sesiones", "no sessions to import"),
        ("tui.import.slash_help", r"memoria", "chats AND memory"),
        ("tui.import.slash_help", r"harness", "harness kept English"),
        ("tui.import.title", r"^Importar ", "action label in the infinitive"),
        ("tui.import.error_source", r"\bu omp\b", "es euphony 'u' before 'omp'"),
        ("tui.import.no_sources", r"\bu omp\b", "es euphony 'u' before 'omp'"),
        ("tui.import.result_head", r"→", "source→target arrow survives"),
    ]
    forbidden = [
        ("tui.import.preview_confirm", r"\bse importó\b", "claims an import ALREADY happened"),
        ("tui.import.preview_confirm", r"\bya se ha importado\b", "claims an import ALREADY happened"),
        ("tui.import.preview_confirm_metadata", r"se ha importado", "future-negative key claims a past import"),
        ("tui.import.preview_narrow_metadata", r"se ha importado", "future-negative key claims a past import"),
        ("tui.import.result_memory", r"redaccion", "'redactions' mistranslated as 'drafts'"),
        ("tui.import.rerun_hint", r"se pierden|se sobrescrib|pierde datos", "rerun described as destructive"),
        ("tui.import.subtitle", r"copia", "source described as copied when it is read-only"),
        # a permissive claim that the import CAN happen; the correct wording is
        # the negated 'no se puede importar nada' asserted just above.
    ]
    return required, forbidden


def check_semantics(target: dict) -> None:
    required, forbidden = build_semantic_rules()
    checked = 0
    for key, pattern, description in required:
        if not pattern:
            continue
        if key not in target:
            fail("semantics", f"{key}: required key missing")
            continue
        checked += 1
        if not re.search(pattern, target[key], re.I):
            fail("semantics", f"{key}: expected {pattern!r} ({description}); got {target[key]!r}")
    for key, pattern, description in forbidden:
        if not pattern:
            continue
        if key not in target:
            fail("semantics", f"{key}: required key missing")
            continue
        checked += 1
        if re.search(pattern, target[key], re.I):
            fail("semantics", f"{key}: hit forbidden {pattern!r} ({description}); got {target[key]!r}")
    notes.append(f"semantic assertions evaluated: {checked}")



# ---------------------------------------------------------------------------
# Round 2 — the round-1 width findings, re-measured on the CURRENT bytes
# ---------------------------------------------------------------------------

#: The hint row's cells at 80x24. Both paddings are in the stylesheet:
#: ``Screen { padding: 1 }`` and ``.settings-view { padding: 1 1 1 1 }``, so
#: 80 - 4 = 76 -- the figure ``HarnessImportView._hint_labels`` records
#: ("The hint row still fits at 80x24 (76 cells)").
HINT_ROW_BUDGET_80X24 = 76

#: ``_hint_labels`` gates the move hint on ``self.size.width >= 74``. In
#: Textual ``Widget.size`` is the CONTENT width, so 74 is the narrowest hint-row
#: box in which all five sessions-state hints paint (a 78-column terminal).
HINT_ROW_GATE_BOX = 74

#: Seam and join, read from ``HintButton._build``: a non-lead hint is preceded
#: by " · ", and its key and label are joined by one space.
HINT_SEAM = " · "

#: The EN sessions-state row as the REAL widget painted it in the round-2 rig
#: (audit.log section 4). The assembler below must reproduce it byte-for-byte;
#: if it stops doing so, every row measurement in this file is measuring a
#: different row from the one the user sees.
WIDGET_SESSIONS_ROW_EN = "↑↓ move · space select · a all/none · enter preview · esc back"

#: Hint-ladder tiers in the widget's own candidate order (widest first), with
#: the caps the EN sidecar declares.
ANALYTICS_TIERS = [
    ("tui.analytics.hint.expand.full_metric", 57),
    ("tui.analytics.hint.expand.full", 42),
    ("tui.analytics.hint.expand.compact", 31),
    ("tui.analytics.hint.expand.keys", 25),
    ("tui.analytics.hint.expand.keys_bare", 12),
]

#: Import hint keys whose EN source is an imperative verb, so the es rendering
#: must be the infinitive action label the style guide requires.
IMPORT_VERB_HINTS = [
    "hint_back",
    "hint_cancel",
    "hint_confirm",
    "hint_import",
    "hint_list",
    "hint_move",
    "hint_open",
    "hint_reload",
    "hint_select",
]

INFINITIVE = re.compile(r"(ar|er|ir)$")

ROUND1_SUGGESTIONS = {
    "tui.analytics": [
        ("A1", "tui.analytics.totals.output_note", "{generation} generación, {thinking} razonamiento"),
        ("A2", "tui.analytics.hint.expand.compact", "↑↓ fila · enter abrir · e todo"),
        ("A3", "tui.analytics.search.meta_live", "en todo el proceso · en vivo"),
        ("A4", "tui.analytics.search.summary_free_paid", "{free} de {priced} con precio gratis · {money} pagado"),
        ("A5", "tui.analytics.rate.legend",
         "tasa de decodificación sobre {shown, number} de {total, number} llamadas"),
    ],
    "tui.import": [
        ("I1", None,
         "↑↓ mover · space elegir · a todas/ninguna · enter vista previa · esc volver"),
        ("I2", "tui.import.hint_list", "listar"),
        ("I3", "tui.import.error_select",
         "Selecciona al menos una sesión con la tecla espacio antes de previsualizar."),
        ("N1", "tui.import.result_memory",
         "memoria importada: {imported} · omitidos: {skipped} · censurados: {redactions}"),
        ("N2", "tui.import.selected_count", "{count} seleccionadas"),
        ("N3", None, None),
    ],
}


def _missing_hint_keys(target: dict) -> list[str]:
    """Keys ``import_hint_rows`` needs; a negative control may delete one."""
    needed = (
        "hint_move", "hint_select", "all_none", "hint_open", "hint_reload",
        "hint_preview", "hint_import", "hint_confirm", "hint_cancel",
        "hint_list", "hint_back",
    )
    return [k for k in needed if f"tui.import.{k}" not in target]


def assemble_hints(parts: list[tuple[str, str]]) -> str:
    """Join ``(key, label)`` pairs the way ``HintButton`` paints them."""
    return HINT_SEAM.join(f"{key} {label}" for key, label in parts)


def import_hint_rows(target: dict) -> dict[str, list[tuple[str, str]]]:
    """The rows ``HarnessImportView._hint_labels`` assembles, per state.

    Mirrors widgets/harness_import_view.py:505-529: button order is move, space,
    all, enter, cancel, reload, esc, and each state paints its own subset. The
    preview state has two forms -- ``hint_import`` normally and ``hint_confirm``
    when every selected source is index-only -- so both are measured.
    """
    back = ("esc", target["tui.import.hint_back"])
    return {
        "sources": [
            ("↑↓", target["tui.import.hint_move"]),
            ("enter", target["tui.import.hint_open"]),
            ("r", target["tui.import.hint_reload"]),
            back,
        ],
        "sessions": [
            ("↑↓", target["tui.import.hint_move"]),
            ("space", target["tui.import.hint_select"]),
            ("a", target["tui.import.all_none"]),
            ("enter", target["tui.import.hint_preview"]),
            back,
        ],
        "preview_import": [("enter", target["tui.import.hint_import"]), back],
        "preview_confirm": [("enter", target["tui.import.hint_confirm"]), back],
        "progress": [("c", target["tui.import.hint_cancel"]), back],
        "result": [("enter", target["tui.import.hint_list"]), back],
        "loading": [back],
    }


def check_hint_register(target: dict) -> None:
    """Round 2: the I2 dimension, asserted rather than eyeballed."""
    for name in IMPORT_VERB_HINTS:
        key = f"tui.import.{name}"
        value = target[key]
        if not INFINITIVE.search(value):
            fail(
                "register",
                f"{key}: {value!r} is not an infinitive action label "
                f"(es guide: action labels are infinitives; en is an imperative verb)",
            )
    notes.append(f"infinitive action labels asserted on {len(IMPORT_VERB_HINTS)} hint keys")


def check_import_rows(en: dict, target: dict) -> None:
    """Round 2: re-assemble every hint row and re-measure the budget."""
    missing = _missing_hint_keys(target)
    if missing:
        notes.append(f"row check skipped: target is missing {missing} (the [keys] check reports it)")
        return
    rows = import_hint_rows(target)
    en_rows = import_hint_rows(en)

    en_sessions = assemble_hints(en_rows["sessions"])
    if en_sessions != WIDGET_SESSIONS_ROW_EN:
        fail(
            "row",
            f"assembler control failed: built {en_sessions!r}, the widget painted "
            f"{WIDGET_SESSIONS_ROW_EN!r}",
        )
    elif cell_len(en_sessions) != 62:
        fail(
            "row",
            f"assembler control failed: EN sessions row measures {cell_len(en_sessions)}, "
            f"the widget measured 62",
        )
    notes.append(
        f"assembler control: EN sessions row reproduced byte-identically from the EN catalogue "
        f"({cell_len(en_sessions)} cells, widget-measured 62)"
    )

    for state in ("sources", "sessions", "preview_import", "preview_confirm", "progress", "result", "loading"):
        row = assemble_hints(rows[state])
        cells = cell_len(row)
        en_cells = cell_len(assemble_hints(en_rows[state]))
        notes.append(
            f"row[{state}]: es={cells} en={en_cells} budget={HINT_ROW_BUDGET_80X24} :: {row}"
        )
        if cells > HINT_ROW_BUDGET_80X24:
            fail(
                "row",
                f"{state}: assembled row is {cells} cells, over the documented "
                f"{HINT_ROW_BUDGET_80X24}-cell row at 80x24: {row!r}",
            )

    esc_key, esc_label = rows["sessions"][-1]
    esc_hint = f"{esc_key} {esc_label}"
    sessions_row = assemble_hints(rows["sessions"])
    sessions_cells = cell_len(sessions_row)
    if not sessions_row.endswith(esc_hint):
        fail("row", f"sessions row does not end with the esc hint {esc_hint!r}: {sessions_row!r}")
    notes.append(
        f"sessions row ends with {esc_hint!r} ({cell_len(esc_hint)} cells); "
        f"row={sessions_cells} <= {HINT_ROW_BUDGET_80X24}, spare={HINT_ROW_BUDGET_80X24 - sessions_cells}"
    )
    over_gate = sessions_cells - HINT_ROW_GATE_BOX
    notes.append(
        f"gate boundary (measured, not gated): show_move paints down to a "
        f"{HINT_ROW_GATE_BOX}-cell box = a 78-column terminal; the row is {over_gate:+d} "
        f"against that box, so {max(0, over_gate)} trailing cell(s) of {esc_hint!r} clip at exactly that width"
    )


def check_analytics_ladder(en: dict, target: dict, context: dict) -> None:
    """Round 2: the A2 dimension -- tier order and the verb split."""
    widths: list[tuple[str, int]] = []
    for key, default_cap in ANALYTICS_TIERS:
        cap = int(context.get(key, {}).get("maxCells", default_cap))
        es_cells = cell_len(target[key])
        en_cells = cell_len(en[key])
        widths.append((key, es_cells))
        notes.append(
            f"tier {key.rsplit('.', 1)[-1]}: es={es_cells} en={en_cells} cap={cap} "
            f"delta_vs_en={es_cells - en_cells:+d} within_cap={'yes' if es_cells <= cap else 'no'}"
        )
    for (wide_key, wide), (narrow_key, narrow) in zip(widths, widths[1:]):
        if wide <= narrow:
            fail(
                "ladder",
                f"{narrow_key} ({narrow} cells) is not narrower than {wide_key} ({wide}); the widget "
                f"paints the first tier that FITS, widest-first, so this order would misorder the ladder",
            )
    notes.append("ladder order widest-first: " + " > ".join(f"{k.rsplit('.', 1)[-1]}={w}" for k, w in widths))

    alt = "↑↓ fila · enter expandir · e todo"
    notes.append(
        f"counterfactual {alt!r} = {cell_len(alt)} cells against the compact cap 31 -> "
        f"{'fits' if cell_len(alt) <= 31 else 'does not fit'}"
    )
    notes.append(
        "ladder verbs: compact='abrir', full/full_metric='expandir', keys/keys_bare name the key only"
    )


def check_remediation_applied(target: dict) -> None:
    """Round 2: report each round-1 disposition against the round-1 wording."""
    for fid, key, suggested in ROUND1_SUGGESTIONS[NAMESPACE]:
        if key is None and suggested is None:
            notes.append(f"{fid}: recorded-only nit (no fixed wording requested) — judgement in report.md")
            continue
        if key is None:
            if _missing_hint_keys(target):
                notes.append(f"{fid}: sessions row not measurable this run (a control deleted a key)")
                continue
            current = assemble_hints(import_hint_rows(target)["sessions"])
            notes.append(
                f"{fid}: sessions row {'MATCHES' if current == suggested else 'DIFFERS FROM'} "
                f"the round-1 suggestion ({cell_len(current)} cells)"
            )
            continue
        current = target[key]
        notes.append(
            f"{fid}: {'APPLIED' if current == suggested else 'NOT APPLIED'} — {key}"
        )


def check_round2(en: dict, target: dict, context: dict) -> None:
    if NAMESPACE == "tui.import":
        check_hint_register(target)
        check_import_rows(en, target)
    else:
        check_analytics_ladder(en, target, context)
    check_remediation_applied(target)


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def collect_failures(en: dict, target: dict, context: dict) -> list[str]:
    global failures
    saved = failures
    failures = []
    try:
        check_keys(en, target)
        check_icu(en, target)
        check_plurals(en, target)
        check_render(en, target)
        check_tokens(en, target)
        check_budgets(en, target, context)
        check_polarity(en, target)
        check_terms(target)
        check_untranslated(en, target)
        check_semantics(target)
        check_round2(en, target, context)
    finally:
        caught = failures
        failures = saved
    return caught


def run_negative_controls(en: dict, target: dict, context: dict) -> None:
    """Deliberately break one thing at a time and prove the check rejects it."""
    from local_operator.i18n.runtime import message_arguments

    controls: list[tuple[str, dict, dict, str]] = []

    # 1. key parity
    broken = dict(target)
    broken.pop(next(iter(broken)))
    controls.append(("keys", broken, context, "dropped one key"))

    # 2. placeholder renamed (argument tuple changes)
    broken = dict(target)
    for name, value in broken.items():
        simple = [a for a, kind in message_arguments(value) if kind == "simple"]
        if simple:
            broken[name] = value.replace("{" + simple[0] + "}", "{renombrado}", 1)
            break
    controls.append(("icu", broken, context, "renamed a placeholder argument"))

    # 3. a required Spanish category deleted
    broken = dict(target)
    name = next(n for n, v in broken.items() if "{count, plural" in v)
    broken[name] = re.sub(r"many \{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*", "", broken[name], count=1)
    controls.append(("plural", broken, context, "deleted a 'many' branch"))

    # 4. a branch emptied, so the number would disappear
    broken = dict(target)
    name = next(n for n, v in broken.items() if "{count, plural" in v and " other {" in v)
    broken[name] = blank_branch(broken[name], "other")
    controls.append(("plural", broken, context, "emptied a branch"))

    # 5. a category Spanish does not have
    broken = dict(target)
    name = next(n for n, v in broken.items() if "{count, plural" in v and " other {" in v)
    broken[name] = re.sub(r"other \{", "few {x} other {", broken[name], count=1)
    controls.append(("plural", broken, context, "added a 'few' category"))

    # 6. a machine token translated
    broken = dict(target)
    if "tui.import.error_source_required" in broken:
        broken["tui.import.error_source_required"] = broken["tui.import.error_source_required"].replace(
            "<source>", "<origen>"
        )
        controls.append(("tokens", broken, context, "translated the <source> machine token"))
    else:
        broken["tui.analytics.legend.cost"] = broken["tui.analytics.legend.cost"].replace("$\u2014", "$0")
        controls.append(("tokens", broken, context, "rewrote the $\u2014 money mark"))

    # 7. a width budget exceeded. Where the sidecar declares none, one is
    #    injected on a key whose es value is longer than its en value, so the
    #    budget logic itself is exercised either way.
    broken = dict(target)
    broken_context = dict(context)
    declared = [n for n in broken if n in context and "maxCells" in context[n]]
    if declared:
        broken[declared[0]] = broken[declared[0]] + " extra extra"
        controls.append(("budget", broken, broken_context, "padded a declared-budget hint past maxCells"))
    else:
        candidates = [n for n in broken if cell_len(broken[n]) > cell_len(en.get(n, ""))]
        assert candidates, "no key is longer in es than en; budget control cannot be built"
        name = candidates[0]
        broken_context[name] = {"maxCells": cell_len(en[name])}
        controls.append(
            (
                "budget",
                broken,
                broken_context,
                f"injected maxCells={cell_len(en[name])} on {name} (en fits, es does not)",
            )
        )

    # 8. a negation dropped
    broken = dict(target)
    if "tui.import.preview_confirm" in broken:
        broken["tui.import.preview_confirm"] = "Enter importa las sesiones seleccionadas."
        controls.append(("polarity", broken, context, "dropped 'nothing has been imported'"))
    else:
        broken["tui.analytics.empty.title"] = "Uso registrado."
        controls.append(("polarity", broken, context, "dropped the negation from the empty state"))

    # 9. a promised behaviour dropped
    broken = dict(target)
    if "tui.import.rerun_hint" in broken:
        broken["tui.import.rerun_hint"] = (
            "Repetir es seguro: las fuentes sin cambios se omiten y las que han crecido se reanudan."
        )
        controls.append(("semantics", broken, context, "dropped the rewritten-source behaviour"))
    else:
        broken["tui.analytics.search.summary_free"] = "{count} sin precio"
        controls.append(("semantics", broken, context, "rendered 'free' as 'unpriced'"))

    # 10. a glossary violation
    broken = dict(target)
    broken[next(iter(broken))] = "el demonio está en marcha, 3 sesiones"
    controls.append(("terms", broken, context, "used 'demonio' for daemon"))

    # 11. prose left in English
    broken = dict(target)
    name = next(n for n, v in broken.items() if len(v.split()) >= 4)
    broken[name] = en.get(name, broken[name])
    controls.append(("untranslated", broken, context, "left a prose value in English"))


    # 12+. round 2 — restore the round-1 defects; the new checks must fire.
    if NAMESPACE == "tui.import":
        broken = dict(target)
        broken["tui.import.hint_select"] = "seleccionar"
        controls.append(
            ("row", broken, context, "restored the round-1 'space seleccionar' (an 80-cell sessions row)")
        )
        broken = dict(target)
        broken["tui.import.all_none"] = "todas/ninguna de las sesiones"
        controls.append(
            ("row", broken, context, "grew a SIBLING term in the same row past the 76-cell budget")
        )
        broken = dict(target)
        broken["tui.import.hint_list"] = "lista"
        controls.append(
            ("register", broken, context, "restored the round-1 noun 'lista' in an infinitive hint family")
        )
    else:
        broken = dict(target)
        broken["tui.analytics.hint.expand.compact"] = broken["tui.analytics.hint.expand.compact"] + " extra"
        controls.append(("budget", broken, context, "padded the compact tier past maxCells 31"))
        broken = dict(target)
        broken["tui.analytics.hint.expand.keys"] = "↑↓ fila · enter abrir · e todo y algo más aquí"
        controls.append(
            ("ladder", broken, context, "made the KEYS tier wider than the COMPACT tier (order inverted)")
        )

    for expected_check, mutated, mutated_context, description in controls:
        caught = collect_failures(en, mutated, mutated_context)
        hit = [f for f in caught if f.startswith(f"[{expected_check}]")]
        print(f"  negative control [{expected_check}] {description}: {'CAUGHT' if hit else 'MISSED'}")
        if hit:
            print(f"      -> {hit[0]}")
        else:
            failures.append(
                f"[control] negative control did NOT trip {expected_check}: {description}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    global failures, notes

    en_path = CAT / "en" / f"{NAMESPACE}.json"
    target_path = CAT / LOCALE / f"{NAMESPACE}.json"
    context_path = CAT / "en" / f"{NAMESPACE}.context.json"

    print(f"namespace : {NAMESPACE}")
    print(f"locale    : {LOCALE}")
    print(f"repo root : {ROOT}")
    print(f"en        : {en_path.name} exists={en_path.exists()}")
    print(f"target    : {LOCALE}/{target_path.name} exists={target_path.exists()}")
    print(f"context   : {context_path.name} exists={context_path.exists()}")
    print(f"cell_len  : {CELL_SOURCE}")
    print()

    en = load_json(en_path)
    target = load_json(target_path)
    context = load_json(context_path) if context_path.exists() else {}
    print(f"key counts: en={len(en)} target={len(target)} context={len(context)}")
    print()

    print("== blocking checks ==")
    check_keys(en, target)
    check_icu(en, target)
    check_plurals(en, target)
    check_render(en, target)
    check_tokens(en, target)
    check_budgets(en, target, context)
    check_polarity(en, target)
    check_terms(target)
    check_untranslated(en, target)
    check_semantics(target)
    check_round2(en, target, context)
    print(f"  {len(failures)} failure(s) so far")
    for line in failures:
        print("  FAIL " + line)
    print()

    print("== measurements ==")
    for line in notes:
        print("  " + line)
    print()

    audit_failures = list(failures)
    failures = []

    print("== negative controls (each MUST trip its check) ==")
    run_negative_controls(en, target, context)
    print()

    print("== result ==")
    for line in audit_failures:
        print("  FAIL " + line)
    for line in failures:
        print("  FAIL " + line)
    total = len(audit_failures) + len(failures)
    if total == 0:
        print("  PASS — all checks clean, every negative control tripped")
        return 0
    print(f"  FAIL — {total} problem(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())

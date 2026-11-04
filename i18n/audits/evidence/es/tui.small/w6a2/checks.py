#!/usr/bin/env python3
"""Independent es audit — mechanical checks for one catalogue namespace (w6a2).

ROUND 2 ADDENDUM
================
This file is the w6a1 script with a round-2 section appended. C1..C9 are
unchanged: they are the mechanical floor re-derived on the CURRENT bytes. The
round-2 section (C10..C13) exists because a remediation round needs evidence
that a plain re-run cannot give:

  C10  the delta is EXACTLY the recorded round-1 findings. Reverting each
       named key to its round-1 value must reproduce the round-1 file digest
       byte-for-byte; any edit outside the recorded delta — a stray sibling
       edit, a reordered key, a whitespace change — makes the digest differ
       and the check fail. This is what turns "no collateral change" from a
       promise into a measurement.
  C11  the changed values still fit, measured with `rich.cells.cell_len` on
       the string the renderer BUILDS (placeholders substituted with the
       values the code actually supplies), not on the raw `{name}` token —
       the raw token overstates a digit slot by five cells.
  C12  a pair whose one member was fixed still reads as one system (the
       goal-panel heading + overflow triplet here).
  C13  a round-1 CONSTRAINT (`tui.small.copy.code_count`) still holds: same
       value, same sidecar note, and the repo's own argument parity still
       refuses the plural upgrade that would be the tempting "fix".

The baseline digests below are copied from each w6a1 `record.json`, so a later
edit of that record cannot silently move this audit's baseline.


WHY THIS FILE EXISTS
====================
The catalogue validator (`scripts/i18n/check.py --catalogues-only`) already
compares key sets, ICU arguments and plural categories against `en`. This
script exists because an auditor has to be able to say WHICH classes of defect
were ruled out, reproduce the numbers, and show the checks can still fail, on
the exact bytes that were read. It therefore:

  * re-derives every mechanical fact from the same files the audit read,
  * measures cell widths with `rich.cells.cell_len` (the renderer's own
    measure), not `len()`,
  * fires the same checks against deliberately broken copies (negative
    controls) and reports what each one caught,
  * exits non-zero when a hard check fails, so a green run is evidence and a
    stale one cannot pass silently.

WHAT IT DOES NOT DO
===================
No semantic judgement lives here. Meaning, idiom and register were read by the
auditor; this is the mechanical floor beneath that reading. It never writes:
it opens the catalogues read-only and prints.

USAGE
=====
    PYTHONPATH=<repo> <python-with-rich> \
        i18n/audits/evidence/es/<namespace>/<audit-id>/checks.py

The namespace and audit id are read from this file's own path, so the bundle
is self-describing and a copied script cannot silently audit the wrong
namespace.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from rich.cells import cell_len

#: Spanish CLDR categories — the only ones an es plural may carry. `few` is not
#: a Spanish category; `many` is exact multiples of 10^6 only.
ES_CATEGORIES = {"one", "many", "other"}

#: Tokens that are machine vocabulary in this surface and legitimately stay
#: Latin/English inside es copy (glossary §1 and the es style guide's
#: kept-English list). The residue scan subtracts these before deciding a value
#: still holds English prose.
KEPT_ENGLISH = {
    "lop", "lo", "omp", "tui", "api", "cli", "json", "yaml", "http", "https",
    "url", "gpu", "js", "css", "html", "python", "node", "git", "github",
    "macos", "windows", "linux", "runtime", "harness", "hook", "hooks",
    "handshake", "backoff", "commit", "push", "pull", "merge", "rebase",
    "token", "tokens", "endpoint", "extra", "script", "prompt", "deploy",
    "daemon", "daemons", "tenant", "heartbeat", "web", "fetch", "read",
    "browser", "bash", "edit", "wake", "monitor", "ctrl", "shift", "alt",
    "esc", "enter", "tab", "space", "pid", "id", "ids", "operator", "y",
    "n", "w", "r", "d", "o", "f", "t", "key", "chain", "provider", "model",
    "goal", "project", "librarian", "copy", "chat", "path", "link",
}

#: Words that must never appear in es copy for this surface. Each is a settled
#: decision recorded in the glossary or the es style guide; a hit is a
#: glossary violation, not a preference.
FORBIDDEN = {
    "demonio": "daemon stays English (es guide, W2 amendment)",
    "demonios": "daemon stays English (es guide, W2 amendment)",
    "inquilino": "tenant stays English (es guide, W2 amendment)",
    "fichero": "neutral Spanish uses 'archivo' (es guide)",
    "celular": "es guide: 'teléfono', never the region-marked words",
    "móvil": "es guide: 'teléfono', never the region-marked words",
    "carpeta de trabajo": "'carpeta' is a folder; a session cwd is 'directorio de trabajo'",
    "usted": "es guide fixes tú, never usted",
    "vosotros": "es guide: vosotros is never used",
    "starter": "'plantilla incluida' is the settled rendering",
    "perfiles iniciales": "a second rendering of one product term (W2 audit)",
    "«": "es typography uses “ ” (es guide)",
    "»": "es typography uses “ ” (es guide)",
}

#: English function words whose presence in a Spanish value means untranslated
#: prose. Checked word-wise against KEPT_ENGLISH, so a kept technical term does
#: not trip it.
ENGLISH_WORDS = {
    "the", "and", "with", "from", "for", "your", "you", "this", "that",
    "are", "was", "were", "have", "has", "will", "can", "cannot", "please",
    "here", "there", "when", "while", "into", "than", "then", "only",
    "should", "would", "must", "may", "might", "not", "yes", "all",
    "some", "more", "less", "none", "unknown", "missing", "failed", "sent",
    "to", "of", "in", "on", "at", "by", "it", "its", "is", "be", "or",
}

BACKTICK_RE = re.compile(r"`[^`]*`")
FLAG_RE = re.compile(r"(?<![\w-])--[a-z0-9][\w-]*")
ANGLE_RE = re.compile(r"<[^<>\n]+>")
KEYCOMBO_RE = re.compile(
    r"\b(?:ctrl|shift|alt|cmd|fn)\+[a-z0-9]+|\b(?:esc|enter|tab|f[0-9]{1,2})\b"
)
SLASH_CMD_RE = re.compile(r"(?<![\w/:])/[a-z][a-z_]*")
MONEY_RE = re.compile(r"\$[\d.—]+")
PLURAL_ARG_RE = re.compile(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(plural|selectordinal)\s*,")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balanced_block(source: str, start: int) -> str:
    """The brace-balanced body of an ICU block whose content starts at `start`."""
    depth = 1
    index = start
    while index < len(source) and depth > 0:
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1
    return source[start : index - 1]


def plural_categories(source: str) -> dict[str, set[str]]:
    """plural/selectordinal argument -> the CLDR categories its branches use."""
    found: dict[str, set[str]] = {}
    for match in PLURAL_ARG_RE.finditer(source):
        body = balanced_block(source, match.end())
        labels = set(re.findall(r"(?:^|\s)(zero|one|two|few|many|other)\s*\{", body))
        found.setdefault(match.group(1), set()).update(labels)
    return found


def check_json_and_keys(en: dict, es: dict) -> list[str]:
    """C1 — valid JSON (already parsed), identical key sets, identical order."""
    problems: list[str] = []
    missing = sorted(set(en) - set(es))
    extra = sorted(set(es) - set(en))
    for key in missing:
        problems.append(f"missing key {key!r}")
    for key in extra:
        problems.append(f"unknown key {key!r} (added by the translation)")
    if list(en) != list(es) and not missing and not extra:
        problems.append("key ORDER differs from en (catalogue order is part of the contract)")
    return problems


def check_arguments(en: dict, es: dict, runtime) -> list[str]:
    """C2 — placeholder names, kinds and order, compared by the repo's own parser."""
    problems: list[str] = []
    for key in en:
        if key not in es:
            continue
        try:
            got = runtime.message_arguments(es[key])
        except Exception as exc:  # noqa: BLE001 - any parse failure is a finding
            problems.append(f"{key}: does not parse: {exc}")
            continue
        try:
            want = runtime.message_arguments(en[key])
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{key}: EN source does not parse (unexpected): {exc}")
            continue
        if got != want:
            problems.append(f"{key}: arguments {got!r} != en's {want!r}")
    return problems


def check_plural_categories(es: dict) -> list[str]:
    """C3 — es plurals carry exactly one/many/other; no few/two/zero."""
    problems: list[str] = []
    for key, value in es.items():
        for arg, cats in plural_categories(value).items():
            if cats != ES_CATEGORIES:
                problems.append(
                    f"{key}: plural {arg!r} carries {sorted(cats)}, "
                    f"es requires exactly {sorted(ES_CATEGORIES)}"
                )
        # selectordinal is not used anywhere in es: an ordinal would need a
        # different (and unapproved) structure.
        if "selectordinal" in value:
            problems.append(f"{key}: selectordinal is not an approved es structure")
    return problems


def check_tokens(en: dict, es: dict) -> list[str]:
    """C4 — every machine token survives byte-identical."""
    problems: list[str] = []
    extractors = {
        "backtick code span": BACKTICK_RE.findall,
        "flag": FLAG_RE.findall,
        "angle token": ANGLE_RE.findall,
        "key name": KEYCOMBO_RE.findall,
        "slash command": SLASH_CMD_RE.findall,
        "money amount": MONEY_RE.findall,
        "arrow": lambda s: re.findall(r"[↑↓←→↕↔]", s),
        "middle dot": lambda s: re.findall(r"·", s),
    }
    for key in en:
        if key not in es:
            continue
        for label, extract in extractors.items():
            want = Counter(extract(en[key]))
            got = Counter(extract(es[key]))
            if want != got:
                problems.append(f"{key}: {label} {sorted(got.elements())} != en's {sorted(want.elements())}")
    return problems


def check_whitespace(en: dict, es: dict) -> list[str]:
    """C5 — load-bearing leading/trailing spaces and newlines are preserved.

    Several values in this block are Rich Text fragments concatenated by the
    caller; their edge whitespace IS the separator or the row indent (the
    sidecar says so per key). Trimming one silently joins two words.
    """
    problems: list[str] = []
    for key in en:
        if key not in es:
            continue
        a, b = en[key], es[key]
        edges_a = (len(a) - len(a.lstrip()), len(a) - len(a.rstrip()), a.count("\n"))
        edges_b = (len(b) - len(b.lstrip()), len(b) - len(b.rstrip()), b.count("\n"))
        if edges_a != edges_b:
            problems.append(
                f"{key}: edge whitespace/newlines {edges_b} != en's {edges_a} "
                f"(en={a!r}, es={b!r})"
            )
        # An INTERNAL run of two or more spaces is a typo unless it is doing
        # layout work that legitimately differs between languages: the column
        # padding after a ``label:`` (es labels are a different length, so the
        # run length must differ to keep the column) and the row indent that
        # follows a newline. Everything else is French spacing or a slip.
        for match in re.finditer(r" {2,}", b):
            start = match.start()
            if start == 0 or b[start - 1] == "\n":
                continue
            if b[start - 1] == ":":
                continue
            # The source's own layout may carry the same run (``buscar  /
            # mirrors ``search  /``): mirroring it is fidelity, not a typo.
            if len(match.group(0)) in {len(m.group(0)) for m in re.finditer(r" {2,}", a)}:
                continue
            problems.append(
                f"{key}: internal space run of {len(match.group(0))} with no column or "
                f"line-start role: {b!r}"
            )
    return problems


def check_budgets(en: dict, es: dict, context: dict | None) -> tuple[list[str], list[tuple]]:
    """C6 — declared maxCells budgets, measured in cells (not characters)."""
    problems: list[str] = []
    table: list[tuple] = []
    if context is None:
        return problems, table
    for key, meta in context.items():
        if not isinstance(meta, dict) or "maxCells" not in meta:
            continue
        if key not in en or key not in es:
            continue
        budget = int(meta["maxCells"])
        es_cells = cell_len(es[key])
        en_cells = cell_len(en[key])
        table.append((key, budget, en_cells, es_cells))
        if es_cells > budget:
            problems.append(
                f"{key}: es {es_cells} cells > maxCells {budget} "
                f"(en {en_cells}) — {es[key]!r}"
            )
    return problems, table


def check_register(es: dict) -> list[str]:
    """C7 — settled terminology and es typography that must not drift."""
    problems: list[str] = []
    for key, value in es.items():
        lowered = value.lower()
        for word, why in FORBIDDEN.items():
            if word in lowered:
                problems.append(f"{key}: contains {word!r} — {why}")
        if re.search(r"(?<!\.)\.\.\.", value) and "…" not in value:
            problems.append(f"{key}: ASCII ellipsis instead of … — {value!r}")
        if " - " in value:
            problems.append(f"{key}: hyphen used as a dash (es uses ' — ') — {value!r}")
    return problems


def check_english_residue(en: dict, es: dict) -> list[str]:
    """C8 — no English prose left in a Spanish value."""
    problems: list[str] = []
    for key, value in es.items():
        stripped = BACKTICK_RE.sub(" ", value)
        stripped = re.sub(r"\{[^{}]*\}", " ", stripped)
        stripped = re.sub(r"<[^<>]*>", " ", stripped)
        for word in re.findall(r"[A-Za-z]+", stripped):
            low = word.lower()
            if low in ENGLISH_WORDS and low not in KEPT_ENGLISH:
                problems.append(f"{key}: English word {word!r} in {value!r}")
    return problems


def run_all(en: dict, es: dict, context: dict | None, runtime) -> dict[str, list[str]]:
    budget_problems, budget_table = check_budgets(en, es, context)
    return {
        "C1 keys": check_json_and_keys(en, es),
        "C2 arguments": check_arguments(en, es, runtime),
        "C3 plural categories": check_plural_categories(es),
        "C4 machine tokens": check_tokens(en, es),
        "C5 whitespace": check_whitespace(en, es),
        "C6 maxCells": budget_problems,
        "C7 register/typography": check_register(es),
        "C8 english residue": check_english_residue(en, es),
    }, budget_table


def negative_controls(en: dict, es: dict, runtime) -> list[tuple[str, bool, str]]:
    """Prove each check can still fail.

    Each control breaks ONE value the way a real translation defect would, then
    runs the relevant check against the broken copy and reports whether the
    check noticed. Where this namespace happens to carry the feature (a
    placeholder, a backtick span, a load-bearing edge space) the control uses
    that REAL value; where it does not, the control builds a minimal synthetic
    pair, because what is being proved is that the CHECK can fail, not that the
    namespace happens to exercise it. A control that is NOT caught fails the
    whole run: a check nobody can break proves nothing about the bytes under
    audit.
    """
    results: list[tuple[str, bool, str]] = []

    # NC1 — a dropped placeholder. Placeholder parity is the defect class that
    # silently renders `{name}` or loses a caller-supplied value.
    source = next((k for k, v in en.items() if re.search(r"\{[a-z_]", v)), None)
    if source:
        en_c, es_c = {source: en[source]}, {source: es[source]}
    else:
        en_c, es_c = {"nc.key": "Hello {name}"}, {"nc.key": "Hola {name}"}
    broken = {k: re.sub(r"\{[a-z_][a-z0-9_]*\}", "", v, count=1) for k, v in es_c.items()}
    problems = check_arguments(en_c, broken, runtime)
    results.append((
        f"NC1 drop a placeholder ({source or 'synthetic fixture'})",
        bool(problems),
        problems[0] if problems else "NOT CAUGHT",
    ))

    # NC2 — a plural branch relabelled as `few`, which Spanish does not have.
    source = next((k for k, v in es.items() if " many " in v), None)
    if source:
        es_c = {source: es[source].replace(" many ", " few ", 1)}
    else:
        es_c = {"nc.key": "{n, plural, one {# x} few {# y} other {# z}}"}
    problems = check_plural_categories(es_c)
    results.append((
        f"NC2 relabel a plural branch as few ({source or 'synthetic fixture'})",
        bool(problems),
        problems[0] if problems else "NOT CAUGHT",
    ))

    # NC3 — a stripped backtick code span.
    source = next((k for k, v in en.items() if "`" in v), None)
    if source:
        en_c = {source: en[source]}
        broken = {source: es[source].replace("`", "", 2)}
    else:
        en_c = {"nc.key": "try `browser` for the page"}
        broken = {"nc.key": "try browser for the page"}
    problems = check_tokens(en_c, broken)
    results.append((
        f"NC3 strip a backtick code span ({source or 'synthetic fixture'})",
        bool(problems),
        problems[0] if problems else "NOT CAUGHT",
    ))

    # NC4 — a trimmed load-bearing edge space.
    source = next((k for k, v in en.items() if v != v.rstrip() or v != v.lstrip()), None)
    if source:
        en_c = {source: en[source]}
        broken = {source: es[source].strip()}
    else:
        en_c = {"nc.key": "allow "}
        broken = {"nc.key": "permitir"}
    problems = check_whitespace(en_c, broken)
    results.append((
        f"NC4 trim a load-bearing edge space ({source or 'synthetic fixture'})",
        bool(problems),
        problems[0] if problems else "NOT CAUGHT",
    ))

    # NC5 — a value inflated past a declared maxCells budget.
    source = next((k for k, v in en.items() if cell_len(v) <= 8))
    if source:
        en_c, es_c = {source: en[source]}, {source: es[source]}
    else:
        en_c, es_c = {"nc.key": "no charge"}, {"nc.key": "sin cargo"}
    context = {k: {"maxCells": 4} for k in en_c}
    problems, _ = check_budgets(en_c, es_c, context)
    results.append((
        f"NC5 inflate a value past a 4-cell budget ({source or 'synthetic fixture'})",
        bool(problems),
        problems[0] if problems else "NOT CAUGHT",
    ))

    # NC6 — a settled-terminology violation (glossary term swapped for a
    # synonym the style guide explicitly rules out).
    problems = check_register({"nc.key": "abre el demonio de la sesión"})
    results.append((
        "NC6 use a forbidden glossary term",
        bool(problems),
        problems[0] if problems else "NOT CAUGHT",
    ))

    # NC7 — untranslated English prose left in a Spanish value.
    problems = check_english_residue(
        {"nc.key": "Open the settings page"},
        {"nc.key": "Open the settings page"},
    )
    results.append((
        "NC7 leave English prose untranslated",
        bool(problems),
        problems[0] if problems else "NOT CAUGHT",
    ))

    # NC8 — a key dropped from the catalogue (a silent en-fallback hole).
    problems = check_json_and_keys({"nc.key": "x", "nc.other": "y"}, {"nc.key": "x"})
    results.append((
        "NC8 drop a key from the target",
        bool(problems),
        problems[0] if problems else "NOT CAUGHT",
    ))

    # NC9 — a plural with no `many` arm: at 10^6 the renderer falls to `other`,
    # which is exactly the silent selection defect C9's marker rule exists for.
    def fake_render(code: str, params: dict, *, locale: str | None = None) -> str:
        return runtime.render_message(
            "{n, plural, one {{n} uno} other {{n} otros}}", params, locale or "es"
        )

    problems, _ = check_rendering(
        "nc", fake_render, [("nc.key", "n", [1, 2, 1000000], {}, " de ")]
    )
    results.append((
        "NC9 render a plural with no many arm at 10^6",
        bool(problems),
        problems[0] if problems else "NOT CAUGHT",
    ))
    return results


#: Plural messages whose branch SELECTION is executed through the shipped
#: renderer. Structure proves the branches exist; only the formatter proves that
#: Spanish selects `many` at 10^6, `one` at 1 and `other` at 0/2, and that `#`
#: formats with the locale's rules while a raw `{shown}` stays ungrouped.
#: Each row: (key, count-arg name, counts, extra args, substring the 10^6 render
#: must contain because only the distinct `many` branch carries it; None when the
#: branches are word-identical, as in "+{label} más").
RENDER_CASES: dict[str, list[tuple[str, str, list[int], dict[str, str], str | None]]] = {
    "tui.small": [
        ("tui.small.draft.attachment_count", "images", [1, 2, 1000000], {}, " de "),
        ("tui.small.network_panel.member_count", "count", [0, 1, 2, 1000000], {}, " de "),
        ("tui.small.copy_picker.more_lines", "count", [1, 5, 1000000], {}, " de "),
        (
            "tui.small.wake_panel.monitors_need_attention",
            "count",
            [1, 3, 1000000],
            {"total": "9"},
            " de ",
        ),
        ("tui.small.wake_panel.scheduled_count", "count", [1, 3, 1000000], {}, " de "),
        ("tui.small.settings_view.hop_count", "count", [1, 3, 1000000], {}, " de "),
        ("tui.small.command_picker.entries", "count", [0, 1, 2, 1000000], {}, " de "),
    ],
    "tui.library": [
        ("tui.library.asset_count", "count", [0, 1, 2, 1000000], {}, " de "),
        ("tui.library.detail_sources", "count", [1, 2, 1000000], {}, " de "),
    ],
    "tui.projects": [
        ("tui.projects.send.more", "count", [1, 3, 1000000], {"label": "3"}, None),
    ],
}


def check_rendering(namespace: str, render, cases=None) -> tuple[list[str], list[str]]:
    """C9 — execute the Spanish plural selection on the shipped renderer.

    A catalogue that parses is not a catalogue that renders: the `many` arm is
    only reachable at exact multiples of 10^6, and a missing or mislabelled arm
    silently falls back to `other` — a defect every structural check accepts.
    Rendering at 0, 1, 2 and 1.000.000 exercises all three Spanish categories.

    ``cases`` exists so the negative control can drive this same function with a
    synthetic message that has no `many` arm, proving the marker rule can fail.
    """
    problems: list[str] = []
    rows: list[str] = []
    for key, arg, counts, extra, many_marker in (
        RENDER_CASES.get(namespace, []) if cases is None else cases
    ):
        rendered: dict[int, str] = {}
        for count in counts:
            params: dict[str, object] = {**extra, arg: count, "shown": str(count)}
            out = render(key, params, locale="es")
            rendered[count] = out
            rows.append(f"{key} [{arg}={count}] -> {out!r}")
            if out == key:
                problems.append(
                    f"{key}: render returned the message CODE at {arg}={count} "
                    "(the message did not render)"
                )
        if many_marker and 1000000 in rendered and many_marker not in rendered[1000000]:
            problems.append(
                f"{key}: the 10^6 render {rendered[1000000]!r} lacks {many_marker!r} — "
                "the `many` branch did not select"
            )
        if 1 in rendered and 2 in rendered and rendered[1] == rendered[2]:
            problems.append(
                f"{key}: 1 and 2 rendered identically ({rendered[1]!r}) — `one` did not select"
            )
    return problems, rows


# ---------------------------------------------------------------------------
# ROUND 2 — remediation verification (w6a2)
# ---------------------------------------------------------------------------

#: Round-1 target digest + the exact value pair each recorded finding named.
#: Copied from i18n/audits/evidence/es/<namespace>/w6a1/record.json.
ROUND1_BASELINE = {
    "tui.small": {
        "target_sha256": "c48357ff20fcacfe996b687a2d035645639a8a6621d79ecf6b1436ca763ed145",
        "changed": {
            "tui.small.info_panel.deeper_below_depth": (
                "  +{deeper} más abajo (bajo profundidad {depth})",
                "  +{deeper} más abajo (bajo la profundidad {depth})",
            ),
            "tui.small.spend.unpriced_partial": (
                "peticiones sin precio",
                "algunas sin precio",
            ),
            "tui.small.network_panel.suspect_tag": (
                "  [sospechoso: puede que la clave se haya copiado]",
                "  [sospechoso: clave quizá copiada]",
            ),
            "tui.small.goal_panel.more_settled": (
                "… {hidden} finalizados más",
                "… {hidden} más finalizados",
            ),
            "tui.small.goal_panel.more_settled_indent": (
                "  … {hidden} finalizados más",
                "  … {hidden} más finalizados",
            ),
        },
    },
    "tui.library": {
        "target_sha256": "4b3e5982fdf26c71896579709f968dd25508f9fa48a1f942c0f616c0f4acbc62",
        "changed": {
            "tui.library.rail_hidden": (
                "Los filtros aparecen cuando el terminal tiene al menos {width} columnas de ancho.",
                "Los filtros aparecen a partir de {width} columnas.",
            ),
            "tui.library.form_name_placeholder": (
                "si no, el nombre del archivo",
                "si lo dejas vacío, se usa el nombre del archivo",
            ),
        },
    },
    "tui.projects": {
        "target_sha256": "4b6375ad8473afd927126efb535072f41426c80908f03c7a90834888f0264d76",
        "changed": {
            "tui.projects.binding_zoom_out": ("Tiempo más grueso", "Menos detalle"),
            "tui.projects.send.compose": (
                "enviar a: {handle} · esc cancela",
                "enviar a: {handle} · esc cancelar",
            ),
        },
    },
}

#: Placeholder substitutions as the CODE supplies them, so C11 measures the
#: painted string rather than the token. `why` names the constant the value
#: comes from.
ROUND2_WIDTH_CASES = {
    "tui.small": [
        ("tui.small.info_panel.deeper_below_depth", {"deeper": "3", "depth": "3"},
         "deeper=3, depth=_MAX_TREE_DEPTH=3"),
        ("tui.small.info_panel.deeper_below_depth", {"deeper": "1200", "depth": "3"},
         "worst case: raw digits stay ungrouped (1200)"),
        ("tui.small.spend.unpriced_partial", {}, "no arguments; maxCells 24"),
        ("tui.small.network_panel.suspect_tag", {}, "no arguments"),
        ("tui.small.goal_panel.more_settled", {"hidden": "3"}, "hidden=3"),
        ("tui.small.goal_panel.more_settled_indent", {"hidden": "3"}, "hidden=3"),
    ],
    "tui.library": [
        ("tui.library.rail_hidden", {"width": "90"}, "RAIL_MIN_WIDTH = 90"),
        ("tui.library.form_name_placeholder", {}, "no arguments"),
    ],
    "tui.projects": [
        ("tui.projects.binding_zoom_out", {}, "no arguments"),
        ("tui.projects.send.compose", {"handle": "◆ abcdefghijkl"},
         "SHORT_ID_CELLS = 12 -> the ◆ handle is 14 cells"),
    ],
}


def substituted(value: str, values: dict[str, str]) -> str:
    """The string the renderer builds, with placeholder slots filled."""
    out = value
    for name, text in values.items():
        out = out.replace("{" + name + "}", text)
    return out


def check_remediation_scope(namespace: str, text: str, es: dict) -> list[str]:
    """C10 — the remediation is exactly the recorded delta and nothing else."""
    spec = ROUND1_BASELINE.get(namespace)
    if spec is None:
        return [f"no round-1 baseline recorded for namespace {namespace!r}"]
    problems: list[str] = []
    clean = True
    for key, (_old, new) in spec["changed"].items():
        if es.get(key) != new:
            problems.append(f"{key}: expected the remediated value {new!r}, found {es.get(key)!r}")
        literal = json.dumps(new, ensure_ascii=False)
        count = text.count(literal)
        if count != 1:
            problems.append(
                f"{key}: the remediated value occurs {count} times in the raw file, expected 1"
            )
            clean = False
    if clean:
        reverted = text
        for key, (old, new) in spec["changed"].items():
            reverted = reverted.replace(
                json.dumps(new, ensure_ascii=False), json.dumps(old, ensure_ascii=False), 1
            )
        digest = hashlib.sha256(reverted.encode("utf-8")).hexdigest()
        if digest != spec["target_sha256"]:
            problems.append(
                "scope drift: reverting the recorded changes does not reproduce round-1 "
                f"bytes ({digest} != {spec['target_sha256']}) — a value OUTSIDE the "
                "recorded delta changed"
            )
    return problems


def check_changed_widths(namespace: str, en: dict, es: dict, context: dict | None):
    """C11 — the changed values, measured on the built string."""
    problems: list[str] = []
    table: list[tuple] = []
    for key, values, why in ROUND2_WIDTH_CASES.get(namespace, []):
        if key not in es or key not in en:
            problems.append(f"{key}: changed key missing from a catalogue")
            continue
        built = substituted(es[key], values)
        en_built = substituted(en[key], values)
        cells = cell_len(built)
        en_cells = cell_len(en_built)
        meta = (context or {}).get(key)
        budget = int(meta["maxCells"]) if isinstance(meta, dict) and "maxCells" in meta else None
        table.append((key, values, budget, en_cells, cells, why))
        if "{" in built:
            problems.append(f"{key}: a placeholder survived substitution — {built!r}")
        if budget is not None and cells > budget:
            problems.append(
                f"{key}: {cells} cells > maxCells {budget} with {values or 'no args'} — {built!r}"
            )
    return problems, table


def check_goal_pair(namespace: str, es: dict) -> list[str]:
    """C12 — the settled heading and both overflow rows still agree."""
    if namespace != "tui.small":
        return []
    problems: list[str] = []
    head = (es.get("tui.small.goal_panel.settled_heading") or "").strip()
    base = es.get("tui.small.goal_panel.more_settled", "")
    indented = es.get("tui.small.goal_panel.more_settled_indent", "")
    if not head:
        return ["tui.small.goal_panel.settled_heading is empty"]
    for key, value, prefix in (
        ("tui.small.goal_panel.more_settled", base, "…"),
        ("tui.small.goal_panel.more_settled_indent", indented, "  …"),
    ):
        if not value.startswith(prefix + " "):
            problems.append(f"{key}: does not open with {prefix + ' '!r} — {value!r}")
        if " más " not in value:
            problems.append(f"{key}: no 'más' before the participle — {value!r}")
        if not value.endswith(" " + head):
            problems.append(
                f"{key}: does not end with the heading noun {head!r} — {value!r}"
            )
        if value.rstrip().endswith("más"):
            problems.append(f"{key}: trailing 'más' (the round-1 nit) is back — {value!r}")
    if indented != "  " + base:
        problems.append("the indented variant is not the base form plus two spaces")
    return problems


def check_code_count(namespace: str, en: dict, es: dict, context: dict | None, runtime) -> list[str]:
    """C13 — the recorded no-plural constraint is intact and still enforced."""
    if namespace != "tui.small":
        return []
    key = "tui.small.copy.code_count"
    problems: list[str] = []
    if es.get(key) != "{shown} código":
        problems.append(f"{key}: recorded constraint no longer holds — {es.get(key)!r}")
        return problems
    if en.get(key) != "{shown} code":
        problems.append(f"{key}: EN source changed — {en.get(key)!r}")
    note = ((context or {}).get(key) or {}).get("note", "")
    if "NO plural branch" not in note:
        problems.append(f"{key}: the sidecar no longer records the no-plural-branch constraint")
    got = runtime.message_arguments(es[key])
    if got != (("shown", "simple"),):
        problems.append(f"{key}: argument kinds are {got!r}, expected the simple slot")
    probe = "{shown, plural, one {# código} many {# de códigos} other {# códigos}}"
    if runtime.message_arguments(probe) == got:
        problems.append(
            f"{key}: the parity rule no longer distinguishes simple from plural — "
            "the constraint is unenforced"
        )
    return problems


def round2_checks(namespace: str, es_path: Path, en: dict, es: dict, context: dict | None, runtime):
    """C10..C13 in the same shape as run_all's result mapping."""
    text = es_path.read_text(encoding="utf-8")
    width_problems, width_table = check_changed_widths(namespace, en, es, context)
    return {
        "C10 remediation scope": check_remediation_scope(namespace, text, es),
        "C11 changed-key widths": width_problems,
        "C12 goal-panel pair": check_goal_pair(namespace, es),
        "C13 code_count constraint": check_code_count(namespace, en, es, context, runtime),
    }, width_table


def round2_negative_controls(namespace: str, es_path: Path, es: dict, context: dict | None, runtime):
    """Round-2 controls: each breaks ONE thing a remediated file can get wrong."""
    spec = ROUND1_BASELINE.get(namespace)
    if spec is None:
        return [("round-2 controls need a recorded baseline", False, f"no baseline for {namespace!r}")]
    text = es_path.read_text(encoding="utf-8")
    en_path = es_path.parents[1] / "en" / f"{namespace}.json"
    en = json.loads(en_path.read_text(encoding="utf-8"))
    out: list[tuple[str, bool, str]] = []

    touched = set(spec["changed"])
    spare = next(
        k for k in es
        if k not in touched and es[k].strip() and not es[k].endswith((" ", "\n"))
    )
    drifted = text.replace(
        json.dumps(es[spare], ensure_ascii=False),
        json.dumps(es[spare] + "Z", ensure_ascii=False),
        1,
    )
    problems = check_remediation_scope(namespace, drifted, es)
    out.append((
        "C10 scope: one edit OUTSIDE the recorded delta",
        bool(problems),
        (problems or ["not caught"])[0],
    ))

    first = next(iter(spec["changed"]))
    stale = dict(es)
    stale[first] = spec["changed"][first][0]
    problems = check_remediation_scope(namespace, text, stale)
    out.append((
        "C10 scope: one recorded key left holding its round-1 value",
        bool(problems),
        (problems or ["not caught"])[0],
    ))

    # A key whose placeholder is load-bearing in this namespace, so the control
    # removes a REAL argument slot rather than inventing one.
    placeholder_key = {
        "tui.small": "tui.small.info_panel.deeper_below_depth",
        "tui.library": "tui.library.rail_hidden",
        "tui.projects": "tui.projects.send.compose",
    }[namespace]
    lost = dict(es)
    stripped = re.sub(r"\{[^{}]*\}", "", es[placeholder_key]).replace("  ", " ").strip()
    lost[placeholder_key] = stripped
    hits = [p for p in check_arguments(en, lost, runtime) if p.startswith(placeholder_key)]
    out.append((
        f"C2 arguments: the placeholder dropped from {placeholder_key}",
        bool(hits),
        (hits or ["not caught"])[0],
    ))

    budget_key = next(
        (k for k in spec["changed"]
         if isinstance((context or {}).get(k), dict) and "maxCells" in (context or {})[k]),
        None,
    )
    if budget_key is not None:
        over = dict(es)
        over[budget_key] = "algunas peticiones sin precio"
        problems = [p for p in check_budgets(en, over, context)[0] if p.startswith(budget_key)]
        out.append((
            "C6 maxCells: a remediated key pushed over its declared budget",
            bool(problems),
            (problems or ["not caught"])[0],
        ))

    dotted = dict(es)
    dotted[placeholder_key] = "... " + es[placeholder_key].lstrip("… ").lstrip()
    problems = [p for p in check_register(dotted) if p.startswith(placeholder_key)]
    out.append((
        "C7 typography: ASCII ellipsis reintroduced in a remediated key",
        bool(problems),
        (problems or ["not caught"])[0],
    ))

    if namespace == "tui.small":
        broken = dict(es)
        broken["tui.small.goal_panel.more_settled"] = "… {hidden} más finalizado"
        problems = [p for p in check_goal_pair(namespace, broken) if "more_settled:" in p]
        out.append((
            "C12 pair: the overflow row loses agreement with its heading",
            bool(problems),
            (problems or ["not caught"])[0],
        ))
    return out


def main() -> int:
    here = Path(__file__).resolve()
    namespace = here.parents[1].name
    audit_id = here.parents[0].name
    repo = here.parents[6]
    base = repo / "local_operator" / "i18n" / "catalogues"

    en_path = base / "en" / f"{namespace}.json"
    es_path = base / "es" / f"{namespace}.json"
    ctx_path = base / "en" / f"{namespace}.context.json"

    en = json.loads(en_path.read_text(encoding="utf-8"))
    es = json.loads(es_path.read_text(encoding="utf-8"))
    context = json.loads(ctx_path.read_text(encoding="utf-8")) if ctx_path.exists() else None

    sys.path.insert(0, str(repo))
    from local_operator.i18n import runtime  # noqa: PLC0415 - after sys.path

    print(f"== es audit {audit_id} :: namespace {namespace}")
    print(f"repo            {repo}")
    print(f"en              {en_path.relative_to(repo)}  sha256 {sha256(en_path)}  messages {len(en)}")
    print(f"target          {es_path.relative_to(repo)}  sha256 {sha256(es_path)}  messages {len(es)}")
    if context is None:
        print("context         (none — this namespace has no English sidecar)")
    else:
        print(f"context         {ctx_path.relative_to(repo)}  sha256 {sha256(ctx_path)}  entries {len(context)}")
    print()

    results, budget_table = run_all(en, es, context, runtime)
    round2_results, width_table = round2_checks(namespace, es_path, en, es, context, runtime)
    results.update(round2_results)
    failed = False
    for name, problems in results.items():
        status = "PASS" if not problems else "FAIL"
        print(f"[{status}] {name}: {len(problems)} problem(s)")
        for problem in problems[:12]:
            print(f"        - {problem}")
        if problems:
            failed = True

    print()
    if budget_table:
        print(f"maxCells budgets measured with rich.cells.cell_len ({len(budget_table)} declared):")
        for key, budget, en_cells, es_cells in sorted(budget_table, key=lambda row: -(row[3] or 0)):
            over = "OVER" if es_cells > budget else "    "
            print(f"  {over} {key:<52} budget {budget:>3}  en {en_cells:>3}  es {es_cells:>3}")
    else:
        print("maxCells budgets: none declared for this namespace (no usable sidecar)")

    print()
    print("C11 changed-key widths, with placeholders substituted as the code supplies them:")
    for key, values, budget, en_cells, cells, why in width_table:
        cap = f"budget {budget:>3}" if budget is not None else "no budget"
        over = "OVER" if budget is not None and cells > budget else "    "
        print(f"  {over} {key:<52} {cap}  en {en_cells:>3}  es {cells:>3}  [{why}]")

    print()
    from local_operator.i18n import messages as i18n_messages  # noqa: PLC0415 - after sys.path

    # C9 runs the real ICU selection on the es bytes. `messages.render` first
    # coerces an unshipped locale to `en` through resolve.eligible_locale, so if
    # this working tree's generated `shipped.json` does not list `es` a plain
    # call renders ENGLISH and proves nothing about the Spanish catalogue. The
    # gate is REPORTED here, never judged: when it is closed, C9 goes through
    # the same lookup + formatter `render_record` uses after the gate.
    gate = i18n_messages.resolve.eligible_locale("es")
    if gate == "es":
        render_fn = i18n_messages.render
        print("C9 ship gate: resolve.eligible_locale('es') -> 'es' (es is in the shipped set)")
    else:
        def render_fn(code, params, *, locale=None):
            ns = i18n_messages._namespace_for(i18n_messages.catalogues.catalogue_root(), code)
            if ns is None:
                return code
            source = i18n_messages._lookup("es", ns, code)
            if source is None:
                return code
            return i18n_messages._runtime.render_message(source, params, "es")
        print(
            "C9 ship gate: resolve.eligible_locale('es') -> "
            f"{gate!r} — the generated shipped.json in this working tree does not "
            "list es, so messages.render() would answer in ENGLISH. C9 therefore "
            "renders through the same catalogue lookup + ICU formatter "
            "render_record uses after the gate; the gate itself is a ledger "
            "state, not a property of these bytes."
        )

    render_problems, render_rows = check_rendering(namespace, render_fn)
    render_status = "PASS" if not render_problems else "FAIL"
    print(f"[{render_status}] C9 plural rendering (executed): {len(render_problems)} problem(s)")
    for row in render_rows:
        print("        " + row)
    for problem in render_problems:
        print("        - " + problem)
    if render_problems:
        failed = True

    print()
    print("negative controls (each must be CAUGHT):")
    controls_ok = True
    for label, caught, detail in negative_controls(en, es, runtime):
        print(f"  [{'CAUGHT' if caught else 'MISSED'}] {label}")
        print(f"           -> {detail}")
        controls_ok = controls_ok and caught

    print()
    print("round-2 negative controls (each must be CAUGHT):")
    for label, caught, detail in round2_negative_controls(namespace, es_path, es, context, runtime):
        print(f"  [{'CAUGHT' if caught else 'MISSED'}] {label}")
        print(f"           -> {detail}")
        controls_ok = controls_ok and caught

    print()
    verdict = "PASS" if not failed and controls_ok else "FAIL"
    print(f"checks.py: {verdict} (hard checks {'clean' if not failed else 'failed'}; "
          f"controls {'all caught' if controls_ok else 'a control was missed'})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

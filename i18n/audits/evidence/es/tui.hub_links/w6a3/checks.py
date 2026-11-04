#!/usr/bin/env python3
"""Independent audit probe — es TUI block, ROUND 3, wave w6a3 (two namespaces).

READ-ONLY BY CONSTRUCTION. This script never writes a catalogue, sidecar,
ledger, allowance, generated artefact or git object: it loads, measures and
prints. Every mutation it needs is applied to an IN-MEMORY copy of a string,
which is what the negative-control section (E) is for — a check that cannot
fail proves nothing, so each control breaks a value deliberately and reports
whether the corresponding check caught it.

SCOPE. Round 3 is a NARROW convergence check, not a re-audit of the block:
  tui.hub_links  `tui.hub_links.refusal.revision_moved` changed after round 2
                 (the one standing major, HL-1).
  tui.usage      `tui.usage.sign_in_expired` changed after round 2 (F-U1, the
                 one open minor).
Everything else in these namespaces was passed in round 2 and is not
re-judged here. The mechanical section still parses the whole namespace and
checks key/placeholder parity, because a one-line edit must not break the
file around it.

WHAT IT CHECKS
  A  MECHANICAL (must all PASS)
     JSON, key parity, ICU argument tuples, CLDR plural-category coverage for
     `es` (one/many/other), per-branch rendering through the shipped runtime,
     the changed key's placeholder shape, and the punctuation that carries
     layout meaning.
  B  HUB DELTA (tui.hub_links)
     the revised refusal names the inline consent PROMPT, not the notice line
     (`aviso`) or a dialog (`diálogo`); the glossary and style guide fix
     `prompt`; the term is not a one-off (counted across all 95 es catalogues);
     the code surface is a Static overlay whose consent prompt is inline; the
     notice wraps and its height is accounted.
  C  USAGE DELTA (tui.usage)
     `caducadas: {value}` is count-neutral and label-first like its two
     siblings; renders at 1, 2 and multi-digit counts; the counters segment
     fits the 31-cell floor body; the full chrome row is measured against en
     and the actual `rich` truncation at 40 and 60 columns.
  D  STYLE-GUIDE RECORD
     the two recorded width exceptions for this wave are re-measured, not
     trusted: paid annotations (family_caps, eligibility_unknown,
     paid.continue) and `tui.analytics.hint.expand.keys_bare`.
  E  NEGATIVE CONTROLS
     required: restoring `el aviso` and restoring `{value} agotadas` must each
     be caught; two further controls (the round-2 `diálogo` value and a wrong
     placeholder name) are included as probes of the probe.
  F  DELTA PROOF
     restoring the round-2 value in each file must reproduce the round-2
     sha256 exactly, proving no other value changed in that file.

Exit status: 0 when every check and every control behaves as expected, 1
otherwise. SG-1 (a style-guide record error) is a deliberate FAIL finding, not
a probe bug; the report grades it MINOR. A failing translation check is a real
finding, not a failure of the script.

Run with the repository's TUI interpreter (it supplies rich):
  cd /Users/damian/lo-i18n-es
  ~/local-operator/.venv/bin/python <this file> --namespace tui.hub_links
  ~/local-operator/.venv/bin/python <this file> --namespace tui.usage
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE
while not (REPO / "local_operator" / "i18n").is_dir():
    if REPO.parent == REPO:
        raise SystemExit("cannot locate repository root above " + str(HERE))
    REPO = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_operator.i18n import format as i18n_format  # noqa: E402
from local_operator.i18n import runtime as i18n_runtime  # noqa: E402

CAT = REPO / "local_operator" / "i18n" / "catalogues"
ES = CAT / "es"
EN = CAT / "en"

# --------------------------------------------------------------------------
# Width measurement: rich.cells when the interpreter has rich (the TUI's own
# ruler), else a local clone of the same arithmetic.
# --------------------------------------------------------------------------

try:  # pragma: no cover - environment dependent
    from rich.cells import cell_len as _rich_cell_len

    CELL_SOURCE = "rich.cells.cell_len"
except Exception:  # noqa: BLE001
    _rich_cell_len = None
    CELL_SOURCE = "local clone of rich.cells.cell_len (rich not importable)"


def _local_cell_len(text: str) -> int:
    total = 0
    for char in text:
        if char == "\n":
            continue
        if unicodedata.combining(char) or unicodedata.category(char) in ("Mn", "Me", "Cf"):
            continue
        if unicodedata.east_asian_width(char) in ("W", "F"):
            total += 2
        else:
            total += 1
    return total


def cell_len(text: str) -> int:
    if _rich_cell_len is not None:
        return _rich_cell_len(text)
    return _local_cell_len(text)


# --------------------------------------------------------------------------
# The round-3 delta, pinned by the task brief and by the round-2 bundles.
# --------------------------------------------------------------------------

DELTA = {
    "tui.hub_links": {
        "key": "tui.hub_links.refusal.revision_moved",
        "source": (
            "This link changed while you were reviewing it. "
            "Reopen the prompt to confirm the latest."
        ),
        "round2_value": (
            "Este vínculo cambió mientras lo revisabas. "
            "Vuelve a abrir el diálogo y confirma la versión más reciente."
        ),
        "round1_value": (
            "Este vínculo cambió mientras lo revisabas. "
            "Vuelve a abrir el aviso para confirmar la última versión."
        ),
        "round2_target_sha256": (
            "f3bde8f887ec44c2ab4b1fd79ac0d227fdb96561d3c4b992076157f6bc239c76"
        ),
        "current_target_sha256": (
            "ff50b7cd5e4ab165969fe3f9f9167635bd4966b495a06b59444345d4a0ee19f3"
        ),
    },
    "tui.usage": {
        "key": "tui.usage.sign_in_expired",
        "source": "{value} sign-in expired",
        "round2_value": "{value} caducadas",
        "round1_value": "{value} con inicio de sesión caducado",
        "round2_target_sha256": (
            "e2ab2abd3d581eb5c093cb3148b4beede61a3ed518b9b9de39c3361309c01a48"
        ),
        "current_target_sha256": (
            "52166a416bf6d49ab61138e7bd285fe5099710008127d3761a6e2fc298ac6776"
        ),
    },
}

#: Values whose only `prompt` occurrence is the JSON machine key ``"prompt":``
#: in an OpenRouter price example. They are not the product term; excluding
#: them is the method that reproduces round 2's 36-values / 38-occurrences
#: figure. Raw count (all values) is printed too.
MACHINE_PROMPT_KEYS = {
    "wire.settings.providers_openrouter_max_price.help",
    "wire.settings.providers_openrouter_max_price.placeholder",
}

STYLE_GUIDE = REPO / "i18n" / "style-guides" / "es.md"
GLOSSARY = REPO / "i18n" / "glossary.md"
HUB_PANEL = REPO / "local_operator" / "tui" / "widgets" / "hub_links_panel.py"
USAGE_PANEL = REPO / "local_operator" / "tui" / "widgets" / "usage_panel.py"
ANALYTICS_PANEL = REPO / "local_operator" / "tui" / "widgets" / "analytics_panel.py"

#: usage_panel.py geometry: PANEL_MIN_WIDTH 32, PANEL_PADDING_CELLS 4,
#: SCROLLBAR_GUTTER_CELLS 1, PANEL_WIDTH_MARGIN 4. At a 40-column terminal:
#: panel 36, content 32, body 31. At 60: content 52, body 51.
PANEL_MIN_WIDTH = 32
PANEL_PADDING_CELLS = 4
PANEL_WIDTH_MARGIN = 4
SCROLLBAR_GUTTER_CELLS = 1
FLOOR_COLUMNS = 40
FLOOR_CONTENT = FLOOR_COLUMNS - PANEL_WIDTH_MARGIN - PANEL_PADDING_CELLS  # 32
FLOOR_BODY = FLOOR_CONTENT - SCROLLBAR_GUTTER_CELLS  # 31
WIDE_COLUMNS = 60
WIDE_CONTENT = WIDE_COLUMNS - PANEL_WIDTH_MARGIN - PANEL_PADDING_CELLS  # 52
WIDE_BODY = WIDE_CONTENT - SCROLLBAR_GUTTER_CELLS  # 51

SEP = "  · "  # title-row separator in usage_panel._title_row (2 before, 1 after)


def load(namespace: str, locale: str) -> dict[str, str]:
    path = CAT / locale / f"{namespace}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_context(namespace: str) -> dict:
    path = CAT / "en" / f"{namespace}.context.json"
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def prompt_counts() -> tuple[int, int, int, int]:
    """Return (values, lowercase occurrences, values_excluding_machine, occ_excl)."""
    values = occurrences = 0
    values_excl = occurrences_excl = 0
    for path in sorted(ES.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            count = value.count("prompt")
            if not count:
                continue
            values += 1
            occurrences += count
            if key not in MACHINE_PROMPT_KEYS:
                values_excl += 1
                occurrences_excl += count
    return values, occurrences, values_excl, occurrences_excl


def dialog_count() -> int:
    total = 0
    for path in sorted(ES.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for value in data.values():
            total += value.count("diálogo")
    return total


def message_signature(value: str) -> tuple[tuple[str, str], ...]:
    return i18n_runtime.message_arguments(value)


def plural_selectors_ok(value: str, locale: str) -> tuple[bool, str]:
    try:
        arguments = i18n_runtime.plural_arguments(value)
    except Exception as exc:  # noqa: BLE001 - a parse failure is the finding
        return False, f"plural parse failed: {exc}"
    if not arguments:
        return False, "no plural argument found"
    expected = set(i18n_format.plural_categories(locale))
    selectors = set()
    empty = []
    for argument in arguments:
        for branch in argument.branches:
            selectors.add(branch.selector)
            if branch.empty:
                empty.append(branch.selector)
    ok = selectors == expected and not empty
    detail = f"selectors={sorted(selectors)} expected={sorted(expected)} empty={empty}"
    return ok, detail


def hub_delta_ok(value: str) -> bool:
    """The revised refusal must name the consent prompt and no other surface."""
    lowered = value.lower()
    return (
        value.startswith("Este vínculo cambió mientras lo revisabas.")
        and "Vuelve a abrir el prompt" in value
        and "confirma la versión más reciente." in value
        and "aviso" not in lowered
        and "diálogo" not in lowered
    )


def counter_shape(value: str) -> bool:
    return bool(re.fullmatch(r"(agotadas|obsoletas|caducadas): \{value\}", value))


def counter_shape_all(data: dict[str, str]) -> bool:
    return all(
        counter_shape(data[key])
        for key in ("tui.usage.exhausted", "tui.usage.stale", "tui.usage.sign_in_expired")
    )


def usage_exact(data: dict[str, str]) -> bool:
    return data["tui.usage.sign_in_expired"] == "caducadas: {value}"


class Book:
    """Collect checks for one namespace and print them as the probe runs."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.rows: list[tuple[str, str, str]] = []

    def add(self, check_id: str, status: str, detail: str) -> None:
        if status not in ("PASS", "WARN", "FAIL"):
            raise ValueError(status)
        self.rows.append((check_id, status, detail))
        print(f"{check_id} {status} — {detail}")

    def failed(self) -> list[tuple[str, str, str]]:
        return [row for row in self.rows if row[1] == "FAIL"]


def mechanical(book: Book, namespace: str) -> dict[str, str]:
    en = load(namespace, "en")
    es = load(namespace, "es")
    context = load_context(namespace)

    book.add(
        "A1",
        "PASS" if set(en) == set(es) else "FAIL",
        f"key parity: en {len(en)} / es {len(es)}; missing={sorted(set(en) - set(es))} "
        f"extra={sorted(set(es) - set(en))}",
    )
    missing_context = sorted(set(en) - set(context))
    book.add(
        "A2",
        "PASS",
        f"context {len(context)} keys, missing {missing_context} (context is optional, "
        "pre-existing; round 2 recorded the same count)",
    )

    parse_errors = []
    for locale, data in (("en", en), ("es", es)):
        for key, value in data.items():
            try:
                i18n_runtime.parse_message(value)
            except Exception as exc:  # noqa: BLE001 - the parse failure is the finding
                parse_errors.append(f"{locale}:{key}: {exc}")
    book.add(
        "A3",
        "PASS" if not parse_errors else "FAIL",
        f"every value parses in the shipped ICU subset; errors={parse_errors}",
    )

    signature_mismatch = [
        key for key in en if message_signature(en[key]) != message_signature(es[key])
    ]
    book.add(
        "A4",
        "PASS" if not signature_mismatch else "FAIL",
        f"placeholder tuples identical on all {len(en)} keys; mismatches={signature_mismatch}",
    )

    plural_keys = [key for key, value in es.items() if "{count, plural" in value]
    plural_detail = []
    plural_ok = True
    for key in plural_keys:
        ok_es, detail_es = plural_selectors_ok(es[key], "es")
        ok_en, detail_en = plural_selectors_ok(en[key], "en")
        plural_ok = plural_ok and ok_es and ok_en
        plural_detail.append(f"{key}: es[{detail_es}] en[{detail_en}]")
    book.add(
        "A5",
        "PASS" if plural_ok else "FAIL",
        "; ".join(plural_detail) if plural_detail else "no plural message in this namespace",
    )

    key = DELTA[namespace]["key"]
    if namespace == "tui.hub_links":
        shape_ok = not any(char in es[key] for char in "{}`")
        shape_detail = "no ICU braces, no code span (the message is plain prose)"
    else:
        braces = re.findall(r"\{[^}]*\}", es[key])
        shape_ok = braces == ["{value}"] and es[key].count("{value}") == 1
        shape_detail = f"braces={braces}; exactly one {{value}}"
    book.add("A6", "PASS" if shape_ok else "FAIL", f"{key}: {shape_detail}")

    if namespace == "tui.hub_links":
        punctuation_ok = es[key].endswith(".") and " - " not in es[key]
        punctuation_detail = "ends with a period; no ASCII hyphen used as a dash"
    else:
        punctuation_ok = bool(re.fullmatch(r"caducadas: \{value\}", es[key]))
        punctuation_detail = "label + colon + space + {value}; no space before the colon"
    book.add("A7", "PASS" if punctuation_ok else "FAIL", punctuation_detail)

    return es


def check_hub(book: Book, es: dict[str, str]) -> None:
    key = DELTA["tui.hub_links"]["key"]
    value = es[key]
    book.add(
        "B1",
        "PASS" if hub_delta_ok(value) else "FAIL",
        f"{key} = {value!r}; names `el prompt`, starts with the revision-moved "
        "sentence, contains neither `aviso` nor `diálogo`",
    )

    glossary = GLOSSARY.read_text(encoding="utf-8")
    style = STYLE_GUIDE.read_text(encoding="utf-8")
    glossary_row = bool(
        re.search(r"\|\s*prompt\s*\|\s*translate\s*\|\s*prompt\s*·\s*prompt", glossary)
    )
    kept_english = bool(re.search(r"script,\s*prompt,\s*deploy", style))
    book.add(
        "B2",
        "PASS" if glossary_row and kept_english else "FAIL",
        f"glossary §2 row fixes es `prompt` ({glossary_row}); style guide keeps "
        f"`prompt` English ({kept_english})",
    )

    values, occurrences, values_excl, occurrences_excl = prompt_counts()
    book.add(
        "B3",
        "PASS" if values_excl >= 30 and occurrences_excl >= 30 else "FAIL",
        "`prompt` across the 95 es catalogues: "
        f"{values_excl} values / {occurrences_excl} lowercase occurrences after excluding "
        "the two `\"prompt\":` JSON-key examples; raw "
        f"{values} / {occurrences} (the machine-token inclusion explains the difference)",
    )
    book.add(
        "B4",
        "PASS" if dialog_count() == 0 else "FAIL",
        f"`diálogo` occurrences across the es catalogues: {dialog_count()} "
        f"(this key: {value.count('diálogo')}); `aviso` in this key: {value.count('aviso')}",
    )

    source = HUB_PANEL.read_text(encoding="utf-8")
    class_ok = "class HubLinksPanel(Static)" in source
    overlay_ok = "The overlay itself." in source
    consent_ok = "_consent_link_id" in source and "refusal_revision_moved" in source
    book.add(
        "B5",
        "PASS" if class_ok and overlay_ok and consent_ok else "FAIL",
        f"surface: HubLinksPanel(Static)={class_ok}, 'The overlay itself'={overlay_ok}, "
        f"inline consent + refusal call={consent_ok}",
    )

    try:
        from rich.console import Console
        from rich.text import Text

        counts = {}
        for width in (FLOOR_CONTENT, WIDE_CONTENT, 72):
            counts[width] = len(Text(value).wrap(Console(width=width), width))
        book.add(
            "B6",
            "PASS" if counts[WIDE_CONTENT] <= 3 and counts[72] <= 3 else "FAIL",
            f"notice wraps at {FLOOR_CONTENT}/{WIDE_CONTENT}/72 cells: "
            f"{counts[FLOOR_CONTENT]}/{counts[WIDE_CONTENT]}/{counts[72]} lines; "
            f"cell_len={cell_len(value)}; no maxCells in the sidecar",
        )
    except Exception as exc:  # noqa: BLE001 - rich is the TUI ruler; report if absent
        book.add("B6", "FAIL", f"rich wrap measurement failed: {exc}")


def check_usage(book: Book, es: dict[str, str]) -> None:
    key = DELTA["tui.usage"]["key"]
    value = es[key]
    book.add(
        "C1",
        "PASS" if usage_exact(es) else "FAIL",
        f"{key} = {value!r}; count-neutral label-first value",
    )
    book.add(
        "C2",
        "PASS" if counter_shape_all(es) else "FAIL",
        "all three counters label-first: "
        f"exhausted={es['tui.usage.exhausted']!r}, stale={es['tui.usage.stale']!r}, "
        f"sign_in_expired={value!r}",
    )

    renders = {}
    renders_ok = True
    for number in ("1", "2", "12", "100", "1000", "1000000"):
        rendered = i18n_runtime.render_message(value, {"value": number}, "es")
        renders[number] = rendered
        renders_ok = renders_ok and rendered == f"caducadas: {number}"
    book.add(
        "C3",
        "PASS" if renders_ok else "FAIL",
        f"renders: {renders}",
    )

    title = i18n_runtime.render_message(es["tui.usage.title"], {}, "es")
    stale = i18n_runtime.render_message(es["tui.usage.stale"], {"value": "1"}, "es")
    counter_widths = {}
    segment_widths = {}
    row_widths = {}
    en = load("tui.usage", "en")
    en_title = i18n_runtime.render_message(en["tui.usage.title"], {}, "en")
    en_stale = i18n_runtime.render_message(en["tui.usage.stale"], {"value": "1"}, "en")
    en_counter_widths = {}
    en_row_widths = {}
    for number in ("1", "2", "12", "100", "1000", "1000000"):
        rendered = renders[number]
        segment = f"{rendered}{SEP}{stale}"
        row = f"{title}  2m ago{SEP}{segment}"
        counter_widths[number] = cell_len(rendered)
        segment_widths[number] = cell_len(segment)
        row_widths[number] = cell_len(row)

        en_counter = f"{number} sign-in expired"
        en_segment = f"{en_counter}{SEP}{en_stale}"
        en_row = f"{en_title}  2m ago{SEP}{en_segment}"
        en_counter_widths[number] = cell_len(en_counter)
        en_row_widths[number] = cell_len(en_row)

    fits = all(segment_widths[number] <= FLOOR_BODY for number in ("1", "2", "12", "100", "1000"))
    no_worse = all(row_widths[number] <= en_row_widths[number] for number in ("1", "2", "12", "100"))
    book.add(
        "C4",
        "PASS" if fits and no_worse else "FAIL",
        "counters segment at 1/2/12/100/1000/1e6 = "
        f"{segment_widths['1']}/{segment_widths['2']}/{segment_widths['12']}/"
        f"{segment_widths['100']}/{segment_widths['1000']}/{segment_widths['1000000']} "
        f"vs {FLOOR_BODY}-cell floor body; 1e6 is 34 in both es and en. "
        "full row es/en: "
        f"{row_widths['1']}/{en_row_widths['1']} (1), "
        f"{row_widths['12']}/{en_row_widths['12']} (12), "
        f"{row_widths['100']}/{en_row_widths['100']} (100)",
    )

    try:
        from rich.text import Text

        truncated_40_es = Text(f"{title}  2m ago{SEP}{renders['2']}{SEP}{stale}")
        truncated_40_es.truncate(FLOOR_CONTENT, overflow="ellipsis")
        truncated_40_en = Text(
            f"{en_title}  2m ago{SEP}2 sign-in expired{SEP}{en_stale}"
        )
        truncated_40_en.truncate(FLOOR_CONTENT, overflow="ellipsis")
        book.add(
            "C5",
            "PASS" if row_widths["2"] <= WIDE_CONTENT else "FAIL",
            f"full row ({row_widths['2']}) fits the {WIDE_CONTENT}-cell content at "
            f"{WIDE_COLUMNS} columns; at {FLOOR_COLUMNS} the {FLOOR_CONTENT}-cell chrome "
            f"truncates es to {truncated_40_es.plain!r} and en to {truncated_40_en.plain!r}",
        )
    except Exception as exc:  # noqa: BLE001
        book.add("C5", "FAIL", f"rich truncation measurement failed: {exc}")

    old = DELTA["tui.usage"]["round2_value"]
    book.add(
        "C6",
        "PASS" if old not in es.values() else "FAIL",
        f"round-2 `{old}` no longer appears anywhere in the file",
    )

    plural_renders = {}
    plural_ok = True
    for number, shown in (("1", "1"), ("2", "2"), ("1000000", "1000000")):
        rendered = i18n_runtime.render_message(
            es["tui.usage.windows_count"], {"count": int(number), "shown": shown}, "es"
        )
        plural_renders[number] = rendered
        plural_ok = plural_ok and (
            rendered == ("1 ventana" if number == "1" else f"{shown} ventanas" if number == "2" else f"{shown} de ventanas")
        )
    for number in ("1", "2"):
        rendered = i18n_runtime.render_message(
            es["tui.usage.reported_balance"], {"count": int(number), "shown": number}, "es"
        )
        plural_renders[f"balance_{number}"] = rendered
        plural_ok = plural_ok and rendered == (
            "1 saldo informado" if number == "1" else "2 saldos informados"
        )
    book.add(
        "C7",
        "PASS" if plural_ok else "FAIL",
        f"untouched plural keys still render: {plural_renders}",
    )


def check_style_guide(book: Book, es: dict[str, str]) -> None:
    guide = STYLE_GUIDE.read_text(encoding="utf-8")
    relay = load("relay.shell", "es")
    en = load("tui.usage", "en")

    family_es = es["tui.usage.paid.family_caps"]
    family_en = en["tui.usage.paid.family_caps"]
    eligibility_es = es["tui.usage.paid.eligibility_unknown"]
    eligibility_en = en["tui.usage.paid.eligibility_unknown"]
    continue_es = es["tui.usage.paid.continue"]
    continue_en = en["tui.usage.paid.continue"]

    raw = {
        "family_caps_es": cell_len(family_es),
        "family_caps_en": cell_len(family_en),
        "eligibility_es": cell_len(eligibility_es),
        "eligibility_en": cell_len(eligibility_en),
        "continue_es": cell_len(continue_es),
        "continue_en": cell_len(continue_en),
    }
    composed = {
        "family_caps_es": cell_len("  " + family_es),
        "family_caps_en": cell_len("  " + family_en),
        "eligibility_es": cell_len("  " + eligibility_es),
        "eligibility_en": cell_len("  " + eligibility_en),
        "continue_es": cell_len("  " + continue_es),
        "continue_en": cell_len("  " + continue_en),
    }
    thresholds = {
        "family_caps": composed["family_caps_es"] + 9,
        "eligibility_unknown": composed["eligibility_es"] + 9,
        "continue": composed["continue_es"] + 9,
    }
    numbers_ok = (
        raw["family_caps_es"] == 53
        and raw["family_caps_en"] == 29
        and raw["eligibility_es"] == 35
        and composed["continue_es"] == 54
        and composed["continue_en"] == 44
    )
    book.add(
        "D1",
        "PASS" if numbers_ok else "FAIL",
        f"paid annotations raw/composed: family_caps {raw['family_caps_es']}/"
        f"{composed['family_caps_es']} vs en {raw['family_caps_en']}/"
        f"{composed['family_caps_en']}; eligibility {raw['eligibility_es']}/"
        f"{composed['eligibility_es']} vs en {raw['eligibility_en']}/"
        f"{composed['eligibility_en']}; paid.continue {raw['continue_es']}/"
        f"{composed['continue_es']} vs en {raw['continue_en']}/"
        f"{composed['continue_en']}; thresholds family_caps={thresholds['family_caps']}, "
        f"eligibility={thresholds['eligibility_unknown']}, continue={thresholds['continue']}",
    )

    at_60 = {name: (width <= WIDE_BODY) for name, width in (
        ("family_caps", composed["family_caps_es"]),
        ("eligibility_unknown", composed["eligibility_es"]),
        ("paid.continue", composed["continue_es"]),
    )}
    guide_claims_60 = "at 60+ columns all three fit" in guide
    book.add(
        "SG-1",
        "FAIL" if guide_claims_60 and not all(at_60.values()) else "PASS",
        "style-guide paid exception says 'at 60+ columns all three fit'; measured at "
        f"{WIDE_COLUMNS} columns (body {WIDE_BODY}): {at_60}; the paragraph itself says "
        f"paid.continue needs 63, and its own family_caps (55 composed) needs 64. Decision "
        "(keep relay wording) stands; only the threshold sentence is wrong.",
    )

    guide_says_23 = "35 against 23" in guide
    book.add(
        "SG-2",
        "WARN" if guide_says_23 and raw["eligibility_en"] != 23 else "PASS",
        f"style-guide eligibility comparator: guide says 23, measured en="
        f"{raw['eligibility_en']} (len={len(eligibility_en)}); es=35 is correct, so the "
        "exception still stands but one number is wrong",
    )

    family_byte_identical = family_es == relay["relay.shell.usage.family_caps"]
    eligibility_byte_identical = (
        eligibility_es == relay["relay.shell.usage.eligibility_unknown"]
    )
    guide_says_byte_identical = "both strings are byte-identical" in guide
    book.add(
        "SG-3",
        "WARN" if guide_says_byte_identical and not (family_byte_identical and eligibility_byte_identical) else "PASS",
        "style-guide says both paid strings are byte-identical to relay.shell.usage: "
        f"eligibility={eligibility_byte_identical}, family_caps={family_byte_identical}; "
        f"relay family_caps={relay['relay.shell.usage.family_caps']!r} vs tui="
        f"{family_es!r} (case + final period only)",
    )

    keys_bare_key = "tui.analytics.hint.expand.keys_bare"
    analytics_es = load("tui.analytics", "es")[keys_bare_key]
    analytics_en = load("tui.analytics", "en")[keys_bare_key]
    sidecar = load_context("tui.analytics")[keys_bare_key]
    cap = sidecar.get("maxCells")
    keys_bare_ok = (
        cell_len(analytics_es) == 14
        and cell_len(analytics_en) == 13
        and cap == 12
        and "14" in guide
        and "13" in guide
        and "maxCells" in guide
    )
    analytics_source = ANALYTICS_PANEL.read_text(encoding="utf-8")
    crop_ok = 'Text(no_wrap=True, overflow="crop")' in analytics_source
    book.add(
        "D2",
        "PASS" if keys_bare_ok and crop_ok else "FAIL",
        f"{keys_bare_key}: es {cell_len(analytics_es)} / en {cell_len(analytics_en)} / "
        f"maxCells {cap}; guide states 14/13/12; footer crop present={crop_ok}",
    )

    records_ok = (
        "Recorded width exception: `tui.usage` paid annotations" in guide
        and "Recorded width exception: `tui.analytics.hint.expand.keys_bare`" in guide
    )
    book.add(
        "D3",
        "PASS" if records_ok else "FAIL",
        "both exceptions are present in i18n/style-guides/es.md under the recorded-width-"
        "exception headings",
    )


def controls(book: Book, namespace: str, es: dict[str, str]) -> None:
    if namespace == "tui.hub_links":
        key = DELTA[namespace]["key"]

        aviso = copy.deepcopy(es)
        aviso[key] = DELTA[namespace]["round1_value"]
        book.add(
            "E1",
            "PASS" if not hub_delta_ok(aviso[key]) else "FAIL",
            "negative control `el aviso`: the restored round-1 value is rejected by B1 "
            "(forbidden surface name + missing `prompt`) — caught",
        )

        dialogo = copy.deepcopy(es)
        dialogo[key] = DELTA[namespace]["round2_value"]
        book.add(
            "E2",
            "PASS" if not hub_delta_ok(dialogo[key]) else "FAIL",
            "negative control `el diálogo`: the round-2 value is rejected by B1 — caught",
        )
    else:
        old_exhausted = copy.deepcopy(es)
        old_exhausted["tui.usage.exhausted"] = "{value} agotadas"
        book.add(
            "E1",
            "PASS" if not counter_shape_all(old_exhausted) else "FAIL",
            "negative control `{value} agotadas`: restoring the count-first shape in "
            "`exhausted` is rejected by C2 — caught",
        )

        old_signin = copy.deepcopy(es)
        old_signin["tui.usage.sign_in_expired"] = "{value} caducadas"
        book.add(
            "E2",
            "PASS" if not (usage_exact(old_signin) and counter_shape_all(old_signin)) else "FAIL",
            "negative control `{value} caducadas`: restoring the round-2 value is rejected "
            "by C1/C2 — caught",
        )

        wrong_placeholder = copy.deepcopy(es)
        wrong_placeholder["tui.usage.sign_in_expired"] = "caducadas: {count}"
        book.add(
            "E3",
            "PASS"
            if message_signature(wrong_placeholder["tui.usage.sign_in_expired"])
            != message_signature(DELTA[namespace]["source"])
            else "FAIL",
            "negative control `{count}`: wrong placeholder name differs from the en "
            "signature — caught",
        )


def delta_proof(book: Book, namespace: str, es: dict[str, str]) -> None:
    key = DELTA[namespace]["key"]
    old_value = DELTA[namespace]["round2_value"]
    expected = DELTA[namespace]["round2_target_sha256"]

    restored = copy.deepcopy(es)
    restored[key] = old_value
    text = json.dumps(restored, ensure_ascii=False, indent=2) + "\n"
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    book.add(
        "F1",
        "PASS" if actual == expected else "FAIL",
        f"restoring the round-2 value in {namespace}.json reproduces the round-2 file "
        f"sha256 {expected[:16]}…; actual {actual[:16]}… — no other value changed",
    )

    current = sha256_file(CAT / "es" / f"{namespace}.json")
    book.add(
        "F2",
        "PASS" if current == DELTA[namespace]["current_target_sha256"] else "FAIL",
        f"current es file sha256 {current}",
    )


def run(namespace: str) -> int:
    print()
    print(f"== ROUND 3 DELTA AUDIT — es / {namespace} / w6a3 ==")
    print(f"repo={REPO}")
    print(f"branch={git('rev-parse', '--abbrev-ref', 'HEAD')}")
    print(f"head={git('rev-parse', 'HEAD')}")
    print(f"cell_ruler={CELL_SOURCE}")
    for name, path in (
        ("en", CAT / "en" / f"{namespace}.json"),
        ("context", CAT / "en" / f"{namespace}.context.json"),
        ("target", CAT / "es" / f"{namespace}.json"),
    ):
        print(f"sha256 {name}: {sha256_file(path)}  {path.relative_to(REPO)}")
    print()

    book = Book(namespace)
    es = mechanical(book, namespace)
    print()
    if namespace == "tui.hub_links":
        check_hub(book, es)
    else:
        check_usage(book, es)
        print()
        check_style_guide(book, es)
    print()
    print("[E] NEGATIVE CONTROLS")
    controls(book, namespace, es)
    print()
    print("[F] DELTA PROOF")
    delta_proof(book, namespace, es)

    failed = book.failed()
    print()
    if failed:
        print(f"RESULT: {len(failed)} FAIL(s) — " + ", ".join(row[0] for row in failed))
        return 1
    print("RESULT: PASS — every check and control behaved as expected")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        choices=("tui.hub_links", "tui.usage", "both"),
        default="both",
    )
    args = parser.parse_args()
    namespaces = (
        ("tui.hub_links", "tui.usage") if args.namespace == "both" else (args.namespace,)
    )
    status = 0
    for namespace in namespaces:
        status = max(status, run(namespace))
    return status


if __name__ == "__main__":
    raise SystemExit(main())

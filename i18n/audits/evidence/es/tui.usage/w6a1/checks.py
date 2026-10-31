#!/usr/bin/env python3
"""Independent audit probe — es TUI block, wave w6a1 (three namespaces).

READ-ONLY BY CONSTRUCTION. This script never writes a catalogue, sidecar,
ledger, allowance or generated artefact: it loads them, measures them and
prints. Every mutation it needs is applied to an IN-MEMORY copy of a string,
which is what the negative-control section is for — a check that cannot fail
proves nothing, so each control breaks a value deliberately and reports
whether the corresponding check caught it.

WHAT IT CHECKS
  A  MECHANICAL (must all PASS for any verdict other than changes_requested)
     key parity, ICU argument tuples, CLDR plural-category coverage for `es`
     (one/many/other — the repo validator only rejects categories OUTSIDE the
     locale's set, so a plural missing `many` passes it), `#` usage, printf
     tokens, `--flags`, backtick code spans, machine tokens, punctuation and
     whitespace padding that carries layout meaning.
  B  FIT
     declared `maxCells` (tui.hub_links), and the measured ceilings the code
     itself pins: the /help key rows' 74-cell bound at 80 columns, and the
     usage panel's body width at the supported 40x24 floor.
  C  SEMANTICS / TERMINOLOGY
     the settled es terminology (`vínculo` vs `enlace`, `contacto` for a
     check-in, `finalizados` for a settled goal), agreement against the real
     placeholder values the code passes, and the money claims in the usage
     readout against the shipped relay.shell copy.
  D  NEGATIVE CONTROLS
     eight deliberate breakages, each of which a section-A/B check must catch.

Run (from the repository root):
    PYTHONPATH=$PWD python3 i18n/audits/evidence/es/tui.hub_links/w6a1/checks.py
or for one namespace:
    PYTHONPATH=$PWD python3 .../checks.py --namespace tui.app

Exit code 0 only when every check passes. Findings are printed as FAIL lines
with the catalogue key, so the audit log doubles as the finding index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

# --------------------------------------------------------------------------
# Repository + runtime import (walk up until the package root is found).
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve()
REPO = HERE
while not (REPO / "local_operator" / "i18n").is_dir():
    if REPO.parent == REPO:
        raise SystemExit("cannot locate repository root above " + str(HERE))
    REPO = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_operator.i18n import runtime as rt  # noqa: E402

CAT = REPO / "local_operator" / "i18n" / "catalogues"
PLURAL_RULES = REPO / "local_operator" / "i18n" / "data" / "plural_rules.json"
LOCALE = "es"
NAMESPACES = ("tui.hub_links", "tui.app", "tui.usage")

FLAG = re.compile(r"(?<![\w-])--[A-Za-z0-9][\w-]*")
CODE = re.compile(r"`[^`]*`")
PRINTF = re.compile(r"%\([^)]+\)[a-zA-Z]|%%")

#: Tokens whose exact spelling and padding are layout or machine contract, per
#: the sidecar notes (NBSPs keep a count glued to its "of"; the two leading
#: spaces are list indents; `·`, `—`, `–` and `…` are the shipped separators).
MACHINE_TOKENS = (
    "/stop all",
    "/notifications read",
    "/agent clear",
    "/agent <name> <message>",
    "/new",
    "config.yml",
    "LOCAL_OPERATOR_NO_TERMINAL_TITLE=1",
    "LOCAL_OPERATOR_NO_NOTIFICATIONS=1",
    "lop config edit display.terminal_title false",
    "lop config edit display.notifications false",
    "ctrl+c",
    "ctrl+b",
    "ctrl+y",
    "Terminal.app",
    "lo ›",
    "lo ⣾",
    "lo !",
    "{first}–{last}/{total}",
)

results: list[tuple[str, str, bool, str]] = []


def ck(section: str, name: str, ok: bool, detail: str = "") -> bool:
    """Record and print one check result."""
    results.append((section, name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    line = f"{mark}  [{name}]"
    if detail:
        print(f"{line} {detail}")
    else:
        print(line)
    return ok


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------
# Width measurement: rich.cells when the interpreter has rich (the TUI's own
# measure), otherwise a local clone of its algorithm. Both are printed so the
# log says which one produced the numbers.
# --------------------------------------------------------------------------

try:  # pragma: no cover - environment dependent
    from rich.cells import cell_len as _rich_cell_len

    CELL_SOURCE = "rich.cells.cell_len"
except Exception:  # noqa: BLE001 - the fallback is the point
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
# Loaders
# --------------------------------------------------------------------------


def load(ns: str, report: bool = True) -> tuple[dict[str, str], dict[str, str], dict[str, dict]]:
    """Read the three inputs and prove the real loader agrees with the bytes.

    `json.loads` only proves well-formed JSON; the shipped catalogues are read
    through `catalogues.load_catalogue`, so the same path is exercised and its
    digest is compared with this script's own sha256 of the file.
    """
    en_path = CAT / "en" / f"{ns}.json"
    target_path = CAT / LOCALE / f"{ns}.json"
    en = json.loads(en_path.read_text(encoding="utf-8"))
    target = json.loads(target_path.read_text(encoding="utf-8"))
    ctx_path = CAT / "en" / f"{ns}.context.json"
    ctx = json.loads(ctx_path.read_text(encoding="utf-8")) if ctx_path.exists() else {}

    from local_operator.i18n import catalogues as cat_api

    loaded = cat_api.load_catalogue(LOCALE, ns)
    own_digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    api_digest = cat_api.catalogue_sha256(LOCALE, ns)
    if report:
        ck(ns, f"{ns}: catalogue loader accepts the file and agrees on its digest",
           loaded == target and api_digest == own_digest,
           f"loader {len(loaded)} keys / api {api_digest[:16]} vs file {own_digest[:16]}")
    return en, target, ctx


def es_categories() -> set[str]:
    table = json.loads(PLURAL_RULES.read_text(encoding="utf-8"))
    return set(table["locales"][LOCALE]["categories"])


def selector_sets(message: str) -> list[tuple[str, ...]]:
    return list(rt.plural_selectors(message))


def renders_value(message: str) -> tuple[tuple[str, bool], ...]:
    """(selector, uses-#) per plural branch, so `#` parity is checkable."""
    out = []
    for argument in rt.plural_arguments(message):
        for branch in argument.branches:
            out.append((branch.selector, branch.renders_value))
    return tuple(out)


def renders_branch_args(node) -> frozenset:
    """Names of every argument a plural branch body renders."""
    found: set[str] = set()
    for child in node:
        if isinstance(child, rt._Arg):
            found.add(child.name)
            for _, body in child.parts:
                found |= renders_branch_args(body)
    return frozenset(found)


def branch_argument_sets(message: str) -> dict[tuple[str, str], frozenset]:
    """(plural argument, selector) -> the arguments that branch body renders.

    A plural branch exists to render the COUNTED value. A branch that drops
    `{shown}` still parses, keeps the message-level argument tuple intact
    (other branches use it) and therefore passes the repo validator — so the
    loss is only visible per branch, which is what this map is for.
    """
    out: dict[tuple[str, str], frozenset] = {}
    for node in rt.parse_message(message):
        if isinstance(node, rt._Arg) and node.kind == "plural":
            for label, body in node.parts:
                out[(node.name, label)] = renders_branch_args(body)
    return out


# --------------------------------------------------------------------------
# A. Mechanical checks
# --------------------------------------------------------------------------


def per_key_defects(key: str, source: str, value: str, categories: set[str]) -> dict[str, list]:
    """Every section-A defect for ONE key, as a dict of check -> problems.

    Factored out so the negative controls can run the SAME predicates over a
    deliberately broken value: a control that re-implements the comparison
    proves nothing about the check that gates the audit.
    """
    out: dict[str, list] = {name: [] for name in (
        "arguments", "categories", "branch_terms", "hash", "printf", "flags", "code",
        "tokens", "padding")}

    if rt.message_arguments(source) != rt.message_arguments(value):
        out["arguments"].append((key, rt.message_arguments(source), rt.message_arguments(value)))

    source_sets, value_sets = selector_sets(source), selector_sets(value)
    if len(source_sets) != len(value_sets):
        out["categories"].append((key, "plural count", source_sets, value_sets))
    for argument in rt.plural_arguments(value):
        for branch in argument.branches:
            if branch.empty:
                out["categories"].append((key, "empty branch", argument.argument,
                                          branch.selector))
    for sel in value_sets:
        if "other" not in sel:
            out["categories"].append((key, "no other", sel, ""))
        unknown = [s for s in sel if not s.startswith("=") and s not in categories]
        if unknown:
            out["categories"].append((key, "not an es category", unknown, ""))
        absent = sorted(c for c in categories if c not in sel and not any(
            s.startswith("=") for s in sel))
        if absent and "other" in sel:
            out["categories"].append((key, "missing es category", absent, sel))

    # Every branch must render the same arguments as its source counterpart.
    # A category `es` adds (many) has no counterpart: it must match the source's
    # `other` branch, because that is the branch carrying the same content.
    source_branches = branch_argument_sets(source)
    for (argument, label), terms in branch_argument_sets(value).items():
        counterpart = source_branches.get((argument, label)) or source_branches.get(
            (argument, "other"))
        if counterpart is not None and terms != counterpart:
            out["branch_terms"].append((key, argument, label, sorted(terms), sorted(counterpart)))

    # `#` usage per selector must match the source for the selectors the
    # SOURCE has; `es` legitimately adds `many` (a category en lacks), so
    # comparing the raw tuples would flag every correct es plural.
    want_hash = dict(renders_value(source))
    got_hash = dict(renders_value(value))
    hash_mismatch = [
        (sel, want_hash[sel], got_hash[sel]) for sel in want_hash
        if sel in got_hash and want_hash[sel] != got_hash[sel]
    ]
    if hash_mismatch:
        out["hash"].append((key, hash_mismatch))

    if PRINTF.findall(source) != PRINTF.findall(value):
        out["printf"].append((key, PRINTF.findall(source), PRINTF.findall(value)))
    if FLAG.findall(source) != FLAG.findall(value):
        out["flags"].append((key, FLAG.findall(source), FLAG.findall(value)))
    if CODE.findall(source) != CODE.findall(value):
        out["code"].append((key, CODE.findall(source), CODE.findall(value)))

    for token in MACHINE_TOKENS:
        if (token in source) != (token in value):
            out["tokens"].append((key, token, token in source, token in value))

    # Padding and separators that the sidecars call load-bearing.
    for mark in ("\u00a0", "\u00b7", "\u2014", "\u2013", "\u2026", "  "):
        if source.count(mark) != value.count(mark):
            out["padding"].append((key, repr(mark), source.count(mark), value.count(mark)))
    if (source[:2] == "  ") != (value[:2] == "  "):
        out["padding"].append((key, "leading 2-space indent", source[:2], value[:2]))
    if (source.endswith(" ") != value.endswith(" ")) or (
        source.startswith(" ") != value.startswith(" ")
    ):
        out["padding"].append((key, "edge space", repr(source[:1] + source[-1:]),
                               repr(value[:1] + value[-1:])))
    return out


def mechanical(ns: str, en: dict[str, str], target: dict[str, str], ctx: dict) -> int:
    failures = 0
    print(f"\n-- {ns}: {len(en)} en / {len(target)} target / {len(ctx)} context keys")

    if not ck(ns, f"{ns}: ordered key parity en==es", list(en) == list(target),
              f"missing={[k for k in en if k not in target]} extra={[k for k in target if k not in en]}"):
        failures += 1
    if not ck(ns, f"{ns}: context sidecar only describes real keys",
              all(k in en for k in ctx), f"unknown={[k for k in ctx if k not in en]}"):
        failures += 1
    missing_ctx = [k for k in en if k not in ctx]
    print(f"INFO  [{ns}: keys without a context entry] {missing_ctx}")

    categories = es_categories()
    print(f"INFO  [{ns}: CLDR categories for es] {sorted(categories)}")

    arg_bad, cat_bad, hash_bad, printf_bad, flag_bad, code_bad = [], [], [], [], [], []
    branch_bad = []
    tok_bad, pad_bad, plural_count = [], [], 0
    for key, source in en.items():
        value = target[key]
        defects = per_key_defects(key, source, value, categories)
        arg_bad += defects["arguments"]
        cat_bad += defects["categories"]
        branch_bad += defects["branch_terms"]
        hash_bad += defects["hash"]
        printf_bad += defects["printf"]
        flag_bad += defects["flags"]
        code_bad += defects["code"]
        tok_bad += defects["tokens"]
        pad_bad += defects["padding"]
        plural_count += len(selector_sets(value))

    checks = [
        ("ICU argument tuples (name and kind) match en", arg_bad),
        ("plural selectors are es CLDR categories, all present, `other` defined", cat_bad),
        ("every plural branch renders its source's arguments", branch_bad),
        ("`#`/value rendering per plural branch matches en", hash_bad),
        ("printf tokens preserved", printf_bad),
        ("--flags preserved", flag_bad),
        ("backtick code spans byte-identical", code_bad),
        ("machine tokens present in both or neither", tok_bad),
        ("layout punctuation/whitespace counts match en", pad_bad),
    ]
    for name, bad in checks:
        if not ck(ns, f"{ns}: {name}", not bad, f"{len(bad)} mismatch(es): {bad[:4]}"):
            failures += 1
    print(f"INFO  [{ns}: plural arguments seen] {plural_count}")
    return failures


# --------------------------------------------------------------------------
# B. Fit
# --------------------------------------------------------------------------

#: `/help` composes every key row as `name.ljust(name_width) + note`. The code
#: derives name_width from the longest slash command; it is 20 on this registry
#: (the source comments' arithmetic — 51+20=71 for ctrl+v, 54+20=74 for cmd+v —
#: confirms it), and pins the composed row <= 74 at 80 columns.
HELP_NAME_WIDTH = 20
HELP_CEILING = 74
HELP_ROW_KEYS = (
    "tui.app.help.paste_note",
    "tui.app.help.copy_note",
    "tui.app.help.copy_note_more",
    "tui.app.help.skill_note",
    "tui.app.help.model_saved_note",
    "tui.app.help.cmd_paste_note",
)

#: usage_panel geometry: PANEL_MIN_WIDTH 32, PANEL_PADDING_CELLS 4,
#: SCROLLBAR_GUTTER_CELLS 1 -> a 31-cell body at the supported 40x24 floor.
#: The paid note row is `TIER_INDENT + note`, truncated to that width with an
#: ellipsis; the title row is `count + '  · ' + count`, truncated likewise.
USAGE_FLOOR_BODY = 31
USAGE_TIER_INDENT = 2
USAGE_PAID_KEYS = (
    "tui.usage.paid.family_caps",
    "tui.usage.paid.continue",
    "tui.usage.paid.eligibility_unknown",
    "tui.usage.paid.unlimited",
    "tui.usage.paid.blocked",
)


def fit(ns: str, en: dict[str, str], target: dict[str, str], ctx: dict) -> int:
    failures = 0
    if ns == "tui.hub_links":
        for key, entry in ctx.items():
            budget = entry.get("maxCells")
            if not isinstance(budget, int):
                continue
            print(f"\n-- {key}: maxCells {budget}")
            for reason in ("provider error", "unsupported precondition", "baseline unknown"):
                e = rt.render_message(en[key], {"reason": reason}, locale="en")
                t = rt.render_message(target[key], {"reason": reason}, locale=LOCALE)
                print(f"   reason={reason!r:28} en {cell_len(e):3d} | {LOCALE} {cell_len(t):3d}"
                      f"  {t!r}")
                if not ck(ns, f"{ns}: {key} <= {budget} cells incl. reason={reason!r}",
                          cell_len(t) <= budget, f"{cell_len(t)} cells"):
                    failures += 1

    if ns == "tui.app":
        print(f"\n-- /help key rows: composed = {HELP_NAME_WIDTH} + note, ceiling "
              f"{HELP_CEILING} at 80 columns")
        for key in HELP_ROW_KEYS:
            e, t = cell_len(en[key]), cell_len(target[key])
            print(f"   {key:32s} en {e:3d} -> {HELP_NAME_WIDTH + e:3d}"
                  f"{' (pinned)' if HELP_NAME_WIDTH + e == HELP_CEILING else ''}"
                  f" | es {t:3d} -> {HELP_NAME_WIDTH + t:3d}")
            if not ck(ns, f"{ns}: {key} composed row <= {HELP_CEILING}",
                      HELP_NAME_WIDTH + t <= HELP_CEILING,
                      f"composed {HELP_NAME_WIDTH + t} cells (ceiling {HELP_CEILING}, "
                      f"en {HELP_NAME_WIDTH + e})"):
                failures += 1

    if ns == "tui.usage":
        print(f"\n-- usage panel body at the 40x24 floor: {USAGE_FLOOR_BODY} cells")
        for key in USAGE_PAID_KEYS:
            e = USAGE_TIER_INDENT + cell_len(en[key])
            t = USAGE_TIER_INDENT + cell_len(target[key])
            note = " (en is over the floor too)" if e > USAGE_FLOOR_BODY else ""
            print(f"   {key:38s} en {e:3d} | es {t:3d}{note}")
            if not ck(ns, f"{ns}: {key} fits the floor body where en does",
                      t <= USAGE_FLOOR_BODY or e > USAGE_FLOOR_BODY,
                      f"es {t} vs floor {USAGE_FLOOR_BODY}, en {e}"):
                failures += 1
        # The mixed-state title row: `2 sign-in expired  · 2 stale`. The code
        # row comment says both counts are kept because they ask for opposite
        # things, and the row truncates from the right — so a longer first
        # clause silently drops the second count at narrow widths.
        parts = []
        for book, name in ((en, "en"), (target, LOCALE)):
            si = book["tui.usage.sign_in_expired"].replace("{value}", "2")
            st = book["tui.usage.stale"].replace("{value}", "2")
            row = f"{si}  \u00b7 {st}"
            parts.append((name, cell_len(row), row))
            print(f"   mixed title row {name}: {cell_len(row):3d} cells  {row!r}")
        (_, en_row, _), (_, es_row, _) = parts
        if not ck(ns, f"{ns}: mixed title row fits the floor body where en does",
                  es_row <= USAGE_FLOOR_BODY or en_row > USAGE_FLOOR_BODY,
                  f"es {es_row} vs floor {USAGE_FLOOR_BODY}, en {en_row}"):
            failures += 1
    return failures


# --------------------------------------------------------------------------
# C. Semantics and terminology (explicit, reviewable assertions)
# --------------------------------------------------------------------------


def shipped(ns: str) -> dict[str, str]:
    """Another es catalogue used as a terminology witness (NOT assumed approved)."""
    return json.loads((CAT / LOCALE / f"{ns}.json").read_text(encoding="utf-8"))


def tracked(ns: str) -> bool:
    """Is that witness file committed at HEAD, or only in the working tree?

    A witness that is untracked is a sibling translator's in-flight work, not
    shipped precedent — the report says which, and never the other way round.
    """
    import subprocess

    rel = (CAT / LOCALE / f"{ns}.json").relative_to(REPO)
    return subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)],
                          cwd=str(REPO), capture_output=True).returncode == 0


def label_values(target: dict[str, str]) -> list[str]:
    """The only values `tui.app.config.toggle_*` ever receive as `{label}`.

    Read from the call site, not assumed: `app.py` passes the two catalogue
    switch names and nothing else.
    """
    return sorted({target["tui.app.config.label_web_search"],
                   target["tui.app.config.label_web_fetch"]})


def agrees(template: str, label: str) -> bool:
    """Does the state adjective in `template` agree with `label`'s head noun?

    Spanish adjectives agree with the noun the label NAMES ("búsqueda web" ->
    feminine), not with the string's last character. Checked longest-first:
    "desactivado" contains "activado".
    """
    head = label.split()[0].lower()
    feminine = head.endswith("a")
    for masculine, fem in (("desactivado", "desactivada"), ("activado", "activada")):
        if fem in template:
            return feminine
        if masculine in template:
            return not feminine
    return True


def render(template: str, label: str) -> str:
    return template.replace("{label}", label)


def states_credits_continue(text: str) -> bool:
    """The money claim: included quota is spent and billing continues on credits."""
    return "créditos" in text and any(
        word in text for word in ("continuar", "continua", "continúa", "sigue"))


def semantics(ns: str, en: dict[str, str], target: dict[str, str]) -> int:
    failures = 0
    if ns == "tui.hub_links":
        # Word order within each key: ES then EN. Reported in the audit log.
        print("\n-- terminology: configured link is `vínculo`, hyperlink is `enlace`")
        vinculo = [k for k, v in target.items() if "vínculo" in v]
        enlace = [k for k, v in target.items() if "enlace" in v]
        print(f"   vínculo in {len(vinculo)} keys: {vinculo}")
        print(f"   enlace in {len(enlace)} keys: {enlace}")
        if not ck(ns, f"{ns}: configured hub links use `vínculo`, never `enlace`",
                  not enlace and len(vinculo) >= 5,
                  f"enlace={enlace} vínculo_count={len(vinculo)}"):
            failures += 1
        link_witness = shipped("wire.settings.hub_defaults")
        print(f"   witness: {link_witness['wire.settings.hub_defaults.teams.label']!r}"
              f" (tracked={tracked('wire.settings.hub_defaults')}, ledger passed at the"
              f" current en hash)")

        # The recovery string must name the object the user has to reopen. The
        # panel shows this text on its own `_notice` line, so "el aviso"
        # points at the message itself, not at the consent prompt.
        moved = target["tui.hub_links.refusal.revision_moved"]
        print(f"\n-- recovery referent: {moved!r}")
        if not ck(ns, f"{ns}: revision_moved names the consent prompt, not the notice line",
                  "prompt" in moved or "confirmación" in moved,
                  "says `el aviso`, which is the panel's own notice line"):
            failures += 1

        # Unsubscribe must not claim that outgoing pushes stop.
        unsub = target["tui.hub_links.state.unsubscribed"]
        if not ck(ns, f"{ns}: unsubscribed keeps both halves (inbound paused, outbound may publish)",
                  "entrantes" in unsub and "salientes" in unsub and "publicarse" in unsub,
                  unsub):
            failures += 1

    if ns == "tui.app":
        # 1. Agreement: {label} is always one of the two feminine switch names.
        labels = label_values(target)
        feminine = all(label.split()[0].lower().endswith("a") for label in labels)
        print(f"\n-- {{label}} values: {labels} (all feminine: {feminine})")
        for key in ("tui.app.config.toggle_on", "tui.app.config.toggle_off"):
            value = target[key]
            if not ck(ns, f"{ns}: {key} agrees with the switch names it receives",
                      all(agrees(value, label) for label in labels),
                      "renders " + repr(render(value, labels[0]))):
                failures += 1
        print(f"   rendered on  : {render(target['tui.app.config.toggle_on'], labels[0])!r}")
        print(f"   rendered off : {render(target['tui.app.config.toggle_off'], labels[1])!r}")

        # 2. `settled` goals: the shipped catalogue already renders this heading.
        goal_panel = shipped("tui.small")["tui.small.goal_panel.settled_heading"]
        goal_tracked = tracked("tui.small")
        settled = target["tui.app.goals.settled_heading"]
        print(f"\n-- settled-goal heading: this block {settled!r} vs the goal panel's own"
              f" heading {goal_panel!r} (tui.small.json tracked={goal_tracked})")
        if not ck(ns, f"{ns}: settled goals match the goal panel's heading for the same list",
                  goal_panel.rstrip("s") in settled or "finalizad" in settled,
                  f"{settled!r} is a second rendering of {goal_panel!r}; the goals it lists are"
                  f" the completed ones (goal_panel.py strikes them through and suffixes `done`)"):
            failures += 1

        # 3. check-in vocabulary, checked against the shipped Aida settings.
        aida = shipped("wire.settings")
        print(f"\n-- check-in: es {target['tui.app.aida.next_checkin']!r} /"
              f" {target['tui.app.aida.extras_used']!r}")
        print(f"   witness: {aida['wire.settings.aida_cadence_at.label']!r}"
              f" (wire.settings.json tracked={tracked('wire.settings')}, "
              f"ledger passed at the current en hash)")
        if not ck(ns, f"{ns}: check-in uses the shipped `contacto` wording",
                  "contacto" in target["tui.app.aida.next_checkin"]
                  and "contactos" in target["tui.app.aida.extras_used"]):
            failures += 1
        # The two NBSPs are contract, not style.
        if not ck(ns, f"{ns}: extras_used keeps both NBSPs around `de`",
                  target["tui.app.aida.extras_used"].count("\u00a0") == 2,
                  repr(target["tui.app.aida.extras_used"])):
            failures += 1

        # 4. A destructive stop must stay explicit: stopped, not left to finish.
        stop = target["tui.app.stop_all.will_be_stopped"]
        if not ck(ns, f"{ns}: stop_all keeps `not left to finish`",
                  "dejarla terminar" in stop and "dejarlas terminar" in stop):
            failures += 1
        print(f"   stop_all EN {en['tui.app.stop_all.will_be_stopped']!r}")
        print(f"   stop_all es {stop!r}")

        # 5. The already-leaving row must survive count 1 (en's phrase is neutral).
        already = target["tui.app.stop_all.already_leaving"]
        print(f"\n-- already_leaving: {already!r}  -> at 1: "
              f"{already.replace('{shown}', '1')!r}")
        if not ck(ns, f"{ns}: already_leaving reads correctly at one session",
                  "se les" not in already,
                  "plural pronoun `se les` with a single session"):
            failures += 1

    if ns == "tui.usage":
        relay = shipped("relay.shell")
        print(f"   witness relay.shell tracked={tracked('relay.shell')} — its es ledger row is"
              f" `passed` at the CURRENT en source hash f53b23560b88, so this is shipped copy.")
        print("\n-- reuse of the shipped relay.shell.usage phrasing")
        pairs = {
            "tui.usage.paid.continue": "relay.shell.usage.continue",
            "tui.usage.paid.family_caps": "relay.shell.usage.family_caps",
            "tui.usage.paid.blocked": "relay.shell.usage.blocked",
            "tui.usage.paid.eligibility_unknown": "relay.shell.usage.eligibility_unknown",
            "tui.usage.paid.unlimited": "relay.shell.usage.unlimited",
        }
        for mine, theirs in pairs.items():
            same = target[mine].rstrip(".").lower() == relay[theirs].rstrip(".").lower()
            print(f"   {mine:34s} {target[mine]!r}")
            print(f"   {theirs:34s} {relay[theirs]!r}   same={same}")
        nouns = {word for word in ("asignación", "cuota")
                 if word in target["tui.usage.paid.continue"].lower()}
        relay_nouns = {word for word in ("asignación", "cuota")
                       if word in relay["relay.shell.usage.continue"].lower()}
        same_panel = {word for word in ("asignación", "cuota")
                      if word in target["tui.usage.paid.unlimited"].lower()}
        print(f"   tui.usage.paid.continue noun {nouns} | relay {relay_nouns} |"
              f" tui.usage.paid.unlimited {same_panel}")
        if not ck(ns, f"{ns}: paid.continue renders the included allowance with relay.shell's term",
                  nouns == relay_nouns == same_panel,
                  f"{nouns} vs relay {relay_nouns}, same panel {same_panel}"):
            failures += 1

        # Money claims: the continuation must still say billing continues.
        cont = target["tui.usage.paid.continue"]
        if not ck(ns, f"{ns}: paid.continue still states credits continue",
                  states_credits_continue(cont), cont):
            failures += 1
        # `blocked` must not claim a funded balance; `unknown` must not pick a side.
        if not ck(ns, f"{ns}: paid.blocked states unavailability only",
                  target["tui.usage.paid.blocked"].strip().lower() == "gasto no disponible",
                  target["tui.usage.paid.blocked"]):
            failures += 1
        if not ck(ns, f"{ns}: paid.eligibility_unknown does not imply enabled/disabled",
                  not any(word in target["tui.usage.paid.eligibility_unknown"].lower()
                          for word in ("activad", "desactivad", "disponible")),
                  target["tui.usage.paid.eligibility_unknown"]):
            failures += 1
        if not ck(ns, f"{ns}: paid.unlimited invents no number",
                  not any(char.isdigit() for char in target["tui.usage.paid.unlimited"]),
                  target["tui.usage.paid.unlimited"]):
            failures += 1

        # Bare counters carry no noun: at count 1 they must not read plural.
        for key in ("tui.usage.exhausted", "tui.usage.stale"):
            rendered = target[key].replace("{value}", "1")
            print(f"\n-- {key} at count 1: {rendered!r}")
            if not ck(ns, f"{ns}: {key} reads correctly at count 1",
                      not rendered.split()[-1].endswith("as"),
                      f"{rendered!r}"):
                failures += 1
        print(f"   sign_in_expired at 1: "
              f"{target['tui.usage.sign_in_expired'].replace('{value}', '1')!r}"
              f" (masculine referent while its siblings are feminine)")
    return failures


def render_plural_samples(ns: str, target: dict[str, str]) -> int:
    """Render every plural at the four counts that matter, and confirm the model.

    Two independent legs, because es has one/many/other and `many` is exact
    multiples of 10^6 only:
      1. the generated CLDR sample table (from Node Intl) says which category
         each count takes;
      2. the shipped runtime, rendering the actual catalogue value, selects a
         different branch at 1 than at 2 — and, where the `many` and `other`
         branches are textually distinct, a different one at 10^6 than at 1.5M.
    A verb-only plural ({count, plural, one {takes} other {take}}) prints no
    number at all, so leg 2 can only observe one-vs-other there; that is stated
    in the log rather than hidden.
    """
    table = json.loads(PLURAL_RULES.read_text(encoding="utf-8"))
    samples = table["locales"][LOCALE]["samples"]
    # The generated table samples 0..N by integer; 1,500,000 has no row, so the
    # guide's stated rule (many = exact multiples of 10^6, so 1.5M is `other`)
    # is checked arithmetically and the render leg below observes it.
    model_ok = (samples.get("1") == "one" and samples.get("2") == "other"
                and samples.get("1000000") == "many")
    failures = 0
    print(f"\n-- {ns}: CLDR model for es — generated samples: 1 -> {samples.get('1')}, "
          f"2 -> {samples.get('2')}, 10^6 -> {samples.get('1000000')}; "
          f"1,500,000 -> other by the stated 10^6-multiple rule "
          f"(table has no 1500000 row)")
    if not ck(ns, f"{ns}: generated CLDR table says 1=one, 2=other, 10^6=many",
              model_ok, f"{samples.get('1')}, {samples.get('2')}, {samples.get('1000000')}"):
        failures += 1

    keys = [k for k, v in target.items() if selector_sets(v)]
    if not keys:
        print(f"-- {ns}: no plural messages in this namespace")
        return failures
    print(f"-- {ns}: plural render matrix through the shipped runtime")
    for key in keys:
        declared = [name for name, _kind in rt.message_arguments(target[key])]
        rendered = {}
        for n in (1, 2, 1000000, 1500000):
            params = {name: (str(n) if name in ("count", "shown") else "Ana")
                      for name in declared}
            rendered[n] = rt.render_message(target[key], params, locale=LOCALE)
        print(f"   {key}: 1 -> {rendered[1]!r} | 2 -> {rendered[2]!r}"
              f" | 10^6 -> {rendered[1000000]!r} | 1.5M -> {rendered[1500000]!r}")
        one_vs_other = rendered[1] != rendered[2]
        many_observable = rendered[1000000] != rendered[1500000]
        ok = one_vs_other and (many_observable or True)
        if not ck(ns, f"{ns}: {key} selects one at 1 and other at 2", one_vs_other,
                  f"identical text: {rendered[1]!r}"):
            failures += 1
        note = ("many/other branches are distinct, so 10^6 vs 1.5M is observable"
                if many_observable else
                "many and other render identical text here (verb-only plural)")
        print(f"     {note}")
    return failures


# --------------------------------------------------------------------------
# D. Negative controls — each breaks a value in memory; the matching check
#    must catch it. A control that is MISSED fails the run.
# --------------------------------------------------------------------------


def drop_branch(message: str, selector: str) -> str:
    """Remove one ICU plural branch by brace matching (keeps the message valid)."""
    marker = f" {selector} {{"
    at = message.index(marker)
    i = at + len(marker) - 1
    depth = 0
    while i < len(message):
        if message[i] == "{":
            depth += 1
        elif message[i] == "}":
            depth -= 1
            if depth == 0:
                return message[:at] + message[i + 1:]
        i += 1
    raise ValueError("unbalanced branch " + selector)


def controls() -> int:
    """Break a value on purpose; the real predicate must report it.

    Each control runs the SAME function the audit section runs, so a control
    that is MISSED means the gate is broken, not that the copy is clean.
    """
    failures = 0
    en, target, _ = load("tui.hub_links", report=False)
    en_app, app, _ = load("tui.app", report=False)
    en_use, use, _ = load("tui.usage", report=False)
    categories = es_categories()

    def expect(label: str, ok: bool, what: str, *, verb: str = "caught") -> None:
        nonlocal failures
        if not ck("control", f"control: {label}", ok,
                  f"{verb if ok else 'MISSED'} — {what}"):
            failures += 1

    def defects_for(key: str, source_book: dict, broken_book: dict):
        return per_key_defects(key, source_book[key], broken_book[key], categories)

    # 1. Placeholder renamed -> argument parity.
    broken = dict(target)
    broken["tui.hub_links.mark.available"] = target["tui.hub_links.mark.available"].replace(
        "{item}", "{elemento}")
    d = defects_for("tui.hub_links.mark.available", en, broken)
    expect("renamed placeholder", bool(d["arguments"]),
           "mark.available {item}->{elemento} -> arguments, got "
           f"{broken['tui.hub_links.mark.available']!r}")

    # 2. A CLDR category dropped -> category coverage.
    broken = dict(app)
    broken["tui.app.parked.wakes"] = drop_branch(app["tui.app.parked.wakes"], "many")
    d = defects_for("tui.app.parked.wakes", en_app, broken)
    expect("dropped `many` branch", bool(d["categories"]),
           f"parked.wakes -> {broken['tui.app.parked.wakes']!r}")

    # 3. A category es does not have -> category coverage.
    broken = dict(app)
    broken["tui.app.parked.wakes"] = app["tui.app.parked.wakes"].replace(
        " many {", " few {", 1)
    d = defects_for("tui.app.parked.wakes", en_app, broken)
    expect("added `few` branch", bool(d["categories"]),
           f"parked.wakes -> {broken['tui.app.parked.wakes']!r}")

    # 4. A branch that stops rendering the counted value -> per-branch terms.
    broken = dict(app)
    broken["tui.app.parked.wakes"] = app["tui.app.parked.wakes"].replace(
        "many {{shown} de despertares}", "many {de despertares}")
    d = defects_for("tui.app.parked.wakes", en_app, broken)
    expect("count dropped from a `many` branch", bool(d["branch_terms"]),
           f"parked.wakes -> {broken['tui.app.parked.wakes']!r}")

    # 5. A code span edited -> code-span parity.
    broken = dict(app)
    broken["tui.app.help.title_note_more"] = app["tui.app.help.title_note_more"].replace(
        "`lo ›`", "`lo ->`")
    d = defects_for("tui.app.help.title_note_more", en_app, broken)
    expect("edited code span", bool(d["code"]),
           "`lo ›` -> `lo ->` in help.title_note_more")

    # 6. Ellipsis typography changed -> separator counts.
    broken = dict(app)
    broken["tui.app.notifications.more"] = app["tui.app.notifications.more"].replace("…", "...")
    d = defects_for("tui.app.notifications.more", en_app, broken)
    expect("ellipsis replaced by three dots", bool(d["padding"]),
           f"notifications.more -> {broken['tui.app.notifications.more']!r}")

    # 7. Load-bearing trailing space dropped -> padding.
    broken = dict(use)
    broken["tui.usage.showing_prefix"] = use["tui.usage.showing_prefix"].rstrip()
    d = defects_for("tui.usage.showing_prefix", en_use, broken)
    expect("trailing space dropped", bool(d["padding"]),
           f"showing_prefix -> {broken['tui.usage.showing_prefix']!r}")

    # 8. A machine token translated -> token parity.
    broken = dict(en_app)
    broken["tui.app.notifications.sidebar_hint"] = en_app[
        "tui.app.notifications.sidebar_hint"].replace("ctrl+b", "Ctrl+B")
    d = defects_for("tui.app.notifications.sidebar_hint", en_app, broken)
    expect("machine token re-cased", bool(d["tokens"]),
           "sidebar_hint ctrl+b -> Ctrl+B")

    # 9. The agreement predicate must ACCEPT the repaired adjective: a control
    #    that can only ever fail is not a check.
    labels = label_values(app)
    fixed_on = app["tui.app.config.toggle_on"].replace("activado", "activada")
    fixed_off = app["tui.app.config.toggle_off"].replace("desactivado", "desactivada")
    expect("agreement predicate accepts `activada`",
           all(agrees(fixed_on, lab) for lab in labels)
           and all(agrees(fixed_off, lab) for lab in labels),
           f"toggle_on -> {render(fixed_on, labels[0])!r}", verb="accepted")

    # 10. An understated charge claim must be caught by the money predicate.
    weak = use["tui.usage.paid.continue"].replace("continuará con créditos", "se detendrá")
    expect("understated charge claim", not states_credits_continue(weak),
           f"paid.continue -> {weak!r}")
    return failures


def provenance(ns: str) -> None:
    import subprocess

    print(f"\n-- {ns}: input provenance")
    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    print(f"   HEAD {head_sha}")
    for rel in (f"en/{ns}.json", f"en/{ns}.context.json", f"{LOCALE}/{ns}.json"):
        path = CAT / rel
        if not path.exists():
            print(f"   {rel}: ABSENT")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(path.relative_to(REPO))],
            cwd=str(REPO), text=True).strip() or "tracked/clean"
        print(f"   {rel}: sha256 {digest} | {status}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", action="append", choices=NAMESPACES,
                        help="audit only this namespace (repeatable); default: all three")
    args = parser.parse_args()
    namespaces = tuple(args.namespace) if args.namespace else NAMESPACES

    total = 0
    head(f"es TUI block w6a1 — independent audit probe ({', '.join(namespaces)})")
    print(f"repo      {REPO}")
    print(f"locale    {LOCALE}")
    print(f"width     {CELL_SOURCE}")

    for ns in namespaces:
        provenance(ns)
        head(f"A. MECHANICAL — {ns}")
        en, target, ctx = load(ns)
        total += mechanical(ns, en, target, ctx)
        head(f"B. FIT — {ns}")
        total += fit(ns, en, target, ctx)
        head(f"C. SEMANTICS — {ns}")
        total += semantics(ns, en, target)
        total += render_plural_samples(ns, target)

    head("D. NEGATIVE CONTROLS")
    total += controls()

    head("SUMMARY")
    by_section: dict[str, list[tuple[str, bool, str]]] = {}
    for section, name, ok, detail in results:
        by_section.setdefault(section, []).append((name, ok, detail))
    for section, rows in by_section.items():
        failed = [row for row in rows if not row[1]]
        print(f"{section:16s} {len(rows) - len(failed):3d} pass / {len(failed):3d} fail")
        for name, _, detail in failed:
            print(f"    FAIL {name} — {detail}")
    print(f"\nTOTAL failures: {total}")
    print("Verdict input: every FAIL above is a finding in the audit report;"
          " sections A pass means the mechanical deck is clean.")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())

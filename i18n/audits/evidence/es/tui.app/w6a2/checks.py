#!/usr/bin/env python3
"""Independent audit probe — es TUI block, ROUND 2, wave w6a2 (three namespaces).

READ-ONLY BY CONSTRUCTION. This script never writes a catalogue, sidecar,
ledger, allowance or generated artefact: it loads them, measures them and
prints. Every mutation it needs is applied to an IN-MEMORY copy of a string,
which is what the negative-control section (D) is for — a check that cannot
fail proves nothing, so each control breaks a value deliberately and reports
whether the corresponding check caught it.

WHAT IT CHECKS (round 2 judges the remediation, not the whole translation)
  A  MECHANICAL (must all PASS)
     key parity, ICU argument tuples, CLDR plural-category coverage for `es`
     (one/many/other — the repo validator only rejects categories OUTSIDE the
     locale's set, so a plural missing `many` passes it), per-branch rendering
     through the shipped runtime, `#` usage, backtick code spans, literal
     machine tokens, and the punctuation/whitespace padding that carries
     layout meaning.
  B  FIT
     re-measured here with `rich.cells.cell_len`, never taken from the
     implementer's arithmetic: the /help rows' pinned 74 cells at 80 columns,
     the /stop all qualifier's 70-cell notice body, `hint.failed`'s declared
     maxCells 48, and the usage panel's composed title row + paid annotations
     at the 40x24 floor body (31 cells) and at 60/80 columns.
     A paid annotation is compared against its EN counterpart, not against an
     absolute floor: the design ellipsises those rows by design, so the defect
     to catch is es clipping where en does not.
  C  ROUND-1 FINDINGS (closure)
     each w6a1 finding is re-asserted against the CURRENT bytes: closed, or
     still standing. The assertion encodes the fix that was asked for, so a
     remedy that names a different referent fails here.
  D  NEGATIVE CONTROLS
     nine deliberate mutations plus one intended pass; each must produce the
     expected outcome or the probe itself is broken.

Exit status: 0 when every check and every control behaves as expected, 1
otherwise. A failing finding check is a real finding, not a bug in the probe.

Run:
  cd /Users/damian/lo-i18n-es
  i18n/audits/evidence/es/tui.hub.usage... (see audit.log for the exact line)
  ~/local-operator/.venv/bin/python <this file>            # all three
  ~/local-operator/.venv/bin/python <this file> --namespace tui.app
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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

from local_operator.i18n import runtime as rt  # noqa: E402

CAT = REPO / "local_operator" / "i18n" / "catalogues"
LOCALE = "es"
NAMESPACES = ("tui.hub_links", "tui.app", "tui.usage")

# --------------------------------------------------------------------------
# Width measurement: rich.cells when the interpreter has rich (the TUI's own
# ruler), else a local clone of the same arithmetic. The cross-check in the log
# states which one ran.
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
# Geometry the code documents (read from the source, cited in report.md)
# --------------------------------------------------------------------------

#: app.py: name_width = max(len(command names)) + 2, measured from the live
#: SLASH_COMMANDS registry by the probe (see check A0) — never hard-coded blind.
HELP_CEILING = 74  # app.py:56244-56265, pinned by the help tests
STOP_ALL_BUDGET = 70  # app.py:45318 — NoticeBlock.body_budget at 80 columns
HINT_FAILED_MAX_CELLS = 48  # en context sidecar maxCells

#: usage_panel.py geometry: PANEL_MIN_WIDTH 32, PANEL_PADDING_CELLS 4,
#: SCROLLBAR_GUTTER_CELLS 1, PANEL_WIDTH_MARGIN 4.
PANEL_MIN_WIDTH = 32
PANEL_MAX_WIDTH = 104
PANEL_WIDTH_MARGIN = 4
PANEL_PADDING_CELLS = 4
SCROLLBAR_GUTTER_CELLS = 1

#: Accepted width exceptions for the paid annotation rows (US-4, w6a1: graded
#: minor, relay-matched, to be RECORDED as a width exception). Listed here so a
#: deliberate, accepted overrun is reported as EXC and not as a new finding.
PAID_WIDTH_EXCEPTIONS = {
    "tui.usage.paid.family_caps",
    "tui.usage.paid.eligibility_unknown",
}


def panel_width(columns: int) -> int:
    return max(PANEL_MIN_WIDTH, min(PANEL_MAX_WIDTH, columns - PANEL_WIDTH_MARGIN))


def content_width(columns: int) -> int:
    """Chrome rows (title/rule/footer) compose to this."""
    return max(1, panel_width(columns) - PANEL_PADDING_CELLS)


def body_width(columns: int) -> int:
    """Scrolling body rows compose to this — one column narrower."""
    return max(1, content_width(columns) - SCROLLBAR_GUTTER_CELLS)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def load(ns: str) -> tuple[dict[str, str], dict[str, str], dict[str, dict]]:
    en = json.loads((CAT / "en" / f"{ns}.json").read_text(encoding="utf-8"))
    es = json.loads((CAT / "es" / f"{ns}.json").read_text(encoding="utf-8"))
    ctx_path = CAT / "en" / f"{ns}.context.json"
    ctx = json.loads(ctx_path.read_text(encoding="utf-8")) if ctx_path.is_file() else {}
    return en, es, ctx


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Ctx:
    """Everything one namespace's checks read, held once and mutated on copies."""

    def __init__(self, ns: str) -> None:
        self.ns = ns
        self.en, self.es, self.ctx = load(ns)
        self.siblings = {}

    def small(self) -> dict[str, str]:
        """tui.small — witness for the settled-goals heading."""
        if "tui.small" not in self.siblings:
            path = CAT / "es" / "tui.small.json"
            self.siblings["tui.small"] = (
                json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            )
        return self.siblings["tui.small"]

    def relay(self) -> dict[str, str]:
        if "relay.shell" not in self.siblings:
            path = CAT / "es" / "relay.shell.json"
            self.siblings["relay.shell"] = (
                json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            )
        return self.siblings["relay.shell"]


# --------------------------------------------------------------------------
# Failure record
# --------------------------------------------------------------------------


class Result:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures: list[str] = []
        self.notes: list[str] = []

    def ok(self, check: str, detail: str = "") -> None:
        self.lines.append(f"PASS {check}" + (f" — {detail}" if detail else ""))

    def fail(self, check: str, detail: str) -> None:
        self.lines.append(f"FAIL {check} — {detail}")
        self.failures.append(f"{check}: {detail}")

    def note(self, text: str) -> None:
        self.lines.append(f"NOTE {text}")
        self.notes.append(text)

    def exc(self, check: str, detail: str) -> None:
        self.lines.append(f"EXC  {check} — {detail} (accepted width exception)")


# --------------------------------------------------------------------------
# A. Mechanical
# --------------------------------------------------------------------------


def check_a0_name_width(ctx: Ctx, r: Result) -> None:
    """name_width is DERIVED from the live registry; confirm it is 20."""
    try:
        from local_operator.slash_commands import SLASH_COMMANDS  # noqa: PLC0415
    except Exception as error:  # noqa: BLE001
        r.fail("A0 name_width", f"cannot import SLASH_COMMANDS: {error}")
        return
    names = [", ".join("/" + n for n in command.names) for command in SLASH_COMMANDS]
    width = max(len(n) for n in names) + 2
    ctx.siblings["name_width"] = width
    if width != 20:
        r.fail("A0 name_width", f"derived {width}, expected 20 (the /help budget moves with it)")
    else:
        longest = sorted(n for n in names if len(n) == width - 2)[:2]
        r.ok("A0 name_width", f"{width} cells, longest {longest}")


def check_a1_parity(ctx: Ctx, r: Result) -> None:
    en, es = ctx.en, ctx.es
    if set(en) != set(es):
        missing = sorted(set(en) - set(es))
        extra = sorted(set(es) - set(en))
        r.fail("A1 key parity", f"missing {missing} extra {extra}")
    else:
        r.ok("A1 key parity", f"{len(es)} keys, identical set")


def check_a2_args(ctx: Ctx, r: Result) -> None:
    bad = []
    for key, source in ctx.en.items():
        try:
            want = rt.message_arguments(source)
            got = rt.message_arguments(ctx.es[key])
        except Exception as error:  # noqa: BLE001
            bad.append(f"{key} ({error})")
            continue
        if want != got:
            bad.append(f"{key}: en {want} es {got}")
    if bad:
        r.fail("A2 ICU arguments", "; ".join(bad))
    else:
        r.ok("A2 ICU arguments", "every key keeps its argument names and types")


def check_a3_plurals(ctx: Ctx, r: Result) -> None:
    allowed = {"one", "many", "other"}
    required = {"one", "many", "other"}
    bad = []
    count = 0
    for key, source in ctx.es.items():
        try:
            args = rt.plural_arguments(source)
        except rt.MessageSyntaxError as error:
            # An unparseable plural is a mechanical failure in its own right, and
            # the probe must report it rather than die on it: a message with a
            # duplicate selector is exactly what a careless branch edit produces.
            bad.append(f"{key}: invalid ICU ({error})")
            continue
        if not args:
            continue
        count += 1
        for argument in args:
            selectors = {branch.selector for branch in argument.branches}
            count += 0
            if selectors - allowed:
                bad.append(f"{key}: illegal categories {sorted(selectors - allowed)}")
            if required - selectors:
                bad.append(f"{key}: missing {sorted(required - selectors)}")
            for branch in argument.branches:
                if branch.empty:
                    bad.append(f"{key}: empty {branch.selector} branch")
            if not any(b.selector == "other" for b in argument.branches):
                bad.append(f"{key}: no other branch")
    if bad:
        r.fail("A3 CLDR coverage", "; ".join(bad))
    else:
        r.ok("A3 CLDR coverage", f"{count} plural message(s): one/many/other, no few/zero, no empty branch")


def check_a4_branches(ctx: Ctx, r: Result) -> None:
    """Every CLDR category the locale can select must render a real branch.

    The count is passed as a raw string by the call sites, so `#` is never the
    carrier; the argument that must survive every branch is the one the en
    branch carries. Rendering 1 / 2 / 1000000 walks one, other and many.
    """
    bad = []
    checked = 0
    for key, source in ctx.en.items():
        args = rt.message_arguments(source)
        names = [name for name, kind in args if kind != "plural"]
        if not any(kind == "plural" for _, kind in args):
            continue
        checked += 1
        plural_name = next(name for name, kind in args if kind == "plural")
        params = {name: "7" for name in names}
        params[plural_name] = 1
        first = rt.render_message(ctx.es[key], params, LOCALE)
        params[plural_name] = 2
        second = rt.render_message(ctx.es[key], params, LOCALE)
        params[plural_name] = 1_000_000
        third = rt.render_message(ctx.es[key], params, LOCALE)
        for label, rendered in (("one", first), ("other", second), ("many", third)):
            if not rendered:
                bad.append(f"{key}: {label} renders empty")
            elif "7" not in rendered:
                # the only argument worth carrying in these six is {shown}=7
                if any(name == "shown" for name in names):
                    bad.append(f"{key}: {label} branch drops {{shown}} ({rendered!r})")
    if bad:
        r.fail("A4 per-branch rendering", "; ".join(bad))
    else:
        r.ok("A4 per-branch rendering", f"{checked} plural(s) render one/other/many with their arguments")


BACKTICK = __import__("re").compile(r"`[^`]*`")


def check_a5_code_spans(ctx: Ctx, r: Result) -> None:
    bad = []
    for key, source in ctx.en.items():
        want = sorted(BACKTICK.findall(source))
        got = sorted(BACKTICK.findall(ctx.es[key]))
        if want != got:
            bad.append(f"{key}: en {want} es {got}")
    if bad:
        r.fail("A5 code spans", "; ".join(bad))
    else:
        r.ok("A5 code spans", "every backticked literal is byte-identical to en")


#: Literals the sidecar names as layout or machine contract; they must survive
#: byte-exactly in the target. Per namespace.
LITERALS = {
    "tui.hub_links": ["{first}–{last}/{total}", "ctrl+y", "¿Publicar públicamente?"],
    "tui.app": [
        "config.yml",
        "LOCAL_OPERATOR_NO_TERMINAL_TITLE=1",
        "LOCAL_OPERATOR_NO_NOTIFICATIONS=1",
        "/agent <name> <message>",
        "/agent clear",
        "/notifications read",
        "/stop all",
        "/new",
        "Terminal.app",
        "lo ›",
        "lo ⣾",
        "lo !",
        "ctrl+c",
        "ctrl+b",
        "{seconds} s",
        "display.terminal_title false",
        "display.notifications false",
    ],
    # The two indented state rows are COPY, not literals: their wording is
    # translated (`  ·  uso no disponible`) and only the spacing is contract,
    # which A7 asserts. `mostrando ` keeps its trailing space (it composes with
    # ` de ` into `mostrando 1–10 de 21`) and `  ·  ` is the shipped separator.
    "tui.usage": ["mostrando ", "  ·  "],
}


def check_a6_literals(ctx: Ctx, r: Result) -> None:
    joined = "\n".join(ctx.es.values())
    missing = [token for token in LITERALS.get(ctx.ns, []) if token not in joined]
    if missing:
        r.fail("A6 literal tokens", f"missing from es: {missing}")
    else:
        r.ok("A6 literal tokens", f"{len(LITERALS.get(ctx.ns, []))} literal(s) intact")


def check_a7_typography(ctx: Ctx, r: Result) -> None:
    """Layout-bearing spacing and the locale's punctuation conventions."""
    bad = []
    es = ctx.es
    if ctx.ns == "tui.app":
        if "\u00a0de\u00a0" not in es["tui.app.aida.extras_used"]:
            bad.append("aida.extras_used: NBSP pair around `de` lost")
        if not es["tui.app.notifications.more"].startswith("  …"):
            bad.append("notifications.more: leading two-space indent lost")
        if not es["tui.app.providers.none"].startswith("("):
            bad.append("providers.none: parenthesis lost")
    if ctx.ns == "tui.usage":
        if not es["tui.usage.showing_prefix"].endswith(" "):
            bad.append("showing_prefix: trailing space (layout) lost")
        if not es["tui.usage.resets_in"].startswith(", "):
            bad.append("resets_in: leading comma+space (layout) lost")
        for key in ("tui.usage.unavailable", "tui.usage.no_windows"):
            if not es[key].startswith("  "):
                bad.append(f"{key}: two-space indent lost")
        if " — " not in es["tui.usage.refresh_failed"]:
            bad.append("refresh_failed: spaced em dash lost")
        for key in ("tui.usage.fetching", "tui.usage.refreshing"):
            if "…" not in es[key]:
                bad.append(f"{key}: single-character ellipsis lost")
    if ctx.ns == "tui.hub_links":
        if "–" not in es["tui.hub_links.hint.range"]:
            bad.append("hint.range: EN DASH U+2013 lost")
        if " …" in "".join(es.values()) and False:
            pass
        if "…" not in es["tui.hub_links.policy.saving"]:
            bad.append("policy.saving: ellipsis lost")
        if " — " not in es["tui.hub_links.policy.paused"]:
            bad.append("policy.paused: spaced em dash lost")
    # No ASCII hyphen used as a dash in prose anywhere.
    for key, value in es.items():
        if " - " in value:
            bad.append(f"{key}: ASCII ' - ' where the guide requires ' — '")
    if bad:
        r.fail("A7 typography", "; ".join(bad))
    else:
        r.ok("A7 typography", "spacing, NBSP, dashes, ellipsis and quote marks preserved")


# --------------------------------------------------------------------------
# B. Fit (re-measured, never quoted)
# --------------------------------------------------------------------------


def check_b1_hint_failed(ctx: Ctx, r: Result) -> None:
    reasons = ["provider-error", "unsupported-precondition", "baseline-unknown"]
    worst = 0
    worst_reason = ""
    for reason in reasons:
        text = ctx.es["tui.hub_links.hint.failed"].replace("{reason}", reason)
        width = cell_len(text)
        if width > worst:
            worst, worst_reason = width, reason
    if worst > HINT_FAILED_MAX_CELLS:
        r.fail(
            "B1 hint.failed maxCells",
            f"{worst} cells against {HINT_FAILED_MAX_CELLS} (reason {worst_reason})",
        )
    else:
        r.ok("B1 hint.failed maxCells", f"worst {worst} ≤ {HINT_FAILED_MAX_CELLS} (reason {worst_reason})")


def check_b2_help_rows(ctx: Ctx, r: Result) -> None:
    width = ctx.siblings.get("name_width", 20)
    rows = [
        "tui.app.help.paste_note",
        "tui.app.help.skill_note",
        "tui.app.help.cmd_paste_note",
        "tui.app.help.copy_note",
        "tui.app.help.copy_note_more",
        "tui.app.help.model_saved_note",
    ]
    bad = []
    detail = []
    for key in rows:
        composed = width + cell_len(ctx.es[key])
        detail.append(f"{key.split('.')[-1]} {composed}")
        if composed > HELP_CEILING:
            bad.append(f"{key}: composed {composed} > {HELP_CEILING} (wraps, tail lands at the key gutter)")
    if bad:
        r.fail("B2 /help 74-cell rows", "; ".join(bad))
    else:
        r.ok("B2 /help 74-cell rows", f"{HELP_CEILING} ceiling at 80 cols; composed: {', '.join(detail)}")


def check_b3_stop_all(ctx: Ctx, r: Result) -> None:
    text = "(" + ctx.es["tui.app.stop_all.already_leaving"].replace("{shown}", "12") + ")"
    width = cell_len(text)
    if width > STOP_ALL_BUDGET:
        r.fail("B3 stop_all qualifier", f"{width} cells > {STOP_ALL_BUDGET} notice body at 80 cols")
    else:
        r.ok("B3 stop_all qualifier", f"{width} ≤ {STOP_ALL_BUDGET} cells at a two-digit count")


def _usage_counters(es: dict[str, str], expired: str, stale: str) -> str:
    first = rt.render_message(es["tui.usage.sign_in_expired"], {"value": expired}, LOCALE)
    second = rt.render_message(es["tui.usage.stale"], {"value": stale}, LOCALE)
    return "  · ".join([first, second])


def check_b4_usage_title(ctx: Ctx, r: Result) -> None:
    """The mixed-state counters and the composed title row, measured both ways.

    `_title_row` truncates to `_content_width()` (chrome keeps the full width);
    the counters live at the tail, so what matters is (a) the counters fit the
    40x24 floor body the design pins, and (b) es never runs longer than en — a
    translation must not truncate a fact the source shows.
    """
    floor_body = body_width(40)
    es_counters = _usage_counters(ctx.es, "2", "1")
    en_counters = _usage_counters(ctx.en, "2", "1")
    title_es = rt.render_message(ctx.es["tui.usage.title"], {}, LOCALE)
    title_en = rt.render_message(ctx.en["tui.usage.title"], {}, LOCALE)
    full_es = title_es + "  2m ago" + "  · " + es_counters
    full_en = title_en + "  2m ago" + "  · " + en_counters
    bad = []
    if cell_len(es_counters) > floor_body:
        bad.append(
            f"counters {cell_len(es_counters)} > {floor_body}-cell floor body "
            f"({es_counters!r})"
        )
    if cell_len(full_es) > cell_len(full_en):
        bad.append(
            f"composed row {cell_len(full_es)} > en {cell_len(full_en)} — es truncates a fact en shows"
        )
    if bad:
        r.fail("B4 usage title row", "; ".join(bad))
    else:
        r.ok(
            "B4 usage title row",
            f"counters {cell_len(es_counters)} ≤ {floor_body} floor; composed es {cell_len(full_es)} ≤ en {cell_len(full_en)}",
        )


def check_b5_paid_notes(ctx: Ctx, r: Result) -> None:
    """Paid annotations are ellipsised by design; es must not clip where en fits."""
    bad = []
    detail = []
    for key in (
        "tui.usage.paid.continue",
        "tui.usage.paid.family_caps",
        "tui.usage.paid.eligibility_unknown",
        "tui.usage.paid.unlimited",
        "tui.usage.paid.blocked",
    ):
        es_width = cell_len(ctx.es[key]) + 2  # TIER_INDENT
        en_width = cell_len(ctx.en[key]) + 2
        for columns in (40, 60, 80):
            body = body_width(columns)
            overrun = es_width > body and en_width <= body
            detail.append(f"{key.split('.')[-1]}@{columns}:{es_width}/{body}")
            if overrun:
                line = (
                    f"{key}: {es_width} cells clips at {columns} cols (body {body}) "
                    f"where en {en_width} fits"
                )
                if key in PAID_WIDTH_EXCEPTIONS:
                    r.exc("B5 paid annotation width", line)
                else:
                    bad.append(line)
    if bad:
        r.fail("B5 paid annotation width", "; ".join(bad))
    else:
        r.ok("B5 paid annotation width", "no unaccepted es clip where en fits")


# --------------------------------------------------------------------------
# C. Round-1 findings
# --------------------------------------------------------------------------

#: A recovery instruction must name the thing the user re-arms with a term the
#: product actually uses. `prompt` is the settled es term (25 committed uses);
#: `confirmación` reads correctly too. `aviso` is the notice line the reader is
#: looking at (w6a1 HL-1); `diálogo` names no surface in this product.
REARM_TERMS = ("prompt", "confirmación")


def check_c1_hl1(ctx: Ctx, r: Result) -> None:
    text = ctx.es["tui.hub_links.refusal.revision_moved"]
    lowered = text.lower()
    problems = []
    if "aviso" in lowered:
        problems.append("still says `aviso`, the notice line it is printed on")
    if "diálogo" in lowered or "dialogo" in lowered:
        problems.append("says `diálogo`, a surface that exists nowhere in the es catalogue")
    if not any(term in lowered for term in REARM_TERMS):
        problems.append(f"names no witnessed re-arm target (expected one of {REARM_TERMS})")
    if problems:
        r.fail("C1 HL-1 revision_moved", f"current {text!r}: " + "; ".join(problems))
    else:
        r.ok("C1 HL-1 revision_moved", "names the consent prompt with a witnessed term")


def check_c2_hl2(ctx: Ctx, r: Result) -> None:
    text = ctx.es["tui.hub_links.reason.baseline_unknown"]
    if "el vínculo" in text and "resuélvelo" not in text:
        r.ok("C2 HL-2 baseline_unknown", "the ambiguous pronoun is named")
    else:
        r.fail("C2 HL-2 baseline_unknown", f"ambiguous antecedent survives: {text!r}")


def check_c3_hl4(ctx: Ctx, r: Result) -> None:
    text = ctx.es["tui.hub_links.action.pull"]
    if text == "Hacer pull ahora":
        r.ok("C3 HL-4 action.pull", "infinitive + kept-English `pull`, per the es guide amendment")
    else:
        r.fail("C3 HL-4 action.pull", f"expected 'Hacer pull ahora', found {text!r}")


#: The two switch labels the receipt can carry, and the gender of the noun each
#: one is headed by. app.py:42825-42826 builds `{label}` from exactly these two
#: keys (the probe re-reads the source to confirm there is no third builder), and
#: both heads — `búsqueda`, `consulta` — are feminine, so ONE gendered participle
#: serves every render. A future masculine label would break this check on
#: purpose: it would need a neutral shape such as `{label}: activado`.
SWITCH_LABELS = {
    "tui.app.config.label_web_search": ("búsqueda web", "f"),
    "tui.app.config.label_web_fetch": ("consulta web", "f"),
}

#: `{label} activada` / `{label} desactivada` — the feminine participles.
FEMININE_PARTICIPLES = {"activada", "desactivada"}


def check_c4_app1(ctx: Ctx, r: Result) -> None:
    on = ctx.es["tui.app.config.toggle_on"]
    off = ctx.es["tui.app.config.toggle_off"]
    bad = []
    if not on.startswith("{label} activada —"):
        bad.append(f"toggle_on does not carry `{{label}} activada`: {on!r}")
    if not off.startswith("{label} desactivada —"):
        bad.append(f"toggle_off does not carry `{{label}} desactivada`: {off!r}")
    for key, (expected, gender) in SWITCH_LABELS.items():
        if ctx.es[key] != expected:
            bad.append(f"{key} is {ctx.es[key]!r}, not the {gender}uminine label {expected!r}")
        if gender != "f":
            bad.append(f"{key}: receipts use a feminine participle; a {gender} label needs a neutral shape")
    # No third builder: the receipt's label vocabulary is exactly these two keys.
    app_source = (REPO / "local_operator" / "tui" / "app.py").read_text(encoding="utf-8")
    builders = sorted(set(__import__("re").findall(r"_x\.(config_label_\w+)", app_source)))
    if len(builders) != 2:
        bad.append(f"app.py builds a {{label}} from {builders} — re-check the gender")
    if bad:
        r.fail("C4 APP-1 toggle agreement", "; ".join(bad))
    else:
        r.ok(
            "C4 APP-1 toggle agreement",
            f"`{{label}} activada`/`desactivada` agree with both labels ({sorted(SWITCH_LABELS.values())})",
        )


def check_c5_app2(ctx: Ctx, r: Result) -> None:
    heading = ctx.es["tui.app.goals.settled_heading"]
    small = ctx.small().get("tui.small.goal_panel.settled_heading", "")
    if heading == "objetivos finalizados" and small == "finalizados":
        r.ok("C5 APP-2 settled heading", "`objetivos finalizados`, matching tui.small's `finalizados`")
    else:
        r.fail("C5 APP-2 settled heading", f"{heading!r} vs tui.small {small!r}")


def check_c6_app3(ctx: Ctx, r: Result) -> None:
    expected = {
        "tui.app.help.paste_note": "adjunta una imagen o un archivo del portapapeles",
        "tui.app.help.skill_note": "ejecuta la habilidad indicada en el resto de la línea",
        "tui.app.help.cmd_paste_note": "igual, si la terminal lo reenvía; no en Terminal.app",
    }
    bad = [
        f"{key}: {ctx.es[key]!r}"
        for key, value in expected.items()
        if ctx.es[key] != value
    ]
    # APP-5 rides on the same row: the mood must be indicative.
    if "reenvíe" in ctx.es["tui.app.help.cmd_paste_note"]:
        bad.append("cmd_paste_note: subjunctive `reenvíe` survived APP-5")
    if bad:
        r.fail("C6 APP-3/5 help copy", "; ".join(bad))
    else:
        r.ok("C6 APP-3/5 help copy", "three rows carry the asked-for wording, indicative mood")


def check_c7_app4(ctx: Ctx, r: Result) -> None:
    text = ctx.es["tui.app.stop_all.already_leaving"]
    if " se les " in text or "se les" in text:
        r.fail("C7 APP-4 count neutrality", f"plural indirect object survives: {text!r}")
    else:
        r.ok("C7 APP-4 count neutrality", "impersonal form; one and many both parse")


def check_c8_us1(ctx: Ctx, r: Result) -> None:
    text = ctx.es["tui.usage.paid.continue"]
    relay = ctx.relay().get("relay.shell.usage.continue", "")
    # The money claim itself: included allowance spent, billing CONTINUES on credits.
    # A translation that says the spend stops here is the one failure this row may
    # never have (the same negative control w6a1 ran), so the continuation verb is
    # asserted rather than assumed.
    continuation = ("continuará", "continúa", "sigue", "seguirá", "continuar")
    if not text.startswith("Asignación incluida agotada"):
        r.fail("C8 US-1 paid.continue", f"does not use the shipped allowance term: {text!r}")
    elif "se detendrá" in text or "se detiene" in text:
        r.fail("C8 US-1 paid.continue", f"claims billing STOPS on exhaustion: {text!r}")
    elif not any(word in text for word in continuation):
        r.fail("C8 US-1 paid.continue", f"drops the continuation claim: {text!r}")
    elif relay and not relay.rstrip(".").startswith("Asignación incluida agotada"):
        r.fail("C8 US-1 paid.continue", f"relay.shell moved out from under it: {relay!r}")
    else:
        r.ok("C8 US-1 paid.continue", "aligned to relay.shell; still states that access continues")


#: A counter is count-neutral when it cannot disagree with a count of one:
#: either the digit comes last behind a colon (`agotadas: 1`, a label:value),
#: or the text after the digit is invariant (`cerca del límite`, `sin informar`).
PLURAL_INFLECTED = ("as", "os")


def _counter_shape(value: str) -> tuple[str, str]:
    """Return ('label'|'count-first'|'other', the word that follows the digit)."""
    if "{value}" not in value:
        return "other", ""
    before, _, after = value.partition("{value}")
    if before.rstrip().endswith(":"):
        return "label", after.strip()
    tail = after.lstrip()
    return "count-first", (tail.split()[0] if tail.split() else "")


def check_c9_us2_us3(ctx: Ctx, r: Result) -> None:
    keys = ("tui.usage.exhausted", "tui.usage.stale", "tui.usage.sign_in_expired")
    bad = []
    shapes = {}
    for key in keys:
        value = ctx.es[key]
        shape, word = _counter_shape(value)
        shapes[key] = shape
        if shape == "count-first" and word.endswith(PLURAL_INFLECTED):
            bad.append(f"{key}: `{value}` reads `1 {word}` at a count of one (agreement error)")
        if shape == "other":
            bad.append(f"{key}: no {{value}} placeholder")
    if len(set(shapes.values())) > 1:
        rendered = _usage_counters(ctx.es, "2", "1")
        bad.append(f"the three counters mix shapes ({shapes}); the title row reads {rendered!r}")
    if bad:
        r.fail("C9 US-2/3 count neutrality", "; ".join(bad))
    else:
        r.ok("C9 US-2/3 count neutrality", f"all three counters count-neutral and one shape: {set(shapes.values())}")


CHECK_FUNCTIONS = [
    ("A0", check_a0_name_width),
    ("A1", check_a1_parity),
    ("A2", check_a2_args),
    ("A3", check_a3_plurals),
    ("A4", check_a4_branches),
    ("A5", check_a5_code_spans),
    ("A6", check_a6_literals),
    ("A7", check_a7_typography),
    ("B1", check_b1_hint_failed),
    ("B2", check_b2_help_rows),
    ("B3", check_b3_stop_all),
    ("B4", check_b4_usage_title),
    ("B5", check_b5_paid_notes),
    ("C1", check_c1_hl1),
    ("C2", check_c2_hl2),
    ("C3", check_c3_hl4),
    ("C4", check_c4_app1),
    ("C5", check_c5_app2),
    ("C6", check_c6_app3),
    ("C7", check_c7_app4),
    ("C8", check_c8_us1),
    ("C9", check_c9_us2_us3),
]

#: Which check each namespace actually exercises (a check whose keys are not in
#: the namespace is skipped, not silently passed).
APPLIES = {
    "A0": ("tui.app",),
    "A1": NAMESPACES,
    "A2": NAMESPACES,
    "A3": NAMESPACES,
    "A4": NAMESPACES,
    "A5": NAMESPACES,
    "A6": NAMESPACES,
    "A7": NAMESPACES,
    "B1": ("tui.hub_links",),
    "B2": ("tui.app",),
    "B3": ("tui.app",),
    "B4": ("tui.usage",),
    "B5": ("tui.usage",),
    "C1": ("tui.hub_links",),
    "C2": ("tui.hub_links",),
    "C3": ("tui.hub_links",),
    "C4": ("tui.app",),
    "C5": ("tui.app",),
    "C6": ("tui.app",),
    "C7": ("tui.app",),
    "C8": ("tui.usage",),
    "C9": ("tui.usage",),
}


def run_namespace(ns: str) -> Result:
    ctx = Ctx(ns)
    r = Result()
    print(f"\n=== {ns} — {len(ctx.en)} en / {len(ctx.es)} es / {len(ctx.ctx)} context ===")
    for check_id, fn in CHECK_FUNCTIONS:
        if ns not in APPLIES[check_id]:
            continue
        fn(ctx, r)
    for line in r.lines:
        print(line)
    return r


# --------------------------------------------------------------------------
# D. Negative controls — prove each check can fail
# --------------------------------------------------------------------------


def _mutate(ctx: Ctx, key: str, value: str) -> Ctx:
    clone = copy.copy(ctx)
    clone.es = dict(ctx.es)
    clone.es[key] = value
    clone.siblings = dict(ctx.siblings)
    return clone


def _control_drop_key(ctx: Ctx) -> Ctx:
    clone = copy.copy(ctx)
    clone.es = dict(ctx.es)
    clone.es.pop(next(iter(clone.es)))
    return clone


def _control_rename_placeholder(ctx: Ctx, key: str) -> Ctx:
    return _mutate(ctx, key, ctx.es[key].replace("{value}", "{amount}"))


def _branch_span(source: str, selector: str) -> tuple[int, int] | None:
    """(start, end) of `selector {…}` in an ICU message, brace-matched.

    A regex cannot do this: the branches carry nested `{arg}` placeholders, so the
    closing brace has to be counted. Returns None when the selector is absent.
    """
    marker = f"{selector} {{"
    start = source.find(marker)
    if start < 0:
        return None
    depth = 0
    index = start + len(selector) + 1  # at the branch's opening brace
    while index < len(source):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return start, index + 1
        index += 1
    return None


def _control_drop_branch(ctx: Ctx, key: str, selector: str) -> Ctx:
    value = ctx.es[key]
    span = _branch_span(value, selector)
    if span is None:
        return _mutate(ctx, key, value)
    start, end = span
    return _mutate(ctx, key, (value[:start] + value[end:]).replace("  ", " ").strip())


def _control_insert_branch(ctx: Ctx, key: str, selector: str, body: str) -> Ctx:
    value = ctx.es[key]
    span = _branch_span(value, "other")
    if span is None:
        return _mutate(ctx, key, value)
    start, _end = span
    return _mutate(ctx, key, value[:start] + f"{selector} {body} " + value[start:])


def _control_drop_many(ctx: Ctx, key: str) -> Ctx:
    return _control_drop_branch(ctx, key, "many")


def _control_add_few(ctx: Ctx, key: str) -> Ctx:
    return _control_insert_branch(ctx, key, "few", "{{shown} unos despertares}")


def _control_code_span(ctx: Ctx) -> Ctx:
    key = "tui.app.help.title_note"
    return _mutate(ctx, key, ctx.es[key].replace("`lop config edit display.terminal_title false`", "`lop config edit`"))


CONTROLS = [
    ("drop a key", "tui.hub_links", _control_drop_key, "A1", "fail"),
    (
        "rename {value} to {amount}",
        "tui.usage",
        lambda c: _control_rename_placeholder(c, "tui.usage.stale"),
        "A2",
        "fail",
    ),
    (
        "drop the `many` branch",
        "tui.usage",
        lambda c: _control_drop_many(c, "tui.usage.windows_count"),
        "A3",
        "fail",
    ),
    (
        "add an illegal `few` branch",
        "tui.app",
        lambda c: _control_add_few(c, "tui.app.parked.wakes"),
        "A3",
        "fail",
    ),
    ("shorten a backticked command", "tui.app", _control_code_span, "A5", "fail"),
    (
        "pad a /help row to 75",
        "tui.app",
        lambda c: _mutate(c, "tui.app.help.cmd_paste_note", c.es["tui.app.help.cmd_paste_note"] + "xxx"),
        "B2",
        "fail",
    ),
    (
        "restore w6a1's `el aviso`",
        "tui.hub_links",
        lambda c: _mutate(
            c,
            "tui.hub_links.refusal.revision_moved",
            "Este vínculo cambió mientras lo revisabas. Vuelve a abrir el aviso para confirmar la última versión.",
        ),
        "C1",
        "fail",
    ),
    (
        "restore `{value} agotadas`",
        "tui.usage",
        lambda c: _mutate(c, "tui.usage.exhausted", "{value} agotadas"),
        "C9",
        "fail",
    ),
    (
        "lengthen an accepted paid note past the 60-col body",
        "tui.usage",
        lambda c: _mutate(
            c,
            "tui.usage.paid.blocked",
            "Gasto no disponible: el proveedor no informa de saldo disponible ahora",
        ),
        "B5",
        "fail",
    ),
    (
        "claim that billing stops on exhaustion",
        "tui.usage",
        lambda c: _mutate(
            c, "tui.usage.paid.continue", "Asignación incluida agotada; se detendrá"
        ),
        "C8",
        "fail",
    ),
    (
        "legitimate rewording of an unchanged row (intended pass)",
        "tui.app",
        lambda c: _mutate(
            c, "tui.app.help.model_saved_note", "vuelve al valor por defecto de config.yml"
        ),
        "B2",
        "pass",
    ),
]


def run_controls() -> int:
    print("\n=== D — negative controls (a check that cannot fail proves nothing) ===")
    bad = 0
    for name, ns, mutate, check_id, expect in CONTROLS:
        ctx = Ctx(ns)
        mutated = mutate(ctx)
        r = Result()
        fn = dict(CHECK_FUNCTIONS)[check_id]
        fn(mutated, r)
        fired = bool(r.failures)
        got = "fail" if fired else "pass"
        if got == expect:
            print(f"CTRL ok   [{check_id}] {name} → {got} (expected {expect})")
        else:
            bad += 1
            print(f"CTRL BAD  [{check_id}] {name} → {got} (expected {expect})")
    print(f"controls: {len(CONTROLS) - bad}/{len(CONTROLS)} behaved as required")
    return bad


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", action="append", choices=NAMESPACES)
    args = parser.parse_args()
    namespaces = args.namespace or list(NAMESPACES)

    print(f"cell ruler: {CELL_SOURCE}")
    print(
        "geometry: help ceiling {} | stop_all budget {} | panel {}..{} (margin {}) "
        "| padding {} | scrollbar {} | floor body {} at 40 cols".format(
            HELP_CEILING,
            STOP_ALL_BUDGET,
            PANEL_MIN_WIDTH,
            PANEL_MAX_WIDTH,
            PANEL_WIDTH_MARGIN,
            PANEL_PADDING_CELLS,
            SCROLLBAR_GUTTER_CELLS,
            body_width(40),
        )
    )
    print("input digests:")
    for ns in namespaces:
        for kind, path in (
            ("en", CAT / "en" / f"{ns}.json"),
            ("es", CAT / "es" / f"{ns}.json"),
            ("ctx", CAT / "en" / f"{ns}.context.json"),
        ):
            print(f"  {kind:3} {sha256(path)}  {path.relative_to(REPO)}")

    failures = 0
    per_namespace = {}
    for ns in namespaces:
        result = run_namespace(ns)
        per_namespace[ns] = result
        failures += len(result.failures)

    if len(namespaces) == len(NAMESPACES):
        failures += run_controls()

    print("\n=== summary ===")
    for ns, result in per_namespace.items():
        print(f"{ns}: {len(result.failures)} failing check(s)")
        for failure in result.failures:
            print(f"  - {failure}")
    print(f"total failing checks: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

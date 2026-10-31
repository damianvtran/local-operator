#!/usr/bin/env python3
"""Round-3 delta audit of ``es/tui.analytics`` + class sweep of code-fixed columns.

WHY THIS EXISTS. Round 2 passed ``tui.analytics`` and independent QA then found,
in a RENDERED frame, that two Spanish Totals labels overran the 22-cell label
column that ``analytics_panel.kv()`` hard-codes (``f"  {name:<22}"``), fusing
label and value (``Tasa de aciertos de caché86%``). No ``maxCells`` is declared
for those keys, so a sidecar-driven width check could not see it. This script
therefore derives each column FROM THE PANEL SOURCE (regex/AST over the .py
files, never a restated literal) and measures the live catalogue against it
with ``rich.cells.cell_len`` — the function the TUI itself budgets with.

READ-ONLY. It never writes a catalogue, sidecar, ledger, allowance or generated
file. It builds one throw-away catalogue root under a temp dir for the negative
control and deletes it.

Run from the worktree root with the checkout's own interpreter::

    env PYTHONPATH=$PWD .venv/bin/python \
        i18n/audits/evidence/es/tui.analytics/w6a3/checks.py

Exit 0 only if no BLOCKING check failed AND every negative control tripped.
Findings that the sweep reports but that are not column-fusion failures of this
namespace are printed as ``FINDING`` lines and do NOT change the exit code: the
auditor's verdict on them is in record.json / report.md, not in an exit status.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from rich.cells import cell_len
from rich.console import Console

ROOT = Path(os.environ.get("AUDIT_REPO_ROOT", os.getcwd()))
CAT = ROOT / "local_operator" / "i18n" / "catalogues"
TUI = ROOT / "local_operator" / "tui"
NS_ALL = ["small", "library", "projects", "analytics", "import", "hub_links", "app", "usage"]

blocking: list[str] = []
findings: list[str] = []
controls: list[tuple[str, bool]] = []
table: list[tuple[str, str, str, str, str, str]] = []  # key, site, column, en, es, verdict


def load(locale: str, ns: str) -> dict:
    with (CAT / locale / f"tui.{ns}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(check: str, msg: str) -> None:
    blocking.append(f"[{check}] {msg}")
    print(f"FAIL  [{check}] {msg}")


def ok(check: str, msg: str) -> None:
    print(f"PASS  [{check}] {msg}")


def cells(s: str) -> int:
    return cell_len(s)


# ---------------------------------------------------------------------------
# 0. identity of what is being audited
# ---------------------------------------------------------------------------
print("=== 0. inputs (sha256 of the CURRENT bytes) ===")
TARGET = CAT / "es" / "tui.analytics.json"
EN = CAT / "en" / "tui.analytics.json"
CTX = CAT / "en" / "tui.analytics.context.json"
for label, p in (("target es", TARGET), ("en", EN), ("context", CTX)):
    print(f"{label:10} {sha(p)}  {p.relative_to(ROOT)}")
print(f"rich.cells.cell_len = {cell_len.__module__}.{cell_len.__name__}")

es = load("es", "analytics")
en = load("en", "analytics")
ctx = json.load(CTX.open(encoding="utf-8"))

# ---------------------------------------------------------------------------
# 1. mechanical
# ---------------------------------------------------------------------------
print("\n=== 1. mechanical ===")
if set(es) != set(en):
    fail("keys", f"es-en={sorted(set(es)-set(en))} en-es={sorted(set(en)-set(es))}")
else:
    ok("keys", f"{len(es)} keys, identical to en")
PH = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)")  # argument NAMES (plural/select heads included)
bad = [k for k in en if set(PH.findall(en[k])) != set(PH.findall(es[k]))]
bad += [k for k in en if "plural" in en[k] and "plural" not in es[k]]
(fail if bad else ok)("placeholders", f"mismatches: {bad}" if bad else "argument names identical on every key")
miss = [k for k in es if "plural" in es[k] and not all(f"{c} {{" in es[k] for c in ("one", "many", "other"))]
(fail if miss else ok)("cldr-es", f"missing one/many/other: {miss}" if miss else "every es plural has one/many/other")

# THE DELTA — exactly the two values the manager named.
EXPECT = {
    "tui.analytics.totals.cache_hit": "Aciertos de caché",
    "tui.analytics.totals.decode": "Decodificación",
}
for key, want in EXPECT.items():
    (ok if es[key] == want else fail)("delta", f"{key} = {es[key]!r} (expected {want!r}), {cells(es[key])} cells")

# ---------------------------------------------------------------------------
# 2. the column: derived from the panel source, not restated
# ---------------------------------------------------------------------------
print("\n=== 2. totals label column (derived from analytics_panel.py) ===")
PANEL = TUI / "widgets" / "analytics_panel.py"
panel_src = PANEL.read_text(encoding="utf-8")
m = re.search(r'row\.append\(f"  \{name:<(\d+)\}", style=dim\)', panel_src)
if not m:
    fail("column", "could not find kv()'s `f\"  {name:<N}\"` pad in analytics_panel.py")
    KV_COL = 22
else:
    KV_COL = int(m.group(1))
    ok("column", f"kv() pads name to {KV_COL} cells (analytics_panel.py:kv); value starts at cell {2 + KV_COL}")
m2 = re.search(r"^_SEARCH_NAME_COL = (\d+)", panel_src, re.M)
SEARCH_COL = int(m2.group(1)) if m2 else 22
m3 = re.search(r"^\s+_VALUE_CELL = (\d+)", panel_src, re.M)
VALUE_CELL = int(m3.group(1)) if m3 else 11

# accessor name -> catalogue key, from the generated typed-key module
keys_src = (ROOT / "local_operator" / "i18n" / "keys" / "tui_analytics.py").read_text(encoding="utf-8")
ACC: dict[str, str] = {}
for node in ast.parse(keys_src).body:
    if isinstance(node, ast.FunctionDef):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "Msg" and sub.args:
                a0 = sub.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    ACC[node.name] = a0.value


def kv_label_sites(tree: ast.AST, call_name: str) -> list[tuple[str, str]]:
    """(prefix literal, catalogue key) for the FIRST argument of every ``call_name(...)``."""
    out: list[tuple[str, str]] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == call_name and n.args:
            first = n.args[0]
            prefix, key = "", None
            for s in ast.walk(first):
                if isinstance(s, ast.Constant) and isinstance(s.value, str):
                    prefix += s.value
                if isinstance(s, ast.Attribute) and isinstance(s.value, ast.Name) and s.value.id == "tui_analytics":
                    key = ACC.get(s.attr)
            if key:
                out.append((prefix, key))
    return out


panel_tree = ast.parse(panel_src)
kv_sites = kv_label_sites(panel_tree, "kv")


def column_check(label_by_key: dict[str, str], tag: str) -> list[str]:
    """Every kv() label, with its literal tree prefix, must leave >= 1 blank cell."""
    problems: list[str] = []
    for prefix, key in kv_sites:
        text = prefix + label_by_key[key]
        n = cells(text)
        if n > KV_COL - 1:
            problems.append(f"{key}: {n} cells ({text!r}) in a {KV_COL}-cell column")
    return problems


live_labels = {k: es[k] for _, k in kv_sites}
print(f"discovered {len(kv_sites)} kv() label sites by AST: {[k.rsplit('.',1)[-1] for _,k in kv_sites]}")
probs = column_check(live_labels, "live")
(fail if probs else ok)("kv-column", "; ".join(probs) if probs else f"all {len(kv_sites)} labels <= {KV_COL-1} cells (>= 1 blank cell before the value)")
for prefix, key in kv_sites:
    text = prefix + es[key]
    etext = prefix + en[key]
    table.append((key, f"analytics_panel.kv  `{{name:<{KV_COL}}}`", f"{KV_COL}", str(cells(etext)), str(cells(text)),
                  "OK" if cells(text) <= KV_COL - 1 else "OVER"))

# search block: same literal, different constant
for k in ("tui.analytics.search.row_total", "tui.analytics.search.row_session"):
    n = cells(es[k])
    table.append((k, f"analytics_panel search row  `{{name:<{SEARCH_COL}}}`", f"{SEARCH_COL}", str(cells(en[k])), str(n),
                  "OK" if n <= SEARCH_COL - 1 else "OVER"))
    (ok if n <= SEARCH_COL - 1 else fail)("search-col", f"{k} {n} cells vs {SEARCH_COL}")

# ---------------------------------------------------------------------------
# 3. NEGATIVE CONTROL A (static): the check must FAIL on the round-2 wording
# ---------------------------------------------------------------------------
print("\n=== 3. negative control A — restore the round-2 labels, the column check must FAIL ===")
OLD = {
    "tui.analytics.totals.cache_hit": "Tasa de aciertos de caché",
    "tui.analytics.totals.decode": "Tasa de decodificación",
}
poisoned = dict(live_labels)
poisoned.update(OLD)
neg = column_check(poisoned, "poisoned")
tripped = len(neg) == 2
controls.append(("static column check trips on round-2 labels", tripped))
print(("TRIPPED " if tripped else "NOT TRIPPED ") + "(expected 2 problems): " + " | ".join(neg))
if not tripped:
    fail("neg-control-A", "the column check did not flag the round-2 wording — the check is blind")

# ---------------------------------------------------------------------------
# 4. RENDERED frame: the same question asked of build_report's real output
# ---------------------------------------------------------------------------
print("\n=== 4. rendered Totals block (build_report), live bytes vs round-2 bytes ===")
sys.path.insert(0, str(ROOT))
from local_operator.analytics.model import COMPONENT_KEYS, UsageAggregate  # noqa: E402
from local_operator.i18n import catalogues, text as i18n_text  # noqa: E402
from local_operator.tui.widgets.analytics_panel import build_report  # noqa: E402


def make_agg() -> UsageAggregate:
    agg = UsageAggregate(
        calls=100, ok_calls=98, input_tokens=500_000, output_tokens=120_000,
        cache_read_tokens=3_000_000, cache_write_tokens=80_000, reasoning_tokens=40_000,
        context_tokens=3_500_000, cost_micro=4_200_000, cost_known_calls=100,
    )
    agg.components = {k: 0 for k in COMPONENT_KEYS}
    agg.components["conversation"] = 1_800_000
    # a measured decode rate so the Decode row carries a real figure (`42 tok/s`)
    agg.decode_tokens, agg.decode_us, agg.decode_calls = 42_000, 1_000_000_000, 10
    return agg


def totals_rows(locale: str, width: int = 120) -> list[str]:
    i18n_text.set_active_locale(locale)
    lines = [ln.plain for ln in build_report(make_agg(), width)]
    out: list[str] = []
    started = False
    for ln in lines:
        for seg in ln.split("\n"):
            if seg.startswith("▌") and not started:
                started = True
                continue
            if started:
                if not seg.strip():
                    return out
                out.append(seg)
    return out


def fused(rows: list[str]) -> list[str]:
    """A row is FUSED when the label's last cell touches the value, i.e. cell 23 is not blank
    or the value does not start exactly at the column the code reserves (2 + KV_COL)."""
    bad = []
    for seg in rows:
        if cells(seg) < 2 + KV_COL + 1:
            continue
        if seg[1 + KV_COL] != " " or seg[2 + KV_COL] == " ":
            bad.append(seg)
    return bad


live_rows = totals_rows("es")
for seg in live_rows:
    print("   |" + seg)
f = fused(live_rows)
(fail if f else ok)("rendered-gap", f"fused rows: {f}" if f else f"all {len(live_rows)} Totals rows keep a blank cell and start the value at cell {2+KV_COL}")
en_rows = totals_rows("en")
(fail if fused(en_rows) else ok)("rendered-gap-en", "English control: no fused row" if not fused(en_rows) else f"EN fused: {fused(en_rows)}")

# NEGATIVE CONTROL B (rendered): a throw-away catalogue root carrying the round-2 bytes
tmp = Path(tempfile.mkdtemp(prefix="es-r3-neg-"))
try:
    for loc in ("en", "es"):
        (tmp / loc).mkdir()
        shutil.copy(CAT / loc / "tui.analytics.json", tmp / loc / "tui.analytics.json")
    shutil.copy(CTX, tmp / "en" / "tui.analytics.context.json")
    old = json.load((tmp / "es" / "tui.analytics.json").open(encoding="utf-8"))
    old.update(OLD)
    (tmp / "es" / "tui.analytics.json").write_text(json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    real_root = catalogues.catalogue_root
    catalogues.catalogue_root = lambda: tmp  # type: ignore[assignment]
    try:
        neg_rows = totals_rows("es")
    finally:
        catalogues.catalogue_root = real_root  # type: ignore[assignment]
finally:
    shutil.rmtree(tmp, ignore_errors=True)
for seg in neg_rows:
    if "Tasa" in seg:
        print("   NEG|" + seg)
neg_fused = fused(neg_rows)
trip = len(neg_fused) == 2 and any("caché86%" in s for s in neg_fused)
controls.append(("rendered frame fuses on round-2 labels and the check sees it", trip))
print(("TRIPPED " if trip else "NOT TRIPPED ") + f"rendered negative control: {len(neg_fused)} fused row(s)")
if not trip:
    fail("neg-control-B", "round-2 bytes did not reproduce the QA fusion in a rendered frame")
i18n_text.set_active_locale("es")

# ---------------------------------------------------------------------------
# 5. the label question: what the row now says, in its row
# ---------------------------------------------------------------------------
print("\n=== 5. label + value + note as the user reads them ===")
for key in EXPECT:
    note_key = key + "_note"
    print(f"  {es[key]:<18}| {'86%' if 'cache' in key else '42 tok/s':<9}| {es[note_key]}")
print(f"  rate.legend (still on screen under the tables): {es['tui.analytics.rate.legend']!r}")

# ---------------------------------------------------------------------------
# 6. CLASS SWEEP — other code-fixed columns, all 8 namespaces
# ---------------------------------------------------------------------------
print("\n=== 6. class sweep ===")
# 6a. mechanical discovery: every site where a catalogue string reaches a pad/justify/
#     fixed-width sink, found by AST over the panel sources (importers of the 8 keys modules).
MODS = {f"tui_{n}" for n in NS_ALL}
SINK_ATTR = {"ljust", "rjust", "center", "add_column"}
SINK_NAME = {"kv", "marked", "_pad_cells"}
found: dict[str, list[str]] = {}
for py in sorted(list((TUI).glob("*.py")) + list((TUI / "widgets").glob("*.py"))):
    src = py.read_text(encoding="utf-8")
    if "local_operator.i18n.keys" not in src:
        continue
    tree = ast.parse(src)
    alias = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "local_operator.i18n.keys":
            for a in n.names:
                if a.name in MODS:
                    alias[a.asname or a.name] = a.name
    if not alias:
        continue

    def direct(node, taint):
        for s in ast.walk(node):
            if isinstance(s, ast.Attribute) and isinstance(s.value, ast.Name) and s.value.id in alias:
                return f"{alias[s.value.id]}.{s.attr}"
            if isinstance(s, ast.Name) and s.id in taint:
                return "~" + s.id
        return None

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        taint: set[str] = set()
        for _ in range(3):
            for n in ast.walk(fn):
                if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and getattr(n, "value", None) is not None and direct(n.value, taint):
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                        for s in ast.walk(t):
                            if isinstance(s, ast.Name):
                                taint.add(s.id)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("append", "extend", "add") \
                        and isinstance(n.func.value, ast.Name) and any(direct(a, taint) for a in n.args):
                    taint.add(n.func.value.id)
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                nm = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
                hit = None
                if nm in SINK_ATTR | SINK_NAME:
                    if nm in ("ljust", "rjust", "center") and isinstance(n.func, ast.Attribute):
                        hit = direct(n.func.value, taint)
                    else:
                        hit = next((direct(a, taint) for a in list(n.args) + [k.value for k in n.keywords] if direct(a, taint)), None)
                if hit:
                    found.setdefault(f"{py.relative_to(ROOT)}:{fn.name}:{nm}", []).append(hit)
            if isinstance(n, ast.FormattedValue) and n.format_spec is not None and re.search(r"[<>^]", ast.unparse(n.format_spec)):
                hit = direct(n.value, taint)
                if hit:
                    found.setdefault(f"{py.relative_to(ROOT)}:{fn.name}:fspec", []).append(hit)
print("catalogue string -> pad/justify/fixed-width sink (AST, taint through locals):")
for site, hits in sorted(found.items()):
    print(f"   {site}  <- {sorted(set(hits))}")
EXPECTED_SITES = {
    "local_operator/tui/widgets/analytics_panel.py:build_report:kv",
    "local_operator/tui/widgets/info_panel.py:_sessions_section:marked",
    # Taint reaches the pad through ``name_col`` (a WIDTH derived from cell_len of the
    # catalogue suffixes: table_tokens/table_tok_s/table_coverage), while the padded VALUE is
    # a provider/model id. Data-sized column, so tolerated; listed so a change is noticed.
    "local_operator/tui/widgets/analytics_panel.py:_model_rate_section:fspec",
}
if set(found) != EXPECTED_SITES:
    fail("sweep-discovery", f"sink set changed: new={sorted(set(found)-EXPECTED_SITES)} gone={sorted(EXPECTED_SITES-set(found))} — re-audit")
else:
    ok("sweep-discovery", "the only direct catalogue->pad sinks are analytics kv() (checked above) and info_panel marked() (a width LADDER, see 6e)")

# 6b. library rail: fixed by CSS (`.library-view-rail { width: N }`), labels truncated by code
tcss = (TUI / "local_operator.tcss").read_text(encoding="utf-8")
rail_w = int(re.search(r"\.library-view-rail \{\s*width: (\d+);", tcss).group(1))
lib_src = (TUI / "widgets" / "library_view.py").read_text(encoding="utf-8")
assert "width = max(10, (self._rail.size.width or 24) - 2)" in lib_src and "label_width = width - 2 - (4 if item.count is not None else 0)" in lib_src, "rail formula moved: re-derive"
rail_inner = max(10, rail_w - 2)
lab_count = rail_inner - 2 - 4  # category and tag rows carry a count
lab_nocount = rail_inner - 2  # status rows
lib_es, lib_en = load("es", "library"), load("en", "library")
print(f"library rail: CSS width {rail_w} -> inner {rail_inner}; label room {lab_count} (rows with a count) / {lab_nocount} (status rows) / {rail_inner} (UPPERCASE header)")
for k in sorted(lib_en):
    n = k.split(".", 2)[2]
    if n.startswith(("category_", "tag_category_")) or n.startswith(("sidebar_",)) or n.startswith("status_"):
        if n.startswith("sidebar_"):
            col, a, b = rail_inner, cells(lib_en[k].upper()), cells(lib_es[k].upper())
        elif n.startswith("status_"):
            col, a, b = lab_nocount, cells(lib_en[k]), cells(lib_es[k])
        elif n.startswith("tag_category_"):
            col, a, b = rail_inner, cells(lib_en[k].upper()), cells(lib_es[k].upper())  # header rows
        else:
            col, a, b = lab_count, cells(lib_en[k]), cells(lib_es[k])
        v = "OK" if b <= col else "TRUNC(ellipsis)"
        table.append((k, "library rail (CSS width:22, truncate_cells)", str(col), str(a), str(b), v))

# 6c. projects board card: BOARD_COLUMN_WIDTH, cell.truncate(W-1, ellipsis)
pr_src = (TUI / "projects_render.py").read_text(encoding="utf-8")
BOARD_W = int(re.search(r"^BOARD_COLUMN_WIDTH = (\d+)", pr_src, re.M).group(1))
pj_es, pj_en = load("es", "projects"), load("en", "projects")
room = BOARD_W - 1
for k, extra_es, extra_en, label in (
    ("tui.projects.render.no_estimate_or_dates", "", "", "facts line (alone)"),
    ("tui.projects.render.no_progress", "", "", "fresh line (alone)"),
    ("tui.projects.render.no_progress_recorded", "", "", "detail line"),
):
    a, b = cells(pj_en[k]), cells(pj_es[k])
    table.append((k, f"projects board card (BOARD_COLUMN_WIDTH={BOARD_W}, {label})", str(room), str(a), str(b), "OK" if b <= room else "TRUNC(ellipsis)"))


def board_line(locale: str, *, age_s: int, stale: bool, live_n: int) -> str:
    from local_operator.tui.projects_render import render_project_board

    i18n_text.set_active_locale(locale)
    now = 1_760_000_000.0
    project = {"id": "id-p", "name": "p", "title": None, "owner": None, "team": None, "status": "active",
               "description": "", "progress": "x", "progress_updated_at": now - age_s, "progress_reported_by": "",
               "tags": [], "sessions": [], "created_at": 0.0, "updated_at": 0.0, "start_date": None,
               "target_date": None, "completed_at": None, "estimate": None, "estimate_unit": "points",
               "milestones": [], "schema": 1}
    sess = [{"session_id": f"s{i}", "exists": True, "title": "t", "created_at": None, "archived": False,
             "runtime": {"state": "live", "busy": None, "heartbeat_age_s": None, "pid": None},
             "subagents": None, "todos": None} for i in range(live_n)]
    view = {"project": project, "progress_stale": stale, "sessions": sess}
    return render_project_board([view], now=now).text.plain.split("\n")[-1]


for tag, kw in (("fresh 2h, 2 live", dict(age_s=7200, stale=False, live_n=2)),
                ("stale 3d, 1 live", dict(age_s=259200, stale=True, live_n=1))):
    e, s_ = board_line("en", **kw), board_line("es", **kw)
    table.append(("tui.projects.render.reported_ago + live_count", f"projects board card fresh line ({tag}) [rendered]", str(room),
                  f"{cells(e.replace('…','')) if '…' not in e else room}{'+' if '…' in e else ''}",
                  f"{cells(s_) if '…' not in s_ else room}{'+' if '…' in s_ else ''}",
                  "OK" if "…" not in s_ else "TRUNC(ellipsis) " + repr(s_)))
    print(f"   board {tag}: en={e!r} es={s_!r}")
i18n_text.set_active_locale("es")

# 6d. goal panel done tag: the budget is a CODE LITERAL, subtracted before the tag is appended
gp_src = (TUI / "widgets" / "goal_panel.py").read_text(encoding="utf-8")
GOAL_TAG = int(re.search(r"^_GOAL_TAG_CELLS = (\d+)", gp_src, re.M).group(1))
sm_es, sm_en = load("es", "small"), load("en", "small")
k = "tui.small.goal_panel.done_suffix"
a, b = cells(sm_en[k]), cells(sm_es[k])
v = "OK" if b <= GOAL_TAG else "OVER"
table.append((k, "goal_panel `_GOAL_TAG_CELLS` (clip budget for the `  — done` tag)", str(GOAL_TAG), str(a), str(b), v))
print(f"goal_panel._GOAL_TAG_CELLS = {GOAL_TAG}; done_suffix en {a} cells, es {b} cells ({sm_es[k]!r})")
if b > GOAL_TAG:
    from local_operator.tui.widgets.goal_panel import build_goal_body

    for loc in ("en", "es"):
        i18n_text.set_active_locale(loc)
        for cw in (40, 66, 76):
            body = build_goal_body(goal="Migrar el servicio de pagos al nuevo proveedor y verificar cada cobro con el equipo de finanzas " * 2,
                                   status="done", judge=None, history=[], width=cw, cap=3, actions=True, max_rows=0)
            row = next(x for x in body.plain.split("\n") if x.startswith("▌"))
            buf = io.StringIO()
            Console(file=buf, width=cw, color_system=None, force_terminal=False, highlight=False).print(row, end="")
            lines = buf.getvalue().split("\n")
            print(f"   goal row [{loc}] content_width={cw}: {cells(row)} cells -> {len(lines)} painted line(s); last={lines[-1][-24:]!r}")
    i18n_text.set_active_locale("es")
    findings.append(f"FINDING F1 goal_panel done_suffix {b} cells > _GOAL_TAG_CELLS {GOAL_TAG}: clipped goal row wraps, `hecho` orphaned on line 2")

# 6e. tui.app /help key reference: name_width derived, ceiling 74 (pinned by test_app_pilot)
from local_operator.slash_commands import SLASH_COMMANDS  # noqa: E402

name_w = max(len(", ".join(f"/{n}" for n in c.names)) for c in SLASH_COMMANDS) + 2
ap_es, ap_en = load("es", "app"), load("en", "app")
CEIL = 74
for k in sorted(ap_en):
    if k.startswith("tui.app.help.") and k.endswith("_note") or k.startswith("tui.app.help.") and k.endswith("_note_more"):
        a, b = name_w + cells(ap_en[k]), name_w + cells(ap_es[k])
        wraps_en, wraps_es = a > CEIL, b > CEIL
        v = "OK" if b <= CEIL else ("OVER (en also wraps: parity)" if wraps_en else "OVER")
        table.append((k, f"app /help gutter name_width={name_w}, ceiling {CEIL} (composed)", str(CEIL), str(a), str(b), v))
        if wraps_es and not wraps_en:
            findings.append(f"FINDING app help {k}: es wraps where en does not")

# 6f. analytics notes beside the kv value: the note column starts at a CODE-FIXED cell
note_start = 2 + KV_COL + VALUE_CELL + 2
print(f"analytics kv note column starts at cell {note_start} (2 + {KV_COL} + {VALUE_CELL} + 2); the report body crops (ReportView) rather than grows")


def note_rows(locale: str, width: int) -> list[tuple[int, str]]:
    i18n_text.set_active_locale(locale)
    out = []
    for ln in build_report(make_agg(), width):
        for seg in ln.plain.split("\n"):
            if seg.startswith("  ") and cells(seg) > note_start and seg[2 + KV_COL: 2 + KV_COL + 1] != "" and any(
                seg.lstrip(" ├└").startswith(x) for x in ("Total", "Contex", "Nuevo", "Lectura", "Escritura", "Salida", "Aciertos", "Decodif",
                                                          "Context", "Fresh", "Cache", "Output", "Decode")):
                out.append((cells(seg), seg))
    return out


for w in (66, 75, 102):
    en_o = [(n, s) for n, s in note_rows("en", w) if n > w]
    es_o = [(n, s) for n, s in note_rows("es", w) if n > w]
    print(f"   body width {w}: rows wider than the body — en {len(en_o)}, es {len(es_o)}")
    for n, s in es_o:
        print(f"      es {n:3}>{w}  {s}")
i18n_text.set_active_locale("es")
for k in ("tui.analytics.totals.fresh_note", "tui.analytics.totals.cache_read_note", "tui.analytics.totals.cache_write_note",
          "tui.analytics.totals.cache_hit_note", "tui.analytics.totals.decode_note"):
    a, b = cells(en[k]), cells(es[k])
    room29 = 66 - note_start
    table.append((k, f"analytics kv note, 80x24 body=66 -> note room {room29} (crop, not grow)", str(room29), str(a), str(b),
                  "OK" if b <= room29 else ("CROP (en also)" if a > room29 else "CROP")))

# 6g. tolerated-by-construction sites (measured live, listed so the sweep is complete)
tol = [
    ("tui.analytics.table.{tokens,tok_s,calls,cache,coverage}", "analytics table builders: columns sized from data + cell_len(suffix) (`_row_overhead`)", "derived"),
    ("tui.usage.*", "usage_panel `_measure_columns`: label/numbers/reset sized from the rows; bar takes the remainder", "derived"),
    ("tui.hub_links.policy.*", "hub_links `_render_row`: name floor 12, policy wraps to line 2 below it", "ladder"),
    ("tui.import.*", "harness_import_view: truncate_cells only; `[x]`/`[ ]` markers identical to en", "truncate"),
    ("tui.small.session_sidebar.*", "session_sidebar `_footer_line`: first rung that fits, never cropped mid-word", "ladder"),
    ("tui.small.todo_panel.more_below", "todo_panel `_expanded_footer`: measures the widest remainder with cell_len", "derived"),
    ("tui.small.info_panel.*", "info_panel marked(): `_fit_pair` ladder (metas) — no fixed pad on the catalogue string", "ladder"),
]
for key, site, kind in tol:
    table.append((key, site, kind, "-", "-", "OK (tolerated)"))

print("\nSWEEP TABLE (key | site | column | en cells | es cells | verdict)")
for row in table:
    print("  | " + " | ".join(row))

# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
print("\n=== result ===")
for name, tripped in controls:
    print(f"negative control: {name}: {'TRIPPED' if tripped else 'NOT TRIPPED'}")
for line in findings:
    print(line)
print(f"target sha256 verified: {sha(TARGET)}")
print(f"blocking failures: {len(blocking)}")
for b in blocking:
    print("  " + b)
sys.exit(1 if blocking or not all(t for _, t in controls) else 0)

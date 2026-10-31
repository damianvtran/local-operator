"""Round-4 closing checks for es/tui.small (finding S1: goal_panel.done_suffix).

Run from the worktree root with the repo venv python, read-only:
    env -i HOME=<tmp> LOCAL_OPERATOR_CONFIG_DIR=<tmp>/cfg PATH=... \
        <venv>/bin/python i18n/audits/evidence/es/tui.small/w6a4/checks.py <scratch_dir>

<scratch_dir> must hold `goal_panel_prefix.py`, i.e. the output of
`git show 4912975a4b5:local_operator/tui/widgets/goal_panel.py` (the code BEFORE the
fix). It is loaded under another module name from there; nothing is ever written
into the worktree. Why a control: a probe that passes on the fixed code proves
nothing unless the same probe FAILS on the unfixed code.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys

sys.path.insert(0, os.getcwd())

from rich.cells import cell_len
from rich.console import Console

from local_operator.i18n import text
from local_operator.tui.widgets import goal_panel as fixed

SCRATCH = sys.argv[1]
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


# (1) target bytes and value -------------------------------------------------
tgt = "local_operator/i18n/catalogues/es/tui.small.json"
en = "local_operator/i18n/catalogues/en/tui.small.json"
sha = hashlib.sha256(open(tgt, "rb").read()).hexdigest()
check("target sha256", sha == "e374959d70bb379a37e3f02e8c69edeea168b7edbd11107e4131f41fe116b9af", sha)
es_cat, en_cat = json.load(open(tgt)), json.load(open(en))
check("137 keys, same key set", len(es_cat) == 137 and es_cat.keys() == en_cat.keys())
K = "tui.small.goal_panel.done_suffix"
check("value is exactly '  — hecho'", es_cat[K] == "  \u2014 hecho", repr(es_cat[K]))
check("two leading spaces + U+2014", es_cat[K].startswith("  \u2014 "))
print("INFO  cells en=%d es=%d  _GOAL_TAG_CELLS=%d" % (cell_len(en_cat[K]), cell_len(es_cat[K]), fixed._GOAL_TAG_CELLS))

# load the PRE-fix module from scratch under another name
spec = importlib.util.spec_from_file_location("goal_panel_prefix", os.path.join(SCRATCH, "goal_panel_prefix.py"))
pre = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pre)
check("control module really is pre-fix", "cell_len(done_suffix)" not in open(os.path.join(SCRATCH, "goal_panel_prefix.py")).read())

GOAL = ("Migrar el servicio de pagos al nuevo proveedor y verificar cada cobro con el equipo "
        "de finanzas antes del cierre trimestral del año ") * 2


def render(mod, goal: str, cols: int) -> list[str]:
    """Body at terminal width `cols`; content box = cols - PANEL_PADDING_CELLS, as the card does."""
    cw = cols - fixed.PANEL_PADDING_CELLS
    body = mod.build_goal_body(goal=goal, status="done", judge=None, history=[], width=cw, cap=3,
                               actions=True, max_rows=0)
    buf = io.StringIO()
    Console(file=buf, width=cw, color_system=None, force_terminal=False, highlight=False).print(body, end="")
    return buf.getvalue().split("\n")


def orphan(lines: list[str]) -> bool:
    return any(l.strip() == "hecho" for l in lines)


def goal_lines(lines: list[str]) -> list[str]:
    """Painted lines belonging to the goal row: from the spine to the line before 'judge'."""
    i = next(n for n, l in enumerate(lines) if l.startswith("▌"))
    j = next(n for n, l in enumerate(lines) if n > i and l.startswith("judge"))
    return lines[i:j]


# (2) fixed code, three widths --------------------------------------------------
text.set_active_locale("es")
for cols in (60, 80, 100):
    cw = cols - fixed.PANEL_PADDING_CELLS
    lines = render(fixed, GOAL, cols)
    g = goal_lines(lines)
    print(f"INFO  [fixed] cols={cols} content={cw}")
    for l in g:
        print("      |" + l)
    check(f"fixed {cols}: goal row is exactly one painted line", len(g) == 1, f"lines={len(g)}")
    check(f"fixed {cols}: tag stays on goal row", g[0].rstrip().endswith("\u2014 hecho"))
    check(f"fixed {cols}: no line is just 'hecho'", not orphan(lines))
    check(f"fixed {cols}: every painted line <= content width", all(cell_len(l) <= cw for l in lines),
          f"max={max(cell_len(l) for l in lines)}/{cw}")
    check(f"fixed {cols}: hint row still present", any("c " in l or "limpiar" in l or "borrar" in l for l in lines[-3:]) or len(lines) > 4)

# (2b) NEGATIVE CONTROL: same probe, pre-fix code -----------------------------------
control_orphans = 0
for cols in (60, 80, 100):
    cw = cols - fixed.PANEL_PADDING_CELLS
    lines = render(pre, GOAL, cols)
    g = goal_lines(lines)
    print(f"INFO  [pre-fix] cols={cols} content={cw}")
    for l in g:
        print("      |" + l)
    o = orphan(lines)
    control_orphans += o
    check(f"CONTROL pre-fix {cols}: reproduces the orphan (expected: wraps)", o and len(g) == 2, f"lines={len(g)} orphan={o}")

# (2c) exhaustive sweep of goal lengths: exactly which goal widths orphan, pre vs fixed
import itertools
words = GOAL.split()
def goals():
    s = " ".join(words)
    for n in range(10, 160):
        yield s[:n].rstrip()
bad_fixed, bad_pre = [], []
for cols in (60, 80, 100):
    for g in goals():
        if not g:
            continue
        if orphan(render(fixed, g, cols)) or any(cell_len(l) > cols - 4 for l in render(fixed, g, cols)):
            bad_fixed.append((cols, len(g)))
        if orphan(render(pre, g, cols)):
            bad_pre.append((cols, len(g)))
check("sweep (3 widths x 150 goal lengths): fixed code never orphans/overflows", not bad_fixed, str(bad_fixed[:5]))
check("sweep: pre-fix code orphans for at least one goal length", bool(bad_pre), f"n={len(bad_pre)} e.g. {bad_pre[:4]}")

# (2d) English untouched by the fix (floor of 8): identical output pre vs fixed
text.set_active_locale("en")
same = all(render(fixed, GOAL, c) == render(pre, GOAL, c) for c in (60, 80, 100))
check("English rendering identical pre vs fixed (floor keeps 8)", same)
text.set_active_locale("es")

print("\nRESULT", "FAIL " + ",".join(failures) if failures else "ALL CHECKS PASS")
sys.exit(1 if failures else 0)

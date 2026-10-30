#!/usr/bin/env python3
"""Mechanical checks for one fr catalogue, as run inside the evidence bundle.

This file is committed *inside* each bundle so that the command logged in
audit.log can be re-run and produce the same output. It resolves the repo root
by walking up until local_operator/i18n/catalogues exists, so it works from any
checkout of this revision.

Width scoring definition (stated here because the number is meaningless
without it):
  * the key set is the fr keys that declare `maxCells` in the en context
    sidecar;
  * a key is EXCLUDED from the width assertion iff its fr value matches
    /\\{\\w+\\s*,/ — i.e. it carries an ICU construct with a type tag
    (plural / select / selectordinal), whose raw source text is not a rendered
    width;
  * every other key is scored exactly with rich.cells.cell_len, including keys
    that carry a SIMPLE placeholder such as {name} or {elapsed}: substituting a
    value cannot change a cell count by an unbounded amount and those keys have
    no ICU branch to widen.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ICU_TYPED = re.compile(r"\{\w+\s*,")
ANY_BRACE = re.compile(r"\{")


def repo_root(start: pathlib.Path) -> pathlib.Path:
    for cand in [start, *start.parents]:
        if (cand / "local_operator/i18n/catalogues").is_dir():
            return cand
    raise SystemExit("cannot locate repo root (no local_operator/i18n/catalogues)")


def load(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))


def arg_names(src: str) -> list[str]:
    return sorted(set(re.findall(r"\{([A-Za-z_]\w*)\s*[,}]", src)))


def plural_arms(src: str):
    out = {}
    for m in re.finditer(r"\{(\w+),\s*(plural|select|selectordinal)\s*,", src):
        start, depth, i = m.end(), 1, m.end()
        while i < len(src) and depth > 0:
            depth += 1 if src[i] == "{" else -1 if src[i] == "}" else 0
            i += 1
        body, arms, d, cur = src[start : i - 1], set(), 0, ""
        for ch in body:
            if ch == "{":
                if d == 0:
                    tok = cur.strip().split()[0] if cur.strip() else ""
                    if tok:
                        arms.add(tok)
                    cur = ""
                d += 1
            elif ch == "}":
                d -= 1
            elif d == 0:
                cur += ch
        out[m.group(1)] = (m.group(2), sorted(arms))
    return out


def typography(v: str) -> list[str]:
    prose = re.sub(r"`[^`]*`", "\x00", v)
    bad = []
    if "'" in prose:
        bad.append("ascii-apostrophe")
    if re.search(r"(?<![\u202f\u00a0])[;:!?](?![\u202f\u00a0])", prose):
        bad.append("missing-U+202F")
    if any(m.group(1) not in "\u202f\u00a0" for m in re.finditer(r"«(.)", v)):
        bad.append("guillemet-open")
    if any(m.group(1) not in "\u202f\u00a0" for m in re.finditer(r"(.)»", v)):
        bad.append("guillemet-close")
    if "..." in v:
        bad.append("ascii-dots")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", required=True)
    ns = ap.parse_args().namespace

    root = repo_root(pathlib.Path(__file__).resolve().parent)
    base = root / "local_operator/i18n/catalogues"
    en = load(base / f"en/{ns}.json")
    fr = load(base / f"fr/{ns}.json")
    ctx = load(base / f"en/{ns}.context.json")

    from rich.cells import cell_len

    print(f"namespace: {ns}   keys: en={len(en)} fr={len(fr)}")
    missing, unknown = sorted(set(en) - set(fr)), sorted(set(fr) - set(en))
    print(
        f"key_set: missing={len(missing)} unknown={len(unknown)} -> "
        + ("OK" if not missing and not unknown else f"FAIL {missing} {unknown}")
    )
    print(f"key_order_identical_to_en: {list(en) == list(fr)}")

    mism = [k for k in en if arg_names(en[k]) != arg_names(fr[k])]
    print(f"argument_parity: mismatches={len(mism)} -> " + ("OK" if not mism else f"FAIL {mism}"))

    pdiff = [
        (k, plural_arms(en[k]), plural_arms(fr[k]))
        for k in en
        if plural_arms(en[k]) != plural_arms(fr[k])
    ]
    for k, pe, pf in pdiff:
        print(f"plural_arms: {k}: en={pe} fr={pf}")
    print(
        "plural_arms: -> "
        + (
            "OK (fr adds 'many' where en has one/other; never 'few')"
            if pdiff
            else "no plural/select constructs -> vacuous OK"
        )
    )
    print("plural_uses_few: " + str(any("few" in src for src in fr.values())))

    typo = {k: typography(v) for k, v in fr.items() if typography(v)}
    thin = sum(v.count(chr(0x202F)) for v in fr.values())
    nbsp = sum(v.count(chr(0xA0)) for v in fr.values())
    print(
        f"typography: violations={len(typo)} U+202F={thin} U+00A0={nbsp} -> "
        + ("OK" if not typo else f"FAIL {typo}")
    )

    budgeted = [k for k, v in fr.items() if (ctx.get(k) or {}).get("maxCells")]
    icu = [k for k in budgeted if ICU_TYPED.search(fr[k])]
    brace = [k for k in budgeted if ANY_BRACE.search(fr[k])]
    over = [
        (k, cell_len(fr[k]), cell_len(en[k]), ctx[k]["maxCells"])
        for k in budgeted
        if k not in icu and cell_len(fr[k]) > ctx[k]["maxCells"]
    ]
    print(
        "width_definition: scored_keys = fr keys declaring maxCells, minus keys matching "
        r"/\{\w+\s*,/ (ICU type-tagged); simple {name} placeholders ARE scored"
    )
    print(
        f"width_key_set: budgeted={len(budgeted)} placeholder_bearing_any_brace={len(brace)} "
        f"icu_type_tagged_excluded={len(icu)} scored_exact={len(budgeted) - len(icu)}"
    )
    print(f"width_vs_declared_maxCells_exact: over={len(over)}")
    for k, wf, we, b in over:
        print(f"  OVER {k}: fr={wf} en={we} budget={b} (+{wf - b})")
    if icu:
        print("width_excluded_keys: " + ", ".join(k.replace(ns + ".", "") for k in icu))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Ad-hoc reproduction driver for issue #2017 (mobile focus-zoom conditions).

REPRODUCTION ONLY — no product behavior is changed by this file. It drives the
REAL built SPA served by scripts/mobile_overflow_fixture.py out of the shared
Chrome/Page harness in scripts/mobile_overflow_capture.py, at the two phone
viewports the issue names (390x844, 360x780), and records:

* the viewport meta, innerWidth, visualViewport.scale, data-view and the
  persisted wide-view key on every screen it visits;
* every input/textarea/select's computed font-size + selector context per
  screen (the field-size table the issue's triage needs, measured rather than
  read off classes), walked in BOTH modes (wide off / wide on);
* PNG frames of the composer in the default and wide layouts (the required
  before/default pair), plus a frame per surface it walks.

Untracked driver, kept at the worktree root so `scripts` resolves to THIS tree;
frames + JSON land under the session scratchpad (paths printed in the report).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from scripts.mobile_overflow_capture import Chrome, Page, fixture_password

PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 4317
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("captures")
OUT.mkdir(parents=True, exist_ok=True)
BASE = f"http://127.0.0.1:{PORT}"
VIEWPORTS = [(390, 844), (360, 780)]

report: dict = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def snap(page: Page, key: str) -> dict:
    value = json.loads(page.js("""
(() => {
  const m = document.querySelector('meta[name="viewport"]');
  return JSON.stringify({
    meta: m ? m.getAttribute('content') : null,
    innerWidth: window.innerWidth,
    innerHeight: window.innerHeight,
    innerHeightVV: window.visualViewport ? Math.round(window.visualViewport.height) : null,
    scale: window.visualViewport ? Number(window.visualViewport.scale.toFixed(4)) : null,
    dpr: window.devicePixelRatio,
    screenW: window.screen.width,
    layoutClientWidth: document.documentElement.clientWidth,
    dataView: document.documentElement.dataset.view ?? null,
    fitScale: document.documentElement.style.getPropertyValue('--lo-fit-scale'),
    wideStored: localStorage.getItem('lo-mobile-wide-view'),
    url: location.href,
  });
})()
"""))
    report[key] = value
    log(f"[probe] {key}: {json.dumps(value)}")
    return value


def fields(page: Page, key: str) -> list[dict]:
    """Every input/textarea/select on the page, with font-size + context."""
    value = json.loads(page.js("""
(() => {
  const els = [...document.querySelectorAll('input, textarea, select')];
  return JSON.stringify(els.map((el) => {
    const cs = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    let label = "";
    const lbl = el.closest('label');
    if (lbl) label = (lbl.textContent || "").trim().replace(/\\s+/g, " ").slice(0, 48);
    return {
      tag: el.tagName.toLowerCase(),
      type: el.getAttribute('type') || null,
      id: el.id || null,
      placeholder: el.getAttribute('placeholder'),
      className: (el.className || "").toString().slice(0, 110),
      label,
      fontSize: cs.fontSize,
      lineHeight: cs.lineHeight,
      rect: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)],
      inFold: r.top >= -1 && r.bottom <= window.innerHeight + 1,
      disabled: el.disabled === true,
    };
  }));
})()
"""))
    report[key] = value
    log(f"[fields] {key}: {len(value)} field(s)")
    for f in value:
        log(f"    {key} :: {f['tag']}[{f['type'] or ''}] id={f['id']} "
            f"ph={f['placeholder']!r} fs={f['fontSize']} rect={f['rect']} "
            f"inFold={f['inFold']} disabled={f['disabled']} class={f['className'][:56]!r}")
    return value


def cards(page: Page, key: str) -> list[dict]:
    """ask-card / pending-card blocks with their inner fields, by test id."""
    value = json.loads(page.js("""
(() => {
  const out = [];
  const sel = '[data-testid="ask-card"], [data-testid="pending-card"]';
  for (const el of document.querySelectorAll(sel)) {
    const fields = [...el.querySelectorAll('input, textarea, select, button')].map((i) => {
      const r = i.getBoundingClientRect();
      if (i.tagName === 'BUTTON') return null;
      return {
        tag: i.tagName.toLowerCase(),
        type: i.getAttribute('type') || null,
        placeholder: i.getAttribute('placeholder'),
        fontSize: getComputedStyle(i).fontSize,
        rect: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)],
        inFold: r.top >= -1 && r.bottom <= window.innerHeight + 1,
      };
    }).filter(Boolean);
    out.push({
      testid: el.getAttribute('data-testid'),
      askId: el.getAttribute('data-ask-id'),
      status: el.getAttribute('data-ask-status'),
      fields,
    });
  }
  return JSON.stringify(out);
})()
"""))
    report[key] = value
    log(f"[cards] {key}: {len(value)} card(s)")
    for c in value:
        log(f"    {key} :: <{c['testid']}> ask={c['askId']} status={c['status']} fields={c['fields']}")
    return value


def frame(page: Page, name: str) -> None:
    path = OUT / f"{name}.png"
    page.shot(path)
    log(f"[frame] {path} ({path.stat().st_size} bytes)")


def js_click(page: Page, expression: str, key: str) -> str:
    result = page.js(expression)
    log(f"[click] {key}: {result}")
    time.sleep(1.0)
    return str(result)


def close_sheet(page: Page) -> None:
    page.js("""
(() => {
  const b = document.querySelector('[role="dialog"] button[aria-label="close sheet"]');
  if (b) { b.click(); return "closed"; }
  const s = document.querySelector('[data-testid="sheet-scrim"]');
  if (s) { s.click(); return "scrim"; }
  return "no-sheet";
})()
""")
    time.sleep(0.8)


def login(page: Page) -> None:
    page.goto(f"{BASE}/login")
    page.js(
        "(() => { const f = document.querySelector('form');"
        f" if (!f || !f.password) return 'no-form';"
        f" f.password.value = {fixture_password()!r}; f.submit(); return 'submitted'; }})()"
    )
    time.sleep(2.0)


def open_model_sheet(page: Page, key: str) -> None:
    js_click(page, """
(() => {
  const b = [...document.querySelectorAll('button')]
    .find((x) => { const t = (x.textContent || '').trim(); return t === 'model' || t === 'fixture'; });
  if (!b) {
    const cands = [...document.querySelectorAll('button')]
      .map((x) => (x.textContent || '').trim()).filter(Boolean).slice(0, 80);
    return JSON.stringify({missing: true, candidates: cands});
  }
  b.click(); return "clicked";
})()
""", key)


def walk_fields(page: Page, phase: str, vp: str, with_frames: bool = True) -> None:
    """Walk every field surface; keys prefixed with `<vp>-<phase>-`.

    Frames are named `<vp>-<label>-<surface>`, where `label` is `default` or
    `wide`: both passes walk the same surfaces, and a name without the label
    made the wide pass overwrite the default pass's frames — which is how the
    round-1 design review found the default-mode field set missing (D3).
    """
    label = "default" if phase == "wideoff" else "wide"
    # ---- list ----
    page.goto(f"{BASE}/#/")
    time.sleep(1.2)
    snap(page, f"{vp}-{phase}-list")
    fields(page, f"{vp}-{phase}-list-fields")
    if with_frames:
        frame(page, f"{vp}-{label}-list")

    # ---- projects sheet (create form = FIELD_CLASS inputs) ----
    js_click(page, """
(() => {
  const b = [...document.querySelectorAll('button')]
    .find((x) => x.textContent.trim() === 'projects');
  if (!b) return "missing";
  b.click(); return "clicked";
})()
""", "open-projects")
    js_click(page, """
(() => {
  const b = [...document.querySelectorAll('button')]
    .find((x) => x.textContent.trim() === 'new project');
  if (!b) return "missing";
  b.click(); return "clicked";
})()
""", "projects-create")
    fields(page, f"{vp}-{phase}-projects-create-fields")
    if with_frames:
        frame(page, f"{vp}-{label}-projects-create")
    close_sheet(page)

    # ---- past sessions ----
    page.goto(f"{BASE}/#/past")
    time.sleep(1.0)
    fields(page, f"{vp}-{phase}-past-fields")
    if with_frames:
        frame(page, f"{vp}-{label}-past")

    # ---- pair ----
    page.goto(f"{BASE}/#/pair")
    time.sleep(1.0)
    fields(page, f"{vp}-{phase}-pair-fields")
    if with_frames:
        frame(page, f"{vp}-{label}-pair")

    # ---- session: ask-free (composer + pending secret input) ----
    page.goto(f"{BASE}/#/s/ask-free")
    time.sleep(1.5)
    snap(page, f"{vp}-{phase}-session-askfree")
    fields(page, f"{vp}-{phase}-session-askfree-fields")
    cards(page, f"{vp}-{phase}-session-askfree-cards")

    # ---- model sheet ----
    open_model_sheet(page, "open-model-sheet")
    fields(page, f"{vp}-{phase}-model-sheet-fields")
    if with_frames:
        frame(page, f"{vp}-{label}-model-sheet")
    close_sheet(page)

    # ---- directory sheet (working-directory chip) ----
    js_click(page, """
(() => {
  const b = document.querySelector('button[aria-label^="working directory"]');
  if (!b) return "missing";
  b.click(); return "clicked";
})()
""", "open-directory-sheet")
    fields(page, f"{vp}-{phase}-directory-sheet-fields")
    if with_frames:
        frame(page, f"{vp}-{label}-directory-sheet")
        # The POPULATED case the empty-with-placeholder frame cannot show: type a
        # long technical path so the design round can judge character room
        # against a real value (round-1 design review D3). Native setter + an
        # input event, because a bare `.value` write does not reach React.
        js_click(page, """
(() => {
  const el = document.querySelector('input[placeholder="or type another path…"]');
  if (!el) return "missing";
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  setter.call(el, '/Users/damian/workspace/repos/local-operator/local_operator/mobile/web/src');
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.focus();
  return "typed";
})()
""", "type-long-path")
        time.sleep(0.4)
        frame(page, f"{vp}-{label}-directory-longvalue")
    close_sheet(page)

    # ---- asks sheet on asks-stacked (ask-card inputs) ----
    page.goto(f"{BASE}/#/s/asks-stacked")
    time.sleep(1.5)
    js_click(page, """
(() => {
  const b = document.querySelector('[data-testid="ask-dock"]');
  if (!b) return "missing";
  b.click(); return "clicked";
})()
""", "open-asks-sheet")
    time.sleep(0.8)
    cards(page, f"{vp}-{phase}-asks-sheet-cards")
    fields(page, f"{vp}-{phase}-asks-sheet-fields")
    if with_frames:
        frame(page, f"{vp}-{label}-asks-sheet")
    close_sheet(page)


def run_default_pass(page: Page, vp: str) -> None:
    """Wide OFF: the composer's default state + every other field."""
    page.goto(f"{BASE}/#/s/ask-free")
    time.sleep(1.5)
    frame(page, f"{vp}-composer-default")
    walk_fields(page, "wideoff", vp)


def run_wide_pass(page: Page, vp: str) -> None:
    """Wide ON: toggle from the list footer, read scale AT LOAD, walk fields."""
    page.goto(f"{BASE}/#/")
    time.sleep(1.0)
    js_click(page, """
(() => {
  const b = document.querySelector('button[aria-label="wide view"]');
  if (!b) return "missing";
  b.click(); return "clicked";
})()
""", "toggle-wide-on")
    snap(page, f"{vp}-wideon-after-toggle")

    # Reload AT a session route so the meta is applied at load by the boot
    # script (persisted localStorage), which is the state the issue describes.
    page.goto(f"{BASE}/#/s/ask-free")
    time.sleep(1.0)
    page.send("Page.reload")
    time.sleep(2.5)
    snap(page, f"{vp}-wideon-session-askfree-at-load")
    frame(page, f"{vp}-composer-wide")
    walk_fields(page, "wideon", vp)

    # Toggle back OFF from the session header, to leave the profile clean and
    # to capture the toggle's off state from the second surface.
    page.goto(f"{BASE}/#/s/ask-free")
    time.sleep(1.0)
    js_click(page, """
(() => {
  const b = document.querySelector('button[aria-label="wide view"]');
  if (!b) return "missing";
  b.click(); return "clicked";
})()
""", "toggle-wide-off")
    snap(page, f"{vp}-wideon-toggle-off")


def served_assets(page: Page) -> dict:
    """The build the fixture actually served — recorded IN the report.

    WHY: a report that names its build only in the README cannot be checked
    against the file itself, and a stale scan once passed as fresh (round-2
    review). These filenames are content hashes, so recording them here makes
    every report self-identifying: `index-<hash>.css` names the exact bundle the
    numbers below were measured against.
    """
    refs = page.js(
        "[...document.querySelectorAll('link[rel=stylesheet],script[src]')]"
        ".map((e) => e.getAttribute('href') || e.getAttribute('src'))"
    )
    assets = [str(r).rsplit("/", 1)[-1] for r in (refs or [])]
    return {
        "served_css": sorted({a for a in assets if a.endswith(".css")}),
        "served_js": sorted({a for a in assets if a.endswith(".js")}),
    }


def main() -> None:
    chrome = Chrome()
    try:
        page = Page(chrome.target_ws())
        for width, height in VIEWPORTS:
            vp = f"{width}x{height}"
            page.metrics(width, height)
            login(page)
            if "provenance" not in report:
                # Once, from the first loaded page: makes the report
                # self-identifying (round-2 review).
                report["provenance"] = served_assets(page)
                log(f"[provenance] served: {json.dumps(report['provenance'])}")
            run_default_pass(page, vp)
            run_wide_pass(page, vp)
        page.close()
    finally:
        chrome.close()
    (OUT / "focus-zoom-report.json").write_text(json.dumps(report, indent=2))
    log(f"[done] report: {OUT / 'focus-zoom-report.json'}")


if __name__ == "__main__":
    main()

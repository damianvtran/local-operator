/* Behavioural coverage for the popup's PAIRING-FLOW STABILITY.
 *
 * A user reported the pairing experience as "jittery". Every defect behind that
 * word lives in render() sequencing or in DOM side effects, not in a predicate:
 * which section ships visible, whether a re-render steals focus, whether two
 * concurrent renders paint in order, and whether the caret survives sanitising.
 * A predicate matrix structurally cannot fail on any of them — the same reason
 * popup-render.integration.test.mjs exists — so this file drives the real
 * module the same way, with only chrome.* and the DOM stubbed.
 *
 * Every test here was confirmed RED against the pre-fix bundle and green after,
 * so each one can actually go red on the defect it names.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { build } from "esbuild";
import { mkdtemp, rm, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

/** The ids popup.ts touches, kept as a literal list for the same reason the
 * sibling suite does: a test that silently stops covering an element is worse
 * than one that fails when an id is renamed. */
const IDS = [
  "connected", "paired", "pairing", "disconnected", "incompatible", "origin", "origin-ack",
  "pending", "origin-host", "origin-again", "origin-scope", "origin-scope-detail",
  "origin-position", "origin-waiting", "origin-allow", "origin-deny", "origin-previous",
  "origin-next", "origin-ack-title", "origin-ack-sub", "origin-ack-check", "card", "retry",
  "retry-incompatible", "connected-all-sites", "connected-all-sites-off", "pair-form",
  "pair-code", "pair-error", "port", "port-row", "connected-label", "connected-detail",
];

/** A DOM stub that models the two behaviours these defects live in: real focus
 * (so a stolen focus is observable) and a real caret (so a value assignment
 * collapsing the selection is observable). */
function installDomStub() {
  const nodes = new Map();
  const doc = { activeElement: null };
  const make = (id) => {
    const node = {
      id,
      tagName: id === "pair-form" ? "FORM" : "DIV",
      textContent: "",
      hidden: false,
      disabled: false,
      children: [],
      dataset: {},
      _value: "",
      selectionStart: 0,
      selectionEnd: 0,
      _classes: new Set(id === "origin-again" ? ["again", "hidden"] : ["state"]),
      classList: {
        add: (c) => node._classes.add(c),
        remove: (c) => node._classes.delete(c),
        contains: (c) => node._classes.has(c),
        toggle: (c, force) => {
          const on = force === undefined ? !node._classes.has(c) : force;
          if (on) node._classes.add(c);
          else node._classes.delete(c);
          return on;
        },
      },
      style: { setProperty: () => {}, removeProperty: () => {} },
      setAttribute: () => {},
      removeAttribute: () => {},
      addEventListener: (event, handler) => {
        (node._handlers[event] ||= []).push(handler);
      },
      // Real focus bookkeeping: this is what J2 is about.
      //
      // A DISABLED element refuses focus — the actual DOM rule, and the one the
      // pairing form's unlock ordering depends on. Modelling it matters: with
      // focus granted unconditionally, a `select()` called while the input is
      // still disabled looks identical to one called after the unlock, and the
      // U8 defect (select() before setPairBusy(false) in the catch branch, so
      // the corrected code is still swallowed) is invisible to every assertion.
      focus: () => {
        if (node.disabled) return;
        doc.activeElement = node;
        node.focusCount++;
      },
      focusCount: 0,
      setSelectionRange: (s, e) => {
        node.selectionStart = s;
        node.selectionEnd = e;
      },
      // A real input's select() focuses and selects the whole value — and, like
      // focus(), gets nothing on a disabled element: the range is set on
      // something that is not focused, so the next keystroke goes to the body.
      // That asymmetry IS U8, so it has to be modelled rather than assumed.
      select: () => {
        node.selectionStart = 0;
        node.selectionEnd = node._value.length;
        if (node.disabled) return;
        doc.activeElement = node;
        node.focusCount++;
      },
      // popup.ts uses form.contains(document.activeElement) to avoid stealing a
      // focus the user placed inside the form themselves.
      contains: (other) => other != null && (other === node || node._contains.has(other)),
      _contains: new Set(),
      replaceChildren: (...kids) => {
        node.children = kids;
        node.value = kids[0]?.value ?? "";
      },
      querySelectorAll: () => [],
      _handlers: {},
      click: () => (node._handlers.click || []).forEach((h) => h()),
      dispatch: (event) =>
        (node._handlers[event] || []).forEach((h) => h({ preventDefault() {}, stopPropagation() {} })),
    };
    // A real <input> collapses the selection to the end when `.value` is
    // assigned a DIFFERENT string, and short-circuits an identical one. That
    // asymmetry IS the caret defect, so it has to be modelled, not assumed.
    Object.defineProperty(node, "value", {
      get: () => node._value,
      set: (v) => {
        const next = String(v);
        if (next === node._value) return; // Chrome's identical-value short circuit.
        node._value = next;
        node.selectionStart = next.length;
        node.selectionEnd = next.length;
      },
    });
    Object.defineProperty(node, "options", { get: () => node.children });
    Object.defineProperty(node, "selectedOptions", {
      get: () => node.children.filter((c) => c.value === node.value),
    });
    return node;
  };
  for (const id of IDS) nodes.set(id, make(id));
  // The pairing form really does contain the code input and the submit button.
  nodes.get("pair-form")._contains.add(nodes.get("pair-code"));

  const submit = make("pair-submit");
  submit.tagName = "BUTTON";
  nodes.set("pair-submit", submit);
  nodes.get("pair-form")._contains.add(submit);

  globalThis.document = {
    getElementById: (id) => nodes.get(id) ?? null,
    createElement: () => make("option"),
    querySelectorAll: () => [],
    querySelector: (sel) => (sel.includes("submit") ? submit : null),
    addEventListener: () => {},
    documentElement: make("html"),
    body: make("body"),
    get activeElement() {
      return doc.activeElement;
    },
    set activeElement(v) {
      doc.activeElement = v;
    },
  };
  globalThis.window = {
    close: () => {},
    matchMedia: () => ({ matches: false, addEventListener: () => {} }),
  };
  // Synchronous storage, which is what lets the first paint size itself before
  // any await. Persisted across installDomStub() calls within a test so the
  // "hint survives to the next open" property is expressible.
  if (!globalThis.localStorage) {
    const store = new Map();
    globalThis.localStorage = {
      getItem: (k) => (store.has(k) ? store.get(k) : null),
      setItem: (k, v) => store.set(k, String(v)),
      removeItem: (k) => store.delete(k),
      clear: () => store.clear(),
    };
  }
  return nodes;
}

function installChromeStub() {
  const areas = { session: new Map(), local: new Map() };
  const listeners = [];
  const sent = [];
  const makeArea = (name) => ({
    get: async (keys) => {
      const out = {};
      for (const key of Array.isArray(keys) ? keys : [keys]) {
        if (areas[name].has(key)) out[key] = areas[name].get(key);
      }
      return out;
    },
    set: async (obj) => {
      const changes = {};
      for (const [key, value] of Object.entries(obj)) {
        changes[key] = { oldValue: areas[name].get(key), newValue: value };
        areas[name].set(key, value);
      }
      for (const listener of listeners) listener(changes, name);
    },
    remove: async (keys) => {
      const changes = {};
      for (const key of Array.isArray(keys) ? keys : [keys]) {
        changes[key] = { oldValue: areas[name].get(key), newValue: undefined };
        areas[name].delete(key);
      }
      for (const listener of listeners) listener(changes, name);
    },
  });
  globalThis.chrome = {
    storage: {
      session: makeArea("session"),
      local: makeArea("local"),
      onChanged: { addListener: (fn) => listeners.push(fn) },
    },
    runtime: {
      sendMessage: async (message) => {
        sent.push(message);
        return { applied: true };
      },
      openOptionsPage: () => {},
      getManifest: () => ({ version: "0.1.9" }),
    },
    tabs: { query: async () => [] },
  };
  return { areas, sent };
}

const tick = (n = 6) => new Promise((r) => setTimeout(r, n));

async function loadPopup() {
  const dir = await mkdtemp(join(tmpdir(), "lop-popup-jitter-"));
  const outfile = join(dir, "popup.mjs");
  await build({
    entryPoints: [join(HERE, "..", "src", "popup", "popup.ts")],
    bundle: true,
    platform: "node",
    format: "esm",
    outfile,
  });
  return {
    import: () => import(pathToFileURL(outfile) + `?t=${Math.random()}`),
    close: () => rm(dir, { recursive: true, force: true }),
  };
}

/** A reachable daemon whose /health answer and LATENCY are both controllable.
 * Latency is what puts two renders in flight at once, which is the whole of
 * the ordering defect. */
function installFetchStub(paired, delayMs = () => 0) {
  globalThis.fetch = async () => {
    const wait = delayMs();
    // The answer is snapshotted when the request STARTS, exactly as a real
    // /health round trip is: the daemon's reply describes the moment it was
    // asked, not the moment it arrives. Reading the flag at resolve time
    // instead makes a slow probe silently return the newest state, which is
    // precisely the stale answer this test needs to exist.
    const snapshot = paired();
    if (wait) await new Promise((r) => setTimeout(r, wait));
    return {
      ok: true,
      json: async () => ({
        paired: snapshot,
        extension_connected: true,
        protocol_version: 1,
        pending_origin: undefined,
      }),
    };
  };
}

/* ------------------------------------------------------------------ J1 ---- */

test("no diagnostic state ships visible: the first paint is neutral (J1)", async () => {
  // This one is about the SHIPPED MARKUP, which is what the browser paints
  // before render()'s awaits resolve. Asserted against popup.html itself
  // because that file, not the module, decides the pre-render frame.
  const html = await readFile(join(HERE, "..", "src", "popup", "popup.html"), "utf8");
  const visible = [...html.matchAll(/<section\s+id="([\w-]+)"\s+class="state([^"]*)"/g)]
    .filter((m) => !m[2].includes("hidden"))
    .map((m) => m[1]);

  assert.deepEqual(
    visible,
    ["pending"],
    "exactly one section may ship visible, and it must be the neutral placeholder",
  );
  // The specific regression: the error card was the pre-render default, so
  // every popup open flashed "Not connected." for the length of the /health
  // round trip.
  assert.ok(
    !visible.includes("disconnected"),
    "the disconnected card must never be the pre-render default",
  );
  // The placeholder must not diagnose anything, or it is the same defect with
  // different copy. Checked against the RENDERED text only — HTML comments
  // legitimately quote the other states' copy to explain the rule.
  const pending = html.slice(html.indexOf('<section id="pending"'));
  const body = pending
    .slice(0, pending.indexOf("</section>"))
    .replace(/<!--[\s\S]*?-->/g, "");
  assert.doesNotMatch(body, /not reachable|isn't reachable|Not connected/i,
    "the placeholder must not assert a connection verdict it has not established");
});

/* ------------------------------------------------------------------ J7 ---- */

test("the pairing card holds one height across placeholder, form and error (J1/J7)", async () => {
  // A Chrome action popup auto-sizes to its content, so any height difference
  // between these is a visible resize of the popup WINDOW — the "jittery"
  // report. The stylesheet decides it, so it is asserted here.
  //
  // These assertions COMPARE the captured values. An earlier version only
  // checked that the regex matched, which meant `min-height: 1px` passed — the
  // exact blindness that let a 188px first-paint regression through review to
  // be caught by looking at a frame instead.
  const css = await readFile(join(HERE, "..", "src", "popup", "popup.css"), "utf8");

  // The placeholder's fallback pin. popup.ts overrides this per user at first
  // paint (see the paired-hint test below); this is what a browser with no
  // usable localStorage gets, so it must be the unpaired/pairing-form height.
  const pendingMin = /#pending\s*\{[^}]*min-height:\s*(\d+)px/.exec(css);
  assert.ok(pendingMin, "the placeholder must pin a height, or the card resizes when render() settles");
  assert.equal(
    Number(pendingMin[1]),
    219,
    "the fallback pin must be the pairing form's height (measured 219px -> card 340px at 300x600)",
  );

  // The error's reserved slot. 36px is two lines at 12px/1.5 — the height the
  // longest message the daemon can send occupies at 300px.
  const errorMin = /#pair-error\s*\{[^}]*min-height:\s*(\d+)px/.exec(css);
  assert.ok(errorMin, "the pairing error must reserve its space rather than reflow the card");
  assert.equal(
    Number(errorMin[1]),
    36,
    "the error slot must reserve exactly two lines; more is height every pairing card pays for nothing",
  );

  // Reserved space is only reserved if the hidden line still occupies it.
  assert.match(
    css,
    /#pair-error\.hidden\s*\{[^}]*visibility:\s*hidden/,
    "the hidden error must stay in flow (visibility), not be removed from it (display:none)",
  );
  // ...and it must not become permanently invisible: role="alert" only
  // announces a message the AT can reach.
  assert.match(
    css,
    /#pair-error\.hidden\s*\{[^}]*display:\s*block/,
    "the hidden error must override the shared .hidden display:none to keep its box",
  );
});

test("every pairing failure message fits the reserved slot (J7)", async () => {
  // The 36px reservation is calibrated to a WRAP, not to one string, and the
  // slot is permanent — a message that wraps to three lines resizes the card
  // exactly as before the fix. So the binding that matters is between the
  // reservation and the longest message that can land in it.
  //
  // Measured at the card's content width (300px body - 2x16px body padding -
  // 2x1px card border - 2x10px card margin = 246px) in the popup's body face at
  // 12px. Rather than re-deriving pixel metrics here, the invariant asserted is
  // the one that produced 36px: no message exceeds two lines' worth of
  // characters at this width.
  const daemon = await readFile(
    join(HERE, "..", "..", "local_operator", "browser_bridge", "daemon.py"),
    "utf8",
  );
  const flow = await readFile(join(HERE, "..", "src", "popup", "pair-flow.ts"), "utf8");
  const popup = await readFile(join(HERE, "..", "src", "popup", "popup.ts"), "utf8");

  // Python implicit string concatenation means a message can be spelled across
  // several source lines, so adjacent literals are JOINED before measuring.
  // Without this the longest message — the one the reservation is calibrated
  // against — is silently skipped the moment someone wraps it, which is exactly
  // how that string was written when this slot was first sized.
  //
  // Done line-wise rather than with one regex: a run of quoted literals on
  // consecutive lines is the only shape Python's concatenation takes here, and
  // it is far easier to read than a nested-quantifier match.
  const joinAdjacent = (text) => {
    const out = [];
    let current = null;
    for (const line of text.split("\n")) {
      const parts = [...line.matchAll(/"((?:[^"\\]|\\.)*)"/g)].map((m) => m[1]);
      if (parts.length === 1 && /^\s*"/.test(line)) {
        // A bare continuation line: append to the message being built.
        current = current === null ? parts[0] : current + parts[0];
        continue;
      }
      if (current !== null) out.push(current);
      current = null;
      // A line that both opens a message and may continue on the next.
      if (parts.length >= 1 && /=\s*\(?\s*"|message=\s*"|else\s+"/.test(line)) current = parts.at(-1);
      else out.push(...parts);
    }
    if (current !== null) out.push(current);
    return out;
  };

  const messages = [
    ...joinAdjacent(daemon).filter((m) =>
      /^(That code|No live pairing code|Too many attempts)/.test(m),
    ),
    /PAIR_MISMATCH_MESSAGE = "([^"]+)"/.exec(flow)?.[1],
    /error\.textContent = "(Could not reach[^"]*)"/.exec(popup)?.[1],
  ].filter(Boolean);

  assert.ok(messages.length >= 5, `expected the real failure strings, found ${messages.length}`);
  // The daemon's mismatch copy is the string the 36px slot is calibrated to.
  assert.ok(
    messages.some((m) => m.startsWith("That code didn't match")),
    "the daemon's mismatch message must be among the strings measured",
  );

  // Word wrapping, not character division: a line breaks at the last word that
  // fits, so a naive length/width underestimates the line count badly enough to
  // miss the case this test exists for. The old mismatch copy (87 chars) is
  // ceil(87/52) = 2 by division but genuinely wraps to THREE lines, which is
  // the regression that sized this slot at 54px in the first place.
  //
  // 41 characters per line is the measured fit at the card's 246px content
  // width in the popup's 12px body face — calibrated against the two strings
  // whose rendered wrap the designer measured: the 87-char copy at 3 lines and
  // the 55-char replacement at 2.
  const CHARS_PER_LINE = 41;
  const LINES_RESERVED = 2;
  const wrappedLines = (message) => {
    let lines = 1;
    let column = 0;
    for (const word of message.split(" ")) {
      const width = word.length + (column === 0 ? 0 : 1);
      if (column + width > CHARS_PER_LINE) {
        lines++;
        column = word.length;
      } else {
        column += width;
      }
    }
    return lines;
  };
  for (const message of messages) {
    const lines = wrappedLines(message);
    assert.ok(
      lines <= LINES_RESERVED,
      `"${message}" wraps to ${lines} lines but the slot reserves ${LINES_RESERVED}; it would resize the card`,
    );
  }
});

test("the error sits above the button that produced it (D2)", async () => {
  // Placement, not styling: below `.actions` the permanently-reserved slot put
  // a 66px void between the last control and the footer rule, and put the
  // message below the button on failure.
  const html = await readFile(join(HERE, "..", "src", "popup", "popup.html"), "utf8");
  const form = html.slice(html.indexOf('<form id="pair-form">'), html.indexOf("</form>"));
  assert.ok(form.includes('id="pair-error"'), "the error must live inside the pairing form");
  assert.ok(
    form.indexOf('id="pair-error"') < form.indexOf('class="actions"'),
    "the error must come before the actions row, not after it",
  );
});

test("the first paint is pinned to the state this browser will actually reach (D1)", async () => {
  // #pending paints on EVERY open, before render()'s awaits resolve, so its
  // pinned height decides how far the card travels. One pin cannot serve both
  // populations: the pairing form is 360.5px and the connected card 239.9px.
  const nodes = installDomStub();
  const { areas } = installChromeStub();

  // An already-paired browser. The hint is synchronous (localStorage), because
  // chrome.storage cannot inform a first paint.
  globalThis.localStorage.setItem("lop:paired-hint", "1");
  installFetchStub(() => true);
  let bundle = await loadPopup();
  try {
    areas.local.set("port", 4099);
    await bundle.import();
    await tick(30);
    assert.equal(
      nodes.get("pending").style.minHeight,
      "86px",
      "a paired browser's first paint must be pinned to the connected card's height",
    );
    assert.equal(
      nodes.get("connected").classList.contains("hidden"),
      false,
      "precondition: it really does settle on connected",
    );
  } finally {
    await bundle.close();
  }

  // A browser that has never paired.
  const fresh = installDomStub();
  const second = installChromeStub();
  globalThis.localStorage.removeItem("lop:paired-hint");
  installFetchStub(() => false);
  bundle = await loadPopup();
  try {
    second.areas.local.set("port", 4099);
    await bundle.import();
    await tick(30);
    assert.equal(
      fresh.get("pending").style.minHeight,
      "219px",
      "an unpaired browser's first paint must be pinned to the pairing form's height",
    );
  } finally {
    await bundle.close();
  }
});

test("the pin is applied BEFORE the first paint, not by the deferred module (Q4)", async () => {
  // The D1 test below reads the inline style AFTER `await bundle.import()`, so
  // it structurally cannot observe the pre-module paint — it passed while the
  // already-paired user still saw a 340px -> 207px resize on 4 of 12 opens,
  // measured off composited frames. popup.js is `<script type="module">` and
  // therefore DEFERRED: anything it does to layout happens after the compositor
  // may already have painted the stylesheet's default pin.
  //
  // The fix is structural, so this guard is too: it pins the load order that
  // closes the window, which a DOM read after import can never distinguish.
  // Chrome is not available in CI, so the composited-frame measurement stays a
  // manual step (recorded on the PR); this is the part that must not silently
  // regress on a refactor.
  const html = await readFile(join(HERE, "..", "src", "popup", "popup.html"), "utf8");
  const head = html.slice(html.indexOf("<head>"), html.indexOf("</head>"));

  const preScript = /<script(?![^>]*\btype\s*=\s*"module")[^>]*src="first-paint\.js"/.exec(head);
  assert.ok(
    preScript,
    "first-paint.js must load from <head> as a CLASSIC script; as a module it is deferred past first paint",
  );
  // Compared on the TAGS, not on the first mention of each filename: the
  // comment above the script names it too, so an indexOf on the raw text finds
  // the comment and reports the right order however the tags are actually
  // arranged. (Confirmed: that spelling survived a mutation that moved the
  // <link> above the <script>.)
  const stripped = head.replace(/<!--[\s\S]*?-->/g, "");
  assert.ok(
    /<script[^>]*src="first-paint\.js"[\s\S]*<link[^>]*href="popup\.css"/.test(stripped),
    "the pre-paint script must precede the stylesheet: a classic script after a <link> blocks on it loading",
  );
  assert.ok(
    !/<script[^>]*src="popup\.js"/.test(head),
    "the module must not be moved into <head> instead — it stays deferred wherever it is",
  );

  // It must stay import-free. A single import makes it a module, which hands
  // back exactly the deferral it exists to avoid.
  const source = await readFile(join(HERE, "..", "src", "popup", "first-paint.js"), "utf8");
  assert.ok(
    !/^\s*import\s/m.test(source) && !/\brequire\s*\(/.test(source),
    "first-paint.js must have no imports; any import makes it a deferred module",
  );

  // Its pins and key must agree with popup.ts, which owns them. They are
  // duplicated because nothing can be shared with code that runs this early,
  // and a silent divergence is a resize for one of the two populations.
  const popup = await readFile(join(HERE, "..", "src", "popup", "popup.ts"), "utf8");
  const owned = /readPairedHint\(\) \? "(\d+)px" : "(\d+)px"/.exec(popup);
  assert.ok(owned, "popup.ts must still own the measured pins");
  const early = /paired \? PAIRED_PIN : UNPAIRED_PIN/.test(source) && {
    paired: /PAIRED_PIN = "(\d+)px"/.exec(source)?.[1],
    unpaired: /UNPAIRED_PIN = "(\d+)px"/.exec(source)?.[1],
  };
  assert.ok(early, "first-paint.js must choose between a paired and an unpaired pin");
  assert.equal(early.paired, owned[1], "the paired pin must match popup.ts");
  assert.equal(early.unpaired, owned[2], "the unpaired pin must match popup.ts");

  const key = /PAIRED_HINT_KEY = "([^"]+)"/.exec(popup)?.[1];
  assert.ok(source.includes(`"${key}"`), `first-paint.js must read the same key (${key}) popup.ts writes`);

  // And the CSS fallback must be the unpaired pin: it is what a browser whose
  // hint cannot be read at all gets.
  const css = await readFile(join(HERE, "..", "src", "popup", "popup.css"), "utf8");
  const fallback = /#pending\s*\{[^}]*min-height:\s*(\d+)px/.exec(css);
  assert.equal(fallback?.[1], owned[2], "the CSS fallback must be the unpaired pin");

  // The store package is an explicit allowlist; an unlisted file ships a popup
  // whose <head> references a 404 and whose pin is never applied.
  const shipped = await readFile(join(HERE, "..", "store-package-files.txt"), "utf8");
  assert.ok(
    shipped.split("\n").includes("popup/first-paint.js"),
    "first-paint.js must be in the store package allowlist",
  );
});

test("the paired hint follows /health in both directions (D1)", async () => {
  // The hint is a layout guess and must never drift from reality: an unpair has
  // to shrink the next first paint back, or the returning user gets the resize
  // the pin exists to remove.
  const nodes = installDomStub();
  const { areas } = installChromeStub();
  globalThis.localStorage.removeItem("lop:paired-hint");
  let paired = true;
  installFetchStub(() => paired);
  const bundle = await loadPopup();
  try {
    areas.local.set("port", 4099);
    await bundle.import();
    await tick(30);
    assert.equal(globalThis.localStorage.getItem("lop:paired-hint"), "1", "pairing must record the hint");

    paired = false;
    await chrome.storage.session.set({ connState: "pairing" });
    await tick(30);
    assert.equal(
      globalThis.localStorage.getItem("lop:paired-hint"),
      "0",
      "an unpair must clear the hint, or the next first paint is pinned to the wrong state",
    );
  } finally {
    await bundle.close();
  }
});

test("the placeholder is gone once render settles (A4)", async () => {
  // #pending ships visible, so it is hidden only by show()'s toggle over the
  // sections list. Dropping it from that list leaves the placeholder stacked
  // above the real card, which no other assertion observes.
  const nodes = installDomStub();
  const { areas } = installChromeStub();
  installFetchStub(() => true);
  const bundle = await loadPopup();
  try {
    areas.local.set("port", 4099);
    await bundle.import();
    await tick(30);
    assert.equal(
      nodes.get("pending").classList.contains("hidden"),
      true,
      "the placeholder must be hidden once a real state is painted, not left stacked above it",
    );
    const visible = ["connected", "paired", "pairing", "disconnected", "pending"].filter(
      (id) => !nodes.get(id).classList.contains("hidden"),
    );
    assert.deepEqual(visible, ["connected"], "exactly one card may be visible after render settles");
  } finally {
    await bundle.close();
  }
});

/* ------------------------------------------------------------------ A1 ---- */

test("a stalled health probe still reaches a state the user can retry from (A1)", async () => {
  // Renders are serialised, so an unbounded probe does not strand ONE render —
  // renderRunning never clears and renderQueued chains off it, so it wedges
  // every later render permanently. The frozen frame is #pending, which has no
  // Retry button, and both Retry handlers are `() => void render()` and become
  // silently inert.
  const nodes = installDomStub();
  const { areas } = installChromeStub();

  let fetches = 0;
  // A daemon that accepts the connection and never answers. Honours the abort
  // signal the way a real fetch does — rejecting — which is what lets the
  // render finish at all.
  globalThis.fetch = (_url, init) => {
    fetches++;
    return new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
    });
  };

  const bundle = await loadPopup();
  try {
    areas.local.set("port", 4099);
    await bundle.import();
    // Longer than the 3s bound the probe must impose.
    await tick(3600);

    assert.equal(
      nodes.get("disconnected").classList.contains("hidden"),
      false,
      "a stalled probe must land on the retryable disconnected card, not freeze on the placeholder",
    );
    assert.equal(
      nodes.get("pending").classList.contains("hidden"),
      true,
      "the popup must not be parked on a card with no Retry button",
    );

    // ...and the popup must still be LIVE: further triggers must issue new
    // probes rather than queue behind a render that never finished.
    const before = fetches;
    nodes.get("retry").click();
    await tick(60);
    assert.ok(
      fetches > before,
      `Retry must issue a new probe (fetches ${before} -> ${fetches}); a wedged render makes it silently inert`,
    );
  } finally {
    await bundle.close();
  }
});

/* ------------------------------------------------------------------ J2 ---- */

test("a re-render does not yank focus back into the pairing code field (J2)", async () => {
  const nodes = installDomStub();
  const { areas } = installChromeStub();
  installFetchStub(() => false);
  const bundle = await loadPopup();
  try {
    areas.local.set("port", 4099);
    await bundle.import();
    await tick(30);

    const input = nodes.get("pair-code");
    assert.equal(nodes.get("pairing").classList.contains("hidden"), false, "precondition: the form is up");
    assert.equal(document.activeElement, input, "precondition: a newly shown form takes focus once");

    // The user tabs OUTSIDE the form — the ordinary case in the report: they
    // click the port row or the footer while reading the code out of the
    // terminal. This is deliberately not the submit button: the button is
    // inside #pair-form, so `alreadyInForm` would be true and `!pairingShown`
    // would never be the deciding term — a mutation dropping `!pairingShown`
    // survived the earlier version of this test for exactly that reason.
    const outside = nodes.get("port");
    outside.focus();
    const focusesBefore = input.focusCount;

    // While unpaired the worker idle-suspends and the alarm floor rewakes it,
    // so connState is rewritten under the user on every teardown and hello_ack.
    await chrome.storage.session.set({ connState: "connecting" });
    await tick(30);
    await chrome.storage.session.set({ connState: "pairing" });
    await tick(30);

    assert.equal(
      document.activeElement,
      outside,
      "a re-render must not steal focus from a user who has tabbed onward",
    );
    assert.equal(
      input.focusCount,
      focusesBefore,
      "the code field must not be re-focused by a render that changed nothing",
    );
  } finally {
    await bundle.close();
  }
});

test("re-entering the pairing state after leaving it focuses the field again (J2)", async () => {
  // The other half of the rule: suppressing focus entirely would be a
  // regression for the case the focus exists to serve.
  const nodes = installDomStub();
  const { areas } = installChromeStub();
  let paired = true;
  installFetchStub(() => paired);
  const bundle = await loadPopup();
  try {
    areas.local.set("port", 4099);
    await bundle.import();
    await tick(30);
    assert.equal(nodes.get("connected").classList.contains("hidden"), false, "precondition: connected");

    // An unpair drops back to the form: this IS a newly-shown form.
    paired = false;
    await chrome.storage.session.set({ connState: "pairing" });
    await tick(30);

    assert.equal(nodes.get("pairing").classList.contains("hidden"), false, "the form is up again");
    assert.equal(
      document.activeElement,
      nodes.get("pair-code"),
      "a form the user has not seen yet must take focus",
    );
  } finally {
    await bundle.close();
  }
});

test("a render never re-focuses a field the user is already editing (J2)", async () => {
  // The consequence the guard exists to prevent, asserted on the state that
  // makes it expensive: a user mid-edit with a SELECTION. Re-focusing an input
  // collapses the selection to the caret, which would silently undo the
  // select() the failure path performs — so this also protects U1.
  const nodes = installDomStub();
  const { areas } = installChromeStub();
  installFetchStub(() => false);
  const bundle = await loadPopup();
  try {
    areas.local.set("port", 4099);
    await bundle.import();
    await tick(30);

    const input = nodes.get("pair-code");
    // The user is mid-edit with a selection — exactly the state the failure
    // path's select() leaves them in.
    input.focus();
    input._value = "123456";
    input.setSelectionRange(0, 6);
    const focusesBefore = input.focusCount;

    // Force the case the guard's second term exists for: the pairing state is
    // re-shown while focus is genuinely inside the form.
    await chrome.storage.session.set({ connState: "connecting" });
    await tick(30);

    assert.equal(
      input.focusCount,
      focusesBefore,
      "a render must not re-focus a field the user is editing: it collapses the selection to the caret",
    );
    assert.equal(input.selectionStart, 0, "the user's selection must survive the render");
    assert.equal(input.selectionEnd, 6, "the user's selection must survive the render");
  } finally {
    await bundle.close();
  }
});

/** The daemon's side of a pairing submit, over the popup's own socket: a
 * hello_ack followed by a pair_result. `ok:false` is the rejection path. */
function installPairSocket({ ok = false, message = "That code didn't match." } = {}) {
  globalThis.WebSocket = class {
    constructor() {
      queueMicrotask(() => this.onopen?.({}));
    }
    send(raw) {
      const frame = JSON.parse(String(raw));
      if (frame.event === "hello") {
        queueMicrotask(() =>
          this.onmessage?.({ data: JSON.stringify({ event: "hello_ack", proto: 1, paired: false }) }),
        );
      } else if (frame.event === "pair") {
        queueMicrotask(() =>
          this.onmessage?.({
            data: JSON.stringify({ event: "pair_result", ok, message, token: ok ? "tok" : undefined }),
          }),
        );
      }
    }
    close() {}
  };
}

/* ------------------------------------------------------------------ U1 ---- */

test("a rejected code is selected so the next keystroke replaces it (U1)", async () => {
  // maxlength="6" is already satisfied by the rejected digits, so with the
  // caret at 6 every further keystroke AND a paste are silently discarded.
  // Observed costing a user two of five attempts on the same wrong code.
  const nodes = installDomStub();
  const { areas } = installChromeStub();
  installFetchStub(() => false);
  installPairSocket({ ok: false });
  const bundle = await loadPopup();
  try {
    areas.local.set("port", 4099);
    await bundle.import();
    await tick(30);

    const input = nodes.get("pair-code");
    input._value = "707770";
    input.setSelectionRange(6, 6);

    // Same spy as the U8 test on the sibling branch: the unlock must precede
    // the selection here too. Without this the identical ordering defect could
    // be introduced on THIS branch and every assertion below would still pass,
    // because `finally` re-enables the input before they run.
    let enabledWhenSelected = null;
    const realSelect = input.select;
    input.select = () => {
      enabledWhenSelected = !input.disabled;
      realSelect();
    };

    // The daemon rejects it over the popup's own socket.
    nodes.get("pair-form").dispatch("submit");
    await tick(120);

    assert.equal(
      nodes.get("pair-error").classList.contains("hidden"),
      false,
      "precondition: the attempt was rejected",
    );
    assert.equal(
      enabledWhenSelected,
      true,
      "the input must already be re-enabled when the rejected code is selected",
    );
    assert.equal(
      input.selectionStart === 0 && input.selectionEnd === input.value.length,
      true,
      `the rejected code must be selected so typing replaces it (got ${input.selectionStart}-${input.selectionEnd} of "${input.value}")`,
    );
  } finally {
    await bundle.close();
  }
});

test("a failed socket upgrade leaves the field usable for the retry (U8)", async () => {
  // The transport-failure branch, which is NOT reached by killing the daemon:
  // an unreachable daemon renders the `disconnected` card, where this copy and
  // this recovery are off screen entirely. It is live only when /health is
  // REACHABLE but the WebSocket upgrade fails — a refused upgrade, a daemon
  // mid-restart still serving HTTP, a stale worker. So the fetch stub stays
  // healthy and only the socket fails.
  //
  // The assertions are on the OPERATIVE facts — the input is enabled, it is
  // focused, and a subsequently typed code actually reaches the field. Checking
  // that `select()` was called would pass on the broken ordering, which is this
  // PR's recurring failure mode: the guard names the right behaviour and
  // measures the wrong quantity.
  const nodes = installDomStub();
  const { areas } = installChromeStub();
  installFetchStub(() => false);
  // Reachable daemon, refused upgrade: the socket errors after construction
  // rather than opening, which is what lands the handler in `catch`.
  globalThis.WebSocket = class {
    constructor() {
      queueMicrotask(() => this.onerror?.({}));
    }
    send() {}
    close() {}
  };

  const bundle = await loadPopup();
  try {
    areas.local.set("port", 4099);
    await bundle.import();
    await tick(30);

    const input = nodes.get("pair-code");
    input._value = "707770";
    input.setSelectionRange(6, 6);

    // Record the input's state AT THE MOMENT the recovery runs. Sampling after
    // the handler returns cannot see this defect: `finally` re-enables the
    // input, so by then `disabled` is false either way. The ordering is the
    // whole finding, so it has to be observed while it is happening.
    //
    // The browser refuses focus to a disabled element, so the observable
    // consequence is that the selection is applied to something the user is not
    // typing into. This spy records enablement at the instant of selection.
    let enabledWhenSelected = null;
    const realSelect = input.select;
    input.select = () => {
      enabledWhenSelected = !input.disabled;
      realSelect();
    };

    nodes.get("pair-form").dispatch("submit");
    await tick(150);

    assert.equal(
      nodes.get("pair-error").classList.contains("hidden"),
      false,
      "precondition: the transport failure surfaced an error",
    );
    assert.match(
      nodes.get("pair-error").textContent,
      /Could not reach/,
      "precondition: this is the catch branch, not the mismatch branch",
    );

    // The ordering defect, stated directly: `setPairBusy(false)` runs in
    // `finally`, i.e. AFTER the catch body, so a select() placed before it acts
    // on a disabled input and the range lands on an element that cannot be
    // typed into.
    assert.equal(
      enabledWhenSelected,
      true,
      "the input must already be re-enabled when the recovery selects it; " +
        "select() on a disabled input sets a range the user cannot type over",
    );
    assert.equal(input.disabled, false, "the input must end the handler enabled");
    assert.equal(document.activeElement, input, "the input must hold focus for the retry");
    assert.equal(input.selectionStart, 0, "the typed code must be selected");
    assert.equal(input.selectionEnd, 6, "the typed code must be selected");

    // The consequence the user feels. maxlength="6" is already satisfied by the
    // digits in the field, so unless the selection is live the corrected code
    // is silently discarded — the exact defect U6 was filed for.
    const typed = "707776";
    if (document.activeElement === input && input.selectionEnd > input.selectionStart) {
      input._value = typed; // the selection is replaced by what the user types
      input.setSelectionRange(typed.length, typed.length);
    }
    assert.equal(
      input.value,
      typed,
      "a corrected code typed over the selection must land, not be swallowed by maxlength",
    );
  } finally {
    await bundle.close();
  }
});

/* ------------------------------------------------------------------ J3 ---- */

test("concurrent renders never paint an older state over a newer one (J3)", async () => {
  const nodes = installDomStub();
  const { areas } = installChromeStub();
  // Render A sees UNPAIRED and is slow; render B sees PAIRED and is fast. A
  // started first, so unserialised it finishes last and repaints the form over
  // the connected card. Latency varying is the ordinary condition; only its
  // size is amplified here.
  //
  // The delay is keyed on WHAT the probe will observe, not on a call index: a
  // positional counter is consumed by whichever renders happen to run first
  // (module load fires one immediately), so it stops landing on the render this
  // test is about and the race silently never happens.
  let paired = false;
  installFetchStub(
    () => paired,
    () => (paired ? 0 : 120),
  );
  const bundle = await loadPopup();
  try {
    areas.local.set("port", 4099);

    // Record every painted state transition. Only the states render() itself
    // paints are tracked, so the sequence is comparable across trees whose
    // shipped markup differs (the pre-fix tree has no `pending` section).
    const TRACKED = ["connected", "paired", "pairing", "disconnected"];
    const painted = [];
    const watch = () => {
      const shown = TRACKED.filter((id) => !nodes.get(id).classList.contains("hidden")).join(",");
      if (!shown) return; // pre-render, before any show() has run
      if (painted[painted.length - 1] !== shown) painted.push(shown);
    };
    const timer = setInterval(watch, 2);

    // NOT awaited: the module-load render must still be in flight when the
    // storage event fires, or there is only ever one render running and the
    // race this test exists for cannot happen. Awaiting the import here made
    // the test pass against the pre-fix bundle — it was guarding nothing.
    void bundle.import();
    // Long enough for that render to reach its slow /health await, short enough
    // that it has not resolved (the unpaired probe takes 120ms).
    await tick(20);
    // The pairing lands while render A is still awaiting its slow /health.
    paired = true;
    await chrome.storage.session.set({ connState: "connected" });
    await tick(600);
    clearInterval(timer);
    watch();

    // The defect is a state the popup had already LEFT being painted again.
    // Expressed as "no state repeats after a different one intervened", which
    // is the bounce the user sees, rather than as a rank ordering — the set of
    // states a tree can paint differs between trees, the bounce does not.
    const bounces = [];
    for (let i = 2; i < painted.length; i++) {
      if (painted[i] === painted[i - 2] && painted[i] !== painted[i - 1]) {
        bounces.push(`${painted[i - 2]} -> ${painted[i - 1]} -> ${painted[i]}`);
      }
    }
    assert.deepEqual(
      bounces,
      [],
      `a render that started earlier must not repaint a state the popup had left (sequence: ${JSON.stringify(painted)})`,
    );
    assert.equal(
      nodes.get("connected").classList.contains("hidden"),
      false,
      `the settled card must be the newest state (sequence: ${JSON.stringify(painted)})`,
    );
  } finally {
    await bundle.close();
  }
});

/* ------------------------------------------------------------------ J8 ---- */

test("sanitising the code keeps the caret where the edit happened (J8)", async () => {
  const nodes = installDomStub();
  const { areas } = installChromeStub();
  installFetchStub(() => false);
  const bundle = await loadPopup();
  try {
    areas.local.set("port", 4099);
    await bundle.import();
    await tick(30);

    const input = nodes.get("pair-code");

    // Digit-only typing: the value is unchanged, so Chrome's identical-value
    // short circuit already protected this. Confirmed with real keystrokes in
    // Chrome 152; asserted here so a "fix" cannot regress it.
    input._value = "12934";
    input.setSelectionRange(3, 3);
    input.dispatch("input");
    assert.equal(input.value, "12934", "digits are kept verbatim");
    assert.equal(input.selectionStart, 3, "a digit-only edit must not move the caret");

    // A stripped character mid-string: the assignment genuinely changes the
    // value, so before the fix the caret jumped to the end and the next
    // character landed in the wrong place.
    input._value = "12x34";
    input.setSelectionRange(3, 3);
    input.dispatch("input");
    assert.equal(input.value, "1234", "the non-digit is stripped");
    assert.equal(
      input.selectionStart,
      2,
      "the caret must stay at the edit point, not jump to the end of the field",
    );

    // A pasted mixed string: every stripped character before the caret shifts
    // it left by one, and none after it do.
    input._value = "1-2-3-4";
    input.setSelectionRange(7, 7);
    input.dispatch("input");
    assert.equal(input.value, "1234", "separators are stripped");
    assert.equal(input.selectionStart, 4, "the caret follows the surviving digits");
  } finally {
    await bundle.close();
  }
});

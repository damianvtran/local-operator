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
      focus: () => {
        doc.activeElement = node;
        node.focusCount++;
      },
      focusCount: 0,
      setSelectionRange: (s, e) => {
        node.selectionStart = s;
        node.selectionEnd = e;
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
      dispatch: (event) => (node._handlers[event] || []).forEach((h) => h({ preventDefault() {} })),
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
  // different copy.
  const pending = html.slice(html.indexOf('<section id="pending"'));
  const body = pending.slice(0, pending.indexOf("</section>"));
  assert.doesNotMatch(body, /not reachable|isn't reachable|Not connected/i,
    "the placeholder must not assert a connection verdict it has not established");
});

/* ------------------------------------------------------------------ J7 ---- */

test("the pairing card holds one height across placeholder, form and error (J1/J7)", async () => {
  // A Chrome action popup auto-sizes to its content, so any height difference
  // between these three is a visible resize of the popup WINDOW — the "jittery"
  // report. The stylesheet is the only thing that decides it, so it is asserted
  // here rather than through the module.
  //
  // The numbers are measured, not guessed: rendered in Chrome 152 at the
  // popup's true 300x600 metrics against a real daemon, the card is 358px for
  // the placeholder, the pairing form, and the form carrying an error alike.
  // Before the fix the same sequence measured 259 -> 292 -> 358.
  const css = await readFile(join(HERE, "..", "src", "popup", "popup.css"), "utf8");

  const pendingMin = /#pending\s*\{[^}]*min-height:\s*(\d+)px/.exec(css);
  assert.ok(pendingMin, "the placeholder must pin a height, or the card resizes when render() settles");

  const errorMin = /#pair-error\s*\{[^}]*min-height:\s*(\d+)px/.exec(css);
  assert.ok(errorMin, "the pairing error must reserve its space rather than reflow the card");

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

    // The user reads the code out of the terminal and tabs onward. While
    // unpaired the worker idle-suspends and the alarm floor rewakes it, so
    // connState is rewritten under them on every teardown and hello_ack.
    const submit = nodes.get("pair-submit");
    submit.focus();
    const focusesBefore = input.focusCount;

    await chrome.storage.session.set({ connState: "connecting" });
    await tick(30);
    await chrome.storage.session.set({ connState: "pairing" });
    await tick(30);

    assert.equal(
      document.activeElement,
      submit,
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

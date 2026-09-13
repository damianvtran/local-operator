/* Behavioural coverage for popup.ts's RENDER SEQUENCING.
 *
 * Every other test in this suite calls the origin-flow predicates directly
 * with hand-built arguments, or greps popup.ts for call shapes. Five separate
 * defects (A1/U1, A6, U7, U9, U10) have shipped through that coverage,
 * because none of them lives in a predicate: they live in WHICH render
 * consumes a piece of state and whether it survives to the render that needs
 * it. A predicate matrix structurally cannot fail on that, and mutation runs
 * proved it — reverting the U9 fix left the whole suite green.
 *
 * So this file drives the real module. popup.ts wires its own listeners and
 * calls render() at import, exactly as it does in the browser, so a storage
 * write here reaches the same code path a real queue change does. The
 * assertions are on the card the user would see, AFTER the second and third
 * render — because every defect in this class is correct on render N and
 * wrong on N+1.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { build } from "esbuild";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

/** The popup's markup, reduced to the ids and classes popup.ts touches.
 * Kept as a literal id list rather than parsing popup.html: a test that
 * silently stops covering an element because the parse missed it is worse
 * than one that fails when an id is renamed. */
const IDS = [
  "connected", "paired", "pairing", "disconnected", "incompatible", "unresponsive",
  "origin", "origin-ack",
  "origin-host", "origin-again", "origin-scope", "origin-scope-detail", "origin-position",
  "origin-waiting", "origin-allow", "origin-deny", "origin-previous", "origin-next",
  "origin-ack-title", "origin-ack-sub", "origin-ack-check", "origin-ack-granted", "card", "retry",
  "retry-incompatible", "retry-unresponsive", "reload-extension",
  "origin-wedge", "origin-wedge-reload", "unresponsive-outcome",
  "connected-all-sites", "connected-all-sites-off",
  "pair-form",
  "pair-code", "pair-error", "port", "port-row",
];

function installDomStub() {
  const nodes = new Map();
  const make = (id) => {
    const node = {
      id,
      textContent: "",
      value: "",
      hidden: false,
      disabled: false,
      children: [],
      dataset: {},
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
      // Records HOW focus was called. `preventScroll` is the whole D7/U8 fix:
      // focusing inside a scroll container scrolls it, which put the title and
      // the danger banner above the fold at zoom. A stub that ignores the
      // options object cannot tell the fix from its absence.
      focus: (options) => {
        node._focusCalls.push(options ?? null);
        globalThis.document.activeElement = node;
        // The real container scrolls to reveal the focused element unless the
        // caller opts out — modelled so the assertion is on the OUTCOME
        // (where the card sits) rather than on the argument alone.
        const body = nodes.get("__body");
        if (body && !(options && options.preventScroll)) body.scrollTop = body._maxScroll ?? 0;
      },
      replaceChildren: (...kids) => {
        node.children = kids;
        // A real <select> adopts the first option's value on replaceChildren;
        // popup.ts then assigns the preselected scope over it. Modelling this
        // matters: without it the "did the preselect run" assertion could pass
        // on a stale value from the previous build.
        node.value = kids[0]?.value ?? "";
      },
      querySelectorAll: () => [],
      _handlers: {},
      _focusCalls: [],
      scrollTop: 0,
      click: () => (node._handlers.click || []).forEach((h) => h()),
    };
    // The scope select reports its options the way popup.ts reads them.
    Object.defineProperty(node, "options", { get: () => node.children });
    Object.defineProperty(node, "selectedOptions", {
      get: () => node.children.filter((c) => c.value === node.value),
    });
    return node;
  };
  for (const id of IDS) nodes.set(id, make(id));
  // The scroll container the bounded card introduced. `_maxScroll` stands in
  // for a card taller than the viewport, which is the only condition under
  // which focus-scrolling is observable at all.
  const bodyNode = make("__body");
  bodyNode._maxScroll = 139;
  nodes.set("__body", bodyNode);
  // Button labels the real markup ships with. popup.ts restores a label it
  // swapped out (the "Checking…" in-flight state), so a stub whose buttons
  // start blank would make a correct restore look like a cleared button.
  nodes.get("retry-unresponsive").textContent = "Check again";

  globalThis.document = {
    getElementById: (id) => nodes.get(id) ?? null,
    querySelector: (selector) => (selector === ".body" ? nodes.get("__body") : null),
    createElement: () => make("option"),
    querySelectorAll: () => [],
    addEventListener: () => {},
    documentElement: make("html"),
    body: make("body"),
    activeElement: null,
  };
  globalThis.window = { close: () => {}, matchMedia: () => ({ matches: false, addEventListener: () => {} }) };
  return nodes;
}

function installChromeStub({ sendMessage } = {}) {
  const areas = { session: new Map(), local: new Map() };
  const listeners = [];
  const sent = [];
  // Every chrome.runtime.reload() the popup made. Counted rather than flagged:
  // "the popup reloaded the extension twice" is a real defect (the second call
  // races the teardown of the first) and a boolean cannot see it.
  const reloads = [];
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
    // The worker resolves the decided entry and writes the queue back. Modelled
    // faithfully, because the ordering under test IS the storage ordering.
    runtime: {
      reload: () => reloads.push(Date.now()),
      sendMessage: async (message) => {
        sent.push(message);
        // A test-supplied worker replaces the faithful one below: an
        // unresponsive worker is the whole subject of the decide() tests, and
        // it is expressed as what sendMessage DOES, not as a flag.
        if (sendMessage) return sendMessage(message);
        if (message?.event !== "origin_decision") return { applied: true };
        const queue = areas.session.get("accessQueue") ?? [];
        const rest = queue.filter((e) => e.entryId !== message.entryId);
        await makeArea("session").set({ accessQueue: rest });
        return { applied: true };
      },
      openOptionsPage: () => {},
      getManifest: () => ({ version: "0.1.8" }),
    },
    tabs: { query: async () => [] },
  };
  return { areas, sent, reloads };
}

const entry = (entryId, origin = "https://app.example.com", broad = { scope: "domain", key: "example.com" }) => ({
  entryId,
  origin,
  displayAuthority: origin.replace(/^https?:\/\//, ""),
  requester: "req-" + entryId,
  kind: "async",
  requestedAt: Date.now(),
  expiresAt: Date.now() + 600_000,
  sequence: 1,
  broad,
});

const tick = (n = 6) => new Promise((r) => setTimeout(r, n));

async function loadPopup() {
  const dir = await mkdtemp(join(tmpdir(), "lop-popup-render-"));
  const outfile = join(dir, "popup.mjs");
  await build({
    entryPoints: ["src/popup/popup.ts"],
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

/** A live daemon on the pinned port. Returning `paired` matters: an
 * unreachable /health renders `disconnected`, whose branch clears the latches,
 * so a broken fix and a working one look identical. */
function installFetchStub(pendingOrigin) {
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({
      paired: true,
      extension_connected: true,
      protocol_version: 1,
      pending_origin: pendingOrigin(),
    }),
  });
}

test("the re-ask card survives a second render and a queue move (U9/U10)", async () => {
  const nodes = installDomStub();
  const { areas, sent } = installChromeStub();
  installFetchStub(() => undefined);
  const bundle = await loadPopup();
  try {
    areas.local.set("token", "t");
    areas.local.set("port", 4099);
    // Two sessions ask for the same origin (dedupe is origin+requester), plus
    // an unrelated third request to navigate to.
    areas.session.set("accessQueue", [entry("gen-1"), entry("gen-2"), entry("gen-3", "https://third.test", null)]);
    areas.session.set("accessQueueVersion", 1);
    await bundle.import();
    await tick(20);

    const scope = nodes.get("origin-scope");
    const again = nodes.get("origin-again");
    assert.equal(nodes.get("origin").classList.contains("hidden"), false, "the prompt should be showing");
    assert.equal(again.classList.contains("hidden"), true, "a first-ever prompt is not a re-ask");

    // The user deliberately narrows, then allows.
    scope.value = "once";
    nodes.get("origin-allow").click();
    await tick(20);

    // RENDER N: the sibling re-ask. Both halves of the U9 fix must be present.
    assert.equal(again.classList.contains("hidden"), false, "render N: the card must say it is asking again");
    assert.equal(scope.value, "once", "render N: the scope must carry the user's choice");

    // RENDER N+1: a further storage event with no new decision. This is where
    // the half-broken first cut of U9 lost the banner while keeping the scope.
    await chrome.storage.session.set({ connState: "connected" });
    await tick(20);
    assert.equal(again.classList.contains("hidden"), false, "render N+1: the banner must not be recomputed away");
    assert.equal(scope.value, "once", "render N+1: the scope must still be the user's choice");

    // RENDER N+2/N+3: Next then Previous. The option list legitimately
    // rebuilds for a different entry and back, which is how U10 discarded
    // both halves and let the next click grant the whole domain.
    nodes.get("origin-next").click();
    await tick(20);
    nodes.get("origin-previous").click();
    await tick(20);
    assert.equal(again.classList.contains("hidden"), false, "after a queue move: the banner must survive");
    assert.equal(scope.value, "once", "after a queue move: the scope must survive");

    // The wire is what actually grants. A reflexive click here must not
    // escalate to the registrable domain.
    nodes.get("origin-allow").click();
    await tick(20);
    const decisions = sent.filter((m) => m.event === "origin_decision");
    assert.equal(decisions.at(-1).decision, "once", "the second grant must not widen what the user chose");
    assert.deepEqual(
      decisions.map((d) => d.decision),
      ["once", "once"],
      "no decision in this flow may be `domain`",
    );
  } finally {
    await bundle.close();
  }
});

test("deciding another queued origin does not evict this one's re-ask state (U12)", async () => {
  const nodes = installDomStub();
  const { areas, sent } = installChromeStub();
  installFetchStub(() => undefined);
  const bundle = await loadPopup();
  try {
    areas.local.set("token", "t");
    areas.local.set("port", 4099);
    // Two origins waiting, A twice so it survives its own decision. Answering
    // one request while others wait is the whole point of the queue controls,
    // so this is the ordinary path, not a contrivance.
    areas.session.set("accessQueue", [
      entry("gen-1"),
      entry("gen-2"),
      entry("gen-3", "https://third.test", null),
    ]);
    areas.session.set("accessQueueVersion", 1);
    await bundle.import();
    await tick(20);

    const scope = nodes.get("origin-scope");
    const again = nodes.get("origin-again");
    scope.value = "once";
    nodes.get("origin-allow").click();
    await tick(20);
    assert.equal(scope.value, "once", "precondition: A's re-ask carries the narrow choice");

    // Move to the OTHER origin and decide it. A single-slot latch is
    // overwritten here, losing A's decision while A is still queued (U12).
    nodes.get("origin-next").click();
    await tick(20);
    assert.equal(nodes.get("origin-host").textContent, "third.test", "precondition: on the sibling origin");
    nodes.get("origin-allow").click();
    await tick(20);

    // Back to A, which never left the queue.
    nodes.get("origin-previous").click();
    await tick(20);
    assert.equal(nodes.get("origin-host").textContent, "app.example.com", "precondition: back on A");
    assert.equal(again.classList.contains("hidden"), false, "after deciding another origin: the banner must survive");
    assert.equal(scope.value, "once", "after deciding another origin: the scope must survive");

    nodes.get("origin-allow").click();
    await tick(20);
    const forA = sent.filter((m) => m.event === "origin_decision" && m.origin === "https://app.example.com");
    assert.deepEqual(
      forA.map((d) => d.decision),
      ["once", "once"],
      "a decision on another origin must not widen what A is granted",
    );
  } finally {
    await bundle.close();
  }
});

test("an unrelated re-render never clobbers an in-flight selection (A13)", async () => {
  const nodes = installDomStub();
  const { areas, sent } = installChromeStub();
  installFetchStub(() => undefined);
  const bundle = await loadPopup();
  try {
    areas.local.set("token", "t");
    areas.local.set("port", 4099);
    areas.session.set("accessQueue", [entry("gen-1")]);
    areas.session.set("accessQueueVersion", 1);
    await bundle.import();
    await tick(20);

    const scope = nodes.get("origin-scope");
    // The user narrows but has NOT clicked yet. No decision exists, so the
    // re-ask latch cannot protect this selection: only renderScopeSelect's
    // "same prompt, already built" guard does. Deleting that guard typechecks
    // and passed every other test while sending `domain` for a user who chose
    // `once` - A1/U1 restored (A13).
    scope.value = "once";

    // An unrelated sibling enqueues, which re-renders the same prompt.
    areas.session.set("accessQueue", [
      entry("gen-1"),
      entry("gen-9", "https://unrelated.test", null),
    ]);
    await chrome.storage.session.set({
      accessQueue: [entry("gen-1"), entry("gen-9", "https://unrelated.test", null)],
    });
    await tick(20);

    assert.equal(scope.value, "once", "an unrelated re-render must not reset the in-flight scope");
    nodes.get("origin-allow").click();
    await tick(20);
    const decisions = sent.filter((m) => m.event === "origin_decision");
    assert.equal(decisions.at(-1).decision, "once", "the wire must carry what the user selected");
  } finally {
    await bundle.close();
  }
});

test("a re-ask latch does not outlive its origin leaving the queue (A14)", async () => {
  const nodes = installDomStub();
  const { areas } = installChromeStub();
  installFetchStub(() => undefined);
  const bundle = await loadPopup();
  try {
    areas.local.set("token", "t");
    areas.local.set("port", 4099);
    // TWO entries for one origin: the decision is only recorded against the
    // origin on a render where that origin is still pending, so a queue that
    // drains on the click never reaches the latch at all.
    areas.session.set("accessQueue", [entry("gen-1"), entry("gen-2")]);
    areas.session.set("accessQueueVersion", 1);
    await bundle.import();
    await tick(20);

    const scope = nodes.get("origin-scope");
    const again = nodes.get("origin-again");
    scope.value = "once";
    nodes.get("origin-allow").click();
    await tick(20);
    assert.equal(again.classList.contains("hidden"), false, "precondition: the latch is set for this origin");

    // The origin now leaves the queue entirely.
    await chrome.storage.session.set({ accessQueue: [] });
    await tick(20);
    assert.equal(nodes.get("origin").classList.contains("hidden"), true, "precondition: queue drained");

    // It asks again later. This is a genuinely NEW request, not a re-ask of
    // one just answered, so it must get the fail-closed default and no
    // banner. An uncleared latch narrows instead: bounded harm, since it can
    // only ever narrow, but the banner would be lying (A14).
    await chrome.storage.session.set({ accessQueue: [entry("gen-77")] });
    await tick(20);

    assert.equal(again.classList.contains("hidden"), true, "a new request must not claim to be a re-ask");
    assert.equal(scope.value, "domain", "a new request gets the fail-closed default, not a carried scope");
  } finally {
    await bundle.close();
  }
});

test("a deny re-ask does not claim the answer was used (U11)", async () => {
  const nodes = installDomStub();
  const { areas } = installChromeStub();
  installFetchStub(() => undefined);
  const bundle = await loadPopup();
  try {
    areas.local.set("token", "t");
    areas.local.set("port", 4099);
    areas.session.set("accessQueue", [entry("gen-1"), entry("gen-2")]);
    areas.session.set("accessQueueVersion", 1);
    await bundle.import();
    await tick(20);

    nodes.get("origin-deny").click();
    await tick(20);

    const again = nodes.get("origin-again");
    assert.equal(again.classList.contains("hidden"), false, "a deny re-ask is still a re-ask");
    assert.match(again.textContent, /denied this site/, "the copy must name the refusal");
    assert.doesNotMatch(
      again.textContent,
      /already been used/,
      "nothing was used: the agent did not visit the site",
    );
  } finally {
    await bundle.close();
  }
});

test("a wedged worker is not painted as connected (D4)", async () => {
  const nodes = installDomStub();
  installChromeStub();
  // A live daemon that reports the state this whole change exists for: paired,
  // socket attached, nothing answered. `extension_unresponsive` is the field the
  // popup must CONSUME — it was in the Health type and never read, so the green
  // card claimed the agent could drive while no command could be answered.
  let health = { paired: true, extension_connected: true, protocol_version: 1, pending_origin: undefined };
  globalThis.fetch = async () => ({ ok: true, json: async () => health });
  const bundle = await loadPopup();
  try {
    await bundle.import();
    await tick(20);
    assert.equal(nodes.get("connected").classList.contains("hidden"), false, "precondition: healthy renders the connected card");
    assert.equal(nodes.get("unresponsive").classList.contains("hidden"), true, "precondition: no wedge, no wedge card");

    // The worker goes mute. /health keeps `paired: true` (and so does the LINK),
    // which is exactly why the connected card used to stay up.
    health = { ...health, extension_connected: false, extension_unresponsive: true, link_attached: true };
    await chrome.storage.session.set({ connState: "connected" });
    await tick(20);
    assert.equal(nodes.get("unresponsive").classList.contains("hidden"), false, "a wedged worker must not render as connected");
    assert.equal(nodes.get("connected").classList.contains("hidden"), true, "the success card must be REPLACED, not merely annotated");

    // RENDER N+1: the same state, one more storage event. Every defect in this
    // class is correct on render N and wrong on N+1.
    await chrome.storage.session.set({ connState: "connected" });
    await tick(20);
    assert.equal(nodes.get("unresponsive").classList.contains("hidden"), false, "must survive a later render");
    assert.equal(nodes.get("connected").classList.contains("hidden"), true);

    // The link comes back: the card must recover, or the fix would be its own
    // sticky lie.
    health = { ...health, extension_connected: true, extension_unresponsive: false };
    await chrome.storage.session.set({ connState: "connected" });
    await tick(20);
    assert.equal(nodes.get("connected").classList.contains("hidden"), false, "must recover to the connected card");
    assert.equal(nodes.get("unresponsive").classList.contains("hidden"), true);

    // THE OTHER HALF OF THE WINDOW, and the reason this card exists at all: the
    // post-drop cooling-off period, where the daemon has severed the link so
    // `/health` reports `paired: false` (it is link-derived) while the pairing on
    // disk — and the daemon's own `paired:` line — are still true. The card used
    // to be gated on `paired`, so this exact payload rendered the PAIRING FORM
    // for a paired browser that is about to re-dial (QA Q2-3 / review R2-5).
    health = {
      ...health,
      paired: false,
      extension_connected: false,
      extension_unresponsive: true,
      link_attached: false,
    };
    await chrome.storage.session.set({ connState: "connected" });
    await tick(20);
    assert.equal(
      nodes.get("unresponsive").classList.contains("hidden"),
      false,
      "the latched half of the window must show the honest card, not the pairing form",
    );
    assert.equal(nodes.get("pairing").classList.contains("hidden"), true, "a paired browser must not be asked for a code");
    assert.equal(nodes.get("connected").classList.contains("hidden"), true);
  } finally {
    await bundle.close();
  }
});

/* --- The stale/unresponsive worker, from the popup's seat -------------------
 *
 * The operator's complaint was "I click the toolbar icon and nothing happens,
 * it takes 2-3 tries". Chrome owns opening the popup, so a click that shows
 * NOTHING means the MV3 worker never started; but a click that shows a popup
 * whose buttons then do nothing is this module's fault, and that is what these
 * cover: decide() awaited the worker with no bound and no catch, so a stale
 * worker left Allow/Deny disabled for the life of the popup with no message.
 *
 * Assertions are on what the user can SEE and DO — the buttons' disabled state
 * and the text in the ack slot — never on the presence of a call.
 */

/** A live daemon whose /health payload the test controls per render. */
function installHealth(get) {
  globalThis.fetch = async () => ({ ok: true, json: async () => get() });
}

const pendingEntry = (entryId = "gen-1") => ({
  entryId,
  origin: "https://app.example.com",
  displayAuthority: "app.example.com",
  requester: "req-1",
  kind: "async",
  requestedAt: Date.now(),
  expiresAt: Date.now() + 600_000,
  sequence: 1,
  broad: { scope: "domain", key: "example.com" },
});

test("a decision whose worker REJECTS leaves the controls usable and says so", async () => {
  const nodes = installDomStub();
  const { areas } = installChromeStub({
    // Exactly what MV3 throws when the worker is gone and nothing receives the
    // message — captured from the operator's own console.
    sendMessage: async () => {
      throw new Error("Could not establish connection. Receiving end does not exist.");
    },
  });
  installHealth(() => ({ paired: true, extension_connected: true, protocol_version: 1 }));
  const rejections = [];
  const onRejection = (error) => rejections.push(error);
  process.on("unhandledRejection", onRejection);
  const bundle = await loadPopup();
  try {
    areas.local.set("token", "t");
    areas.session.set("accessQueue", [pendingEntry()]);
    areas.session.set("accessQueueVersion", 1);
    await bundle.import();
    await tick(20);
    assert.equal(nodes.get("origin").classList.contains("hidden"), false, "precondition: the prompt is up");
    assert.equal(nodes.get("origin-allow").disabled, false, "precondition: Allow is clickable");

    nodes.get("origin-allow").click();
    // Past the notice hold (1500ms), so the settled state is what is asserted.
    await tick(1800);

    // THE DEFECT: setOriginBusy(true) ran, the await rejected, and nothing ever
    // re-enabled these. From the user's seat the popup stopped responding.
    assert.equal(nodes.get("origin-allow").disabled, false, "Allow must not be left disabled by an unreachable worker");
    assert.equal(nodes.get("origin-deny").disabled, false, "Deny must not be left disabled by an unreachable worker");

    // And the click must be ACKNOWLEDGED honestly rather than silently dropped.
    const title = nodes.get("origin-ack-title").textContent;
    const sub = nodes.get("origin-ack-sub").textContent;
    assert.match(title, /didn't|did not|no answer/i, `the user must be told the extension did not answer, got ${JSON.stringify(title)}`);
    assert.doesNotMatch(
      title + " " + sub,
      /request changed/i,
      "'Request changed.' means a REPLACED generation — a live worker's answer — and must not be reused for an unreachable one",
    );
    assert.deepEqual(rejections, [], "the failed round-trip escaped as an unhandled rejection");
  } finally {
    process.off("unhandledRejection", onRejection);
    await bundle.close();
  }
});

test("a decision whose worker NEVER ANSWERS is bounded and ends the same way", async () => {
  const nodes = installDomStub();
  const { areas } = installChromeStub({
    // The nastier half, and the one the operator actually hit: the worker is
    // loaded but mute, so the send neither resolves nor rejects — ever. Without
    // a bound this await is permanent and the popup is dead until it closes.
    sendMessage: () => new Promise(() => {}),
  });
  installHealth(() => ({ paired: true, extension_connected: true, protocol_version: 1 }));
  const rejections = [];
  const onRejection = (error) => rejections.push(error);
  process.on("unhandledRejection", onRejection);
  const bundle = await loadPopup();
  try {
    areas.local.set("token", "t");
    areas.session.set("accessQueue", [pendingEntry()]);
    areas.session.set("accessQueueVersion", 1);
    await bundle.import();
    await tick(20);
    assert.equal(nodes.get("origin-allow").disabled, false, "precondition: Allow is clickable");

    nodes.get("origin-allow").click();
    // Still inside the 5s bound: the popup is legitimately waiting, and saying
    // "no answer" here would libel a merely-slow worker.
    await tick(300);
    assert.equal(nodes.get("origin-allow").disabled, true, "while the round-trip is in flight the controls stay busy");

    // Past the bound plus the notice hold.
    await tick(7000);
    assert.equal(nodes.get("origin-allow").disabled, false, "the timeout path must re-enable Allow");
    assert.equal(nodes.get("origin-deny").disabled, false, "the timeout path must re-enable Deny");
    assert.match(
      nodes.get("origin-ack-title").textContent,
      /didn't|did not|no answer/i,
      "a click that timed out must still be acknowledged",
    );
    assert.deepEqual(rejections, [], "the timed-out round-trip escaped as an unhandled rejection");
  } finally {
    process.off("unhandledRejection", onRejection);
    await bundle.close();
  }
});

test("the wedged-worker card offers a one-click reload, and a healthy one never does", async () => {
  const nodes = installDomStub();
  const { reloads } = installChromeStub();
  // The daemon is the authority here, deliberately: the worker's own
  // `connState` reads "connected" long after the worker is dead (it is that
  // worker's last write), so a reload offered from it would appear on a healthy
  // popup and disappear on a wedged one — backwards.
  let health = { paired: true, extension_connected: true, protocol_version: 1 };
  installHealth(() => health);
  const bundle = await loadPopup();
  try {
    await bundle.import();
    await tick(20);

    // HEALTHY: the remedy is not on offer. A reload discards open tab handles,
    // snapshot refs and pending site decisions, so offering it to a working
    // browser invites a user to pay that for nothing.
    assert.equal(nodes.get("connected").classList.contains("hidden"), false, "precondition: the healthy card is up");
    assert.equal(
      nodes.get("unresponsive").classList.contains("hidden"),
      true,
      "a healthy worker must not be offered a reload",
    );
    // "Hidden section" is only the same thing as "no reload on offer" because
    // the control lives INSIDE that section. This harness's DOM is flat (every
    // id is a sibling), so it cannot express containment and a click on it here
    // would fire in a state a real user cannot reach. Assert the containment
    // against the real markup instead, which is what makes the line above a
    // statement about what the user can do rather than about a CSS class.
    const markup = await readFile(join(HERE, "..", "src", "popup", "popup.html"), "utf8");
    const section = markup.slice(markup.indexOf('<section id="unresponsive"'));
    const body = section.slice(0, section.indexOf("</section>"));
    assert.ok(
      body.includes('id="reload-extension"'),
      "the reload control must live inside #unresponsive, or hiding that card does not withdraw the offer",
    );
    assert.deepEqual(reloads, [], "nothing on the healthy path may reload the extension");

    // WEDGED: the daemon says the extension is not answering.
    health = { ...health, extension_connected: false, extension_unresponsive: true, link_attached: true };
    await chrome.storage.session.set({ connState: "connected" });
    await tick(20);
    assert.equal(nodes.get("unresponsive").classList.contains("hidden"), false, "precondition: the wedge card is up");

    nodes.get("reload-extension").click();
    await tick(20);
    assert.equal(reloads.length, 1, "the wedge card's primary action must reload the extension exactly once");
  } finally {
    await bundle.close();
  }
});

/* --- Round 1 remediation: the wedged worker WITH a decision pending ---------
 *
 * The state three review streams found independently (reviewer/UX U1, QA Q5,
 * design D3) and the shape of the reported incident: the operator's dead clicks
 * happened while the agent was driving, i.e. with a request queued. renderOnce()
 * paints the consent card and returns before the `unresponsive` branch, so the
 * PR's own remedy was in a hidden section exactly when it was needed.
 */

test("a decision pending against a WEDGED worker surfaces the remedy inline", async () => {
  const nodes = installDomStub();
  const { areas, reloads } = installChromeStub();
  let health = { paired: true, extension_connected: true, protocol_version: 1 };
  installHealth(() => health);
  const bundle = await loadPopup();
  try {
    areas.local.set("token", "t");
    areas.session.set("accessQueue", [pendingEntry()]);
    areas.session.set("accessQueueVersion", 1);
    await bundle.import();
    await tick(20);

    // HEALTHY worker, same pending decision: the prompt is a normal prompt. No
    // banner, and no remedy on offer — a working worker must never be told to
    // reload, and this is the control that keeps the banner honest.
    assert.equal(nodes.get("origin").classList.contains("hidden"), false, "precondition: the prompt is up");
    assert.equal(
      nodes.get("origin-wedge").classList.contains("hidden"),
      true,
      "a healthy worker must not be told its extension stopped answering",
    );

    // The worker goes mute while the request is still queued. The prompt STAYS
    // — it is what the user opened the popup for, and hiding it would answer
    // "the remedy is unreachable" by making the request unreachable instead.
    health = { ...health, extension_connected: false, extension_unresponsive: true, link_attached: true };
    await chrome.storage.session.set({ connState: "connected" });
    await tick(20);
    assert.equal(nodes.get("origin").classList.contains("hidden"), false, "the pending decision stays visible");
    assert.equal(
      nodes.get("origin-wedge").classList.contains("hidden"),
      false,
      "the card must say the decision cannot be applied right now",
    );

    // And the remedy is reachable from here, in one click, without finding
    // another card first.
    nodes.get("origin-wedge-reload").click();
    await tick(20);
    assert.equal(reloads.length, 1, "the inline remedy must reload the extension");

    // RENDER N+1 — every defect in this class is right on N and wrong on N+1.
    await chrome.storage.session.set({ connState: "connected" });
    await tick(20);
    assert.equal(nodes.get("origin-wedge").classList.contains("hidden"), false, "the banner must survive a re-render");

    // Recovery: the worker answers again, so the banner must go. A sticky
    // warning would be its own lie.
    health = { ...health, extension_connected: true, extension_unresponsive: false };
    await chrome.storage.session.set({ connState: "connected" });
    await tick(20);
    assert.equal(
      nodes.get("origin-wedge").classList.contains("hidden"),
      true,
      "the banner must clear when the worker answers again",
    );
  } finally {
    await bundle.close();
  }
});

test("a decision in flight does not claim success until the worker confirms", async () => {
  const nodes = installDomStub();
  let release;
  const { areas } = installChromeStub({
    // A worker that answers only when the test says so: the window between the
    // click and the answer IS the subject, and it was 4.9s of green "Site
    // allowed." over a decision that was never applied (UX U2).
    sendMessage: () => new Promise((resolve) => { release = () => resolve({ applied: true }); }),
  });
  installHealth(() => ({ paired: true, extension_connected: true, protocol_version: 1 }));
  const bundle = await loadPopup();
  try {
    areas.local.set("token", "t");
    areas.session.set("accessQueue", [pendingEntry()]);
    areas.session.set("accessQueueVersion", 1);
    await bundle.import();
    await tick(20);
    nodes.get("origin-allow").click();
    await tick(50);

    // IN FLIGHT. The click is acknowledged — the user must see it landed — but
    // none of the three things that read as "done" may be on screen.
    assert.match(
      nodes.get("origin-ack-title").textContent,
      /allowing/i,
      "an in-flight decision must acknowledge the click and name which way it went",
    );
    assert.equal(nodes.get("origin-ack-check").classList.contains("hidden"), true, "no check before the worker confirms");
    assert.equal(
      nodes.get("origin-ack-granted").classList.contains("hidden"),
      true,
      "the granted trough means 'this is what you granted' and must not precede the grant",
    );

    // CONFIRMED: now, and only now, the success card.
    release();
    await tick(60);
    // The entry's default scope is `domain`, so the confirmed ack names the
    // domain grant — asserting the CONFIRMED vocabulary, not a fixed string:
    // what matters is that it switched out of the progressive in-flight voice.
    assert.match(
      nodes.get("origin-ack-title").textContent,
      /allowed\./i,
      "a confirmed decision reads as done, not as in progress",
    );
    assert.doesNotMatch(nodes.get("origin-ack-title").textContent, /allowing/i);
    assert.equal(nodes.get("origin-ack-check").classList.contains("hidden"), false, "a confirmed decision shows the check");
  } finally {
    await bundle.close();
  }
});

test("the unreachable-worker notice carries no granted value", async () => {
  const nodes = installDomStub();
  // The trough is filled by a CONFIRMED ack, so the only way to reach the bug
  // is the real sequence: one decision lands (printing the granted host), the
  // next one cannot be delivered. Asserting it from a first, never-confirmed
  // click would pass with the fix reverted, because the in-flight ack no longer
  // fills the trough at all — the guard has to be proven where it bites.
  let worker = "alive";
  const { areas } = installChromeStub({
    sendMessage: async (message) => {
      if (worker === "dead") {
        throw new Error("Could not establish connection. Receiving end does not exist.");
      }
      const queue = areas.session.get("accessQueue") ?? [];
      await chrome.storage.session.set({
        accessQueue: queue.filter((e) => e.entryId !== message.entryId),
      });
      return { applied: true };
    },
  });
  installHealth(() => ({ paired: true, extension_connected: true, protocol_version: 1 }));
  const bundle = await loadPopup();
  try {
    areas.local.set("token", "t");
    areas.session.set("accessQueue", [pendingEntry("gen-1"), pendingEntry("gen-2")]);
    areas.session.set("accessQueueVersion", 1);
    await bundle.import();
    await tick(20);

    // Decision one lands, so the granted host is printed in the trough.
    nodes.get("origin-allow").click();
    await tick(80);
    assert.equal(
      nodes.get("origin-ack-granted").classList.contains("hidden"),
      false,
      "precondition: a confirmed decision prints what it granted",
    );
    const granted = nodes.get("origin-ack-granted").textContent;
    assert.ok(granted, "precondition: the trough actually carries a value");

    // The worker dies before the next decision can be delivered.
    worker = "dead";
    await tick(40);
    nodes.get("origin-allow").click();
    await tick(1800);

    // The trough is the element a user reads as the granted value. Leaving the
    // previous decision's host under "may not have been applied" prints an
    // answer to a question the popup just said it cannot answer (UX U3).
    assert.match(nodes.get("origin-ack-title").textContent, /didn't|did not|no answer/i);
    assert.equal(
      nodes.get("origin-ack-granted").classList.contains("hidden"),
      true,
      "a decision that may not have been applied must not print a granted value",
    );
    assert.equal(
      nodes.get("origin-ack-granted").textContent,
      "",
      "the granted trough must be cleared, not merely hidden — a later ack would reveal the stale host",
    );
    assert.equal(nodes.get("origin-ack-check").classList.contains("hidden"), true, "no check on an unapplied decision");
  } finally {
    await bundle.close();
  }
});

test("a decision does not resurrect queue controls the renderer disabled", async () => {
  const nodes = installDomStub();
  const { areas } = installChromeStub();
  installHealth(() => ({ paired: true, extension_connected: true, protocol_version: 1 }));
  const bundle = await loadPopup();
  try {
    areas.local.set("token", "t");
    // Two entries, so Previous/Next start live; deciding one leaves a single
    // request, and a single-request prompt must not offer two inert stops (U4).
    areas.session.set("accessQueue", [pendingEntry("gen-1"), pendingEntry("gen-2")]);
    areas.session.set("accessQueueVersion", 1);
    await bundle.import();
    await tick(20);
    assert.equal(nodes.get("origin-previous").disabled, false, "precondition: two entries, navigation is live");

    nodes.get("origin-allow").click();
    await tick(60);

    // The decision owns its own three controls. Previous/Next are DERIVED from
    // queue length by renderQueueControls, and the decision's cleanup runs
    // after that render — so re-enabling them here overwrote the renderer.
    assert.equal(nodes.get("origin-allow").disabled, false, "the decision's own controls come back");
    assert.equal(
      nodes.get("origin-previous").disabled,
      true,
      "one entry left: queue navigation stays the renderer's call, not the decision's",
    );
    assert.equal(nodes.get("origin-next").disabled, true, "same for Next");
  } finally {
    await bundle.close();
  }
});

test("Check again shows that it checked, and says so when nothing changed", async () => {
  const nodes = installDomStub();
  installChromeStub();
  const health = {
    paired: true,
    extension_connected: false,
    extension_unresponsive: true,
    link_attached: true,
    protocol_version: 1,
  };
  let probes = 0;
  globalThis.fetch = async () => {
    probes += 1;
    return { ok: true, json: async () => health };
  };
  const bundle = await loadPopup();
  try {
    await bundle.import();
    await tick(20);
    assert.equal(nodes.get("unresponsive").classList.contains("hidden"), false, "precondition: the wedge card is up");
    const before = probes;

    nodes.get("retry-unresponsive").click();
    // Mid-probe: the click must own the button, or a re-check that changes
    // nothing is indistinguishable from a dead click — the exact complaint this
    // PR was filed about, reappearing on the recovery card (UX U4).
    await tick(30);
    assert.equal(nodes.get("retry-unresponsive").disabled, true, "the button acknowledges the click while probing");
    assert.match(nodes.get("retry-unresponsive").textContent, /checking/i, "and says what it is doing");

    await tick(700);
    assert.ok(probes > before, "the re-check must actually re-probe /health");
    assert.equal(nodes.get("retry-unresponsive").disabled, false, "the button comes back");
    assert.equal(nodes.get("retry-unresponsive").textContent, "Check again", "with its label restored");
    // An unchanged verdict is the one outcome the DOM cannot express by itself.
    assert.equal(
      nodes.get("unresponsive-outcome").classList.contains("hidden"),
      false,
      "an unchanged answer must still be stated",
    );
    assert.match(nodes.get("unresponsive-outcome").textContent, /still not answering/i);
  } finally {
    await bundle.close();
  }
});

/* --- Round 2: the banner must not over-claim, and the card must open at its top */

test("the banner stays hidden while the decision is still deliverable (M1)", async () => {
  const nodes = installDomStub();
  const { areas } = installChromeStub();
  // The post-drop latch: the daemon severed the socket, so `link_attached` is
  // false, but `extension_unresponsive` stays true for LINK_DROP_TTL_S = 60s
  // while the worker is usually alive and re-dialling. A decision travels over
  // chrome.runtime.sendMessage, which never touched that socket — so it is
  // deliverable, and the banner claiming otherwise steers the user at a reload
  // that would destroy the very decision they came to answer (review M1).
  let health = {
    paired: true,
    extension_connected: false,
    extension_unresponsive: true,
    link_attached: false,
    protocol_version: 1,
  };
  installHealth(() => health);
  const bundle = await loadPopup();
  try {
    areas.local.set("token", "t");
    areas.session.set("accessQueue", [pendingEntry()]);
    areas.session.set("accessQueueVersion", 1);
    await bundle.import();
    await tick(20);
    assert.equal(nodes.get("origin").classList.contains("hidden"), false, "precondition: the prompt is up");
    assert.equal(
      nodes.get("origin-wedge").classList.contains("hidden"),
      true,
      "a severed-but-re-dialling link must not be reported as an undeliverable decision",
    );

    // And the decision really is deliverable in that state — which is what
    // makes the banner a false claim rather than a cautious one.
    nodes.get("origin-allow").click();
    await tick(80);
    assert.match(
      nodes.get("origin-ack-title").textContent,
      /allowed\./i,
      "the decision must apply normally while the link is merely dropped",
    );

    // ATTACHED and mute is the case the banner is actually for: sendMessage
    // genuinely has nowhere to land.
    health = { ...health, link_attached: true };
    areas.session.set("accessQueue", [pendingEntry("gen-2")]);
    await chrome.storage.session.set({ connState: "connected" });
    await tick(30);
    assert.equal(
      nodes.get("origin-wedge").classList.contains("hidden"),
      false,
      "an attached-but-mute worker must still raise the banner",
    );
  } finally {
    await bundle.close();
  }
});

test("a fresh prompt opens at the top of its card, not scrolled past its banner (D7/U8)", async () => {
  const nodes = installDomStub();
  const { areas } = installChromeStub();
  installHealth(() => ({
    paired: true,
    extension_connected: false,
    extension_unresponsive: true,
    link_attached: true,
    protocol_version: 1,
  }));
  const bundle = await loadPopup();
  try {
    areas.local.set("token", "t");
    areas.session.set("accessQueue", [pendingEntry()]);
    areas.session.set("accessQueueVersion", 1);
    await bundle.import();
    await tick(30);

    const scope = nodes.get("origin-scope");
    const body = globalThis.document.querySelector(".body");
    assert.equal(nodes.get("origin-wedge").classList.contains("hidden"), false, "precondition: the banner is up");

    // The keyboard landing point is unchanged — the fix must not cost it.
    assert.ok(scope._focusCalls.length > 0, "the scope select is still focused for the keyboard");
    assert.equal(globalThis.document.activeElement, scope, "and it really holds focus");

    // THE DEFECT: focusing inside the scroll container scrolled the card to its
    // maximum, putting the title and the whole danger banner above the fold at
    // >=125% zoom — a zoomed user saw an ordinary consent prompt with live
    // Allow/Deny and no sign their answer could not land.
    assert.equal(
      body.scrollTop,
      0,
      "the card must open at its top: the question and the banner qualifying it are the first things to read",
    );
    assert.ok(
      scope._focusCalls.some((options) => options && options.preventScroll === true),
      "focus must opt out of scrolling rather than rely on the card being short enough",
    );
  } finally {
    await bundle.close();
  }
});

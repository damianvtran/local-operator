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
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

/** The popup's markup, reduced to the ids and classes popup.ts touches.
 * Kept as a literal id list rather than parsing popup.html: a test that
 * silently stops covering an element because the parse missed it is worse
 * than one that fails when an id is renamed. */
const IDS = [
  "connected", "paired", "pairing", "disconnected", "incompatible", "origin", "origin-ack",
  "origin-host", "origin-again", "origin-scope", "origin-scope-detail", "origin-position",
  "origin-waiting", "origin-allow", "origin-deny", "origin-previous", "origin-next",
  "origin-ack-title", "origin-ack-sub", "origin-ack-check", "card", "retry",
  "retry-incompatible", "connected-all-sites", "connected-all-sites-off", "pair-form",
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
      focus: () => {},
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

  globalThis.document = {
    getElementById: (id) => nodes.get(id) ?? null,
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
    // The worker resolves the decided entry and writes the queue back. Modelled
    // faithfully, because the ordering under test IS the storage ordering.
    runtime: {
      sendMessage: async (message) => {
        sent.push(message);
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
  return { areas, sent };
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

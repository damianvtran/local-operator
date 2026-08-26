import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function load(entry) {
  const dir = await mkdtemp(join(tmpdir(), "lop-extension-test-"));
  const outfile = join(dir, "module.mjs");
  await build({ entryPoints: [entry], bundle: true, platform: "node", format: "esm", outfile });
  const loaded = await import(pathToFileURL(outfile));
  return { loaded, close: () => rm(dir, { recursive: true, force: true }) };
}

test("origin policy only permits stored HTTP origins", async () => {
  const module = await load("src/origin-policy.ts");
  try {
    const url = module.loaded.safeHttpUrl("https://example.com/path");
    assert.equal(module.loaded.storedOriginAllowed({ "https://example.com": "allow" }, url), true);
    assert.equal(module.loaded.storedOriginAllowed({}, url), false);
    assert.throws(() => module.loaded.safeHttpUrl("chrome://settings"));
  } finally { await module.close(); }
});

test("AX compaction assigns epoch-scoped click refs", async () => {
  const module = await load("src/ax-compact.ts");
  try {
    const rendered = module.loaded.compactAX([
      { nodeId: "1", role: { value: "main" }, name: { value: "Content" }, childIds: ["2"] },
      { nodeId: "2", role: { value: "button" }, name: { value: "Continue" }, backendDOMNodeId: 42 },
    ], 7);
    assert.match(rendered.snapshot, /button "Continue" \[e1\]/);
    assert.deepEqual(rendered.refs.e1, { backendNodeId: 42, epoch: 7 });
  } finally { await module.close(); }
});

test("scroll expressions force instant behavior in every mode", async () => {
  const module = await load("src/scroll-expressions.ts");
  try {
    const { scrollExpressionFor, defaultScrollExpression, deltaScrollExpression, SCROLL_INTO_VIEW_FN } = module.loaded;
    // Pages can opt into CSS scroll-behavior:smooth, and Chrome throttles rAF
    // to zero in hidden tabs, so a smooth scroll never progresses in our
    // background surface. Every fixed expression must override with 'instant'.
    for (const direction of ["top", "bottom", "up", "down", "left", "right"]) {
      const expr = scrollExpressionFor(direction);
      assert.match(expr, /behavior: 'instant'/, `${direction} must scroll instantly`);
      assert.match(expr, /window\.scroll(By|To)\(\{/, `${direction} must use the options form`);
    }
    assert.match(defaultScrollExpression(), /behavior: 'instant'/);
    assert.match(deltaScrollExpression(10, -20), /left: 10, top: -20, behavior: 'instant'/);
    assert.match(SCROLL_INTO_VIEW_FN, /behavior: 'instant'/);
    // Unknown direction stays a no-op, never interpolated page-bound code.
    assert.equal(scrollExpressionFor("sideways"), "void 0");
  } finally { await module.close(); }
});

test("log filter keeps level matches and limits to the most recent", async () => {
  const module = await load("src/log-capture.ts");
  try {
    const { filterEntries } = module.loaded;
    const entries = [
      { level: "log", text: "a" },
      { level: "error", text: "b" },
      { level: "log", text: "c" },
      { level: "error", text: "d" },
    ];
    // "all" keeps everything, order preserved.
    assert.deepEqual(filterEntries(entries, "all", 0).map((e) => e.text), ["a", "b", "c", "d"]);
    // level filter keeps only matches.
    assert.deepEqual(filterEntries(entries, "error", 0).map((e) => e.text), ["b", "d"]);
    // limit keeps the most recent n, still oldest->newest.
    assert.deepEqual(filterEntries(entries, "all", 2).map((e) => e.text), ["c", "d"]);
  } finally { await module.close(); }
});

test("pair verdict renders success only with a storable token", async () => {
  const module = await load("src/popup/pair-flow.ts");
  try {
    const { pairVerdict, PAIR_MISMATCH_MESSAGE } = module.loaded;
    assert.deepEqual(pairVerdict({ event: "pair_result", ok: true, token: "t1" }), {
      ok: true,
      token: "t1",
    });
    // ok without a token is a failure: nothing to authenticate future
    // connections with, so "Patched in." would be a lie.
    assert.deepEqual(pairVerdict({ ok: true }), { ok: false, message: PAIR_MISMATCH_MESSAGE });
    // daemon-provided message wins over the default mismatch copy.
    assert.deepEqual(pairVerdict({ ok: false, message: "No live pairing code" }), {
      ok: false,
      message: "No live pairing code",
    });
    assert.deepEqual(pairVerdict({}), { ok: false, message: PAIR_MISMATCH_MESSAGE });
  } finally { await module.close(); }
});

test("health render holds the success view during the pair/health race", async () => {
  const module = await load("src/popup/pair-flow.ts");
  try {
    const { viewForHealth } = module.loaded;
    // The race: pair_result.ok arrived but the worker has not reconnected, so
    // health still says unpaired — the form must NOT come back.
    assert.equal(viewForHealth(false, true), "paired");
    // Health caught up: the connected view (with URL details) takes over.
    assert.equal(viewForHealth(true, true), "connected");
    assert.equal(viewForHealth(true, false), "connected");
    // Never paired in this popup: the form is the right offer.
    assert.equal(viewForHealth(false, false), "pairing");
  } finally { await module.close(); }
});

test("origin decision acks render per decision, deny staying neutral", async () => {
  const module = await load("src/popup/origin-flow.ts");
  try {
    const { ackForDecision } = module.loaded;
    const once = ackForDecision("once");
    assert.equal(once.title, "Site allowed.");
    assert.equal(once.sub, "The agent is continuing.");
    assert.deepEqual([once.tone, once.check], ["success", true]);
    // "always" is a standing grant: the ack must say it persists and where to
    // revoke it.
    const always = ackForDecision("always");
    assert.equal(always.tone, "success");
    assert.match(always.sub, /Always-allowed sites can be taken back any time in Settings\./);
    // Deny is a completed choice, not a failure: neutral, no check.
    const deny = ackForDecision("deny");
    assert.deepEqual([deny.title, deny.tone, deny.check], ["Site denied.", "neutral", false]);
  } finally { await module.close(); }
});

test("origin render holds the ack through the decision round-trip race", async () => {
  const module = await load("src/popup/origin-flow.ts");
  try {
    const { originPromptView } = module.loaded;
    const decided = { origin: "https://example.com", decision: "always" };
    // The race: the prompt is still echoed after the click — hold the ack, do
    // not resurrect the buttons.
    assert.equal(originPromptView("https://example.com", decided), "ack");
    // A DIFFERENT pending origin is a new prompt even mid-settle (A6).
    assert.equal(originPromptView("https://other.example", decided), "prompt");
    // Round-trip landed: nothing pending, caller clears its latch.
    assert.equal(originPromptView(undefined, decided), "none");
    // Never decided in this popup: the prompt is correct.
    assert.equal(originPromptView("https://example.com", null), "prompt");
  } finally { await module.close(); }
});

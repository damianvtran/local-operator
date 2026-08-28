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

test("origin policy preserves exact grants and scopes loopback all-port grants", async () => {
  const module = await load("src/origin-policy.ts");
  try {
    const exact = module.loaded.safeHttpUrl("https://example.com/path");
    assert.equal(module.loaded.storedOriginAllowed({ "https://example.com": "allow" }, exact), true);
    assert.equal(module.loaded.storedOriginAllowed({}, exact), false);
    assert.throws(() => module.loaded.safeHttpUrl("chrome://settings"));

    const eligible = ["http://LOCALHOST:5173", "http://127.0.0.1:3000", "http://[::1]:8000"];
    for (const href of eligible) assert.equal(module.loaded.isLoopbackHost(new URL(href)), true, href);
    const ineligible = [
      "http://localhost.:5173",
      "http://api.localhost:5173",
      "http://127.0.0.2",
      "http://0.0.0.0",
      "http://[::ffff:127.0.0.1]",
      "http://192.168.1.1",
      "http://example.com",
      "http://locаlhost", // Cyrillic 'a' is not ASCII localhost.
    ];
    for (const href of ineligible) assert.equal(module.loaded.isLoopbackHost(new URL(href)), false, href);
    assert.throws(() => module.loaded.safeHttpUrl("http://127.1"), /127\.0\.0\.1 exactly/);

    assert.equal(module.loaded.displayAuthority(new URL("http://localhost:5173")), "localhost:5173");
    assert.equal(module.loaded.displayAuthority(new URL("http://localhost:80")), "localhost");
    assert.equal(module.loaded.displayAuthority(new URL("https://localhost:443")), "localhost");
    assert.equal(module.loaded.displayAuthority(new URL("http://[::1]:8000")), "[::1]:8000");

    const source = new URL("http://localhost:5173");
    const key = module.loaded.loopbackHostGrantKey(source);
    const hostGrants = { version: 1, grants: { [key]: { scope: "all_ports", createdAt: 1 } } };
    assert.equal(module.loaded.storedOriginAllowed({}, new URL("http://localhost:9999"), hostGrants), true);
    assert.equal(module.loaded.storedOriginAllowed({}, new URL("https://localhost:9999"), hostGrants), false);
    assert.equal(module.loaded.storedOriginAllowed({}, new URL("http://127.0.0.1:9999"), hostGrants), false);
    assert.equal(module.loaded.storedOriginAllowed({}, new URL("http://api.localhost:9999"), hostGrants), false);
    assert.equal(module.loaded.storedOriginAllowed({}, source, { version: 2, grants: {} }), false);
    assert.equal(module.loaded.loopbackHostGrantLabel(key), "http://localhost");
  } finally { await module.close(); }
});

test("settings list and revoke exact and all-port grants independently", async () => {
  const module = await load("src/options/grant-list.ts");
  try {
    const key = JSON.stringify(["http:", "localhost"]);
    const origins = { "http://localhost:5173": "allow" };
    const hostGrants = {
      version: 1,
      grants: { [key]: { scope: "all_ports", createdAt: 1 } },
    };
    const rows = module.loaded.grantRows(origins, hostGrants);
    assert.deepEqual(rows.map((row) => row.label), [
      "http://localhost · all ports",
      "http://localhost:5173 · this port",
    ]);
    assert.equal(hostGrants.grants[key].scope, "all_ports", "broader grant remains");
    assert.equal(rows[0].key, key, "host revoke targets the canonical authority");
    const accessibleNames = rows.map(module.loaded.removeGrantAccessibleName);
    assert.deepEqual(accessibleNames, [
      "Remove all-ports grant for http://localhost",
      "Remove this-port grant for http://localhost:5173",
    ]);
    assert.equal(new Set(accessibleNames).size, rows.length, "each Remove control is distinguishable");
    const compactRows = module.loaded.grantRows({}, hostGrants);
    assert.deepEqual(compactRows.map((row) => row.label), ["http://localhost · all ports"]);
    assert.equal(origins["http://localhost:5173"], "allow", "exact grant remains");

    assert.deepEqual(
      module.loaded.grantRows(origins, { hostGrants: { version: 2, grants: null } }),
      [{ key: "http://localhost:5173", label: "http://localhost:5173 · this port", scope: "origin" }],
      "malformed host state must not hide exact grants",
    );
  } finally { await module.close(); }
});

test("settings mutation helper reports negative acknowledgements and transport failures", async () => {
  const previousChrome = globalThis.chrome;
  try {
    globalThis.chrome = { runtime: { sendMessage: async () => ({ applied: false }) } };
    let module = await load("src/options/mutation-flow.ts");
    assert.deepEqual(await module.loaded.runWorkerMutation({ event: "x" }, "Done"), {
      ok: false,
      message: "Could not update site access. Try again.",
    });
    await module.close();
    globalThis.chrome.runtime.sendMessage = async () => { throw new Error("worker stopped"); };
    module = await load("src/options/mutation-flow.ts");
    assert.deepEqual(await module.loaded.runWorkerMutation({ event: "x" }, "Done"), {
      ok: false,
      message: "Could not reach the extension worker. Try again.",
    });
    await module.close();
  } finally {
    globalThis.chrome = previousChrome;
  }
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

test("AX compaction walks through ignored wrapper nodes", async () => {
  const module = await load("src/ax-compact.ts");
  try {
    // Real headful Chrome wraps every page's content in ignored generic
    // containers (html/body render as role "none", ignored: true) directly
    // under the RootWebArea. Pruning ignored subtrees therefore drops the
    // whole page — the live one-line-snapshot bug. This mirrors the exact
    // shape captured from Chrome 151 on http://127.0.0.1 test pages.
    const rendered = module.loaded.compactAX([
      { nodeId: "1", role: { value: "RootWebArea" }, name: { value: "Page" }, backendDOMNodeId: 1, childIds: ["2"], properties: [{ name: "focusable", value: { value: true } }] },
      { nodeId: "2", role: { value: "none" }, ignored: true, childIds: ["3"] },
      { nodeId: "3", role: { value: "none" }, ignored: true, childIds: ["4", "5"] },
      { nodeId: "4", role: { value: "heading" }, name: { value: "Title" } },
      { nodeId: "5", role: { value: "link" }, name: { value: "More" }, backendDOMNodeId: 9 },
    ], 3);
    const lines = rendered.snapshot.split("\n");
    assert.equal(lines.length, 3, "ignored wrappers must not hide the page");
    // Children of an ignored wrapper indent relative to the last RENDERED
    // ancestor, not the wrapper chain: depth stays flat through them.
    assert.match(lines[1], /^  - heading "Title"$/);
    assert.match(lines[2], /^  - link "More" \[e2\]$/);
    assert.deepEqual(rendered.refs.e2, { backendNodeId: 9, epoch: 3 });
  } finally { await module.close(); }
});

test("AX compaction terminates on cyclic and duplicated childIds", async () => {
  const module = await load("src/ax-compact.ts");
  try {
    // The walk trusts protocol data; a malformed payload with a cycle
    // (2 -> 3 -> 2) or the same child listed twice must neither hang the
    // service worker nor emit duplicate lines (review round 1, MINOR-1).
    const rendered = module.loaded.compactAX([
      { nodeId: "1", role: { value: "RootWebArea" }, name: { value: "Page" }, childIds: ["2", "2"] },
      { nodeId: "2", role: { value: "heading" }, name: { value: "Loop" }, childIds: ["3"] },
      { nodeId: "3", role: { value: "link" }, name: { value: "Back" }, backendDOMNodeId: 4, childIds: ["2"] },
    ], 1);
    assert.deepEqual(rendered.snapshot.split("\n"), [
      '- RootWebArea "Page"',
      '  - heading "Loop"',
      '    - link "Back" [e1]',
    ]);
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
    // The once-ack names the async grant's real semantics: the NEXT visit,
    // within the 10-minute grant window (n2 — the old "The agent is
    // continuing." copy was written for the in-flight navigation case).
    assert.match(once.sub, /next visit/);
    assert.match(once.sub, /10 minutes/);
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

test("access request verdicts: idempotent repeat, replace on new origin, deny cool-down", async () => {
  const module = await load("src/access-flow.ts");
  try {
    const { requestVerdict, newRequest, ACCESS_REQUEST_TTL_MS } = module.loaded;
    const now = 1_000_000;
    const record = newRequest("https://a.example", "a.example", "req-A", now);
    // Already allowed (stored or once-grant) short-circuits without a prompt.
    assert.equal(requestVerdict(undefined, true, false, "https://a.example", "req-A", now), "allowed");
    assert.equal(requestVerdict(undefined, false, true, "https://a.example", "req-A", now), "allowed");
    // No record: raise a fresh prompt.
    assert.equal(requestVerdict(undefined, false, false, "https://a.example", "req-A", now), "raise");
    // Repeat for the SAME pending origin BY THE SAME requester is idempotent —
    // pending, TTL kept (no polling-extension).
    assert.equal(requestVerdict(record, false, false, "https://a.example", "req-A", now + 1), "pending");
    // A DIFFERENT origin replaces (single popup slot), never queues.
    assert.equal(requestVerdict(record, false, false, "https://b.example", "req-A", now + 1), "raise");
    // The SAME origin from a DIFFERENT requester replaces too — the displaced
    // requester reads "superseded" (B1b), never a silent steal.
    assert.equal(requestVerdict(record, false, false, "https://a.example", "req-B", now + 1), "raise");
    // A fresh deny answers denied without re-prompting (no nagging retries)...
    const denied = { ...record, decision: "deny" };
    assert.equal(requestVerdict(denied, false, false, "https://a.example", "req-A", now + 1), "denied");
    // ...until the TTL cool-down lapses, when a deliberate re-ask may raise.
    assert.equal(
      requestVerdict(denied, false, false, "https://a.example", "req-A", now + ACCESS_REQUEST_TTL_MS),
      "raise",
    );
  } finally { await module.close(); }
});

test("access state machine: pending, resolve paths, TTL expiry, grants, supersession", async () => {
  const module = await load("src/access-flow.ts");
  try {
    const {
      accessState, activeRequest, newRequest, consumableGrant, tombstoneFor, receiptKey,
      ACCESS_REQUEST_TTL_MS,
    } = module.loaded;
    const now = 5_000_000;
    const record = newRequest("https://a.example", "a.example", "req-A", now);
    // Undecided and live: pending — the only state await_access blocks on.
    assert.equal(accessState(record, undefined, false, false, "https://a.example", "req-A", now + 1), "pending");
    // Each decision resolves to its terminal state.
    assert.equal(
      accessState({ ...record, decision: "once" }, undefined, false, false, "https://a.example", "req-A", now),
      "allowed",
    );
    assert.equal(
      accessState({ ...record, decision: "always" }, undefined, false, false, "https://a.example", "req-A", now),
      "allowed",
    );
    assert.equal(
      accessState({ ...record, decision: "deny" }, undefined, false, false, "https://a.example", "req-A", now),
      "denied",
    );
    // Past the TTL the record reads as absent — "none", never a stale pending.
    const later = now + ACCESS_REQUEST_TTL_MS;
    assert.equal(activeRequest(record, later), undefined);
    assert.equal(accessState(record, undefined, false, false, "https://a.example", "req-A", later), "none");
    // A record for another origin is not this origin's request.
    assert.equal(accessState(record, undefined, false, false, "https://b.example", "req-A", now), "none");
    // Requester-bound grants: live for the owner inside the window; dead past
    // it; INVISIBLE to another requester and to an anonymous caller (B1a).
    const grants = { "https://a.example": { expiresAt: now + 60_000, requester: "req-A" } };
    assert.ok(consumableGrant(grants, "https://a.example", "req-A", now));
    assert.equal(consumableGrant(grants, "https://a.example", "req-A", now + 60_000), undefined);
    assert.equal(consumableGrant(grants, "https://a.example", "req-B", now), undefined);
    assert.equal(consumableGrant(grants, "https://a.example", "", now), undefined);
    assert.equal(consumableGrant(grants, "https://b.example", "req-A", now), undefined);
    // Supersession: the displaced requester reads "superseded" from its OWN
    // receipt (keyed origin+requester — round-2 M1); anyone else reads the
    // neutral "none"; past the tombstone's TTL the receipt is gone too.
    const tomb = tombstoneFor(record);
    const tombs = { [receiptKey("https://a.example", "req-A")]: tomb };
    assert.equal(accessState(undefined, tombs, false, false, "https://a.example", "req-A", now), "superseded");
    assert.equal(accessState(undefined, tombs, false, false, "https://a.example", "req-B", now), "none");
    assert.equal(accessState(undefined, tombs, false, false, "https://a.example", "req-A", later), "none");
    // Requester-aware live verdicts (round-2 M1): a record resolved by A
    // answers ONLY A. B asking about the same origin gets its receipt or
    // none — never A's pending/allowed/denied.
    const resolvedByA = { ...record, decision: "once" };
    assert.equal(accessState(resolvedByA, undefined, false, false, "https://a.example", "req-B", now), "none");
    assert.equal(accessState(record, undefined, false, false, "https://a.example", "req-B", now), "none");
  } finally { await module.close(); }
});

test("approval queue selection, generation, expiry, and result bounds", async () => {
  const module = await load("src/access-queue.ts");
  try {
    const {
      ACCESS_RESULT_CAP, adjacentEntryId, cleanResults, liveQueue, newEntry,
      receiptFor, selectEntry,
    } = module.loaded;
    const now = 10_000;
    const a1 = newEntry("https://a.example", "a.example", "A", "async", now, 1, undefined, "A1");
    const b = newEntry("https://b.example", "b.example", "B", "async", now, 2, undefined, "B");
    const a2 = newEntry("https://a.example", "a.example", "A", "async", now, 3, undefined, "A2");
    const queue = [a1, b, a2];
    assert.equal(selectEntry(queue, "B").entryId, "B");
    assert.equal(adjacentEntryId(queue, "B", -1), "A1");
    assert.equal(adjacentEntryId(queue, "B", 1), "A2");
    assert.equal(adjacentEntryId(queue, "A1", -1), "A1");
    assert.equal(selectEntry(queue.filter((entry) => entry.entryId !== "A1"), "A1").entryId, "B");
    assert.notEqual(a1.entryId, a2.entryId, "A→B→A generations remain distinct");
    assert.equal(liveQueue(queue, a1.expiresAt).length, 0);
    const results = {};
    for (let index = 0; index < ACCESS_RESULT_CAP + 5; index++) {
      const entry = { ...a1, entryId: String(index) };
      results[String(index)] = { ...receiptFor(entry, "denied", now + index), expiresAt: now + 100_000 };
    }
    assert.equal(Object.keys(cleanResults(results, now)).length, ACCESS_RESULT_CAP);
    assert.equal(Object.keys(cleanResults(results, now + 100_000)).length, 0);
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

test("rejected decision notice skips the interstitial for fallback renders", async () => {
  const module = await load("src/popup/origin-flow.ts");
  try {
    const { noticeForRejectedDecision } = module.loaded;
    // An EMPTY prompt id means the popup rendered from the /health fallback —
    // the request was never "replaced", so no interstitial: the origin
    // fallback already retried the click, and looping the notice on every
    // click was the reported bug.
    assert.equal(noticeForRejectedDecision("", true), null);
    assert.equal(noticeForRejectedDecision("", false), null);
    // A real miss with the origin still pending under a NEWER generation.
    assert.deepEqual(noticeForRejectedDecision("gen-1", true), {
      title: "Request changed.",
      sub: "The site request was replaced while this window was open. Review the new request.",
    });
    // A real miss with the origin gone from the live queue entirely.
    assert.deepEqual(noticeForRejectedDecision("gen-1", false), {
      title: "Request expired.",
      sub: "It timed out or was cancelled, so nothing was granted or denied.",
    });
  } finally { await module.close(); }
});

test("surface tokens resolve only with an exact nonce-bearing handle", async () => {
  const module = await load("src/state.ts");
  try {
    const { surfaceToken, parseSurface, resolveSurfaceToken, atSurfaceCap, MAX_SURFACES } = module.loaded;
    const surface = { tabId: 42, nonce: "abc123", epoch: 3, createdAt: 1, lastUsedAt: 2 };
    const token = surfaceToken(surface);
    assert.equal(token, "bridge:42:abc123");
    assert.deepEqual(parseSurface(token), { tabId: 42, nonce: "abc123" });
    const surfaces = { [token]: surface };
    // Exact token resolves; a guessed nonce or the bare tab id does not — the
    // nonce is the anti-guessing property that keeps parallel sessions from
    // driving each other's tabs.
    assert.equal(resolveSurfaceToken(token, surfaces), surface);
    assert.equal(resolveSurfaceToken("bridge:42:guessed", surfaces), undefined);
    assert.equal(resolveSurfaceToken("bridge:42", surfaces), undefined);
    assert.equal(resolveSurfaceToken(42, surfaces), undefined);
    // Cap math: at MAX_SURFACES entries a fresh open must be refused.
    const many = {};
    for (let i = 0; i < MAX_SURFACES; i += 1) many[`bridge:${i}:n${i}`] = { ...surface, tabId: i };
    assert.equal(atSurfaceCap(surfaces), false);
    assert.equal(atSurfaceCap(many), true);
  } finally { await module.close(); }
});

test("redacted handles are recognizable by their owner but not driveable", async () => {
  const module = await load("src/state.ts");
  try {
    const { surfaceToken, redactToken, ownsRedacted, resolveSurfaceToken } = module.loaded;
    const surface = { tabId: 42, nonce: "abcdef0123456789abcdef0123456789", epoch: 1, createdAt: 1, lastUsedAt: 2 };
    const token = surfaceToken(surface);
    const redacted = redactToken(token);
    // Truncated to a 6-char prefix + ellipsis: enough to prefix-match, far
    // too little to reconstruct the 32-char nonce (finding M1).
    assert.equal(redacted, "bridge:42:abcdef…");
    // The owner (holding the full token) recognises the listing entry...
    assert.equal(ownsRedacted(token, redacted), true);
    // ...another session's token does not...
    assert.equal(ownsRedacted("bridge:42:ffffff0123456789abcdef0123456789", redacted), false);
    // ...and the redacted form itself resolves NOTHING: it is not a handle.
    assert.equal(resolveSurfaceToken(redacted, { [token]: surface }), undefined);
    // Unredacted comparison still exact-matches (defensive path).
    assert.equal(ownsRedacted(token, token), true);
  } finally { await module.close(); }
});

test("reconnect timing: alarm is the guaranteed floor, setTimeout the alive-only fast path", async () => {
  const module = await load("src/reconnect.ts");
  try {
    const {
      RECONNECT_ALARM_PERIOD_MINUTES,
      MAX_BACKOFF_MS,
      backoffDelayMs,
      shouldArmFastPath,
      shouldDialOnAlarm,
    } = module.loaded;

    // The alarm period must sit ABOVE Chrome's 30s clamp. Below 0.5 min Chrome
    // refuses to honour the period, and 0.5 min itself sits exactly on the clamp
    // edge where the tick can be dropped/delayed (the original bug) — so a
    // `>= 0.5` assertion would re-admit the buggy edge. Require strictly above.
    assert.ok(RECONNECT_ALARM_PERIOD_MINUTES > 0.5, "alarm period not strictly above Chrome's 30s clamp edge");

    // Fast-path backoff is exponential and capped so a dead daemon is not
    // hammered while a live socket still recovers in seconds.
    assert.equal(backoffDelayMs(0), 1_000);
    assert.equal(backoffDelayMs(3), 8_000);
    assert.equal(backoffDelayMs(99), MAX_BACKOFF_MS);

    // The setTimeout fast path arms ONLY while the worker is alive (a suspended
    // worker cannot run it — the alarm covers that) and never stacks a second
    // pending timer.
    assert.equal(shouldArmFastPath({ alive: true, fastPathPending: false }), true);
    assert.equal(shouldArmFastPath({ alive: true, fastPathPending: true }), false);
    assert.equal(shouldArmFastPath({ alive: false, fastPathPending: false }), false);

    // The guaranteed-wake alarm dials whenever the socket is neither connected
    // nor mid-dial. This is the cold-wake-after-suspension case: globals have
    // reset to false, so the alarm re-dials with NO page interaction — the fix.
    assert.equal(shouldDialOnAlarm({ connected: false, connecting: false }), true);
    // ...but stays a no-op when a socket is up or a dial is already in flight,
    // so the alarm never storms a second socket past connect()'s own guard.
    assert.equal(shouldDialOnAlarm({ connected: true, connecting: false }), false);
    assert.equal(shouldDialOnAlarm({ connected: false, connecting: true }), false);
  } finally { await module.close(); }
});

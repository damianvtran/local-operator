import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { mkdtemp, readFile, rm } from "node:fs/promises";
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

test("registrable domain follows the bundled Public Suffix List and refuses unbounded keys", async () => {
  const module = await load("src/origin-policy.ts");
  try {
    const { registrableDomain, broadGrantFor } = module.loaded;
    const domain = (href) => registrableDomain(new URL(href));
    // Ordinary ICANN suffixes.
    assert.equal(domain("https://qa-app.qa.gominerva.com"), "gominerva.com");
    assert.equal(domain("https://app.gominerva.com"), "gominerva.com");
    assert.equal(domain("https://gominerva.com"), "gominerva.com");
    assert.equal(domain("https://GoMinerva.COM:8443/x"), "gominerva.com");
    // Multi-label public suffix.
    assert.equal(domain("https://foo.bar.co.uk"), "bar.co.uk");
    assert.equal(domain("https://bar.co.uk"), "bar.co.uk");
    assert.equal(domain("https://co.uk"), null);
    // Wildcard rule `*.ck` and its exception `!www.ck`.
    assert.equal(domain("https://foo.bar.ck"), "foo.bar.ck");
    assert.equal(domain("https://bar.ck"), null);
    assert.equal(domain("https://www.ck"), "www.ck");
    assert.equal(domain("https://a.www.ck"), "www.ck");
    // PRIVATE section: a tenant on shared hosting is its own domain.
    assert.equal(domain("https://me.github.io"), "me.github.io");
    assert.equal(domain("https://github.io"), null);
    // IDN rule matched through punycode.
    assert.equal(domain("https://\u0441\u0430\u0439\u0442.\u0440\u0444"), "xn--80aswg.xn--p1ai");
    // No domain option at all for these: a grant would be unbounded or
    // meaningless.
    for (const href of [
      "http://10.0.0.5",
      "http://[2001:db8::1]:8080",
      "http://intranet",
      "https://example.com.",
      "https://com",
    ]) assert.equal(domain(href), null, href);
    // Loopback gets a host grant instead; an IP literal gets nothing.
    assert.deepEqual(broadGrantFor(new URL("http://localhost:5173")), { scope: "host", key: "localhost" });
    assert.deepEqual(broadGrantFor(new URL("http://[::1]:8000")), { scope: "host", key: "[::1]" });
    assert.deepEqual(broadGrantFor(new URL("https://qa-app.qa.gominerva.com")), { scope: "domain", key: "gominerva.com" });
    assert.equal(broadGrantFor(new URL("http://10.0.0.5")), null);
  } finally { await module.close(); }
  const psl = await load("src/psl.gen.ts");
  try {
    assert.ok(psl.loaded.PSL_RULE_COUNT > 9000, `bundled list looks truncated: ${psl.loaded.PSL_RULE_COUNT}`);
    assert.equal(psl.loaded.PSL_RULES.split("\n").length, psl.loaded.PSL_RULE_COUNT);
    assert.match(psl.loaded.PSL_GENERATED_AT, /^\d{4}-\d{2}-\d{2}$/);
  } finally { await psl.close(); }
});

test("site grants admit by domain or loopback host, in lookup order, and fail closed", async () => {
  const module = await load("src/origin-policy.ts");
  try {
    const { matchingGrantScope, storedOriginAllowed } = module.loaded;
    const siteGrants = {
      version: 1,
      grants: {
        "gominerva.com": { scope: "domain", createdAt: 1 },
        localhost: { scope: "host", createdAt: 1 },
      },
    };
    const NONE = Symbol("no site grants");
    const scope = (href, origins = {}, hostGrants, grants = siteGrants) =>
      matchingGrantScope(origins, hostGrants, new URL(href), grants === NONE ? undefined : grants);
    // Exact origin wins first.
    assert.equal(scope("https://gominerva.com", { "https://gominerva.com": "allow" }), "origin");
    // Domain grant: every subdomain, both schemes, any port.
    assert.equal(scope("https://qa-app.qa.gominerva.com"), "domain");
    assert.equal(scope("https://gominerva.com"), "domain");
    assert.equal(scope("http://gominerva.com"), "domain");
    assert.equal(scope("https://gominerva.com:8443"), "domain");
    assert.equal(scope("https://gominerva.co"), null);
    assert.equal(scope("https://notgominerva.com"), null);
    assert.equal(scope("https://gominerva.com.evil.example"), null);
    // Host grant: any port, both schemes, literal hostname only.
    assert.equal(scope("http://localhost:9999"), "host");
    assert.equal(scope("https://localhost:9999"), "host");
    assert.equal(scope("http://127.0.0.1:9999"), null);
    assert.equal(scope("http://api.localhost:9999"), null);
    // Legacy v1 loopback grant is still honoured, still same-scheme only,
    // and ranks after the site grants.
    const legacy = { version: 1, grants: { '["http:","localhost"]': { scope: "all_ports", createdAt: 1 } } };
    assert.equal(scope("http://localhost:1", {}, legacy, NONE), "loopback_all_ports");
    assert.equal(scope("https://localhost:1", {}, legacy, NONE), null);
    assert.equal(scope("http://localhost:1", {}, legacy), "host");
    // Fail closed: unknown version, malformed entry, wrong scope for key.
    assert.equal(scope("https://gominerva.com", {}, undefined, { version: 2, grants: siteGrants.grants }), null);
    assert.equal(scope("https://gominerva.com", {}, undefined, { version: 1, grants: { "gominerva.com": { scope: "domain" } } }), null);
    assert.equal(scope("https://gominerva.com", {}, undefined, { version: 1, grants: { "gominerva.com": { scope: "host", createdAt: 1 } } }), null);
    assert.equal(scope("https://gominerva.com", {}, undefined, { version: 1, grants: [] }), null);
    assert.equal(storedOriginAllowed({}, new URL("https://x.gominerva.com"), undefined, siteGrants), true);
    // A "deny" verdict is typed but never written; it is simply not an allow.
    assert.equal(scope("https://gominerva.com", { "https://gominerva.com": "deny" }, undefined, NONE), null);
  } finally { await module.close(); }
});

test("policyCovers reconciles by domain and loopback host regardless of scheme", async () => {
  const module = await load("src/access-queue.ts");
  try {
    const { policyCovers } = module.loaded;
    assert.equal(policyCovers("https://qa-app.qa.gominerva.com", "http://app.gominerva.com:8080", "domain"), true);
    assert.equal(policyCovers("https://gominerva.com", "https://gominerva.co", "domain"), false);
    assert.equal(policyCovers("http://10.0.0.5", "http://10.0.0.5", "domain"), false, "no domain, no coverage");
    assert.equal(policyCovers("http://localhost:3000", "https://localhost:5173", "host"), true);
    assert.equal(policyCovers("http://localhost:3000", "http://127.0.0.1:5173", "host"), false);
    assert.equal(policyCovers("http://example.com", "http://example.com:81", "host"), false);
    assert.equal(policyCovers("http://localhost:3000", "https://localhost:5173", "loopback_all_ports"), false, "legacy stays same-scheme");
    assert.equal(policyCovers("https://a.example", "https://a.example"), true);
  } finally { await module.close(); }
});

test("normalizedSiteGrants refuses any record this build cannot preserve", async () => {
  const module = await load("src/access-grants.ts");
  try {
    const { normalizedSiteGrants } = module.loaded;
    const good = { version: 1, grants: { "gominerva.com": { scope: "domain", createdAt: 1 }, "[::1]": { scope: "host", createdAt: 2 } } };
    assert.deepEqual(normalizedSiteGrants(good), good);
    for (const bad of [
      { version: 2, grants: {} },
      { version: 1, grants: { "gominerva.com": { scope: "domain", createdAt: "1" } } },
      { version: 1, grants: { "co.uk": { scope: "domain", createdAt: 1 } } },
      { version: 1, grants: { "www.gominerva.com": { scope: "domain", createdAt: 1 } } },
      { version: 1, grants: { "example.com": { scope: "host", createdAt: 1 } } },
      { version: 1, grants: { "https://gominerva.com": { scope: "domain", createdAt: 1 } } },
      { version: 1, grants: { "gominerva.com": { scope: "all_ports", createdAt: 1 } } },
    ]) assert.equal(normalizedSiteGrants(bad), null, JSON.stringify(bad));
  } finally { await module.close(); }
});

test("popup scope options come from the entry and default to the broad grant when present", async () => {
  const module = await load("src/popup/origin-flow.ts");
  try {
    const { scopeOptions } = module.loaded;
    const domain = scopeOptions({ origin: "https://qa-app.qa.gominerva.com", broad: { scope: "domain", key: "gominerva.com" } });
    assert.deepEqual(domain.options.map((option) => option.value), ["domain", "site", "once"]);
    assert.equal(domain.defaultValue, "domain");
    assert.deepEqual(domain.options.map((option) => option.label), ["All pages on this domain", "Only this site", "Just this once"]);
    assert.deepEqual(domain.options.map((option) => option.detail), [
      "gominerva.com and every subdomain, any port",
      "https://qa-app.qa.gominerva.com",
      "one navigation within 10 minutes",
    ]);
    const host = scopeOptions({ origin: "http://localhost:5173", broad: { scope: "host", key: "localhost" } });
    assert.equal(host.options[0].label, "Any port on this host");
    assert.equal(host.options[0].detail, "localhost, any port");
    assert.equal(host.defaultValue, "domain");
    // No broad field (IP literal, a 0.1.7 entry, or a /health-only render):
    // no domain option, and the narrow grant is the default.
    for (const entry of [{ origin: "http://10.0.0.5" }, undefined]) {
      const narrow = scopeOptions(entry);
      assert.deepEqual(narrow.options.map((option) => option.value), ["site", "once"]);
      assert.equal(narrow.defaultValue, "site");
    }
    for (const option of [...domain.options, ...host.options]) {
      assert.doesNotMatch(option.label, /\u2014/);
      assert.doesNotMatch(option.detail, /\u2014/);
    }
  } finally { await module.close(); }
});

test("allow-all switch writes only after an acknowledged Enable and reverts on cancel", async () => {
  const module = await load("src/options/allow-all-flow.ts");
  try {
    const { nextAllowAllView, allowAllView } = module.loaded;
    const off = allowAllView(false);
    assert.deepEqual(off, { switchOn: false, dialogOpen: false, acked: false, banner: false });
    // Checking opens the dialog, shows the switch on, writes nothing.
    const opened = nextAllowAllView(false, { type: "toggle", checked: true }, off);
    assert.deepEqual(opened, { switchOn: true, dialogOpen: true, acked: false, banner: false });
    // Enable is inert until acked.
    assert.deepEqual(nextAllowAllView(false, { type: "enable" }, opened), opened);
    const acked = nextAllowAllView(false, { type: "ack", checked: true }, opened);
    assert.equal(acked.acked, true);
    assert.equal(acked.write, undefined);
    // Cancel (or Escape) reverts with no write.
    assert.deepEqual(nextAllowAllView(false, { type: "cancel" }, acked), off);
    // Ack then Enable is the only way to write true.
    const enabled = nextAllowAllView(false, { type: "enable" }, acked);
    assert.deepEqual(enabled, { switchOn: true, dialogOpen: false, acked: false, banner: true, write: true });
    // Unchecking, or the banner's Turn off, writes false with no dialog.
    const on = allowAllView(true);
    assert.deepEqual(nextAllowAllView(true, { type: "toggle", checked: false }, on), { ...off, write: false });
    assert.deepEqual(nextAllowAllView(true, { type: "turn_off" }, on), { ...off, write: false });
    // Re-checking an already-on switch is a no-op.
    assert.deepEqual(nextAllowAllView(true, { type: "toggle", checked: true }, on), on);
  } finally { await module.close(); }
});

// The all-sites bypass must stay reachable ONLY from the options page: no
// wire method and no worker message may set it, or a daemon-side caller (an
// agent) could grant itself every site. This pins that property so a future
// "convenience" RPC fails a test rather than a review.
test("no wire method or worker message can set the all-sites bypass", async () => {
  const worker = await readFile(new URL("../src/worker.ts", import.meta.url), "utf8");
  const table = worker.match(/const HANDLERS[\s\S]*?=\s*\{([\s\S]*?)\n\};/);
  assert.ok(table, "worker.ts must declare the HANDLERS table");
  const handlerNames = table[1]
    .split("\n")
    .filter((line) => !line.trim().startsWith("//"))
    .flatMap((line) => [...line.matchAll(/^\s{2}([A-Za-z_][A-Za-z0-9_]*)\s*[,:]/g)].map((m) => m[1]));
  assert.ok(handlerNames.length > 10, `parsed too few handlers: ${handlerNames}`);
  assert.deepEqual(handlerNames.filter((name) => /allow|site|grant/i.test(name)), []);
  const events = [...worker.matchAll(/message\?\.event === "([a-z_]+)"/g)].map((m) => m[1]);
  assert.ok(events.includes("site_grant_revoke"), "revoke path is a worker message");
  assert.deepEqual(events.filter((name) => /allow_all|allowAll/i.test(name)), []);
  const generated = await readFile(new URL("../src/protocol.gen.ts", import.meta.url), "utf8");
  assert.doesNotMatch(generated, /allow_all|allowAllSites/);
  // The only writer is the options page, as a plain storage write.
  const writers = [];
  for (const file of ["../src/worker.ts", "../src/origins.ts", "../src/approval-store.ts", "../src/access-grants.ts", "../src/commands/access.ts", "../src/popup/popup.ts"]) {
    const source = await readFile(new URL(file, import.meta.url), "utf8");
    if (/set\(\{[^}]*allowAllSites/.test(source)) writers.push(file);
  }
  assert.deepEqual(writers, []);
  const options = await readFile(new URL("../src/options/options.ts", import.meta.url), "utf8");
  assert.match(options, /chrome\.storage\.local\.set\(\{ allowAllSites: view\.write \}\)/);
});

test("settings list labels and revokes every grant scope independently", async () => {
  const module = await load("src/options/grant-list.ts");
  try {
    const key = JSON.stringify(["http:", "localhost"]);
    const origins = { "http://localhost:5173": "allow" };
    const hostGrants = {
      version: 1,
      grants: { [key]: { scope: "all_ports", createdAt: 1 } },
    };
    const siteGrants = {
      version: 1,
      grants: {
        "gominerva.com": { scope: "domain", createdAt: 1 },
        "127.0.0.1": { scope: "host", createdAt: 1 },
      },
    };
    const rows = module.loaded.grantRows(origins, hostGrants, siteGrants);
    assert.deepEqual(rows.map((row) => row.label), [
      "127.0.0.1 · any port",
      "gominerva.com · all subdomains, any port",
      "http://localhost · all ports (http only)",
      "http://localhost:5173 · this site",
    ]);
    assert.deepEqual(rows.map((row) => row.scope), ["host", "domain", "legacy_host", "origin"]);
    assert.equal(hostGrants.grants[key].scope, "all_ports", "legacy grant remains");
    assert.equal(rows[2].key, key, "legacy revoke targets the canonical authority");
    const accessibleNames = rows.map(module.loaded.removeGrantAccessibleName);
    assert.deepEqual(accessibleNames, [
      "Remove any-port grant for 127.0.0.1",
      "Remove domain grant for gominerva.com",
      "Remove legacy all-ports grant for http://localhost",
      "Remove this-site grant for http://localhost:5173",
    ]);
    assert.equal(new Set(accessibleNames).size, rows.length, "each Remove control is distinguishable");
    // Each scope lives in its own storage record and so has its own revoke.
    assert.deepEqual(rows.map(module.loaded.revokeMessageFor), [
      { event: "site_grant_revoke", key: "127.0.0.1" },
      { event: "site_grant_revoke", key: "gominerva.com" },
      { event: "host_grant_revoke", canonicalKey: key },
      { event: "origin_grant_revoke", origin: "http://localhost:5173" },
    ]);
    const compactRows = module.loaded.grantRows({}, hostGrants);
    assert.deepEqual(compactRows.map((row) => row.label), ["http://localhost · all ports (http only)"]);
    assert.equal(origins["http://localhost:5173"], "allow", "exact grant remains");

    assert.deepEqual(
      module.loaded.grantRows(origins, { hostGrants: { version: 2, grants: null } }, { version: 2, grants: {} }),
      [{ key: "http://localhost:5173", label: "http://localhost:5173 · this site", scope: "origin" }],
      "malformed or unknown-version broad state must not hide exact grants",
    );
    assert.deepEqual(
      module.loaded.grantRows({}, undefined, { version: 1, grants: { "x.com": { scope: "wildcard", createdAt: 1 } } }),
      [],
      "an unknown site-grant scope renders no row",
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
    // "site" and "domain" are standing grants: the ack must say exactly what
    // persists and where to revoke it.
    const site = ackForDecision("site");
    assert.equal(site.title, "Site allowed.");
    assert.equal(site.tone, "success");
    assert.match(site.sub, /This exact site \(address and port\) stays allowed; take it back any time in Settings\./);
    const domain = ackForDecision("domain", "domain");
    assert.equal(domain.title, "Domain allowed.");
    assert.match(domain.sub, /Every page on this domain and its subdomains, on any port, stays allowed/);
    const host = ackForDecision("domain", "host");
    assert.equal(host.title, "Domain allowed.");
    assert.match(host.sub, /Every port on this loopback host stays allowed/);
    for (const ack of [site, domain, host]) assert.doesNotMatch(ack.sub, /\u2014/, "no em dashes in user copy");
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
      accessState({ ...record, decision: "site" }, undefined, false, false, "https://a.example", "req-A", now),
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
    const decided = { origin: "https://example.com", decision: "site" };
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
    const { surfaceToken, redactToken, ownsRedacted, resolveSurfaceToken, isRedactedToken } =
      module.loaded;
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
    // A redacted token is DETECTABLE as such, which is what stops worker.ts
    // forwarding it as a tab_update handle. Doing so keyed a second driven
    // record for one tab, and that duplicate outlived the real close as a
    // phantom advertising a dead URL — the ghost this release removes.
    assert.equal(isRedactedToken(redacted), true);
    assert.equal(isRedactedToken(token), false);
    assert.equal(isRedactedToken(""), false);
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

// The wire protocol's own docstring (local_operator/browser_bridge/protocol.py)
// promises that "a method added here without a handler fails a test rather than
// timing out on the wire". Python's side of that is `set(METHODS) ==
// set(COMMAND_TIMEOUTS)`; nothing checked the EXTENSION side, so a method could
// reach the generated union with no handler behind it and only be discovered as
// an opaque timeout against a real browser. This closes that half (review round
// 1, R4).
test("every wire method has a worker handler", async () => {
  const generated = await readFile(new URL("../src/protocol.gen.ts", import.meta.url), "utf8");
  const union = generated.match(/export type Method = ([^;]+);/);
  assert.ok(union, "protocol.gen.ts must declare the Method union");
  const methods = [...union[1].matchAll(/'([^']+)'/g)].map((m) => m[1]);
  assert.ok(methods.length > 10, `parsed too few methods: ${methods}`);

  const worker = await readFile(new URL("../src/worker.ts", import.meta.url), "utf8");
  const table = worker.match(/const HANDLERS[\s\S]*?=\s*\{([\s\S]*?)\n\};/);
  assert.ok(table, "worker.ts must declare the HANDLERS table");
  // Keys are either `name,` (shorthand) or `name: fn,`; comments are ignored
  // because they legitimately mention method names in prose.
  //
  // The `\s{2}` anchors on the table's two-space indentation, which is a real
  // dependency on formatting. It FAILS SAFE: a reformat to four spaces yields
  // an empty handler set, so the assertion below reports every method as
  // unhandled rather than silently passing — loud and diagnosable, which is why
  // the regex is acceptable here instead of a TypeScript parser (review round
  // 2, N6). If you are here because this test failed after a reformat, widen
  // the indentation class rather than deleting the check.
  const handlers = new Set(
    table[1]
      .split("\n")
      .filter((line) => !line.trim().startsWith("//"))
      .flatMap((line) => [...line.matchAll(/^\s{2}([A-Za-z_][A-Za-z0-9_]*)\s*[,:]/g)].map((m) => m[1])),
  );

  const missing = methods.filter((method) => !handlers.has(method));
  assert.deepEqual(missing, [], `wire methods with no handler: ${missing}`);
});

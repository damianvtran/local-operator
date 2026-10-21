import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { ENFORCED, GENERATED_SOURCES, scanSource, fingerprints, NONTRANSLATABLE } from "./check-i18n.mjs";
const script = fileURLToPath(new URL("./check-i18n.mjs", import.meta.url));
const root = dirname(dirname(script));
function fixture(source, baselineSource = source, scope = "src/store.ts") {
	const directory = mkdtempSync(join(process.env.LOCAL_OPERATOR_SCRATCHPAD ?? tmpdir(), "relay-i18n-mutation-"));
	for (const path of ENFORCED) { mkdirSync(dirname(join(directory, path)), { recursive: true }); writeFileSync(join(directory, path), path === scope ? source : ""); }
	writeFileSync(join(directory, "i18n-baseline.json"), JSON.stringify({ schema: 1, enforced: ENFORCED, legacy: { [scope]: fingerprints(scanSource(baselineSource, scope)) } }));
	return { directory, run: () => spawnSync(process.execPath, [script, "--root", directory], { encoding: "utf8" }) };
}
test("template substitutions, JSX and DOM copy assignments fail actual mandatory gate", () => {
	for (const source of [
		'const message = `${"New visible copy"}`;',
		'const view = <span>{`${"New visible copy"}`}</span>;',
		'const message = `${["New visible copy"].join("")}`;',
		'const message = `${decorate("New visible copy")}`;',
		'document.title = "New visible copy";',
		'document.body.textContent = `${"New visible copy"}`;',
		'document.body.setAttribute("aria-label", "New visible copy");',
	]) {
		const rig = fixture(source, "", "src/main.tsx");
		try { const result = rig.run(); assert.equal(result.status, 1, source); assert.match(result.stderr, /New visible copy/); }
		finally { rmSync(rig.directory, { recursive: true, force: true }); }
	}
});
test("exact sink/literal multiset rejects equal-count replacement through actual gate", () => {
	const rig = fixture('const label = "Changed visible copy";', 'const label = "Legacy visible copy";');
	try { const result = rig.run(); assert.equal(result.status, 1); assert.match(result.stderr, /Changed visible copy/); }
	finally { rmSync(rig.directory, { recursive: true, force: true }); }
});
test("same-count sink reassignment and duplicate literal growth fail", () => {
	for (const source of ['const message = "Legacy visible copy";', 'const label = "Legacy visible copy"; function second() { const label = "Legacy visible copy"; }']) {
		const rig = fixture(source, 'const label = "Legacy visible copy";');
		try { assert.equal(rig.run().status, 1); } finally { rmSync(rig.directory, { recursive: true, force: true }); }
	}
});
test("JSX, accessible attributes, template copy and errors are scanned; keys/comments/types are technical", () => {
	const source = 'const view = <><button aria-label="Accessible name">Visible copy</button>{`Count ${count} items`}</>; throw new Error("Visible refusal");';
	const findings = scanSource(source, "src/main.tsx");
	for (const text of ["Accessible name", "Visible copy", "Count", "items", "Visible refusal"]) assert(findings.some(item => item.text === text));
	assert.deepEqual(scanSource('/* Developer prose */ const key = "wire.exec.error_line"; type Status = "technical";', "src/store.ts"), []);
	assert.deepEqual(scanSource('throw new Error("catalogue response limit")', "src/i18n/wire-cache.ts"), []);
	assert(NONTRANSLATABLE.every(item => item.reason));
	assert.deepEqual(scanSource('const view = <span>{t("wire.exec.error_line", {error: value})}</span>;', "src/main.tsx"), []);
	assert.deepEqual(scanSource('const view = <span className="technical-css-token"/>;', "src/main.tsx"), []);
});
test("legacy exact match passes; outside-scope source is advisory, not exempted from inventory", () => {
	const rig = fixture('const label = "Legacy visible copy";');
	try { writeFileSync(join(rig.directory, "src/advisory.ts"), 'const message = "Unextracted app copy";'); const result = rig.run(); assert.equal(result.status, 0); assert.match(result.stdout, /1 advisory/); }
	finally { rmSync(rig.directory, { recursive: true, force: true }); }
});
test("a NEW hardcoded string in a touched relay.shell path fails the actual mandatory gate", () => {
	// C activates the two shell screens it extracts from. Their residual
	// (unextracted) copy is frozen as exact legacy fingerprints, so the gate
	// must now refuse a NEW literal in either — the mutation is the future
	// extraction slice's own regression, run through the SAME mandatory entry
	// point the build/check/test scripts use.
	for (const scope of ["src/screens/session-list.tsx", "src/screens/session-view.tsx"]) {
		const baselineSource = readFileSync(join(root, scope), "utf8");
		const mutatedSource = `${baselineSource}\nconst message = "New hardcoded shell copy";\n`;
		const rig = fixture(mutatedSource, baselineSource, scope);
		try {
			const result = rig.run();
			assert.equal(result.status, 1, `${scope} must refuse a new literal`);
			assert.match(result.stderr, /New hardcoded shell copy/);
		} finally { rmSync(rig.directory, { recursive: true, force: true }); }
	}
});
test("a NEW hardcoded string in a touched relay.session path fails the actual mandatory gate", () => {
	// C2 activates the history/start surfaces it extracts from: the session-list
	// body copy sits in a file C already activated, but past-sessions.tsx and the
	// two sheets that carry the retired new-session screen's copy are NEW here.
	// A new literal in any of them must fail the same mandatory entry point.
	for (const scope of ["src/screens/past-sessions.tsx", "src/components/directory-sheet.tsx", "src/components/model-sheet.tsx"]) {
		const baselineSource = readFileSync(join(root, scope), "utf8");
		const mutatedSource = `${baselineSource}\nconst message = "New hardcoded session copy";\n`;
		const rig = fixture(mutatedSource, baselineSource, scope);
		try {
			const result = rig.run();
			assert.equal(result.status, 1, `${scope} must refuse a new literal`);
			assert.match(result.stderr, /New hardcoded session copy/);
		} finally { rmSync(rig.directory, { recursive: true, force: true }); }
	}
});
test("the activated relay.shell screens are exactly the reviewed scope", () => {
	assert(ENFORCED.includes("src/screens/session-list.tsx"));
	assert(ENFORCED.includes("src/screens/session-view.tsx"));
	// agent-view.tsx is a LATER namespace: activating it here would freeze copy
	// this slice deliberately left out.
	assert(!ENFORCED.includes("src/screens/agent-view.tsx"));
	const baseline = JSON.parse(readFileSync(join(root, "i18n-baseline.json"), "utf8"));
	assert.deepEqual([...baseline.enforced].sort(), [...ENFORCED].sort());
	for (const scope of ["src/screens/session-list.tsx", "src/screens/session-view.tsx"]) {
		assert(Object.keys(baseline.legacy[scope] ?? {}).length > 0, `${scope} needs frozen residual fingerprints`);
	}
});
test("the activated relay.session history/start surfaces are exactly the reviewed scope", () => {
	// C2 extracts from these three files beyond the shell screens. The retired
	// `screens/new-session.tsx` is gone from main (removed in 6abef9783c); its
	// copy now lives in the two sheets and the session-list footer, which is why
	// they — not a `new-session.tsx` — are the activated scope here.
	const sessionScope = ["src/screens/past-sessions.tsx", "src/components/directory-sheet.tsx", "src/components/model-sheet.tsx"];
	for (const scope of sessionScope) assert(ENFORCED.includes(scope), `${scope} must be activated`);
	assert(!ENFORCED.includes("src/components/composer.tsx"), "composer is a later namespace");
	assert(!ENFORCED.includes("src/screens/agent-view.tsx"), "agent-view is a later namespace");
	const baseline = JSON.parse(readFileSync(join(root, "i18n-baseline.json"), "utf8"));
	for (const scope of sessionScope) {
		assert(Object.keys(baseline.legacy[scope] ?? {}).length > 0, `${scope} needs frozen residual fingerprints`);
	}
});
test("the activated asks/cards surfaces are exactly the reviewed scope", () => {
	const asksScope = [
		"src/components/ask-card.tsx",
		"src/components/ask-dock.tsx",
		"src/components/ask-row.tsx",
		"src/components/asks-sheet.tsx",
		"src/components/gate-sheet.tsx",
		"src/components/pending-card.tsx",
		"src/components/ui/disclosure.tsx",
		"src/lib/asks.ts",
	];
	for (const scope of asksScope) assert(ENFORCED.includes(scope), `${scope} must be activated`);
	assert(!ENFORCED.includes("src/screens/pair.tsx"), "pair is a later namespace");
	assert(!ENFORCED.includes("src/screens/agent-view.tsx"), "agent-view is a later namespace");
	const baseline = JSON.parse(readFileSync(join(root, "i18n-baseline.json"), "utf8"));
	// This slice extracts the files completely, so any allowance row left behind
	// would be exactly the file exemption the gate forbids.
	for (const scope of asksScope) assert.deepEqual(baseline.legacy[scope], {}, `${scope} must carry no residual allowance`);
});
test("the generated owned-copy bundle is not counted as app debt", () => {
	const rig = fixture("");
	try {
		mkdirSync(join(rig.directory, "src/i18n"), { recursive: true });
		writeFileSync(
			join(rig.directory, "src/i18n/bundled.gen.ts"),
			'export const BUNDLED_OWNED = { "relay.demo": { "relay.demo.title": "Generated bundle copy", "relay.demo.label": "Another generated label" } };',
		);
		const result = rig.run();
		assert.equal(result.status, 0);
		assert.match(result.stdout, /; 0 advisory legacy sink literal\(s\)/);
		assert(GENERATED_SOURCES.has("src/i18n/bundled.gen.ts"));
	} finally { rmSync(rig.directory, { recursive: true, force: true }); }
});
test("build/check/test scripts all execute the mandatory gate", () => {
	const scripts = JSON.parse(readFileSync(join(root, "package.json"), "utf8")).scripts;
	for (const mode of ["build", "check", "test"]) assert(scripts[mode].startsWith("node scripts/check-i18n.mjs &&"));
	assert(scripts.test.includes("node --test scripts/check-i18n.node-check.mjs && vitest run"));
	assert(!/\.(?:test|spec)\.[cm]?[jt]sx?$/.test(fileURLToPath(import.meta.url)), "Node-only checks must not match Vitest default discovery");
});

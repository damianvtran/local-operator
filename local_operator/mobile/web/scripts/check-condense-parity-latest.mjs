#!/usr/bin/env node
/**
 * THE CROSS-REPO DRIFT CHECK FOR THE PARITY FIXTURES.
 *
 * WHY THIS EXISTS. A fixture both sides pin only to ITSELF cannot detect drift: the
 * relay's copy stayed byte-identical to its own hash for nine days while the
 * desktop's copy moved on, because each repository compared its copy with a number
 * stored in the same repository. Copy-plus-local-hash catches a LOCAL edit and
 * nothing else - the exact failure this whole feature was opened to remove
 * (agent review round 1, C1-M1).
 *
 * TWO MODES, and both are answers about different questions:
 *
 * 1. DEFAULT (no flag) - SELF-CONSISTENCY, always runnable, wired into `npm run
 *    check`. It hashes the copies in `src/lib/` and compares them with
 *    `src/lib/parity-sources.json`. A copy edited without its manifest, or a
 *    manifest re-pinned without the copy, fails here. This cannot detect drift from
 *    the desktop; it detects a broken pin, which is the precondition for everything
 *    else.
 *
 * 2. `--latest` - DRIFT AGAINST THE DESKTOP ITSELF. Reads the desktop checkout's own
 *    generated manifest (`scripts/fixtures/condense.parity.manifest.json` in
 *    `local-operator-ui`) and compares its `sha256`/`sourceHash` with ours. When the
 *    desktop model changes and its generator is re-run, this fails until the relay
 *    re-copies the fixture - which is the re-port trigger the design names.
 *
 * WHY `--latest` IS ADVISORY WHEN THE SIBLING IS ABSENT. This repository's CI checks
 * out one repository; a cross-repo comparison there would either fail forever or be
 * deleted. So a missing checkout prints a NAMED notice and exits 0, while a present
 * checkout that DISAGREES exits 1. `--require-sibling` turns the absence into a
 * failure, for a lane that genuinely has both trees (a developer machine, or a job
 * that checks both out).
 *
 * USAGE
 *   node scripts/check-condense-parity-latest.mjs                  # self-consistency
 *   node scripts/check-condense-parity-latest.mjs --latest         # drift vs sibling
 *   node scripts/check-condense-parity-latest.mjs --latest --require-sibling
 *   node scripts/check-condense-parity-latest.mjs --sibling <path> # explicit checkout
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const LIB = join(HERE, "..", "src/lib");
const SOURCES_PATH = join(LIB, "parity-sources.json");

/**
 * THE REPOSITORY ROOT, and the sibling checkout beside it.
 *
 * `HERE` is `<repo>/local_operator/mobile/web/scripts`, so FOUR levels up is the
 * repository root and the sibling is its SIBLING DIRECTORY. Round 2 measured the old
 * six-level climb resolving to the HOME directory (or, inside a `.worktrees/<x>`
 * checkout, to `local-operator-ui` under the worktrees directory), so the default
 * silently found nothing and printed the advisory skip on the very layout it was
 * written for (C1-M1).
 */
const REPO_ROOT = resolve(HERE, "..", "..", "..", "..");
const DEFAULT_SIBLING = resolve(REPO_ROOT, "..", "local-operator-ui");

const args = process.argv.slice(2);
const latest = args.includes("--latest");
const requireSibling = args.includes("--require-sibling");
const siblingFlag = process.argv.indexOf("--sibling");
const sibling =
	siblingFlag === -1
		? (process.env.LOCAL_OPERATOR_UI_REPO ?? DEFAULT_SIBLING)
		: resolve(process.argv[siblingFlag + 1] ?? DEFAULT_SIBLING);

const digestOf = (path) => createHash("sha256").update(readFileSync(path)).digest("hex");
const sources = JSON.parse(readFileSync(SOURCES_PATH, "utf8"));

/** The fixture entries the manifest pins, in a stable order. */
const entries = Object.entries(sources).filter(([key]) => !key.startsWith("_"));

let failed = false;
const fail = (message) => {
	console.error(`check-condense-parity: FAILED - ${message}`);
	process.exitCode = 1;
	failed = true;
};

if (!latest) {
	for (const [file, entry] of entries) {
		const path = join(LIB, file);
		if (!existsSync(path)) {
			fail(`${file} is pinned in parity-sources.json but not present`);
			continue;
		}
		const actual = digestOf(path);
		if (actual !== entry.sha256)
			fail(`${file} is ${actual.slice(0, 12)}… but parity-sources.json pins ${entry.sha256.slice(0, 12)}… — re-copy the fixture and re-pin it together`);
	}
	if (!failed)
		console.log(
			`check-condense-parity: ${entries.length} fixture(s) match their pins in src/lib/parity-sources.json`,
		);
	process.exit(process.exitCode ?? 0);
}

/* ------------------------------ --latest ------------------------------ */

const desktopManifest = join(sibling, "scripts/fixtures/condense.parity.manifest.json");
const desktopFixture = join(sibling, "scripts/fixtures/condense.parity.json");

if (!existsSync(desktopManifest) || !existsSync(desktopFixture)) {
	/*
	 * ONE MESSAGE PER MODE, because they make opposite claims. The required mode is
	 * failing on purpose, so telling the reader to "pass --require-sibling" (a flag
	 * they already passed) sent whoever hit it looking in the wrong place - measured
	 * while staging the CI ref swap: the required run printed the advisory sentence
	 * with a FAILED prefix.
	 */
	const where = `the desktop checkout (${sibling}) has no generated condense fixture manifest`;
	if (requireSibling)
		fail(`${where}, so drift cannot be compared - and this lane requires the comparison (--require-sibling)`);
	else
		console.log(
			`check-condense-parity: SKIPPED (advisory) - ${where}. Set LOCAL_OPERATOR_UI_REPO to a checkout of ` +
				"local-operator-ui, or pass --sibling <path>; pass --require-sibling to make its absence a failure.",
		);
	process.exit(process.exitCode ?? 0);
}

const theirs = JSON.parse(readFileSync(desktopManifest, "utf8"));
const theirFixture = digestOf(desktopFixture);
const ours = sources["condense.parity.json"];

if (theirFixture !== theirs.sha256)
	fail(
		`the desktop's manifest does not match its own fixture (${theirFixture.slice(0, 12)}… vs ${theirs.sha256.slice(0, 12)}…) — regenerate there first`,
	);
if (ours.sha256 !== theirFixture)
	fail(
		`this repository's condense.parity.json is ${ours.sha256.slice(0, 12)}… but the desktop's is ${theirFixture.slice(0, 12)}… — re-copy it and re-pin parity-sources.json (the re-port trigger)`,
	);
if (ours.sourceHash !== theirs.sourceHash)
	fail(
		`the pinned desktop model sourceHash is ${ours.sourceHash.slice(0, 12)}… but the desktop's manifest records ${String(theirs.sourceHash).slice(0, 12)}… — the model moved; port the diff and re-pin`,
	);

if (!failed)
	console.log(
		`check-condense-parity: in step with the desktop checkout at ${sibling} (sha256 ${theirFixture.slice(0, 12)}…, sourceHash ${String(theirs.sourceHash).slice(0, 12)}…)`,
	);
process.exit(process.exitCode ?? 0);

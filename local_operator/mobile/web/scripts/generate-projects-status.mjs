#!/usr/bin/env node
/**
 * Generates `src/projects-status.generated.ts` from the daemon's own board
 * order AND the store's own link cap.
 *
 *     node scripts/generate-projects-status.mjs           # write
 *     node scripts/generate-projects-status.mjs --check   # verify, exit 1 if stale
 *
 * ## Why this exists
 *
 * Two numbers the phone must not hand-copy. The Projects sheet groups its board
 * by project status, and the section order must be the daemon's rank order
 * (`STATUS_RANK` in `local_operator/server/models/desktop_projects.py`) or a row
 * lands under the wrong heading. And the link picker has to know
 * `SESSIONS_MAX` (`local_operator/projects.py`) to say the cap is reached
 * BEFORE the tap: that cap is shared by the work and filed lists, so it cannot
 * be derived from either one alone.
 *
 * The sheet used to carry a HAND-COPIED array of the seven statuses, and it had
 * already drifted once: the store grew four statuses into seven between 0.63.13
 * and 0.67.4, and every one the copy did not know fell into an "unknown"
 * trailing section instead of its lifecycle position — a silent mis-group,
 * invisible until a reader noticed a card in the wrong place.
 *
 * A comment pointing at the Python file does not survive that; a generated file
 * does. Those modules are the single source, this script reads them, and
 * `src/projects-status.test.tsx` fails when they diverge — the guard that makes
 * the next server-side status (or a changed cap) a red test rather than a
 * mis-grouped board or a control that lies about being available.
 *
 * ## Why not generate at build time
 *
 * `prebuild` deliberately does NOT run this. A generator on the build path
 * would rewrite the committed file and the divergence test could never fail
 * under CI (`pnpm build` runs before `pnpm test`), which is exactly the silent
 * repair this file exists to prevent. The generated module is committed; the
 * test proves it is current; `--check` gives a local reader the same answer.
 *
 * ## The workflow has to run for the files this reads
 *
 * `mobile-web.yml` is paths-gated, so its `paths:` list names BOTH Python
 * modules below — a status or a cap changed on the server alone must bring the
 * guard with it (review round 6, M2); a guard that cannot fire on the change it
 * guards is decoration.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
/** This package's root (`local_operator/mobile/web`). */
const ROOT = join(HERE, "..");
/** The repository root, four levels above the package. */
const REPO = join(HERE, "../../../..");
/** The Python module that owns the board order — the ONE source. */
/** The Python module that owns the board order — the ONE source. */
const PY_SOURCE = join(REPO, "local_operator/server/models/desktop_projects.py");
/** The Python module that owns the store's link cap. */
const PY_STORE = join(REPO, "local_operator/projects.py");
const OUT = join(ROOT, "src/projects-status.generated.ts");

export { PY_SOURCE, PY_STORE, OUT };

/**
 * The ordered status list, read out of the Python `STATUS_RANK` dict.
 *
 * Sorted by the RANK VALUE, not by the order the keys happen to appear in the
 * literal — the daemon's board sorts on the number, so a rank edited without
 * moving its line must still move the section.
 *
 * Deliberately strict: the dict literal is located by its assignment, its body
 * is brace-balanced, and each entry must be a quoted lowercase word mapped to
 * an integer. A parse that cannot find a well-formed dict THROWS rather than
 * returning an empty list — a generator that silently emits nothing is a
 * gate that passes a drifted tree, which is the failure mode this file is
 * built to avoid.
 */
export function readStatusOrder(pySource) {
	const start = pySource.indexOf("\nSTATUS_RANK = {");
	if (start === -1) throw new Error("STATUS_RANK assignment not found in " + PY_SOURCE);
	const open = pySource.indexOf("{", start);
	const close = pySource.indexOf("}", open);
	if (close === -1) throw new Error("STATUS_RANK dict is not closed");
	const body = pySource.slice(open + 1, close);
	const entries = [...body.matchAll(/"([a-z][a-z0-9_]*)"\s*:\s*(-?\d+)\s*,/g)];
	if (entries.length === 0) throw new Error("STATUS_RANK carries no parsable entries");
	const order = entries
		.map((match, index) => ({ status: match[1], rank: Number(match[2]), index }))
		.sort((left, right) => left.rank - right.rank || left.index - right.index)
		.map((entry) => entry.status);
	if (new Set(order).size !== order.length) throw new Error("STATUS_RANK carries a duplicate status");
	return order;
}

/** How many sessions one project may carry, read out of the Python store.
 *
 * `SESSIONS_MAX` is shared by the work and the coordination lists (the store's
 * own rule), which is why the sheet cannot derive the cap from the work list
 * alone. Strict for the same reason as the status parse: a value that cannot be
 * read is a THROW, never a default, because a silently wrong cap is a control
 * that lies about being available. */
export function readSessionLinkCap(pySource) {
	const match = /^SESSIONS_MAX = (\d+)$/m.exec(pySource);
	if (!match) throw new Error("SESSIONS_MAX not found in " + PY_STORE);
	return Number(match[1]);
}

/** The generated module's text for one ordered status list and one cap. */
export function renderModule(order, sessionLinkCap) {
	const list = order.map((status) => `\t"${status}",`).join("\n");
	return `/**
 * GENERATED — do not edit.
 *
 * \`STATUS_ORDER\` is the daemon's board order for project statuses, read out of
 * \`STATUS_RANK\` in \`local_operator/server/models/desktop_projects.py\` by
 * \`scripts/generate-projects-status.mjs\` and sorted by each status's rank.
 * \`SESSION_LINK_CAP\` is \`SESSIONS_MAX\` from \`local_operator/projects.py\`,
 * the cap the work and coordination lists share. Regenerate with
 * \`pnpm gen-projects-status\`; \`src/projects-status.test.ts\` fails when this
 * file is stale, which is what keeps a status added on the server from silently
 * mis-grouping the phone's board.
 *
 * \`readonly string[]\` rather than a literal union on purpose: a status this
 * build has never heard of still has to be able to flow through the sheet's
 * grouping (it renders in a visibly-unrecognised trailing section), and a
 * literal union would turn that runtime case into a type error at the boundary
 * where it is handled.
 */
export const STATUS_ORDER: readonly string[] = [
${list}
];

/** How many sessions one project may carry, work and filed links TOGETHER
 * (\`SESSIONS_MAX\` in \`local_operator/projects.py\`). The sheet uses it to say
 * the cap is reached BEFORE the tap, instead of letting the reader learn it
 * from a refusal. */
export const SESSION_LINK_CAP = ${sessionLinkCap};
`;
}

function main(argv) {
	const check = argv.includes("--check");
	let rendered;
	try {
		rendered = renderModule(
			readStatusOrder(readFileSync(PY_SOURCE, "utf8")),
			readSessionLinkCap(readFileSync(PY_STORE, "utf8")),
		);
	} catch (error) {
		console.error(String(error.message ?? error));
		process.exit(1);
	}
	const current = (() => {
		try {
			return readFileSync(OUT, "utf8");
		} catch {
			return null;
		}
	})();
	if (check) {
		if (current === rendered) {
			console.log(`${OUT} is up to date with ${PY_SOURCE} and ${PY_STORE}`);
			return;
		}
		console.error(
			`${OUT} is STALE against ${PY_SOURCE} / ${PY_STORE} — run \`pnpm gen-projects-status\``,
		);
		process.exit(1);
	}
	if (current === rendered) {
		console.log(`${OUT} unchanged`);
		return;
	}
	writeFileSync(OUT, rendered);
	console.log(`wrote ${OUT}`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
	main(process.argv.slice(2));
}

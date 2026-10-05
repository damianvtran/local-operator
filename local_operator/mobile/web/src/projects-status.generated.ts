/**
 * GENERATED — do not edit.
 *
 * `STATUS_ORDER` is the daemon's board order for project statuses, read out of
 * `STATUS_RANK` in `local_operator/server/models/desktop_projects.py` by
 * `scripts/generate-projects-status.mjs` and sorted by each status's rank.
 * Regenerate with
 * `pnpm gen-projects-status`; `src/projects-status.test.ts` fails when this
 * file is stale, which is what keeps a status added on the server from silently
 * mis-grouping the phone's board.
 *
 * `readonly string[]` rather than a literal union on purpose: a status this
 * build has never heard of still has to be able to flow through the sheet's
 * grouping (it renders in a visibly-unrecognised trailing section), and a
 * literal union would turn that runtime case into a type error at the boundary
 * where it is handled.
 */
export const STATUS_ORDER: readonly string[] = [
	"planning",
	"active",
	"qa",
	"validation",
	"paused",
	"done",
	"archived",
];

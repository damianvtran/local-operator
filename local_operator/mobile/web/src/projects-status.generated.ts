/**
 * GENERATED — do not edit.
 *
 * `STATUS_ORDER` is the daemon's board order for project statuses, read out of
 * `STATUS_RANK` in `local_operator/server/models/desktop_projects.py` by
 * `scripts/generate-projects-status.mjs` and sorted by each status's rank.
 * `SESSION_LINK_CAP` is `SESSIONS_MAX` from `local_operator/projects.py`,
 * the cap the work and coordination lists share. Regenerate with
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

/** How many sessions one project may carry, work and filed links TOGETHER
 * (`SESSIONS_MAX` in `local_operator/projects.py`). The sheet uses it to say
 * the cap is reached BEFORE the tap, instead of letting the reader learn it
 * from a refusal. */
export const SESSION_LINK_CAP = 64;

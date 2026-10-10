/**
 * Image generation on the phone — the detection set and the ONE
 * entry→view-model adapter for the progress card.
 *
 * The programme wires agent-driven image generation end to end
 * (provider → harness tool `generate_image` → every surface). This module is
 * the phone's whole share of the contract, and it exists as a module rather
 * than inline in the card because the LIVE DETAIL FIELDS ARE FROZEN HERE:
 * every `generate_image` update carries the canonical bag — `stage`,
 * `queue_position`, `progress_fraction`, `log_lines`, `error`, `error_type` —
 * every key present, `None` when no provider supplied a value (the harness
 * lane's freeze, 2026-10-09). Everything that reads one of those names lives
 * here and nowhere else, so a future rename is a one-file change — and the
 * card, its tests and the capture fixtures keep rendering.
 *
 * THE WIRE VOCABULARY IS NOT THIS VOCABULARY. The canonical `stage` words are
 * `queued` / `in_progress` / `completed` / `cancelled` / `cancelling`, and
 * `None` on a mid-walk failure whose semantics ride `error`/`error_type`.
 * FAL's own IN_QUEUE / IN_PROGRESS / COMPLETED / CANCELLED, its
 * `queue_position: int`, its `logs` rows and its error prose are folded into
 * that bag ABOVE this module — a surface never sees vendor names. This module
 * maps the canonical shape (plus the `TranscriptEntry` fields every tool call
 * already uses — `tool_state`, `error`) onto the card's view states:
 * queued → running → done | failed | cancelled, plus `cancelling`.
 *
 * ABSENCE RENDERS THE REDUCED STATE HONESTLY. Every live field below is
 * presence-gated: a field the feed does not carry yet (or carries malformed)
 * maps to null/[] and the card renders the reduced state rather than a
 * fabricated number. Numbers that ARE carried are never rounded, clamped or
 * extrapolated into a claim the feed did not make.
 */
import type { TranscriptEntry, TranscriptImageRef } from "../types";

/**
 * The detection set — the ONE exported detection point per surface.
 *
 * ONE tool since the harness lane froze it: `generate_image`, whose
 * image-to-image case is a `source_image_path` argument on the SAME call —
 * there is no separate `generate_altered_image`. If the harness ever adds a
 * second image-generation call, its name is added HERE and nowhere else:
 * the transcript branch and this module's tests both read this constant.
 * Keys are lowercase; lookups normalise, mirroring `tool-row.tsx`'s own
 * `DIFF_FIRST_TOOLS` lookup beside it.
 */
export const IMAGE_GEN_TOOLS: ReadonlySet<string> = new Set([
	"generate_image",
]);

/**
 * The card's view states. NOT the wire vocabulary: `cancelling` never
 * appears on the wire — it is the honest hold between the user's cancel and
 * its confirmation (the provider note: never optimistically "cancelled"),
 * and `cancelled` is this surface's word for the wire's `interrupted`.
 *
 * `finished` is the cancel-conflict settle: the frozen provider contract says
 * a cancel landing on an already-completed job answers
 * `error_type: media_already_completed`, and the surface must render THAT as
 * "already finished" — never as an error, because nothing failed.
 */
export type ImageGenCardState =
	| "queued"
	| "running"
	| "cancelling"
	| "done"
	| "failed"
	| "cancelled"
	| "finished";

/** Everything the card's body renders, and nothing else. */
export interface ImageGenView {
	state: ImageGenCardState;
	/**
	 * Whether the call had actually started generating when the current
	 * presentation began — the desktop's F3 fact, mirrored so a call that
	 * never generated never grows a generating tile or bar. `running` implies
	 * true; a `cancelling` hold keeps it only when the state the hold
	 * replaced was running (a press on the queued card holds without
	 * conjuring a body the card never showed). False everywhere else — no
	 * settled state renders a body.
	 */
	generating: boolean;
	/** The canonical `queue_position`, when the feed carries an integer. */
	queuePosition: number | null;
	/** The canonical `progress_fraction` (0..1), when the feed carries one. */
	progress: number | null;
	/** The canonical `log_lines` rows' messages, tail only; empty when absent. */
	logs: string[];
	/**
	 * The failure message, verbatim — and it IS the provider's actual error in
	 * its sanctioned form: the frozen contract says the surfaces never receive
	 * vendor free-text, only a stable platform sentence safe to render as-is.
	 * Empty when the settle carried none (the reduced state; never substituted
	 * with a sentence of this surface's own).
	 */
	error: string;
	/**
	 * The structured failure category (FAL's own code when it exists, else a
	 * platform code: `media_rejected | media_failed | media_rate_limited |
	 * media_unavailable`, and `media_already_completed` for the cancel
	 * conflict). Empty when absent.
	 */
	errorType: string;
	/** Finished artifact refs, in the entry's image index space. */
	images: TranscriptImageRef[];
}

/**
 * The provider's structured code for "the job this cancel targeted had
 * already completed" — the one failure category that is NOT an error on this
 * surface (the card says "already finished"). One spelling, frozen by the
 * provider contract.
 */
const ERROR_TYPE_ALREADY_COMPLETED = "media_already_completed";

/**
 * The canonical `stage` words this surface acts on. The freeze defines two
 * more (`completed`, `cancelled`); the mapping below explains why those are
 * deliberately NOT refinements of a live row. One spelling each, read from
 * the one module in `src/` that reads field names.
 */
const STAGE_QUEUED = "queued";
const STAGE_IN_PROGRESS = "in_progress";
const STAGE_CANCELLING = "cancelling";
const STAGE_CANCELLED = "cancelled";

/**
 * How many log lines the card shows. A phone card is a glance, not a log
 * viewer: the TAIL is what a watcher reads ("what is it doing now"), and
 * three one-line rows fit under the progress bar without pushing the
 * controls off screen. The adapter slices, so the card never re-decides.
 */
const LOG_TAIL_LINES = 3;

/**
 * The paint text of one canonical `log_lines` row, or null.
 *
 * The canonical bag passes the provider's own rows through verbatim —
 * `{message, timestamp}` — and the card paints the message. A row of any
 * other shape (a bare string, a number, an object without a string message)
 * is a shape this adapter does not recognise: it drops the row rather than
 * guessing a rendering the feed did not send.
 */
function logLineText(row: unknown): string | null {
	if (typeof row === "object" && row !== null) {
		const message = (row as { message?: unknown }).message;
		if (typeof message === "string") return message;
	}
	return null;
}

/**
 * Map one image-gen tool entry onto the card's view model.
 *
 * `cancelRequested` is the card's own local fact — the user pressed Cancel
 * and no confirmation has landed yet — held by the component, not the wire.
 * It outranks every not-yet-settled state (composing/queued/running) but
 * NEVER a settle: the moment the entry reports done/failed/interrupted, that
 * verdict is what renders, because the confirmation is the settle itself and
 * a press is not a result.
 *
 * Callers gate on `IMAGE_GEN_TOOLS` (same module) before mounting the card;
 * this function maps whatever it is given and does not re-check the name.
 */
export function imageGenView(
	entry: TranscriptEntry,
	cancelRequested = false,
): ImageGenView {
	const details = (entry.details ?? {}) as Record<string, unknown>;

	/* The structured failure category, when the settle carried one — read
	   BEFORE the state mapping because it refines two of its arms below. */
	const errorType =
		typeof details.error_type === "string" ? details.error_type : "";

	/* The canonical live word. Only the THREE words that name a live interim
	   refine the live view below — `queued` (a provider-side queue the card's
	   own `tool_state` is already `running` through), `in_progress`, and the
	   wire's own `cancelling` hold. `completed` and `cancelled` are
	   settled-end words: while the row is live they are IGNORED, because the
	   settle that follows IS the confirmation — a beat of optimism before it
	   is exactly what the cancel contract forbids. `None` rides mid-walk
	   failures, whose semantics live in `error`/`error_type` while the walk
	   continues; a live row keeps its own state and the settle carries the
	   outcome. An unknown word reads as absent — a stranger's vocabulary may
	   not repaint the card. */
	const rawStage = details.stage;
	const stage = typeof rawStage === "string" ? rawStage : "";

	/* `interrupted` is the fold's own word for a call a stop or a steer cut
	   off; the card states it as `cancelled`, the frozen vocabulary's word.
	   `composing` (the model still dictating the call) maps to `queued`: the
	   view has no pre-run distinction, and inventing one would put a state on
	   screen the frozen vocabulary cannot read back. An unknown wire state (a
	   newer runtime's) maps to `queued` too — the pre-run resting state claims
	   nothing about execution or outcome, which is the least a stranger's
	   state may assert.

	   A settle carrying `media_already_completed` is the CANCEL CONFLICT — the
	   stop raced a job that had already finished — and it maps to `finished`
	   from EITHER failure-shaped settle (a conflicted cancel reaches the phone
	   as whichever verdict its emitter chose), never to `failed`/`cancelled`,
	   because nothing failed and nothing was stopped. A `done` settle stays
	   `done`: the artifact could exist, and "already finished" is what a
	   failed-looking settle means, not what a success is re-worded into.

	   And a failure-shaped settle whose canonical `stage` says `cancelled`
	   (with no conflict code) is a cancel that LANDED, not a failure: the
	   cancel path builds its result error-shaped while its details name the
	   cancel, so without this read a deliberate stop would paint in the
	   failure ink. Only an otherwise-unclassified failure is refined — any
	   other `error_type` keeps its own arm. */
	let state: ImageGenCardState;
	switch (entry.tool_state) {
		case "running":
			state = "running";
			break;
		case "done":
			state = "done";
			break;
		case "failed":
			state =
				errorType === ERROR_TYPE_ALREADY_COMPLETED
					? "finished"
					: errorType === "" && stage === STAGE_CANCELLED
						? "cancelled"
						: "failed";
			break;
		case "interrupted":
			state =
				errorType === ERROR_TYPE_ALREADY_COMPLETED ? "finished" : "cancelled";
			break;
		default:
			state = "queued";
			break;
	}

	/* THE GENERATING FACT (the desktop's F3 rule, mirrored — `view.generating`
	   there): whether the call had started generating when the current live
	   presentation began, so a call that never generated never grows a
	   generating tile or bar. It is tracked as "where the call stood before
	   any cancelling word", because every cancelling transition erases that
	   answer from `state`: `running` has generated by definition; `queued`
	   (the reduced state — announced, or a provider-side queue this surface
	   renders reduced) has not; the wire's own `cancelling` keeps whatever
	   the row's word last said (the frame that carries it no longer carries
	   the stage it replaced); and a press keeps the state it replaced, so a
	   hold can never conjure a body the press's own card did not show. */
	let generating = state === "running";

	/* The live stage word refines only the live view (see above). */
	if (state === "queued" || state === "running") {
		if (stage === STAGE_QUEUED) {
			state = "queued";
			generating = false;
		} else if (stage === STAGE_IN_PROGRESS) {
			state = "running";
			generating = true;
		} else if (stage === STAGE_CANCELLING) {
			/* The wire's own hold: `generating` keeps the row's word. */
			state = "cancelling";
		}
	}

	/* A pending cancel outranks every not-yet-settled state, and never a
	   settle: the press is a request, the settle is the result. `generating`
	   is deliberately untouched — the hold keeps the body of the state the
	   press replaced (see above). */
	if (
		cancelRequested &&
		(state === "queued" || state === "running" || state === "cancelling")
	) {
		state = "cancelling";
	}

	/* The canonical `queue_position` when the feed carries an integer (the
	   harness folds FAL's own field under this name). Integer only — a float,
	   a numeric string or a negative is a shape this adapter does not
	   recognise, and rendering one would dress a maybe for a fact. */
	const rawQueuePosition = details.queue_position;
	const queuePosition =
		typeof rawQueuePosition === "number" &&
		Number.isInteger(rawQueuePosition) &&
		rawQueuePosition >= 0
			? rawQueuePosition
			: null;

	/* The canonical `progress_fraction` (0..1). It stays absent until a
	   provider reports one — the harness lane deliberately never synthesizes
	   one — and out-of-range values are NOT clamped: clamping 42 into a full
	   bar would state 100% off a value that never meant a fraction, so
	   anything outside 0..1 renders the reduced state instead — no bar at
	   all (the fraction-less bar was removed with the desktop's D2 mirror;
	   the card reduces to the tile alone), honestly. */
	const rawProgress = details.progress_fraction;
	const progress =
		typeof rawProgress === "number" &&
		Number.isFinite(rawProgress) &&
		rawProgress >= 0 &&
		rawProgress <= 1
			? rawProgress
			: null;

	const rawLogLines = details.log_lines;
	const logs = Array.isArray(rawLogLines)
		? rawLogLines
				.map(logLineText)
				.filter((line): line is string => line !== null)
				.slice(-LOG_TAIL_LINES)
		: [];

	/* The failure message as the settle carried it, verbatim. Per the frozen
	   provider contract the surfaces never receive vendor free-text: `error`
	   is a stable platform sentence already safe to render, so this surface
	   neither rewrites it nor substitutes a sentence of its own when it is
	   absent (absence renders the reduced state). */
	const error = typeof entry.error === "string" ? entry.error : "";

	/* Artifact refs. The frozen attachment contract indexes them like a user
	   turn's images and serves their bytes from the same phone image route,
	   so this reads the SAME `images` field the transcript's user case
	   renders — when the emitter lane lands refs on tool rows they arrive
	   here. */
	const rawImages = entry.images;
	const images = Array.isArray(rawImages)
		? rawImages.filter(
				(ref): ref is TranscriptImageRef =>
					ref != null &&
					typeof ref.index === "number" &&
					typeof ref.mime_type === "string",
			)
		: [];

	return {
		state,
		generating,
		queuePosition,
		progress,
		logs,
		error,
		errorType,
		images,
	};
}

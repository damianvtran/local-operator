/**
 * Image-generation progress card — the phone's live view of a
 * `generate_image` call (the harness lane's ONE image tool; the image-to-image
 * case rides a `source_image_path` argument on the same call).
 *
 * The row itself is still the ordinary `ToolRow` (state glyph, name, summary,
 * elapsed — details behind its disclosure), and the card's figure mounts
 * inside that SAME container through ToolRow's `children` slot, so the card
 * is one object: row plus figure share the state ground, and the row's
 * existing tap-to-expand still answers "what was the prompt". Everything
 * below the row comes from `imageGenView` (`lib/image-gen.ts`) — the ONE
 * place the live-detail wire fields are read — and this component never
 * touches those fields itself.
 *
 * THE STATE MACHINE, as the frozen programme contract fixes it:
 *
 *   queued → running → done | failed | cancelled,
 *   plus `cancelling` (the hold) and `finished` (the cancel conflict).
 *
 * `cancelling` is the honest hold between the user's press on Cancel and the
 * confirmation landing on the wire — the card NEVER paints `cancelled`
 * optimistically, because a cancel is a request: the provider may still
 * complete. Confirmation is the row settling (interrupted → cancelled, or
 * whatever verdict actually arrived); from the press until then the card
 * says "cancelling…".
 *
 * `finished` is the same contract's later-frozen edge: a cancel landing on an
 * ALREADY-COMPLETED job answers `error_type: media_already_completed`, and
 * the card states that as "already finished" — never as an error, because
 * nothing failed.
 *
 * THE FAILURE TEXT IS NOT THIS CARD'S TO WORD. The frozen provider contract
 * says the surfaces never receive vendor free-text: `error` is a stable
 * platform sentence already safe to render as-is, so the failed state renders
 * it verbatim and adds no generic sentence of its own.
 *
 * THE CANCEL CONTROL RIDES THE EXISTING TURN-INTERRUPT PATH. `{op:"abort"}`
 * through `sendCommand` is the SAME mechanism the composer's stop button
 * (and an in-flight dictation's cancel) already uses — there is no second
 * cancel mechanism for image generation, deliberately: v1's restart/steer
 * story is "interrupt the turn, then a new generate call", so making this
 * control a turn interrupt keeps every cancel on one rail. The control is
 * offered while the call is UNSETTLED — queued as well as running (wave-2
 * conformance: the TUI's hint, the desktop card and the native app all stop
 * a queued call, and the relay's queued card showed no control at all, a
 * dead end at card level) — and both presses engage the same latch, shed
 * the control and paint the same hold. The restart and steer affordances
 * are SLOTS (optional props) that the transcript does not wire yet — the
 * named surface op is not defined — and are demonstrated in tests only.
 *
 * THE GENERATING BODY IS GATED ON A GENERATION HAVING STARTED (the
 * desktop's F3 rule, mirrored — `view.generating` in the adapter): the
 * tile, the bar and the log tail render while `running`, and during a
 * `cancelling` hold only when the state the hold replaced was running. A
 * hold that replaced the queued card holds the word ALONE — in the control
 * slot, the running hold's own place (UX round 1, U4) — and never grows the
 * body it never had.
 *
 * ONE MOTION PER SURFACE (the desktop's D2 ruling, mirrored). The tile's
 * sweep is this card's ONE indefinite element; the progress bar draws ONLY
 * against a fraction the feed carried. The indeterminate full-width bar
 * that used to sweep beside the tile doubled the rhythm inches away from it
 * and is gone — a fraction-less row reserves the bar's space so the control
 * does not move when a number lands.
 *
 * WHAT THIS CARD DOES NOT DO: it does not call the provider, invent a
 * progress number, or add transport. Live fields render only as the feed
 * carries them (see the adapter); absence renders the reduced state.
 */
import { useEffect, useState } from "react";
import { sendCommand } from "../api";
import { imageGenView } from "../lib/image-gen";
import { AttachmentImage } from "./attachment-image";
import { ToolRow } from "./tool-row";
import type { TranscriptEntry } from "../types";

/** The quiet state line every state's body carries (design vocabulary:
    `text-meta` on the dim ink, the app's receipt voice), except states whose
    body already states more. */
const STATE_LINE = "text-meta text-ink-dim";

/** The card's cancel control — ONE recipe for BOTH live states (queued and
    running): the same words, the same danger ink and the same 44px touch
    floor the running case shipped. A second copy would be a drift hazard,
    and the two states must not read as two different controls (wave-2
    conformance). */
const CANCEL_BUTTON =
	"flex min-h-11 shrink-0 items-center rounded-sm border border-danger-border px-3 text-body-sm text-danger active:bg-danger-wash";

/** The restart/steer slot buttons. The ask sheet's own secondary-button
    recipe (`ask-card.tsx`), because a settled card's controls are a small
    decision, not an alarm — the danger ink belongs to Cancel alone. */
const SLOT_BUTTON =
	"flex min-h-11 items-center rounded-sm border border-control px-3 text-body-sm text-ink-muted active:bg-elevated";

export function ImageGenCard({
	entry,
	pid,
	onRestart,
	onSteer,
}: {
	entry: TranscriptEntry;
	/** The session's route id (the daemon calls it pid; every caller on this
	    surface passes the session id, the composer included). */
	pid: string;
	/** UNWIRED. The v1 restart flow is "interrupt + a NEW generate call" and
	    the named surface op does not exist yet, so the transcript passes
	    nothing and this slot renders no control. Fixtures and tests pass a
	    handler to demonstrate the affordance. */
	onRestart?: () => void;
	/** UNWIRED, same contract as `onRestart` (see above). */
	onSteer?: () => void;
}) {
	/* Component-local, because this fact does not exist on the wire: a cancel
	   hold is ENGAGED — pressed on this client, or carried by the feed's own
	   `cancelling` interim — and no confirmation has landed. It never turns
	   into a rendered outcome by itself (the adapter gives a settle priority
	   over it), so its lifecycle is exactly one live view: engaged by the
	   first cancelling mapping, retired when the view settles. */
	const [cancelRequested, setCancelRequested] = useState(false);
	const view = imageGenView(entry, cancelRequested);

	/* A SETTLE RETIRES THE HOLD, AND THE WIRE'S OWN HOLD LATCHES IT (review
	   round 1, F2; design round 1, D1). Rendered behaviour was already
	   honest — a settle always outranks the flag, and never paints
	   "cancelled" off the press — but two facts needed fixing: the flag
	   outlived the settle (a wire that ever re-reported the same entry as
	   live would resurrect "cancelling…" with no new press), and the FEED's
	   own interim un-said itself in the terminal-update window — the
	   producer emits `cancelling` then `cancelled` ONE UPDATE before the
	   result, and the second fell back to `running`, re-offering the abort
	   control mid-cancel (a second tap sends a second `{op:abort}`). The
	   latch engages the moment a live view maps to `cancelling` — the press
	   or the wire — and the adapter's own pin then carries the held word
	   over later live updates. It clears only when the view leaves every
	   not-yet-settled state, so a settle always wins (`done` included), and
	   a cold `cancelled` that never had an interim latches nothing. */
	const live =
		view.state === "queued" ||
		view.state === "running" ||
		view.state === "cancelling";
	useEffect(() => {
		if (view.state === "cancelling") setCancelRequested(true);
		else if (!live) setCancelRequested(false);
	}, [view.state, live]);

	/* THE ROW IS PART OF THE CARD'S STATEMENT, so it must not disagree with
	   it. The row's own palette is keyed off the wire settle, and for the
	   cancel conflict that settle is failure-shaped (✗ on the danger wash)
	   while the honest state is neither a failure nor a success — the job
	   had already finished and this call returned no artifact. For exactly
	   that view state the row renders as the NEUTRAL settle (dim "–", no
	   wash — the glyph a stop-cut call wears), and the body says why:
	   "already finished". A LANDED cancel (review round 1, F2) reaches the
	   phone the same way — error-shaped, its details naming the cancel — so
	   it takes the same swap: the body's quiet "cancelled" must not sit
	   beside a red ✗ and a danger wash. Every other state passes the entry
	   through untouched, so ToolRow stays the one renderer of every other
	   row. */
	const rowEntry =
		view.state === "finished" || view.state === "cancelled"
			? { ...entry, tool_state: "interrupted" as const }
			: entry;

	const cancel = async () => {
		/* Flip the hold BEFORE the request: the button must not stay live
		   for the round-trip, and the honest immediate state is "cancelling",
		   not "running" — the press HAS happened. */
		setCancelRequested(true);
		try {
			await sendCommand(pid, { op: "abort" });
		} catch {
			/* The abort never reached a runtime (a refusal or a transport
			   failure): no confirmation will ever land, so the hold is
			   released and the card returns to the live state with the
			   control available again. No error copy is invented here —
			   this control's failures are the turn's, and the composer's
			   own stop already owns that message when the user stops a
			   turn deliberately. */
			setCancelRequested(false);
		}
	};

	/* THE GENERATING BODY'S GATE (the desktop's F3 rule, mirrored): the tile,
	   the bar and the log tail render while `running` — a running call had a
	   generation by definition — and during a `cancelling` hold only when a
	   generation had started (`view.generating`). A hold that replaced the
	   queued card draws the reduced line below: neither a tile nor a bar it
	   never had. */
	const generatingBody =
		view.state === "running" ||
		(view.state === "cancelling" && view.generating);

	return (
		<div data-testid="image-gen-card" className="min-w-0">
			<ToolRow entry={rowEntry}>
				{generatingBody ? (
					<div className="flex flex-col items-start gap-2 pt-0.5 pb-1.5 pl-6">
						{/* The tile: the square the finished picture will land
						    in (the transcript's own 160px attachment frame), so
						    the done state does not reflow the card. The sweep
						    rides `lo-gen-tile` (styles/index.css) — the same
						    motion the streaming text uses, on the ground ramp:
						    the placeholder must not wear the accent, which
						    belongs to the one live control below. */}
						<span
							aria-hidden
							className="lo-gen-tile block h-40 w-40 shrink-0 rounded-sm border border-hairline"
						/>
						{/* THE ROW'S HEIGHT IS THE 44px TOUCH FLOOR IN EVERY STATE
						    (design D1). While the control is a button it is the row's
						    tallest child; when the hold replaces it with a text span, a
						    shorter row would re-centre the bar and lift the card's
						    bottom edge at the stop tap — a reflow the user reads as the
						    card flinching exactly when they asked it to stop. `min-h-11`
						    pins the slot so the running -> cancelling swap moves
						    nothing. */}
						<div className="flex min-h-11 w-full items-center gap-2 pr-1.5">
							{view.progress !== null ? (
								/* The determinate branch, rendered ONLY off a
								   fraction the feed carried (the adapter maps
								   anything else to null). */
								<span
									role="progressbar"
									aria-label="generating image"
									aria-valuemin={0}
									aria-valuemax={100}
									aria-valuenow={Math.round(view.progress * 100)}
									className="h-0.5 min-w-0 flex-1 overflow-hidden rounded-full bg-sunken"
								>
									<span
										className="block h-full rounded-full bg-accent"
										/* One rounding for BOTH the painted width and
										   aria-valuenow, so the bar a screen reader
										   states and the bar a sighted user sees can
										   never be two numbers. */
										style={{
											width: `${Math.round(view.progress * 100)}%`,
										}}
									/>
								</span>
							) : (
								/* NO FRACTION, NO BAR (the desktop's D2 ruling,
								   mirrored): the tile's sweep is this card's ONE
								   indefinite element, and the fraction-less bar
								   that used to sweep beside it doubled the
								   rhythm inches away from the tile. The empty slot
								   reserves the row so the control does not move
								   when a fraction lands. */
								<span aria-hidden className="min-w-0 flex-1" />
							)}
							{view.state === "running" ? (
								<button
									type="button"
									onClick={() => void cancel()}
									className={CANCEL_BUTTON}
								>
									cancel
								</button>
							) : (
								/* The hold, spelled out. Never "cancelled": the
								   press is a request and this state ends when the
								   wire settles one way or the other. The test id is
								   shipped markup, not a harness-only hook — the
								   capture rig's geometry dump reads it (review
								   round 1, F3: a dump that names a line it cannot
								   see is an instrument reporting nothing). */
								<span
									data-testid="image-gen-hold"
									className="shrink-0 text-meta text-ink-dim"
								>
									cancelling…
								</span>
							)}
						</div>
						{view.logs.length > 0 ? (
							<div className="flex w-full flex-col gap-0.5 pr-1.5">
								{view.logs.map((line, i) => (
									<p
										key={i}
										className="truncate font-mono text-mono-sm text-ink-dim"
									>
										{line}
									</p>
								))}
							</div>
						) : null}
					</div>
				) : view.state === "cancelling" ? (
					/* THE REDUCED HOLD (F3): the call never generated, so the
					   hold word stands alone — no tile, no bar, nothing the card
					   never showed — in the CONTROL SLOT, trailing edge, where the
					   cancel control was: the same slot the running hold keeps
					   (UX round 1, U4: the one word must not move left-to-right
					   between the two holds). The 44px row floor holds (design
					   D1: the press must not lift the card's bottom edge under the
					   thumb that just stopped it; the vertical padding rides the
					   CONTAINER, the running body's own composition, so the padded
					   row measures the same with and without the control the hold
					   replaced). */
					<div className="w-full pt-0.5 pb-1.5 pl-6">
						<div className="flex min-h-11 w-full items-center gap-2 pr-1.5">
							<span aria-hidden className="min-w-0 flex-1" />
							<span
								data-testid="image-gen-hold"
								className="shrink-0 text-meta text-ink-dim"
							>
								cancelling…
							</span>
						</div>
					</div>
				) : view.state === "done" ? (
					<div className="flex flex-col gap-1.5 pt-0.5 pb-1.5 pl-6">
						{/* The finished artifact renders through the transcript's
						    EXISTING attachment path (the same component a user
						    turn's images use), fetched lazily from the existing
						    image route — the frozen attachment contract's
						    requirement, and the reason that component now
						    lives in its own module. Absent refs (the emitter
						    lane has not landed, or the artifact never was)
						    render the state line alone rather than a broken
						    frame. */}
						{view.images.length > 0 ? (
							<div className="flex flex-wrap gap-1.5">
								{view.images.map((img) => (
									<AttachmentImage
										key={img.index}
										pid={pid}
										entryId={entry.id}
										index={img.index}
									/>
								))}
							</div>
						) : null}
						<p className={STATE_LINE}>image ready</p>
					</div>
				) : view.state === "failed" ? (
					<div className="flex flex-col gap-1 pt-0.5 pb-1.5 pl-6 pr-1.5">
						{view.error ? (
							/* THE MESSAGE IS THE PROVIDER'S OWN, verbatim: the
							   frozen contract says the surfaces never receive
							   vendor free-text — `error` is a stable platform
							   sentence already safe to render as-is — so the card
							   adds NO lead sentence of its own (a generic "image
							   generation failed" on top would substitute this
							   surface's wording for the sanctioned one). Empty
							   text renders the reduced state: the quiet word,
							   nothing invented. */
							<p className="text-body-sm break-words whitespace-pre-wrap text-danger">
								{view.error}
							</p>
						) : (
							<p className={STATE_LINE}>failed</p>
						)}
					</div>
				) : view.state === "finished" ? (
					/* The cancel conflict (`media_already_completed`): the stop
					   raced a job that had already finished. Stated plainly and
					   quietly — NEVER as an error, because nothing failed. */
					<p className={`${STATE_LINE} pt-0.5 pb-1.5 pl-6 pr-1.5`}>
						already finished
					</p>
				) : view.state === "cancelled" ? (
					<div className="flex flex-col gap-1.5 pt-0.5 pb-1.5 pl-6">
						<p className={STATE_LINE}>cancelled</p>
						{onRestart || onSteer ? (
							<div className="flex gap-2">
								{onRestart ? (
									<button type="button" onClick={onRestart} className={SLOT_BUTTON}>
										restart
									</button>
								) : null}
								{onSteer ? (
									<button type="button" onClick={onSteer} className={SLOT_BUTTON}>
										steer
									</button>
								) : null}
							</div>
						) : null}
					</div>
				) : (
					/* The queued card is a LIVE card (wave-2 conformance): it
					   offers the same cancel as the running body — TUI,
					   desktop and the native app all stop a queued call, and
					   the relay showed no control, a dead end at card level.
					   The press engages the same latch (the header's rail
					   note) and the hold replacing the control IS the
					   double-press guard. The copy states the D3-ruled
					   meaning of the carried number: it counts the requests
					   AHEAD of this one, never the off-by-one reading
					   "position N" invited. */
					<div className="w-full pt-0.5 pb-1.5 pl-6">
						<div className="flex min-h-11 w-full items-center gap-2 pr-1.5">
							<p className={`${STATE_LINE} min-w-0 flex-1`}>
								queued
								{/* The queue position, ONLY when the feed carries it —
								    the line stands alone until the relay sends one. */}
								{view.queuePosition !== null
									? ` · ${view.queuePosition} ahead`
									: ""}
							</p>
							<button
								type="button"
								onClick={() => void cancel()}
								className={CANCEL_BUTTON}
							>
								cancel
							</button>
						</div>
					</div>
				)}
			</ToolRow>
		</div>
	);
}

/**
 * The session header's status dot, its explanation sheet, and the resume path
 * an ended session offers from it.
 *
 * WHY A DOT AND NOT A BANNER. The ended session used to carry a permanent
 * full-bleed band above the composer ("Session ended · history kept" + hint +
 * Resume) and a degraded one carried a second warning band in the same slot.
 * Both spent the thumb zone on a fact that changes rarely and needs no action
 * until the reader wants one, and the ended band sat flush on the composer card.
 * The fact now lives where the subtitle already said it ("Ended", "Not
 * answering"): one quiet dot beside that label, and the detail plus the one
 * action (Resume) one tap away in the shared Sheet.
 *
 * THE STATE COMES FROM RUNTIME SIGNALS THE RELAY SENDS, never from the last
 * message: see {@link sessionHealth} for the exact fields and the rule.
 */
import { type ReactNode, useEffect, useRef, useState } from "react";
import { resumeSession } from "../api";
import { cn } from "../lib/cn";
import { resumeRefusalText } from "../lib/refusal";
import { navigate } from "../router";
import { messageRuntime } from "../store";
import { type HealthInput, sessionHealth, type SessionHealth } from "../lib/session-health";
import type { SessionProjection } from "../types";
import { SessionSubtitle } from "./session-subtitle";
import { Button } from "./ui/button";
import { Sheet } from "./ui/sheet";

/** The dot's fill per state — existing theme tones only. Each is held at the
    4.5:1 text floor against `canvas` (the header's ground) by
    `scripts/contrast-contract.mjs` in every shipped palette (`ink-dim` as an ink
    weight, `success`/`warning` as tones), so the non-text 3:1 floor a status
    mark needs is met with margin by construction, not by one palette's luck. */
export const HEALTH_DOT_TONE: Record<SessionHealth, string> = {
	live: "bg-success",
	not_answering: "bg-warning",
	reconnecting: "bg-warning",
	ended: "bg-ink-dim",
};

export { sessionHealth, type SessionHealth };

/** The state's own word: the control's `title`, the sheet's title, and what
    the polite live region announces on a transition. Colour is never the only
    signal. The dot itself is `aria-hidden`; the control is named by its VISIBLE
    words (the subtitle), plus this word as screen-reader text where the subtitle
    does not already say it (`live`). `not_answering` and `ended` reuse the
    subtitle's own translated keys, so the two surfaces cannot disagree;
    `reconnecting` has its own bare title (`relay.health.reconnecting`), since
    the link key it once borrowed carries an ellipsis meant for a status line
    (review round 1, N2/N3). */
export function healthLabel(health: SessionHealth): string {
	switch (health) {
		case "live":
			return messageRuntime.t("relay.health.live");
		case "not_answering":
			return messageRuntime.t("relay.chrome.session.subtitle.not_answering");
		case "reconnecting":
			return messageRuntime.t("relay.health.reconnecting");
		case "ended":
			return messageRuntime.t("relay.chrome.session.subtitle.ended");
	}
}

/** The sheet's ONE explanation sentence per state: what is true and what a
    send does now (review round 1, U1/D2/D4/U6). Each is checked against the
    daemon, not hoped:
    - `degraded`: the relay's dial is down, so the entry has no writer;
      `request()` raises and the command route answers 409 at once, and the
      composer puts the text back. Nothing queues.
    - `heartbeat_stale` with the dial up: the frame IS written, the route waits
      15 s for an ack and answers 504 — the message may still land once the
      runtime catches up (and a retry under the same `command_id` is deduped by
      the runtime), so the sentence says check before retrying.
    - `reconnecting`: the PHONE's link is down; nothing is known about the
      computer, which may well still be running the session.
    - `ended`: history kept, and a send or Resume reopens it in the home folder
      (the daemon wakes a host there for a prompt, issue #1875).
    `degraded` wins over a stale beat for the same reason it does in
    `sessionHealth`: a dial that is down is the fact a send meets first. */
export function healthDetail(health: SessionHealth, projection: HealthInput): string {
	switch (health) {
		case "live":
			return messageRuntime.t("relay.health.live_detail");
		case "not_answering":
			return projection.degraded
				? messageRuntime.t("relay.health.dropped_detail")
				: messageRuntime.t("relay.health.quiet_detail");
		case "reconnecting":
			return messageRuntime.t("relay.health.reconnecting_detail");
		case "ended":
			return messageRuntime.t("relay.health.ended_detail");
	}
}

/* THE ACCEPTANCE LINE'S TWO SPELLINGS (UX round 2, U24). The first is what a
   cold child start honestly looks like; the second is the same fact once "a
   few seconds" has stopped describing it — the session still has not come
   back, which is what the reader needs to decide to wait or retry. 20s is
   generous for the ordinary respawn (the live leg's own real child took ~2s)
   and short enough that a stuck one is not dressed up as a normal start.
   Moved here verbatim with the resume path from the retired ended strip. */
const REOPENING = "reopening — this can take a few seconds";
const STILL_REOPENING = "still reopening — it has not come up yet";
const REOPEN_REPORT_MS = 20_000;

/**
 * The documented resume affordance for an ENDED session (mobile UX batch 2,
 * U7), moved from the retired `EndedSessionStrip` so there is ONE copy of it.
 * docs/mobile.md: "the phone card flips to *ended*, offering resume". A tap
 * reopens the conversation as a NEW live session — the same route the
 * past-sessions screen uses (`POST /api/sessions/resume`: the daemon spawns a
 * child that resumes the transcript) — and the router takes the phone to it.
 *
 * HELD BY THE HEADER, NOT THE SHEET. The sheet unmounts when it closes, and the
 * acceptance line has to survive the reader closing it to watch the dot. The
 * strip used to end the line by unmounting when a live frame flipped `ended`;
 * `ended` going false is that same signal here, so it clears the state.
 */
export function useResume(sessionId: string, ended: boolean) {
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState("");
	/* THE ACCEPTANCE LINE (UX round 2, U24). A resume whose POST succeeds but
	   whose session does not come back live ends the same way it started
	   otherwise — one click, no word — and the reader cannot tell accepted
	   from failed from still-coming-up. The line says which; it is also capped:
	   "a few seconds" stops being true long before the session does, and a
	   stale acceptance reads like a promise the daemon is not keeping — past the
	   cap the sentence says what is actually true instead. */
	const [notice, setNotice] = useState("");
	useEffect(() => {
		if (ended) return;
		setBusy(false);
		setError("");
		setNotice("");
	}, [ended]);
	const resume = async () => {
		if (busy) return;
		setBusy(true);
		setError("");
		setNotice("");
		try {
			const r = await resumeSession(sessionId);
			/* THE SAME-ROUTE NO-OP (UX round 1, U16 / agent-review MINOR 1). The
			   route echoes the id it was given, so this navigation usually lands on
			   the route that is ALREADY mounted — this hook, and its `busy`, survive
			   it. Until this reset, a POST that succeeded against a session that did
			   not come back live left `resuming…` disabled forever. Retrying is
			   safe: `spawn_session` coalesces concurrent resumes onto one
			   constructor and a later attempt reattaches the same owner. */
			setBusy(false);
			setNotice(REOPENING);
			navigate(`/s/${encodeURIComponent(r.session_id)}`);
		} catch (e) {
			/* One refusal voice (design round 1, D5): prefixed like the pin's,
			   instead of the raw daemon string. */
			setError(resumeRefusalText(e));
			setBusy(false);
		}
	};
	useEffect(() => {
		if (notice !== REOPENING) return;
		const t = setTimeout(() => setNotice(STILL_REOPENING), REOPEN_REPORT_MS);
		return () => clearTimeout(t);
	}, [notice]);
	return { busy, error, notice, resume };
}

/**
 * The header's title block as ONE control: the conversation name (passed in as
 * `children`), then the dot and the liveness words; a tap opens the
 * explanation sheet.
 *
 * WHY THE WHOLE BLOCK IS THE BUTTON. The status line alone is one 17 px meta
 * row, and the guidelines' 44 px target cannot come from growing the 56 px
 * header. Stretching a pseudo-element over the title instead would cover the
 * name's own `title` (its truncation recovery, operator finding B) with this
 * button's. As the block, the target is the block's own `min-h-11` inside the
 * existing row, and the innermost `title` still wins on hover: the name over
 * the name, the state word over the status line.
 */
export function SessionHealthControl({
	projection,
	connected,
	open,
	onOpenChange,
	resume: { busy, error, notice, resume },
	children,
}: {
	projection: SessionProjection;
	connected: boolean;
	/* OWNED BY THE HEADER, with the resume state (UX U3): the ⋮ menu's Resume
	   runs the same `useResume` and then opens THIS sheet, so its acceptance
	   line or refusal is seen where the action belongs — never a second copy of
	   the path, and never a refusal left behind a closed menu. */
	open: boolean;
	onOpenChange: (open: boolean) => void;
	resume: ReturnType<typeof useResume>;
	children: ReactNode;
}) {
	const health = sessionHealth(projection, connected);
	const label = healthLabel(health);
	const buttonRef = useRef<HTMLButtonElement>(null);
	const setOpen = onOpenChange;
	return (
		<>
			<button
				ref={buttonRef}
				type="button"
				data-health={health}
				title={label}
				aria-haspopup="dialog"
				aria-expanded={open}
				onClick={() => setOpen(true)}
				className="flex min-h-11 w-full min-w-0 flex-col justify-center rounded-sm text-start active:bg-elevated"
			>
				{children}
				<span className="flex min-w-0 items-center gap-1.5">
				<span aria-hidden data-health-dot="" className={cn("inline-block size-2 shrink-0 rounded-full", HEALTH_DOT_TONE[health])} />
				{/* THE NAME IS THE VISIBLE WORDS, plus the state word where they do not
				    already say it. The subtitle reads "Ended" / "Not answering" /
				    "Reconnecting · …" for the other states, but "Idle" / "Working · 3s"
				    / "Waiting for you" while live — so only `live` adds its word, as
				    screen-reader text. An `aria-label` would replace the visible
				    words and break label-in-name for speech users. */}
				{health === "live" ? <span className="sr-only">{label} · </span> : null}
				<span className="min-w-0">
					<SessionSubtitle projection={projection} connected={connected} />
				</span>
				</span>
			</button>
			{/* THE TRANSITION ANNOUNCEMENT (UX round 1, U2). The retired ended and
			    degraded bands were `role="status"` regions inserted on the change,
			    so a screen-reader user composing in a session that died or wedged
			    heard it happen; with the bands gone the dot alone would say it to
			    sighted readers only. One polite region, OUTSIDE the button (inside,
			    its words would be read twice as part of the control's name), that
			    holds the state's own word while not live and is empty while live —
			    a region only speaks on a CHANGE, so mounting into a state is not
			    repeated and each move into another state is said once. */}
			<span className="sr-only" aria-live="polite" data-health-announce="">{health === "live" ? "" : label}</span>
			<Sheet open={open} onClose={() => setOpen(false)} title={label} returnFocusRef={buttonRef}>
				<div className="flex flex-col gap-3 p-3">
					{/* ONE explanation register for every state (design D4): a single
					    `ink-muted` paragraph that adds what the title does not say. */}
					<p className="text-body-sm text-ink-muted">{healthDetail(health, projection)}</p>
					{health === "ended" ? (
						<>
							{/* THE SHEET'S MAIN ACTION SPENDS THE ACCENT (design D1). With the
							    sheet open Resume is the one action and the reason it was
							    opened; `tonal` measured 1.05:1 (light) / 1.11:1 (dark) against
							    the sheet and read as a tinted slab. `accent` on `elevated` is
							    >=4.53:1 in every shipped theme, and the other one-action sheets
							    (metadata save, project create) already use `primary`. */}
							<Button variant="primary" disabled={busy} onClick={() => void resume()}>
								{busy ? messageRuntime.t("relay.chrome.session.resuming") : messageRuntime.t("relay.chrome.session.resume")}
							</Button>
							{notice ? <p role="status" className="text-meta text-ink-dim">{notice}</p> : null}
							{error ? <p role="alert" className="text-meta break-words text-danger">{error}</p> : null}
						</>
					) : null}
				</div>
			</Sheet>
		</>
	);
}

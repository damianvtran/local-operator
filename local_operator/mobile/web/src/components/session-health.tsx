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
import { sessionHealth, type SessionHealth } from "../lib/session-health";
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

/** The state's own word: the dot's accessible name, its `title`, and the
    sheet's title. Colour is never the only signal. Every arm but `live` reuses a
    key that already ships translated (the subtitle's own words), so the two
    surfaces say the same thing in every locale. */
export { sessionHealth, type SessionHealth };

export function healthLabel(health: SessionHealth): string {
	switch (health) {
		case "live":
			return messageRuntime.t("relay.health.live");
		case "not_answering":
			return messageRuntime.t("relay.chrome.session.subtitle.not_answering");
		case "reconnecting":
			return messageRuntime.t("relay.chrome.link.reconnecting");
		case "ended":
			return messageRuntime.t("relay.chrome.session.subtitle.ended");
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
	sessionId,
	projection,
	connected,
	children,
}: {
	sessionId: string;
	projection: SessionProjection;
	connected: boolean;
	children: ReactNode;
}) {
	const health = sessionHealth(projection, connected);
	const label = healthLabel(health);
	const [open, setOpen] = useState(false);
	const buttonRef = useRef<HTMLButtonElement>(null);
	const { busy, error, notice, resume } = useResume(sessionId, health === "ended");
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
			{/* The link-down announcement, moved out of the subtitle so it sits
			    OUTSIDE the button: inside, its words would be read twice as part
			    of the control's name. Only the transition is announced. */}
			<span className="sr-only" aria-live="polite">{health === "reconnecting" ? messageRuntime.t("relay.chrome.link.reconnecting") : ""}</span>
			<Sheet open={open} onClose={() => setOpen(false)} title={label} returnFocusRef={buttonRef}>
				<div className="flex flex-col gap-3 p-3">
					{health === "ended" ? (
						<>
							<div className="flex flex-col gap-1">
								<p className="text-body-sm text-ink">{messageRuntime.t("relay.chrome.session.ended")}</p>
								{/* WHERE IT REOPENS, AND THAT A SEND DOES IT TOO (UX round 1, U18;
								    issue #1875). The daemon resumes the transcript in the owner's
								    home and wakes a host for a prompt to an ended session, so resume
								    is the way to reopen WITHOUT composing, not a prerequisite. */}
								<p className="text-meta text-ink-dim">{messageRuntime.t("relay.chrome.session.ended_hint")}</p>
							</div>
							<Button disabled={busy} onClick={() => void resume()}>
								{busy ? messageRuntime.t("relay.chrome.session.resuming") : messageRuntime.t("relay.chrome.session.resume")}
							</Button>
							{notice ? <p role="status" className="text-meta text-ink-dim">{notice}</p> : null}
							{error ? <p role="alert" className="text-meta break-words text-danger">{error}</p> : null}
						</>
					) : (
						<p className="text-body-sm text-ink-muted">
							{health === "live"
								? messageRuntime.t("relay.health.live_detail")
								: health === "not_answering"
									? messageRuntime.t("relay.chrome.session.degraded")
									: messageRuntime.t("relay.chrome.drawer.status", { state: "retrying" })}
						</p>
					)}
				</div>
			</Sheet>
		</>
	);
}

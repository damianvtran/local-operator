/**
 * Pending request card — pinned above the composer, accent-bordered, the
 * most prominent element on screen (branding §7: a question for the user is
 * the only thing that needs a decision, and it must be unmissable).
 *
 * Approval: tool name + detail + Approve/Deny (+ remember). Ask: the
 * question plus options as tap targets (each with its consequence line, U3),
 * or a paste field — masked when the ask is a secret credential (D1/U2) —
 * when the daemon offered no options. A multi-question ask shows a
 * "Question N of M" header and re-renders the next question after each answer
 * (U1).
 *
 * The card is an UNSHRINKABLE sibling of the transcript inside the session
 * column, which is `h-dvh overflow-hidden` (screens/session-view.tsx): whatever
 * height the card claims is taken off the transcript, and anything past the
 * column's foot is clipped with no way to scroll to it. An uncapped card
 * therefore did not merely look bad — a 10-option ask measured 1426px against
 * an 844px viewport, so the last four options and the composer were off screen
 * and unreachable by any gesture.
 *
 * THREE regions, and which region a thing lives in is the whole contract:
 *
 *   meta row   shrink-0, pinned ABOVE the scroller
 *   body       min-h-0 flex-1 `lo-scroll` — title, detail, options
 *   controls   shrink-0, pinned BELOW the scroller — and the error line
 *
 * The split is what review round 1 bought. Putting the whole body in one
 * scroller (the first shape of this fix) capped the card correctly and then
 * let the DECISION scroll out of reach: an approval with a long detail arrived
 * with `approve` showing 30 of its 44px at 390x844 and 0 of 44px — invisible —
 * at 360x780, where the uncapped card before it had shown both buttons whole
 * (U2/Q1). The stale-tap error landed ~26px below the fold for the same reason,
 * so a refused tap greyed every option out and explained nothing (U3). Content
 * may scroll; the control that answers the card may not. A long option list can
 * still cost one gesture to reach its foot — that is inherent to a list longer
 * than the card — but the thing the user must press never does.
 *
 * The title stays INSIDE the scroller with the detail, deliberately: on a phone
 * a paragraph-length question can outgrow the cap on its own, and pinning it
 * would rebuild the same unreachable tail the cap exists to prevent. Only the
 * one-line meta row is pinned, which cannot outgrow anything and is what keeps
 * the card legible as a decision at the moment of the decision (D2).
 *
 * The cap is a fraction of the COLUMN, not of the dynamic viewport — see
 * `lib/column.ts` for why those are different numbers the moment a keyboard
 * opens, and for the 360x780 measurement where a `60dvh` card put `send` under
 * the column's clipped foot (C1/U1). It applies to every variant — options,
 * free-text/secret, and the approval's approve/deny pair — because all three
 * overflow the same column.
 *
 * Transient card state (`busy`/`error`/`remember`/free-text draft) lives in
 * component-local `useState`, so it MUST NOT survive from one question to the
 * next: a stale `busy` leaves the next question's options disabled and
 * untappable, a stale draft carries a typed answer forward. The invariant is
 * kept structurally by the render site (screens/session-view.tsx), which keys
 * this card on `request_id`+`question_index` — the daemon keeps the same
 * `request_id` pending across a multi-question ask and only advances
 * `question_index`, so that key changes per question and React remounts the
 * whole card with fresh state. The key lives at the render site, not inside
 * this component, because a component cannot key itself; that is why there is
 * no per-field remount key here.
 */
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { sendCommand } from "../api";
import { cn } from "../lib/cn";
import { PENDING_CARD_FRACTION, columnCap } from "../lib/column";
import type { PendingRequest } from "../types";

/** Turn a raw command error into copy a phone user can act on. The daemon and
    registrant speak in developer terms (HTTP status strings, "session not
    connected"); a person staring at a phone needs the human version (U4/U7). */
function humanizeError(message: string): string {
	const m = message.toLowerCase();
	if (m.includes("moved on")) {
		/* The picker advanced (usually a terminal answer to this question)
		   while the tap was in flight; the card is about to repaint to the
		   current question (U8/U9) — distinct from "already answered". */
		return "That question moved on — showing the current one.";
	}
	if (m.includes("already answered")) {
		/* Stale tap: the question settled on another surface first (U4). */
		return "Already answered — this question was settled on the terminal.";
	}
	if (m.includes("no longer waiting")) {
		return "Already answered — this question is no longer waiting.";
	}
	if (m.includes("not connected") || m.includes("409")) {
		return "The terminal session went away — reopen it to answer.";
	}
	if (m.includes("did not answer") || m.includes("504")) {
		return "The session didn’t respond in time — try again.";
	}
	return message;
}

export function PendingCard({
	pid,
	pending,
	count = 1,
}: {
	pid: string;
	pending: PendingRequest;
	/** Total requests waiting, including this one. A parallel tool batch can
	    open several approvals at once; when >1 the card shows a "1 of N" badge
	    so the user knows more follow, and answering this one reveals the next
	    on the repaint. */
	count?: number;
}) {
	const [remember, setRemember] = useState(false);
	const [freeText, setFreeText] = useState("");
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState("");

	/* Whether the body has content below the fold, which drives the bottom fade
	   (U4). Derived by measurement rather than from the content, because whether
	   the card overflows depends on the cap, the keyboard and the option count at
	   once — a 3-option ask does not overflow and must not wear the cue. Measured
	   on scroll, on resize (the keyboard changes the cap), and on content change
	   via ResizeObserver, since none of those fire the others. */
	const scrollerRef = useRef<HTMLDivElement>(null);
	const [moreBelow, setMoreBelow] = useState(false);
	const syncOverflow = useCallback(() => {
		const el = scrollerRef.current;
		if (!el) return;
		/* 4px of slack: sub-pixel layout leaves a fraction of a pixel at the true
		   bottom, which would otherwise keep the fade painted on a fully-read
		   card and teach the user to distrust it. */
		setMoreBelow(el.scrollHeight - el.scrollTop - el.clientHeight > 4);
	}, []);
	useLayoutEffect(syncOverflow);
	useEffect(() => {
		const el = scrollerRef.current;
		if (!el || typeof ResizeObserver === "undefined") return;
		const ro = new ResizeObserver(syncOverflow);
		ro.observe(el);
		for (const child of Array.from(el.children)) ro.observe(child);
		window.addEventListener("resize", syncOverflow);
		return () => {
			ro.disconnect();
			window.removeEventListener("resize", syncOverflow);
		};
	}, [syncOverflow]);

	const answer = async (fn: () => Promise<unknown>) => {
		if (busy) return;
		setBusy(true);
		setError("");
		try {
			await fn();
		} catch (e) {
			setError(humanizeError(String((e as Error).message ?? e)));
			setBusy(false);
		}
		/* No local reset on the success path. For a single-question ask the
		   next repaint clears `pending` and unmounts this card; for a
		   multi-question ask `pending` stays non-null (same request_id, next
		   question_index) and the render-site key remounts the card fresh.
		   Either way this instance's `busy`/`error` never need clearing here —
		   and must not be cleared, or D7 (options inert while in flight) breaks
		   within the current question. */
	};

	const approve = (approved: boolean) =>
		answer(() =>
			sendCommand(pid, {
				op: "approval_answer",
				request_id: pending.request_id,
				approved,
				remember,
			}),
		);

	const answerAsk = (value: string) =>
		answer(() =>
			sendCommand(pid, {
				op: "ask_answer",
				request_id: pending.request_id,
				value,
				/* The question this card is currently showing. The daemon
				   rejects the answer if the picker advanced past it (U8). */
				question_index: pending.question_index,
			}),
		);

	/* A multi-question ask advances one question at a time (U1): show which
	   question this is so the user knows the card is not the whole prompt. */
	const multiQuestion = pending.question_total > 1;

	/* When an answer is rejected as stale/moved-on, the card is either about to
	   unmount (terminal settled it) or repaint to a new question. Until that
	   repaint lands, its options must stop reading as tappable — otherwise the
	   error line sits under buttons that still look live (D7). */
	const inert = busy || error !== "";

	const optionCount = pending.kind === "approval" ? 0 : pending.options.length;

	return (
		/* `shrink-0` keeps the card at its content height (up to the cap) rather
		   than letting the column squeeze it toward nothing when the transcript is
		   long; the cap is what stops it taking the whole column. The padding
		   stays on this element so the scrollbar rides the card's inner edge
		   instead of overlapping the accent border.

		   `data-testid` rather than the accent border class as the test handle:
		   `border-accent` is also applied by the composer on drag-over and by the
		   new-session screen on selection, so a class selector picks the wrong
		   node the first time one of those states renders alongside a card (C5). */
		<div
			data-testid="pending-card"
			style={columnCap(PENDING_CARD_FRACTION)}
			className="border-accent bg-accent-wash mx-2 flex shrink-0 flex-col rounded-md border p-2.5"
		>
			{/* PINNED meta row (D2). One line, fixed height, cannot outgrow
			    anything — so pinning it costs no reachability while keeping the
			    card legible as a decision at the moment of the decision. Scrolled
			    to the option the user means to tap, 0% of the kind label and 0% of
			    the counter were still on screen; a card that says nothing about
			    what is being asked is a bare list of buttons.

			    The option total rides here too (U4): the cap trades first-glance
			    information (6 visible options of 10 became 3 at 390x844) for a
			    bounded card, and stating the count is the one affordance that
			    costs no vertical space at all. */}
			<span className="flex shrink-0 items-center justify-between text-meta text-accent">
				<span>
					{pending.kind === "approval"
						? "approval needed"
						: pending.secret
							? "secret requested"
							: "question"}
					{optionCount > 0 ? (
						<span className="text-ink-dim"> · {optionCount} options</span>
					) : null}
				</span>
				{multiQuestion ? (
					<span className="font-mono text-mono-sm text-ink-dim">
						Question {pending.question_index + 1} of{" "}
						{pending.question_total}
					</span>
				) : count > 1 ? (
					<span className="font-mono text-mono-sm text-ink-dim">
						1 of {count}
					</span>
				) : null}
			</span>

			{/* SCROLLING content: the title, the detail, and the option list.
			    The title rides inside rather than pinned beside the meta row on
			    purpose — on a phone a paragraph-length question outgrows the cap
			    by itself, and pinning it would rebuild the unreachable tail the
			    cap exists to prevent. `min-h-0` is required: a flex child defaults
			    to `min-height: auto` and would refuse to shrink below its content,
			    which is precisely how the card grew past the viewport.

			    `overflow-x-hidden` + `break-words` are the backstop the
			    transcript's scroller documents for the same reason (C7): option
			    labels and descriptions are arbitrary agent-supplied strings, and
			    one long unbroken token would otherwise scroll the card sideways. */}
			<div
				ref={scrollerRef}
				onScroll={syncOverflow}
				className={cn(
					"lo-scroll mt-1 flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto overflow-x-hidden break-words",
					/* The one static cue that the card scrolls (U4). The overlay
					   scrollbar has zero layout width and paints nothing at rest,
					   so before this the only hint was a half-clipped row — absent
					   entirely on the approval card, where the clipped thing is a
					   button. A bottom fade steals no row and reads at arm's
					   length; it is removed the moment the foot is reached so a
					   fully-read card does not look truncated. */
					moreBelow && "lo-fade-b",
				)}
			>
				<div className="flex flex-col gap-0.5">
					<span className="text-body font-medium">{pending.title}</span>
					{pending.detail ? (
						<p className="text-body-sm text-ink-muted whitespace-pre-wrap">
							{pending.detail}
						</p>
					) : null}
				</div>

				{optionCount > 0 ? (
					<div className="flex flex-col gap-2">
						{pending.options.map((opt) => (
							<button
								key={opt.label}
								type="button"
								disabled={inert}
								onClick={() => answerAsk(opt.label)}
								/* Accent-tinted left edge + elevated fill so an option
								   reads as a tap target, not a static label or a text
								   field (D2). Disabled dims (D3/D4) — including while an
								   error is shown, so a stale option stops reading as
								   live under the message (D7). */
								className={cn(
									"flex min-h-11 flex-col justify-center rounded-sm border border-l-2 border-control border-l-accent bg-elevated px-3 py-2 text-left active:bg-accent-wash disabled:opacity-50",
								)}
							>
								<span className="text-body-sm font-medium text-ink">
									{opt.label}
								</span>
								{opt.description ? (
									<span className="text-body-sm text-ink-muted">
										{opt.description}
									</span>
								) : null}
							</button>
						))}
					</div>
				) : null}
			</div>

			{/* PINNED controls (U2/Q1) and the error line (U3), `shrink-0` and
			    OUTSIDE the scroller. The option list above may cost a gesture to
			    reach its foot — inherent to a list longer than the card — but the
			    control that answers the card may not. Inside the scroller, an
			    approval with a long detail arrived with `approve` showing 30 of
			    44px at 390x844 and 0 of 44px at 360x780, and the stale-tap error
			    landed ~26px below the fold because it is appended below wherever
			    the user is standing.

			    `pr-1.5` keeps the rows clear of the overlay scrollbar's paint band
			    (D3): the thumb has no layout width, so it painted over the right
			    edge of `deny` and `send` at `gapToContentEdge = 0`. Only the
			    control rows carry it — over prose the overlay behaviour is the
			    app-wide `lo-scroll` idiom and was ruled acceptable, and a gutter
			    would cost 4px of line length on every card at 360px. */}
			{pending.kind === "approval" ? (
				<div className="mt-2 flex shrink-0 flex-col gap-2 pr-1.5">
					<label className="flex min-h-11 items-center gap-2 text-body-sm text-ink-muted select-none">
						<input
							type="checkbox"
							checked={remember}
							onChange={(e) => setRemember(e.target.checked)}
							className="size-4 accent-accent"
						/>
						remember this choice
					</label>
					<div className="flex gap-2">
						<button
							type="button"
							disabled={busy}
							onClick={() => approve(true)}
							className="flex min-h-11 flex-1 items-center justify-center rounded-sm bg-accent text-body-sm font-medium text-on-accent active:bg-accent-active disabled:opacity-50"
						>
							{busy ? "…" : "approve"}
						</button>
						<button
							type="button"
							disabled={busy}
							onClick={() => approve(false)}
							className="flex min-h-11 flex-1 items-center justify-center rounded-sm border border-danger-border bg-danger-wash text-body-sm text-danger active:bg-danger-wash disabled:opacity-50"
						>
							{busy ? "…" : "deny"}
						</button>
					</div>
				</div>
			) : optionCount === 0 ? (
				/* Same footer rhythm as the approval above (design D5), which is
				   the variant that had one: `gap-2` between a grounding non-control
				   row and the controls, and the card's own `p-2.5` as the only
				   thing below them. Measured before this, the two cards' footers
				   sat on different ground — the approval's controls opened 60px
				   below the scroller with the `remember` row between, while the
				   secret's input sat 8px under text that scrolls beneath it with
				   nothing between, and its reassurance line hanging BELOW the
				   controls made the card bottom-heavy at 32.4px against the
				   approval's 11px.

				   The reassurance line is what grounds this footer, so it moves
				   ABOVE the input rather than a hairline being added beside it: a
				   second divider idiom next to the approval's row is the
				   competing-patterns bug, and a warning about a credential is one
				   a user should read BEFORE pasting it, not after. */
				<div className="mt-2 flex shrink-0 flex-col gap-2 pr-1.5">
					{pending.secret ? (
						<p className="text-meta text-ink-dim">
							secret — sent directly, not shown in the transcript
						</p>
					) : null}
					<div className="flex gap-2">
						<input
							/* No per-field remount key needed: the whole card is keyed
							   on request_id+question_index at the render site, so this
							   input (and the free-text draft) is already fresh per
							   question of a multi-part ask (U1). */
							value={freeText}
							onChange={(e) => setFreeText(e.target.value)}
							/* Secret asks are credentials: mask the value on screen
							   and suppress the keyboard's learn/suggest so a token is
							   not shoulder-surfable or captured (D1/U2). */
							type={pending.secret ? "password" : "text"}
							autoComplete={pending.secret ? "off" : undefined}
							autoCapitalize={pending.secret ? "none" : undefined}
							autoCorrect={pending.secret ? "off" : undefined}
							spellCheck={pending.secret ? false : undefined}
							placeholder={pending.secret ? "paste secret" : "your answer"}
							className="min-h-11 min-w-0 flex-1 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim"
						/>
						<button
							type="button"
							disabled={inert || !freeText.trim()}
							onClick={() => answerAsk(freeText.trim())}
							className="flex min-h-11 items-center justify-center rounded-sm bg-accent px-4 text-body-sm font-medium text-on-accent active:bg-accent-active disabled:opacity-50"
						>
							{busy ? "…" : "send"}
						</button>
					</div>
				</div>
			) : null}

			{error ? (
				<p className="mt-1 shrink-0 pr-1.5 text-body-sm text-danger">{error}</p>
			) : null}
		</div>
	);
}

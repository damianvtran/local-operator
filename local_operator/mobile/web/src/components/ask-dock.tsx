/**
 * Ask dock — the MINIMIZED ask affordance (design §5.0, R7).
 *
 * ONE LINE, ABOVE THE COMPOSER, and it is the *minimized* half of the two-state
 * model the design fixes for every surface: an ask is either MINIMIZED (this
 * chip, no answer surface mounted) or EXPANDED (the asks sheet, which is the
 * answer surface).
 *
 * A CHIP, NOT A STRIP (design round 1, D1). The first version was a full-bleed
 * tinted row: the wash spanned x=0..390 with no radius and no inset, 1.20:1
 * against the canvas — read as a banner, which §5.0 names as the thing this must
 * NOT be ("a chip … not a modal, a toast or a banner"), and it was the only
 * edge-to-edge tint in a composer stack whose siblings (`pending-card.tsx`, the
 * working line) are either inset-and-rounded or canvas-backed. It now wears the
 * approval card's own shape — inset, rounded, accent-bordered, wash-backed — so
 * the two things that can sit in this slot read as the same kind of object.
 *
 * WHAT IT SAYS, AND WHAT IT MUST NOT SAY. The count is the number of QUESTIONS
 * still waiting across this session's outstanding asks — the number the user can
 * act on — followed by the head ask's own question, truncated. It never says
 * "needs you" and never carries danger ink: the agent keeps working while an ask
 * is queued (§5's header rule), so an outstanding ask is not a run held hostage
 * the way an approval is.
 *
 * THE NAMED ASK IS THE ONE THE SHEET LEADS WITH. The preview comes from
 * `dockAsk` (the oldest open ask, or the first answerable timeout when nothing
 * is open) and `AsksSheet` orders its rows the same way, because a thumb that
 * taps a chip reading "Which sequencing…" and lands on a different question has
 * been told one thing and shown another (UX round 1, U8).
 *
 * ABSENT AT ZERO. The bar is not rendered (never rendered empty, never a zero
 * badge), so its presence is itself the statement.
 */
import { cn } from "../lib/cn";
import { dockAsk, outstandingAsks } from "../lib/asks";
import type { PendingAsk } from "../types";

export function AskDock({
	rows,
	onOpen,
}: {
	rows: PendingAsk[] | undefined | null;
	/** Summon the asks sheet — the answer surface. */
	onOpen: () => void;
}) {
	const outstanding = outstandingAsks(rows);
	if (outstanding.length === 0) return null;
	const head = dockAsk(rows);
	/* Count QUESTIONS, not asks: "2 asks waiting" is true but tells a user
	   nothing about how much they owe, and the design's own line counts
	   questions (`? 3 questions waiting`). A timed-out ask is still counted —
	   it is still answerable — which is why this reads the outstanding set
	   rather than only the open one. */
	const questions = outstanding.reduce(
		(total, row) => total + (Array.isArray(row.questions) ? row.questions.length : 0),
		0,
	);
	const label = questions === 1 ? "1 question waiting" : `${questions} questions waiting`;
	const preview = String(head?.questions?.[0]?.question ?? "").trim();

	return (
		<button
			type="button"
			data-testid="ask-dock"
			data-ask-count={outstanding.length}
			onClick={onOpen}
			/* The whole chip is the target, so the chevron is decoration rather
			   than a second control: two hit targets on one line is how a bar
			   becomes a toolbar. `text-ink-muted` on the count and the preview
			   `text-ink` — the ORDER the design asked for is that the thing being
			   asked is the most legible mark on the line (design round 1, D3). */
			className={cn(
				"mx-2 mb-1 flex min-h-11 items-center gap-2",
				"rounded-md border border-accent bg-accent-wash px-3 py-1.5 text-left active:bg-elevated",
			)}
		>
			{/* A persistent accent on the glyph, never animated: a pulse would be
			   the focus steal this design exists to avoid, in colour instead of
			   keys (§5.0). */}
			<span aria-hidden className="shrink-0 font-mono text-body-sm text-accent">
				?
			</span>
			<span className="shrink-0 text-body-sm text-ink-muted">{label}</span>
			{preview ? (
				<span className="min-w-0 flex-1 truncate text-body-sm text-ink">· {preview}</span>
			) : (
				<span className="min-w-0 flex-1" />
			)}
			<span aria-hidden className="shrink-0 text-accent">
				▸
			</span>
		</button>
	);
}

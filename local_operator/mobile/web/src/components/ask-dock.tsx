/**
 * Ask dock — the MINIMIZED ask affordance (design §5.0, R7).
 *
 * ONE LINE, ABOVE THE COMPOSER, and it is the *minimized* half of the two-state
 * model the design fixes for every surface: an ask is either MINIMIZED (this
 * chip, no answer surface mounted) or EXPANDED (the asks sheet, which is the
 * answer surface). It is a chip in the `pending-card.tsx` family rather than a
 * modal, a toast or a banner — a badge that slides in over the transcript is
 * the focus steal the design exists to avoid, in pixels instead of keys.
 *
 * WHAT IT SAYS, AND WHAT IT MUST NOT SAY. The count is the number of questions
 * still waiting across this session's outstanding asks — the number the user
 * can act on — followed by the HEAD ask's own question, truncated. It never
 * says "needs you" and never carries danger ink: the agent keeps working while
 * an ask is queued (§5's header rule), so an outstanding ask is not a run held
 * hostage the way an approval is.
 *
 * ABSENT AT ZERO. The bar is not rendered (never rendered empty, never a zero
 * badge), so its presence is itself the statement.
 *
 * THE HEAD IS THE OLDEST OPEN ASK, not the first row of the wire list: the list
 * leads with the NEWEST, and a bar whose label jumped to each new arrival would
 * change under a thumb that is already moving toward it (`lib/asks.headAsk`,
 * the same rule `asks/render.mirror_card` uses for the legacy card).
 */
import { cn } from "../lib/cn";
import { headAsk, outstandingAsks } from "../lib/asks";
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
	const head = headAsk(rows);
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
	const preview = String(head?.questions?.[0]?.question || "").trim();

	return (
		<button
			type="button"
			data-testid="ask-dock"
			data-ask-count={outstanding.length}
			onClick={onOpen}
			/* The whole bar is the target (a 44px row), so the chevron is
			   decoration rather than a second control: two hit targets on one
			   line is how a bar becomes a toolbar. */
			className={cn(
				"flex min-h-11 w-full items-center gap-2 border-t border-hairline",
				"bg-accent-wash px-3 py-1.5 text-left active:bg-elevated",
			)}
		>
			{/* A persistent accent on the glyph, never animated: a pulse would be
			   the focus steal this design exists to avoid, in colour instead of
			   keys (§5.0). */}
			<span aria-hidden className="shrink-0 font-mono text-body-sm text-accent">
				?
			</span>
			<span className="shrink-0 text-body-sm text-ink">{label}</span>
			{preview ? (
				<span className="min-w-0 flex-1 truncate text-body-sm text-ink-muted">
					· {preview}
				</span>
			) : (
				<span className="min-w-0 flex-1" />
			)}
			<span className="shrink-0 text-meta text-ink-dim">tap to answer</span>
			<span aria-hidden className="shrink-0 text-accent">
				▸
			</span>
		</button>
	);
}

/**
 * The S2b grouping control: one 44px toggle beside a retry for a failed
 * catalogue read, on its own line under the search field, with one status line
 * underneath that explains what the toggle is currently doing.
 *
 * COPY IS ADVISORY HERE, following S2a's Option B (`session-tags-sheet.tsx`):
 * this module is outside the i18n enforcement scope, so the phone's view-only
 * copy does not grow the enforced catalogue allowance. `session-list.tsx` is
 * enforced and carries no literals from this feature.
 *
 * THE VISIBLE LABEL IS STATIC and names the dimension ("group by tags"), not
 * the state: the earlier `group`/`grouped` pair read as a verb whose past tense
 * was the state, which told a reader nothing about WHAT it grouped (round 1,
 * D1/U1). The on/off is carried by `aria-pressed` and by a pressed treatment
 * copied from `Chip`'s selected branch — the same two-token pair, so on and off
 * cannot drift apart the way two ad-hoc class lists would.
 *
 * THE STATUS LINE IS ONE SLOT, not a stack: an error outranks the paused hint,
 * which outranks the first-read status, because each later state makes the one
 * before it meaningless (there is nothing to wait for after a failure, and no
 * point naming a read while a search is holding the grouping off). `role` is
 * `status` throughout rather than `alert`: the flat list still works in every
 * one of these states, so nothing here interrupts a screen reader.
 *
 * The retry only exists while the screen reports a failed read, so it can never
 * be a dead control: the screen keeps the ordinary flat view until a catalogue
 * answers.
 */
import { cn } from "../lib/cn";

const TOGGLE_LABEL = "group by tags";
const RETRY_LABEL = "retry tags";

/** The one line under the controls, or `null` when grouping is simply on. */
function statusLine({
	busy,
	failed,
	paused,
}: {
	busy: boolean;
	failed: boolean;
	paused: boolean;
}): string | null {
	if (failed) return "could not load tags";
	if (paused) return "grouping paused while searching";
	if (busy) return "loading tags…";
	return null;
}

export function SessionGroupToggle({
	grouped,
	busy,
	failed,
	paused,
	onToggle,
	onRetry,
}: {
	grouped: boolean;
	/** The first catalogue read is outstanding; the list is still flat. */
	busy: boolean;
	failed: boolean;
	/** A non-empty search is holding the grouped view off. */
	paused: boolean;
	onToggle: () => void;
	onRetry: () => void;
}) {
	const status = statusLine({ busy, failed, paused });
	return (
		<div className="mx-2 mb-2 flex flex-col gap-1">
			{/* WRAPS, because both controls are sized by their text: at 195px CSS
			    (200% text) `group by tags` beside `retry tags` overflows one line,
			    and a `gap-2` row that cannot wrap pushes the retry off-screen
			    instead of onto a second line. The status line below is a block
			    paragraph, so it wraps within the available width already. */}
			<div className="flex flex-wrap items-center gap-2">
				<button
					type="button"
					aria-pressed={grouped}
					aria-label="Group by tags"
					onClick={onToggle}
					className={cn(
						"min-h-11 rounded-sm border px-3 text-body-sm",
						grouped
							? "border-accent bg-accent-wash text-accent"
							: "border-control bg-surface text-ink-muted",
					)}
				>
					{TOGGLE_LABEL}
				</button>
				{failed ? (
					<button
						type="button"
						aria-label="Retry tags"
						onClick={onRetry}
						className="min-h-11 rounded-sm border border-control bg-surface px-3 text-body-sm text-ink-muted"
					>
						{RETRY_LABEL}
					</button>
				) : null}
			</div>
			{status ? (
				<p role="status" className="px-1 text-body-sm text-ink-muted">
					{status}
				</p>
			) : null}
		</div>
	);
}

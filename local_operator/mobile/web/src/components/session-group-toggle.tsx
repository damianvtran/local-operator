/**
 * The S2b grouping control: one 44px toggle beside a retry for a failed
 * catalogue read, on its own line under the search field.
 *
 * COPY IS ADVISORY HERE, following S2a's Option B (`session-tags-sheet.tsx`):
 * this module is outside the i18n enforcement scope, so the phone's view-only
 * copy does not grow the enforced catalogue allowance. `session-list.tsx` is
 * enforced and carries no literals from this feature.
 *
 * The button is a real `<button>` with `aria-pressed`; the accessible name is
 * static ("Group by primary tag") because the pressed state already says on or
 * off, while the visible word stays terse enough for a phone row.
 *
 * The retry only exists while the screen reports a failed read, so it can never
 * be a dead control: the screen keeps the ordinary flat view until a catalogue
 * answers.
 */
export function SessionGroupToggle({
	grouped,
	failed,
	onToggle,
	onRetry,
}: {
	grouped: boolean;
	failed: boolean;
	onToggle: () => void;
	onRetry: () => void;
}) {
	return (
		<div className="mx-2 mb-2 flex items-center gap-2">
			<button
				type="button"
				aria-pressed={grouped}
				aria-label="Group by primary tag"
				onClick={onToggle}
				className="min-h-11 rounded-sm border border-control px-3 text-body-sm text-ink-muted"
			>
				{grouped ? "grouped" : "group"}
			</button>
			{failed ? (
				<button
					type="button"
					aria-label="Retry loading the tag catalogue"
					onClick={onRetry}
					className="min-h-11 rounded-sm border border-control px-3 text-body-sm text-ink-muted"
				>
					retry
				</button>
			) : null}
		</div>
	);
}

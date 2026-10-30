/**
 * SkillTray — the non-modal `$skill` list that sits above the composer.
 *
 * WHY IT IS NOT A SHEET. `SlashSheet` is the phone's modal picker and it works
 * because a slash command is a complete act: the user typed `/`, and focus
 * moving to the sheet's filter keeps the next characters going somewhere
 * useful. A skill name is typed INSIDE a sentence. A sheet here would take
 * focus out of the textarea, drop the software keyboard mid-word, and hand the
 * rest of the draft to a filter that rewrites the buffer when accepted. So this
 * tray never takes focus: each row answers `onPointerDown` with
 * `preventDefault` (the caret stays in the field, the keyboard stays up, and
 * the compatibility click never reaches the row), and the component installs no
 * key handler at all — hardware Enter still sends the message, exactly as it
 * does with the tray closed.
 *
 * PLACEMENT AND RHYTHM. `docs/design/mobile-relay-guidelines.md` puts
 * everything stacked between the transcript and the composer inside the one
 * above-composer slot (`components/above-composer.tsx`): one 8 px gap and the
 * slot's edge on the composer's `mx-2`. The tray carries those same numbers
 * (`mx-2`, `mb-2`; no top margin) but the Composer renders it, immediately
 * before its own card, rather than inside the slot's scroller. That is
 * deliberate: the list is anchored to the FIELD it edits, so it must not scroll
 * away with a stack of pending cards, and an item keeps its own height in the
 * slot anyway (`*:shrink-0`). When the slot has items its own `mb-2` supplies
 * the 8 px above the tray; the tray's `mb-2` supplies the 8 px below it. The
 * one gap survives either way.
 *
 * WHAT A ROW SHOWS. `$name`, the skill's one-line description (already cut to
 * 160 chars by the relay), and a `hidden` tag for a skill whose `hide`
 * frontmatter suppresses automatic selection — the picker still offers it,
 * because being TYPED is exactly the case `hide` leaves open. `$name` is a
 * Latin technical token and keeps `dir="ltr"` under the RTL document; the tag
 * and description follow the document's direction.
 */
import type { SkillEntry } from "../types";
import { messageRuntime } from "../store";

export function SkillTray({
	rows,
	onPick,
}: {
	/** The matched rows, already filtered and ordered by the caller. */
	rows: SkillEntry[];
	/** Accept one row: the composer replaces the token's span and restores the
	    caret. Called from `onPointerDown`, so it runs before focus could move. */
	onPick: (name: string) => void;
}) {
	if (rows.length === 0) return null;
	return (
		<ul
			data-skill-tray=""
			aria-label={messageRuntime.t("relay.skills.tray.label")}
			className="lo-scroll mx-2 mb-2 max-h-[min(45vh,22rem)] shrink-0 overflow-y-auto overflow-x-hidden rounded-frame bg-elevated py-1"
		>
			{rows.map((row) => (
				<li key={row.name}>
					<button
						type="button"
						/* The caret must not leave the field: `preventDefault` on
						   pointerdown suppresses focus transfer and the compatibility
						   mouse events, so the tap inserts without a keyboard blink.
						   The click handler is for a KEYBOARD-generated activation
						   only (`detail === 0`); pointer clicks have a non-zero
						   `detail` and are already handled. */
						onPointerDown={(event) => {
							event.preventDefault();
							onPick(row.name);
						}}
						onClick={(event) => {
							if (event.detail === 0) onPick(row.name);
						}}
						className="flex min-h-11 w-full select-none items-center gap-2 rounded-sm px-3 text-start active:bg-surface"
					>
						<span dir="ltr" className="shrink-0 font-mono text-mono-sm text-ink">
							${row.name}
						</span>
						{row.hidden ? (
							<span className="shrink-0 rounded-full bg-surface px-2 py-0.5 text-meta text-ink-dim">
								{messageRuntime.t("relay.skills.tray.hidden")}
							</span>
						) : null}
						<span className="min-w-0 flex-1 truncate text-body-sm text-ink-dim">
							{row.description}
						</span>
					</button>
				</li>
			))}
		</ul>
	);
}

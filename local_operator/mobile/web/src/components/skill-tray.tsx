/**
 * SkillTray — the non-modal `$skill` list that sits above the composer.
 *
 * WHY IT IS NOT A SHEET. `SlashSheet` is the phone's modal picker and it works
 * because a slash command is a complete act: the user typed `/`, and focus
 * moving to the sheet's filter keeps the next characters going somewhere
 * useful. A skill name is typed INSIDE a sentence. A sheet here would take
 * focus out of the textarea, drop the software keyboard mid-word, and hand the
 * rest of the draft to a filter that rewrites the buffer when accepted. So this
 * tray does not take focus BY POINTER: each row answers `onPointerDown` with
 * `preventDefault` (the caret stays in the field, the keyboard stays up, and
 * the compatibility click never reaches the row), and the pick itself fires on
 * the pointer RELEASE so a drag that scrolls the list cannot insert a name. The
 * component installs no key handler at all — hardware Enter still sends the
 * message, exactly as it does with the tray closed, and a row's own activation
 * path is a real button's native Enter/Space. The KEYBOARD's route into the
 * rows is the composer's (ArrowDown/ArrowUp from the field, which focus a row;
 * the same keys then walk the list): sequential focus cannot supply it, because
 * the tray precedes the composer card in DOM order and the field's neighbours
 * are the card's own attach/mic controls (UX round 3 U7, agent review M3).
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
 * Latin technical token and keeps `dir="ltr"` under the RTL document; the
 * description takes its own first-strong direction (skill descriptions are
 * author-written English in every shipped locale, and without `dir` the RTL
 * document also decides WHICH END a truncated description loses — the leading
 * ellipsis ate the subject and verb, design D2); the tag follows the document.
 *
 * THE CAP IS IN COLUMN UNITS (`columnCap`, lib/column.ts). The tray exists for
 * the keyboard-up state, where `vh` still measures the device height: the old
 * `min(45vh, 22rem)` took 69 % of a 508 px column and pushed the composer's
 * foot under the clipped edge (design D1 / agent review M2).
 *
 * WHILE A QUESTION IS WAITING the cap drops to TWO rows (`skillTrayCap`). The
 * card outranks the list, and at every phone column with the keyboard up the
 * 45 % tray pushed the question itself above the slot — the user was asked to
 * approve what they could not read (design D5). Two rows keep `$name`
 * discoverable and the list scrolls inside them.
 *
 * THE FOOT CUE is the repo's existing one — `lo-fade-y` plus `useScrollCues`,
 * the pair every other scroller here uses. `lo-scroll`'s scrollbar is an
 * overlay that paints nothing at rest, so a capped list whose last row is cut
 * by 4 px advertised nothing below the fold (design D3).
 */
import { useRef } from "react";
import type { SkillEntry } from "../types";
import { messageRuntime } from "../store";
import { textDirection } from "../lib/bidi";
import { skillTrayCap } from "../lib/column";
import { useScrollCues } from "../lib/use-scroll-cues";

export function SkillTray({
	rows,
	onPick,
	questionPending,
}: {
	/** The matched rows, already filtered and ordered by the caller. */
	rows: SkillEntry[];
	/** Accept one row: the composer replaces the token's span and restores the
	    caret. Called from a tap's pointer release (or a keyboard activation), so
	    focus never left the field on the way here. */
	onPick: (name: string) => void;
	/** True while an ask or approval is waiting. The list yields the column to
	    the question — two rows instead of 45 % (design D5). */
	questionPending: boolean;
}) {
	/* A tap picks; a DRAG does not. The tray is a scroller on a short column
	   (45vh), and picking on touch-down made a scroll gesture start by inserting
	   a name into the draft — every drag over a row fired the pick before the
	   browser knew it was a scroll. Pointer-down still runs (that is where focus
	   is kept in the field); the pick waits for the release and requires the
	   finger to have stayed within a tap's slop. A scroll that begins fires
	   pointercancel, so it picks nothing at all. */
	const start = useRef<{ id: number; x: number; y: number } | null>(null);
	/* The "more below" cue's JS half (Safari without scroll timelines); on
	   Chromium the class alone tracks the scroll position in CSS. */
	const scrollCue = useScrollCues();
	if (rows.length === 0) return null;
	return (
		<ul
			data-skill-tray=""
			aria-label={messageRuntime.t("relay.skills.tray.label")}
			ref={scrollCue}
			style={skillTrayCap(questionPending)}
			className="lo-scroll lo-fade-y mx-2 mb-2 shrink-0 overflow-y-auto overflow-x-hidden rounded-frame bg-elevated py-1"
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
							start.current = { id: event.pointerId, x: event.clientX, y: event.clientY };
						}}
						onPointerUp={(event) => {
							const origin = start.current;
							start.current = null;
							if (!origin || origin.id !== event.pointerId) return;
							if (Math.hypot(event.clientX - origin.x, event.clientY - origin.y) > 8) return;
							onPick(row.name);
						}}
						onPointerCancel={() => {
							start.current = null;
						}}
						onClick={(event) => {
							/* Only a KEYBOARD-generated activation has detail 0; a pointer
							   click (detail >= 1) was already decided by the release. */
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
						<span
							dir={textDirection(row.description)}
							className="min-w-0 flex-1 truncate text-body-sm text-ink-dim"
						>
							{row.description}
						</span>
					</button>
				</li>
			))}
		</ul>
	);
}

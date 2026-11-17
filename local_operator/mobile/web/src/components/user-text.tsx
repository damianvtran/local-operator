/**
 * The text of the person's own message, foldable when its turn is condensed.
 *
 * WHAT IT REPLACES. A condensed turn used to keep only the message's first line,
 * cut by CSS `truncate` to a bare "…". Anything after it was gone with no cue, and
 * the only way back was the separate "Took 8s · 7 actions" bar underneath, which
 * says nothing about the message and also opens every tool row (the operator's
 * report: a message that "gets cut off … you can't expand or collapse it").
 *
 * WHAT IT DOES.
 *  - `folded` false (the turn is open, or live): the whole text, exactly as before.
 *    No clamp, no control.
 *  - `folded` true AND the text is VERY long (`MESSAGE_FOLD_MIN_LINES` wrapped
 *    lines at the reader's width): the WHOLE text stays in the DOM (a screen
 *    reader reads all of it) but the box is clamped to `MESSAGE_PREVIEW_LINES`
 *    lines, with a labelled "Show more" / "Show less" button (`aria-expanded`,
 *    `aria-controls`, chevron down to open and up to close). A shorter message
 *    shows in full with no control: the fold must not appear all the time.
 *  - the reader's choice is this component's own state, separate from the turn
 *    bar: opening the turn (full text, no control) and condensing it again leaves
 *    an opened message opened, and "Show less" is still on screen.
 *
 * NO FADE ON THE PREVIEW. The transcript scroller already masks its own edges
 * (`lo-fade-y`); a second gradient here would stack with it at nested edges. The
 * ellipsis the clamp draws plus a worded control says "there is more, and this
 * opens it" more plainly than a fade does.
 *
 * THE CLAMP IS AN INLINE STYLE, not a class. An inline style resolves in the
 * vitest layer (no stylesheet there), so a test can fail when the clamp is
 * dropped; a class name asserted against itself passes for a clamp that is
 * present but inert (see `session-view.overflow.test.tsx`).
 *
 * MEASURING. `scrollHeight` is the whole text's height clamped or not, so
 * "is it very long" has the same answer folded, open or unfolded: no latch is
 * needed and an opened message keeps its "Show less" at any width. It is re-read
 * on resize (rotation, large text, a late font).
 *
 * THE SCROLL ANCHOR. The control announces itself with `MESSAGE_PRESSED_EVENT`
 * before the message changes height and with `CONTENT_RESIZED_EVENT` after, so
 * the transcript can hold something still through the toggle. WHICH thing depends
 * on the direction (the Transcript's listener decides): opening holds the
 * message's own top, so the new lines fall below what the reader already sees;
 * closing holds the control, so it lands under the thumb.
 */
import { useId, useLayoutEffect, useRef, useState, type CSSProperties } from "react";
import { cn } from "../lib/cn";
import { CONTENT_RESIZED_EVENT } from "../lib/content-resized";
import { MESSAGE_FOLD_MIN_LINES, MESSAGE_PREVIEW_LINES, MESSAGE_PRESSED_EVENT } from "../lib/message-fold";
import { messageRuntime } from "../store";

const CLAMP: CSSProperties = {
	display: "-webkit-box",
	WebkitBoxOrient: "vertical",
	WebkitLineClamp: MESSAGE_PREVIEW_LINES,
	overflow: "hidden",
};

export function UserText({ entryId, text, folded }: { entryId: string; text: string; folded: boolean }) {
	const textId = useId();
	const box = useRef<HTMLDivElement>(null);
	const [open, setOpen] = useState(false);
	const [long, setLong] = useState(false);
	/* Fold only a VERY long message (`MESSAGE_FOLD_MIN_LINES` wrapped lines). */
	const foldable = folded && long;
	const clamped = foldable && !open;

	/* Read how many wrapped lines the whole text takes. `scrollHeight` is the full
	   content height whether or not the box is clamped, so this answer is the same
	   folded, open or unfolded: no latch is needed, and an opened message never
	   loses its "Show less" because the text now happens to fit. The first measure
	   is in a layout effect, so a long message is clamped before it first paints. */
	useLayoutEffect(() => {
		const el = box.current;
		if (!el) return;
		const measure = () => {
			const parsed = parseFloat(getComputedStyle(el).lineHeight);
			const lineHeight = Number.isFinite(parsed) && parsed > 0 ? parsed : 20;
			setLong(el.scrollHeight >= lineHeight * MESSAGE_FOLD_MIN_LINES);
		};
		measure();
		/* A late web-font swap can add a wrapped line without resizing a clamped box,
		   so the observer below would not fire; re-read once fonts settle. */
		let live = true;
		void document.fonts?.ready.then(() => { if (live) measure(); });
		if (typeof ResizeObserver === "undefined") return () => { live = false; };
		const observer = new ResizeObserver(measure);
		observer.observe(el);
		return () => { live = false; observer.disconnect(); };
	}, [text]);

	/* The message changed height after it painted: say so, so the transcript can
	   hold the pressed control. Not on the first paint, which is not a change. */
	const painted = useRef(false);
	useLayoutEffect(() => {
		if (!painted.current) {
			painted.current = true;
			return;
		}
		box.current?.dispatchEvent(new CustomEvent(CONTENT_RESIZED_EVENT, { bubbles: true }));
	}, [open]);

	return (
		<>
			<div
				ref={box}
				id={textId}
				data-user-text=""
				data-message-id={entryId}
				style={clamped ? CLAMP : undefined}
				className="text-body leading-normal break-words whitespace-pre-wrap"
			>
				{text}
			</div>
			{foldable ? (
				<button
					type="button"
					data-message-toggle={entryId}
					aria-expanded={open}
					aria-controls={textId}
					aria-label={messageRuntime.t(open ? "relay.transcript.message.show_less_label" : "relay.transcript.message.show_more_label")}
					onClick={(event) => {
						/* Before the state flips, so the transcript reads the control's top
						   while it is still where the finger is. */
						event.currentTarget.dispatchEvent(
							new CustomEvent(MESSAGE_PRESSED_EVENT, { bubbles: true, detail: { id: entryId } }),
						);
						setOpen((value) => !value);
					}}
					className={cn(
						"lo-hit -my-0.5 flex h-8 shrink-0 items-center gap-1 self-end rounded-sm px-1",
						"text-meta text-ink-muted select-none active:bg-elevated",
					)}
				>
					<span>{messageRuntime.t(open ? "relay.transcript.message.show_less" : "relay.transcript.message.show_more")}</span>
					{/* Down opens, up closes: the direction the text moves. (The shared
					    `Chevron` is a right/down disclosure caret, which says "expand a
					    section", not "show the rest of this text".) */}
					<svg
						aria-hidden
						data-direction={open ? "up" : "down"}
						viewBox="0 0 12 12"
						width="12"
						height="12"
						fill="none"
						stroke="currentColor"
						strokeWidth="1.5"
						strokeLinecap="round"
						strokeLinejoin="round"
						className="shrink-0"
					>
						<path d={open ? "M2.5 7.5 6 4l3.5 3.5" : "M2.5 4.5 6 8l3.5-3.5"} />
					</svg>
				</button>
			) : null}
		</>
	);
}

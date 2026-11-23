import { useScrollCues } from "../../lib/use-scroll-cues";
/**
 * Sheet — the bottom-anchored overlay every picker on the phone uses
 * (model sheet, effort rungs, slash sheet, subagent detail). Bottom sheets
 * sit under the thumb; a centred dialog would not.
 *
 * The panel takes the elevated ground and the overlay shadow — the one
 * shadow in the system, reserved for objects that leave the flow. The scrim
 * click dismisses; the title/grabber also dismisses by a downward drag.
 *
 * A sheet can be summoned by a gesture that is STILL HELD when it mounts (the
 * list's long-press pin). The platform then synthesises the release click at
 * lift-off, hit-tests whatever the sheet now paints at those coordinates, and
 * a scrim or ✕ receives it — dismissing the sheet the user never touched. The
 * guard below swallows exactly that one click; every other path is unchanged.
 */
import { useCallback, useEffect, useId, useRef, useState, type ReactNode, type RefObject } from "react";
import { cn } from "../../lib/cn";
import { useModalLayer } from "./use-modal-layer";
import {
	SHEET_FOOTER_FRACTION,
	SHEET_READING_MIN_FRACTION,
	columnBox,
	columnCap,
	columnFloor,
} from "../../lib/column";

export function Sheet({
	open,
	onClose,
	onEscape,
	title,
	children,
	returnFocusRef,
	initialFocusRef,
	headerAction,
	headerDetail,
	footer,
	scrollerRef,
	closeFocusSignal,
}: {
	open: boolean;
	onClose: () => void;
	/** Consulted on Escape before `onClose`; true means "handled, stay open".
	    Only the usage sheet passes it, to dismiss its inline panel first. The ✕,
	    the scrim and the drag never consult it (U11). */
	onEscape?: () => boolean;
	title?: string;
	/** One control rendered immediately before ✕ (the usage sheet's refresh).
	    Lives in the header, outside the scroller, so it stays reachable while
	    the content scrolls. Absent on every other sheet, which render as before. */
	headerAction?: ReactNode;
	/** One line under the title row, also outside the scroller: the usage
	    sheet's refresh status, which must stay visible beside the button that
	    caused it however far the content is scrolled (UX round 1, U2). A
	    sticky line inside the scroller would sit under `lo-fade-y`'s top mask.
	    Absent on every other sheet. */
	headerDetail?: ReactNode;
	/** A PINNED BOTTOM BAR, outside the scroller: the control that answers the
	    sheet's content may not scroll away with it (the asks sheet's send and
	    decline). It sits after the scroller inside the panel, so it clears the
	    home indicator by the panel's own safe-area padding, and the sheet adds
	    no padding around it: the caller owns the bar's look. The sheet draws the
	    top hairline on its own non-scrolling wrapper and hides the wrapper while
	    the caller's bar element is empty, so an idle bar leaves no stripe. Absent on every other sheet, which render
	    as before. */
	footer?: ReactNode;
	/** The scrolling region, for a caller that has to move it (the asks sheet
	    puts the open card back at the top when it moves to the next ask). */
	scrollerRef?: RefObject<HTMLDivElement | null>;
	/** Bump it to put keyboard focus on the ✕: for a caller whose focused control
	    just left (the asks sheet's last answer), so the reader's place is not
	    dropped on the page behind. 0/undefined does nothing. */
	closeFocusSignal?: number;
	children: ReactNode;
	returnFocusRef?: RefObject<HTMLElement | null>;
	/** Where focus lands when the sheet opens. Defaults to the ✕; the slash
	    sheet passes its filter so a sheet opened by typing `/…` keeps
	    receiving type-ahead instead of stranding it on a button (U8). */
	initialFocusRef?: RefObject<HTMLElement | null>;
}) {
	const titleId = useId();
	const scrollCue = useScrollCues();
	/* ONE REF CALLBACK FOR TWO JOBS. `scrollCue` returns a cleanup that React 19
	   runs instead of calling the ref with null, so the merge must forward it. */
	const scrollRef = useCallback(
		(node: HTMLDivElement | null) => {
			const cleanup = scrollCue(node) as unknown;
			if (scrollerRef) scrollerRef.current = node;
			return () => {
				if (typeof cleanup === "function") cleanup();
				if (scrollerRef) scrollerRef.current = null;
			};
		},
		[scrollCue, scrollerRef],
	);
	const [dragY, setDragY] = useState(0);
	const panelRef = useRef<HTMLDivElement>(null);
	const drag = useRef<{ y: number; at: number; pointer: number } | null>(null);
	/* Only the title/grabber drags, never the content scroller. A tap still
	   closes through ✕; horizontal travel is irrelevant and scrolling content
	   cannot accidentally dismiss a long answer. The distance threshold scales
	   with the actual viewport-capped panel, not the layout viewport. */
	const dragHandlers = {
		onPointerDown: (event: React.PointerEvent<HTMLDivElement>) => {
			if ((event.target as HTMLElement).closest("button")) return;
			drag.current = { y: event.clientY, at: performance.now(), pointer: event.pointerId };
			event.currentTarget.setPointerCapture?.(event.pointerId);
		},
		onPointerMove: (event: React.PointerEvent<HTMLDivElement>) => {
			if (drag.current?.pointer === event.pointerId) setDragY(Math.max(0, event.clientY - drag.current.y));
		},
		onPointerUp: (event: React.PointerEvent<HTMLDivElement>) => {
			const start = drag.current;
			drag.current = null;
			const distance = start ? Math.max(0, event.clientY - start.y) : 0;
			const speed = start ? distance / Math.max(1, performance.now() - start.at) : 0;
			setDragY(0);
			if (distance > (panelRef.current?.clientHeight ?? Infinity) * 0.3 || (distance > 16 && speed > 0.5)) onClose();
		},
		onPointerCancel: () => { drag.current = null; setDragY(0); },
	};
	const { dialogRef, closeRef, focusEdge } = useModalLayer({ open, onClose, onEscape, returnFocusRef, initialFocusRef });
	useEffect(() => {
		if (closeFocusSignal) closeRef.current?.focus();
		// `closeRef` is a stable ref object.
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [closeFocusSignal]);

	if (!open) return null;
	/* VIEWPORT-ANCHORED, AND THAT IS THE WHOLE POINT OF THIS ELEMENT.

	   `absolute inset-0` resolves against the nearest POSITIONED ancestor, and
	   that is not always the column. The session view wraps its header in a
	   `relative` div (so the session-health ladder can hang off that div's
	   bottom, see `screens/session-view.tsx`), and the approvals sheet is
	   mounted INSIDE that header. Measured on the real bundle at 390x844: the
	   dialog's box was 390x53 — the header — so the bottom-anchored panel landed
	   at top=-263/bottom=53: readable only as a 53px sliver at the top of the
	   screen, cut mid-sentence, with its buttons and its ✕ 263px above the top
	   of a screen that cannot scroll up to them.

	   `columnBox()` supplies the anchor: `position: fixed`, and the two numbers
	   the column is itself pinned to (`--lo-vvh`, `--lo-vvh-top`, published by
	   the same `visualViewport` handler in `screens/session-view.tsx`). The
	   position sits in that helper rather than in this class list on purpose —
	   an inline declaration is the only one the happy-dom layer can assert, and
	   a revert to `absolute` is exactly what has to fail in CI.

	   Two constraints are deliberately kept:

	   1. NO PORTAL. The overlay still renders in place, inside the phone column:
	      the cmux screenshot surface is that column, and a body portal paints
	      outside it. Anchoring it to the viewport does not move the node; and the
	      box is the column's own — same width, same centre
	      (`--lo-column-max`, the var the column itself is capped by) — so the
	      captured surface contains the sheet.
	   2. THE VISUAL VIEWPORT, NOT THE LAYOUT ONE. A `fixed` box resolves against
	      the LAYOUT viewport, which a virtual keyboard does not shrink
	      (`resizes-visual`), so a bare `inset-0` would hold the panel's foot
	      under the keyboard — while the column it belongs with has already been
	      pinned to the visual one.

	   Safe areas: the panel keeps the bottom inset as padding (it is the panel's
	   foot that meets the home indicator). The top needs no term — the panel is
	   bottom-anchored and capped at 85% of the visual viewport, so 15% of that
	   viewport always stands above it, more than any phone's status bar. */
	return (
		<div
			ref={dialogRef}
			className="inset-x-0 z-50 mx-auto w-full max-w-[var(--lo-column-max,28rem)]"
			style={columnBox()}
			role="dialog"
			aria-modal="true"
			aria-labelledby={title ? titleId : undefined}
		>
			{/* Focus guards enforce containment even when a native WebKit Tab move
			    occurs before JavaScript receives a keyboard event. */}
			<span data-focus-guard tabIndex={0} aria-hidden onFocus={() => setTimeout(() => focusEdge(true), 0)} />
			{/* TAP-TO-DISMISS, NOT A SECOND CLOSE CONTROL (UX round 1, U7). The
			    scrim is a full-viewport button, and naming it "close" gave
			    assistive tech a control that collides with the panel's own ✕
			    ("close sheet") -- two ways to say one thing, one of them
			    invisible. It stays a button (the click is the affordance for a
			    pointer) and keeps its `tabIndex={-1}` opt-out of the focus trap,
			    but it leaves the ACCESSIBILITY tree entirely: a reader who
			    cannot see the backdrop has nothing to do with it. */}
			<button
				type="button"
				tabIndex={-1}
				aria-hidden
				/* The one handle a test CAN hold on to now that the scrim is out
				   of the a11y tree -- and the same idiom the pending card already
				   uses for an element selected for behaviour rather than meaning
				   (``data-testid="pending-card"``). */
				data-testid="sheet-scrim"
				className="lo-scrim absolute inset-0 bg-scrim"
				onClick={onClose}
			/>
			<div
				ref={panelRef}
				className={cn(
					"lo-sheet-panel absolute inset-x-0 bottom-0",
					"flex flex-col rounded-t-lg bg-elevated shadow-overlay",
					/* The panel clears the home indicator; content sits above it. */
					"pb-[env(safe-area-inset-bottom)]",
				)}
				/* COLUMN UNITS, never `85dvh` (`lib/column.ts` explains why): `dvh`
				   does not follow the keyboard pin, and this cap used to be the one
				   bounded region still written in it. Measured with the column pinned
				   to a 480px visual viewport (the state the pin exists for, a 390x844
				   phone with the keyboard up): the asks sheet's panel rendered 717px
				   tall — 85dvh of an 844px layout viewport — so 237px of it sat above
				   the top of the screen with the ✕ in that lost part. 0.85 is the
				   fraction it always was; only the unit changes. */
				style={{ ...columnCap(0.85), transform: dragY ? `translateY(${dragY}px)` : undefined, transition: drag.current ? "none" : "transform 180ms var(--ease-out-quart)" }}
			>
				<div {...dragHandlers} className="shrink-0 touch-none py-2"><div aria-hidden className="lo-grabber mx-auto h-1 w-9 rounded-full" /></div>
				{title ? (
					<div {...dragHandlers} className="flex touch-none items-center justify-between px-3 pb-1">
						<span id={titleId} className="line-clamp-2 min-w-0 text-start text-heading font-semibold text-ink">
							{title}
						</span>
						{headerAction ? <div className="ms-auto flex items-center">{headerAction}</div> : null}
						<button
							ref={closeRef}
							type="button"
							onClick={onClose}
							className="flex min-h-11 min-w-11 items-center justify-center rounded-sm text-ink-muted active:bg-surface"
							aria-label="close sheet"
						>
							✕
						</button>
					</div>
				) : null}
				{headerDetail ? <div className="shrink-0">{headerDetail}</div> : null}
				{/* With a footer, the reader keeps a floor of the column to read the question
				    in: the footer is capped below and scrolls inside itself, so large text
				    or open reasons cannot squeeze the reading area to nothing. */}
				<div
					ref={scrollRef}
					className="lo-scroll lo-fade-y min-h-0 flex-1 overflow-y-auto"
					style={footer ? columnFloor(SHEET_READING_MIN_FRACTION) : undefined}
				>
					{children}
				</div>
				{footer ? (
					<div
						data-sheet-footer
						/* `shrink-0`, NOT `shrink` (QA Q-3): the scroller's `flex-1` resolves to
						   its content height while the panel's height is indefinite, so the
						   overflow was shared by shrink factor x content size and a shrinkable
						   footer lost height to the scroller even at default text. The cap
						   below (`max-height`) and `overflow-y-auto` still bound it.

						   THE RULE LIVES HERE, not on the bar inside (design D9): this box does
						   not scroll, so its top edge stays put while the bar scrolls under
						   it. A caller whose bar is empty collapses the whole footer
						   (`has-[>:empty]`), so an idle bar leaves no stripe. */
						className="min-h-0 shrink-0 overflow-y-auto overscroll-contain border-t border-hairline has-[>:empty]:hidden"
						style={columnCap(SHEET_FOOTER_FRACTION)}
					>
						{footer}
					</div>
				) : null}
			</div>
			<span data-focus-guard tabIndex={0} aria-hidden onFocus={() => setTimeout(() => focusEdge(false), 0)} />
		</div>
	);
}

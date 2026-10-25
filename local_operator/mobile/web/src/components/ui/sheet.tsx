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
import { useId, useRef, useState, type ReactNode, type RefObject } from "react";
import { cn } from "../../lib/cn";
import { useModalLayer } from "./use-modal-layer";
import { columnBox, columnCap } from "../../lib/column";

export function Sheet({
	open,
	onClose,
	title,
	children,
	returnFocusRef,
	initialFocusRef,
}: {
	open: boolean;
	onClose: () => void;
	title?: string;
	children: ReactNode;
	returnFocusRef?: RefObject<HTMLElement | null>;
	/** Where focus lands when the sheet opens. Defaults to the ✕; the slash
	    sheet passes its filter so a sheet opened by typing `/…` keeps
	    receiving type-ahead instead of stranding it on a button (U8). */
	initialFocusRef?: RefObject<HTMLElement | null>;
}) {
	const titleId = useId();
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
	const { dialogRef, closeRef, focusEdge } = useModalLayer({ open, onClose, returnFocusRef, initialFocusRef });

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
						<span id={titleId} className="text-heading font-semibold text-ink">
							{title}
						</span>
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
				<div className="lo-scroll lo-fade-y min-h-0 flex-1 overflow-y-auto">
					{children}
				</div>
			</div>
			<span data-focus-guard tabIndex={0} aria-hidden onFocus={() => setTimeout(() => focusEdge(false), 0)} />
		</div>
	);
}

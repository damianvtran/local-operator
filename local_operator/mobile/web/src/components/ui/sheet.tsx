/**
 * Sheet — the bottom-anchored overlay every picker on the phone uses
 * (model sheet, effort rungs, slash sheet, subagent detail). Bottom sheets
 * sit under the thumb; a centred dialog would not.
 *
 * The panel takes the elevated ground and the overlay shadow — the one
 * shadow in the system, reserved for objects that leave the flow. The scrim
 * click dismisses; there is no drag gesture in v1.
 *
 * A sheet can be summoned by a gesture that is STILL HELD when it mounts (the
 * list's long-press pin). The platform then synthesises the release click at
 * lift-off, hit-tests whatever the sheet now paints at those coordinates, and
 * a scrim or ✕ receives it — dismissing the sheet the user never touched. The
 * guard below swallows exactly that one click; every other path is unchanged.
 */
import { useEffect, useId, useRef, type ReactNode, type RefObject } from "react";
import { cn } from "../../lib/cn";
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
	const dialogRef = useRef<HTMLDivElement>(null);
	const closeRef = useRef<HTMLButtonElement>(null);
	const openerRef = useRef<HTMLElement | null>(null);
	const onCloseRef = useRef(onClose);
	const titleId = useId();
	onCloseRef.current = onClose;

	useEffect(() => {
		if (!open) return;
		const dialog = dialogRef.current;
		if (!dialog) return;
		openerRef.current = returnFocusRef?.current ?? (document.activeElement instanceof HTMLElement
			? document.activeElement
			: null);
		const background = Array.from(dialog.parentElement?.children ?? [])
			.filter((node): node is HTMLElement => node instanceof HTMLElement && node !== dialog);
		const previous = background.map((node) => ({
			node,
			inert: node.hasAttribute("inert"),
			ariaHidden: node.getAttribute("aria-hidden"),
		}));
		/* aria-modal alone does not remove the covered application from keyboard
		   or accessibility navigation. Both signals keep the in-flow sheet modal
		   without moving it into a body portal outside the phone column. */
		for (const node of background) {
			node.setAttribute("inert", "");
			node.setAttribute("aria-hidden", "true");
		}
		/* The initial focus target: the ✕ by default, or a caller's control —
		   the slash sheet's filter, so a sheet opened by typing keeps receiving
		   its type-ahead (U8, mobile UX batch 1). */
		(initialFocusRef?.current ?? closeRef.current)?.focus();
		const onKey = (event: KeyboardEvent) => {
			if (event.key === "Escape") {
				event.preventDefault();
				onCloseRef.current();
				return;
			}
			/* cmux/WKWebView reports a chord as key="Shift+Tab" rather than
			   key="Tab" + shiftKey, while browsers with a hardware keyboard use
			   the standard form. Supporting both keeps the same modal contract. */
			const reverse = event.shiftKey || event.key === "Shift+Tab";
			if (event.key !== "Tab" && !reverse) return;
			const focusable = Array.from(
				dialog.querySelectorAll<HTMLElement>(
					'button:not([disabled]):not([tabindex="-1"]), input:not([disabled]):not([tabindex="-1"]), [href]:not([tabindex="-1"]), [tabindex]:not([tabindex="-1"]):not([data-focus-guard])',
				),
			).filter((node) => !node.hasAttribute("inert"));
			if (focusable.length === 0) return;
			const first = focusable[0];
			const last = focusable[focusable.length - 1];
			if (reverse && document.activeElement === first) {
				event.preventDefault();
				/* WKWebView applies its native Tab move after key dispatch even when
				   prevented. Deferring restoration wins that ordering deterministically. */
				setTimeout(() => last.focus(), 0);
			} else if (!reverse && document.activeElement === last) {
				event.preventDefault();
				setTimeout(() => first.focus(), 0);
			}
		};
		/* Capture on the window because WKWebView can move focus out of an
		   in-flow dialog before a React bubble handler sees hardware Tab. */
		window.addEventListener("keydown", onKey, true);

		return () => {
			window.removeEventListener("keydown", onKey, true);
			for (const state of previous) {
				if (!state.inert) state.node.removeAttribute("inert");
				if (state.ariaHidden == null) state.node.removeAttribute("aria-hidden");
				else state.node.setAttribute("aria-hidden", state.ariaHidden);
			}
			/* Row navigation replaces the route and may remove the opener; only a
			   still-connected control is a valid restoration target. */
			if (openerRef.current?.isConnected) openerRef.current.focus();
		};
	}, [open, returnFocusRef, initialFocusRef]);

	/* THE RELEASE CLICK OF THE PRESS THAT OPENED THE SHEET (mobile UX batch 1,
	   U1) is swallowed; every other click passes. Measured in Chromium against
	   the long-press pin: `pointerdown` lands on the row, the sheet mounts
	   ~450ms later, and the finger's release dispatches `pointerup` (implicit
	   touch capture targets it at the ROW outside this dialog, which is why the
	   listeners below are on the window in the capture phase) followed ~1ms
	   later by a `click` whose target is the sheet's own scrim — which ran
	   `onClose` and dismissed the sheet. iOS Safari fires the same release
	   click; Android may suppress it.

	   THE DISCRIMINATOR IS THE GESTURE, NOT A CLOCK, AND IT IS PER-POINTER
	   (mobile UX batch 2). The click belongs to the opening press iff the
	   pointer that RELEASED was never seen pressing since mount — the opening
	   gesture's own pointerdown predates the sheet, because it is what started
	   the hold. The first revision tracked one document-wide boolean, so with
	   two fingers the interleave collapsed: A long-presses (sheet opens, A
	   still held), B taps the scrim, A releases first — A's `pointerup` cleared
	   the flag, B's then read "no down since mount" and armed, and B's genuine
	   tap was eaten (agent review round 1, MINOR 1). The `seenDown` set makes
	   the LAST release decide: `armed = !seenDown.has(pointerId)` at each
	   `pointerup`, so a seen pointer's release disarms just as naturally as an
	   unseen one arms. A fresh press answers scrim/✕ as always (its pointerdown
	   disarms), a keyboard activation passes (a keydown precedes Enter/Space
	   activation and disarms), and a platform that synthesises no click leaves
	   nothing armed past the next press. This also covers the ✕ and any action
	   row a release click could land on, not just the scrim the defect was
	   measured on. */
	useEffect(() => {
		if (!open) return;
		/* Pointer ids, "seen down since mount". The opening gesture's down
		   happened before this effect existed, so its id is absent — the ghost.
		   A cancelled pointer's id stays: no click can follow a cancel, and a
		   reused id will have pressed since mount anyway. */
		const seenDown = new Set<number>();
		let armed = false;
		const onPointerDown = (event: PointerEvent) => {
			seenDown.add(event.pointerId);
			armed = false;
		};
		const onPointerUp = (event: PointerEvent) => {
			/* The LAST release decides — an unseen id arms the ghost click, a
			   seen id disarms whatever an earlier release armed. */
			armed = !seenDown.has(event.pointerId);
		};
		const onKeyDown = () => {
			armed = false;
		};
		const onClick = (event: MouseEvent) => {
			if (!armed) return;
			armed = false;
			event.stopPropagation();
			event.preventDefault();
		};
		window.addEventListener("pointerdown", onPointerDown, true);
		window.addEventListener("pointerup", onPointerUp, true);
		window.addEventListener("keydown", onKeyDown, true);
		window.addEventListener("click", onClick, true);
		return () => {
			window.removeEventListener("pointerdown", onPointerDown, true);
			window.removeEventListener("pointerup", onPointerUp, true);
			window.removeEventListener("keydown", onKeyDown, true);
			window.removeEventListener("click", onClick, true);
		};
	}, [open]);

	const focusEdge = (last: boolean) => {
		const focusable = Array.from(
			dialogRef.current?.querySelectorAll<HTMLElement>(
				'button:not([disabled]):not([tabindex="-1"]), input:not([disabled]):not([tabindex="-1"]), [href]:not([tabindex="-1"]), [tabindex]:not([tabindex="-1"]):not([data-focus-guard])',
			) ?? [],
		).filter((node) => !node.hasAttribute("inert"));
		focusable[last ? focusable.length - 1 : 0]?.focus();
	};

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
				className={cn(
					"lo-sheet-panel absolute right-0 bottom-0 left-0",
					"flex flex-col rounded-t-lg border-t border-control bg-elevated shadow-overlay",
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
				style={columnCap(0.85)}
			>
				{title ? (
					<div className="flex items-center justify-between px-3 pt-2 pb-1">
						<span id={titleId} className="text-meta font-medium tracking-[0.08em] text-ink-muted">
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
				<div className="lo-scroll min-h-0 flex-1 overflow-y-auto">
					{children}
				</div>
			</div>
			<span data-focus-guard tabIndex={0} aria-hidden onFocus={() => setTimeout(() => focusEdge(false), 0)} />
		</div>
	);
}

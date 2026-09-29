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

	   THE DISCRIMINATOR IS THE GESTURE, NOT A CLOCK. The click belongs to the
	   opening press iff NO pointerdown has been seen since mount — the opening
	   gesture's own pointerdown predates the sheet, because it is what started
	   the hold. So: arm on a pointerup with no since-mount pointerdown, and
	   swallow the single click that follows it. A fresh press answers scrim/✕
	   as always (its pointerdown disarms), a keyboard activation passes (a
	   keydown precedes Enter/Space activation and disarms), and a platform that
	   synthesises no click leaves nothing armed past the next press. This also
	   covers the ✕ and any action row a release click could land on, not just
	   the scrim the defect was measured on. */
	useEffect(() => {
		if (!open) return;
		let downSinceMount = false;
		let armed = false;
		const onPointerDown = () => {
			downSinceMount = true;
			armed = false;
		};
		const onPointerUp = () => {
			if (!downSinceMount) armed = true;
			downSinceMount = false;
		};
		const onPointerCancel = () => {
			downSinceMount = false;
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
		window.addEventListener("pointercancel", onPointerCancel, true);
		window.addEventListener("keydown", onKeyDown, true);
		window.addEventListener("click", onClick, true);
		return () => {
			window.removeEventListener("pointerdown", onPointerDown, true);
			window.removeEventListener("pointerup", onPointerUp, true);
			window.removeEventListener("pointercancel", onPointerCancel, true);
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
	/* In-flow overlay, not a portal: the cmux screenshot surface is the
	   phone column, and a body portal paints outside it. The session
	   column is `relative` and no longer uses transform, so `absolute
	   inset-0` covers exactly the column. */
	return (
		<div
			ref={dialogRef}
			className="absolute inset-0 z-50"
			role="dialog"
			aria-modal="true"
			aria-labelledby={title ? titleId : undefined}
		>
			{/* Focus guards enforce containment even when a native WebKit Tab move
			    occurs before JavaScript receives a keyboard event. */}
			<span data-focus-guard tabIndex={0} aria-hidden onFocus={() => setTimeout(() => focusEdge(true), 0)} />
			<button
				type="button"
				tabIndex={-1}
				aria-label="close"
				className="lo-scrim absolute inset-0 bg-scrim"
				onClick={onClose}
			/>
			<div
				className={cn(
					"lo-sheet-panel absolute right-0 bottom-0 left-0 max-h-[85dvh]",
					"flex flex-col rounded-t-lg border-t border-control bg-elevated shadow-overlay",
					/* The panel clears the home indicator; content sits above it. */
					"pb-[env(safe-area-inset-bottom)]",
				)}
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

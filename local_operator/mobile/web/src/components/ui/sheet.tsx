/**
 * Sheet — the bottom-anchored overlay every picker on the phone uses
 * (model sheet, effort rungs, slash sheet, subagent detail). Bottom sheets
 * sit under the thumb; a centred dialog would not.
 *
 * The panel takes the elevated ground and the overlay shadow — the one
 * shadow in the system, reserved for objects that leave the flow. The scrim
 * click dismisses; there is no drag gesture in v1.
 */
import { useEffect, useId, useRef, type ReactNode, type RefObject } from "react";
import { cn } from "../../lib/cn";

export function Sheet({
	open,
	onClose,
	title,
	children,
	returnFocusRef,
}: {
	open: boolean;
	onClose: () => void;
	title?: string;
	children: ReactNode;
	returnFocusRef?: RefObject<HTMLElement | null>;
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
		closeRef.current?.focus();
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
	}, [open, returnFocusRef]);

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

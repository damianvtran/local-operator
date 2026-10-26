/** One in-flow modal contract for drawers and sheets. Keeping inert background,
 * focus return/trap and the opening-pointer release guard together prevents a
 * destination drawer from behaving differently from an object action sheet. */
import { useEffect, useRef, type RefObject } from "react";

export function useModalLayer({ open, onClose, returnFocusRef, initialFocusRef }: {
    open: boolean;
    onClose: () => void;
    returnFocusRef?: RefObject<HTMLElement | null>;
    initialFocusRef?: RefObject<HTMLElement | null>;
}) {
	const dialogRef = useRef<HTMLDivElement>(null);
	const closeRef = useRef<HTMLButtonElement>(null);
	const openerRef = useRef<HTMLElement | null>(null);
	const onCloseRef = useRef(onClose);
	onCloseRef.current = onClose;

	useEffect(() => {
		if (!open) return;
		const dialog = dialogRef.current;
		if (!dialog) return;
		openerRef.current = returnFocusRef?.current ?? (document.activeElement instanceof HTMLElement
			? document.activeElement
			: null);
		/* A gate sheet may be nested in the header while a drawer is a direct
		   column child. Walk siblings at every ancestor level so BOTH shapes
		   inert the whole background, without a portal or inerting the modal's
		   own ancestors (which would disable the dialog too). */
		const background: HTMLElement[] = [];
		let branch: HTMLElement = dialog;
		while (branch.parentElement) {
			const parent = branch.parentElement;
			for (const node of Array.from(parent.children)) {
				if (node instanceof HTMLElement && node !== branch) background.push(node);
			}
			if (parent === document.body) break;
			branch = parent;
		}
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

	return { dialogRef, closeRef, focusEdge };
}

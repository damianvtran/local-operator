/**
 * Disclosure — the one expand/collapse idiom app-wide (branding §7: two
 * competing patterns is a bug). The chevron SWAPS between right and down
 * glyphs; it never rotates, because a rotating chevron animates a pixel
 * shape that was designed to point one way.
 *
 * `forceClosed` is part of this one idiom on purpose. The session column needs
 * its panels shut while a request is pending, and the alternative — a second
 * collapse mechanism beside this one, or lifting every panel's open state into
 * the screen — is exactly the competing-patterns bug above.
 */
import { useState, type ReactNode } from "react";
import { cn } from "../../lib/cn";

/**
 * The dim that says "held shut, not broken" — applied per PART, never to the
 * header button as a whole.
 *
 * It began on the button and cannot stay there. `opacity` composites the whole
 * subtree and a descendant cannot undo it, so a blanket dim also dims the
 * roster's failure count: measured from the painted frame at 3.30:1 against the
 * card's background where the undimmed count is 7.08:1 (design D4). A failed
 * fan-out matters most in exactly the state that dimmed it — the user is being
 * asked for a decision. WCAG's inactive-control exemption does cover a
 * `disabled` header, so this is not a conformance defect; it is the dim taxing
 * the one glyph U5 exists to surface.
 *
 * Each header therefore opts its own parts in, and anything that must stay
 * legible while held simply does not carry it. That is the only thing that
 * works given how opacity composites.
 */
export const HELD_DIM = "opacity-60";

export function Chevron({ open, className }: { open: boolean; className?: string }) {
	return (
		<span
			aria-hidden
			className={cn("inline-block w-4 text-center text-ink-dim select-none", className)}
		>
			{open ? "▾" : "▸"}
		</span>
	);
}

export function Disclosure({
	header,
	children,
	defaultOpen = false,
	forceClosed = false,
	className,
	headerClassName,
}: {
	header: ReactNode;
	children: ReactNode;
	defaultOpen?: boolean;
	/** Hold the panel shut regardless of its own state. The one caller is the
	    session column while a request is pending (see session-view): a panel
	    that expands beside a decision card pushes the decision off a column
	    that cannot scroll. The panel's own state is preserved rather than
	    reset, so answering the question returns the user to what they opened.

	    The header it renders must stay readable while held — see HELD_DIM: the
	    dim is applied by each header to its own parts, not to this button. */
	forceClosed?: boolean;
	className?: string;
	headerClassName?: string;
}) {
	const [open, setOpen] = useState(defaultOpen);
	const shown = open && !forceClosed;
	return (
		<div className={className}>
			<button
				type="button"
				aria-expanded={shown}
				disabled={forceClosed}
				/* Dimmed-and-inert is the same vocabulary this product uses for
				   "busy", so a tap that does nothing reads as a fault rather than
				   as a rule (UX U8). The rule is stated in the header itself
				   rather than in a `title` alone, because a phone has no hover and
				   a long-press on a disabled control surfaces nothing. */
				title={
					forceClosed
						? "Held shut while a question is waiting — answer it to reopen this panel."
						: undefined
				}
				onClick={() => setOpen(!shown)}
				className={cn(
					/* The compact label stays unchanged while the hit box matches the
					   navigation controls users alternate with on a phone. */
					"flex min-h-11 w-full items-center gap-1 text-left select-none",
					headerClassName,
				)}
			>
				{/* The chevron stops inviting a tap that would do nothing. */}
				<Chevron open={shown} className={cn(forceClosed && HELD_DIM)} />
				{header}
				{forceClosed ? (
					/* Says why it is inert, in the row itself, at no vertical cost.
					   `shrink-0` so a long panel label truncates before this does —
					   the panel budget this state enforces is measured in rows, so
					   the explanation must not wrap the header onto a second one. */
					<span className="shrink-0 text-meta text-ink-dim">· answer first</span>
				) : null}
			</button>
			{shown ? children : null}
		</div>
	);
}

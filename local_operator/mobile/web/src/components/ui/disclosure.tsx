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
	    reset, so answering the question returns the user to what they opened. */
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
				onClick={() => setOpen(!shown)}
				className={cn(
					/* The compact label stays unchanged while the hit box matches the
					   navigation controls users alternate with on a phone. */
					"flex min-h-11 w-full items-center gap-1 text-left select-none",
					/* Held shut reads as inert rather than broken: the count stays
					   legible, the chevron stops inviting a tap that would do
					   nothing. */
					forceClosed && "opacity-60",
					headerClassName,
				)}
			>
				<Chevron open={shown} />
				{header}
			</button>
			{shown ? children : null}
		</div>
	);
}

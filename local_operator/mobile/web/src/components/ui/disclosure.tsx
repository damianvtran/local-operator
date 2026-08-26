/**
 * Disclosure — the one expand/collapse idiom app-wide (branding §7: two
 * competing patterns is a bug). The chevron SWAPS between right and down
 * glyphs; it never rotates, because a rotating chevron animates a pixel
 * shape that was designed to point one way.
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
	className,
	headerClassName,
}: {
	header: ReactNode;
	children: ReactNode;
	defaultOpen?: boolean;
	className?: string;
	headerClassName?: string;
}) {
	const [open, setOpen] = useState(defaultOpen);
	return (
		<div className={className}>
			<button
				type="button"
				aria-expanded={open}
				onClick={() => setOpen(!open)}
				className={cn(
					/* The compact label stays unchanged while the hit box matches the
					   navigation controls users alternate with on a phone. */
					"flex min-h-11 w-full items-center gap-1 text-left select-none",
					headerClassName,
				)}
			>
				<Chevron open={open} />
				{header}
			</button>
			{open ? children : null}
		</div>
	);
}

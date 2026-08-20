/**
 * Chip — the small tappable label (model chip, effort chip) in the session
 * header. 10px radius is panel scale; a control-scale 6 would disappear at
 * this size and full would read as a button, which it is not quite.
 */
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "../../lib/cn";

export function Chip({
	className,
	children,
	...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { children: ReactNode }) {
	return (
		<button
			type="button"
			className={cn(
				"inline-flex min-h-8 items-center gap-1 rounded-sm border border-control bg-surface px-2 text-mono-sm text-ink-muted select-none active:bg-elevated",
				className,
			)}
			{...rest}
		>
			{children}
		</button>
	);
}

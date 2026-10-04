/**
 * Chip — a small, tappable label carrying ONE piece of state.
 *
 * A plain button on the system's control ground, held to the same 44 px floor
 * as every other control on these surfaces (`min-h-11`) and rounded at the
 * CONTROL scale (`rounded-sm`) so it reads as something you press rather than
 * as a panel. (The note that used to stand here described usages that did not
 * exist and argued for a radius the code does not apply; the chip had no
 * consumer to contradict either. Design round 2, N4.)
 *
 * THE COMPOSER'S WORKING-DIRECTORY CHIP IS ITS FIRST CONSUMER
 * (`components/directory-sheet.tsx`). It lives here rather than inline there
 * because the alternative is a second chip-shaped control beside it, and the
 * two would then drift.
 *
 * Callers supply their own glyph as a child; this file takes no icon
 * dependency, which is why it imports nothing beyond React and `cn`.
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
				"inline-flex min-h-11 items-center gap-1 rounded-sm border border-control bg-surface px-2 text-mono-sm text-ink-muted select-none active:bg-elevated",
				className,
			)}
			{...rest}
		>
			{children}
		</button>
	);
}

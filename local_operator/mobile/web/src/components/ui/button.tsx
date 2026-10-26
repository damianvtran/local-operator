/**
 * Button — the one pressable primitive. 44px minimum height: the iOS tap
 * target floor. Variants are roles, never hues; `primary` spends the
 * accent, which the branding budget reserves for the main action on screen.
 */
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "../../lib/cn";

type Variant = "primary" | "tonal" | "ghost" | "icon" | "danger";

const VARIANTS: Record<Variant, string> = {
	/* The disabled treatment is COLOUR, never opacity — `styles/index.css`
	   pins `opacity: 1` for `:disabled` for exactly that reason. Without it
	   a disabled primary was pixel-identical to its live self (round-1 design,
	   D2), so a tap on an inert control read as a broken one; the pattern is
	   the composer's own (`components/composer.tsx`). */
	primary: "bg-accent text-on-accent active:bg-accent-active disabled:bg-sunken disabled:text-ink-disabled",
	tonal: "bg-surface text-ink active:bg-elevated disabled:text-ink-disabled",
	ghost: "text-ink-muted active:bg-surface disabled:text-ink-disabled",
	icon: "size-11 shrink-0 !px-0 text-ink-muted active:bg-surface disabled:text-ink-disabled",
	danger: "bg-danger-wash text-danger active:bg-elevated disabled:text-ink-disabled",
};

export function Button({
	variant = "tonal",
	className,
	children,
	...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
	variant?: Variant;
	children: ReactNode;
}) {
	return (
		<button
			type="button"
			className={cn(
				"inline-flex min-h-11 items-center justify-center gap-2 rounded-md px-4 text-body-sm font-medium transition-colors duration-fast select-none",
				VARIANTS[variant],
				className,
			)}
			{...rest}
		>
			{children}
		</button>
	);
}

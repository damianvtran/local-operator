/**
 * Sheet — the bottom-anchored overlay every picker on the phone uses
 * (model sheet, effort rungs, slash sheet, subagent detail). Bottom sheets
 * sit under the thumb; a centred dialog would not.
 *
 * The panel takes the elevated ground and the overlay shadow — the one
 * shadow in the system, reserved for objects that leave the flow. The scrim
 * click dismisses; there is no drag gesture in v1.
 */
import { useEffect, type ReactNode } from "react";
import { cn } from "../../lib/cn";

export function Sheet({
	open,
	onClose,
	title,
	children,
}: {
	open: boolean;
	onClose: () => void;
	title?: string;
	children: ReactNode;
}) {
	/* Escape dismisses for hardware keyboards; touch users tap the scrim. */
	useEffect(() => {
		if (!open) return;
		const onKey = (e: KeyboardEvent) => {
			if (e.key === "Escape") onClose();
		};
		window.addEventListener("keydown", onKey);
		return () => window.removeEventListener("keydown", onKey);
	}, [open, onClose]);

	if (!open) return null;
	/* In-flow overlay, not a portal: the cmux screenshot surface is the
	   phone column, and a body portal paints outside it. The session
	   column is `relative` and no longer uses transform, so `absolute
	   inset-0` covers exactly the column. */
	return (
		<div
			className="absolute inset-0 z-50"
			role="dialog"
			aria-modal="true"
		>
			<button
				type="button"
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
						<span className="text-meta font-medium tracking-[0.08em] text-ink-muted">
							{title}
						</span>
						<button
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
		</div>
	);
}

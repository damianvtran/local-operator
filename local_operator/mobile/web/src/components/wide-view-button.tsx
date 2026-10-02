/**
 * The wide-view toggle (issue #1870): the one control that enters and leaves the
 * wider reading layout, rendered wherever a reader can reach it.
 *
 * ONE COMPONENT, TWO PLACES, because the preference is one preference. It ships on
 * the conversation list's footer (beside the theme, the other per-phone
 * preference) and on the SESSION screen's header — the surface the issue is
 * actually about, where a reader who notices the narrow column is standing
 * (design round 1, D2). A second copy of this markup is how the two drift into
 * different labels or different pressed states.
 *
 * THE LABEL IS A WORD, not a glyph alone (design round 1, D3). `⇔` reads as
 * swap/pan and sits beside the theme's `◐`, so a first-time reader has nothing to
 * learn the meaning from; the word `wide` fits the same 44x44 target at 320, 360
 * and 390 with no overflow (measured). `aria-pressed` carries the state to
 * assistive tech, which the accessible name alone never did.
 *
 * PRESSED IS FILLED, NOT JUST HUE (design round 1, D4). The pressed and default
 * glyph colours sit within ~1.5:1 luminance in 28 of the 31 shipped palettes, so a
 * hue change alone is invisible to many readers; a filled block against an
 * outlined one differs in weight. `accent`-on-`on-accent` is asserted at >= 4.5:1
 * for every palette by `scripts/contrast-contract.mjs`.
 */
import { useState } from "react";
import { cn } from "../lib/cn";
import { applyWideView, getWideView } from "../lib/viewport";

export function WideViewButton({ className }: { className?: string }) {
	/* Seeded from the persisted preference on mount, so the control states the
	   truth in whichever screen opened it. The two instances never share a screen
	   (the list and the session are separate routes), so there is no second source
	   of truth to synchronise — `applyWideView` writes the one. */
	const [wide, setWide] = useState(getWideView);
	return (
		<button
			type="button"
			onClick={() => {
				applyWideView(!wide);
				setWide(!wide);
			}}
			aria-label="wide view"
			aria-pressed={wide}
			className={cn(
				"flex min-h-11 min-w-11 shrink-0 items-center justify-center rounded-md border text-meta select-none",
				wide
					? "border-accent bg-accent text-on-accent active:bg-accent-active"
					: "border-control bg-surface text-ink-muted active:bg-elevated",
				className,
			)}
		>
			wide
		</button>
	);
}

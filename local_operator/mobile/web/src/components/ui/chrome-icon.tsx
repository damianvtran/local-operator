/** Chrome glyphs have an explicit optical footprint, independent of the 44px
 * button. Font symbols vary by platform and leave a nominally large target
 * looking tiny. A 20px box leaves the paths larger than those text glyphs but
 * avoids oversized chrome; the shared 24-unit viewBox keeps optical proportions.
 *
 * `temporary` is lucide 0.507's `message-square-dashed`, inlined like the
 * other paths: the dashed twin of the "New chat" bubble, so "solid = kept,
 * dashed = not kept" reads as a pair on every surface. It is also drawn inline
 * beside text (the subtitle, a list row), hence the optional smaller `size`. */
export function ChromeIcon({ name, size = 20 }: {
	name: "menu" | "search" | "close" | "refresh" | "temporary";
	size?: number;
}) {
	return <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
		stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
		aria-hidden="true" focusable="false" className="pointer-events-none shrink-0">
		{name === "menu" ? <path d="M2 5h20M2 12h20M2 19h20" /> :
			name === "search" ? <><circle cx="10.25" cy="10.25" r="8.25" /><path d="m16.5 16.5 5.5 5.5" /></> :
				name === "refresh" ? <><path d="M20 11a8 8 0 1 0-2.34 5.66" /><path d="M20 4v7h-7" /></> :
				name === "temporary" ? <path d="M10 17H7l-4 4v-7M14 17h1M14 3h1M19 3a2 2 0 0 1 2 2M21 14v1a2 2 0 0 1-2 2M21 9v1M3 9v1M5 3a2 2 0 0 0-2 2M9 3h1" /> :
				<path d="m3 3 18 18M21 3 3 21" />}
	</svg>;
}

/** Chrome glyphs have an explicit optical footprint, independent of the 44px
 * button. Font symbols vary by platform and leave a nominally large target
 * looking tiny. A 20px box leaves the paths larger than those text glyphs but
 * avoids oversized chrome; the shared 24-unit viewBox keeps optical proportions. */
export function ChromeIcon({ name }: { name: "menu" | "search" | "close" | "refresh" }) {
	return <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
		stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
		aria-hidden="true" focusable="false" className="pointer-events-none shrink-0">
		{name === "menu" ? <path d="M2 5h20M2 12h20M2 19h20" /> :
			name === "search" ? <><circle cx="10.25" cy="10.25" r="8.25" /><path d="m16.5 16.5 5.5 5.5" /></> :
				name === "refresh" ? <><path d="M20 11a8 8 0 1 0-2.34 5.66" /><path d="M20 4v7h-7" /></> :
				<path d="m3 3 18 18M21 3 3 21" />}
	</svg>;
}

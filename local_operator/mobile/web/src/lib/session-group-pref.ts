/**
 * The local grouping preference (`lo-mobile-group-tags`).
 *
 * VIEW STATE, NOT CONTENT: grouping changes only how this browser paints the
 * list, so the key lives in `localStorage` beside the theme and wide-view
 * preferences and, like them, deliberately survives sign-out. Signing out must
 * not silently change how the home screen is arranged when the same reader
 * signs back in.
 *
 * DEFAULT OFF, and OFF REMOVES THE KEY rather than writing "0" — the shipped
 * default is the absence of the key, so a reset (clearing site data, or the
 * toggle turned off) returns this browser to exactly the released behaviour.
 */
const KEY = "lo-mobile-group-tags";

export function getGroupTags(): boolean {
	return localStorage.getItem(KEY) === "1";
}

export function setGroupTags(grouped: boolean): void {
	if (grouped) localStorage.setItem(KEY, "1");
	else localStorage.removeItem(KEY);
}

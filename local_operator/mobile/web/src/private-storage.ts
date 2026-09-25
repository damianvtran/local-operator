import { clearPendingEchoes } from "./pending-echo";

const PRIVATE_STORAGE_PREFIXES = ["lo-mobile-command:", "lo-mobile-draft:"];

/** Remove content-bearing state when authentication changes ownership.
 * Theme and other non-private preferences deliberately survive sign-out. */
export function clearPrivateSessionStorage(): void {
	for (let index = localStorage.length - 1; index >= 0; index--) {
		const key = localStorage.key(index);
		if (key && PRIVATE_STORAGE_PREFIXES.some((prefix) => key.startsWith(prefix))) {
			localStorage.removeItem(key);
		}
	}
	/* The pending echoes are content-bearing state too — they hold what the user
	   typed — and they live in MEMORY, so the prefix sweep above cannot reach
	   them. Both callers currently follow this with a page load (the 401 handler
	   and the login page's own inline sweep), which is what actually discards
	   module state today; dropping them here is what makes this function's
	   promise — one purge point for private content — true on its own rather
	   than one reload away from being true. The login page's own script cannot
	   do this half: it runs in a page that never loaded this bundle. */
	clearPendingEchoes();
}

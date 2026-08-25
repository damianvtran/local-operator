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
}

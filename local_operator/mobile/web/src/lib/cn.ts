/**
 * clsx-style class joiner, hand-rolled to keep the bundle free of deps.
 * Falsy values drop out; strings join with single spaces.
 */
export function cn(...parts: Array<string | false | null | undefined>): string {
	return parts.filter(Boolean).join(" ");
}

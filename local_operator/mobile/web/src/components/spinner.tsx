/**
 * A small loading wheel — the phone's "this session is working" mark beside
 * a title. Pure CSS (a border ring that rotates), so it needs no icon font:
 * the ⟳ / ☐ Unicode glyphs it replaces are the ones that rendered as tofu
 * boxes on phones whose system font lacks those codepoints. A spinning ring
 * is read as "loading" in any font.
 *
 * The braille tuple used inside a session (WorkingLine) is text and survives
 * every font; this ring is for the roster, where a compact glyph-free mark
 * sits cleaner next to a title.
 */
import { cn } from "../lib/cn";

export function Spinner({ className }: { className?: string }) {
	return (
		<span
			role="status"
			aria-label="working"
			className={cn(
				"lo-spinner inline-block size-3 shrink-0 rounded-full",
				className,
			)}
		/>
	);
}

/**
 * The STATIC tag line a conversation row carries — session list and past
 * sessions, one implementation for both.
 *
 * It deliberately has no controls. Assignment is a long-press action on the
 * list (`SessionTagsSheet`), and a row is a single button in its entirety:
 * putting a checkbox or a chip button inside it would nest an interactive
 * control in a button, which the platform does not hit-test sanely. The line is
 * a LABEL, not a seat for interaction.
 *
 * BUDGET. Two names, then a `+N` count (`lib/session-tags.ts` owns the rule and
 * the order). Long names truncate inside their share of the row rather than
 * pushing the count off it — the count is the only thing that says the rest
 * exist, so it must never be the part that is clipped.
 *
 * THE PRIMARY MARK IS A SHAPE PLUS A WORD. The leading dot is `aria-hidden`;
 * the word "primary" is `sr-only`, so a reader hears which label the dot is on
 * and nobody depends on colour alone.
 */
import { cn } from "../lib/cn";
import { compressedTags } from "../lib/session-tags";
import { messageRuntime } from "../store";
import type { TagRef } from "../types";

export function SessionTagLine({
	tags,
	primary,
	className,
}: {
	tags: readonly TagRef[];
	primary: string | null;
	className?: string;
}) {
	const { shown, overflow } = compressedTags(tags);
	/* Untagged is NO LINE, not an empty one: a placeholder would change the
	   row's height between a tagged and an untagged conversation, and the list
	   would visibly jitter on every repaint. */
	if (shown.length === 0) return null;
	return (
		<span
			data-testid="session-tags"
			className={cn(
				"flex min-w-0 items-center gap-2 overflow-hidden text-meta text-ink-dim",
				className,
			)}
		>
			{shown.map((tag) => (
				<span key={tag.id} className="flex min-w-0 items-center gap-1">
					{tag.id === primary ? (
						<>
							<span aria-hidden>•</span>
							<span className="sr-only">
								{messageRuntime.t("relay.session.tags.primary")}
							</span>
						</>
					) : null}
					<span className="truncate">{tag.name}</span>
				</span>
			))}
			{overflow > 0 ? (
				<span
					className="shrink-0"
					aria-label={messageRuntime.t("relay.session.tags.more", {
						count: overflow,
					})}
				>
					+{overflow}
				</span>
			) : null}
		</span>
	);
}

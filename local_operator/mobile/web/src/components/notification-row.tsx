/**
 * One inbox reel row on the phone (design §9): a 56 px two-line row whose
 * trailing 44×44 control is ALWAYS visible (touch has no hover): ✕ dismisses
 * an update in one tap, `›` opens the Questions sheet for a live ask (asks
 * keep one answering surface, decision "link, don't duplicate").
 *
 * The lead slot is the session list's reserved-indicator idea: one fixed box
 * that holds the unread dot, or the checkbox in selection mode, so the title
 * starts at the same x in every state and nothing reflows on a mode change.
 *
 * A row NEVER marks itself read. Rendering, scrolling past or focusing it is
 * not engagement (decision D11); only the reader's explicit act is.
 */
import { useEffect, useRef } from "react";
import { cn } from "../lib/cn";
import { ageText, sourceName } from "../lib/notifications";
import { messageRuntime } from "../store";
import type { NotificationItem } from "../types";

/* ICU `select` tokens for the accessible name (machine values, not copy). */
const UNREAD = "unread";
const READ = "read";

export function NotificationRow({
	item,
	selecting,
	selected,
	writesPaused,
	onOpen,
	onToggleSelect,
	onLongPress,
	onDismiss,
	onAnswer,
	nowMs,
	names,
}: {
	item: NotificationItem;
	selecting: boolean;
	selected: boolean;
	/** A refresh failed or a write is in flight: the list stays readable but
	    writes are held. A real `disabled` (review RW10): with `aria-disabled`
	    the tap still fired and was dropped without a word. */
	writesPaused: boolean;
	onOpen: () => void;
	onToggleSelect: () => void;
	onLongPress: () => void;
	onDismiss: () => void;
	onAnswer: () => void;
	nowMs: number;
	/** session id → the list frame's conversation name (`sourceName` fallback). */
	names?: ReadonlyMap<string, string>;
}) {
	const isAsk = item.source === "ask";
	const unread = !item.read;
	const source = sourceName(item, names);
	const age = ageText(item.updated_at, nowMs);
	/* Long-press enters selection mode, the session list's own 450 ms / 10 px
	   rule, so the two lists answer the same gesture the same way. */
	const pressTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
	const pressedAt = useRef<{ x: number; y: number } | null>(null);
	const suppressClick = useRef(false);
	const cancelPress = () => {
		if (pressTimer.current) clearTimeout(pressTimer.current);
		pressTimer.current = null;
		pressedAt.current = null;
	};
	useEffect(() => cancelPress, []);
	const label = messageRuntime.t("relay.notifications.row.a11y", {
		state: unread ? UNREAD : READ,
		priority: item.priority,
		title: item.title,
		source,
		age,
	});
	return (
		<div className="relative flex min-h-14 items-stretch">
			<button
				type="button"
				aria-label={selecting ? messageRuntime.t("relay.notifications.row.select", { title: item.title }) : label}
				aria-pressed={selecting ? selected : undefined}
				onPointerDown={event => {
					if (selecting) return;
					suppressClick.current = false;
					pressedAt.current = { x: event.clientX, y: event.clientY };
					pressTimer.current = setTimeout(() => {
						pressTimer.current = null;
						suppressClick.current = true;
						onLongPress();
					}, 450);
				}}
				onPointerMove={event => {
					const start = pressedAt.current;
					if (start && Math.hypot(event.clientX - start.x, event.clientY - start.y) > 10) cancelPress();
				}}
				onPointerUp={cancelPress}
				onPointerLeave={cancelPress}
				onPointerCancel={cancelPress}
				onContextMenu={event => { event.preventDefault(); onLongPress(); }}
				onClick={() => {
					if (suppressClick.current) { suppressClick.current = false; return; }
					if (selecting) onToggleSelect();
					else if (isAsk) onAnswer();
					else onOpen();
				}}
				className={cn("flex min-w-0 flex-1 items-center gap-3 ps-4 pe-13 py-2 text-start select-none active:bg-surface", selected && "bg-accent-wash")}
			>
				<span aria-hidden className="flex size-4 shrink-0 items-center justify-center">
					{selecting ? (
						<span className={cn("flex size-4 items-center justify-center rounded-sm text-meta",
							selected ? "bg-accent text-on-accent" : "border border-control")}>{selected ? "✓" : ""}</span>
					) : unread ? <span className="inline-block size-1.5 rounded-full bg-accent" /> : null}
				</span>
				<span aria-hidden className={cn("flex size-5 shrink-0 items-center justify-center text-body-sm",
					isAsk ? "text-info" : item.priority === "high" ? "text-warning" : "text-ink-dim")}>
					{isAsk ? "?" : item.priority === "high" ? "▁▃▅" : item.priority === "normal" ? "▁▃" : "▁"}
				</span>
				<span className="flex min-w-0 flex-1 flex-col gap-0.5">
					<span className="flex min-w-0 items-center gap-1.5">
						<span className={cn("min-w-0 flex-1 truncate text-body", unread ? "font-semibold text-ink" : "text-ink-muted")}>{item.title}</span>
						{item.priority === "high" && !isAsk ? <span className="shrink-0 rounded-sm bg-warning-wash px-1.5 text-meta text-warning">{messageRuntime.t("relay.notifications.row.high")}</span> : null}
						{item.repeat_count > 1 ? <span className="shrink-0 rounded-sm bg-surface px-1.5 text-meta tabular-nums text-ink-dim">{messageRuntime.t("relay.notifications.row.repeat", { count: item.repeat_count })}</span> : null}
					</span>
					{/* The age gets its OWN non-shrinking slot (past-sessions' rule): as
					    one truncating string, the age was the part clipped away at 130 %
					    text on a 360 px phone, and "how recent" is what the reel answers. */}
					<span className="flex min-w-0 gap-1.5 text-meta text-ink-dim">
						<span className="min-w-0 truncate">{source}</span>
						<span aria-hidden>·</span>
						<span className="shrink-0">{age}</span>
					</span>
				</span>
			</button>
			{selecting ? null : isAsk ? (
				<button type="button" onClick={onAnswer} aria-label={messageRuntime.t("relay.notifications.row.answer", { title: item.title })}
					className="absolute end-1 top-1/2 flex size-11 -translate-y-1/2 items-center justify-center rounded-md text-ink-muted active:bg-surface">›</button>
			) : item.actions.dismiss ? (
				<button type="button" onClick={onDismiss} disabled={writesPaused}
					aria-label={messageRuntime.t("relay.notifications.row.dismiss", { title: item.title })}
					className={cn("absolute end-1 top-1/2 flex size-11 -translate-y-1/2 items-center justify-center rounded-md active:bg-surface",
						writesPaused ? "text-ink-disabled" : "text-ink-muted")}>✕</button>
			) : null}
		</div>
	);
}

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
				className={cn("@container flex min-w-0 flex-1 items-center gap-3 ps-4 py-2 text-start select-none active:bg-surface", selecting ? "pe-4" : "pe-13", selected && "bg-accent-wash")}
			>
				<span aria-hidden className="flex size-4 shrink-0 items-center justify-center">
					{selecting ? (
						<span className={cn("flex size-4 items-center justify-center rounded-sm text-meta",
							selected ? "bg-accent text-on-accent" : "border border-control")}>{selected ? "✓" : ""}</span>
					) : unread ? <span className="inline-block size-1.5 rounded-full bg-accent" /> : null}
				</span>
				{/* ONE FIXED SLOT WIDE ENOUGH FOR THE WIDEST GLYPH (`▁▃▅`, 1.75em at
				    every size): the old `size-5` box measured 40 px against 57 px of bars
				    at 200 % and spilled into the title (design RW-D1), and a slot that
				    grew with its glyph put each row's title at a different x. It is decorative — priority is also
				    the High chip and the accessible name — so a row whose content box is
				    under 260 px drops it and gives the width to the title. A px bound,
				    NOT rem: a rem bound grows with the text it is meant to measure
				    against, and hid the glyph at 100 % (360 px phone: content box 292,
				    130 %: 272, 200 %: ~224 — only the last one drops it). */}
				<span aria-hidden className={cn("flex h-5 w-8 shrink-0 items-center justify-center text-body-sm @max-[260px]:hidden",
					isAsk ? "text-info" : item.priority === "high" ? "text-warning" : "text-ink-dim")}>
					{isAsk ? "?" : item.priority === "high" ? "▁▃▅" : item.priority === "normal" ? "▁▃" : "▁"}
				</span>
				{/* THE TITLE OWNS LINE ONE AND MAY WRAP TO TWO (design RW-D1). It used to
				    share one line with the shrink-0 High/×N chips, and every other part
				    of the row scales with text size, so at 200 % on 360 px the High
				    row's title was 0 px wide — the most urgent row was the unreadable
				    one. The chips now lead line two; the source keeps a floor of 8ch so
				    the row still says where it came from; the age keeps its own slot. */}
				<span className="flex min-w-0 flex-1 flex-col gap-0.5">
					<span className={cn("line-clamp-2 break-words text-body", unread ? "font-semibold text-ink" : "text-ink-muted")}>{item.title}</span>
					<span className="flex min-w-0 flex-wrap items-center gap-x-1.5 text-meta text-ink-dim">
						{item.priority === "high" && !isAsk ? <span className="shrink-0 rounded-sm bg-warning-wash px-1.5 text-warning">{messageRuntime.t("relay.notifications.row.high")}</span> : null}
						{item.repeat_count > 1 ? <span className="shrink-0 rounded-sm bg-surface px-1.5 tabular-nums">{messageRuntime.t("relay.notifications.row.repeat", { count: item.repeat_count })}</span> : null}
						{/* The source keeps an 8ch floor and the age never clips; when the
						    two cannot share a line (360 px at 200 % text: 157 px for ~220 px
						    of content, measured in round 1) the age wraps under the source
						    instead of running into the ✕ column. */}
						<span className="flex max-w-full flex-1 flex-wrap gap-x-1.5">
							{/* The separator travels with the source, so a wrapped age starts
							    its line clean rather than with a dangling "·". */}
							<span className="flex min-w-[8ch] max-w-full gap-1.5"><span className="min-w-0 truncate">{source}</span><span aria-hidden>·</span></span>
							<span className="shrink-0">{age}</span>
						</span>
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

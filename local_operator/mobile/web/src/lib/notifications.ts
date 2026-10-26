/**
 * Pure helpers for the phone inbox (`#/notifications`): the receipt sentence,
 * the age line, the day groups and the write failure. Kept out of the screens so
 * the rules the operator asked for — exact receipts, an honest partial, no
 * invented counts — are testable without a DOM.
 *
 * READ SEMANTICS (decision D11) are NOT here and deliberately so: nothing in
 * this module marks anything read. The only reads this lane sends are explicit
 * (a button, or opening an item's LOADED detail). The relay's
 * `POST /api/notifications/seen` door accepts native-surface evidence only, so
 * the web inbox never calls it.
 */
import type { MessageKey } from "../i18n/keys.gen";
import { messageRuntime } from "../store";
import type { NotificationItem, NotificationsReceipt, NotificationToken } from "../types";

export type NotificationsOp = "read" | "unread" | "dismiss" | "undismiss";

const CHANGED_KEY: Record<NotificationsOp, MessageKey> = {
	read: "relay.notifications.receipt.read",
	unread: "relay.notifications.receipt.unread",
	dismiss: "relay.notifications.receipt.dismissed",
	undismiss: "relay.notifications.receipt.restored",
};

/** One sentence for one write, naming EVERY bucket that is not empty.

    The PATCH answers `changed` plus `skipped` with a reason per id. A count that
    dropped the skipped ones would be the silent partial success the session
    list's mark-all already refuses (`markAllReceipt`), so each reason the store
    gave is stated: a live question stays, a row updated after the reader looked
    is left alone, an id the store no longer has is gone. */
export function receiptText(op: NotificationsOp, receipt: Pick<NotificationsReceipt, "changed" | "skipped">): string {
	const parts: string[] = [];
	const changed = receipt.changed.length;
	if (changed > 0) parts.push(messageRuntime.t(CHANGED_KEY[op] as "relay.notifications.receipt.read", { count: changed }));
	const tally = (reason: string) => receipt.skipped.filter(entry => entry.reason === reason).length;
	const liveAsk = tally("live_ask");
	const stale = tally("stale_revision");
	const gone = tally("unknown");
	const already = tally("already");
	if (liveAsk > 0) parts.push(messageRuntime.t("relay.notifications.receipt.live_ask", { count: liveAsk }));
	if (stale > 0) parts.push(messageRuntime.t("relay.notifications.receipt.stale", { count: stale }));
	if (gone > 0) parts.push(messageRuntime.t("relay.notifications.receipt.gone", { count: gone }));
	if (already > 0) parts.push(messageRuntime.t("relay.notifications.receipt.already", { count: already }));
	return parts.join(" · ") || messageRuntime.t("relay.notifications.receipt.nothing");
}

/** A failed write in the reader's words: nothing changed, why, and what to do.
    `fetch` rejects with browser-ese ("Failed to fetch"), which is not a reason a
    reader can act on, so an unreachable relay gets its own clause. */
export function writeFailureText(error: unknown): string {
	const message = error instanceof Error ? error.message : String(error);
	const unreachable = message === "" || /failed to fetch|networkerror|load failed|network request failed/i.test(message);
	return messageRuntime.t("relay.notifications.error.write", {
		reason: unreachable ? messageRuntime.t("relay.notifications.error.unreachable") : message,
	});
}

/** Epoch MILLISECONDS (the inbox wire's unit) → a short localized age. */
export function ageText(epochMs: number, nowMs: number = Date.now()): string {
	const seconds = Math.max(0, (nowMs - epochMs) / 1000);
	if (seconds < 45) return messageRuntime.t("relay.notifications.age.now");
	if (seconds < 3600) return messageRuntime.t("relay.notifications.age.minutes", { count: Math.max(1, Math.round(seconds / 60)) });
	if (seconds < 86400) return messageRuntime.t("relay.notifications.age.hours", { count: Math.round(seconds / 3600) });
	return messageRuntime.t("relay.notifications.age.days", { count: Math.round(seconds / 86400) });
}

export type DayGroup = "today" | "yesterday" | "earlier";

/** Time grouping (design §3): the newest priority item stays at the top of
    Today instead of being buried inside whichever conversation sorts first. */
export function dayGroup(epochMs: number, now: Date = new Date()): DayGroup {
	const midnight = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
	if (epochMs >= midnight) return "today";
	if (epochMs >= midnight - 86_400_000) return "yesterday";
	return "earlier";
}

export function groupByDay(items: NotificationItem[], now: Date = new Date()): { group: DayGroup; items: NotificationItem[] }[] {
	const order: DayGroup[] = ["today", "yesterday", "earlier"];
	const buckets = new Map<DayGroup, NotificationItem[]>();
	for (const item of items) {
		const group = dayGroup(item.updated_at, now);
		buckets.set(group, [...(buckets.get(group) ?? []), item]);
	}
	return order.filter(group => buckets.has(group)).map(group => ({ group, items: buckets.get(group)! }));
}

/** The conversation's name. The wire's own title first; then the list frame's
    `conversation_name` for that id (an ask row arrives with an empty title when
    the conversation catalogue has no cached entry, measured on #2939 head
    3cacd0cad013: `waiting[].session.title == ""` beside a titled notify row of
    the same session); "Unknown conversation" only when neither names it. */
export function sourceName(
	item: Pick<NotificationItem, "session">,
	names?: ReadonlyMap<string, string>,
): string {
	return item.session?.title?.trim()
		|| (item.session?.id ? names?.get(item.session.id)?.trim() : "")
		|| messageRuntime.t("relay.notifications.row.unknown_source");
}

export function tokenOf(item: Pick<NotificationItem, "id" | "revision">): NotificationToken {
	return { id: item.id, revision: item.revision };
}

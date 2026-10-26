/**
 * The phone inbox (`#/notifications`): ONE reel across every conversation —
 * the live questions an agent is blocked on, pinned as "Waiting on you", then
 * the updates agents raised with `notify`, newest first in day groups
 * (design §§2-3, §9; contract §2).
 *
 * WHAT THIS SCREEN NEVER DOES (decision D11). Rendering, scrolling, focusing
 * or returning to this list marks nothing read: a reader coming back after an
 * hour must find every missed item still unread. Reads here are explicit —
 * Mark all read (confirmed), the selection bar's Read, or opening an item's
 * loaded detail (`notification-detail.tsx`). The relay's
 * `POST /api/notifications/seen` door is native-surface only, so this web
 * client has no in-view auto-read at all; that is a stated gap, not an
 * oversight.
 *
 * EVERY BULK WRITE IS BOUNDED BY WHAT WAS LOADED. The list answers
 * `observed` tokens and a `high_water` per read; Mark all read echoes the
 * union of loaded tokens plus the FIRST page's `high_water` (later pages are
 * read with that same `through_seq`, so the whole loaded set is one read). An
 * item published after the read is in neither, and the store leaves it
 * unread — the late-arrival rule the operator asked for.
 *
 * ASKS KEEP ONE ANSWERING SURFACE. A live question opens the redesign's own
 * Questions sheet (`AsksSheet`, cross-conversation mode) rather than a second
 * answer UI with its own receipts.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getNotifications, patchNotifications, type NotificationsPatchBody } from "../api";
import { AsksSheet } from "../components/asks-sheet";
import { NotificationReceipt } from "../components/notification-receipt";
import { NotificationRow } from "../components/notification-row";
import { Button } from "../components/ui/button";
import { Chip } from "../components/ui/chip";
import { Sheet } from "../components/ui/sheet";
import { cn } from "../lib/cn";
import { failureReason, groupByDay, inboxServed, mergeObserved, receiptText, tokenOf, writeFailureText, type NotificationsOp } from "../lib/notifications";
import { navigate, navigateUp } from "../router";
import { messageRuntime, retainSessionListStream, useCapabilities, useNotificationsBlock, useSessions } from "../store";
import type { NotificationItem, NotificationsReceipt, NotificationToken } from "../types";

type Tab = "unread" | "all";

/* Machine tokens held outside JSX so the relay copy gate (check-i18n.mjs),
   which reads string literals in expressions as copy, scans this file clean. */
const SECTION_KEY = {
	today: "relay.notifications.section.today",
	yesterday: "relay.notifications.section.yesterday",
	earlier: "relay.notifications.section.earlier",
} as const;
/** Width reservation only (RW-D4): never shown, never announced. */
const PLACEHOLDER_COUNT = 8;
const CLOCK: Intl.DateTimeFormatOptions = { hour: "numeric", minute: "2-digit" };

/** The tab is a per-device preference (design §2: "the choice persists per
    device"). Storage can throw (private mode, quota); the default then holds. */
const TAB_KEY = "lo-mobile-notifications-tab";
function readTab(): Tab {
	try { return localStorage.getItem(TAB_KEY) === "all" ? "all" : "unread"; } catch { return "unread"; }
}
function writeTab(tab: Tab): void {
	try { localStorage.setItem(TAB_KEY, tab); } catch { /* preference only */ }
}

type Loaded = {
	items: NotificationItem[];
	waiting: NotificationItem[];
	observed: NotificationToken[];
	/** The first page's boundary; every later page and every bulk write uses it. */
	highWater: number;
	nextCursor: string | null;
	/** Pages read so far, the depth a refresh re-reads to (RW3). */
	pages: number;
	unreadCount: number;
	waitingCount: number;
};

export type Receipt = {
	text: string;
	danger: boolean;
	/** The write that takes this one back, when anything changed. */
	undo?: NotificationsPatchBody;
};

const INVERSE: Partial<Record<NotificationsOp, NotificationsOp>> = { read: "unread", dismiss: "undismiss" };

/** Run one write and describe it exactly. The Undo targets the ids that
    CHANGED, at the revisions the reader saw — never the skipped ones, which
    the write did not touch and Undo must not touch either. */
export async function runWrite(
	op: NotificationsOp,
	body: NotificationsPatchBody,
	tokens: NotificationToken[],
): Promise<{ receipt: Receipt; answer?: NotificationsReceipt }> {
	try {
		const answer = await patchNotifications(body);
		const inverse = INVERSE[op];
		const byId = new Map(tokens.map(token => [token.id, token]));
		const undoTokens = answer.changed.map(id => byId.get(id)).filter((token): token is NotificationToken => token !== undefined);
		return {
			answer,
			receipt: {
				text: receiptText(op, answer),
				danger: false,
				undo: inverse && undoTokens.length > 0 ? { op: inverse, ids: undoTokens } : undefined,
			},
		};
	} catch (error) {
		return { receipt: { text: writeFailureText(error), danger: true } };
	}
}

export function NotificationsScreen() {
	useEffect(() => retainSessionListStream(), []);
	const block = useNotificationsBlock();
	const { sessions } = useSessions();
	const capabilities = useCapabilities();
	/* A DEEP LINK ON A RELAY WITHOUT THE INBOX (review RW5). The drawer offers
	   no row there, but a bookmark or shared `#/notifications` still lands here,
	   and its first read would 404 into "Couldn't load notifications". Once a
	   list frame has answered (capabilities known) without the inbox, leave for
	   the list, as an unknown route does; before that, nothing is decided. An
	   inbox whose STORE is unreadable is not this case (QA Q6): it stays, and
	   the first read paints the relay's own `store_unavailable` sentence. */
	const unavailable = capabilities !== null && !inboxServed(capabilities, block);
	useEffect(() => { if (unavailable) navigateUp("/"); }, [unavailable]);

	const [tab, setTab] = useState<Tab>(readTab);
	const [loaded, setLoaded] = useState<Loaded | null>(null);
	const [loadError, setLoadError] = useState("");
	/* A failed REFRESH keeps the list (never blank what the reader is reading)
	   and pauses writes until a read succeeds again (design §5). */
	const [refreshFailedAt, setRefreshFailedAt] = useState<number | null>(null);
	const [paging, setPaging] = useState(false);
	const [selecting, setSelecting] = useState(false);
	const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
	/* The exact set Mark all read will send, FROZEN WHEN THE SHEET OPENS (review
	   RW1). The sheet covers the list, so a frame landing while it is up must not
	   widen the write to rows the reader never saw; the snapshot is what the
	   title counts and what the PATCH carries. Unread rows only (RW6), so the
	   title, the body and the receipt describe one set. */
	const [confirmMarkAll, setConfirmMarkAll] = useState<{ tokens: NotificationToken[]; highWater: number } | null>(null);
	const [writing, setWriting] = useState(false);
	const [receipt, setReceipt] = useState<Receipt | null>(null);
	const [asksOpen, setAsksOpen] = useState(false);
	const [nowMs, setNowMs] = useState(() => Date.now());
	const seq = useRef(0);
	/* Whether a list is on screen, read by the loader to tell a failed first
	   load (replace the body) from a failed refresh (keep the list). A ref, so
	   the loader need not re-create on every page it loads. */
	const loadedRef = useRef<Loaded | null>(null);
	loadedRef.current = loaded;

	useEffect(() => {
		const timer = setInterval(() => setNowMs(Date.now()), 30_000);
		return () => clearInterval(timer);
	}, []);

	/* RE-READ TO THE SAME DEPTH (review RW3). Every notify anywhere moves the
	   frame and lands here; replacing the list with page one alone collapsed a
	   reader who had loaded more back to the first 50 rows, and left bulk acts
	   covering only those. So the refresh re-reads page one, then follows
	   `next_cursor` (pinned to the NEW page one's `high_water`, one consistent
	   read) until every row that was on screen is accounted for, or the list
	   ends. Counting rows instead was wrong: arrivals at the top fill page one
	   and stop the walk before the reader's tail (measured in this PR's test).
	   A row that left the store is simply not found, so the walk also stops at
	   the old page count. The whole set is replaced at once, never merged with
	   rows from an older read: a merged tail could carry tokens the new
	   boundary no longer vouches for. */
	const loadFirst = useCallback(async () => {
		const mine = ++seq.current;
		const previous = loadedRef.current;
		const wanted = new Set(previous?.items.map(row => row.id) ?? []);
		const pages = previous?.pages ?? 1;
		try {
			const page = await getNotifications({ state: tab });
			if (mine !== seq.current) return;
			let items = page.items;
			let observed = page.observed;
			let nextCursor = page.next_cursor;
			let read = 1;
			const missing = () => [...wanted].some(id => !items.some(row => row.id === id));
			while (nextCursor && read < pages + 1 && missing()) {
				read += 1;
				const more = await getNotifications({ state: tab, cursor: nextCursor, through_seq: page.high_water });
				if (mine !== seq.current) return;
				items = [...items, ...more.items.filter(item => !items.some(row => row.id === item.id))];
				observed = mergeObserved(observed, more.observed);
				nextCursor = more.next_cursor;
			}
			setLoaded({
				items,
				waiting: page.waiting ?? [],
				observed,
				highWater: page.high_water,
				nextCursor,
				pages: read,
				unreadCount: page.unread_count,
				waitingCount: page.waiting_count,
			});
			setLoadError("");
			setRefreshFailedAt(null);
		} catch (error) {
			if (mine !== seq.current) return;
			if (loadedRef.current) setRefreshFailedAt(Date.now());
			/* The cause in the reader's words, not transport text (design RW-D6). */
			else setLoadError(failureReason(error));
		}
	}, [tab]);

	/* The list frame's badge block moves whenever the store does (the daemon
	   fingerprints `notifications.db`), so its numbers are the refetch signal;
	   no second stream and no poll. */
	const signal = block ? `${block.high_water}:${block.unread_count}:${block.waiting_count}:${block.waiting_unread_count}` : "";
	useEffect(() => { if (!unavailable) void loadFirst(); }, [loadFirst, signal, unavailable]);

	const loadMore = async () => {
		if (!loaded?.nextCursor || paging) return;
		setPaging(true);
		const mine = seq.current;
		try {
			const page = await getNotifications({ state: tab, cursor: loaded.nextCursor, through_seq: loaded.highWater });
			if (mine !== seq.current) return;
			setLoaded(current => current && {
				...current,
				items: [...current.items, ...page.items.filter(item => !current.items.some(row => row.id === item.id))],
				observed: mergeObserved(current.observed, page.observed),
				nextCursor: page.next_cursor,
				pages: current.pages + 1,
			});
		} catch {
			if (mine === seq.current) setRefreshFailedAt(Date.now());
		} finally {
			setPaging(false);
		}
	};

	const writesPaused = refreshFailedAt !== null;
	const write = async (op: NotificationsOp, body: NotificationsPatchBody, tokens: NotificationToken[]) => {
		if (writing || writesPaused) return;
		setWriting(true);
		setReceipt(null);
		const { receipt: next } = await runWrite(op, body, tokens);
		setReceipt(next);
		setWriting(false);
		await loadFirst();
	};

	const undo = async () => {
		const body = receipt?.undo;
		if (!body || !("ids" in body)) return;
		setReceipt(null);
		setWriting(true);
		const { receipt: next } = await runWrite(body.op, body, body.ids);
		setReceipt({ ...next, undo: undefined });
		setWriting(false);
		await loadFirst();
	};

	const changeTab = (next: Tab) => {
		setTab(next);
		writeTab(next);
		setSelected(new Set());
		setLoaded(null);
		setLoadError("");
	};

	/* Conversation names for rows whose own `session.title` is empty (an ask's,
	   when the catalogue has no cached entry): the list frame's names first, then
	   any loaded notify row of the same conversation, which carries the stored
	   `session_title` the agent raised it under. */
	/* CUMULATIVE for this screen's life: dismissing a conversation's last update
	   must not turn its still-pending question into "Unknown conversation"
	   (measured on the real relay: exactly that, after a multi-select dismiss). */
	const learnedNames = useRef(new Map<string, string>());
	const names = useMemo(() => {
		for (const row of loaded?.items ?? []) if (row.session?.id && row.session.title) learnedNames.current.set(row.session.id, row.session.title);
		const map = new Map(learnedNames.current);
		for (const row of sessions) if (row.conversation_name) map.set(row.session_id, row.conversation_name);
		return map;
	}, [sessions, loaded]);
	const allRows = useMemo(() => [...(loaded?.waiting ?? []), ...(loaded?.items ?? [])], [loaded]);
	const unreadLoaded = allRows.filter(item => !item.read);
	const groups = useMemo(() => groupByDay(loaded?.items ?? [], new Date(nowMs)), [loaded, nowMs]);
	const selectedRows = allRows.filter(item => selected.has(item.id));
	const selectedDismissible = selectedRows.filter(item => item.actions.dismiss);
	const selectedQuestions = selectedRows.filter(item => item.source === "ask").length;

	const toggle = (id: string) => setSelected(current => {
		const next = new Set(current);
		if (next.has(id)) next.delete(id); else next.add(id);
		return next;
	});
	const startSelect = (id?: string) => {
		setSelecting(true);
		setSelected(id ? new Set([id]) : new Set());
	};
	const endSelect = () => { setSelecting(false); setSelected(new Set()); };

	const openMarkAll = () => {
		if (!loaded) return;
		setConfirmMarkAll({ tokens: unreadLoaded.map(tokenOf), highWater: loaded.highWater });
	};
	const markAll = async () => {
		const plan = confirmMarkAll;
		setConfirmMarkAll(null);
		if (!plan || plan.tokens.length === 0) return;
		await write("read", { op: "read", all: true, through_seq: plan.highWater, observed: plan.tokens }, plan.tokens);
	};

	const renderRow = (item: NotificationItem) => (
		<NotificationRow key={item.id} item={item} nowMs={nowMs} names={names} selecting={selecting} selected={selected.has(item.id)}
			writesPaused={writesPaused || writing}
			onOpen={() => navigate(`/notifications/${encodeURIComponent(item.id)}`)}
			onAnswer={() => setAsksOpen(true)}
			onToggleSelect={() => toggle(item.id)}
			onLongPress={() => startSelect(item.id)}
			onDismiss={() => void write("dismiss", { op: "dismiss", ids: [tokenOf(item)] }, [tokenOf(item)])} />
	);
	/* Section headers stick BELOW the Mark-all band while it shows (design
	   RW-D5): at `top-0` both pinned at the same y and the band hid Today. */
	const bandShown = !selecting && unreadLoaded.length > 0;
	const sectionClass = cn("sticky z-10 truncate bg-canvas px-4 pt-4 pb-1 text-meta font-medium text-ink-dim", bandShown ? "top-11" : "top-0");
	const empty = loaded !== null && allRows.length === 0;

	return (
		<div className="relative mx-auto flex h-dvh w-full max-w-[var(--lo-column-max,28rem)] flex-col">
			<header className="flex h-[calc(52px+env(safe-area-inset-top))] shrink-0 items-center gap-1 px-1 pt-[env(safe-area-inset-top)]">
				<button type="button" onClick={() => navigateUp("/")} aria-label={messageRuntime.t("relay.notifications.back")}
					className="flex size-11 shrink-0 items-center justify-center rounded-md text-ink-muted active:bg-surface">‹</button>
				<h1 className="min-w-0 flex-1 truncate text-heading font-semibold">{selecting
					? messageRuntime.t("relay.notifications.select.count", { count: selected.size })
					: messageRuntime.t("relay.notifications.title")}</h1>
				{selecting ? <Button variant="ghost" onClick={endSelect}>{messageRuntime.t("relay.notifications.select.cancel")}</Button>
					: allRows.length > 0 ? <Button variant="ghost" onClick={() => startSelect()}>{messageRuntime.t("relay.notifications.select")}</Button> : null}
			</header>
			<div role="group" aria-label={messageRuntime.t("relay.notifications.filter.label")} className="flex shrink-0 items-center gap-2 px-4 pb-2">
				<Chip selected={tab === "unread"} aria-pressed={tab === "unread"} onClick={() => changeTab("unread")}>
					{/* D16: an Unread filter's count equals the rows that filter shows.
					    This tab lists every unread update (`unread_count`, across all
					    pages, not just the loaded one) plus EVERY live question —
					    pending is not unread, so a read ask stays pinned here — hence
					    `waiting_count`, not the badge's `waiting_unread_count`. Shown
					    only once the tab's own read answered; unknown renders no number. */}
					{/* NO JUMP WHEN THE COUNT ARRIVES (design RW-D4). Before the first
					    read the chip still lays out the counted message, with an
					    invisible placeholder figure, so the All chip beside it does not
					    move 19 px under the thumb when the number lands. The figure is
					    hidden from sight AND from AT (D16: unknown is never stated as a
					    number); the plain label is what is read until then. */}
					{tab !== "unread" ? messageRuntime.t("relay.notifications.filter.unread_plain")
						: loaded ? messageRuntime.t("relay.notifications.filter.unread", { count: loaded.unreadCount + loaded.waitingCount })
						: <span className="relative">
							<span aria-hidden className="invisible">{messageRuntime.t("relay.notifications.filter.unread", { count: PLACEHOLDER_COUNT })}</span>
							<span className="absolute inset-y-0 start-0 whitespace-nowrap">{messageRuntime.t("relay.notifications.filter.unread_plain")}</span>
						</span>}
				</Chip>
				<Chip selected={tab === "all"} aria-pressed={tab === "all"} onClick={() => changeTab("all")}>{messageRuntime.t("relay.notifications.filter.all")}</Chip>
			</div>
			<main className="lo-scroll lo-fade-y min-h-0 flex-1 overflow-y-auto pb-[calc(88px+env(safe-area-inset-bottom))]">
				{refreshFailedAt !== null && loaded ? <p role="alert" className="mx-4 mb-2 rounded-md bg-danger-wash px-3 py-2 text-body-sm text-danger">
					{messageRuntime.t("relay.notifications.error.refresh", { time: new Date(refreshFailedAt).toLocaleTimeString(undefined, CLOCK) })}
					<button type="button" className="ms-2 min-h-11 font-semibold underline" onClick={() => void loadFirst()}>{messageRuntime.t("relay.notifications.error.retry")}</button>
				</p> : null}
				{bandShown ? <div className="sticky top-0 z-20 bg-canvas">
					<button type="button" onClick={openMarkAll} disabled={writing || writesPaused} aria-busy={writing}
						className="min-h-11 w-full px-4 text-start text-body-sm text-ink-muted active:bg-surface disabled:text-ink-disabled">{messageRuntime.t("relay.notifications.mark_all")}</button>
				</div> : null}
				{loaded === null ? loadError ? <div role="alert" className="flex min-h-[55vh] flex-col items-center justify-center gap-3 px-6 text-center">
					<h2 className="text-heading font-semibold">{messageRuntime.t("relay.notifications.error.load")}</h2>
					<p className="text-body-sm text-ink-muted">{loadError}</p>
					<Button onClick={() => { setLoadError(""); void loadFirst(); }}>{messageRuntime.t("relay.notifications.error.retry")}</Button>
				</div> : <div aria-label={messageRuntime.t("relay.notifications.loading")} aria-busy="true">
					{/* THE SKELETON HAS THE CONTENT'S GEOMETRY (design RW-D4): a band the
					    height of Mark all read and a section-header slot come first, so the
					    first real row lands where the first skeleton row was instead of
					    73 px lower. Bars are `bg-hairline` (1.41 / 1.25 on canvas; the
					    `bg-surface` bars measured 1.07:1 and read as a blank screen,
					    RW-D3) and pulse like the Library's, stilled under reduced motion. */}
					<div className="min-h-11" />
					<div className="flex h-[1lh] items-center px-4 pt-4 pb-1 text-meta box-content"><div className="h-3 w-24 rounded-sm bg-hairline motion-safe:animate-pulse" /></div>
					{[0, 1, 2, 3, 4].map(i => <div key={i} className="flex min-h-14 items-center gap-3 ps-4 pe-13 py-2"><div className="size-4 shrink-0" /><div className="h-5 w-8 shrink-0" />
						<div className="flex flex-1 flex-col gap-2"><div className="h-3 w-3/5 rounded-sm bg-hairline motion-safe:animate-pulse" /><div className="h-2 w-[35%] rounded-sm bg-hairline motion-safe:animate-pulse" /></div></div>)}
				</div> : empty ? <div className="flex min-h-[55vh] flex-col items-center justify-center gap-2 px-6 text-center">
					<span aria-hidden className="mb-2 text-title text-ink-dim">✓</span>
					<h2 className="text-heading font-semibold">{tab === "unread" ? messageRuntime.t("relay.notifications.empty.caught_up.title") : messageRuntime.t("relay.notifications.empty.none.title")}</h2>
					<p className="text-body-sm text-ink-muted">{tab === "unread" ? messageRuntime.t("relay.notifications.empty.caught_up.body") : messageRuntime.t("relay.notifications.empty.none.body")}</p>
					{tab === "unread" ? <Button variant="ghost" onClick={() => changeTab("all")}>{messageRuntime.t("relay.notifications.empty.caught_up.show_all")}</Button> : null}
				</div> : <>
					{loaded.waiting.length > 0 ? <section>
						<h2 className={sectionClass}>{messageRuntime.t("relay.notifications.section.waiting", { count: loaded.waiting.length })}</h2>
						{loaded.waiting.map(renderRow)}
					</section> : null}
					{groups.map(group => <section key={group.group}>
						<h2 className={sectionClass}>{messageRuntime.t(SECTION_KEY[group.group])}</h2>
						{group.items.map(renderRow)}
					</section>)}
					{loaded.nextCursor ? <div className="px-4 py-3"><Button className="w-full" onClick={() => void loadMore()} disabled={paging} aria-busy={paging}>
						{messageRuntime.t("relay.notifications.load_more")}</Button></div> : null}
				</>}
			</main>
			{selecting ? <div className="absolute inset-x-0 bottom-0 z-20 flex items-center gap-2 bg-surface px-3 pt-2 pb-[max(env(safe-area-inset-bottom),0.5rem)] shadow-overlay">
				<span className="flex min-w-0 flex-1 flex-col">
					<span className="truncate text-body-sm text-ink-muted">{messageRuntime.t("relay.notifications.select.count", { count: selected.size })}</span>
					{selectedQuestions > 0 ? <span className="text-meta text-ink-dim">{messageRuntime.t("relay.notifications.receipt.live_ask", { count: selectedQuestions })}</span> : null}
				</span>
				<Button disabled={selectedRows.length === 0 || writing || writesPaused}
					onClick={() => { const tokens = selectedRows.map(tokenOf); endSelect(); void write("read", { op: "read", ids: tokens }, tokens); }}>
					{messageRuntime.t("relay.notifications.select.read")}</Button>
				{/* The count is the rows a dismiss can act on: live questions are
				    skipped by the store (`live_ask`), and the receipt names them. */}
				{/* Dismiss acts on updates only: a question stays until answered, and
				    the bar says so BEFORE the tap (design RW-D2), with the store's own
				    receipt wording. Nothing dismissible selected = nothing to offer. */}
				<Button variant="danger" disabled={selectedDismissible.length === 0 || writing || writesPaused}
					onClick={() => { const tokens = selectedRows.map(tokenOf); endSelect(); void write("dismiss", { op: "dismiss", ids: tokens }, tokens); }}>
					{messageRuntime.t("relay.notifications.select.dismiss", { count: selectedDismissible.length })}</Button>
			</div> : null}
			<NotificationReceipt text={receipt?.text ?? ""} danger={receipt?.danger} onUndo={receipt?.undo ? () => void undo() : undefined} onDismiss={() => setReceipt(null)} />
			<Sheet open={confirmMarkAll !== null} onClose={() => setConfirmMarkAll(null)} title={messageRuntime.t("relay.notifications.confirm.mark_all", { count: confirmMarkAll?.tokens.length ?? 0 })}>
				<div className="flex flex-col gap-3 p-3">
					<p className="text-body-sm text-ink-muted">{messageRuntime.t("relay.notifications.confirm.mark_all_body")}</p>
					<div className="flex justify-end gap-2">
						<Button variant="ghost" onClick={() => setConfirmMarkAll(null)}>{messageRuntime.t("relay.notifications.confirm.cancel")}</Button>
						<Button variant="primary" onClick={() => void markAll()}>{messageRuntime.t("relay.notifications.confirm.read")}</Button>
					</div>
				</div>
			</Sheet>
			<AsksSheet open={asksOpen} onClose={() => setAsksOpen(false)} currentSessionId="" />
		</div>
	);
}

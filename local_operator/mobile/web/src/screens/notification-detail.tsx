/**
 * One inbox item (`#/notifications/<id>`): the body and links the agent wrote,
 * the transcript line it was raised beside, the conversation it came from, its
 * related projects, the questions still waiting in that conversation, and the
 * earlier revisions a repeat collapsed (design §2 detail, §9 phone layout).
 *
 * OPENING A LOADED DETAIL IS THE EXPLICIT READ (decision D11). The read is sent
 * only AFTER the item's own content has rendered, with the revision this
 * screen is showing, so a newer revision published meanwhile stays unread
 * (`stale_revision`). A route mount before content arrives, a failed load, a
 * document hidden when the content landed (a restored background tab) or an
 * already-read item sends nothing, and neither refocus nor a later
 * visibilitychange sends it after the fact. Once per revision shown.
 *
 * Questions are answered in the existing Questions sheet, not here: one
 * answering surface on the phone.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getNotification, getNotifications, HttpError, patchNotifications } from "../api";
import { AsksSheet } from "../components/asks-sheet";
import { Markdown } from "../components/markdown";
import { NotificationReceipt } from "../components/notification-receipt";
import { Button } from "../components/ui/button";
import { Sheet } from "../components/ui/sheet";
import { ageText, sourceName, tokenOf } from "../lib/notifications";
import { navigate, navigateUp } from "../router";
import { messageRuntime, retainSessionListStream, useSessions } from "../store";
import type { NotificationDetail } from "../types";
import { runWrite, type Receipt } from "./notifications";

/** The ICU `select` token for the From card (a machine value, not copy). */
function sessionState(live: boolean): string {
	return live ? LIVE : IDLE;
}
const LIVE = "live";
const IDLE = "idle";

type LoadState = { state: "loading" } | { state: "missing" } | { state: "error"; message: string } | { state: "ready"; detail: NotificationDetail };

export function NotificationDetailScreen({ itemId }: { itemId: string }) {
	useEffect(() => retainSessionListStream(), []);
	const { sessions } = useSessions();
	const names = useMemo(() => new Map(sessions.map(row => [row.session_id, row.conversation_name])), [sessions]);
	const [load, setLoad] = useState<LoadState>({ state: "loading" });
	const [receipt, setReceipt] = useState<Receipt | null>(null);
	const [writing, setWriting] = useState(false);
	const [asksOpen, setAsksOpen] = useState(false);
	const [confirmSession, setConfirmSession] = useState<{ count: number; through: number; observed: { id: string; revision: number }[] } | null>(null);
	/* `<id>@<revision>` already read by this screen, so a refetch after a write
	   cannot re-send the read the reader just undid with Mark unread. */
	const readSent = useRef<string>("");
	const [nowMs] = useState(() => Date.now());

	const refetch = useCallback(async () => {
		try {
			const detail = await getNotification(itemId);
			setLoad({ state: "ready", detail });
		} catch (error) {
			if (error instanceof HttpError && error.status === 404) setLoad({ state: "missing" });
			else setLoad({ state: "error", message: error instanceof Error ? error.message : String(error) });
		}
	}, [itemId]);
	useEffect(() => { void refetch(); }, [refetch]);

	const detail = load.state === "ready" ? load.detail : null;
	const item = detail?.item ?? null;

	/* The explicit read. Runs after the render that painted `item`'s content. */
	useEffect(() => {
		if (!item || item.read || item.dismissed) return;
		const key = `${item.id}@${item.revision}`;
		if (readSent.current === key) return;
		/* Hidden at load (a tab restored in the background, a prefetch) is not
		   opening it, and a later visibilitychange alone is not engagement
		   either, so nothing is sent; Mark read stays one tap away. */
		if (document.visibilityState !== "visible") return;
		readSent.current = key;
		void patchNotifications({ op: "read", ids: [tokenOf(item)] })
			.then(answer => { if (answer.changed.includes(item.id)) void refetch(); })
			.catch(() => { /* The item stays unread, and this screen still says so. */ });
	}, [item, refetch]);

	const act = async (op: "read" | "unread" | "dismiss" | "undismiss") => {
		if (!item || writing) return;
		setWriting(true);
		const tokens = [tokenOf(item)];
		/* Mark unread is the reader overruling the automatic read, so this screen
		   must not read it again on the refetch that follows. */
		if (op === "unread") readSent.current = `${item.id}@${item.revision}`;
		const { receipt: next } = await runWrite(op, { op, ids: tokens }, tokens);
		setReceipt(next);
		setWriting(false);
		await refetch();
	};

	/* Dismiss-all-from-this-conversation reads THAT conversation's own set first,
	   so the confirmation states the exact number and the write echoes exactly
	   those tokens and that boundary: an update raised after the read stays. */
	const prepareSessionDismiss = async () => {
		const sessionId = detail?.session.id;
		if (!sessionId) return;
		try {
			const page = await getNotifications({ state: "all", session_id: sessionId, limit: 100 });
			const observed = page.items.filter(row => row.actions.dismiss).map(tokenOf);
			setConfirmSession({ count: observed.length, through: page.high_water, observed });
		} catch (error) {
			setReceipt({ text: messageRuntime.t("relay.notifications.error.write", { reason: error instanceof Error ? error.message : String(error) }), danger: true });
		}
	};
	const dismissSession = async () => {
		const plan = confirmSession;
		const sessionId = detail?.session.id;
		setConfirmSession(null);
		if (!plan || !sessionId) return;
		setWriting(true);
		const { receipt: next } = await runWrite("dismiss", { op: "dismiss", session_id: sessionId, through_seq: plan.through, observed: plan.observed }, plan.observed);
		setReceipt(next);
		setWriting(false);
		await refetch();
	};
	const undo = async () => {
		const body = receipt?.undo;
		if (!body || !("ids" in body)) return;
		setReceipt(null);
		setWriting(true);
		const { receipt: next } = await runWrite(body.op, body, body.ids);
		setReceipt({ ...next, undo: undefined });
		setWriting(false);
		await refetch();
	};

	const sectionClass = "px-1 pt-3 pb-1 text-meta font-medium text-ink-dim";
	const earlier = detail && item ? detail.history.filter(entry => entry.revision !== item.revision) : [];
	return (
		<div className="relative mx-auto flex h-dvh w-full max-w-[var(--lo-column-max,28rem)] flex-col">
			<header className="flex h-[calc(52px+env(safe-area-inset-top))] shrink-0 items-center gap-1 px-1 pt-[env(safe-area-inset-top)]">
				<button type="button" onClick={() => navigateUp("/notifications")} aria-label={messageRuntime.t("relay.notifications.detail.back")}
					className="flex size-11 shrink-0 items-center justify-center rounded-md text-ink-muted active:bg-surface">‹</button>
				<h1 className="min-w-0 flex-1 truncate text-heading font-semibold">{messageRuntime.t("relay.notifications.title")}</h1>
			</header>
			<main className="lo-scroll min-h-0 flex-1 overflow-y-auto px-4 pb-[calc(88px+env(safe-area-inset-bottom))]">
				{load.state === "loading" ? <p aria-busy="true" className="pt-4 text-body-sm text-ink-dim">{messageRuntime.t("relay.notifications.detail.loading")}</p>
				: load.state === "missing" ? <div className="flex min-h-[55vh] flex-col items-center justify-center gap-2 text-center">
					<h2 className="text-heading font-semibold">{messageRuntime.t("relay.notifications.detail.missing.title")}</h2>
					<p className="text-body-sm text-ink-muted">{messageRuntime.t("relay.notifications.detail.missing.body")}</p>
					<Button variant="ghost" onClick={() => navigate("/notifications")}>{messageRuntime.t("relay.notifications.detail.back")}</Button>
				</div>
				: load.state === "error" ? <div role="alert" className="flex min-h-[55vh] flex-col items-center justify-center gap-3 text-center">
					<h2 className="text-heading font-semibold">{messageRuntime.t("relay.notifications.error.load")}</h2>
					<p className="text-body-sm text-ink-muted">{load.message}</p>
					<Button onClick={() => { setLoad({ state: "loading" }); void refetch(); }}>{messageRuntime.t("relay.notifications.error.retry")}</Button>
				</div>
				: detail && item ? <article className="flex flex-col gap-2 pt-1">
					<p className={item.priority === "high" ? "text-meta font-medium text-warning" : "text-meta text-ink-dim"}>
						{messageRuntime.t("relay.notifications.detail.kind", { priority: item.priority, kind: item.source })}</p>
					<h2 className="text-title text-ink">{item.title}</h2>
					<p className="text-meta text-ink-dim">
						{messageRuntime.t("relay.notifications.row.meta", { source: sourceName(item, names), age: ageText(item.updated_at, nowMs) })}
						{item.repeat_count > 1 ? ` · ${messageRuntime.t("relay.notifications.row.repeat", { count: item.repeat_count })}` : ""}
					</p>
					{item.dismissed ? <p role="status" className="rounded-md bg-surface px-3 py-2 text-body-sm text-ink-muted">{messageRuntime.t("relay.notifications.detail.dismissed")}</p> : null}
					<div className="flex flex-col gap-2 pt-1">
						{detail.session.id ? <Button variant="primary" className="w-full" onClick={() => navigate(`/s/${encodeURIComponent(detail.session.id)}`)}>
							{messageRuntime.t("relay.notifications.detail.open_conversation")}</Button> : null}
						<div className="flex gap-2">
							<Button className="flex-1" disabled={writing} onClick={() => void act(item.read ? "unread" : "read")}>
								{item.read ? messageRuntime.t("relay.notifications.detail.mark_unread") : messageRuntime.t("relay.notifications.detail.mark_read")}</Button>
							{item.source === "notify" ? <Button className="flex-1" variant={item.dismissed ? "tonal" : "danger"} disabled={writing}
								onClick={() => void act(item.dismissed ? "undismiss" : "dismiss")}>
								{item.dismissed ? messageRuntime.t("relay.notifications.detail.restore") : messageRuntime.t("relay.notifications.detail.dismiss")}</Button> : null}
						</div>
					</div>
					{item.body_md ? <div className="pt-2 text-body text-ink"><Markdown text={item.body_md} /></div> : item.excerpt ? <p className="pt-2 text-body text-ink">{item.excerpt}</p> : null}
					{item.links && item.links.length > 0 ? <section>
						<h3 className={sectionClass}>{messageRuntime.t("relay.notifications.detail.section.links")}</h3>
						<ul className="flex flex-col">{item.links.map(link => <li key={link.url}>
							{/* http(s) only: the store's `normalize_links` drops anything else. */}
							<a href={link.url} target="_blank" rel="noopener noreferrer" className="flex min-h-11 items-center break-all text-body-sm text-accent underline">{link.label}</a>
						</li>)}</ul>
					</section> : null}
					{detail.anchor ? <section>
						<h3 className={sectionClass}>{messageRuntime.t("relay.notifications.detail.section.anchor")}</h3>
						<p className="whitespace-pre-wrap rounded-md bg-surface px-3 py-2 text-body-sm text-ink-muted">{detail.anchor.excerpt}</p>
					</section> : null}
					{detail.asks.length > 0 ? <section>
						<h3 className={sectionClass}>{messageRuntime.t("relay.notifications.detail.section.waiting")}</h3>
						<ul className="flex flex-col gap-1">{detail.asks.map(ask => <li key={ask.ask_id} className="rounded-md bg-surface px-3 py-2 text-body-sm text-ink">
							{ask.questions[0]?.question ?? ""}</li>)}</ul>
						<Button variant="ghost" className="mt-1" onClick={() => setAsksOpen(true)}>{messageRuntime.t("relay.notifications.detail.waiting.answer")}</Button>
					</section> : null}
					<section>
						<h3 className={sectionClass}>{messageRuntime.t("relay.notifications.detail.section.from")}</h3>
						<div className="flex flex-col gap-1 rounded-md bg-surface px-3 py-2">
							<span className="truncate text-body text-ink">{sourceName(detail, names)}</span>
							<span className="text-meta text-ink-dim">{messageRuntime.t("relay.notifications.detail.from.state", { state: sessionState(detail.session.live) })}</span>
							{detail.session.id ? <Button variant="ghost" className="self-start !px-0" disabled={writing} onClick={() => void prepareSessionDismiss()}>
								{messageRuntime.t("relay.notifications.detail.from.dismiss_all")}</Button> : null}
						</div>
					</section>
					{detail.projects && detail.projects.length > 0 ? <section>
						<h3 className={sectionClass}>{messageRuntime.t("relay.notifications.detail.section.projects")}</h3>
						<ul className="flex flex-col">{detail.projects.map(project => <li key={project.key} className="flex min-h-11 items-center gap-2 text-body-sm">
							<span className="min-w-0 flex-1 truncate text-ink">{project.name}</span>
							<span className="shrink-0 text-meta text-ink-dim">{project.status}</span>
						</li>)}</ul>
					</section> : null}
					{/* `history` holds earlier revisions (the current one may or may not be
					    included), so the section shows whenever any OTHER revision exists. */}
					{earlier.length > 0 ? <section>
						<h3 className={sectionClass}>{messageRuntime.t("relay.notifications.detail.section.history")}</h3>
						<ul className="flex flex-col gap-1">{earlier.map(entry => <li key={entry.revision} className="flex flex-col py-1">
							<span className="text-body-sm text-ink">{entry.title}</span>
							<span className="text-meta text-ink-dim">{ageText(entry.at, nowMs)}{entry.excerpt ? ` · ${entry.excerpt}` : ""}</span>
						</li>)}</ul>
					</section> : null}
				</article> : null}
			</main>
			<NotificationReceipt text={receipt?.text ?? ""} danger={receipt?.danger} onUndo={receipt?.undo ? () => void undo() : undefined} onDismiss={() => setReceipt(null)} />
			<Sheet open={confirmSession !== null} onClose={() => setConfirmSession(null)}
				title={messageRuntime.t("relay.notifications.confirm.dismiss_session", { count: confirmSession?.count ?? 0, source: detail ? sourceName(detail, names) : "" })}>
				<div className="flex flex-col gap-3 p-3">
					<p className="text-body-sm text-ink-muted">{messageRuntime.t("relay.notifications.confirm.dismiss_session_body")}</p>
					<div className="flex justify-end gap-2">
						<Button variant="ghost" onClick={() => setConfirmSession(null)}>{messageRuntime.t("relay.notifications.confirm.cancel")}</Button>
						<Button variant="danger" disabled={(confirmSession?.count ?? 0) === 0} onClick={() => void dismissSession()}>{messageRuntime.t("relay.notifications.confirm.dismiss")}</Button>
					</div>
				</div>
			</Sheet>
			<AsksSheet open={asksOpen} onClose={() => setAsksOpen(false)} currentSessionId="" />
		</div>
	);
}

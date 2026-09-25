/**
 * The client store: plain React hooks over module state, fed by the two SSE
 * channels (session list, per-session projection). No library — the shape is
 * a list plus a map keyed by durable session identity.
 *
 * SSE discipline: each stream is wrapped with manual backoff (1s doubling to
 * 15s). EventSource's built-in retry is immediate on some server-close
 * shapes, which hammers a dying daemon, so on error we close, wait, and
 * reopen ourselves.
 */
import { useEffect, useState, useSyncExternalStore } from "react";
import type {
	SessionProjection,
	SubagentRow,
	TodoPhase,
	TranscriptEntry,
	SessionSummary,
} from "./types";

function list<T>(value: T[] | undefined | null): T[] {
	return Array.isArray(value) ? value : [];
}

function normalizeTodoPhase(phase: TodoPhase): TodoPhase {
	return { ...phase, items: list(phase.items) };
}

function normalizeSubagent(row: SubagentRow): SubagentRow {
	/* The daemon fills these defaults for rolling upgrades, but SSE is the last
	   trust boundary before React. Preserve the repaint if a hand-built relay or
	   mixed-version proxy omits a newly introduced nested collection: one absent
	   peer list must not unmount the entire phone session. */
	return {
		...row,
		ancestors: list(row.ancestors),
		ancestor_ids: list(row.ancestor_ids),
		child_ids: list(row.child_ids),
		peer_ids: list(row.peer_ids),
		transcript: list<TranscriptEntry>(row.transcript),
		todos: list(row.todos).map(normalizeTodoPhase),
	};
}

export function normalizeProjection(incoming: SessionProjection): SessionProjection {
	return {
		...incoming,
		transcript: list(incoming.transcript),
		todos: list(incoming.todos).map(normalizeTodoPhase),
		subagents: list(incoming.subagents).map(normalizeSubagent),
	};
}

export interface ProjectionSlot {
	projection: SessionProjection | null;
	/** True once at least one snapshot has landed; false = still connecting. */
	connected: boolean;
}


let sessions: SessionSummary[] = [];
let sessionsConnected = false;
/* The pins the user has ASKED FOR and the daemon has not answered for yet,
   held as the value that was asked for. An overlay on the server's list, never
   a rewrite of it, and the distinction is the whole point.

   `applySessionPin` used to write straight into `sessions`, and the list
   PARTITIONS by `pinned`, so pressing pin lifted the row into ★ Pinned before
   the daemon agreed. That reorder is what moved the reader's rows: the browser
   reacts to a list that reorders under it by adjusting the scroll (scroll
   anchoring), measured at +88.0px on a successful pin and −51.0px at 100% /
   −101.5px at 200% root font on an ordinary refusal — against a screen that
   wrote `scrollTop` zero times in 72 instrumented runs.

   A mark is enough for the instant feedback the gesture needs: the row's ★ is
   drawn from it at once, while every DECISION (which section a row renders in,
   and therefore the order the reader sees) stays on the confirmed flag until
   the daemon's own list repaint says so. */
let pinMarks: ReadonlyMap<string, boolean> = new Map();
const projections = new Map<string, ProjectionSlot>();
// useSyncExternalStore requires referentially stable snapshots, including the
// first render before the route's effect has subscribed its SSE stream.
const EMPTY_PROJECTION_SLOT: ProjectionSlot = { projection: null, connected: false };

const listeners = new Set<() => void>();

function emit() {
	for (const l of listeners) l();
}

function subscribe(l: () => void): () => void {
	listeners.add(l);
	return () => listeners.delete(l);
}

/**
 * The bridge into module state. Selector results are the module values
 * themselves (primitives, or Map lookups returning the same object between
 * emissions), so useSyncExternalStore stays stable without a cache.
 */
export function useSessions(): {
	sessions: SessionSummary[];
	connected: boolean;
} {
	const list = useSyncExternalStore(subscribe, () => sessions);
	const connected = useSyncExternalStore(subscribe, () => sessionsConnected);
	return { sessions: list, connected };
}

/** The pins the user has asked for and the daemon has not confirmed yet, keyed
    by session id. A selector of its own, like `useProjection`: it is a slice of
    store state with its own change signal, and a caller that reads the list
    alone (the sections' order) must not be made to re-render by a mark, nor to
    miss one. The Map is replaced rather than mutated, so its identity is what
    `useSyncExternalStore` compares. */
export function usePinMarks(): ReadonlyMap<string, boolean> {
	return useSyncExternalStore(subscribe, () => pinMarks);
}

export function useProjection(sessionId: string): ProjectionSlot {
	return useSyncExternalStore(
		subscribe,
		() =>
			projections.get(sessionId) ?? EMPTY_PROJECTION_SLOT,
	);
}

/* ------------------------------------------------------------------ */
/* Streams                                                             */
/* ------------------------------------------------------------------ */

const BACKOFF_MIN_MS = 1000;
const BACKOFF_MAX_MS = 15000;

export function openEventStream(
	url: string,
	event: string,
	onData: (data: string) => void,
	onOpen: () => void,
	onDisconnect: () => void,
): () => void {
	let closed = false;
	let attempt = 0;
	let es: EventSource | null = null;
	let timer: ReturnType<typeof setTimeout> | null = null;

	const connect = () => {
		if (closed) return;
		const source = new EventSource(url);
		es = source;
		// Closing an EventSource does not revoke callbacks already queued by the
		// browser. A retired source must neither publish nor close its replacement.
		source.addEventListener(event, (e) => {
			if (closed || es !== source) return;
			attempt = 0;
			onData((e as MessageEvent).data as string);
		});
		source.onopen = () => {
			if (closed || es !== source) return;
			attempt = 0;
			onOpen();
		};
		source.onerror = () => {
			if (closed || es !== source) return;
			source.close();
			es = null;
			onDisconnect();
			const delay = Math.min(
				BACKOFF_MAX_MS,
				BACKOFF_MIN_MS * 2 ** attempt++,
			);
			timer = setTimeout(connect, delay);
		};
	};

	connect();
	return () => {
		closed = true;
		if (timer) clearTimeout(timer);
		es?.close();
	};
}

/* ------------------------------------------------------------------ */
/* Session list stream, refcounted                                     */
/* ------------------------------------------------------------------ */

let listStreamClose: (() => void) | null = null;
let listStreamRefs = 0;

export function retainSessionListStream(): () => void {
	listStreamRefs++;
	if (listStreamRefs === 1) {
		listStreamClose = openEventStream(
			"/api/sessions/events",
			"sessions",
			(data) => {
				try {
					const payload = JSON.parse(data) as {
						sessions: SessionSummary[];
					};
					sessions = payload.sessions;
					sessionsConnected = true;
					/* A frame is the daemon's answer for every row it carries, so a mark
					   it AGREES with has been confirmed and is dropped: from here the
					   confirmed flag alone renders the row, in the section it belongs in.
					   A mark the frame contradicts is kept — a repaint older than the
					   press is not an answer to it — and a refused POST is cleared by the
					   screen that sent it. */
					settlePinMarks(payload.sessions);
					emit();
				} catch {
					/* A malformed frame is dropped; the next one repaints. */
				}
			},
			() => {
				sessionsConnected = true;
				emit();
			},
			() => {
				sessionsConnected = false;
				emit();
			},
		);
	}
	return () => {
		listStreamRefs--;
		if (listStreamRefs === 0 && listStreamClose) {
			listStreamClose();
			listStreamClose = null;
		}
	};
}

/* ------------------------------------------------------------------ */
/* Per-session projection streams, refcounted                          */
/* ------------------------------------------------------------------ */

const projectionStreams = new Map<
	string,
	{ close: () => void; refs: number }
>();

export function retainProjectionStream(sessionId: string): () => void {
	const existing = projectionStreams.get(sessionId);
	if (existing) {
		existing.refs++;
	} else {
		projections.set(sessionId, { projection: null, connected: false });
		let awaitingSnapshot = true;
		const close = openEventStream(
			`/api/sessions/${encodeURIComponent(sessionId)}/events`,
			"projection",
			(data) => {
				let incoming: SessionProjection;
				try {
					incoming = normalizeProjection(JSON.parse(data) as SessionProjection);
				} catch {
					return;
				}
				if (incoming.session_id !== sessionId) return;
				const current = projections.get(sessionId);
				/* The daemon reconciles owner epochs while it is alive. Its counter
				   can restart after reconnection, whose first authenticated snapshot
				   is authoritative. Within that fenced source, keep normal ordering. */
				if (
					!awaitingSnapshot && current?.projection &&
					incoming.version < current.projection.version
				) {
					return;
				}
				awaitingSnapshot = false;
				projections.set(sessionId, { projection: incoming, connected: true });
				emit();
			},
			() => {
				// Connection establishment alone does not validate the old rendered
				// result. Only the new source's first snapshot makes it viewable.
				awaitingSnapshot = true;
			},
			() => {
				awaitingSnapshot = true;
				const cur = projections.get(sessionId);
				if (cur) {
					/* Retain the last good projection while making its staleness
					   explicit. Recovery swaps only the connection flag until a fresh
					   frame arrives, so a flaky link never blanks selected detail. */
					projections.set(sessionId, { ...cur, connected: false });
					emit();
				}
			},
		);
		projectionStreams.set(sessionId, { close, refs: 1 });
	}
	return () => {
		const entry = projectionStreams.get(sessionId);
		if (!entry) return;
		entry.refs--;
		if (entry.refs > 0) return;
		entry.close();
		projectionStreams.delete(sessionId);
		projections.delete(sessionId);
		emit();
	};
}

/* ------------------------------------------------------------------ */
/* Seen handshake, optimistic half                                     */
/* ------------------------------------------------------------------ */

/** Optimistic half of the seen handshake (the POST is `markSessionSeen` in
    api.ts): opening a session IS viewing it, so clear `unseen` locally at
    once — back-navigation must never flash a stale `new` mark while the POST
    is in flight. The daemon's next list repaint confirms the clear; if the
    POST fails, that repaint simply restores the mark, which is the honest
    state. */
export function clearSessionUnseen(sessionId: string): void {
	const target = sessions.find((s) => s.session_id === sessionId);
	if (!target || !target.unseen) return;
	sessions = sessions.map((s) =>
		s.session_id === sessionId ? { ...s, unseen: false } : s,
	);
	emit();
}

/** Optimistic half of the pin handshake (the POST is `setSessionPin` in
    api.ts): the row must show its ★ the instant the user acts, or a tap that
    visibly changed nothing reads as a fault. It is a MARK, not a move — see
    the note on `pinMarks` for why the row must not change section until the
    daemon's next list repaint (which `set_pins` already woke) confirms it. */
export function applySessionPin(sessionId: string, pinned: boolean): void {
	if (pinMarks.get(sessionId) === pinned) return;
	pinMarks = new Map(pinMarks).set(sessionId, pinned);
	emit();
}

/** Drops the mark for a row the server has answered for and could NOT pin —
    the POST's own failure path. The row never moved, so there is nothing to
    put back; only the ★ this mark drew has to go. */
export function clearSessionPinMark(sessionId: string): void {
	if (!pinMarks.has(sessionId)) return;
	const next = new Map(pinMarks);
	next.delete(sessionId);
	pinMarks = next;
	emit();
}

/** Confirmation sweep for one list frame: a mark the frame agrees with has
    been confirmed and is retired, so the row is rendered from the daemon's own
    value from that commit on. */
function settlePinMarks(frame: SessionSummary[]): void {
	if (pinMarks.size === 0) return;
	const aged = new Set<string>();
	for (const row of frame) {
		const mark = pinMarks.get(row.session_id);
		if (mark !== undefined && mark === Boolean(row.pinned)) {
			aged.add(row.session_id);
		}
	}
	if (aged.size === 0) return;
	const next = new Map(pinMarks);
	for (const id of aged) next.delete(id);
	pinMarks = next;
}

/* ------------------------------------------------------------------ */
/* Tab title aggregate                                                 */
/* ------------------------------------------------------------------ */

/* The browser tab title is the one attention signal visible from the
   phone's app switcher, so it must track the list even while a session
   view (which renders nothing of the list) is on screen. A permanent
   module subscription instead of a hook: a number in the chrome must not
   re-render any component. n = sessions with unseen || needs_attention —
   finished reading matter plus decisions, the two states that want the
   user (spec §4). */
function syncTabTitle(): void {
	/* The store also loads in non-DOM contexts (the node-env unit suite). */
	if (typeof document === "undefined") return;
	const n = sessions.filter((s) => s.unseen || s.needs_attention).length;
	const title = n > 0 ? `(${n}) local operator` : "local operator";
	if (document.title !== title) document.title = title;
}

subscribe(syncTabTitle);
syncTabTitle();

/* ------------------------------------------------------------------ */
/* Composer drafts, per pid, in localStorage                           */
/* ------------------------------------------------------------------ */

const DRAFT_PREFIX = "lo-mobile-draft:";

export function getDraft(sessionId: string): string {
	return localStorage.getItem(DRAFT_PREFIX + sessionId) ?? "";
}

export function setDraft(sessionId: string, text: string): void {
	if (text) {
		localStorage.setItem(DRAFT_PREFIX + sessionId, text);
	} else {
		localStorage.removeItem(DRAFT_PREFIX + sessionId);
	}
}

/**
 * Draft hook: initialises from localStorage, writes through on change.
 * Drafts survive navigation away and back, which is the phone case — the
 * user answers a message mid-compose and returns.
 */
export function useDraft(sessionId: string): [string, (t: string) => void] {
	const [text, setText] = useState(() => getDraft(sessionId));
	useEffect(() => {
		setText(getDraft(sessionId));
	}, [sessionId]);
	return [
		text,
		(t: string) => {
			setText(t);
			setDraft(sessionId, t);
		},
	];
}

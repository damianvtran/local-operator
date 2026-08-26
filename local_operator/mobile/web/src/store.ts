/**
 * The client store: plain React hooks over module state, fed by the two SSE
 * channels (session list, per-session projection). No library — the shape is
 * a list plus a map keyed by pid.
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
const projections = new Map<string, ProjectionSlot>();

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

export function useProjection(sessionId: string): ProjectionSlot {
	return useSyncExternalStore(
		subscribe,
		() =>
			projections.get(sessionId) ?? { projection: null, connected: false },
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
		es = new EventSource(url);
		es.addEventListener(event, (e) => {
			attempt = 0;
			onData((e as MessageEvent).data as string);
		});
		es.onopen = () => {
			attempt = 0;
			onOpen();
		};
		es.onerror = () => {
			es?.close();
			es = null;
			if (closed) return;
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
				const current = projections.get(sessionId);
				/* Drop stale repaints: the daemon's epoch only moves forward. */
				if (
					current?.projection &&
					incoming.version < current.projection.version
				) {
					return;
				}
				projections.set(sessionId, { projection: incoming, connected: true });
				emit();
			},
			() => {
				const cur = projections.get(sessionId);
				if (cur) {
					projections.set(sessionId, { ...cur, connected: true });
					emit();
				}
			},
			() => {
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

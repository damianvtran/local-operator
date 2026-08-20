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
import type { SessionProjection, SessionSummary } from "./types";

export interface ProjectionSlot {
	projection: SessionProjection | null;
	/** True once at least one snapshot has landed; false = still connecting. */
	connected: boolean;
}

interface StoreState {
	sessions: SessionSummary[];
	sessionsConnected: boolean;
	projections: Map<number, ProjectionSlot>;
}

let sessions: SessionSummary[] = [];
let sessionsConnected = false;
const projections = new Map<number, ProjectionSlot>();

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

export function useProjection(pid: number): ProjectionSlot {
	return useSyncExternalStore(
		subscribe,
		() =>
			projections.get(pid) ?? { projection: null, connected: false },
	);
}

/* ------------------------------------------------------------------ */
/* Streams                                                             */
/* ------------------------------------------------------------------ */

const BACKOFF_MIN_MS = 1000;
const BACKOFF_MAX_MS = 15000;

function openEventStream(
	url: string,
	event: string,
	onData: (data: string) => void,
	onOpen: () => void,
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
	number,
	{ close: () => void; refs: number }
>();

export function retainProjectionStream(pid: number): () => void {
	const existing = projectionStreams.get(pid);
	if (existing) {
		existing.refs++;
	} else {
		projections.set(pid, { projection: null, connected: false });
		const close = openEventStream(
			`/api/sessions/${pid}/events`,
			"projection",
			(data) => {
				let incoming: SessionProjection;
				try {
					incoming = JSON.parse(data) as SessionProjection;
				} catch {
					return;
				}
				const current = projections.get(pid);
				/* Drop stale repaints: the daemon's epoch only moves forward. */
				if (
					current?.projection &&
					incoming.version < current.projection.version
				) {
					return;
				}
				projections.set(pid, { projection: incoming, connected: true });
				emit();
			},
			() => {
				const cur = projections.get(pid);
				if (cur) {
					projections.set(pid, { ...cur, connected: true });
					emit();
				}
			},
		);
		projectionStreams.set(pid, { close, refs: 1 });
	}
	return () => {
		const entry = projectionStreams.get(pid);
		if (!entry) return;
		entry.refs--;
		if (entry.refs > 0) return;
		entry.close();
		projectionStreams.delete(pid);
		projections.delete(pid);
		emit();
	};
}

/* ------------------------------------------------------------------ */
/* Composer drafts, per pid, in localStorage                           */
/* ------------------------------------------------------------------ */

const DRAFT_PREFIX = "lo-mobile-draft:";

export function getDraft(pid: number): string {
	return localStorage.getItem(DRAFT_PREFIX + pid) ?? "";
}

export function setDraft(pid: number, text: string): void {
	if (text) {
		localStorage.setItem(DRAFT_PREFIX + pid, text);
	} else {
		localStorage.removeItem(DRAFT_PREFIX + pid);
	}
}

/**
 * Draft hook: initialises from localStorage, writes through on change.
 * Drafts survive navigation away and back, which is the phone case — the
 * user answers a message mid-compose and returns.
 */
export function useDraft(pid: number): [string, (t: string) => void] {
	const [text, setText] = useState(() => getDraft(pid));
	useEffect(() => {
		setText(getDraft(pid));
	}, [pid]);
	return [
		text,
		(t: string) => {
			setText(t);
			setDraft(pid, t);
		},
	];
}

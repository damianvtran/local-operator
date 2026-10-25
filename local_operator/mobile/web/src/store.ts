/**
 * The client store: plain React hooks over module state, fed by the two SSE
 * channels (session list, per-session projection). No library — the shape is
 * a list plus a map keyed by durable session identity.
 *
 * SSE discipline: fetch-streamed snapshots with invisible lease rotation,
 * a byte-silence watchdog, bounded retries and delayed outage reporting.
 */
import { useEffect, useState, useSyncExternalStore } from "react";
import { handleUnauthorized } from "./api";
import { supplementsFromWire } from "./lib/supplement";
import { createMessageRuntime, directionFor, isResolvedLocale, type LocaleSnapshot } from "./i18n/runtime";
import { createWireLoader } from "./i18n/wire-cache";
import { BUNDLED_OWNED } from "./i18n/bundled.gen";
import type {
	Capabilities,
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
	const projection: SessionProjection = {
		...incoming,
		transcript: list(incoming.transcript),
		todos: list(incoming.todos).map(normalizeTodoPhase),
		subagents: list(incoming.subagents).map(normalizeSubagent),
	};
	/* Turn supplements: the sibling map is re-read through its ONE reader at
	   this last trust boundary, like every nested shape beside it — a beat
	   whose anchor/state does not parse is dropped rather than rendered from
	   a guess. ABSENT stays absent: the daemon emits the key only when it
	   carries beats, and every renderer reads a missing key and an empty map
	   the same way (the distinction is deliberately not laundered here). */
	if (incoming.supplements !== undefined) {
		projection.supplements = supplementsFromWire(incoming.supplements);
	}
	return projection;
}

export interface ProjectionSlot {
	projection: SessionProjection | null;
	/** True once at least one snapshot has landed; false = still connecting. */
	connected: boolean;
}


let sessions: SessionSummary[] = [];
let sessionsConnected = false;
/* Monotonic count of ACCEPTED list frames. Consumers that need "has a newer
   authoritative frame arrived?" read this, never the `sessions` array's
   identity: identity is only a change signal for `useSyncExternalStore`
   itself, and a caller may legally hand back a fresh array every render (the
   screen tests do exactly that). Counting HERE — in the one handler that
   accepts a frame — is what keeps the counter tied to frames rather than to
   renders. */
let sessionsRevision = 0;
/* Server capabilities off the SAME list frames that carry the sessions (see
   the daemon's ``_list_frame``): one answer, two transports. ``null`` means an
   older daemon omitted the key — read it as "no capabilities", never as
   "unchanged" — and the voice mic's rule for it is the rule for
   ``available: false``: hide. */
let capabilities: Capabilities | null = null;
/* Backend preference and actual UI copy language are different facts. Until
   C bundles own copy, the SPA remains English/LTR under every backend frame.
   No OS/device/storage guess is allowed to impersonate backend authority. */
let localeSnapshot: LocaleSnapshot = Object.freeze({
	backendLocale: null, uiLocale: "en", direction: "ltr", revision: 0,
});
export const messageRuntime = createMessageRuntime(() => localeSnapshot, () => publishLocaleSnapshot());
export const wireCatalogueLoader = createWireLoader({
	context: () => ({ origin: typeof location === "object" ? location.origin : "", locale: localeSnapshot.backendLocale }),
	capabilities: () => capabilities,
	activate: (namespace, locale, messages) => { messageRuntime.registerWire(namespace, locale, messages); },
	deactivate: (namespace, locale) => { messageRuntime.unregisterWire(namespace, locale); },
});
export function initializeLocaleMetadata(): void {
	if (typeof document === "object") {
		document.documentElement.lang = localeSnapshot.uiLocale;
		document.documentElement.dir = localeSnapshot.direction;
	}
}
function publishLocaleSnapshot(backendLocale = localeSnapshot.backendLocale): void {
	const uiLocale = messageRuntime.uiLocaleFor(backendLocale);
	localeSnapshot = Object.freeze({ backendLocale, uiLocale, direction: directionFor(uiLocale), revision: localeSnapshot.revision + 1 });
	wireCatalogueLoader.invalidateContext();
	initializeLocaleMetadata();
	emit(); // one existing notification channel, including same-locale registration
}
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
/* THE ASK SIGNATURE'S REVISION (design §5.3). The aggregate asks sheet reads
   `GET /api/asks` on demand — an HTTP route, deliberately not a feed — so it
   needs a signal for WHEN re-reading is worth doing. The list frame carries
   every session's `asks_open`, and a change in that signature is exactly "a
   queued ask appeared, settled, or changed hands" — the events a reader of the
   sheet would want to see. A counter rather than a boolean so two changes
   between renders cannot collapse into one; the signature compares the count
   per session, so an unrelated repaint (a streaming session bumping its
   activity 30x/s) does not move it.

   WHY NOT EVERY LIST FRAME: a list frame is emitted on every projection push,
   which is ~30/s while a turn streams. Re-reading the whole aggregate on each
   would be a poll disguised as an event. */
let asksRevision = 0;
let asksSignature = "";
const projections = new Map<string, ProjectionSlot>();// useSyncExternalStore requires referentially stable snapshots, including the
// first render before the route's effect has subscribed its SSE stream.
const EMPTY_PROJECTION_SLOT: ProjectionSlot = { projection: null, connected: false };

const listeners = new Set<() => void>();
let streamEmitPending = false;
/** Both streams can update in one paint. Gestures still emit synchronously. */
function emitStream() {
	if (streamEmitPending) return;
	streamEmitPending = true;
	const flush = () => { streamEmitPending = false; emit(); };
	if (typeof requestAnimationFrame === "function") requestAnimationFrame(flush);
	else setTimeout(flush, 16);
}


/* THE SPA'S OWN ENGLISH COPY, registered before any consumer renders the relay
   (the RFC's "bundle a bootstrap set", §2.1/§3.3). This is what makes the
   owned-copy language an ACTUAL rendered fact rather than a claim: `uiLocaleFor`
   only returns a non-English UI locale when SOME bundle carries the WHOLE
   English key set plus the resolved locale, so with English-only copy the SPA
   stays EN/LTR under every backend frame — `lang`/`dir` describe the copy on
   screen, not the backend preference (that stays on `useLanguage`). The bundle
   is generated from the explicit owned namespaces (§ bundled.gen.ts); no wire
   prose is bundled, so this cannot smuggle a served namespace into the client.
   Registration runs through the ONE store channel — `registerBundle` ->
   `changed()` -> `publishLocaleSnapshot` — so it repaints and re-labels the
   document instead of creating a second locale truth. Placed AFTER the
   listener set is declared: an init-time registration notifies that set, so
   registering above it would emit into a `listeners` binding still in its
   temporal dead zone and abort module evaluation. */
for (const [namespace, messages] of Object.entries(BUNDLED_OWNED)) {
	messageRuntime.registerBundle(namespace, "en", { ...messages });
}

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
	/** Every accepted list frame increments this once; it never moves on a
	    component render. A write that compares it before and after can tell
	    whether a newer authoritative frame landed meanwhile. */
	revision: number;
} {
	const list = useSyncExternalStore(subscribe, () => sessions);
	const connected = useSyncExternalStore(subscribe, () => sessionsConnected);
	const revision = useSyncExternalStore(subscribe, () => sessionsRevision);
	return { sessions: list, connected, revision };
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

/** The mark-all receipt, as MODULE state rather than component state.

    Two things the component-local version could not answer (design D3, UX
    U2/U8): a glance into a conversation and back (``#/`` -> ``#/s/<id>`` ->
    ``#/``) unmounted the screen and erased the receipt -- the only explanation
    of a partial or failed clear -- and the receipt never expired, so it
    outlived the gesture indefinitely. Module state carries it across a route
    change; the TTL bounds it; ``null`` clears it (the next press, or the
    dismiss control on the line itself).

    Same shape as ``pinMarks`` above and for the same reason: this is a slice
    of state with its own change signal, so a caller that only reads the list
    must not be made to re-render by it. */
export interface MarkNotice {
	text: string;
	danger: boolean;
}

let markNotice: MarkNotice | null = null;

/** How long a receipt survives once it is ON SCREEN, without the reader
    dismissing it -- long enough to read after a glance away, short enough that
    it does not outlive the gesture it describes (UX round 1, U8). */
export const MARK_NOTICE_TTL_MS = 12000;

export function useMarkNotice(): MarkNotice | null {
	const notice = useSyncExternalStore(subscribe, () => markNotice);
	/* THE CLOCK RUNS ONLY WHILE THE RECEIPT IS BEING LOOKED AT (UX round 2, U9).
	   A store-owned timer armed at publish kept ticking through a route change,
	   so a glance into a conversation longer than the TTL still erased the
	   receipt on return -- and the receipt is the only explanation of a partial
	   or failed clear (the same U2 reason it outlives the route change at all).
	   Owning the timer HERE, and clearing it on unmount, ties the window to the
	   reader's own viewing time: leaving the screen pauses the clock, coming
	   back restarts it, and a fresh receipt gets its own full window. */
	useEffect(() => {
		if (notice === null) return;
		const timer = setTimeout(() => publishMarkNotice(null), MARK_NOTICE_TTL_MS);
		return () => clearTimeout(timer);
	}, [notice]);
	return notice;
}

/** Publish (``notice``) or clear (``null``) the receipt. No timer lives here:
    the consuming hook owns the countdown, so an unmounted screen cannot expire
    a receipt nobody is looking at (U9), and ``null`` is also how the dismiss
    control and every new press clear the line. */
export function publishMarkNotice(notice: MarkNotice | null): void {
	markNotice = notice;
	emit();
}

/** The daemon's capability answer (``capabilities`` on the list payload). */
export function useCapabilities(): Capabilities | null {
	return useSyncExternalStore(subscribe, () => capabilities);
}

export function useLanguage(): string | null {
	return useSyncExternalStore(subscribe, () => localeSnapshot.backendLocale);
}
export function useLocaleSnapshot(): LocaleSnapshot {
	return useSyncExternalStore(subscribe, () => localeSnapshot);
}

export function useProjection(sessionId: string): ProjectionSlot {
	return useSyncExternalStore(
		subscribe,
		() =>
			projections.get(sessionId) ?? EMPTY_PROJECTION_SLOT,
	);
}

/** A counter that moves whenever the outstanding-ask population changes.

    The aggregate asks sheet subscribes to it so a new ask (or one settled on
    another surface) re-reads `GET /api/asks` without anybody pressing
    refresh — and without re-reading on every unrelated repaint. */
export function useAsksRevision(): number {
	return useSyncExternalStore(subscribe, () => asksRevision);
}

/** Fold one list frame's per-session ask counts into a signature.

    PURE and exported so the rule is testable on its own (agent review round 1,
    R4): the sheet mocks the hook, so nothing exercised the part that decides
    WHEN a re-read is worth doing. The rule is threefold and each clause is a
    deliberate answer: per-SESSION counts (so an unrelated repaint of a session
    streaming at 30/s cannot move it), ORDER-INSENSITIVE (the daemon's row order
    is its own rank and a re-rank is not new information), and a runtime that
    does not publish asks contributes NOTHING rather than a zero (absence is the
    capability proxy, §4 — a zero would be a claim it cannot make).

    A TIMED-OUT ASK IS NOT IN THE COUNT (#2388), so a session holding only
    lapsed asks contributes nothing, and dismissing or late-answering one does
    not move this signature. An open asks sheet learns that from the
    conversation's own projection rows instead (`askRowsSignature`). */
export function asksPopulationSignature(rows: SessionSummary[]): string {
	const parts: string[] = [];
	for (const row of rows) {
		const count = row?.asks_open;
		if (typeof count !== "number" || count <= 0) continue;
		parts.push(`${row.session_id}:${count}`);
	}
	parts.sort();
	return parts.join(",");
}

function publishAsksRevision(rows: SessionSummary[]): void {
	const next = asksPopulationSignature(rows);
	if (next === asksSignature) return;
	asksSignature = next;
	asksRevision += 1;
}

/* ------------------------------------------------------------------ */
/* Streams                                                             */
/* ------------------------------------------------------------------ */

const BACKOFF_MIN_MS = 250;
const BACKOFF_MAX_MS = 10000;
const SILENCE_MS = 35000;
const OUTAGE_GRACE_MS = 2500;
// A seed is sent even by a leg that immediately dies. Require continuous
// residency before calling it recovered; shorter than the outage grace so a
// quick working reconnect never paints a false outage while proving itself.
const STABLE_STREAM_MS = 2000;

/** Incremental SSE framing, including comments and CR/LF/CRLF boundaries.
 * Resynchronization uses full snapshots, never Last-Event-ID replay. */
export function sseFramer(deliver: (event: string, data: string) => void) {
	let line = "", event = "", data: string[] = [], afterCR = false;
	const finish = () => {
		if (!line) {
			if (data.length) deliver(event || "message", data.join("\n"));
			event = ""; data = [];
		} else if (!line.startsWith(":")) {
			const colon = line.indexOf(":");
			const key = colon < 0 ? line : line.slice(0, colon);
			let value = colon < 0 ? "" : line.slice(colon + 1);
			if (value.startsWith(" ")) value = value.slice(1);
			if (key === "event") event = value;
			if (key === "data") data.push(value);
		}
		line = "";
	};
	return (chunk: string) => {
		for (const char of chunk) {
			if (afterCR && char === "\n") { afterCR = false; continue; }
			afterCR = char === "\r";
			if (char === "\r" || char === "\n") finish();
			else line += char;
		}
	};
}

/** Fetch rather than EventSource: the tunnel's clean 60 s lease rotation is
 * routine, comments prove liveness, and HTTP 401/503 must remain observable.
 * The last good view survives brief failures; only a sustained outage reaches
 * onDisconnect. Each onOpen establishes a NEW snapshot/version fence.
 * onData returns false for an invalid/stale snapshot; its connected argument
 * stays false during a reported outage. onRecovered restores connectivity
 * only after an accepted snapshot and a continuously open stability window. */
export function openEventStream(
	url: string, event: string, onData: (data: string, connected: boolean) => boolean | void,
	onOpen: () => void, onDisconnect: () => void,
	onRecovered: () => void = () => {},
): () => void {
	let closed = false, failures = 0, generation = 0, lastByte = Date.now();
	let controller: AbortController | null = null;
	let retry: ReturnType<typeof setTimeout> | null = null;
	let watchdog: ReturnType<typeof setTimeout> | null = null;
	let outage: ReturnType<typeof setTimeout> | null = null;
	let reported = false, inBackoff = false;
	let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
	let stableTimer: ReturnType<typeof setTimeout> | null = null;
	const clearWatchdog = () => { if (watchdog !== null) clearTimeout(watchdog); watchdog = null; };
	const unhealthy = () => {
		if (reported || outage !== null || closed) return;
		outage = setTimeout(() => { outage = null; reported = true; onDisconnect(); }, OUTAGE_GRACE_MS);
	};
	const healthy = () => {
		if (outage !== null) clearTimeout(outage);
		outage = null;
		const wasReported = reported;
		reported = false; failures = 0;
		if (wasReported) onRecovered();
	};
	const schedule = (delay: number, backoff: boolean) => {
		if (closed) return;
		inBackoff = backoff;
		retry = setTimeout(() => { retry = null; inBackoff = false; void connect(); }, delay);
	};
	const stopLeg = () => {
		controller?.abort();
		void reader?.cancel().catch(() => {});
		reader = null;
		clearWatchdog();
		if (stableTimer !== null) clearTimeout(stableTimer);
		stableTimer = null;
	};
	const connect = async () => {
		if (closed) return;
		const id = ++generation;
		stopLeg();
		const ctl = new AbortController(); controller = ctl;
		const current = () => !closed && generation === id;
		let opened: number | null = null, eof = false, stalled = false;
		let retryAfter: number | undefined;
		let acceptedSnapshot = false;
		const recoverIfStable = () => {
			if (current() && acceptedSnapshot && opened !== null && Date.now() - opened >= STABLE_STREAM_MS) healthy();
		};
		const arm = () => {
			clearWatchdog();
			watchdog = setTimeout(() => {
				if (!current()) return;
				stalled = true; unhealthy(); stopLeg();
			}, SILENCE_MS);
		};
		// Covers a fetch whose headers never arrive as well as a stalled body.
		lastByte = Date.now(); arm();
		try {
			const response = await fetch(url, {
				credentials: "same-origin", cache: "no-store",
				headers: { Accept: "text/event-stream" }, signal: ctl.signal,
			});
			if (!current()) { void response.body?.cancel(); return; }
			if (response.status === 401) {
				dispose(); handleUnauthorized(); return;
			}
			if (!response.ok || !response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
				if (response.status === 503) {
					const header = response.headers.get("retry-after");
					if (header) {
						const value = /^\d+(\.\d+)?$/.test(header) ? Number(header) * 1000 : Date.parse(header) - Date.now();
						if (Number.isFinite(value)) retryAfter = Math.max(250, Math.min(30000, value));
					}
				}
				void response.body?.cancel(); throw response;
			}
			opened = Date.now(); onOpen();
			stableTimer = setTimeout(recoverIfStable, STABLE_STREAM_MS);
			const localReader = response.body.getReader(); reader = localReader;
			const decoder = new TextDecoder();
			const parse = sseFramer((name, data) => {
				if (!current() || name !== event) return;
				// Accept every snapshot BEFORE batching renders. Last-arrival raw
				// coalescing hid version 9 behind version 8 before the fence could
				// see it. The daemon already caps projection/list rates at 10/2.5Hz.
				if (onData(data, !reported) !== false) acceptedSnapshot = true;
				recoverIfStable();
			});
			for (;;) {
				const { done, value } = await localReader.read();
				if (!current()) return;
				if (done) { eof = true; break; }
				if (value?.length) {
					lastByte = Date.now(); arm();
					parse(decoder.decode(value, { stream: true }));
				}
			}
		} catch { /* Network failure follows the bounded retry policy below. */ }
		finally {
			if (current()) {
				clearWatchdog(); reader = null;
				if (stableTimer !== null) clearTimeout(stableTimer);
				stableTimer = null;
			}
		}
		if (!current()) return;
		if (!stalled && eof && acceptedSnapshot && opened !== null && Date.now() - opened >= 5000) {
			// Security lease unchanged: reauthorize with a fresh request, no UI blip.
			schedule(Math.random() * 250, false); return;
		}
		unhealthy();
		if (stalled) { schedule(0, false); return; }
		const delay = Math.min(BACKOFF_MAX_MS, BACKOFF_MIN_MS * 2 ** Math.min(failures++, 10));
		schedule(retryAfter ?? delay * (0.5 + Math.random() * 0.5), true);
	};
	const wake = () => {
		if (closed || document.visibilityState === "hidden") return;
		if (inBackoff || Date.now() - lastByte >= 20000) {
			if (retry !== null) clearTimeout(retry);
			retry = null; inBackoff = false;
			unhealthy(); void connect();
		}
	};
	const hasDOM = typeof window !== "undefined" && typeof document !== "undefined";
	if (hasDOM) {
		window.addEventListener("pageshow", wake); window.addEventListener("online", wake);
		document.addEventListener("visibilitychange", wake);
	}
	const dispose = () => {
		closed = true; generation++;
		if (retry !== null) clearTimeout(retry);
		if (outage !== null) clearTimeout(outage);
		stopLeg();
		if (hasDOM) {
			window.removeEventListener("pageshow", wake); window.removeEventListener("online", wake);
			document.removeEventListener("visibilitychange", wake);
		}
	};
	void connect();
	return dispose;
}

/* ------------------------------------------------------------------ */
/* Session list stream, refcounted                                     */
/* ------------------------------------------------------------------ */

let listStreamClose: (() => void) | null = null;
let listStreamRefs = 0;
let listStreamLinger: ReturnType<typeof setTimeout> | null = null;

export function retainSessionListStream(): () => void {
	listStreamRefs++;
	if (listStreamLinger !== null) clearTimeout(listStreamLinger);
	listStreamLinger = null;
	if (listStreamClose === null) {
		listStreamClose = openEventStream(
			"/api/sessions/events",
			"sessions",
			(data, connected) => {
				try {
					const payload = JSON.parse(data) as {
						sessions: SessionSummary[];
						capabilities?: Capabilities;
					};
					if (!Array.isArray(payload.sessions)) return false;
					sessions = payload.sessions;
					sessionsRevision++;
					capabilities = payload.capabilities ?? null;
					wireCatalogueLoader.invalidateCapabilities();
					sessionsConnected = connected;
					publishAsksRevision(payload.sessions);
					/* A frame is the daemon's answer for every row it carries, so a mark
					   it AGREES with has been confirmed and is dropped: from here the
					   confirmed flag alone renders the row, in the section it belongs in.
					   A mark the frame contradicts is kept — a repaint older than the
					   press is not an answer to it — and a refused POST is cleared by the
					   screen that sent it. */
					settlePinMarks(payload.sessions);
					emitStream();
					return true;
				} catch {
					/* A malformed frame is dropped; the next one repaints. */
					return false;
				}
			},
			() => {}, // HTTP headers cannot certify a retained list as current.
			() => {
				sessionsConnected = false;
				emitStream();
			},
			() => { sessionsConnected = true; emitStream(); },
		);
	}
	return () => {
		listStreamRefs--;
		if (listStreamRefs === 0 && listStreamClose) {
			// React destroys before creating the next route. A brief linger makes
			// list/conversation navigation reuse the same authenticated stream.
			listStreamLinger = setTimeout(() => {
				listStreamLinger = null;
				if (listStreamRefs) return;
				listStreamClose?.(); listStreamClose = null;
			}, 10000);
		}
	};
}

/* ------------------------------------------------------------------ */
/* Config stream (resolved language), refcounted                       */
/* ------------------------------------------------------------------ */

const CONFIG_STREAM_PATH = "/api/config/events";

let configStreamClose: (() => void) | null = null;
let configStreamCancel: (() => void) | null = null;
let configStreamRefs = 0;

function applyConfigFrame(data: string): void {
	try {
		const payload = JSON.parse(data) as {
			changed_keys?: string[];
			language?: string;
		};
		if (!isResolvedLocale(payload.language)) return;
		publishLocaleSnapshot(payload.language);
	} catch {
		/* A malformed frame is dropped; the next one repaints. */
	}
}

/* Probe ONCE whether this daemon serves the config stream.

   An older daemon answers 404 on ``/api/config/events``. Keep capability
   negotiation separate from transport retry: subscribing blind would retry a
   route that will never exist. One probe fetch settles it: headers
   land immediately even for a live event-stream (``fetch`` resolves on them;
   the body is then cancelled).

   ONLY 404 is PERMANENT absence (CI round 1, MINOR-3); 401 is a terminal
   authentication transition. Every other non-ok answer — a 5xx behind a restarting proxy, a captive portal, a tunnel that
   is not up yet — is TRANSIENT and branches to the retry path, so it
   self-heals instead of disabling live language for the whole load; a
   NETWORK failure is the same retry. A daemon that disappears AFTER a
   successful probe keeps the stream wrapper's normal reconnect behaviour;
   the probe covers the mixed-version load. */
async function configEndpointPresence(): Promise<"present" | "absent" | "retry" | "unauthorized"> {
	try {
		const res = await fetch(CONFIG_STREAM_PATH, {
			headers: { Accept: "text/event-stream" },
		});
		const isStream =
			(res.headers.get("content-type") ?? "").includes("text/event-stream");
		try {
			await res.body?.cancel();
		} catch {
			/* dropping the probe's body is best-effort */
		}
		if (res.status === 401) {
			// A pre-stream capability probe has the same dead-cookie rule as
			// the stream itself; never retry it as a transient capability error.
			handleUnauthorized();
			return "unauthorized";
		}
		if (res.ok && isStream) return "present";
		return res.status === 404 ? "absent" : "retry";
	} catch {
		return "retry";
	}
}

export function retainConfigStream(): () => void {
	configStreamRefs++;
	if (configStreamRefs === 1) {
		let disposed = false;
		let attempt = 0;
		let timer: ReturnType<typeof setTimeout> | null = null;
		const start = async () => {
			if (disposed) return;
			const presence = await configEndpointPresence();
			if (disposed) return;
			if (presence === "retry") {
				const delay = Math.min(BACKOFF_MAX_MS, BACKOFF_MIN_MS * 2 ** attempt++);
				timer = setTimeout(start, delay);
				return;
			}
			if (presence === "absent" || presence === "unauthorized") return; // terminal, no retries
			configStreamClose = openEventStream(
				CONFIG_STREAM_PATH,
				"config",
				applyConfigFrame,
				() => {},
				() => {},
			);
		};
		configStreamCancel = () => {
			disposed = true;
			if (timer) clearTimeout(timer);
			configStreamClose?.();
			configStreamClose = null;
		};
		void start();
	}
	return () => {
		configStreamRefs--;
		if (configStreamRefs === 0) {
			configStreamCancel?.();
			configStreamCancel = null;
		}
	};
}

/* ------------------------------------------------------------------ */
/* Per-session projection streams, refcounted                          */
/* ------------------------------------------------------------------ */

const projectionStreams = new Map<
	string,
	{ close: () => void; refs: number; linger: ReturnType<typeof setTimeout> | null }
>();

export function retainProjectionStream(sessionId: string): () => void {
	const existing = projectionStreams.get(sessionId);
	if (existing) {
		existing.refs++;
		if (existing.linger !== null) clearTimeout(existing.linger);
		existing.linger = null;
	} else {
		projections.set(sessionId, { projection: null, connected: false });
		let awaitingSnapshot = true;
		const close = openEventStream(
			`/api/sessions/${encodeURIComponent(sessionId)}/events`,
			"projection",
			(data, connected) => {
				let incoming: SessionProjection;
				try {
					incoming = normalizeProjection(JSON.parse(data) as SessionProjection);
				} catch {
					return false;
				}
				if (incoming.session_id !== sessionId) return false;
				const current = projections.get(sessionId);
				/* The daemon reconciles owner epochs while it is alive. Its counter
				   can restart after reconnection, whose first authenticated snapshot
				   is authoritative. Within that fenced source, keep normal ordering. */
				if (
					!awaitingSnapshot && current?.projection &&
					incoming.version < current.projection.version
				) {
					return false;
				}
				awaitingSnapshot = false;
				projections.set(sessionId, { projection: incoming, connected });
				emitStream();
				return true;
			},
			() => {
				// Connection establishment alone does not validate the old rendered
				// result. Only the new source's first snapshot makes it viewable.
				awaitingSnapshot = true;
			},
			() => {
				// Outage reporting can land during a new leg's stability window.
				// Only onOpen resets ordering; a status change must not reopen it.
				const cur = projections.get(sessionId);
				if (cur) {
					/* Retain the last good projection while making its staleness
					   explicit. Recovery swaps only the connection flag until a fresh
					   frame arrives, so a flaky link never blanks selected detail. */
					projections.set(sessionId, { ...cur, connected: false });
					emitStream();
				}
			},
			() => {
				const current = projections.get(sessionId);
				if (current?.projection) {
					projections.set(sessionId, { ...current, connected: true });
					emitStream();
				}
			},
		);
		projectionStreams.set(sessionId, { close, refs: 1, linger: null });
	}
	return () => {
		const entry = projectionStreams.get(sessionId);
		if (!entry) return;
		entry.refs--;
		if (entry.refs > 0) return;
		// A glance at the list and back must not tear down the viewer.
		entry.linger = setTimeout(() => {
			entry.linger = null;
			if (entry.refs > 0) return;
			entry.close();
			projectionStreams.delete(sessionId);
			projections.delete(sessionId);
			emitStream();
		}, 5000);
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
/* Tab title                                                           */
/* ------------------------------------------------------------------ */

/* Two writers decide the title, and they COMPOSE rather than race.

   THE LIST owns the attention aggregate: n = sessions with unseen ||
   needs_attention — finished reading matter plus decisions, the two states that
   want the user (spec §4). It is computed from a module subscription instead of
   a hook because a number in the chrome must not re-render any component.

   A SESSION OR AGENT ROUTE owns its own context (U4, batch 2). A phone's task
   switcher reads `document.title`, and a session route wearing the list's
   `(3) local operator` described the wrong screen — the audit's exact finding.
   The route writer sets `routeTitle`; the list writer keeps applying the count
   whenever no route overrides it, so the count returns the moment the reader is
   back on the list while a conversation route names the conversation. */
let listTitle = "local operator";
let routeTitle: string | null = null;

function applyTabTitle(): void {
	/* The store also loads in non-DOM contexts (the node-env unit suite). */
	if (typeof document === "undefined") return;
	const title = routeTitle ?? listTitle;
	if (document.title !== title) document.title = title;
}

function syncTabTitle(): void {
	const n = sessions.filter((s) => s.unseen || s.needs_attention).length;
	listTitle = n > 0 ? `(${n}) local operator` : "local operator";
	applyTabTitle();
}

subscribe(syncTabTitle);
syncTabTitle();

/** The route's own title while it is mounted; `null` releases the list's.
    A hook rather than a module call so unmount restores the list title without
    the screen having to remember what it was. */
export function useRouteTitle(title: string | null): void {
	useEffect(() => {
		routeTitle = title;
		applyTabTitle();
		return () => {
			routeTitle = null;
			applyTabTitle();
		};
	}, [title]);
}

/* ------------------------------------------------------------------ */
/* Composer drafts, per pid, in localStorage                           */
/* ------------------------------------------------------------------ */

const DRAFT_PREFIX = "lo-mobile-draft:";

/** The answer-draft key for one ask: the CHAT draft's prefix plus an ask
    namespace, so an ask draft is purged by the same auth sweep
    (`private-storage.ts` matches on the prefix) without that sweep needing to
    learn a second spelling. */
const ASK_DRAFT_PREFIX = `${DRAFT_PREFIX}ask:`;

/** What an answer draft holds: the chosen labels (or the typed text) per
 *  question id, plus the questions the user explicitly skipped.
 *
 *  WHY THIS SURVIVES UNMOUNTING (QA round 1 Q-1 = UX round 1 U1, design §5.0's
 *  R7). The draft used to live in `AskCard`'s own `useState`, and the sheet
 *  unmounts when it closes — so collapsing to read the transcript and coming
 *  back re-answered every question from scratch, which is exactly what R7's
 *  "toggling preserves BOTH drafts" forbids (it keeps the chat draft, which
 *  has lived here all along). Storage is the same home the chat draft uses, for
 *  the same reasons, and it comes free with the reload/navigation survival and
 *  the sign-out purge. */
export interface AskDraft {
	answers: Record<string, string[]>;
	skipped: string[];
}

const EMPTY_ASK_DRAFT: AskDraft = { answers: {}, skipped: [] };

function readAskDraft(askId: string): AskDraft {
	const raw = localStorage.getItem(ASK_DRAFT_PREFIX + askId);
	if (!raw) return EMPTY_ASK_DRAFT;
	try {
		const parsed = JSON.parse(raw) as Partial<AskDraft>;
		return {
			answers: parsed.answers && typeof parsed.answers === "object" ? parsed.answers : {},
			skipped: Array.isArray(parsed.skipped) ? parsed.skipped.map(String) : [],
		};
	} catch {
		/* Unreadable is the same as absent: a half-written value must not take
		   the card down, and the user's next gesture overwrites it. */
		return EMPTY_ASK_DRAFT;
	}
}

/** Drop one ask's draft, called when the ask SETTLES.

    A settled ask's draft is spent: leaving it would re-fill a card the user
    already answered (and the sheet re-reads the aggregate, so a settled card can
    come back in the list). Ids are unique per ask, so this is hygiene rather
    than collision defence. */
export function clearAskDraft(askId: string): void {
	localStorage.removeItem(ASK_DRAFT_PREFIX + askId);
}

/** A value, or an updater over the current one — React's own two spellings. */
export type AskDraftSetter = (next: AskDraft | ((current: AskDraft) => AskDraft)) => void;

function persistAskDraft(askId: string, next: AskDraft): void {
	if (Object.keys(next.answers).length === 0 && next.skipped.length === 0) {
		clearAskDraft(askId);
	} else {
		localStorage.setItem(ASK_DRAFT_PREFIX + askId, JSON.stringify(next));
	}
}

/**
 * Answer-draft hook, keyed by ask id: initialises from storage, writes through
 * on change. The ask twin of `useDraft`, deliberately the same shape so the two
 * buffers cannot drift in how they survive a collapse.
 *
 * THE SETTER IS FUNCTIONAL, and that is not decoration. The card holds one draft
 * map for ALL of an ask's questions, so answering question 2 has to build on
 * question 1's pick — and two taps in the same tick (a fast thumb, or the
 * capture rig's own scripted pair) both closed over the SAME pre-tick `draft`,
 * so the second overwrote the first and the form stayed incomplete with the send
 * control disabled. The rig found it: `scripts/mobile_asks_capture.py` clicks
 * both of the head ask's options in one evaluate, and its log said "send
 * disabled — the form is incomplete" while the frame showed one option pressed.
 */
export function useAskDraft(askId: string): [AskDraft, AskDraftSetter] {
	const [draft, setDraftState] = useState<AskDraft>(() => readAskDraft(askId));
	useEffect(() => {
		setDraftState(readAskDraft(askId));
	}, [askId]);
	const setDraft: AskDraftSetter = (next) =>
		setDraftState((current) => {
			const value = typeof next === "function" ? next(current) : next;
			/* Written from inside the updater so the stored copy cannot lag the
			   rendered one. Idempotent, so a double-invoked updater is harmless. */
			persistAskDraft(askId, value);
			return value;
		});
	return [draft, setDraft];
}

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

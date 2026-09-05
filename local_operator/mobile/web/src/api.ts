/**
 * REST client for the mobile daemon. Same-origin; the vite dev server
 * proxies /api to 127.0.0.1:4097.
 *
 * The 401 rule: the auth cookie has died and every further call will fail
 * the same way, so reload the page and let the server 303 to /login. There
 * is no client-side login form — login is server-rendered.
 */
import { clearPrivateSessionStorage } from "./private-storage";
import type {
	CommandOp,
	Directories,
	ModelEntry,
	PastSession,
	SessionSummary,
	SlashCommand,
	SubagentDetail,
	TranscriptEntry,
} from "./types";

export class HttpError extends Error {
	constructor(readonly status: number, message: string) {
		super(message);
		this.name = "HttpError";
	}
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(path, {
		credentials: "same-origin",
		...init,
	});
	if (res.status === 401) {
		/* A replaced/login-expired browser session must not expose drafts or
		   retry envelopes to whoever authenticates next on this device. */
		clearPrivateSessionStorage();
		location.reload();
		/* Never reached in practice; satisfies the type when reload is slow. */
		throw new Error("unauthorized");
	}
	if (!res.ok) {
		let detail = `${res.status}`;
		try {
			const body = (await res.json()) as { error?: string };
			if (body.error) detail = body.error;
		} catch {
			/* A non-JSON error body carries no more than the status did. */
		}
		throw new HttpError(res.status, detail);
	}
	return (await res.json()) as T;
}

export function getSessions(): Promise<{ sessions: SessionSummary[] }> {
	return request("/api/sessions");
}

export function getCommands(): Promise<{ commands: SlashCommand[] }> {
	return request("/api/commands");
}

export function getModels(): Promise<{ models: ModelEntry[] }> {
	return request("/api/models");
}

export function getDirectories(): Promise<Directories> {
	return request("/api/directories");
}

export function getPastSessions(): Promise<{ sessions: PastSession[] }> {
	return request("/api/sessions/past");
}

/** Search past sessions by name, id, or conversation body (the /resume
    picker's mechanism). Empty query returns the recent list. */
export function searchSessions(
	q: string,
	limit = 40,
): Promise<{ sessions: PastSession[]; query: string }> {
	const params = new URLSearchParams({ q, limit: String(limit) });
	return request(`/api/sessions/search?${params}`);
}

/** Reopen a past session as a new live session the phone attaches to. */
export function resumeSession(
	sessionId: string,
): Promise<{ ok: boolean; pid: number; session_id: string }> {
	return request("/api/sessions/resume", {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ session_id: sessionId }),
	});
}

export function startSession(input: {
	cwd: string;
	provider?: string;
	model_id?: string;
}): Promise<{ ok: boolean; pid: number; session_id: string }> {
	return request("/api/sessions/start", {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(input),
	});
}

/** Mark a session's finished activity as viewed. Fire-and-forget from the
    session view's mount: the daemon clears `unseen` and repaints the list,
    but the client store clears its own copy optimistically (markSessionSeen)
    so back-navigation never flashes a stale mark while this POST is in
    flight. The response body carries nothing the UI needs. */
export function markSessionSeen(sessionId: string): Promise<{ ok: boolean }> {
	return request(`/api/sessions/${encodeURIComponent(sessionId)}/seen`, {
		method: "POST",
	});
}

export function sendCommand(
	sessionId: string,
	op: CommandOp,
): Promise<{ ok: boolean; detail: string }> {
	return request(`/api/sessions/${encodeURIComponent(sessionId)}/command`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(op),
	});
}

/** URL for one image attachment on a user turn. The bytes are served lazily
    from the transcript (never carried in the projection), keyed by the entry
    id plus the image-only index the ref emitted. Same-origin, cacheable and
    immutable — a message's attachments never change — so an <img src> can use
    it directly. */
export function imageUrl(sessionId: string, entryId: string, index: number): string {
	const q = new URLSearchParams({ entry: entryId, i: String(index) });
	return `/api/sessions/${encodeURIComponent(sessionId)}/image?${q}`;
}

/** Older transcript entries for lazy loading. ``before`` is the id of the
    oldest entry the client already has; the daemon returns the page
    immediately older than it (chronological within the page) plus whether
    more history exists beyond. */
export function getHistory(
	sessionId: string,
	before: string | null,
	limit = 80,
): Promise<{ entries: TranscriptEntry[]; has_more: boolean }> {
	const q = new URLSearchParams({ limit: String(limit) });
	if (before) q.set("before", before);
	return request(`/api/sessions/${encodeURIComponent(sessionId)}/history?${q}`);
}

export function getSubagentDetail(
	sessionId: string,
	jobId: string,
	signal?: AbortSignal,
): Promise<SubagentDetail> {
	return request(
		`/api/sessions/${encodeURIComponent(sessionId)}/agents/${encodeURIComponent(jobId)}`,
		{ signal },
	);
}

/** Child history has its own lineage-checked endpoint. Reusing the root route
    here was the paging bug: once a child scrolled above its live tail, root
    user/tool rows appeared inside the child's conversation. */
export function getSubagentHistory(
	sessionId: string,
	jobId: string,
	before: string | null,
	limit = 80,
	signal?: AbortSignal,
): Promise<{ entries: TranscriptEntry[]; has_more: boolean }> {
	const q = new URLSearchParams({ limit: String(limit) });
	if (before) q.set("before", before);
	return request(
		`/api/sessions/${encodeURIComponent(sessionId)}/agents/${encodeURIComponent(jobId)}/history?${q}`,
		{ signal },
	);
}

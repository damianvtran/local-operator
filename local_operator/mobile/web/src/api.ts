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
	CompletionAttention,
	Directories,
	ModelEntry,
	PastSession,
	SessionSummary,
	SlashCommand,
	SubagentDetail,
	TranscriptEntry,
} from "./types";

export class HttpError extends Error {
	/** The refusal's CATEGORY, when the far side sent one.

	    A refusal crosses as a typed code plus the copy, and callers that have to
	    DECIDE something (re-sign? offer pairing? name the install step?) key on the
	    code rather than on prose this repo has rewritten twice already. Absent for
	    every ordinary failure and for a runtime built before the field existed, so
	    a caller that needs it must have a copy-based fallback. */
	constructor(
		readonly status: number,
		message: string,
		readonly code = "",
	) {
		super(message);
		this.name = "HttpError";
	}
}

/** The two authority refusals, as the enumerated codes the wire carries.

    ONE definition, imported by everything that has to distinguish them from an
    ordinary failure — the card's re-sign decision, the gate sheet's copy, a test.
    Two lists would be two rules, and the whole reason the code exists is that the
    copy it rides beside has already been rewritten twice (UX round 6, U6). */
export const AUTHORITY_REFUSAL_CODES = [
	"operator_authority_required",
	"operator_authority_unconfigured",
] as const;

/** Whether a typed code names an authority refusal. */
export function isAuthorityRefusalCode(code: string): boolean {
	return (AUTHORITY_REFUSAL_CODES as readonly string[]).includes(code);
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
		let code = "";
		try {
			const body = (await res.json()) as { error?: string; code?: string };
			if (body.error) detail = body.error;
			if (typeof body.code === "string") code = body.code;
		} catch {
			/* A non-JSON error body carries no more than the status did. */
		}
		throw new HttpError(res.status, detail, code);
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

/**
 * A receipt names what was rendered, not whichever completion exists on arrival.
 *
 * The `attention` the daemon answers with is the whole verdict of the handshake,
 * so it is typed here rather than left for the caller to read out of an untyped
 * body: 2xx means the conversation is READ (`unseen: false`) and nothing else
 * does. An older daemon answered a superseded token with a 200 whose body still
 * said `unseen: true`, and a caller that took the resolved call for a read
 * latched on a completion that never cleared (see docs/ATTENTION.md).
 */
export function markSessionSeen(
	sessionId: string,
	completionToken: string,
): Promise<{ ok: boolean; attention?: CompletionAttention }> {
	return request(`/api/sessions/${encodeURIComponent(sessionId)}/seen`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ completion_token: completionToken }),
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

/** A command frame, plus the proof fields a paired phone adds to it.

    `operator_sig`/`operator_key_id`/`operator_cert` are SIGNATURE material the
    runtime checks against its pinned operator key, not the machine-held
    `operator_cap` (which the relay drops from any HTTP body, always). The two are
    different classes: a capability is proof material this machine can mint and a
    body carrying one can only be a forgery, while a signature is unforgeable and
    its challenge is single-use, so the relay carries it. */
export type SignedCommand = CommandOp & {
	operator_sig?: string;
	operator_key_id?: string;
	operator_cert?: string;
};

export function sendCommandWithProof(
	sessionId: string,
	op: SignedCommand,
): Promise<{ ok: boolean; detail: string }> {
	return request(`/api/sessions/${encodeURIComponent(sessionId)}/command`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(op),
	});
}

/** Ask the runtime to mint a per-action challenge for THIS session's connection.

    The relay forwards one ordinary frame and hands back the challenge; the phone
    signs it. A challenge is worth exactly one signature, so holding one lets
    nobody sign — and the signature is the only thing that ever carries authority
    here. `action` chooses what the signature will be accepted FOR, and the runtime
    derives the same field from the frame it is deciding, so a signature minted for
    one action cannot be presented as the other. */
export function requestOperatorChallenge(
	sessionId: string,
	input: { action: "loosen" | "approve"; request_id?: string },
): Promise<{
	challenge: string;
	expires_s: number;
	session_id: string;
	action: string;
	request_id: string;
}> {
	return request(`/api/sessions/${encodeURIComponent(sessionId)}/operator/challenge`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ request_id: "", ...input }),
	});
}

/** Claim a pairing code with this device's PUBLIC key. The private half never
    leaves the phone, so there is no field here for it. */
export function claimPairingCode(input: {
	code: string;
	spki: string;
	name: string;
}): Promise<{ ok: boolean; device_id: string }> {
	return request("/api/pair", {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(input),
	});
}

/** Whether the operator has approved a claimed code yet, and the certificate they
    signed. The certificate is minted on the MACHINE by the operator's gesture, so
    this is the only way the phone ever learns the string it must present. */
export function pairingStatus(deviceId: string): Promise<{
	paired: boolean;
	device_id: string;
	certificate?: string;
	operator_key_id?: string;
	scope?: string[];
	exp?: number;
	name?: string;
	/** Whether the MACHINE can verify this device's signatures yet — see
	    `GateSheet`/`PairScreen`: false between `lop operator init` and
	    `lop operator install`, which is the state the old success box lied about. */
	authority_ready?: boolean;
}> {
	return request(`/api/pair/${encodeURIComponent(deviceId)}`);
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

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
	AttentionUnread,
	Capabilities,
	CommandOp,
	CompletionAttention,
	Directories,
	ModelEntry,
	PastSession,
	PendingAsk,
	ProjectLinkedSession,
	ProjectSummary,
	ProjectView,
	LibraryAsset,
	LibraryChatLink,
	LibraryListing,
	LibraryTag,
	SessionSummary,
	SessionTagsResponse,
	SlashCommand,
	SubagentDetail,
	TagsResponse,
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

/** The 401 rule, shared by every transport in this module: the auth cookie is
    dead and every further call fails the same way, so the page reloads and the
    server 303s to /login. The multipart upload below is NOT the JSON helper but
    must not invent a second answer to this. */
function handleUnauthorized(): void {

	/* A replaced/login-expired browser session must not expose drafts or
	   retry envelopes to whoever authenticates next on this device. */
	clearPrivateSessionStorage();
	location.reload();
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(path, {
		credentials: "same-origin",
		...init,
	});
	if (res.status === 401) {
		handleUnauthorized();
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

export type AccountRoutingPayload = {
	capability: "account-routing-v1";
	setting: {
		key: "retry.accountSelectionPolicy";
		label: string;
		help: string;
		kind: "enum";
		value: string;
		default: string;
		is_default: boolean;
		choices: { value: string; label: string; description: string }[];
	};
};

export function isAccountRoutingPayload(payload: unknown): payload is AccountRoutingPayload {
	if (!payload || typeof payload !== "object") return false;
	const result = payload as Partial<AccountRoutingPayload>;
	const row = result.setting;
	return result.capability === "account-routing-v1" && row?.key === "retry.accountSelectionPolicy" && row.kind === "enum"
		&& typeof row.label === "string" && typeof row.help === "string" && typeof row.value === "string"
		&& Array.isArray(row.choices) && row.choices.length > 0
		&& row.choices.every(choice => typeof choice?.value === "string" && typeof choice.label === "string" && typeof choice.description === "string")
		&& row.choices.some(choice => choice.value === row.value);
}

export function getAccountRouting(): Promise<AccountRoutingPayload> {
	return request("/api/settings/account-routing");
}

export function setAccountRouting(value: string): Promise<AccountRoutingPayload> {
	return request("/api/settings/account-routing", {
		method: "PATCH",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ value }),
	});
}

export function getSessions(): Promise<{
	sessions: SessionSummary[];
	capabilities?: Capabilities;
}> {
	return request("/api/sessions");
}

export type PhoneUsageLimit = {
	amount: { used: number | null; limit: number | null; remaining: number | null;
		used_fraction: number | null; unit: string };
	paid_fallback: boolean;
	available: boolean | null;
	unlimited: boolean;
	supersedes_tiers: boolean;
	family_cap: boolean;
	shared: boolean;
	window: "5h" | "7d" | "1mo" | "lifetime" | "unknown";
	family: "opus" | "sonnet" | "fable" | "zread" | "unknown";
	resets_at_ms: number | null;
	payable: boolean;
	exhausted: boolean;
};
export type PhoneUsageReport = {
	provider: string; limits: PhoneUsageLimit[]; unavailable: boolean; fetched_at: number;
};
/** The paired gate and persisted-account admission remain server-owned. */
export function getUsage(): Promise<{ reports: PhoneUsageReport[]; source: "cached";
	may_be_stale?: boolean; cached_at?: number | null }> {
	return request("/api/usage");
}

export function getCommands(): Promise<{ commands: SlashCommand[] }> {
	return request("/api/commands");
}

/** Every queued ask worth showing, across conversations (design §4/§5.3).

    Index-backed on the daemon side and read on demand rather than fed: each row
    is the frozen `PendingAsk` shape plus `session_id` and `cwd`, because a row
    drawn under another conversation's name must be answerable against THAT
    conversation's route. An ask outlives the runtime that queued it, so this
    answer never needs one — but answering one sent to a session with no live
    runtime is a refusal naming the reason, not a silent failure. */
export function getAsks(signal?: AbortSignal): Promise<{ asks: PendingAsk[] }> {
	return request("/api/asks", { signal });
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

/** The SHARED conversation-tag registry (`GET /api/tags`), read only.

    There is deliberately no catalogue write here: identity (create/rename) lives
    with the desktop app and agent callers, and a second implementation of it on
    the phone is how the two surfaces start disagreeing. `order` is the
    registry's own ordering field; the sheet renders rows in it. */
export function getTags(): Promise<TagsResponse> {
	return request("/api/tags");
}

/** Assign a conversation's tags as DESIRED STATE.

    `{tags, primary}` and nothing else: the relay refuses a body without
    `primary` (never a silent re-primary) or with an unknown key. Sending the
    state wanted rather than a delta is the property that makes a retry land on
    the same assignment, and `[]` + `null` is the clear. The answer is the store's
    canonical read-back — paint it rather than the request, because the store may
    have pruned a category deleted between the catalogue read and this write. */
export function setSessionTags(
	sessionId: string,
	tags: string[],
	primary: string | null,
): Promise<SessionTagsResponse> {
	return request(`/api/sessions/${encodeURIComponent(sessionId)}/tags`, {
		method: "PUT",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ tags, primary }),
	});
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

/** Start a session, optionally in a named directory.

    THE WORKING DIRECTORY IS OPTIONAL, and that is the feature: the list's
    one-tap start posts no ``cwd`` at all, and the daemon resolves it
    server-side (``GET /api/directories`` publishes the same answer as
    ``default``).
    A caller with a directory in hand passes ``cwd``; nothing else about the
    request changes, so both callers share one route and one response shape. */
export function startSession(
	input: { cwd?: string; provider?: string; model_id?: string } = {},
): Promise<{ ok: boolean; pid: number; session_id: string }> {
	return request("/api/sessions/start", {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(input),
	});
}

/** Point a PRISTINE session at a different working directory.

    The session's identity does not change: the daemon retires the unused
    runtime and starts a successor in ``cwd`` under the SAME ``session_id``, so
    the route stays put and the transcript, the stream and the conversation the
    user is looking at are the same ones.

    A refusal carries a user-facing sentence in ``error`` (rendered verbatim by
    the phone) and a stable ``code`` the caller branches on — ``session_has_history``
    (a conversation with messages can never move), ``session_busy`` (a turn is in
    flight; retry when it lands) and ``move_unavailable`` (the fallback for
    transport, capability and too-old-owner cases). */
export function changeDirectory(
	sessionId: string,
	cwd: string,
): Promise<{ ok: boolean; pid: number; session_id: string }> {
	return request(`/api/sessions/${encodeURIComponent(sessionId)}/directory`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ cwd }),
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

/** The unread badge, as ``GET /api/attention/unread`` serves it.

    Why the client wants it apart from the list frames: the summaries carry the
    ``unseen`` FLAG but never the completion TOKEN, and the one-gesture clear
    must name the completions it rendered (see ``markAllSeen``). Read at the
    moment of the press, so the pair the store compares is one read's set —
    listing and clearing are one set, the TUI's own rule. */
export function getAttentionUnread(): Promise<AttentionUnread> {
	return request("/api/attention/unread");
}

/** What the bulk seen route answers with: one entry per input item, bucketed.

    ``read`` entries are the store's own post-write states (the caller needs no
    second read to learn what its rows now say); ``superseded`` and ``unknown``
    name the two verdicts that mean "not cleared", by session id, so a caller
    can name the remainder instead of reporting a clean sweep it did not get. */
export type AttentionSeenManyReceipt = {
	ok: boolean;
	read: CompletionAttention[];
	superseded: string[];
	unknown: string[];
};

/** Clear several rendered completions in ONE write (issue #2016).

    The bulk half of ``markSessionSeen``, and the same receipt rule: each pair
    names a completion the caller actually RENDERED, and the daemon compares it
    against the conversation's CURRENT token inside one transaction, so a newer
    result stays unread (it answers ``superseded``). A batch that clears nothing
    is still a resolved 2xx — the three buckets ARE the answer — so callers must
    read the buckets rather than the status, exactly as the single call's caller
    reads ``attention.unseen`` rather than the resolution. Mirrors the desktop
    plane's ``POST /v1/desktop/attention/seen``. */
export function markAllSeen(
	items: { session_id: string; completion_token: string }[],
): Promise<AttentionSeenManyReceipt> {
	return request("/api/attention/seen", {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ items }),
	});
}

/** Pin or unpin a conversation in the SHARED durable store.

    Sends the DESIRED STATE rather than a toggle, matching the desktop route: a
    dropped response must not flip the pin back when the caller retries, so the
    body names the state wanted and a retry lands on the same one. The daemon
    stores it in `sidebar-pins.json`, the same file the terminal's F10 and the
    desktop app write, so a pin made here appears there and vice versa. */
export function setSessionPin(
	sessionId: string,
	pinned: boolean,
): Promise<{ ok: boolean; pinned: boolean }> {
	return request(`/api/sessions/${encodeURIComponent(sessionId)}/pin`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ pinned }),
	});
}

/** What a control command answers with.

    ``tone`` and ``refused`` are ADDITIVE fields the daemon sends for a routed
    slash outcome (design round 1, D1; UX round 1, U3): the runtime gives a
    completed write and a plain report the same ``style``, so the container and
    the draft-keeping rule are decided where the outcome is read. Both are absent
    for every other op, and a caller that ignores them reads ``detail`` exactly as
    it always did. */
export type CommandReceipt = {
	ok: boolean;
	detail: string;
	tone?: "success" | "neutral";
	refused?: boolean;
};

export function sendCommand(sessionId: string, op: CommandOp): Promise<CommandReceipt> {
	return request(`/api/sessions/${encodeURIComponent(sessionId)}/command`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(op),
	});
}

/** The recorder's media types → a suffix for the uploaded part's filename.
    The server allowlist is the authority on what is accepted; this only keeps
    the multipart part named like what it contains (a nameless part makes the
    upstream's own decoder guesses worse). */
const RECORDING_FILENAME_SUFFIX: Record<string, string> = {
	"audio/mp4": "m4a",
	"audio/x-m4a": "m4a",
	"audio/webm": "webm",
	"audio/ogg": "ogg",
	"audio/mpeg": "mp3",
	"audio/wav": "wav",
	"audio/aac": "aac",
};

function recordingFilename(blob: Blob): string {
	const bare = blob.type.split(";", 1)[0]?.trim().toLowerCase() ?? "";
	return `dictation.${RECORDING_FILENAME_SUFFIX[bare] ?? "bin"}`;
}

/** POST one recorded clip to the daemon; the transcript is the answer.

    Multipart, so this does NOT ride the JSON `request()` helper — but it keeps
    the same 401 rule (`handleUnauthorized`) and the same error contract: the
    body's `error` becomes the `HttpError` message (413/422/402/503 carry the
    daemon's own actionable copy) and `code` its typed category. `path` is the
    token that ACTUALLY ran — the caller stores it as the send envelope's
    `input_path` and must not re-derive it from a cached capability.

    `signal` is for U5's cancel: the composer's status row aborts an in-flight
    transcription rather than only hiding the wait; a fetch aborted through it
    rejects with an AbortError, which the caller already treats as a discard. */
export function transcribeAudio(
	blob: Blob,
	signal?: AbortSignal,
): Promise<{
	text: string;
	provider: string;
	model: string | null;
	path: string;
}> {
	const form = new FormData();
	form.append("audio", blob, recordingFilename(blob));
	// The daemon normalises this to an ISO-639-1 code before it reaches Radient
	// or the client rules; an unmapped tag (e.g. fil-PH -> fil) is omitted, not
	// guessed. English-only cleanup on a Tagalog take is the failure direction
	// the cleanup design refuses.
	const language = typeof navigator !== "undefined" ? navigator.language : "";
	if (language) form.append("language", language);
	return fetch("/api/transcribe", { method: "POST", body: form, signal }).then(async (res) => {
		if (res.status === 401) {
			handleUnauthorized();
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
		return (await res.json()) as {
			text: string;
			provider: string;
			model: string | null;
			path: string;
		};
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
): Promise<CommandReceipt> {
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

/** One attachment artifact's bytes, keyed by the entry id plus the
    kind-scoped index the ref emitted. The one RANGE-CAPABLE data route:
    phone Safari will not play a <video> that arrives as a full-body 200, so
    the server answers byte-Range requests with 206 + Content-Range. Same
    content key as `imageUrl` (a message uuid plus the family index), so the
    same immutable caching applies. */
export function attachmentUrl(
	sessionId: string,
	entryId: string,
	index: number,
	kind: "image" | "video",
): string {
	const q = new URLSearchParams({ entry: entryId, i: String(index), kind });
	return `/api/sessions/${encodeURIComponent(sessionId)}/attachment?${q}`;
}

/** One supplement sound's bytes, keyed by the beat's own
    `sounds[].attachment` digest. A separate route from `attachmentUrl` on
    purpose: a turn's generated sound is a supplement artifact with no message
    block to index, so `(entry, kind, i)` cannot address it. Range-capable — the
    player's media element needs 206s — and the digest is the cache key. */
export function soundUrl(sessionId: string, digest: string): string {
	const q = new URLSearchParams({ digest });
	return `/api/sessions/${encodeURIComponent(sessionId)}/sound?${q}`;
}

/** The waveform sidecar for one supplement sound, keyed by the same digest.
    404 (never fabricated zeros) means "draw the flat scrubber". */
export function soundWaveformUrl(sessionId: string, digest: string): string {
	const q = new URLSearchParams({ digest });
	return `/api/sessions/${encodeURIComponent(sessionId)}/waveform?${q}`;
}

/* ---- turn supplements ("Highlights", memo §2.7) --------------------------- */

/** The component document route: `{html}` assembled server-side; the bytes
    NEVER load as a document from this URL — the block mounts them as a
    sandboxed `data:` frame instead, so nothing navigates a browsing context
    to the relay route (memo §4.1's relay row). */
export function getSupplementDocument(
	sessionId: string,
	digest: string,
): Promise<{ html: string }> {
	return request(
		`/api/sessions/${encodeURIComponent(sessionId)}/supplements/${encodeURIComponent(digest)}/document`,
	);
}

/** The file-callout route, addressed by the ROW'S OWN index — there is no
    path parameter by design, so this can only ask for a file the row already
    recorded. Bytes are fetched (never navigated to): the caller renders them
    in-app, and `Content-Disposition: attachment` on every non-image class
    means a navigation would download, not display, anyway. */
export function supplementFileUrl(sessionId: string, job: string, index: number): string {
	const q = new URLSearchParams({ i: String(index) });
	return `/api/sessions/${encodeURIComponent(sessionId)}/supplements/${encodeURIComponent(job)}/file?${q}`;
}

/** Fetch one callout's bytes. NOT the JSON helper: the success body is the
    file itself and the type is the response header's. The failure contract is
    the same one every transport in this module keeps — the 401 rule, and the
    daemon's own `error` sentence as the `HttpError` message (404 "no such
    file" covers a denylisted path and an out-of-range index alike by design;
    409 names a cold session's cwd-relative file; 413 a file over the preview
    cap). */
export function getSupplementFile(
	sessionId: string,
	job: string,
	index: number,
): Promise<{ blob: Blob; mime: string }> {
	return fetch(supplementFileUrl(sessionId, job, index), {
		credentials: "same-origin",
	}).then(async (res) => {
		if (res.status === 401) {
			handleUnauthorized();
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
		return {
			blob: await res.blob(),
			mime: res.headers.get("content-type") ?? "",
		};
	});
}

/** The three supplement control ops, over the session command route the whole
    surface already mutates through (see `CommandOp`'s supplement family). The
    runtime's receipt travels in the answer's `detail` verbatim. */
export function supplementCancel(
	sessionId: string,
	anchor: string,
	job: string,
): Promise<CommandReceipt> {
	return sendCommand(sessionId, { op: "supplement_cancel", anchor, job });
}

export function supplementSteer(
	sessionId: string,
	anchor: string,
	job: string,
	text: string,
): Promise<CommandReceipt> {
	return sendCommand(sessionId, { op: "supplement_steer", anchor, job, text });
}

export function supplementRestart(
	sessionId: string,
	anchor: string,
	job: string,
): Promise<CommandReceipt> {
	return sendCommand(sessionId, { op: "supplement_restart", anchor, job });
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

/* ---- projects ----------------------------------------------------------- */

/** The listing — every project as a summary, in the daemon's board order
    (status rank, then freshest). The Projects sheet's list and its
grouped board render this one array; neither re-sorts it. */
export function getProjects(): Promise<{ projects: ProjectSummary[] }> {
	return request("/api/projects");
}

/** One project plus its linked sessions — the detail sheet's read. */
export function getProject(
	key: string,
	signal?: AbortSignal,
): Promise<{ project: ProjectView; links: ProjectLinkedSession[] }> {
	return request(`/api/projects/${encodeURIComponent(key)}`, { signal });
}

/** Create one project. It starts UNLINKED by design (the desktop create body
    carries no sessions either): linking a session is a deliberate act, never
    a side effect of creating a row.

    The body is the desktop route's own create vocabulary
    (``ProjectCreate``): ``description``, ``status`` and ``tags`` are the four
    keys the daemon accepts beyond the name, and every one of them is optional
    — an absent key is a field the daemon leaves at its default, not one the
    client has to spell out. ``status`` defaults server-side to ``active``.

    The response is a SUMMARY, not a full view: a create has no milestones,
    no links and no history yet, so the caller that wants the view re-reads it.
    A refusal carries the daemon's sentence in ``error`` and a machine ``code``
    (``project_name_exists`` for a taken name — case-insensitively — and
    ``project_invalid`` for a name or tag the store's grammar refuses). */
export function createProject(input: {
	name: string;
	description?: string;
	status?: string;
	tags?: string[];
}): Promise<{ ok: boolean; project: ProjectSummary }> {
	return request("/api/projects", {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(input),
	});
}

/** Partially edit one project. ONLY the keys carried are touched — the
    daemon forwards ``model_fields_set`` into the store, so an omitted key
    keeps its stored value while an explicit ``""`` CLEARS a date.

    THE ESTIMATE IS THE EXCEPTION, and it is the store's rule rather than this
    client's: ``update_project``'s apply arm is ``if "estimate" in supplied
    and fields.estimate is not None``, so an explicit ``null`` is a silent
    NO-OP that leaves the stored number in place (measured against the real
    daemon — a ``null`` estimate came back as ``3.0``). A caller that wants to
    leave an estimate alone must therefore OMIT the key; one that wants to
    clear it has no request at all. (An earlier revision of this comment
    claimed ``null`` cleared it, which was exactly the sort of unverified
    claim this file should not carry.)

    The tri-state is the whole reason the caller builds the body rather than
    posting a whole record: a form that re-sent every field it rendered would
    have to guess the ones it never loaded (``owner``, ``team``, ``progress``)
    and would clear them as a side effect of an edit that had nothing to do
    with them. */
export function patchProject(
	key: string,
	input: {
		name?: string;
		description?: string;
		status?: string;
		tags?: string[];
		start_date?: string;
		target_date?: string;
		estimate?: number | null;
		estimate_unit?: string;
	},
): Promise<{ ok: boolean; project: ProjectSummary }> {
	return request(`/api/projects/${encodeURIComponent(key)}`, {
		method: "PATCH",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(input),
	});
}

/** Link one session to one project. The response is a SUMMARY (the write's
    own answer, the same shape every link mutation returns), so a caller
    showing the composed view re-reads it rather than patching one field of a
    document this route never returned. */
export function linkProjectSession(
	key: string,
	sessionId: string,
): Promise<{ ok: boolean; project: ProjectSummary }> {
	return request(`/api/projects/${encodeURIComponent(key)}/links`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ session_id: sessionId }),
	});
}

/** Unlink one session. The id travels in the path — the route carries no body
    — and the session id is percent-encoded because a hand-typed id is the one
    place a caller could hand one through with a character the path forbids. */
export function unlinkProjectSession(
	key: string,
	sessionId: string,
): Promise<{ ok: boolean; project: ProjectSummary }> {
	return request(
		`/api/projects/${encodeURIComponent(key)}/links/${encodeURIComponent(sessionId)}`,
		{ method: "DELETE" },
	);
}

/** Delete one project. The confirmation the daemon requires is the NAME it
    showed the reader — this call takes it from the same row the sheet rendered
    rather than reconstructing it, so a renamed project can never be deleted by
    a stale button. */
export function deleteProject(key: string, confirm: string): Promise<{ ok: boolean; deleted: boolean }> {
	return request(`/api/projects/${encodeURIComponent(key)}`, {
		method: "DELETE",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ confirm }),
	});
}

/** Add-or-update one milestone by name. ``completed`` is the detail sheet's
    toggle (true stamps today, false clears it); ``target_date: ""`` clears a
    date. Answered with the whole project view, so the sheet re-renders from
    the store's own answer rather than a local guess at what changed. */
export function setProjectMilestone(
	key: string,
	input: { name: string; target_date?: string; completed?: boolean },
): Promise<{ ok: boolean; project: ProjectView }> {
	return request(`/api/projects/${encodeURIComponent(key)}/milestones`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(input),
	});
}

/** Remove one milestone by name (case-insensitively, the same key the
    add-or-update route uses). Answered with the whole project view, so a
    caller renders the store's own list rather than a locally filtered one.

    THE NAME TRAVELS IN THE PATH, and the route upstream is ``{name:str}``
    (``[^/]+``), so a name containing a slash can never be addressed here —
    measured: ``DELETE …/milestones/ship%2Fv2`` answers 404, even though the
    same name was accepted on create. Percent-encoding is not the defect and
    cannot fix it (the segment is decoded before routing). A caller must not
    create such a name, and must refuse the removal with that explanation
    rather than sending a request that cannot match. */
export function removeProjectMilestone(
	key: string,
	name: string,
): Promise<{ ok: boolean; project: ProjectView }> {
	return request(
		`/api/projects/${encodeURIComponent(key)}/milestones/${encodeURIComponent(name)}`,
		{ method: "DELETE" },
	);
}

/* ---------------------------------------------------------------------------
 * Library
 *
 * Every call is the relay's own `/api/library*`; ids are opaque `la_` values and
 * no caller can name a path. `bytes`/`thumb` are URLs rather than fetch wrappers
 * because the media elements and a download anchor need the URL itself.
 * ------------------------------------------------------------------------- */

export type LibraryQuery = {
	q?: string;
	categories?: string[];
	kinds?: string[];
	origin_groups?: string[];
	origins?: string[];
	tag?: string;
	tag_mode?: "all" | "any";
	highlighted?: boolean;
	organized?: boolean;
	favorite?: boolean;
	session?: string;
	sort?: "recent" | "name" | "kind" | "size" | "created" | "modified" | "organized";
	order?: "asc" | "desc";
	limit?: number;
	cursor?: string;
};

function libraryParams(query: LibraryQuery): string {
	const params = new URLSearchParams();
	for (const [key, value] of Object.entries(query)) {
		if (value === undefined || value === null || value === "") continue;
		if (Array.isArray(value)) {
			if (value.length) params.set(key, value.join(","));
			continue;
		}
		params.set(key, String(value));
	}
	return params.toString();
}

export function getLibrary(query: LibraryQuery = {}): Promise<LibraryListing> {
	const encoded = libraryParams(query);
	return request(`/api/library/assets${encoded ? `?${encoded}` : ""}`);
}

export function getLibraryAsset(assetId: string): Promise<LibraryAsset> {
	return request(`/api/library/assets/${encodeURIComponent(assetId)}`);
}

export function patchLibraryAsset(
	assetId: string,
	fields: Partial<Pick<LibraryAsset, "name" | "description" | "favorite" | "locked" | "archived">>,
): Promise<LibraryAsset> {
	return request(`/api/library/assets/${encodeURIComponent(assetId)}`, {
		method: "PATCH",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(fields),
	});
}

export function getLibraryTags(): Promise<{ tags: LibraryTag[] }> {
	return request("/api/library/tags");
}

export function createLibraryTag(
	name: string,
	category = "custom",
): Promise<{ tag: LibraryTag }> {
	return request("/api/library/tags", {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ name, category }),
	});
}

export function renameLibraryTag(tagId: string, name: string): Promise<{ tag: LibraryTag }> {
	return request(`/api/library/tags/${encodeURIComponent(tagId)}`, {
		method: "PATCH",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ name }),
	});
}

export function mergeLibraryTag(tagId: string, into: string): Promise<{ tag: LibraryTag }> {
	return request(`/api/library/tags/${encodeURIComponent(tagId)}/merge`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ into }),
	});
}

export function setLibraryTags(
	assetId: string,
	tagIds: string[],
	primary?: string | null,
): Promise<LibraryAsset> {
	/* `primary` is OMITTED unless the caller states one. The relay (like the
	   desktop store) reads a present null as "clear the primary", so a metadata
	   save that only renamed a file must not send one — it silently dropped the
	   tag the reader had chosen elsewhere (review R-3). Passing an explicit
	   null still clears, for a caller that means it. */
	const body: { tags: string[]; primary?: string | null } = { tags: tagIds };
	if (primary !== undefined) body.primary = primary;
	return request(`/api/library/assets/${encodeURIComponent(assetId)}/tags`, {
		method: "PUT",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(body),
	});
}

export function linkLibraryChat(
	assetId: string,
	sessionId: string,
): Promise<{ chat: LibraryChatLink }> {
	return request(`/api/library/assets/${encodeURIComponent(assetId)}/chat`, {
		method: "PUT",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ session_id: sessionId }),
	});
}

/** The preview URL for one asset. `download` flips the relay's disposition. */
export function libraryBytesUrl(assetId: string, download = false): string {
	const suffix = download ? "?download=1" : "";
	return `/api/library/assets/${encodeURIComponent(assetId)}/content${suffix}`;
}

export function libraryThumbUrl(assetId: string, width = 480): string {
	return `/api/library/assets/${encodeURIComponent(assetId)}/thumbnail?w=${width}`;
}

/** One audio asset's waveform payload (`no-store`). A 404 is the honest
    no-waveform state, which the player draws as a flat scrubber. */
export function libraryWaveformUrl(assetId: string): string {
	return `/api/library/assets/${encodeURIComponent(assetId)}/waveform`;
}

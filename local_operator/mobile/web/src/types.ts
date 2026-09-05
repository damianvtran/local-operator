/**
 * Wire types for the mobile control plane.
 *
 * These mirror `local_operator/mobile/types.py` field-for-field: the daemon
 * serialises its dataclasses with `asdict`, so every field has a default and
 * always appears on the wire. The projection is a FULL snapshot on every push
 * (repaint, not deltas) — the client never merges.
 */

export type EntryKind =
	| "user"
	| "assistant"
	| "tool"
	| "notice"
	| "steer"
	| "compaction"
	| "parent_message"
	| "subagent_message"
	// An inbound message from another local lop session (`lop send`). Rendered
	// as a distinct cross-session card, never as the user's own turn.
	| "peer_message";

export type ToolState =
	| "composing"
	| "running"
	| "done"
	| "failed"
	| "interrupted";

export type TodoStatus = "pending" | "done" | "blocked" | "dropped";

export type SubagentStatus =
	| "running"
	| "completed"
	| "failed"
	| "cancelled"
	| "parked";

export interface TranscriptEntryDetails {
	/* The fold serializes these in the shape the tool produced, NOT always
	   strings: args rides through as a dict ({path, old_text, …}), diff as
	   a list of unified-diff lines. Callers must normalize (see toLines in
	   tool-row.tsx) — treating them as strings and calling .split() throws,
	   which unmounts the whole tree and reads as "tap → blank screen". */
	args?: string | Record<string, unknown>;
	output?: string;
	diff?: string | string[];
	partial?: string;
	/* Sender identity on a `peer_message` entry (`lop send`): pid /
	   conversation_name / model_label / session_id / cwd, all advisory. Rides
	   through the fold's `details` so the card can label who reached in. */
	sender?: PeerSender;
}

export interface PeerSender {
	pid?: number;
	session_id?: string;
	conversation_name?: string;
	model_label?: string;
	cwd?: string;
}

export interface TranscriptEntry {
	id: string;
	kind: EntryKind;
	text: string;
	/* tool rows */
	tool_call_id: string;
	tool_name: string;
	tool_state: ToolState;
	/** The one-line args summary (compacted path etc.). */
	summary: string;
	/** The model's own narration, when it gave one. */
	intent: string;
	diff_added: number;
	diff_removed: number;
	elapsed_s: number;
	error: string;
	details: TranscriptEntryDetails;
	/** Image attachments on a user turn, as lightweight references (never the
	    bytes): each is `{index, mime_type}`. The pixels are fetched lazily from
	    the image endpoint — see `imageUrl` in api.ts. */
	images?: TranscriptImageRef[];
	/** Assistant rows stream: `final` flips true on message_end. */
	final: boolean;
}

/** A reference to one image block on a user turn. The bytes live in the
    on-disk transcript and are served on demand, keyed by the entry id plus
    this image-only index. */
export interface TranscriptImageRef {
	index: number;
	mime_type: string;
}

export interface TodoItem {
	text: string;
	status: TodoStatus;
	reason: string;
}

/** One named group of todos. The server stores todos phased; a single
    implicit `"Todos"` phase carries a flat list and renders headerless (see
    `TodosPanel`). */
export interface TodoPhase {
	name: string;
	items: TodoItem[];
}

export interface SubagentRow {
	job_id: string;
	label: string;
	agent: string;
	status: SubagentStatus;
	/** Latest step line while running. */
	progress: string;
	elapsed_s: number;
	model_label: string;
	/** Settled outcome, one line. */
	result_text: string;
	error_text: string;
	parent_job_id: string | null;
	session_id: string | null;
	prompt: string;
	launch_message_id: string;
	effort: string;
	ancestors: string[];
	ancestor_ids: string[];
	child_ids: string[];
	peer_ids: string[];
	transcript: TranscriptEntry[];
	todos: TodoPhase[];
	activity: string;
}

/** Selected-child payload. Root snapshots remain compatible with the legacy
    aggregate shape, but current daemons leave transcript/todos empty there and
    serve these fields only for the active route. */
export interface SubagentDetail extends SubagentRow {
	version: number;
}

/** One selectable answer on an ask question. Carries the consequence line the
    terminal shows under each option so the phone user decides with the same
    information (U3). */
export interface AskOption {
	label: string;
	description: string;
}

export interface PendingRequest {
	request_id: string;
	kind: "approval" | "ask";
	title: string;
	detail: string;
	/** Ask pickers; empty means a free-text / secret paste field. */
	options: AskOption[];
	/** True when this ask requests a credential: the paste field is masked and
	    labelled as a secret (D1/U2). The value never rides the projection. */
	secret: boolean;
	/** Position of the current question within a multi-question ask, so the card
	    can show "Question 1 of 2" and the user knows more follow (U1). */
	question_index: number;
	question_total: number;
}

export interface CompletionAttention {
	conversation_id: string;
	completion_token: string | null;
	anchor_id: string | null;
	kind: "complete" | "error" | "interrupted" | null;
	unseen: boolean;
	revision: [number, number];
}

export interface SessionProjection {
	attention?: CompletionAttention;
	session_id: string;
	pid: number;
	kind: string;
	conversation_name: string;
	cwd: string;
	model_label: string;
	/** provider/model_id — the model sheet's value. */
	model_selector: string;
	/** Current rung; "" when the model has no ladder. */
	effort: string;
	effort_ladder: string[];
	streaming: boolean;
	/** What the turn is doing right now, TUI-working-line style: "thinking",
	    "responding", or a running tool's intent. Empty when idle. */
	activity: string;
	/** Seconds since the activity began (server-computed). */
	activity_started_s: number;
	/** Why streaming last stopped — "completed" | "aborted" | "" before the
	    first turn ends. The resume affordance reads this, never an inference
	    from the streaming flag flipping. */
	stop_reason: string;
	/** User messages waiting for the turn boundary. */
	queued_count: number;
	/** Process gone; history still resumable. */
	ended: boolean;
	/** Record fresh but socket unreachable. */
	degraded: boolean;
	transcript: TranscriptEntry[];
	/** Todos grouped into phases. One implicit `"Todos"` phase carries a flat
	    list and renders without a header. */
	todos: TodoPhase[];
	subagents: SubagentRow[];
	pending: PendingRequest | null;
	/** How many requests are waiting in total (>= 1 while `pending` is set).
	    A parallel tool batch can open several approvals at once; the card
	    shows "1 of N" so the user knows more follow this one. */
	pending_count: number;
	/** input/output tokens. */
	usage: Record<string, number>;
	/** Projection epoch; drop stale repaints. */
	version: number;
}

export interface SessionSummary {
	session_id: string;
	section: "active" | "previous";
	conversation_name: string;
	cwd: string;
	model_label: string;
	streaming: boolean;
	needs_attention: boolean;
	/** A turn finished while no relay client was viewing the session, and it
	    has not been opened since. Renders the calm accent "new" mark (never
	    danger, never a pulse — those are reserved for decisions); cleared by
	    POST /api/sessions/{id}/seen when the session is opened. Older daemons
	    omit the field entirely, so readers must treat absence as false. */
	unseen?: boolean;
	pending_kind: "approval" | "ask" | "" | null;
	subagents_running: number;
	todos_open: number;
	mtime: number;
}

export interface SlashCommand {
	name: string;
	description: string;
	aliases: string[];
	arguments: "none" | "optional" | "required";
}

export interface ModelEntry {
	selector: string;
	provider: string;
	model_id: string;
	name: string;
}

export interface PastSession {
	id: string;
	name: string;
	mtime: number;
	/** True when this row matched only on what was SAID in the conversation,
	    not its name/id — the UI marks these so the hit doesn't look arbitrary. */
	body_match?: boolean;
	/** True while this session is a FORK still wearing the title it inherited
	    from its parent. Such a row is otherwise byte-identical to the parent's
	    — same name, same age — so the list tags it, exactly as the TUI's
	    /resume picker does. Clears the moment the fork names itself. */
	forked?: boolean;
}

export interface Directories {
	home: string;
	recent: string[];
	/** The system temp dir, offered as a scratch start directory. */
	tmp?: string;
}

/* ---- command ops (POST /api/sessions/{session_id}/command) --------------- */

/** A pasted / dropped image, base64 — the wire form the handles decode. */
export interface PromptImage {
	data_b64: string;
	mime_type: string;
}

export type CommandOp =
	| { op: "prompt" | "steer"; command_id: string; text: string; images?: PromptImage[] }
	| { op: "abort" }
	| { op: "set_model"; provider: string; model_id: string }
	| { op: "set_effort"; effort: string }
	| { op: "slash"; command: string; args: string }
	| { op: "new_conversation" }
	| { op: "resume_session"; session_id: string }
	| { op: "approval_answer"; request_id: string; approved: boolean; remember: boolean }
	| {
			op: "ask_answer";
			request_id: string;
			value: string;
			/** The question the card was showing when the user answered. The
			    daemon rejects the answer if the picker has advanced past it
			    (U8), so a tap in flight during a terminal advance is never
			    recorded against the wrong question. */
			question_index: number;
	  }
	| { op: "snapshot" };

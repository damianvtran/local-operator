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
	| "compaction";

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
	args?: string;
	output?: string;
	diff?: string;
	partial?: string;
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
	/** Assistant rows stream: `final` flips true on message_end. */
	final: boolean;
}

export interface TodoItem {
	text: string;
	status: TodoStatus;
	reason: string;
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
}

export interface PendingRequest {
	request_id: string;
	kind: "approval" | "ask";
	title: string;
	detail: string;
	/** Ask pickers; empty means free text. */
	options: string[];
}

export interface SessionProjection {
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
	/** User messages waiting for the turn boundary. */
	queued_count: number;
	/** Process gone; history still resumable. */
	ended: boolean;
	/** Record fresh but socket unreachable. */
	degraded: boolean;
	transcript: TranscriptEntry[];
	todos: TodoItem[];
	subagents: SubagentRow[];
	pending: PendingRequest | null;
	/** input/output tokens. */
	usage: Record<string, number>;
	/** Projection epoch; drop stale repaints. */
	version: number;
}

export interface SessionSummary {
	pid: number;
	kind: string;
	session_id: string;
	conversation_name: string;
	cwd: string;
	model_label: string;
	streaming: boolean;
	ended: boolean;
	degraded: boolean;
	needs_attention: boolean;
	pending_kind: "approval" | "ask" | "" | null;
	subagents_running: number;
	todos_open: string;
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
}

export interface Directories {
	home: string;
	recent: string[];
}

/* ---- command ops (POST /api/sessions/{pid}/command) ---------------------- */

export type CommandOp =
	| { op: "prompt"; text: string }
	| { op: "steer"; text: string }
	| { op: "abort" }
	| { op: "set_model"; provider: string; model_id: string }
	| { op: "set_effort"; effort: string }
	| { op: "slash"; command: string; args: string }
	| { op: "new_conversation" }
	| { op: "resume_session"; session_id: string }
	| { op: "approval_answer"; request_id: string; approved: boolean; remember: boolean }
	| { op: "ask_answer"; request_id: string; value: string }
	| { op: "snapshot" };

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
	| "peer_message"
	// The model's own PRIVATE reasoning, streamed while it thinks. Transient by
	// construction: it never joins the durable transcript, so the row is gone
	// after the next sync, and it is never the assistant's answer. Listed here
	// because this union mirrors `local_operator/mobile/types.py` field-for-field
	// -- the client renders it through its unknown-kind path today, which is what
	// it rendered before the runtime emitted reasoning at all.
	| "reasoning";

export type ToolState =
	| "composing"
	/* Announced and finished being written, waiting for its turn to execute. The
	   producer's terminal dictation frame sets it; `composing` would claim the
	   model is still writing a call it finished minutes ago, and `running` would
	   claim execution nothing has started. */
	| "queued"
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
	/* Severity ink for a `notice` row, mirroring the TUI's NoticeBlock kind.
	   A refusal, a failed turn and a failed compaction are `error`; an
	   unattended gate timeout and a declined compaction are `warning`.
	   Absent means the quiet default — most notices are receipts nobody has
	   to read, and tinting those would spend the loudest ink in the palette
	   on routine chrome. */
	severity?: "info" | "warning" | "error";
	/* A wake delivery, which the TUI gives its own affordance. Carried so
	   the phone can tell a wake receipt from an arbitrary notice rather
	   than flattening both into the same grey line. */
	notice_kind?: "wake";
	/* Bang-mode (`! cmd`): the user ran this command themselves, so its card
	   opens EXPANDED — they are waiting to read the output, not to be told a
	   command they typed has finished. Mirrors the TUI's ToolCard user_run. */
	user_run?: boolean;
}

export interface PeerSender {
	pid?: number;
	session_id?: string;
	conversation_name?: string;
	model_label?: string;
	cwd?: string;
}

export interface TranscriptEntry {
	/** Explicitly true only when transport caps retained the real row ending. */
	text_complete?: boolean;
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
	/** The child's age in seconds, or `null` when this roster has no age for it.
	 *
	 * `null` is NOT `0`: the roster computes an age only for a child whose job row
	 * carries a start, and the drill-in renders whatever it gets through the
	 * `WorkingLine` gate — where a plain `0` from an ageless child used to license
	 * a `0s` clock counting from the viewer's own mount, while the TUI withholds
	 * that number (design round 3, D8). Withholding is now expressible. */
	elapsed_s: number | null;
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
	/** Seconds since the activity began (server-computed), or `null` when the
	    server has no instant it can honestly date the phase from — a label the
	    fold joined mid-flight whose producer stated none. `null` means WITHHOLD
	    the digits, not `0`: a known zero is a real reading (`0.0`, the phase edge
	    the server watched begin) and paints `0s` and counts up. One nullable
	    number carries both because the client's only question is whether an
	    instant exists, and a value-plus-flag pair could disagree with itself. */
	activity_started_s: number | null;
	/** Why streaming last stopped — "completed" | "aborted" | "" before the
	    first turn ends. The resume affordance reads this, never an inference
	    from the streaming flag flipping. */
	stop_reason: string;
	/** True when that turn was CUT OFF by the harness rather than stopped on
	    purpose. The composer's word follows it, so the button agrees with the
	    danger notice above it. Optional because an older daemon omits the field
	    entirely; absent means false, which is also what a deliberate stop is. */
	cut_off?: boolean;
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
	/** The record's own phrase when the runtime has been SIGNALLED and is
	    finishing the work in flight before it exits; `""` otherwise (an older
	    daemon omits it, so readers normalise with `?? ""`).

	    RANKED HERE RATHER THAN DRAWN, and that is the point: a draining session
	    has children still running, so a mark derived from the counts alone would
	    advertise delegated work for a row the list itself is about to describe as
	    leaving. The daemon already refuses to report counts for such an entry
	    (`_advertisable_counts`); this field is the second guard for a client
	    talking to a build that predates that refusal. */
	leaving?: string;
	/** How many of this session's OWN delegated children are RUNNING. `null`
	    means the daemon did not report a count, and MUST NOT be read as zero: a
	    row that could not be asked must not be told "no subagents". The daemon
	    reports `null` for an entry it cannot vouch for (its dial is degraded, the
	    owner stopped beating, the runtime is leaving) as well as for a session
	    with no live record at all, and an older daemon omits the field — so
	    readers normalise with `typeof … === "number"`. */
	subagents_running?: number | null;
	/** Delegated children parked waiting for a capacity slot. Separate from the
	    running count for the reason the record keeps them apart: a parked child
	    spends nothing, but "queued with nothing running" is still not idle. The
	    two fields are declared on the same terms — both nullable AND optional —
	    because a reader that tolerates absence for one and not the other is a
	    reader whose two arms drift (review round 1, R2). */
	subagents_queued?: number | null;
	todos_open: number;
	mtime: number;
	/** Immutable conversation birth; absent on older daemons, never activity. */
	created_at?: number;
	/** Latest outcome; only unseen non-streaming outcomes affect list priority. */
	completion_kind?: string;
}

export interface SlashCommand {
	name: string;
	description: string;
	aliases: string[];
	arguments: "none" | "optional" | "required";
}

/** One offerable model, as `GET /api/models` ranks it.
 *
 * The array order IS the ranking — direct-connected providers first, newest
 * version first, aggregators last — computed server-side by the same
 * `rank_rows` the desktop `/model` picker uses. Anything here that re-sorts or
 * regroups the array throws that away, which is the bug these fields were added
 * alongside: the sheet grouped by provider and put ~445 Radient rows ahead of
 * the first direct provider.
 *
 * The fields below `name` are additive; `selector`, `provider`, `model_id` and
 * `name` keep the meanings they always had, so a stale cached bundle still
 * renders against a current daemon.
 *
 * The field set is deliberately WHAT THE PHONE RENDERS. An earlier revision also
 * shipped `routed`, `context_window`, `input_price` and `output_price` as a
 * forward contract; no `.tsx` read any of them and they cost 159 KB of a 301 KB
 * response on a mobile link. Add a field back here and on the daemon's row when
 * a surface actually renders it — a payload is not free just because it is
 * additive. */
export interface ModelEntry {
	selector: string;
	provider: string;
	model_id: string;
	/** The model's display name — the listing's own, falling back to its id. */
	name: string;
	/** The picker's resolved label, exactly as the desktop spells it — equal to
	    `selector` when no name can be vouched for (always so for a reseller,
	    whose listing names cannot say which route is answering). This is the
	    parity contract, not a display string; render `name`. */
	label?: string;
	/** Whether the provider has a credential that can run this model now. */
	connected?: boolean;
	/** The provider RESELLS this model rather than serving it; the direct route
	    for the same model ranks ahead of it. */
	aggregated?: boolean;
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
	/* `slash_result` is the ROUTED slash op — the one the runtime's authority seam
	   was built for, and the one the desktop backend and the TUI's attached pane
	   already use. `slash` is the off-terminal SUBSET (`/goal`, `/compact`) and
	   answers `/approvals` with "terminal-only here", which is the dead end this
	   phone surface exists to remove: `/approvals auto` is authority-increasing and
	   only reaches a sink through this op. Typed here so the refusal cannot come
	   back through the client. */
	| { op: "slash_result"; command: string; args: string; images?: PromptImage[] }
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

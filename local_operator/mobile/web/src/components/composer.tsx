/**
 * Composer — the bottom-docked input cluster: textarea, send/steer, stop,
 * queued count, plus the three sheets it can raise (slash commands, model,
 * effort rungs) and the post-abort "resume" row.
 *
 * Ergonomics: the textarea renders at 16px so iOS does not zoom on focus;
 * it auto-grows to six lines then scrolls. While the session is streaming
 * the send button switches to the `steer` op (same action, different
 * command) and a stop button appears beside it.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { getCommands, HttpError, sendCommand, transcribeAudio } from "../api";
import {
	getPendingContinuation,
	submitContinuation,
} from "../continuation-command";
import {
	annotationForSend,
	applyEdit,
	computeEdit,
	emptyProvenance,
	formatDuration,
	joinDraft,
	MAX_RECORDING_MS,
	noteDictation,
	pickRecorderMime,
	type DictationProvenance,
} from "../lib/dictation";
import {
	markPendingEchoAccepted,
	projectionCarriesCommand,
	registerPendingEcho,
	withdrawPendingEcho,
} from "../pending-echo";
import { Sheet } from "./ui/sheet";
import { WorkingDirectoryChip } from "./directory-sheet";
import { cn } from "../lib/cn";
import { useCapabilities, useDraft } from "../store";
import type { PromptImage, SessionProjection, SlashCommand } from "../types";

/** One attached image, kept as the wire form plus a local object URL for the
    thumbnail strip. */
interface AttachedImage extends PromptImage {
	/** Object URL for the thumbnail preview; revoked on remove/send. */
	preview: string;
}

/* Read a pasted/dropped image File into the wire form. Oversize images are
   downscaled first: a 12 MP phone photo is several MB of base64 that the
   provider would reject anyway, and the rebound the session would do on the
   way in costs the same pixels. 1568px matches the session's own bound. */
const MAX_IMAGE_EDGE = 1568;
async function fileToImage(file: File): Promise<AttachedImage | null> {
	if (!file.type.startsWith("image/")) return null;
	const preview = URL.createObjectURL(file);
	try {
		const bmp = await createImageBitmap(file);
		let { width, height } = bmp;
		const scale = Math.min(1, MAX_IMAGE_EDGE / Math.max(width, height));
		width = Math.round(width * scale);
		height = Math.round(height * scale);
		const canvas = document.createElement("canvas");
		canvas.width = width;
		canvas.height = height;
		const ctx = canvas.getContext("2d");
		if (!ctx) return null;
		ctx.drawImage(bmp, 0, 0, width, height);
		const blob: Blob | null = await new Promise((res) =>
			canvas.toBlob(res, file.type === "image/png" ? "image/png" : "image/jpeg", 0.9),
		);
		if (!blob) return null;
		const buf = await blob.arrayBuffer();
		let bin = "";
		const bytes = new Uint8Array(buf);
		for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
		return {
			data_b64: btoa(bin),
			mime_type: blob.type,
			preview,
		};
	} catch {
		URL.revokeObjectURL(preview);
		return null;
	}
}


/** Whether two paths name the same directory, allowing for a trailing slash.

    ONE spelling difference is worth absorbing here and no more: the daemon
    resolves its own input, so a trailing slash is the difference between what
    a reader types and what the projection publishes. Everything else (``~``,
    relative paths) is resolved above the client, which is why the bridge also
    yields on a changed projection rather than relying on this alone. */
function sameDirectory(a: string, b: string): boolean {
	const norm = (p: string) => (p.length > 1 ? p.replace(/\/+$/, "") : p);
	return norm(a) === norm(b);
}

/** Detect "/cmd args" at the very start of the draft — the slash trigger. */
function slashQuery(text: string): string | null {
	if (!text.startsWith("/")) return null;
	if (text.includes("\n")) return null;
	const space = text.indexOf(" ");
	return (space === -1 ? text.slice(1) : text.slice(1, space)).toLowerCase();
}

/** True while the draft is still just the command TOKEN — no arguments yet.

    U5 (mobile UX batch 1): the sheet opens on the token and yields the composer
    the moment a space arrives — `/delete x` is the user composing arguments,
    not a sheet query, and re-opening the sheet over that keystroke stole focus
    (and the keyboard) mid-typing.

    U13 (batch 2): "the moment a space arrives" covers both arrival paths. On a
    phone the space usually arrives in the SHEET'S FILTER (focus lives there
    after typing `/`), which used to strand the arguments in a filter matching
    nothing; the filter's onChange now hands the composed line back to the
    composer (see `onSlashSpace`), so both paths converge on the same draft and
    this predicate stays the single rule. */
function slashTokenOnly(text: string): boolean {
	return slashQuery(text) !== null && !text.includes(" ");
}

/** The voice state machine's four resting points (error surfaces as `idle` + copy). */
type DictationState = "idle" | "recording" | "transcribing";

/**
 * The failure copy for one transcription attempt.
 *
 * 402/413/422/503 carry the DAEMON's own actionable sentence (top up, shorten,
 * wrong format, no path) and are shown verbatim — the user can act on every
 * one. 502/500 and transport failures are transient or diagnostic, so they get
 * the one retry sentence; the server's upstream diagnostics belong in a log,
 * not on a phone.
 */
function dictationErrorCopy(error: unknown): string {
	if (error instanceof HttpError && [402, 413, 422, 503].includes(error.status)) {
		return error.message;
	}
	return "Couldn't transcribe that. Try again.";
}

/**
 * What a tap on a command row does: run it now, or wait for the text it takes.
 *
 * The catalogue's `arguments` field means "does a space open a value list", and
 * that is exactly the question here: a command with an argument surface needs
 * the user's text before it can mean anything, so the tap FILLS the composer and
 * stops; a command with none runs on the tap.
 *
 * Named and exported rather than left inline because the answer changed for two
 * commands without this file being touched: the backend moved `/goal` and
 * `/loop` from `none` to `optional` when their flag rows landed, so tapping
 * either now inserts `/goal ` / `/loop ` and waits where it used to run the bare
 * word. That is the same treatment `/rename`, `/theme`, `/stop`, `/mcp` and
 * `/approvals` already get — and for `/loop` it is the safer one, since the bare
 * word used to start iterations — but a shipped surface changing under a remote
 * field is exactly the kind of thing that should be pinned by a test, which is
 * what `composer-slash-tap.test.ts` does.
 */
export function tapFillsOnly(argumentsField: SlashCommand["arguments"]): boolean {
	return argumentsField !== "none";
}

function SlashSheet({
	open,
	onClose,
	onPick,
	onSpace,
	query,
}: {
	open: boolean;
	onClose: () => void;
	/** fill: text to place in the composer; submit: send immediately. */
	onPick: (fill: string, submit: boolean) => void;
	/** A space typed in the FILTER — the composed line belongs to the
	    composer now (U13, batch 2). Receives the filter's full value. */
	onSpace: (value: string) => void;
	/** The token after `/` in the composer — seeds the filter. */
	query: string;
}) {
	const [commands, setCommands] = useState<SlashCommand[]>([]);
	const [filter, setFilter] = useState(query);
	const [loaded, setLoaded] = useState(false);
	/* U8: focus lands on the FILTER, not the ✕. The sheet is opened by typing
	   `/…` on a phone; focusing the ✕ moved the caret out of every text field,
	   which closes the software keyboard, and the characters typed after `/`
	   reached nothing. The filter is already seeded with the token typed so far
	   — continuing to type filters, exactly what the seed implies. */
	const filterRef = useRef<HTMLInputElement>(null);

	useEffect(() => {
		if (!open) return;
		setFilter(query);
		setLoaded(false);
		getCommands()
			.then((r) => {
				setCommands(r.commands);
				setLoaded(true);
			})
			.catch(() => {
				setCommands([]);
				setLoaded(true);
			});
	}, [open, query]);

	const filtered = useMemo(() => {
		const q = filter.trim().toLowerCase();
		if (!q) return commands;
		return commands.filter(
			(c) =>
				c.name.toLowerCase().includes(q) ||
				c.aliases.some((a) => a.toLowerCase().includes(q)) ||
				c.description.toLowerCase().includes(q),
		);
	}, [commands, filter]);

	return (
		<Sheet open={open} onClose={onClose} title="commands" initialFocusRef={filterRef}>
			<div className="flex flex-col gap-1 p-2">
				<input
					ref={filterRef}
					value={filter}
					onChange={(e) => {
						const next = e.target.value;
						setFilter(next);
						/* U13 (batch 2): a space means the reader is composing
						   ARGUMENTS, which the filter cannot match (measured:
						   `delete x` → "no matching commands", the text surviving
						   only here, discarded on dismissal). Close and hand the
						   line back — the hand-off the sheet's contract already
						   promises for the draft path. */
						if (next.includes(" ")) onSpace(next);
					}}
					placeholder="filter commands"
					spellCheck={false}
					autoCapitalize="off"
					autoCorrect="off"
					className="mb-1 min-h-11 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim"
				/>
				{filtered.map((c) => (
					<button
						key={c.name}
						type="button"
						onClick={() => {
							/* Fill and wait when the command takes an argument; run it
							   outright when it takes none. See `tapFillsOnly`. */
							if (tapFillsOnly(c.arguments)) {
								onPick(`/${c.name} `, false);
							} else {
								onPick(`/${c.name}`, true);
							}
							onClose();
						}}
						className="flex min-h-11 items-center gap-2 rounded-sm px-2 text-left active:bg-surface"
					>
						<span className="shrink-0 font-mono text-mono-sm text-ink">
							/{c.name}
							{c.arguments === "required" ? (
								<span className="text-ink-dim"> …</span>
							) : null}
						</span>
						<span className="min-w-0 flex-1 truncate text-body-sm text-ink-dim">
							{c.description}
						</span>
					</button>
				))}
				{!loaded ? (
					<p className="px-3 py-2 text-body-sm text-ink-dim">
						loading…
					</p>
				) : filtered.length === 0 ? (
					<p className="px-3 py-2 text-body-sm text-ink-dim">
						no matching commands
					</p>
				) : null}
			</div>
		</Sheet>
	);
}

function EffortSheet({
	open,
	onClose,
	pid,
	projection,
}: {
	open: boolean;
	onClose: () => void;
	pid: string;
	projection: SessionProjection;
}) {
	const set = async (effort: string) => {
		try {
			await sendCommand(pid, { op: "set_effort", effort });
		} catch {
			/* The next repaint shows the truth; a failed set leaves it. */
		}
		onClose();
	};
	return (
		<Sheet open={open} onClose={onClose} title="effort">
			<div className="flex flex-col p-2">
				{projection.effort_ladder.map((rung) => (
					<button
						key={rung}
						type="button"
						onClick={() => set(rung)}
						className="flex min-h-11 items-center gap-2 rounded-sm px-2 text-left active:bg-surface"
					>
						<span
							className={cn(
								"size-2 shrink-0 rounded-full",
								rung === projection.effort
									? "bg-accent"
									: "bg-hairline",
							)}
							aria-hidden
						/>
						<span className="font-mono text-mono text-ink">
							{rung}
						</span>
					</button>
				))}
			</div>
		</Sheet>
	);
}

const MAX_TEXTAREA_PX = 6 * 22; /* six lines at body line-height */
const CONTINUATION_ERROR = "Couldn’t continue this conversation. Try again.";
/* U15's other half (UX round 1, U20; reworded for issue #1875): on an ENDED
   session the generic line's `Try again.` is not the whole story — the runtime
   is gone — and the old sentence ("tap resume to continue") was WRONG about the
   mechanism: a send to an ended session is itself resume-then-send (the daemon
   wakes a host for a prompt with no live entry and the continuation carries the
   text into it), so resume is never a prerequisite. A send that FAILED here means
   that wake failed, and the honest remedies are the two that exist: send again
   (the retained draft goes under the same envelope), or resume, which is the one
   thing a send cannot do — reopen the session without composing. (The composer
   is NOT disabled for an ended session: a disabled composer would change the
   draft, attachment and retained-envelope flows, and sending is a real way back.) */
const ENDED_CONTINUATION_ERROR =
	"This session has ended and couldn’t be woken just now. Tap “Retry earlier instruction” to send it again, or reopen it from the sessions list.";
const SLASH_ERROR = "Couldn’t run that command. Try again.";
const STEER_ERROR = "Couldn’t send this instruction. Try again.";
/* U5: one vocabulary for the retained instruction across the alert, the retry
   button, and the delivered acknowledgement — "earlier" throughout, matching
   the "Earlier instruction delivered." success line. */
const RETAINED_RETRY_ERROR =
	"An earlier instruction may have been delivered. Retry that earlier instruction before sending your current draft.";
const RETRY_BUTTON_LABEL = "Retry earlier instruction";
/* D11: the positive acknowledgement is a SUCCESS, not a failure — it renders in
   the neutral/success notice, never the danger alert container.

   U4: the sent draft is no longer the only thing that can be in the field when
   the receipt lands — a send now MOVES the draft out, so the common case is a
   SECOND message the user started while this one was in flight. "Your edited
   draft" named a noun that describes only the retry-of-a-retained-instruction
   case, which is now the rarer one. */
const RETRY_ACK_NOTICE = "Earlier instruction delivered. Your draft is ready to send.";
/* U4: the reason the primary send is disabled while an uncertain envelope is
   pending — stated inline so the dead control reads as intentional. */
const RETRY_DISABLED_HINT = "Resolve the earlier instruction first.";
const CONNECTING_STATUS = "Connecting…";
/* U3: a submit while a send is in flight is a NO-OP — the send control is
   disabled and `send()` returns on its `sending` guard — so the phone's return
   key (`enterKeyHint="send"`) answered nothing at all. The disabled control and
   the status line below were both already on screen, so the answer belongs in
   that line's own sentence rather than in a fourth element beside the row's
   caption and the button's glyph (UX round 1, U3, and nit N1 which asks for FEWER
   ways of saying one thing). It appears only when there IS a draft that cannot
   go, which is the moment the question is asked. */
const SENDING_HINT = `${CONNECTING_STATUS} — send again when this one lands.`;
/* D1: the placeholder must FIT the textarea's narrowest resting width (184px
   beside the mic at 390×844) or the empty box wraps to two lines and the
   first keystroke collapses it 45→24 with a visible reflow above the
   composer (design round 1, D1). The old string measured 188.5px at 16px
   system-ui — 4.5px over — this one measures 77.8px, which survives narrower
   viewports and font fallbacks too. The product name is carried by the app
   shell (document title, header mark). */
const COMPOSER_PLACEHOLDER = "Message…";

/* THE CLUSTER'S BOX, IN ONE PLACE. The composer is the biggest thing on this
   screen that arrives with the projection, and the conversation's first frame
   has to reserve exactly the room it will take (first-paint lane T2): the
   screen paints `<ComposerFrame>` before the projection lands, and a reserved
   box that does not match the real one is a layout shift with extra steps.
   Sharing the class strings is what makes the match structural rather than a
   pair of numbers someone keeps in step by hand — the frame and the composer
   cannot disagree about the shell, the input row, the field or the control size. */
const COMPOSER_SHELL =
	"flex flex-col gap-1.5 px-3 pt-1.5 pb-[max(env(safe-area-inset-bottom),0.5rem)]";
const COMPOSER_CWD_ROW = "flex min-w-0 flex-1 items-center px-0.5";
const COMPOSER_INPUT_ROW = "flex items-end gap-2";
const COMPOSER_FIELD_BOX =
	"flex min-w-0 flex-1 items-end rounded-md border bg-elevated px-3 py-2";
const COMPOSER_FIELD =
	"lo-scroll max-h-33 min-h-6 w-full resize-none bg-transparent text-[16px] leading-[1.4] text-ink outline-none placeholder:text-ink-muted";
const COMPOSER_ROUND = "flex size-11 shrink-0 items-center justify-center rounded-full";
/* The model/effort chips row, which is BELOW the field. It is easy to miss as
   part of the box and it is 44 px of it: a reserve without it left the field
   59 px lower than the real one, which is exactly the shift this whole frame
   exists to prevent (measured before it was added: field rect
   [64, 812, 262, 24] against the settled [77, 753, 236, 24]). */
const COMPOSER_CHIP_ROW = "flex items-center gap-2 px-0.5";
const COMPOSER_CHIP = "flex min-h-11 min-w-11 items-center font-mono text-mono-sm text-ink-dim";

/** The composer's box, empty, for the frames before a session's projection.

    WHY IT EXISTS. `SessionScreen` used to paint a header and one status
    sentence until the projection arrived, then mount the whole column in a
    single commit: the composer, the working line and the transcript all
    appeared at once, and everything below the header MOVED. The reserved frame
    keeps the column's geometry fixed so that arrival is a content fill.

    INERT ON PURPOSE, and it is not a second composer: there is no draft, no
    send, no attachment path and no data to act on yet, so `disabled` on the
    field is the honest state and a live field here would be a control that
    silently drops what the reader types. Nothing inside is focusable, and the
    whole box is `aria-hidden` — a screen reader gets the status line in the
    transcript area instead of an unlabelled row of dead controls.

    THE HEIGHT FLOOR UNDER THE FIELD is the working-directory chip's own 44 px
    (`ui/chip.tsx`), rendered as a plain reserve: the chip itself is a control
    that opens the directory sheet, and a dead button in a frame that is gone
    in ~50 ms is worse than a gap of the same size. The model/effort row BELOW
    the field is reserved the same way and for the same reason — it, too, is
    44 px the settled composer has and this one must hold.

    The field's own box is shared with the real composer through
    `COMPOSER_FIELD_BOX`: the settled field sits inside a bordered container
    (`px-3 py-2`), which is 24 px of width and 16 px of height the reserve
    would otherwise be missing. */
/* The attach disc's paperclip, drawn so it needs no icon font — ONE definition,
   shared by the settled composer and by its reserved frame, because design
   review round 1 (D1) measured the swap moving three glyphs and a placeholder
   into place when all four are static and knowable before any data arrives. The
   reserve is the settled empty composer or it is a different composer. */
function PaperclipGlyph() {
	return (
		<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
			<path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" />
		</svg>
	);
}

export function ComposerFrame() {
	return (
		<div className={COMPOSER_SHELL} aria-hidden>
			<div className={COMPOSER_CWD_ROW}>
				<div className="min-h-11" />
			</div>
			<div className={COMPOSER_INPUT_ROW}>
				{/* The settled empty composer's own three glyphs and placeholder, in
				   their DISABLED styling: the attach disc at `opacity-50`, the field
				   empty with `Message…`, the send disc `bg-sunken`
				   /`text-ink-disabled`. Design review round 1 (D1) measured the gap
				   this closes: the reserve read as "ring + outline" with the
				   right-hand control invisible (the sunken disc sits at 1.05:1
				   against the canvas) and then three glyphs and a placeholder
				   popped in at the reveal. */}
				<span
					className={cn(
						COMPOSER_ROUND,
						"border border-control text-ink-muted opacity-50",
					)}
				>
					<PaperclipGlyph />
				</span>
				<div className={cn(COMPOSER_FIELD_BOX, "border-control")}>
					<textarea
						disabled
						rows={1}
						aria-hidden
						tabIndex={-1}
						placeholder={COMPOSER_PLACEHOLDER}
						className={COMPOSER_FIELD}
					/>
				</div>
				<span
					className={cn(
						COMPOSER_ROUND,
						"bg-sunken text-ink-disabled",
					)}
				>
					<span aria-hidden>↑</span>
				</span>
			</div>
			<div className={COMPOSER_CHIP_ROW}>
				<span className={cn(COMPOSER_CHIP, "flex-1")} />
				<span className={COMPOSER_CHIP} />
			</div>
		</div>
	);
}

/* The dictation outcome lines (U2/U3/D2), rendered in the same polite status
   row the live states use. DICTATION_ADDED is the completion announcement:
   focus is deliberately NOT returned to the field with it — a programmatic
   focus() on iOS pops the keyboard over wherever the user moved on to — so a
   screen reader hears the line while a sighted user taps the field to keep
   typing. DICTATION_EMPTY answers a transcript that came back empty; the
   discarded line surfaces the dictation a send/steer just dropped, which used
   to be silent. */
const DICTATION_ADDED = "Transcript added";
const DICTATION_EMPTY = "Didn't catch that — try again.";
const DICTATION_DISCARDED = "Voice input discarded.";

export function Composer({
	pid,
	projection,
	onOpenModels,
	onOpenEffort,
	effortOpen,
	onCloseEffort,
	autoFocus = false,
}: {
	/** Route pid — the discovery record's, not the fold's (which stamps 0). */
	pid: string;
	projection: SessionProjection;
	onOpenModels: () => void;
	onOpenEffort: () => void;
	effortOpen: boolean;
	onCloseEffort: () => void;
	/** Whether the textarea takes focus once this composer mounts.

	    A MOUNT-TIME ONE-SHOT rather than the DOM ``autoFocus`` attribute, and the
	    difference matters here: this composer mounts only AFTER the session's
	    projection arrives (the screen renders a waiting state until then), so the
	    caller's intent — "this conversation was just created FOR the user to type
	    into" — has to survive the gap between the tap and the mount. It is a
	    one-shot flag the caller consumes (see ``lib/pending-focus.ts``), so an
	    ordinary navigation into an existing conversation never steals focus.

	    BEST-EFFORT for the phone's keyboard, and accepted as such: on iOS the
	    initial tap IS the gesture that raises the keyboard, and a programmatic
	    ``focus()`` later in the same response may not raise it again. The
	    alternative — forcing a keyboard open from script — fights the platform
	    and is worse than a focused field with a closed keyboard. */
	autoFocus?: boolean;
}) {
	const [text, setText] = useDraft(pid);
	/* U5/U8 (mobile UX batch 1): the sheet follows the draft only while the draft
	   is still the command TOKEN (see `slashTokenOnly`), and once the user has
	   closed it for this draft it stays closed until the draft is no longer a
	   slash draft at all — so typing arguments is never fought by a re-opening
	   sheet, while a fresh `/x` still opens it. */
	const [slashDismissed, setSlashDismissed] = useState(false);
	const slashOpen = slashTokenOnly(text) && !slashDismissed;
	const [sending, setSending] = useState(false);
	const [retryEnvelope, setRetryEnvelope] = useState(() => getPendingContinuation(pid));
	const retryPending = retryEnvelope !== null;
	const [error, setError] = useState(() => retryPending ? RETAINED_RETRY_ERROR : "");
	/* Success acknowledgement lives apart from `error` so it renders in the
	   success token, not the danger alert (D11). */
	const [notice, setNotice] = useState("");
	/* WHICH CONTAINER the notice paints in (design round 1, D1). A receipt reports
	   an OUTCOME, and the runtime leaves a completed write and a plain report at
	   the same `style="info"` — so the daemon classifies and the tone travels on
	   the reply. The success wash stays for a real acknowledgement (the delivered
	   instruction above), which is what it was built for. */
	const [noticeTone, setNoticeTone] = useState<"success" | "neutral">("success");
	const [images, setImages] = useState<AttachedImage[]>([]);
	const textRef = useRef(text);
	const imagesRef = useRef(images);
	textRef.current = text;
	imagesRef.current = images;
	const [dragOver, setDragOver] = useState(false);
	const textareaRef = useRef<HTMLTextAreaElement>(null);
	const fileInputRef = useRef<HTMLInputElement>(null);

	/* THE MOVE'S LOCAL BRIDGE (see ``directory-sheet.tsx``). The directory route
	   answers only once the successor runtime is ready, and the successor's
	   first projection lands a scan tick later; without this the chip would keep
	   naming the OLD directory for that second, which reads as "the change did
	   not take".

	   IT YIELDS ON THE SUCCESSOR'S OWN ANSWER, not on string equality (review
	   round 1, nit 4). Equality only holds when the spelling the client sent is
	   the spelling the daemon resolves to, which is NOT guaranteed -- a typed
	   ``~`` path or a relative one is resolved server-side, so the strings
	   differ and an equality test would hold the optimistic value for the life
	   of the page, leaving the chip disagreeing with the session it describes.
	   The value the projection carried BEFORE the move is the discriminator: a
	   projection that now names the directory we asked for (any spelling), or
	   any directory other than that pre-move one, is the successor answering.
	   What the chip shows is therefore always either the session's real cwd or a
	   directory the daemon has just confirmed. */
	const [movedCwd, setMovedCwd] = useState("");
	const cwdBeforeMove = useRef("");
	const projectedCwd = projection.cwd;
	const shownCwd = movedCwd || projectedCwd;
	const requestMove = (target: string) => {
		cwdBeforeMove.current = projectedCwd;
		setMovedCwd(target);
	};
	useEffect(() => {
		if (!movedCwd || !projectedCwd) return;
		if (sameDirectory(projectedCwd, movedCwd) || projectedCwd !== cwdBeforeMove.current) {
			setMovedCwd("");
		}
	}, [movedCwd, projectedCwd]);

	useEffect(() => {
		if (!autoFocus) return;
		textareaRef.current?.focus();
	}, [autoFocus]);

	/* ---- voice dictation (mobile STT) ------------------------------------ */

	/* The mic exists iff the daemon knows a path that can RUN and this browser
	   can record one: a secure context, getUserMedia and MediaRecorder. The old
	   plain-HTTP case is deliberately excluded — a mic that appears and fails is
	   worse than one that never does. */
	const capabilities = useCapabilities();
	const micAvailable =
		capabilities?.stt?.available === true &&
		typeof window !== "undefined" &&
		window.isSecureContext === true &&
		typeof navigator !== "undefined" &&
		!!navigator.mediaDevices?.getUserMedia &&
		typeof MediaRecorder !== "undefined";
	const [dictation, setDictation] = useState<DictationState>("idle");
	/* The transient outcome line (added / empty / discarded); lives in the same
	   status row as the live states so nothing moves between them (D3). */
	const [dictationNotice, setDictationNotice] = useState("");
	const dictationRef = useRef<DictationState>("idle");
	dictationRef.current = dictation;
	const [recordSeconds, setRecordSeconds] = useState(0);
	const recorderRef = useRef<MediaRecorder | null>(null);
	const streamRef = useRef<MediaStream | null>(null);
	const chunksRef = useRef<Blob[]>([]);
	const dictationTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
	const dictationCapRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	/* An epoch per attempt: the permission prompt is async, so a cancel or a
	   send during it must refuse the stream when it arrives (and a stale result
	   must never append to a draft it did not record into). */
	const dictationEpochRef = useRef(0);
	const dictationActiveRef = useRef(false);
	const dictationCancelledRef = useRef(false);
	/* U5: the in-flight transcribe request, so the status row's cancel aborts
	   it rather than only hiding the wait. Null whenever nothing is in flight. */
	const transcribeAbortRef = useRef<AbortController | null>(null);
	/* U1: set when a transcript has just been committed, consumed by the [text]
	   layout effect to reveal the appended span (scroll, never focus). */
	const revealAppendRef = useRef(false);
	/* The send window's provenance (see lib/dictation.ts). A ref, not state: it
	   changes with every keystroke and renders nothing itself. */
	const provenanceRef = useRef<DictationProvenance>(emptyProvenance());

	/* U1: what the retry button is about to resend.
	 *
	 * The withdrawn row took the failed message's only visible copy with it, and
	 * the composer's restore is deliberately gated on an EMPTY field (text typed
	 * since is the user's) — so the operator who carried on typing was being asked
	 * to retry a message whose words appeared nowhere on screen, while the empty
	 * state said "no messages yet". The first line names WHICH instruction; CSS
	 * clips it to whatever width the alert has, and the body itself stays in the
	 * envelope behind the button. An image-only instruction has no first line, so
	 * the count stands in for it — the same words the pending row uses. */
	const retainedPreview = (() => {
		if (!retryEnvelope) return "";
		const firstLine = retryEnvelope.text.split("\n")[0]?.trim() ?? "";
		if (firstLine) return firstLine;
		const attached = retryEnvelope.images?.length ?? 0;
		if (attached === 1) return "1 image attached";
		return attached > 1 ? `${attached} images attached` : "";
	})();

	/* The resume affordance is driven by the WIRE fact (stop_reason), not an
	   inference from the streaming flag: a turn that completes also flips
	   streaming off, and only an aborted turn should offer "resume".

	   AN ENDED SESSION YIELDS TO THE STRIP'S RESUME (UX round 1, U15). `ended`
	   and `stop_reason="aborted"` coexist on the shape the daemon serves after
	   a mid-turn death (the terminal repaint fills the end from the durable
	   record), and this button was the SECOND resume path — the prominent one,
	   in the thumb zone — sending `continue` to a runtime that is gone, over
	   and over. On an ended session the strip's `resume` is the one action: it
	   is the path that respawns the session, and one act must not read as two
	   different controls. */
	const showResume = projection.stop_reason === "aborted" && !projection.ended;

	/* WHICH word, when it does, follows the verdict the notice above it states.
	   `aborted` covers a deliberate stop and a harness cut-off alike, and the
	   red `Stopped with an error — ...` row sitting a few rows up made
	   `interrupted — tap to resume` name one act two ways (design round 2, D7).
	   `cut_off` is absent from an older daemon's payload, so absence keeps
	   today's word rather than inventing a verdict nobody sent. */
	const resumeLabel = projection.cut_off
		? "turn cut off — tap to resume"
		: "interrupted — tap to resume";

	/* Auto-grow: reset to auto so shrink works, then clamp at six lines; when a
	   transcript just landed, also reveal it (U1). The reveal is a plain
	   scrollTop assignment on the field — never focus(), which on iOS would
	   pop the keyboard over whatever the user was doing while it transcribed. */
	useEffect(() => {
		const el = textareaRef.current;
		if (!el) return;
		el.style.height = "auto";
		el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_PX)}px`;
		if (revealAppendRef.current) {
			revealAppendRef.current = false;
			el.scrollTop = el.scrollHeight;
		}
	}, [text]);

	const disabled = false;

	useEffect(() => {
		/* Envelopes are scoped per session and survive navigation (U1): moving to
		   another of my conversations must NOT delete this one's uncertain
		   instruction. On mount for a route, restore its own pending envelope so
		   returning to it re-shows the recovery affordance. Cross-session storage
		   is bounded by count inside the continuation module, and logout/401 is
		   what clears every route's private state — not navigation. */
		const pending = getPendingContinuation(pid);
		setRetryEnvelope(pending);
		setError(pending ? RETAINED_RETRY_ERROR : "");
	}, [pid]);

	const addFiles = async (files: Iterable<File>) => {
		for (const f of files) {
			const img = await fileToImage(f);
			if (img) setImages((cur) => [...cur, img]);
		}
	};

	/* The attach button's file picker. The pipeline is image-only today
	   (paste/drop already are), so the input scopes to images; a non-image
	   pick is ignored rather than sent as a broken base64 block. */
	const onPickFiles = (list: FileList | null) => {
		if (!list) return;
		void addFiles(Array.from(list));
		/* Reset so picking the SAME file twice still fires change. */
		if (fileInputRef.current) fileInputRef.current.value = "";
	};

	const removeImage = (preview: string) => {
		setImages((cur) => {
			const hit = cur.find((i) => i.preview === preview);
			if (hit) URL.revokeObjectURL(hit.preview);
			return cur.filter((i) => i.preview !== preview);
		});
	};

	/* ---- the dictation state machine -------------------------------------- */

	const releaseDictationTracks = () => {
		streamRef.current?.getTracks().forEach((track) => track.stop());
		streamRef.current = null;
	};

	const clearDictationTimers = () => {
		if (dictationTimerRef.current) {
			clearInterval(dictationTimerRef.current);
			dictationTimerRef.current = null;
		}
		if (dictationCapRef.current) {
			clearTimeout(dictationCapRef.current);
			dictationCapRef.current = null;
		}
	};

	/** Stop the in-flight dictation and DISCARD it: tracks released, timers
	    cleared, any in-flight result dropped when it lands. Idempotent, and
	    safe while the permission prompt is still up (the epoch refuses the
	    stream when it resolves). */
	const cancelDictation = () => {
		dictationCancelledRef.current = true;
		dictationActiveRef.current = false;
		dictationEpochRef.current++;
		clearDictationTimers();
		/* U5: a transcription in flight is cancelled for REAL — the abort reaches
		   the request, not just the UI. A no-op when nothing is in flight. */
		transcribeAbortRef.current?.abort();
		transcribeAbortRef.current = null;
		const recorder = recorderRef.current;
		recorderRef.current = null;
		if (recorder && recorder.state !== "inactive") {
			try {
				recorder.stop();
			} catch {
				/* Already stopping. */
			}
		}
		releaseDictationTracks();
		chunksRef.current = [];
		setDictation("idle");
		setRecordSeconds(0);
		setDictationNotice("");
	};

	const startDictation = async () => {
		if (dictationRef.current !== "idle" || !micAvailable) return;
		setError("");
		setNotice("");
		setDictationNotice("");
		const epoch = ++dictationEpochRef.current;
		dictationCancelledRef.current = false;
		dictationActiveRef.current = true;
		let stream: MediaStream;
		try {
			stream = await navigator.mediaDevices.getUserMedia({
				audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
			});
		} catch (e) {
			if (epoch !== dictationEpochRef.current) return;
			dictationActiveRef.current = false;
			setError(
				e instanceof DOMException && e.name === "NotAllowedError"
					? "Microphone access is blocked for this site. Allow it in your browser settings and try again."
					: "Couldn't start recording. Check your microphone and try again.",
			);
			return;
		}
		if (epoch !== dictationEpochRef.current) {
			/* A cancel (or a send) resolved while the permission prompt was up. */
			stream.getTracks().forEach((track) => track.stop());
			return;
		}
		let recorder: MediaRecorder;
		try {
			const mimeType = pickRecorderMime((mime) => MediaRecorder.isTypeSupported(mime));
			recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
		} catch {
			stream.getTracks().forEach((track) => track.stop());
			dictationActiveRef.current = false;
			setError("Couldn't start recording. Check your microphone and try again.");
			return;
		}
		streamRef.current = stream;
		recorderRef.current = recorder;
		chunksRef.current = [];
		recorder.ondataavailable = (event) => {
			if (event.data && event.data.size > 0) chunksRef.current.push(event.data);
		};
		recorder.onstop = () => {
			void finishDictation(recorder);
		};
		recorder.start(1000);
		setRecordSeconds(0);
		setDictation("recording");
		dictationTimerRef.current = setInterval(() => setRecordSeconds((s) => s + 1), 1000);
		/* The 120 s cap stops the recorder and TRANSCRIBES what it has — the cap
		   is a bound on the recording, not a discard. */
		dictationCapRef.current = setTimeout(() => {
			if (recorderRef.current === recorder) stopDictation();
		}, MAX_RECORDING_MS);
	};

	const stopDictation = () => {
		const recorder = recorderRef.current;
		if (!recorder || recorder.state === "inactive") return;
		clearDictationTimers();
		setDictation("transcribing");
		try {
			recorder.stop();
		} catch {
			/* Racing the cap timer's own stop. */
		}
	};

	const finishDictation = async (recorder: MediaRecorder) => {
		clearDictationTimers();
		releaseDictationTracks();
		if (recorderRef.current === recorder) recorderRef.current = null;
		const chunks = chunksRef.current;
		chunksRef.current = [];
		if (dictationCancelledRef.current) return;
		if (chunks.length === 0) {
			dictationActiveRef.current = false;
			setDictation("idle");
			setRecordSeconds(0);
			return;
		}
		const blob = new Blob(chunks, { type: recorder.mimeType });
		/* U5: the request is abortable, so the status row's own cancel reaches
		   the network rather than only hiding the wait. */
		const controller = new AbortController();
		transcribeAbortRef.current = controller;
		let result: { text: string; path: string };
		try {
			result = await transcribeAudio(blob, controller.signal);
		} catch (e) {
			if (dictationCancelledRef.current) return;
			dictationActiveRef.current = false;
			setError(dictationErrorCopy(e));
			setDictation("idle");
			setRecordSeconds(0);
			return;
		} finally {
			/* Only THIS attempt's controller is unmounted; the cancel or a fresher
			   attempt may already own the ref. */
			if (transcribeAbortRef.current === controller) transcribeAbortRef.current = null;
		}
		/* A send (or unmount) may have cancelled while the request was out: the
		   result is DISCARDED — appending it after the draft moved out would be
		   a clobber by another name. */
		if (dictationCancelledRef.current) return;
		dictationActiveRef.current = false;
		const previous = textRef.current;
		const joined = joinDraft(previous, result.text);
		const cleaned = result.text.trim();
		if (cleaned !== "" && joined !== previous) {
			provenanceRef.current = noteDictation(provenanceRef.current, {
				start: joined.length - cleaned.length,
				end: joined.length,
				path: result.path ?? "",
			});
			/* U1: the [text] layout effect reveals the appended span after React
			   commits it — scroll, never focus. */
			revealAppendRef.current = true;
			setText(joined);
			/* U3: a polite completion line, the announcement a screen reader was
			   missing; see the DICTATION_ADDED comment for the focus tradeoff. */
			setDictationNotice(DICTATION_ADDED);
		} else if (cleaned === "") {
			/* D2: silence (or a cut-off) is answered instead of the silent
			   return to idle the design review caught. */
			setDictationNotice(DICTATION_EMPTY);
		}
		setDictation("idle");
		setRecordSeconds(0);
	};

	/** Clear the draft AND its provenance window: a fresh message starts clean. */
	const clearDraft = () => {
		provenanceRef.current = emptyProvenance();
		setText("");
	};

	/** Classify one USER-driven draft change (typing, a slash fill): sticky
	    `sawTyping`, spans trimmed, reset when the draft empties. */
	const applyUserEdit = (value: string) => {
		const edit = computeEdit(textRef.current, value);
		if (edit) provenanceRef.current = applyEdit(provenanceRef.current, edit);
		if (value.trim() === "") provenanceRef.current = emptyProvenance();
	};

	/* Navigating away must not leave the microphone hot (or a permission
	   prompt pending): the cleanup releases tracks and discards any in-flight
	   result. */
	useEffect(() => {
		return () => cancelDictation();
		// eslint-disable-next-line react-hooks/exhaustive-deps -- refs + stable setters only
	}, []);

	const send = async (raw: string, op?: "prompt" | "steer") => {
		/* SEND CANCELS AN IN-FLIGHT DICTATION: tracks released and a result still
		   in the air discarded (aborted, U5). A transcript landing after this
		   message left the composer is a clobber by another name — the
		   transcript belongs to the NEXT message, and the next tap of the mic
		   starts one. THE DROP IS SAID (U2): a brief line in the status row,
		   because speech the user just gave otherwise vanishes with nothing said,
		   and send/steer is not a deliberate cancel. */
		setDictationNotice("");
		if (dictationActiveRef.current) {
			cancelDictation();
			setDictationNotice(DICTATION_DISCARDED);
		}
		const trimmed = raw.trim();
		if ((!trimmed && images.length === 0 && !retryPending) || sending || disabled) return;
		setSending(true);
		setError("");
		setNotice("");
		/* The echo the prompt branch paints, held so the failure branch can take it
		   back down and hand the text over again. A HOLDER rather than a plain
		   local because the callback that fills it runs in a closure TS's
		   control-flow analysis cannot follow — a `let x: T | null = null` reads as
		   permanently `null` at the catch. `null` on every other route: a slash
		   command paints no row. */
		const submitted: { echo: { commandId: string; text: string } | null } = { echo: null };
		/* A refused COMMAND and a failed CONTINUATION need different words: the first
		   carries the runtime's own refusal sentence, the second is the retry copy. */
		const isSlash =
			trimmed.startsWith("/") && !trimmed.includes("\n") && images.length === 0;
		try {
			/* Slash input routes to the slash op rather than prompt — and only
			   when there is no attachment, since a "/…" caption with an image
			   is a prompt, not a command. */
			if (isSlash) {
				const space = trimmed.indexOf(" ");
				const command =
					space === -1 ? trimmed.slice(1) : trimmed.slice(1, space);
				const args = space === -1 ? "" : trimmed.slice(space + 1);
				/* THE ROUTED OP, not `slash`. `slash` is the owner's off-terminal subset
				   (`/goal`, `/compact`) and refuses every other word with "terminal-only
				   here"; `slash_result` is the seam the runtime's dispatcher answers for
				   everything the sheet offers (the daemon builds that catalogue from the
				   same scope table, so the two cannot disagree), and the one the gate
				   sheet already uses. Plain `sendCommand`, not the proof variant: only
				   `/approvals auto|off|yolo` is authority-increasing, and its signed path
				   lives in the gate sheet — typed here it is REFUSED with the runtime's own
				   sentence, which `humanizeGateError` carries to the alert below. */
				const receipt = await sendCommand(pid, {
					op: "slash_result",
					command,
					args,
					images: [],
				});
				/* THE DRAFT IS KEPT WHEN THE RUNTIME REFUSED THE ARGUMENT (UX round 1,
				   U3). `/model nonexistent` answers 200 with "usage: /model <provider>/
				   <model-id>" — the command ran and declined — and clearing the field
				   there cost the reader the whole line to fix one word, while the 422
				   path keeps it. One rule, both paths: keep what the user typed until
				   the command actually did something. */
				if (!receipt.refused) clearDraft();
				/* What the command DID, in the runtime's words — a run that says nothing
				   reads as a dead tap. */
				setNotice(receipt.detail);
				setNoticeTone(receipt.tone === "success" ? "success" : "neutral");
				return;
			} else {
				const chosen =
					op ?? (projection.streaming ? "steer" : "prompt");
				const payloadImages = images.length
					? images.map(({ data_b64, mime_type }) => ({ data_b64, mime_type }))
					: undefined;
				/* A retry's operation belongs to the immutable envelope. Streaming may
				   change between admission and acknowledgement, so neither transport nor
				   correlation may derive prompt/steer from the current repaint. */
				const submittedEnvelope = retryEnvelope ?? {
					op: chosen,
					text: trimmed,
					images: payloadImages,
				};
				const receipt = await submitContinuation(
					pid,
					submittedEnvelope.op,
					trimmed,
					payloadImages,
					/* The annotation is consulted only for a FRESH send (a retry replays
					   the stored envelope's own bytes, annotation included). */
					annotationForSend(provenanceRef.current),
					/* The user's message leaves the composer HERE, into a row of its own,
					   before the daemon has answered — the whole point of the optimistic
					   path. It is painted under the ENVELOPE's id rather than a fresh one,
					   because that id is also the id the session will write the real row
					   under, which is what reconciles the two. */
					(envelope) => {
						/* A retry of an envelope the session already wrote paints nothing:
						   the projection owns that row and a second one under the same id is
						   the duplicate this reconciliation exists to prevent. */
						if (projectionCarriesCommand(projection.transcript, envelope.command_id)) return;
						registerPendingEcho(pid, {
							commandId: envelope.command_id,
							text: envelope.text,
							imageCount: envelope.images?.length ?? 0,
							/* The envelope's op, not the local `chosen`: they agree today, and the
							   envelope is the immutable identity the row is painted for. */
							op: envelope.op,
							accepted: false,
						});
						submitted.echo = { commandId: envelope.command_id, text: envelope.text };
						/* The draft leaves the composer only when the envelope IS the body it
						   holds. A retry replays the RETAINED envelope, whose body can differ
						   from a draft typed since — the exact case `RETRY_ACK_NOTICE` exists
						   for — and clearing the composer there would discard that draft, the
						   loss `action_stop` forbids. */
						if (
							envelope.text === trimmed &&
							JSON.stringify(envelope.images) === JSON.stringify(payloadImages)
						) {
							clearDraft();
						}
					},
				);
				setRetryEnvelope(null);
				/* The session admitted this command — the receipt IS that fact, and it is
				   what lets the row stop saying it is still going out. A no-op when the
				   projection already answered the echo, or when none was painted. */
				markPendingEchoAccepted(pid, receipt.envelope.command_id);
				const currentPayloadImages = imagesRef.current.length
					? imagesRef.current.map(({ data_b64, mime_type }) => ({ data_b64, mime_type }))
					: undefined;
				const acknowledgedCurrentDraft =
					receipt.envelope.op === submittedEnvelope.op &&
					/* An EMPTY composer is covered by any acknowledgement — there is nothing
					   left in it for the ACK to be behind. That is the ordinary path now, the
					   send having moved the draft into its own row; the comparison is what is
					   left of the original predicate and it still catches the case it was
					   written for, a message typed while this one was in flight. */
					(textRef.current.trim() === "" ||
						receipt.envelope.text === textRef.current.trim()) &&
					JSON.stringify(receipt.envelope.images) === JSON.stringify(currentPayloadImages);
				if (acknowledgedCurrentDraft) {
					imagesRef.current.forEach((i) => URL.revokeObjectURL(i.preview));
					setImages([]);
					clearDraft();
				} else {
					/* The acknowledged envelope covers the submission, but NOT everything
					   the user is looking at: they started another message while this one was
					   in flight, or attached something to it. Keep that as the next command
					   rather than presenting an idempotent ACK as delivery of content the
					   owner never received. This is a positive outcome, so it renders in the
					   success notice, not the danger alert (D11). The pending row stays up
					   either way — it is a row the session genuinely owes. */
					setNotice(RETRY_ACK_NOTICE);
				}
				return;
			}
			clearDraft();
		} catch (failure) {
			if (isSlash) {
				/* The runtime's refusal (a bad argument, `/approvals auto` without a
				   signature) is the answer — the daemon sends it as a 422 whose message is its
				   own sentence; anything else (a dropped connection) gets the one retry
				   line, never the raw fetch string. The draft stays so it can be
				   corrected. There is no envelope, echo or retry to unwind here. */
				setError(
					failure instanceof HttpError && failure.status === 422 ? failure.message : SLASH_ERROR,
				);
				return;
			}
			/* Previous conversations can fail at every layer between fetch and
			   provider construction. Those mechanics are intentionally invisible:
			   retain the exact draft, images, and command id for a safe retry while
			   giving every failure one actionable, USER-facing product message.
			   The raw fetch string ("Load failed") must never surface — that was
			   the developer-worded first impression of U3. Every non-streaming
			   send, tui-originated or daemon, reads as "couldn't send/continue";
			   only a live steer uses the steer copy.

			   The pending row comes DOWN with it. A row left standing would be a
			   message the user believes went — the phantom this path exists to
			   refuse — and the composer's alert below is the honest state from here:
			   it names the failure and offers the retry under the SAME envelope id,
			   which is also what keeps an ambiguous 408/502/504 from being reported as
			   a refusal. The text goes back with it, because the submit moved it out
			   of the composer and this is the only copy the user has left. */
			if (submitted.echo) withdrawPendingEcho(pid, submitted.echo.commandId);
			/* Only into an empty composer, on the same reading the acknowledgement
			   uses: text the user has typed since is theirs, and the retained envelope
			   behind the retry button still carries the body this submit sent. */
			if (submitted.echo && textRef.current.trim() === "") {
				/* A restored draft opens a FRESH provenance window: the retained
				   envelope behind the retry button still carries the failed send's own
				   annotation, which is what a retry replays. */
				provenanceRef.current = emptyProvenance();
				setText(submitted.echo.text);
			}
			setRetryEnvelope(getPendingContinuation(pid));
			setError(
				projection.ended
					? ENDED_CONTINUATION_ERROR
					: projection.streaming
						? STEER_ERROR
						: CONTINUATION_ERROR,
			);
		} finally {
			setSending(false);
		}
	};

	const abort = async () => {
		try {
			await sendCommand(pid, { op: "abort" });
		} catch (e) {
			setError(String((e as Error).message ?? e));
		}
	};

	const onChange = (value: string) => {
		applyUserEdit(value);
		setText(value);
		/* Typing is the next action: a lingering dictation outcome line is done
		   being useful. */
		setDictationNotice("");
	};

	/* The sheet ALSO watches the value, because driver/IME paths set it without
	   an onChange. The only piece of retained state is the DISMISSAL: a draft
	   that stops being a slash draft at all (emptied, or the slash deleted)
	   clears it, and so does a bare `/` — while within one slash draft, closing
	   the sheet (Escape, scrim, ✕, a pick, the filter's space hand-off) keeps
	   it closed (U5).

	   THE BARE SLASH IS A FRESH QUERY (batch 2, agent review MINOR 2). The
	   dismissal used to outlive the draft's own token: after a pick or Escape
	   on `/delete`, backspacing `/delete` → `/` left the sheet shut, so wanting
	   a different command meant deleting the slash itself. Deciding this
	   together with U13 settled it: the space hand-off parks arguments in the
	   composer for the CURRENT command, and the moment that draft returns to
	   just `/` the reader is starting over — which is what typing `/` fresh
	   does. Mid-token drafts stay dismissed (the strict half U5 pinned). */
	useEffect(() => {
		const q = slashQuery(text);
		if (q === null || q === "") setSlashDismissed(false);
	}, [text]);


	const onSlashPick = (fill: string, submit: boolean) => {
		applyUserEdit(fill);
		setText(fill);
		/* The pick CLOSES the sheet for this draft: an argument-taking command
		   lands mid-edit ("/rename ") and the sheet must not come back while the
		   arguments are typed (U5). */
		setSlashDismissed(true);
		if (submit) {
			void send(fill);
		} else {
			textareaRef.current?.focus();
		}
	};

	/* U13 (batch 2): a space typed while the FILTER has focus hands the composed
	   line back to the composer. The filter holds the command as typed
	   (`delete`), so the composed draft is `/` + that value including the space
	   the reader just typed; focus returns to the field they will keep typing
	   in, and the dismissal covers the backspace that removes the space again
	   (no mid-edit re-open) until the draft is a bare `/`. */
	const onSlashSpace = (value: string) => {
		const composed = `/${value}`;
		applyUserEdit(composed);
		setText(composed);
		setSlashDismissed(true);
		textareaRef.current?.focus();
	};

	return (
		<div className={COMPOSER_SHELL}>
			{showResume && !projection.streaming && !disabled ? (
				<button
					type="button"
					onClick={() => void send("continue", "prompt")}
					className="flex min-h-11 items-center justify-center rounded-sm border border-control bg-surface text-body-sm text-ink active:bg-elevated"
				>
					{resumeLabel}
				</button>
			) : null}

			{sending ? (
				<p role="status" aria-live="polite" className="text-body-sm text-ink-muted">
					{text.trim() ? SENDING_HINT : CONNECTING_STATUS}
				</p>
			) : null}

			{notice ? (
				/* D11: a delivered acknowledgement is a success — neutral/success
				   token, never the danger container the failure alert uses. D1: a
				   COMMAND RECEIPT is not a success — the runtime marks a refusal and
				   a report the same way it marks a completed write, so a receipt
				   paints in the neutral surface unless the daemon named it a real
				   mutation. The distinction is then carried by the surface rather
				   than by colour alone. */
				<div
					className={cn(
						"rounded-sm border px-3 py-2 text-body-sm",
						noticeTone === "success"
							? "border-success-border bg-success-wash text-success"
							: "border-control bg-elevated text-ink",
					)}
				>
					<p role="status" aria-live="polite">{notice}</p>
				</div>
			) : null}

			{error ? (
				<div className="rounded-sm border border-danger-border bg-danger-wash px-3 py-2 text-body-sm text-danger">
					<p role="alert" aria-live="assertive">{error}</p>
					{retryPending ? (
						<>
							{retainedPreview ? (
								<p className="mt-2 truncate text-body-sm text-ink-muted">{retainedPreview}</p>
							) : null}
							<button
								type="button"
								onClick={() => void send(text)}
								className="mt-2 min-h-11 rounded-sm border border-danger-border px-3"
							>
								{RETRY_BUTTON_LABEL}
							</button>
							{/* U4: name why the primary send is dead while the retry is
							    unresolved, so the disabled ↑ reads as intentional. */}
							<p className="mt-2 text-meta text-danger">{RETRY_DISABLED_HINT}</p>
						</>
					) : null}
				</div>
			) : null}

			{images.length > 0 ? (
				<div className="flex flex-wrap gap-1.5">
					{images.map((img) => (
						<button
							key={img.preview}
							type="button"
							onClick={() => removeImage(img.preview)}
							aria-label="remove attachment"
							className="relative size-14 overflow-hidden rounded-sm border border-control"
						>
							<img
								src={img.preview}
								alt=""
								className="size-full object-cover"
							/>
							<span className="absolute inset-0 flex items-center justify-center bg-scrim text-meta text-on-accent opacity-0 active:opacity-100">
								remove
							</span>
						</button>
					))}
				</div>
			) : null}

			{/* The dictation status row: recording shows a live dot + elapsed time and
			    a cancel; transcribing shows the wait and its own cancel (U5); the
			    outcome lines (added / empty / discarded) reuse the row, so one
			    `min-h-11` height covers every state and stopping a recording does not
			    move the line (design round 1, D3). `role="status"` (polite) so a
			    screen reader hears the state without stealing focus; colour is never
			    the only carrier (dot + word + timer). */}
			{dictation !== "idle" || dictationNotice ? (
				<div className="flex min-h-11 items-center gap-2 px-0.5">
					{dictation === "recording" ? (
						<>
							<p
								role="status"
								aria-live="polite"
								className="flex items-center gap-1.5 text-body-sm text-ink-muted"
							>
								<span aria-hidden className="lo-dictation-dot size-2 rounded-full bg-danger" />
								Recording {formatDuration(recordSeconds)}
							</p>
							<button
								type="button"
								onClick={cancelDictation}
								aria-label="cancel recording"
								className="flex min-h-11 items-center rounded-sm px-2 text-body-sm text-ink-muted active:text-ink"
							>
								cancel
							</button>
						</>
					) : dictation === "transcribing" ? (
						<>
							<p role="status" aria-live="polite" className="text-body-sm text-ink-muted">
								Transcribing…
							</p>
							{/* U5: the wait has an exit. It discards what was recorded (the same
							    deal as cancelling a recording) and says nothing extra — a
							    deliberate cancel needs no notice. */}
							<button
								type="button"
								onClick={cancelDictation}
								aria-label="cancel transcription"
								className="flex min-h-11 items-center rounded-sm px-2 text-body-sm text-ink-muted active:text-ink"
							>
								cancel
							</button>
						</>
					) : (
						<p role="status" aria-live="polite" className="text-body-sm text-ink-muted">
							{dictationNotice}
						</p>
					)}
				</div>
			) : null}

			{/* THE WORKING-DIRECTORY CHIP. It sits directly above the input cluster
			    (rather than beside the send controls) so it never competes with them
			    for width on a narrow phone, and so the row it opens is where a reader
			    already looks when asking "where is this session working?". */}
			<div className={COMPOSER_CWD_ROW}>
				<WorkingDirectoryChip
					sessionId={pid}
					cwd={shownCwd}
					onMoved={requestMove}
				/>
			</div>

			<div
				className={COMPOSER_INPUT_ROW}
				onDragOver={(e) => {
					e.preventDefault();
					if (!disabled) setDragOver(true);
				}}
				onDragLeave={() => setDragOver(false)}
				onDrop={(e) => {
					e.preventDefault();
					setDragOver(false);
					if (!disabled) void addFiles(e.dataTransfer.files);
				}}
			>
				<input
					ref={fileInputRef}
					type="file"
					accept="image/*"
					multiple
					className="hidden"
					onChange={(e) => onPickFiles(e.target.files)}
				/>
				<button
					type="button"
					onClick={() => fileInputRef.current?.click()}
					disabled={disabled}
					aria-label="attach image"
					className={cn(
						COMPOSER_ROUND,
						"border border-control text-ink-muted active:bg-elevated disabled:opacity-50",
					)}
				>
					<PaperclipGlyph />
				</button>
				{/* The voice mic: same 44px round control as attach/send/stop, left
				    cluster beside attach. Visible iff the daemon advertised an available
				    path AND this browser can record one; `sending`/`retryPending` do NOT
				    disable it — the draft stays editable while a send is in flight. */}
				{micAvailable ? (
					<button
						type="button"
						onClick={() => {
							if (dictationRef.current === "recording") stopDictation();
							else if (dictationRef.current === "idle") void startDictation();
						}}
						disabled={dictation === "transcribing"}
						aria-label={
							dictation === "recording"
								? "stop and transcribe"
								: dictation === "transcribing"
									? "transcribing"
									: "start voice input"
						}
						className={cn(
							"flex size-11 shrink-0 items-center justify-center rounded-full border active:bg-elevated disabled:opacity-50",
							dictation === "recording"
								? "border-danger-border bg-danger-wash text-danger"
								: "border-control text-ink-muted",
						)}
					>
						{dictation === "transcribing" ? (
							<span aria-hidden className="text-meta">…</span>
						) : (
							<svg width="18" height="18" viewBox="0 0 24 24" fill={dictation === "recording" ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
								<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
								<path d="M19 10v2a7 7 0 0 1-14 0v-2" />
								<line x1="12" y1="19" x2="12" y2="22" />
							</svg>
						)}
					</button>
				) : null}
				<div
					className={cn(
						COMPOSER_FIELD_BOX,
						dragOver ? "border-accent" : "border-control",
					)}
				>
					<textarea
						ref={textareaRef}
						value={text}
						onChange={(e) => onChange(e.target.value)}
						onPaste={(e) => {
							const files = Array.from(e.clipboardData.files);
							if (files.length > 0) {
								e.preventDefault();
								void addFiles(files);
							}
						}}
						placeholder={COMPOSER_PLACEHOLDER}
						disabled={disabled}
						rows={1}
						enterKeyHint="send"
						onKeyDown={(e) => {
							/* Hardware keyboards: Enter sends, Shift+Enter
							   newline. Touch keyboards use the button. */
							if (e.key === "Enter" && !e.shiftKey) {
								e.preventDefault();
								void send(text);
							}
						}}
						className={COMPOSER_FIELD}
					/>
				</div>

				{projection.streaming ? (
					<button
						type="button"
						onClick={abort}
						aria-label="stop"
						className="flex size-11 shrink-0 items-center justify-center rounded-full border border-danger-border text-danger active:bg-danger-wash"
					>
						■
					</button>
				) : null}

				<button
					type="button"
					onClick={() => void send(text)}
					disabled={retryPending || (!text.trim() && images.length === 0) || sending || disabled}
					aria-label={sending ? "Connecting" : projection.streaming ? "steer" : "send"}
					className="flex size-11 shrink-0 items-center justify-center rounded-full bg-accent text-on-accent active:bg-accent-active disabled:bg-sunken disabled:text-ink-disabled"
				>
					<span aria-hidden>{sending ? "…" : "↑"}</span>
				</button>
			</div>

			<div className={COMPOSER_CHIP_ROW}>
				{projection.queued_count > 0 ? (
					<span className="font-mono text-mono-sm text-ink-dim">
						{projection.queued_count} queued
					</span>
				) : null}
				<span className="flex-1" />
				{/* D5 (mobile UX batch): the model chip is ONE line with an ellipsis —
				    under a long model label it used to wrap to two full-width lines
				    and push the row's second chip off-screen. The chip shrinks under
				    its content down to the 44px floor, and the truncating span
				    supplies the ellipsis (the row itself stays on one line — flex
				    rows never wrap). */}
				<button
					type="button"
					onClick={onOpenModels}
					className={cn(COMPOSER_CHIP, "active:text-ink-muted")}
				>
					<span className="truncate">{projection.model_label || "model"}</span>
				</button>
				{projection.effort_ladder.length > 0 ? (
					<button
						type="button"
						onClick={onOpenEffort}
						className={cn(COMPOSER_CHIP, "active:text-ink-muted")}
					>
						{projection.effort || "effort"}
					</button>
				) : null}
			</div>

			<SlashSheet
				open={slashOpen && !disabled}
				onClose={() => setSlashDismissed(true)}
				onPick={onSlashPick}
				onSpace={onSlashSpace}
				query={slashQuery(text) ?? ""}
			/>
			<EffortSheet
				open={effortOpen}
				onClose={onCloseEffort}
				pid={pid}
				projection={projection}
			/>
		</div>
	);
}

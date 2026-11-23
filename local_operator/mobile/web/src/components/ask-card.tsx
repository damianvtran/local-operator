/**
 * Ask card — ONE queued ask, with the controls that answer it.
 *
 * WHAT THIS IS. The phone's answer surface for a queued ask (design
 * `docs/design/ask-nonblocking.md` §5.3). It is the queued twin of
 * `pending-card.tsx`, and the differences between the two are exactly the
 * differences the queue introduced:
 *
 *  * The queued wire carries the ask's WHOLE question list at once (§4), so
 *    this card is a FORM — every question visible, one submit — where the
 *    blocking card was a wizard advanced one question at a time. That is not a
 *    style choice: `AskQueue.respond` is atomic per ask and refuses an
 *    incomplete map, so a wizard would have to hold the answers anyway, and a
 *    form is what makes "you have answered 2 of 3" visible.
 *  * A queued ask has a DEADLINE, and the card says when it is (`expires_at`,
 *    rendered on the client clock per §5) rather than leaving the user to guess
 *    whether anybody is still waiting.
 *  * A queued ask can be DISMISSED. Dismissal asks the model nothing — it buys
 *    no turn — which is why it is offered as a quiet third action rather than a
 *    second button beside Decline, and why its own state line says "no reply
 *    was sent". What it can still OWE is the deadline notice that ask never got
 *    (design `ask-nonblocking.md` §14), delivered on the conversation's next
 *    run: the agent's fact, not this card's action.
 *
 * THE SEND BUTTON IS ENABLED ONLY BY A COMPLETE MAP. Every question id must
 * carry a value; a question the user deliberately skipped is sent as an EMPTY
 * LIST, which is the queue's own spelling for "no answer" (§2.4). Letting a
 * partial map reach the wire would be refused anyway (`_partial_answer_error`),
 * so the card states the rule instead of collecting a refusal for it.
 *
 * REFUSALS KEEP THE DAEMON'S OWN SENTENCE. The queue's copy is the shared one
 * every surface shows — "already answered by desktop", "this ask expired 7 days
 * ago — ask again if it is still needed" — and the whole point of `render.
 * refusal_copy` living in core is that the phone must not author a fourth
 * wording of one rule. The card renders `error.message` verbatim, and only
 * falls back to a plain line when the daemon said nothing usable.
 */
import { contentDir } from "../lib/content-direction";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useKeyboardUp } from "../lib/use-keyboard-up";
import type { ReactNode } from "react";
import { HttpError, sendCommand } from "../api";
import { cn } from "../lib/cn";
import { ASK_QUESTION_FRACTION, columnCap } from "../lib/column";
import {
	type AskTone,
	answeredPairs,
	askStateLine,
	dismissedReasonKey,
	isAnswerable,
	isDeadConversation,
	isLapsed,
	recommendedRows,
	unansweredQuestions,
} from "../lib/asks";
import { clearAskDraft, messageRuntime, useAskDraft } from "../store";
import type { AskDraft } from "../store";
import type { AskQuestion, PendingAsk } from "../types";

/** The two dismissal reasons, in the order the card offers them (design §17:
    "No longer needed" first, then "Answered in chat"), with the catalogue keys
    each row paints.

    A LOOKUP RATHER THAN A TEMPLATE. The keys used to be built by interpolation
    (``relay.ask.dismiss.${reason}`` / ``..._note``), which no static check can
    verify — a renamed key would render as the key text on screen — and which
    the relay copy gate counts as unextracted literals. Spelling each key here
    keeps them greppable and type-checked against the generated key union. */
const DISMISS_COPY = {
	moot: { key: "relay.ask.dismiss.moot", note: "relay.ask.dismiss.moot_note" },
	answered_in_chat: {
		key: "relay.ask.dismiss.answered_in_chat",
		note: "relay.ask.dismiss.answered_in_chat_note",
	},
} as const;

const DISMISS_REASONS = ["moot", "answered_in_chat"] as const;

/** The one line a queued ask's own state is stated in, for the open card and
    for the sheet's collapsed row of the same ask.

    A DISMISSED ROW STATES ITS REASON (SA-D8, design §17). The reason is a
    catalogue sentence rather than a literal here, so the card says it in the
    operator's language: `askStateLine`'s `dismissed` arm is the fallback for a
    row that recorded none (and for an older daemon that never sent the key). */
export function askStateOf(row: PendingAsk, nowMs: number): { text: string; tone: AskTone } {
	const state = askStateLine(row, nowMs, messageRuntime.t);
	const reason = dismissedReasonKey(row);
	return reason ? { ...state, text: messageRuntime.t(reason) } : state;
}

/** The ink a state line is painted in, one place for the open card and the
    sheet's collapsed rows. INK FOLLOWS THE PROMISE, not the mood (design round
    1, D2): an open ask's line was `text-accent` and a settled one
    `text-success`, two greens that are ΔE 17 apart here and IDENTICAL in three of
    the 28 shipped themes — so the one distinction this card exists to draw
    (still waiting vs already settled) could vanish. `accent` is no longer spent
    on the state line at all: "the agent is continuing" is information, not an
    achievement, and success green is reserved for the states that are receipts. */
export function askToneClass(tone: AskTone): string {
	return tone === "attention"
		? "text-warning"
		: tone === "settled"
			? "text-success"
			: tone === "gone"
				? "text-ink-dim"
				: "text-ink-muted";
}

/** The daemon's sentence, or the plainest honest thing when it gave none.
 *
 *  `api.request` falls back to the bare status when an error body is not JSON
 *  ("422" under a button explains nothing), and a fetch-level failure arrives as
 *  the browser's own TypeError. Neither is the queue's refusal copy, so neither
 *  may be rendered as though it were. */
function refusalText(error: unknown): string {
	if (error instanceof TypeError) return messageRuntime.t("relay.asks.error.reach_failed");
	const message = error instanceof Error ? error.message : String(error);
	if (message === "" || /^\d{3}$/.test(message)) return messageRuntime.t("relay.asks.error.no_reason");
	return message;
}

/** The stale-ask context line (SA-D7 / R5), or `null` when there is none.
 *
 *  "3 messages from you since this was asked (last 19:36)": operator messages
 *  admitted since the ask was raised, and when the latest one was. It is the one
 *  line on this card that says a question may ALREADY have been answered in
 *  chat — the reason the card can be closed with `answered_in_chat`.
 *
 *  ABSENCE IS THE CONTRACT, exactly as `asks/store.pending_row` writes it: an
 *  older runtime sends no field, and that draws NOTHING rather than a zero the
 *  reader would have to disbelieve. A zero is a fact (the ask is newer than every
 *  message) and draws nothing either — there is no staleness to report. An OPEN
 *  ask only: a settled one has no wait left to be stale.
 *
 *  The clock is formatted HERE rather than through ICU: it is a wall-clock time,
 *  not prose, and every locale's catalogue carries the finished string inside the
 *  sentence it belongs to. Latin digits in 24-hour form, matching the terminal's
 *  own rendering of the same pair (`tui.widgets.ask_queue.context_text`). */
function contextLine(row: PendingAsk): string | null {
	if (String(row.status || "open") !== "open") return null;
	const count = row.user_messages_since;
	if (typeof count !== "number" || count <= 0) return null;
	const last = row.last_user_message_at;
	if (typeof last !== "number" || last <= 0) return null;
	const at = new Date(last);
	const time = `${String(at.getHours()).padStart(2, "0")}:${String(at.getMinutes()).padStart(2, "0")}`;
	return messageRuntime.t("relay.ask.context.messages", { count, time });
}

/** The sentence a row shows when the aggregate ALREADY reports its conversation
 *  as DEAD (`isDeadConversation`: no durable transcript AND nothing live to
 *  deliver to) — the state the daemon refuses every op on with
 *  `ask_session_gone` (design round 2, D6).
 *
 *  IT MIRRORS `daemon._ASK_GONE_RESPONSE` rather than waiting for the refusal,
 *  and that duplication is the point of a pre-emptive state: fetching the
 *  sentence would mean buying the guaranteed 409 the state exists to avoid. It
 *  is the ANSWER's wording, remedy included (design round 3, D7's NIT: one
 *  voice), because the card offers no control at all — so the sentence that fits
 *  is the one the primary control would have produced, and `start a new
 *  conversation and ask again` is the user's actual next move. The ops that
 *  carry their own sentence (a dismissal, a handed-over decision) are only told
 *  apart when the refusal comes FROM THE WIRE, which this branch never does.
 *  The sentence itself is `relay.asks.card.gone`. */

/** Not a wire cap: the daemon stores whatever string it is given. 32768 is the
    bound the desktop client's answer schema uses, and the phone matches it so the
    same typed answer is accepted on both surfaces and a runaway paste stops at the
    field rather than after the tap. */
const ANSWER_MAX_CHARS = 32768;

/** The card's own record of its `Other` row: whether it is the selected one, and
 *  what its field holds (UNTRIMMED).
 *
 *  KEPT BESIDE THE DRAFT CELL, NOT DERIVED FROM IT, for the two reasons the
 *  desktop panel learned the hard way. The cell is trimmed on the way in, so a
 *  field fed from it would eat a space typed between two words on the keystroke.
 *  And a text equal to an option's label is indistinguishable from that option in
 *  the cell, so a field whose visibility followed the cell would close mid-
 *  sentence the moment `No` was typed beside an option `No`. Single-select keeps
 *  `text` while `open` is false, so choosing an option and coming back does not
 *  make the user type it again. */
interface OtherState {
	open: boolean;
	text: string;
}

/** The marker the cell holds for an `Other` that is open but EMPTY: the empty
 *  string, exactly (the desktop's own spelling). It is what lets the open state
 *  survive an unmount through the store-backed draft, and what keeps the
 *  question incomplete — see `filled`. It is never sent. */
const isOtherPending = (value: string): boolean => value === "";

/** The `Other` state on mount: open, with the text, when the cell already holds a
 *  value the options did not offer, or the blank marker. Seeded open but never
 *  focused — a mount takes no focus, only the user's own press does.
 *
 *  More than one out-of-list value cannot come from this card (it writes at most
 *  one); they are joined rather than hidden so a draft written by something else
 *  loses nothing the user could see. */
function otherSeed(cell: readonly string[], labels: readonly string[]): OtherState {
	const values = cell.filter((value) => value.trim() !== "" && !labels.includes(value));
	return { open: values.length > 0 || cell.some(isOtherPending), text: values.join(", ") };
}

/** The option labels ticked in the cell. A typed `Other` equal to an option's
 *  label sits in the cell as that label, so the LAST such entry is the Other's,
 *  not a tick — otherwise typing `No, thanks` through `No` would tick an option
 *  the user never chose. */
function otherTicks(
	cell: readonly string[],
	labels: readonly string[],
	other: OtherState,
): string[] {
	const labelled = cell.filter((value) => labels.includes(value));
	const entry = other.open ? other.text.trim() : "";
	if (entry === "" || !labels.includes(entry)) return labelled;
	const at = labelled.lastIndexOf(entry);
	return labelled.filter((_, index) => index !== at);
}

/** The cell for a set of ticks and an `Other` state. The Other entry goes LAST,
 *  after the ticked labels, which is the order the answer travels in. */
function cellWith(ticks: readonly string[], multi: boolean, other: OtherState): string[] {
	if (!other.open) return multi ? [...ticks] : [];
	const entry = other.text.trim();
	return multi ? [...ticks, entry] : [entry];
}

/** What travels for one cell: every value trimmed, blanks dropped, the first of
 *  any duplicate kept, tick order preserved with the `Other` text last. A blank
 *  string must never reach `answers` — the runtime would store it as a deliberate
 *  empty answer — and the open-but-empty marker is exactly such a string. */
function answerValues(cell: readonly string[]): string[] {
	return Array.from(new Set(cell.map((value) => value.trim()).filter((value) => value !== "")));
}

/** One question's control, plus whether it currently holds a complete answer. */
/**
 * A question's text, capped at a share of the column and scrolling inside
 * itself (`ASK_QUESTION_FRACTION`), so a paragraph- or brief-length question
 * can never push this question's options and free-text field off screen.
 * `whitespace-pre-line` keeps an agent's line breaks (a proposal summary is
 * three lines); `break-words` stops a long token from widening the card.
 * `tabIndex` makes the scroller keyboard-reachable once it overflows.
 */
function QuestionText({ text }: { text: string }) {
	return (
		<span
			dir={contentDir(text)}
			data-testid="ask-question-text"
			tabIndex={0}
			style={columnCap(ASK_QUESTION_FRACTION)}
			className="lo-scroll block min-w-0 text-start overflow-y-auto overflow-x-hidden break-words whitespace-pre-line"
		>
			{text}
		</span>
	);
}

/** How long the Copy button reads "Copied" after a successful copy. Feedback
    timing only (long enough to be read at a glance, short enough that a second
    tap on a re-rendered card is not met by a stale "Copied"); nothing waits on it. */
const COMMAND_COPIED_MS = 2000;

/**
 * The question's suggested terminal command (`AskQuestion.command`), as a
 * monospace block with a Copy button.
 *
 * COPY ONLY, BY DESIGN. The phone has no terminal: the command is meant for the
 * agent's machine or the user's computer (`command_target`), never the phone,
 * so this card offers nothing about running it — the desktop app owns the
 * confirmed Insert/Run flow. Showing it separately from the prose is still the
 * point: inline code inside a capped, scrolling question is hard to select on a
 * touch screen, and one tap here puts the exact bytes on the clipboard.
 *
 * `dir="ltr"` regardless of the UI locale: a shell command is LTR text, and an
 * RTL paragraph would visually reorder its tokens (core already refuses bidi
 * control characters in it). `break-all` + `whitespace-pre-wrap` show every
 * character, wrapped rather than truncated, so what is copied is what was read.
 *
 * A clipboard write can fail (no secure context, permission denied); the block
 * then selects the command so the platform's own copy gesture works, and says so
 * in a `role="status"` note, so a screen-reader user hears the failure too (the
 * button's own live region only ever announces "Copied") (code review m1).
 *
 * It must read as CODE, not as a field (design round 1, D5/D6): a free-text
 * question puts an editable `bg-canvas` input right under it, so the block wears
 * the transcript's code surface (`bg-sunken`, as `markdown.tsx`'s code blocks)
 * and a leading `$` like the TUI's `$ <command>` line. The `$` is `aria-hidden`,
 * `select-none` and OUTSIDE the `<code>` node: Copy writes the `command` prop
 * and the failure fallback selects only the `<code>` node, so neither path can
 * put the prompt mark on the clipboard.
 */
function CommandBlock({ command }: { command: string }) {
	const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");
	const code = useRef<HTMLElement>(null);
	useEffect(() => {
		if (copyState !== "copied") return;
		const timer = window.setTimeout(() => setCopyState("idle"), COMMAND_COPIED_MS);
		return () => window.clearTimeout(timer);
	}, [copyState]);
	const copy = async () => {
		try {
			await navigator.clipboard.writeText(command);
			setCopyState("copied");
		} catch {
			const node = code.current;
			const selection = window.getSelection();
			if (node && selection) {
				const range = document.createRange();
				range.selectNodeContents(node);
				selection.removeAllRanges();
				selection.addRange(range);
			}
			setCopyState("failed");
		}
	};
	return (
		<div
			role="group"
			aria-label={messageRuntime.t("relay.asks.card.command.label")}
			data-testid="ask-command"
			className="flex flex-col gap-1"
		>
			<div className="flex min-w-0 items-start gap-2 rounded-sm bg-sunken py-1 ps-3 pe-1">
				{/* `dir="ltr"` on the row holding `$` and the command, so the mark
				    stays at the command's start in an RTL locale too. */}
				<div dir="ltr" className="flex min-w-0 flex-1 gap-1.5 self-center py-1.5 font-mono text-mono-sm">
					<span aria-hidden="true" data-testid="ask-command-prompt" className="shrink-0 select-none text-ink-muted">
						$
					</span>
					<code
						ref={code}
						dir="ltr"
						data-testid="ask-command-text"
						className="min-w-0 flex-1 select-text break-all whitespace-pre-wrap font-mono text-mono-sm text-ink"
					>
						{command}
					</code>
				</div>
				<button
					type="button"
					data-testid="ask-command-copy"
					onClick={() => void copy()}
					aria-live="polite"
					className="flex min-h-11 shrink-0 items-center rounded-sm px-3 text-body-sm text-ink-muted active:bg-elevated"
				>
					{/* Both labels share one grid cell, the inactive one `invisible` and
					    `aria-hidden`, so the button is always as wide as the LONGER label
					    in this locale (Copy→Copied is 4→6 chars in en, 8→11 in vi). A
					    width that changed on the tap rewrapped a long command beside it,
					    and again 2 s later (design round 1, D4) — the same "the label is
					    the width, so it stays put" rule as Dismiss below. The visible
					    label is the live text, so "Copied" is still announced. */}
					<span className="grid">
						<span data-testid="ask-command-copy-label" className="col-start-1 row-start-1">
							{copyState === "copied"
								? messageRuntime.t("relay.asks.card.command.copied")
								: messageRuntime.t("relay.asks.card.command.copy")}
						</span>
						<span aria-hidden="true" className="invisible col-start-1 row-start-1">
							{copyState === "copied"
								? messageRuntime.t("relay.asks.card.command.copy")
								: messageRuntime.t("relay.asks.card.command.copied")}
						</span>
					</span>
				</button>
			</div>
			{copyState === "failed" ? (
				<p role="status" className="text-meta text-ink-dim">
					{messageRuntime.t("relay.asks.card.command.copy_failed")}
				</p>
			) : null}
		</div>
	);
}

/**
 * The look of one option row (and the `Other` row), the web twin of the native
 * app's `choiceRowClasses` so the two panels draw the same rows.
 *
 * - REST: a tinted fill (the accent wash), a 2 px accent start edge, and a
 *   trailing EMPTY ring (one choice) or square (several) in `inkDim`.
 * - SELECTED: the same tint, a 4 px edge, a semibold label, and the indicator
 *   FILLS to an accent disc/square holding a check (`Indicator`) — selection is
 *   shape and weight as well as colour, so it survives greyscale. The edge takes
 *   the 2 px it gains out of the start padding, so the label does not shift.
 *   (The native app also deepens the fill on selection; the relay's palette has
 *   no stronger tint than the wash, and a runtime blend would sit outside what
 *   `scripts/contrast-contract.mjs` can verify.)
 * - PRESSED: a brief dim under the finger.
 * - DISABLED (and not selected): sunken ground, hairline edge and ring, disabled
 *   ink. A selected row keeps its look while an answer is in flight, so the
 *   thing being sent stays readable.
 *
 * Contrast: ink, inkMuted, inkDim and accent on accentWash and onAccent on accent
 * are pinned at 4.5:1 in every theme by the contrast contract. The ring is
 * `inkDim` rather than the native app's control edge because `control` on
 * accentWash measures 2.96 (ayuLight) and 2.89 (iceberg), under the 3:1 non-text
 * floor, where `inkDim` is at least 4.55.
 */
function optionRowLook({
	selected,
	disabled,
	multi,
}: {
	selected: boolean;
	disabled: boolean;
	multi: boolean;
}) {
	const dimmed = disabled && !selected;
	return {
		row: cn(
			"flex min-h-11 items-center gap-2 rounded-sm py-2 pe-3 text-start",
			selected ? "border-s-4 ps-3" : "border-s-2 ps-3.5",
			dimmed ? "border-s-hairline bg-sunken" : "border-s-accent",
			!dimmed && "bg-accent-wash active:brightness-95",
		),
		inkMain: cn(selected && "font-semibold", dimmed ? "text-ink-disabled" : "text-ink"),
		note: dimmed ? "text-ink-disabled" : "text-ink-dim",
		desc: dimmed ? "text-ink-disabled" : "text-ink-muted",
		indicator: cn(
			"ms-auto flex h-5 w-5 shrink-0 items-center justify-center border-2 text-meta leading-none",
			multi ? "rounded-xs" : "rounded-full",
			selected
				? "border-accent bg-accent text-on-accent"
				: dimmed
					? "border-hairline"
					: "border-ink-dim",
		),
	};
}

/** The trailing ring (or square) of a row: empty at rest, a disc holding a check
    when chosen. Decorative: `aria-checked` carries the state for assistive tech. */
function Indicator({ className, on }: { className: string; on: boolean }) {
	return (
		<span aria-hidden className={className} data-ask-indicator={on ? "on" : "off"}>
			{on ? "✓" : null}
		</span>
	);
}

function QuestionField({
	question,
	value,
	onChange,
	disabled,
}: {
	question: AskQuestion;
	/** The draft cell: chosen labels, or the typed text for a text/secret
	    question. An empty array means "not answered yet" — which is why a
	    deliberate skip is a separate state (`skipped`), not an empty value. */
	value: string[];
	onChange: (next: string[]) => void;
	disabled: boolean;
}) {
	const options = Array.isArray(question.options) ? question.options : [];
	const labels = options.map((option) => option.label);
	const multi = question.multi === true;
	/* The free-text door is on every option question except a secret one: a
	   secret's value is a credential the agent asked for by name, and an arbitrary
	   typed string beside its options would not be one. A question with no options
	   already IS a text field. */
	const offersOther = !question.secret && options.length > 0;
	const [other, setOther] = useState<OtherState>(() => otherSeed(value, labels));
	const ticks = otherTicks(value, labels, other);
	const otherField = useRef<HTMLInputElement>(null);
	/* FOCUS IS TAKEN ONLY INSIDE THE USER'S PRESS ON THE ROW, never on mount: a
	   seeded-open field (a draft restored after a collapse) must not pull the
	   keyboard up under a thumb that did nothing. The flag is raised by the press
	   and spent here, once the field it points at is mounted. The dependency is
	   the trigger (the field mounting), not a value read in the body. */
	const focusOther = useRef(false);
	useLayoutEffect(() => {
		if (!focusOther.current) return;
		focusOther.current = false;
		otherField.current?.focus();
	}, [other.open]);
	const writeOther = (next: OtherState) => {
		setOther(next);
		onChange(cellWith(ticks, multi, next));
	};
	const chosen = new Set(ticks);
	const otherLook = optionRowLook({ selected: other.open, disabled, multi });
	if (options.length === 0) {
		/* A free-text or secret question: the answer IS the typed string. */
		return (
			<div className="flex flex-col gap-1">
				{question.secret ? (
					<p className="text-meta text-ink-dim">
						{question.persist
							? messageRuntime.t("relay.asks.card.secret_notice_persisted")
							: messageRuntime.t("relay.asks.card.secret_notice")}
					</p>
				) : null}
				<input
					value={value[0] ?? ""}
					onChange={(event) => onChange([event.target.value])}
					disabled={disabled}
					type={question.secret ? "password" : "text"}
					autoComplete={question.secret ? "off" : undefined}
					autoCapitalize={question.secret ? "none" : undefined}
					autoCorrect={question.secret ? "off" : undefined}
					spellCheck={question.secret ? false : undefined}
					placeholder={
						question.secret
							? messageRuntime.t("relay.asks.card.placeholder_secret")
							: messageRuntime.t("relay.asks.card.placeholder_answer")
					}
					className="min-h-11 w-full rounded-sm bg-canvas px-3 text-body text-ink outline-none placeholder:text-ink-dim disabled:opacity-50"
				/>
			</div>
		);
	}

	return (
		<div
			role={multi ? "group" : "radiogroup"}
			aria-label={question.question}
			className="flex flex-col gap-2"
		>
			{/* Describes the option list, so it sits with it, above the rows. */}
			{multi ? (
				<span className="text-meta text-ink-dim">
					{messageRuntime.t("relay.asks.card.choose_any")}
				</span>
			) : null}
			{options.map((option, index) => {
				const on = chosen.has(option.label);
				/* THE RECOMMENDATION IS AN INDEX, not a flag (see `AskQuestion`): the
				   runtime hoists the recommended option to 0 and STATES the position,
				   because the harness option model has no boolean to read. Any index
				   is honoured — the wire, the type and the TUI all treat it as a
				   position, and pinning it to 0 would badge nothing the day the
				   runtime stops hoisting (agent review round 2, N1). A multi-select
				   may recommend several rows, so every one in the set is badged. */
				const recommended = recommendedRows(question).includes(index);
				const rowLook = optionRowLook({ selected: on, disabled, multi });
				return (
					<button
						key={option.label}
						type="button"
						/* A RADIO OR A CHECKBOX, NOT A TOGGLE BUTTON (ask-panel parity R3): the
						   rows are one choice among several, which is what a screen reader
						   should hear ("radio, 1 of 3, checked"), and `aria-pressed` only ever
						   said "toggle". They stay real <button>s, so Enter and Space work
						   without a key handler of ours. */
						role={multi ? "checkbox" : "radio"}
						aria-checked={on}
						disabled={disabled}
						onClick={() => {
							if (!multi) {
								/* An option and `Other` are one choice in single-select, so
								   picking an option CLOSES `Other` — its typed text stays in
								   card state (the user need not retype it on coming back) but
								   is no longer the answer. */
								setOther({ ...other, open: false });
								onChange([option.label]);
								return;
							}
							/* Multi-select toggles; the answer map holds a list either
							   way, so the wire shape does not depend on `multi`. `Other`
							   is additive here and is rebuilt LAST, so a tick never
							   reorders the typed text ahead of the labels. */
							const next = new Set(chosen);
							if (next.has(option.label)) next.delete(option.label);
							else next.add(option.label);
							onChange(cellWith(Array.from(next), true, other));
						}}
						className={rowLook.row}
					>
						<span className="flex min-w-0 flex-1 flex-col justify-center">
							<span className={cn("text-body-sm font-medium", rowLook.inkMain)}>
								{option.label}
								{recommended ? (
									/* The model's recommendation, as a word. §5's contract keeps
									   position as the only channel for older clients, but this
									   client is told outright, and a badge that says so is
									   cheaper to read than "the first one means it". */
									<span className={rowLook.note}> {messageRuntime.t("relay.asks.card.recommended")}</span>
								) : null}
							</span>
							{option.description ? (
								<span className={cn("text-body-sm", rowLook.desc)}>{option.description}</span>
							) : null}
						</span>
						<Indicator className={rowLook.indicator} on={on} />
					</button>
				);
			})}
			{offersOther ? (
				<>
					{/* THE `Other` ROW: the free-text door on every non-secret option
					    question (the desktop and the terminal already have it), so how
					    to give an answer the list did not offer is on the card rather
					    than something the user has to know. The hint is the same
					    "your answer" the field's placeholder says; it goes while the
					    field is open, where it would print the same words twice.
					    A real button with the option rows' radio/checkbox semantics. */}
					<button
						type="button"
						data-ask-option-other=""
						disabled={disabled}
						role={multi ? "checkbox" : "radio"}
						aria-checked={other.open}
						onClick={() => {
							/* A single-select `Other` is the selected one of a radio-like
							   group, which does not deselect: pressing it again re-focuses
							   the field. A multi-select one is a checkbox and unticks. */
							if (multi && other.open) {
								writeOther({ ...other, open: false });
								return;
							}
							focusOther.current = true;
							if (other.open) {
								focusOther.current = false;
								otherField.current?.focus();
							} else {
								writeOther({ ...other, open: true });
							}
						}}
						className={otherLook.row}
					>
						<span className={cn("text-body-sm font-medium", otherLook.inkMain)}>
							{messageRuntime.t("relay.asks.card.other")}
						</span>
						{other.open ? null : (
							<span className={cn("text-body-sm", otherLook.note)}>
								{/* TYPED TEXT SURVIVES CLOSING THE ROW (it stays in card state), and
								    the row says so: otherwise a closed row reads as an empty door
								    while the text is still there, unsent. */}
								{other.text.trim() !== ""
									? messageRuntime.t("relay.asks.card.other_kept")
									: messageRuntime.t("relay.asks.card.placeholder_answer")}
							</span>
						)}
						<Indicator className={otherLook.indicator} on={other.open} />
					</button>
					{other.open ? (
						<input
							ref={otherField}
							type="text"
							value={other.text}
							maxLength={ANSWER_MAX_CHARS}
							onChange={(event) => writeOther({ open: true, text: event.target.value })}
							onKeyDown={(event) => {
								/* A single-line field: Enter neither submits nor inserts anything,
								   it only dismisses the keyboard. An IME confirming a composition
								   also sends Enter, which must reach the IME untouched — Safari
								   delivers that keydown with `isComposing` already false and the
								   legacy `keyCode` 229, so both signals are checked. */
								if (
									event.key !== "Enter" ||
									event.nativeEvent.isComposing ||
									event.keyCode === 229
								)
									return;
								event.preventDefault();
								event.currentTarget.blur();
							}}
							disabled={disabled}
							aria-label={question.question}
							placeholder={messageRuntime.t("relay.asks.card.placeholder_answer")}
							className="min-h-11 w-full rounded-sm bg-canvas px-3 text-body text-ink outline-none placeholder:text-ink-dim disabled:opacity-50"
						/>
					) : null}
				</>
			) : null}
		</div>
	);
}

export function AskCard({
	row,
	sessionId,
	nowMs,
	onSettled,
	onDismissed,
	footerSlot,
	lead,
	hideState = false,
	focusQuestion = false,
	onFocused,
}: {
	row: PendingAsk;
	sessionId: string;
	/** The client's clock, for the deadline line (§5: the countdown is rendered
	    from `expires_at` locally, never from a server-computed at-push string). */
	nowMs: number;
	/** Called after this surface settles, declines or dismisses the ask, so an
	    aggregate list can re-read the index rather than guess the new state. It
	    names the OP: a sheet holding a dismissal receipt must retire it when the
	    operator's next act was an answer, not keep describing the old one
	    (agent review round 2, M1 = UX round 2, U9). */
	onSettled?: (kind: "respond" | "decline" | "dismiss") => void;
	/** Called after a DISMISSAL is accepted, with the ask it was about and the
	    reason that was sent. The sheet needs both: the reason for the receipt's
	    words, and the id so a later dismissal replaces this one instead of being
	    confused with it (M1). */
	onDismissed?: (askId: string, reason: "moot" | "answered_in_chat" | undefined) => void;
	/** WHERE THE ANSWERING CONTROLS GO. The controls that answer the card (send,
	    decline, dismiss, the progress and refusal lines) are drawn in the sheet's
	    PINNED bottom bar, not at the end of the scrolling card: content may
	    scroll, the control that answers it may not (ask-panel parity, R2).

	    `undefined` draws them at the card's own foot (a card used on its own);
	    an element portals them into it; `null` is a bar that has not mounted yet,
	    which draws nothing for that one commit rather than a duplicate. They stay
	    this component's own children either way, so the draft, the in-flight
	    state and the refusal never have to be lifted into the sheet. */
	footerSlot?: HTMLElement | null;
	/** Drawn right under the state line: the conversation's name when the sheet
	    holds several conversations' asks, and the not-running wait. Data and
	    sheet-level copy the card has no reason to know how to compute. */
	lead?: ReactNode;
	/** The state line is stated once by the collapsed row an expanded receipt
	    sits under, so the card does not say it a second time. */
	hideState?: boolean;
	/** Move focus to the first question's text once, on mount (after a send, the
	    reader's place moves to the next ask), then say so through `onFocused`. */
	focusQuestion?: boolean;
	onFocused?: () => void;
}) {
	/* The draft map: question id → the labels (or the typed string) the user has
	   chosen but not yet sent. IT LIVES IN THE STORE, not in this component
	   (QA round 1 Q-1 = UX round 1 U1): a card-local `useState` died with the
	   sheet's unmount, so collapsing the sheet discarded the draft that §5.0-R7
	   requires be kept. The chat buffer has lived in the store all along; this is
	   its twin, keyed by ask id so one collapse cannot mix two asks' answers. */
	/* SECRET QUESTIONS ARE NEVER WRITTEN TO THE STORED DRAFT: the value is held in
	   this card's memory while it is open and nowhere else (`useAskDraft`). Joined
	   to a string so the list's identity does not change per render. */
	const secretKey = (Array.isArray(row.questions) ? row.questions : [])
		.filter((question) => Boolean(question.secret))
		.map((question) => String(question.id))
		.join("\u0000");
	const secretIds = useMemo(() => (secretKey === "" ? [] : secretKey.split("\u0000")), [secretKey]);
	const [draft, setDraft] = useAskDraft(row.ask_id, secretIds);
	const [busy, setBusy] = useState<"" | "respond" | "decline" | "dismiss">("");
	const [error, setError] = useState("");
	/** Which control produced the refusal, so the sentence can sit beside IT
	    (design round 1, D5): a dismiss refusal rendered above the send/decline row
	    landed 60 px from the button that produced it, with the other pair between
	    them. */
	const [errorOp, setErrorOp] = useState<"" | "respond" | "decline" | "dismiss">("");
	/** The conversation is GONE (`ask_session_gone`): terminal, so no control may
	    stay live (design round 1, D3). */
	const [gone, setGone] = useState(false);
	/** The two reason buttons are showing (`close…` pressed on an open ask).
	    A separate state from `busy`: revealing the choice sends NOTHING, so a
	    card that showed the in-flight mark here would claim a dismissal the
	    operator has not made. */
	const [closing, setClosing] = useState(false);
	/** Which reason was SENT, so the button that carries it keeps its label while
	    the dismissal is in flight (UX round 1, U3). Both buttons used to collapse
	    to `…`, which erased the only on-screen statement of what the operator
	    chose. */
	const [sentReason, setSentReason] = useState<"moot" | "answered_in_chat" | undefined>(
		undefined,
	);
	/** A dismiss refusal has been read: the reasons stay visible, inert (U8). */
	const [dismissRefused, setDismissRefused] = useState(false);

	/* THE COLD CASE, and only the known one (design round 1, D1/D4 = UX U1/U3/U4):
	   `runtime_live === false` means the relay will have to bring the conversation
	   up before the op can land — FOR ANSWER AND DECLINE, which is a ~30 s envelope
	   against a warm answer's 1–3 s. A dismissal is the exception that takes the
	   no-boot door on every surface (design §14, fork A), so `cold` buys it nothing
	   and nothing here mentions it. `undefined` (an older daemon, or a row from a
	   projection) promises nothing and keeps the bare `…`.

	   BOTH FACTS ARE READ OFF THE ROW, not passed as props: they are the
	   aggregate's own fields (`PendingAsk.runtime_live` / `.durable`), and a
	   second copy of them on the component was exactly the drift D7 was made of. */
	const cold = row.runtime_live === false;
	const state = askStateOf(row, nowMs);
	const stateLine = state.text;
	const status = String(row.status || "open");
	const answerable = isAnswerable(status);
	/* A LAPSED ASK IS SHOWN, NOT OFFERED (Ben's report, 2026-10-15). Its deadline
	   has passed and the agent has gone ahead without it, so options and a send under
	   it read as though the agent were still waiting on them. The store would still
	   take a late answer on `timed_out` (`isAnswerable` keeps saying so); the phone
	   deliberately stops inviting one. `lapsed` gates the controls only — the dead
	   and dismiss rules below are unchanged. */
	const lapsed = isLapsed(row);
	const offered = answerable && !lapsed;

	/* A ROW THE AGGREGATE ALREADY KNOWS IS DEAD IS TERMINAL BEFORE THE TAP
	   (design round 2, D6 + UX U7), and DEAD IS A CONJUNCTION (design round 3,
	   D7). `isDeadConversation` is the route's own two facts: no durable
	   transcript to read AND nothing live to deliver to. Gating on `durable`
	   alone made the client MORE PESSIMISTIC THAN THE SERVER — a live row with no
	   transcript was shown as a dead end while `ask_respond` on it returned
	   `200 {"ok": true, "detail": "answered"}` — the same class of lie in the
	   other direction, and the reason the predicate lives in `lib/asks.ts` and is
	   shared by this card, the sheet's strip and the sheet's title count. Gated
	   further on `answerable`: a SETTLED row in a dead conversation is a receipt,
	   not a dead end, and must keep its receipt. */
	const goneFromWire = isDeadConversation(row) && answerable;
	const terminal = gone || goneFromWire;
	/* A LAPSED ASK KEEPS ITS ONE-TAP DISMISS (D11, §14), and an OPEN one now has a
	   door of its own (R5 / SA-D4): `AskQueue.dismiss` accepts `open` on every live
	   session (the live refusal was retired — see the design's amended §15 D6), so
	   the refusal this comment used to record ("only a timed-out ask can be
	   dismissed; it is still open.") no longer exists to collect. The two doors
	   differ in what they ASK the operator, not in whether they are allowed: a
	   lapsed ask was already walked past by the agent, so it goes at once; an open
	   one is still owed an answer, so the card asks WHY first and sends nothing
	   until a reason is chosen. */
	const dismissible = status === "timed_out";
	const reason_needed = status === "open";
	const questions = useMemo(() => (Array.isArray(row.questions) ? row.questions : []), [row]);
	const open_questions = useMemo(
		() => (offered ? unansweredQuestions(row) : []),
		[offered, row],
	);

	const skipped = useMemo(() => new Set(draft.skipped), [draft.skipped]);
	/* FUNCTIONAL, so two picks in one tick compose instead of overwriting (see
	   `useAskDraft`); the options of two different questions are the normal case. */
	const patchDraft = (patch: (current: AskDraft) => Partial<AskDraft>) =>
		setDraft((current) => {
			/* The patch runs ONCE per update (agent review round 2, N3): calling it
			   in both fields was harmless only because every caller passes a pure
			   function, which is not a property this signature can promise. */
			const next = patch(current);
			return {
				answers: next.answers ?? current.answers,
				skipped: next.skipped ?? current.skipped,
			};
		});

	/** Whether every still-open question carries a usable cell.
	 *
	 *  A question counts as answered when it has a non-empty draft, or the user
	 *  explicitly skipped it (which is sent as the empty list the queue's own
	 *  contract defines). A text answer that is only whitespace is not an
	 *  answer — the runtime would store a blank string, which reads as a
	 *  deliberate empty answer to the model. */
	const filled = (id: string): boolean => {
		if (skipped.has(id)) return true;
		const cell = draft.answers[id] ?? [];
		/* DELIBERATELY NOT THE DESKTOP'S RULE. The desktop holds a question open
		   while ANY entry is the blank `Other` marker; here a tick IS a complete
		   answer, so an open-but-empty `Other` beside one does not grey send. The
		   blank marker is dropped by `answerValues` on submit. On a single-select
		   the marker is the cell's only entry, which is blank, so it still holds
		   the question open. */
		return cell.some((value) => value.trim() !== "");
	};
	const complete = open_questions.length > 0 && open_questions.every((q) => filled(String(q.id)));

	async function run(
		kind: "respond" | "decline" | "dismiss",
		send: () => Promise<unknown>,
		onAccepted?: () => void,
	) {
		if (busy) return;
		setBusy(kind);
		setError("");
		setErrorOp("");
		try {
			await send();
			onAccepted?.();
			/* THE DRAFT IS SPENT ONCE THE ASK SETTLES. Cleared here rather than left
			   for the next mount: the sheet re-reads the aggregate, so a settled card
			   can come back in the list, and a re-filled card the user already
			   answered reads as if their answer never landed. */
			clearAskDraft(row.ask_id);
			onSettled?.(kind);
		} catch (failure) {
			setError(refusalText(failure));
			setErrorOp(kind);
			/* A GONE CONVERSATION IS TERMINAL (design round 1, D3). The ask can
			   never be read, so every further tap is a guaranteed 409 — the
			   sentence, already written for THIS op by the daemon (D2), is the
			   remedy, and the controls must stop inviting a tap that cannot
			   work. */
			if (failure instanceof HttpError && failure.code === "ask_session_gone") {
				setGone(true);
			}
			/* A REFUSED DISMISSAL STOPS INVITING ANOTHER (UX round 1, U8). The
			   daemon's most common sentence here is "you already dismissed this":
			   the ask is settled and every further tap is a guaranteed repeat. The
			   reasons stay on screen — the sentence belongs beside the control that
			   produced it (design D5) — but they go inert, and `cancel` is the way
			   back to a fresh attempt if the refusal was momentary (a dead
			   connection rather than a settled ask). */
			if (kind === "dismiss") setDismissRefused(true);
			/* The controls come back: a refusal is a state the user can act on
			   (answer again, or read who beat them to it), and a card left inert
			   by its own error is the greyed-out-buttons defect the blocking card
			   already paid for. */
		} finally {
			/* THE OUTCOME ENDS THE IN-FLIGHT STATE, WHICHEVER OUTCOME IT IS (UX round
			   2, U5). `setBusy("")` lived only in the catch, so a SUCCESSFUL answer
			   left the card asserting `bringing the session up — this can take up to
			   ~30 s` over a dead `…` until — and only if — the sheet's aggregate
			   re-read replaced the row. A refused read (the sheet's own 8 s bound, or
			   a hung backstop) left that lie on screen indefinitely. The promise has
			   settled, so the claim that it is still settling must stop.

			   The controls coming back after a refusal is deliberate: a refusal is a
			   state the user can act on (answer again, or read who beat them to
			   it), and a card left inert by its own error is the
			   greyed-out-buttons defect the blocking card already paid for. */
			setBusy("");
		}
	}

	const respond = () =>
		run("respond", () => {
			/* One body for every question of the ask — the atomic op. A skipped
			   question rides as an empty list, which is the contract's own
			   spelling for "no answer"; omitting the key is what the queue
			   refuses. */
			const answers: Record<string, string[]> = {};
			for (const question of questions) {
				const id = String(question.id);
				if (skipped.has(id)) answers[id] = [];
				else answers[id] = answerValues(draft.answers[id] ?? []);
			}
			return sendCommand(sessionId, { op: "ask_respond", ask_id: row.ask_id, answers });
		});

	const decline = () =>
		run("decline", () =>
			sendCommand(sessionId, { op: "ask_decline", ask_id: row.ask_id }),
		);

	const dismiss = (reason?: "moot" | "answered_in_chat") => {
		setSentReason(reason);
		return run(
			"dismiss",
			() =>
			/* The reason rides only when there is one: a lapsed ask's door is the
			   one-key dismissal the queue has always taken, whose body must stay the
			   shape every older daemon accepts. An OPEN ask's reason is the whole
			   point of the two-choice row — `moot` closes the question, and
			   `answered_in_chat` tells the agent the answer is already in the
			   conversation, so it stops waiting and reads what was typed. */
				sendCommand(
					sessionId,
					reason
						? { op: "ask_dismiss", ask_id: row.ask_id, reason }
						: { op: "ask_dismiss", ask_id: row.ask_id },
				),
			() => onDismissed?.(row.ask_id, reason),
		);
	};

	const settledPairs = answeredPairs(row.questions, row.answers);
	/* INK FOLLOWS THE PROMISE, not the mood (design round 1, D2). An open ask's
	   line was `text-accent` and a settled one `text-success`, two greens that are
	   ΔE 17 apart here and IDENTICAL in three of the 28 shipped themes — so the
	   one distinction this card exists to draw (still waiting vs already settled)
	   could vanish. `accent` is no longer spent on the state line at all: "the
	   agent is continuing" is information, not an achievement, and success green
	   is reserved for the states that are receipts. */
	const toneClass = askToneClass(state.tone);

	const body = (
		<>
			{/* THE STATE LINE STANDS DOWN WHEN THE CARD IS TERMINAL. "Queued — the
			    agent is continuing; expires in 11 m" beside "this conversation no
			    longer exists" is the same contradiction §5's preamble forbids — the
			    timer and the agent both belong to a conversation that is not there.
			    The refusal sentence is the whole state. */}
			{terminal || hideState ? null : (
				<span className={cn("flex flex-wrap items-center gap-x-2 text-meta", toneClass)}>
					<span role="status">{stateLine}</span>
					{row.urgent && offered ? (
						<span className="font-medium text-warning">{messageRuntime.t("relay.asks.card.urgent")}</span>
					) : null}
				</span>
			)}
			{terminal ? null : lead}

			{terminal ? (
				/* TERMINAL, AND IT IS THE SENTENCE THAT SAYS SO (design round 1, D3 for
				   the refusal that arrived; design round 2, D6 for the row the aggregate
				   already reported dead, where the same sentence is shown BEFORE the tap
				   that could only ever be refused). The controls are withheld rather
				   than handed back enabled, because every further tap is a guaranteed
				   409 and the surface must not invite one. The remedy is a NEW message
				   in a new conversation, which is what the daemon's per-op sentence
				   names (D2) when the refusal came from the wire — a dismissal is told
				   nothing was sent, an answer is told to ask again, a decision is told
				   the ask can never be read (U9). */
				<p className="text-body-sm text-danger">
					{gone ? error : messageRuntime.t("relay.asks.card.gone")}
				</p>
			) : lapsed ? (
				<>
					{/* READ-ONLY: the question the agent asked, so the reader knows what
					    lapsed, and nothing to answer it with. Dismiss is the only op, in the
					    pinned bar. A
					    lapsed row is always `timed_out` (`isLapsed`; the index readers
					    re-derive a stale `open` past its deadline), so the Dismiss gate and
					    the queue's own rule are the same statement. It stays apart from any
					    Decline: declining ANSWERS the agent, dismissing answers nobody and
					    buys no turn (the agent is told on its next run — §14). */}
					{questions.map((question, index) => (
						<span
							key={String(question.id ?? index)}
							className="flex items-baseline gap-2 text-body-sm text-ink-muted"
						>
							{questions.length > 1 ? (
								<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
									{index + 1}/{questions.length}
								</span>
							) : null}
							<QuestionText text={question.question} />
						</span>
					))}
				</>
			) : offered ? (
				<>
					{open_questions.map((question, index) => {
						const id = String(question.id);
						return (
							<div key={id} className="flex flex-col gap-1.5">
								<span className="flex items-baseline gap-2 text-body font-medium">
									{questions.length > 1 ? (
										<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
											{index + 1}/{questions.length}
										</span>
									) : null}
									<QuestionText text={question.question} />
								</span>
								{typeof question.command === "string" && question.command.trim() ? (
									<CommandBlock command={question.command.trim()} />
								) : null}
								<QuestionField
									question={question}
									value={draft.answers[id] ?? []}
									disabled={busy !== "" || skipped.has(id)}
									onChange={(next) =>
										patchDraft((current) => ({
											answers: { ...current.answers, [id]: next },
										}))
									}
								/>
								{/* A 44 px TARGET, NOT A 17 px LINK (UX round 1, U5 = design round
								    1, D5). This was `text-meta underline`: measured 136x17 against
								    44 px for every other control in the card, under WCAG 2.5.8's
								    24 px floor and with no spacing exception to claim (its centre
								    sits 8 px under the send button). It is the ONLY way to say
								    "no answer to this question", so it is a control, and it now
								    wears the card family's quiet control shape. */}
								{questions.length > 1 ||
								(Array.isArray(question.options) && question.options.length === 0) ? (
									<button
										type="button"
										disabled={busy !== ""}
										onClick={() => {
											const next = new Set(skipped);
											if (next.has(id)) next.delete(id);
											else next.add(id);
											patchDraft(() => ({ skipped: Array.from(next) }));
										}}
										className={cn(
											"flex min-h-11 self-start items-center rounded-sm px-3 text-body-sm active:bg-elevated disabled:opacity-50",
											skipped.has(id) ? "text-ink" : "text-ink-muted",
										)}
									>
										{skipped.has(id)
											? messageRuntime.t("relay.asks.card.unskip")
											: messageRuntime.t("relay.asks.card.skip")}
									</button>
								) : null}
							</div>
						);
					})}

					{/* R6 (agent review round 1): an answerable ask whose questions are all
					    already taken elsewhere used to render no fields and a permanently
					    disabled send, with nothing saying why. The queue is right (a draft
					    elsewhere is not this card's to submit); the card just has to say
					    so instead of inviting a tap that cannot land. */}
					{open_questions.length === 0 ? (
						<p className="text-body-sm text-ink-muted">
							{messageRuntime.t("relay.asks.card.nothing_left")}
						</p>
					) : null}

					{/* THE STALE-ASK CONTEXT LINE (SA-D7 / R5), UNDER the question it
					    refers to (design round 1, D3): it used to sit above the question,
					    so "this was asked" pointed at a sentence the reader had not read
					    yet — and the terminal paints the same line under its row. It is
					    the card's one statement that the question may already be settled
					    in chat, which is exactly the case the `answered_in_chat` reason
					    exists for. No line at all when the runtime sent no count (an older
					    runtime), when the count is zero, or on a settled card — see
					    `contextLine`. */}
					{(() => {
						const context = terminal ? null : contextLine(row);
						return context ? (
							<p className="text-meta text-ink-dim" data-testid="ask-context">
								{context}
							</p>
						) : null;
					})()}

				</>
			) : (
				<>
					{/* A settled ask: the questions and the answers chosen, so the
					    card is a receipt rather than a sentence. `row.answers` holds a
					    secret answer as the KEY the runtime stored, never a value. */}
					{/* PAIRING CARRIED BY SPACE, not only by ink (design round 1, N1):
					    the receipt's inside-pair gap and its between-pair gap measured
					    ~10 px vs ~12 px, so a two-question receipt read as one
					    four-line block. The gap BETWEEN pairs is now the container's;
					    the gap inside a pair stays tight. */}
					{status === "dismissed" && settledPairs.length > 0 ? (
						/* A DISMISSED ROW HAS NO PAIRS, so it is not a `<dl>` (UX round
						   2, U5). U3 removed the answer slot on a dismissal, which left
						   each `<dt>` without its `<dd>` — a description list whose
						   entries describe nothing, for a reader on assistive tech as
						   much as on screen. The questions are the whole receipt here, so
						   they are plain lines in the same register, one per question:
						   dropping them would lose what the row is a receipt FOR, and a
						   visually hidden `<dd>` would keep a value that does not exist. */
						<div className="flex flex-col gap-1">
							{settledPairs.map((pair) => (
								<span key={pair.question} className="text-body-sm text-ink-muted">
									{pair.question}
								</span>
							))}
						</div>
					) : settledPairs.length > 0 ? (
						<dl className="flex flex-col gap-3">
							{settledPairs.map((pair) => (
								<div key={pair.question} className="flex flex-col">
									<dt className="text-body-sm text-ink-muted">{pair.question}</dt>
									{/* The bare `—` is the settled register's placeholder for
									    "an answer was expected here and is missing" (a declined
									    row's, pre-existing); every status that reaches THIS
									    branch answered or was put away by someone else, so it
									    keeps the placeholder exactly as it was. */}
									<dd className="text-body-sm text-ink">{pair.answer || "—"}</dd>
								</div>
							))}
						</dl>
					) : (
						/* GUARDED like every other read of a wire list (agent review round
						   1, R5): a settled row that arrives without `questions` used to throw
						   during render, which unmounts the whole sheet rather than
						   degrading to a line. */
						<span className="text-body-sm text-ink-muted">
							{questions[0]?.question ?? ""}
						</span>
					)}
					{/* NO DISMISS HERE, deliberately (agent review round 1, R1): dismiss is
					    a `timed_out` action and a timed-out ask renders in the LAPSED
					    branch above, so every status that reaches this branch (answered,
					    late, declined, dismissed, withdrawn, expired, unknown) can only
					    be refused. An EXPIRED ask additionally keeps no error register at
					    all (§5) — the remedy is in the state line. */}
				</>
			)}
		</>
	);

	/* THE ANSWERING CONTROLS (ask-panel parity, R2). Everything the reader does
	   to ANSWER this card, and everything that explains why they cannot yet, lives
	   in one block so it can be pinned: the refusal, the cold-start wait, why send
	   is grey, the send/decline pair, and the quiet way out. Nothing in it is part
	   of the question, so a long question can scroll without taking the buttons
	   with it. Absent on a terminal card (its sentence is the whole state), a
	   settled receipt (nothing to do) and an `expired` one. */
	const keyboardUp = useKeyboardUp();
	const hasBar = !terminal && (offered || (lapsed && dismissible));
	const unfilled = open_questions.filter((question) => !filled(String(question.id)));
	/* WHY SEND IS GREY, said in words (the native panel's rule): a disabled button
	   alone gives the reader no reason. Only while the form is incomplete and
	   nothing is in flight (a busy button is dimmed for a different reason). A
	   multi-question ask names how far along it is, counting questions taken
	   elsewhere as answered, as the gate does; a single question says what it takes. */
	const progressLine =
		offered && busy === "" && unfilled.length > 0
			? questions.length > 1
				? messageRuntime.t("relay.asks.card.progress_tally", {
						answered: String(Math.max(0, questions.length - unfilled.length)),
						total: String(questions.length),
					})
				: Array.isArray(open_questions[0]?.options) && open_questions[0].options.length > 0
					? messageRuntime.t("relay.asks.card.progress_pick_or_type")
					: messageRuntime.t("relay.asks.card.progress_type")
			: null;
	const bar = !hasBar ? null : lapsed ? (
		/* A LAPSED ASK'S ONLY OP, with no reason asked: the agent already walked past
		   its deadline, so nothing turns on the operator naming one. It stays apart
		   from any Decline: declining ANSWERS the agent, dismissing answers nobody and
		   buys no turn (the agent is told on its next run — §14). */
		<div className="flex flex-col gap-2" data-testid="ask-bar">
			<button
				type="button"
				disabled={busy !== ""}
				aria-busy={busy === "dismiss" || undefined}
				onClick={() => dismiss()}
				className="flex min-h-11 w-full items-center justify-center rounded-sm px-3 text-body-sm text-ink-muted active:bg-surface disabled:opacity-50"
			>
				{/* The label IS the width, so it stays put while the dismissal is in
				    flight: it dims (`disabled`) and says it is busy, and never collapses
				    to a bare glyph under the thumb (design round 1, D1).

				    NO COLD LINE, AND NO COLD CLOCK (fork A, 2026-10-16): a dismissal takes
				    the no-boot door on every surface, one append against the log, so
				    "dismiss — send no reply" is the WHOLE truth, warm or cold. */}
				{messageRuntime.t("relay.asks.card.dismiss")}
			</button>
			{error && errorOp === "dismiss" ? <p className="text-body-sm text-danger">{error}</p> : null}
		</div>
	) : (
		<div className="flex flex-col gap-2" data-testid="ask-bar">
			{closing ? (
				/* THE REASONS REPLACE THE ANSWERING PAIR while they are up (ask-panel
				   parity, as on the native panel). They used to sit 8.5 px under a live
				   `decline`, so a reader with a drafted answer could read "Answered in
				   chat" as "send my answer" (design round 1, D1 = UX U6). The heading
				   names the act (the same words the terminal uses) and each row states
				   what the agent will be told.

				   A TAP ON `Dismiss…` SENDS NOTHING: the reason row is a choice, not a
				   dismissal. IT IS NOT A SECOND `decline`: declining ANSWERS the agent
				   ("no"), buys nothing and is irreversible; closing says the question
				   stopped mattering and asks the agent for nothing at all, and on a live
				   conversation it need not even start a turn (a dismissal is held durably
				   until the agent's next turn). */
				<div className="flex flex-col gap-1">
					<span className="text-meta text-ink-dim">
						{messageRuntime.t("relay.ask.dismiss.title", { count: questions.length })}
					</span>
					{DISMISS_REASONS.map((reason) => (
						<button
							key={reason}
							type="button"
							disabled={busy !== "" || dismissRefused}
							aria-busy={(busy === "dismiss" && sentReason === reason) || undefined}
							onClick={() => dismiss(reason)}
							className={cn(
								"flex min-h-11 w-full flex-col items-start justify-center gap-0.5 rounded-sm px-2 py-1.5 text-start active:bg-surface disabled:opacity-50",
								sentReason === reason && busy === "dismiss" ? "text-ink" : "text-ink-muted",
							)}
						>
							{/* THE PRESSED LABEL STAYS PUT (UX round 1, U3): the pressed row
							    keeps its words and gains the in-flight mark, the sibling dims. */}
							<span className="text-body-sm">
								{messageRuntime.t(DISMISS_COPY[reason].key)}
								{busy === "dismiss" && sentReason === reason ? " …" : ""}
							</span>
							<span className="text-meta text-ink-dim">
								{messageRuntime.t(DISMISS_COPY[reason].note)}
							</span>
						</button>
					))}
					<button
						type="button"
						disabled={busy !== ""}
						onClick={() => {
							/* THE REFUSAL GOES WITH THE BLOCK IT EXPLAINED (UX round 2, U10 =
							   agent review round 2, m1): leaving it behind put an old refusal
							   under a fresh-looking `Dismiss…`. */
							setClosing(false);
							setDismissRefused(false);
							setError("");
							setErrorOp("");
						}}
						className="flex min-h-11 items-center self-start rounded-sm px-2 text-body-sm text-ink-dim active:bg-surface disabled:opacity-50"
					>
						{messageRuntime.t("relay.ask.dismiss.cancel")}
					</button>
				</div>
			) : (
				<>
					{/* THE COLD ANSWER'S CLOCK, STATED (design round 1, D1 = UX U1). A cold
					    answer is an engage plus a dial plus an ack in ONE call — ~30 s
					    measured, against the 1–3 s a warm answer pays — and until this line
					    existed the whole in-flight affordance was a bare `…`. NOT FOR A
					    DISMISSAL (agent review round 1, M1): the engine takes the no-boot
					    door on every surface, so the ~30 s bring-up belongs to the ANSWER and
					    DECLINE below, which really do engage. The gate is the op, not the row:
					    `cold` says what an answer would cost. */}
					{busy !== "" && busy !== "dismiss" && cold ? (
						<p className="text-body-sm text-ink-muted">
							{messageRuntime.t("relay.asks.card.cold_wait")}
						</p>
					) : null}
					{/* THE REFUSAL SITS WITH THE CONTROLS IT ANSWERS (design round 1, D4/D5): a
					    dismiss refusal renders beside the reasons instead, above. */}
					{error && errorOp !== "dismiss" ? (
						<p className="text-body-sm text-danger">
							{error}
						</p>
					) : null}
					{/* EQUAL HEIGHT, SIDE BY SIDE; STACKED when the labels would not fit. A
					    wrapping row of `flex-1 basis-0 min-w-fit` buttons is both at once
					    with no measuring: while both labels fit on one line they share the
					    row and stretch to one height, and when their no-wrap widths no longer
					    fit (a 320 px phone, a long locale, a larger root font size, since the
					    widths are rem-based) the row wraps and each takes a full row. Side by
					    side at large text, "Send answer" wrapped to two lines beside a
					    one-line Decline and the bar took a third of the sheet. The label
					    STAYS put while in flight: the button dims and says it is busy. */}
					<div className="flex flex-wrap gap-2">
						<button
							type="button"
							disabled={busy !== "" || !complete}
							aria-busy={busy === "respond" || undefined}
							onClick={respond}
							className="flex min-h-11 min-w-fit flex-1 basis-0 items-center justify-center rounded-sm bg-accent px-4 text-body-sm font-medium whitespace-nowrap text-on-accent active:bg-accent-active disabled:opacity-50"
						>
							{open_questions.length > 1
								? messageRuntime.t("relay.asks.card.send_answers")
								: messageRuntime.t("relay.asks.card.send_answer")}
						</button>
						<button
							type="button"
							disabled={busy !== ""}
							aria-busy={busy === "decline" || undefined}
							onClick={decline}
							className="flex min-h-11 min-w-fit flex-1 basis-0 items-center justify-center rounded-sm bg-danger-wash px-3 text-body-sm whitespace-nowrap text-danger active:bg-danger-wash disabled:opacity-50"
						>
							{messageRuntime.t("relay.asks.card.decline")}
						</button>
					</div>
					{/* WHY SEND IS GREY and the quiet way out share ONE row, so the bar stays
					    compact. `Dismiss…` is a text link, not a button: dismissing answers
					    nobody and buys no turn, so it must not compete with Send and Decline
					    (R5, §15's amended D6; it asks WHY first, hence the ellipsis). It
					    steps aside while the on-screen keyboard is up, which is when the
					    question needs the room. */}
					{progressLine || (reason_needed && !keyboardUp) ? (
						<div className="flex items-center gap-x-3">
							<p className="min-w-0 flex-1 text-start text-meta text-ink-dim">{progressLine}</p>
							{reason_needed && !keyboardUp ? (
								<button
									type="button"
									disabled={busy !== "" || dismissRefused}
									onClick={() => setClosing(true)}
									className="flex min-h-11 shrink-0 items-center rounded-sm px-1 text-meta text-ink-muted active:bg-surface disabled:opacity-50"
								>
									{messageRuntime.t("relay.ask.dismiss.open")}
								</button>
							) : null}
						</div>
					) : null}
				</>
			)}
			{/* THE REFUSAL SITS WITH THE CONTROL THAT PRODUCED IT (design round 1, D5): a
			    relay or bridge ask refuses the dismissal in the queue's own words, and
			    that sentence belongs beside the reason the operator pressed. */}
			{error && errorOp === "dismiss" ? (
				<p className="text-body-sm text-danger">
					{error}
				</p>
			) : null}
		</div>
	);

	const card = useRef<HTMLDivElement>(null);
	/* AFTER A SEND, THE READER'S PLACE MOVES TO THE NEXT QUESTION (ask-panel parity):
	   the control that was focused (send) is gone with the card it belonged to, so
	   without this focus falls to the page and a keyboard or screen-reader user
	   starts over. The question text is the first thing the card says after its
	   state line, and it is focusable already (the capped scroller's own tab stop).
	   Spent once, on mount: a card that is merely re-rendered must not take focus. */
	useEffect(() => {
		if (!focusQuestion) return;
		card.current
			?.querySelector<HTMLElement>('[data-testid="ask-question-text"]')
			?.focus({ preventScroll: true });
		onFocused?.();
		// Mount-only on purpose: `focusQuestion` is the trigger, not a value to track.
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [focusQuestion]);

	return (
		<>
			<div
				ref={card}
				data-testid="ask-card"
				data-ask-id={row.ask_id}
				data-ask-status={String(row.status || "open")}
				className="flex flex-col gap-2 rounded-md border border-hairline bg-surface p-2.5"
			>
				{body}
				{footerSlot === undefined ? bar : null}
			</div>
			{footerSlot ? createPortal(bar, footerSlot) : null}
		</>
	);
}

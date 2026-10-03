/**
 * Session list (`#/`) — the phone's home. One card per live session, kept
 * current by the list SSE; footer row with new session, past sessions,
 * projects and the theme picker.
 *
 * Visual contract: a streaming session shimmers its name (the row itself is
 * the indicator — no spinner); a session waiting on the user carries the
 * danger dot and a word ("approval" / "question"), because that is the one
 * card that needs a decision (branding §7).
 *
 * SECTIONS AND ORDER COME FROM THE DAEMON, and this screen re-derives neither.
 * The server sorts every row on the shared catalog key
 * (`session.catalog.CatalogEntry.rank` — the same key the terminal sidebar and
 * the desktop app sort on) and marks each row `active`/`previous` with the
 * shared `active` rule, so the list is STABLE across activity refreshes (the
 * jitter this change removes) and identical to the other two surfaces. This
 * screen only GROUPS what it is given: ★ Pinned, Active Sessions, Previous
 * Sessions — the sidebar's own section order, minus the subagent layer the
 * phone does not carry. A pin is a display lift like the sidebar's, so it never
 * changes the daemon's ranking and pinning a row never moves it inside its own
 * section.
 */
import {
	useCallback,
	useEffect,
	useLayoutEffect,
	useRef,
	useState,
	type Ref,
} from "react";
import { getDirectories, setSessionPin } from "../api";
import { ProjectsSheet } from "../components/projects-sheet";
import { Sheet } from "../components/ui/sheet";
import { Spinner } from "../components/spinner";
import { navigate } from "../router";
import {
	applySessionPin,
	clearSessionPinMark,
	retainSessionListStream,
	usePinMarks,
	useSessions,
} from "../store";
import { WideViewButton } from "../components/wide-view-button";
import { applyTheme, getTheme, THEMES } from "../theme";
import { shortenHome } from "../lib/format";
import { MARK_DATA_URI } from "../lib/mark";
import { clampPinReason, pinRefusalReason } from "../lib/pin-refusal";
import type { SessionSummary } from "../types";
import { cn } from "../lib/cn";

/** The shared noun for a delegated child, in the singular at one.

    "subagent" and not "agent": that is the record's own field name, the word
    `/info` tallies in and the word the TUI's own stop notice uses. "Agent"
    alone is ambiguous in a product with an "Agents" page of reusable PROFILES.
    Singular at 1 matches the `Scheduled (1 wake)` style the rest of the product
    uses. */
function subagentNoun(n: number): string {
	return n === 1 ? "subagent" : "subagents";
}

/** Past this, the chip stops spelling the count and CAPS it.

    A `shrink-0` chip is paid for by the row's name, and the name is the row's
    identity while the count is context (UX round 1, U4 — the 360px frame shows
    `Parent at capacity, twelve parked …` truncated to ~31 cells beside a
    12-cell chip). Three digits is where that trade stops being worth making,
    and the exact figure is one tap away in the session view's roster header. */
const COUNT_CAP = 99;

/** The right-cluster chip for a session's delegated children.

    IT COUNTS `running + queued` BY DESIGN, which is deliberately NOT how the
    catalogue label counts: `CatalogEntry.status` prints `2 subagents running ·
    1 queued`, the precise form for a row with room for a sentence. This chip
    has room for a number, and the number it must agree with is the one the
    SESSION VIEW shows after a tap — the roster header prints
    `{running}/{direct.length} running`, and both sides of that come from the
    folded projection, where a parked child is drawn as running
    (`mobile/projection.py` maps `queued` to the `running` mobile status).

    So `1 running + 1 queued` reads `2 subagents` here and `2/2 running` there:
    one number for one population across the tap (UX round 1, U2, which caught
    the two disagreeing). The fold itself — queued drawn as running — is a
    pre-existing presentation in a different subsystem, recorded on the PR as a
    deferred finding rather than changed here.

    The noun is always present, including for a parent with nothing running,
    where the count-only form would have read `3 queued` and left the reader to
    guess WHAT was queued (UX round 1, U3). */
function delegatedChip(children: number): string {
	const shown = children > COUNT_CAP ? `${COUNT_CAP}+` : String(children);
	return `${shown} ${subagentNoun(children)}`;
}

/** The right-cluster word `new`. It lingers through a 120ms opacity fade when
    the mark clears (session opened) instead of blinking out — but it MUST
    unmount once the fade lands: an opacity-0 `shrink-0` span would keep
    pushing the `N agents` / `N todo` chips rightward forever. The fade is a
    transition, so the global prefers-reduced-motion block caps it to instant
    for free; the unmount timer still runs its course. */
function NewMark({ visible }: { visible: boolean }) {
	const [mounted, setMounted] = useState(visible);
	/* The unmount timer is driven by the visible -> hidden TRANSITION, so the
	   effect tracks the previous `visible` in a ref rather than depending on
	   `mounted` — the state it sets itself. Depending on your own output is
	   the shape that becomes a re-entrant timer the moment someone adds a
	   branch: correct here only because of a guard, and a trap next edit. */
	const wasVisible = useRef(visible);
	useEffect(() => {
		const had = wasVisible.current;
		wasVisible.current = visible;
		if (visible) {
			setMounted(true);
			return;
		}
		if (!had) return;
		/* Matches --transition-duration-fast (120ms): the timeout only removes
		   the node after the CSS fade has landed. */
		const timer = setTimeout(() => setMounted(false), 120);
		return () => clearTimeout(timer);
	}, [visible]);
	if (!mounted) return null;
	return (
		<span
			/* Sighting users see `new`; assistive tech hears the unambiguous
			   phrase. While fading out the word is already meaningless, so it
			   leaves the accessibility tree at once. */
			aria-label={visible ? "new activity" : undefined}
			aria-hidden={visible ? undefined : true}
			className="shrink-0 text-meta text-accent"
			style={{
				opacity: visible ? 1 : 0,
				transition:
					"opacity var(--transition-duration-fast, 120ms) ease-out",
			}}
		>
			new
		</span>
	);
}

function SessionCard({
	s,
	home,
	pinned,
	onLongPress,
	ref,
}: {
	s: SessionSummary;
	home: string;
	/* The pin as RENDERED — the user's unanswered mark over the daemon's
	   confirmed flag. A separate input from `s.pinned` on purpose: a row may
	   show its ★ long before the daemon confirms the pin, and the mark must
	   never be mistaken for the fact that decides which section the row renders
	   in. */
	pinned: boolean;
	/* A long-press opens the pin action sheet for this row. Passed in rather
	   than handled here so the card stays a pure presentation of one summary and
	   the gesture's timer lives with the screen that owns the sheet — and so the
	   card can be rendered in a test with no gesture machinery at all. */
	onLongPress?: () => void;
	/* FLIP anchor: the list measures every card before/after a reorder so it
	   can settle it into its new slot instead of teleporting it. */
	ref?: Ref<HTMLButtonElement>;
}) {
	/* A POINTER, NOT A HOVER: a phone has no hover, so a long-press is the one
	   gesture that reliably means "more actions for this row" without a visible
	   control on every card (which would cost width the title needs). A press
	   that moves — a scroll — CANCELS the timer, or a flick through the list
	   would open the sheet on whatever row the finger happened to pass over. */
	const pressTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
	const pressedAt = useRef<{ x: number; y: number } | null>(null);
	/* Set by the long-press firing; read and reset by the click that follows it. */
	const suppressClick = useRef(false);
	const cancelPress = () => {
		if (pressTimer.current !== null) {
			clearTimeout(pressTimer.current);
			pressTimer.current = null;
		}
		pressedAt.current = null;
	};
	/* CLEANUP ON UNMOUNT (review round 1, MINOR 2). The list re-renders rows
	   constantly, and a press in flight when a row unmounts — or when the whole
	   screen is left by a navigation — would otherwise fire its ``setTimeout``
	   against a torn-down tree: ``onLongPress`` would open a sheet for a row that
	   is no longer on screen. */
	useEffect(() => cancelPress, []);
	const onPointerDown = (event: React.PointerEvent) => {
		if (!onLongPress) return;
		/* A NEW PRESS CLEARS THE SUPPRESSION, so it can never outlive the gesture
		   that set it (review round 1, NIT 1). Without this a long-press whose
		   finger lifted off the card (a ``pointerleave`` with no following click)
		   left the flag set, and the user's NEXT genuine tap was swallowed. */
		suppressClick.current = false;
		pressedAt.current = { x: event.clientX, y: event.clientY };
		pressTimer.current = setTimeout(() => {
			pressTimer.current = null;
			/* A long-press must not ALSO fire the card's onClick and navigate:
			   `cancelPress` clears the timer on pointerup, and this flag is what
			   tells the click handler to stand down. */
			suppressClick.current = true;
			onLongPress();
		}, 450);
	};
	const onPointerMove = (event: React.PointerEvent) => {
		const start = pressedAt.current;
		if (start === null) return;
		/* 10px of travel is a scroll, not a press held still. */
		if (Math.hypot(event.clientX - start.x, event.clientY - start.y) > 10) {
			cancelPress();
		}
	};
	/* THE MIRRORED WORD, AND WHEN IT IS RETIRED (UX round 1, U4 = agent review
	   round 1, R3). `pending_kind === "ask"` is the daemon's projection of the
	   LEGACY single-slot mirror — the same card the session screen already
	   refuses to draw twice once `asks` is published (§4's client rule N3). The
	   rule applies to the WORD too: with `asks_open` present this row rendered
	   "question" immediately followed by its own asks chip, two quantities for
	   one queue under two spellings. The mirrored word survives only where the
	   asks field is ABSENT, which is exactly the old-client skew it exists for. */
	const mirroredAskWord = s.pending_kind === "ask" && s.asks_open === undefined;
	const pendingLabel =
		s.pending_kind === "approval"
			? "approval"
			: mirroredAskWord
				? "question"
				: null;
	/* Flags coexist in data; exactly one state renders. The daemon classifies
	   non-streaming unread outcomes above work in progress, but a resumed
	   session must show its current work rather than its stale outcome. A session
	   blocked on a decision must be opened anyway, so the unread mark would
	   add noise; a streaming session is drawing the eye already, and "new"
	   marks COMPLETED unviewed activity, never in-flight work. */
	const decision = Boolean(s.needs_attention && pendingLabel);
	const unread = Boolean(s.unseen) && !decision && !s.streaming;
	/* The delegated-work counts, normalised once, and the derived state the slot
	   ladder and the chip both read.

	   `null` (or a field an older daemon never sent) means THE DAEMON DID NOT
	   REPORT A COUNT, which is not the same fact as zero children: a durable-only
	   row has no live record to read, and a pre-field record has no field. Both
	   arms below stay silent about it — no mark, no chip — rather than rendering
	   "0", which would tell the operator there are no subagents on a session the
	   phone never managed to ask.

	   "Queued with nothing running" COUNTS as delegating. A child parked waiting
	   for a capacity slot is not spending anything, but the parent is certainly
	   not idle, and that is the one shape a reader cannot infer from the running
	   count alone.

	   A LEAVING SESSION ADVERTISES NOTHING (UX round 1, U1). A signalled runtime
	   is still working — that is what makes it leave politely — and its children
	   are still running, but the phrase the list itself prints for that row is
	   "Leaving…", so a mark drawn from the counts alone would say two things at
	   once. The daemon already refuses to REPORT counts for an entry it cannot
	   vouch for (a degraded dial, a stopped heartbeat, a runtime on its way out —
	   `_advertisable_counts`); this gate is what covers a client talking to a
	   build that predates that refusal. */
	const leaving = Boolean(s.leaving);
	const running = typeof s.subagents_running === "number" ? s.subagents_running : null;
	const queued = typeof s.subagents_queued === "number" ? s.subagents_queued : null;
	const children = (running ?? 0) + (queued ?? 0);
	const delegating = !leaving && children >= 1;
	/* The second line's cwd half, computed once so the D1 reservation below can
	   test emptiness against exactly what is rendered. */
	const cwdText = home ? shortenHome(s.cwd, home) : s.cwd;
	return (
		<button
			ref={ref}
			type="button"
			onPointerDown={onPointerDown}
			onPointerMove={onPointerMove}
			onPointerUp={cancelPress}
			onPointerLeave={cancelPress}
			onPointerCancel={cancelPress}
			onContextMenu={(event) => event.preventDefault()}
			onClick={() => {
				if (suppressClick.current) {
					suppressClick.current = false;
					return;
				}
				navigate(`/s/${encodeURIComponent(s.session_id)}`);
			}}
			className="flex min-h-11 w-full flex-col gap-0.5 rounded-md px-2 py-1.5 text-left select-none active:bg-elevated"
		>
			<div className="flex items-center gap-2">
				{/* ONE reserved indicator slot serving every state, so every title
				    starts at the same x forever — indicators change colour,
				    never geometry.

				    The slot is sized to the LARGEST occupant (the 12px spinner)
				    and the 6px dot is centred inside it. Rendering the spinner
				    BESIDE the slot, as this did before, defeated the whole point:
				    a streaming row paid slot + gap + spinner and its title sat
				    ~19.5px right of every other row's, so the reserved-slot
				    promise held for three states and broke on the fourth — the
				    one that changes most often. Spinner itself is untouched; it
				    is a shared component and its own size is correct.

				    THE LADDER decides what occupies the slot, not `streaming`.
				    An approval gate runs INSIDE a turn (the harness blocks in
				    _execute_tool_calls, between agent_start and agent_end), so
				    the ordinary "may I run this?" carries streaming AND pending
				    at once. Testing `streaming` first therefore replaced the red
				    pulse with a neutral spinner on exactly the loudest state in
				    the ladder — the pulse is the only motion reserved for danger,
				    and it was being spent on ordinary work. Decision wins the
				    slot; "working" is still carried on that row by the title's
				    shimmer (applied below whenever `s.streaming`), which is
				    sufficient and costs no geometry — a second mark would have
				    to come out of the same 12px box the alignment depends on. */}
				<span
					className="flex size-3 shrink-0 items-center justify-center"
					aria-hidden={!decision && s.streaming ? undefined : true}
				>
					{decision ? (
						<span className="lo-pulse inline-block size-1.5 rounded-full bg-danger" />
					) : s.streaming ? (
						/* The obvious in-progress mark beside the title: a small
						   loading wheel, not just the text sweep — the sweep alone
						   was too subtle to catch at a glance. */
						<Spinner />
					) : unread ? (
						<span className="inline-block size-1.5 rounded-full bg-accent" />
					) : delegating ? (
						/* DELEGATING — the parent's own turn is not running but it still
						   owns children. BELOW unread on purpose: the unread dot is a
						   receipt for an outcome the operator has not seen yet, and a
						   receipt must never be masked by live activity (the same rule
						   the desktop and TUI ladders follow). ABOVE plain idle, which
						   is the whole point of the state.

						   TWO 4px DOTS rather than one, because this package carries no
						   icon dependency and a single dot would be indistinguishable
						   from the unread mark sitting one rung above it in the same
						   slot and the same accent ink. The pair reads as "more than
						   one thing moving", and the SHAPE is what separates the two
						   states — the TUI's own reasoning for putting `⇉` in its
						   column instead of recolouring `●`.

						   2 × 4px + 2px gap = 10px inside the existing 12px slot, so the
						   ladder's geometry and the title's start x are untouched. The
						   dots are aria-hidden (the slot already is): the COUNT is
						   carried as text by the chip below, which is what a screen
						   reader should hear — a decorative mark is the wrong place
						   for a number. */
						<span className="flex items-center gap-[2px]">
							<span className="inline-block size-1 rounded-full bg-accent" />
							<span className="inline-block size-1 rounded-full bg-accent" />
						</span>
					) : (
						<span className="inline-block size-1.5 rounded-full bg-transparent" />
					)}
				</span>
				{decision && s.streaming ? (
					/* The "working" announcement for assistive tech, carried as TEXT
					   rather than as a mark in the slot. The spinner owns
					   role="status" aria-label="working" and covers every OTHER
					   streaming row; but the ladder gives the slot to the danger
					   dot on a decision+streaming row and hides the slot there, so
					   a screen-reader user heard the pending word and nothing about
					   the turn being mid-flight — the shimmer is a pure CSS paint
					   and conveys nothing to AT. Rendered ONLY for that row, so a
					   plain streaming row is not announced twice. Zero geometry
					   cost: sr-only is clipped out of the layout, so this cannot
					   revive the title shift D2 fixed. */
					<span className="sr-only" role="status">
						working
					</span>
				) : null}
				<span
					className={cn(
						"min-w-0 flex-1 truncate text-body-sm font-medium",
						s.streaming && "lo-shimmer",
					)}
				>
					{s.conversation_name || "untitled"}
				</span>
				{decision && pendingLabel ? (
					<span className="shrink-0 text-meta text-danger">
						{pendingLabel}
					</span>
				) : null}
				{/* SESSION HEALTH, SAID OUT LOUD (mobile UX batch 2, U7). The summary
				    now carries the daemon's own `ended`/`degraded` receipts (the same
				    facts its projection serves), and a row that rendered neither was
				    indistinguishable from a live one: an ended conversation looked
				    openable-and-running forever, and a degraded one — whose socket the
				    relay cannot reach — looked exactly as live as a healthy session.
				    Quiet by design (`text-ink-dim`, no dot, no danger ink): this is
				    state a reader should notice on the row that has it, not an alarm
				    that outranks the decision word beside it. `not answering` is the
				    register the TUI uses for the same condition. */}
				{s.ended ? (
					<span className="shrink-0 text-meta text-ink-dim">ended</span>
				) : null}
				{s.degraded ? (
					<span className="shrink-0 text-meta text-ink-dim">not answering</span>
				) : null}
				{/* THE AGENT-OPENED MARK (design §10.4). This conversation is a
				    listed workstream an agent opened on the operator's behalf, not
				    one the operator started — the 2026-09-18 confusion class, one
				    surface out. Drawn from the PRESENCE of `opened_by`, never from a
				    member: a top-level requester's object has all three members
				    null and the row is still not the operator's own. Quiet like
				    `ended`/`not answering` beside it — a fact to notice on the row
				    that has it, not an alarm — and text, never a glyph (this
				    package carries no icon dependency). `agent` is the word the
				    product already uses for the certain half of the opener: the
				    TUI's own gutter falls back to exactly it when the role cannot
				    be read (`session/preview.py`). */}
				{s.opened_by != null ? (
					<span className="shrink-0 text-meta text-ink-dim">
						{/* D1 (round 1): a bare `agent` is a noun, and a screen reader
						    heard "Anonymous workstream agent" as the session's identity
						    rather than its origin. The sr-only prefix makes the
						    accessible name read "… opened by agent" — the TUI's own
						    words for the same fact — at zero geometry cost: sr-only is
						    clipped out of the layout, the same idiom as the `working`
						    status span above, so the measured 40 px chip budget stands. */}
						<span className="sr-only">opened by </span>
						agent
					</span>
				) : null}
				{/* A PINNED ROW CARRIES ITS ★, and it rides the RIGHT cluster rather than
				    the state slot. The TUI made exactly this call (`session_sidebar.py`
				    `_special_mark`): a ★ in the state column made the sessions a user
				    cares about most the only ones that could not report being blocked,
				    broken or finished, because the pin is the DURABLE fact and the state
				    glyph is the volatile one — so the pin moves, not the state. The ★ is
				    also the shape the ★ Pinned heading uses, so the mark and its section
				    cannot disagree about what it means.

				    Drawn from the RENDERED pin, not `s.pinned`: the reader's own press
				    must show its ★ in the commit that handles the tap, while the row
				    itself waits for the daemon (the mark/section split in `store.ts`). */}
				{pinned ? (
					<span
						className="shrink-0 text-meta text-accent"
						aria-label="pinned"
					>
						★
					</span>
				) : null}
				{/* State word rides BEFORE the count chips in the right cluster
				    (spec §1): `new` truncates the title only, row height never
				    changes. */}
				<NewMark visible={unread} />
				{delegating ? (
					/* The delegated-work chip. It keeps its place in the right cluster (the
					   state word `new` rides before it) and its own geometry; what it says
					   is `delegatedChip`'s business — the shared noun, one number for the
					   population the session view counts, and a cap past three digits so a
					   wide count cannot shrink-0 the name away (UX round 1, U2/U3/U4).

					   `subagents` and not `agents`: the record's own field name, the word
					   `/info` tallies in, and the word the TUI's own stop notice uses.
					   `agents` alone is ambiguous here — this product has an "Agents" page of
					   reusable profiles.

					   Text, not a glyph: ⟳ and ☐ render as tofu boxes on phones whose
					   system font lacks those codepoints. Text marks survive every font. */
					<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
						{delegatedChip(children)}
					</span>
				) : null}
				{s.todos_open ? (
					<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
						{s.todos_open} todo
					</span>
				) : null}
				{/* THE OUTSTANDING-ASKS CHIP (design §4/§5.0). DISTINCT FROM THE APPROVAL
				    STATE and wearing the accent for exactly that reason: the row's
				    `needs_attention`/`pending` arm means a decision is holding the run,
				    while a queued ask is a question the agent keeps working through — a
				    reader who could not tell them apart would learn to ignore both. The
				    accent is the same ink the minimized bar's `?` glyph uses, so the two
				    surfaces name one state with one colour.

				    ABSENT AT ZERO, never a zero badge: the field itself is absent while
				    the runtime cannot report asks, and `0` is a session with nothing
				    outstanding, which needs no mark. */}
				{typeof s.asks_open === "number" && s.asks_open > 0 ? (
					/* THE UNIT IS THE FIELD'S OWN (agent review round 1, R3): `asks_open`
					   counts ASKS (open, or timed-out and still answerable — the
					   outstanding set), and this chip used to print that number with the
					   word "questions" — so the PR's own fixture showed "2 questions"
					   where the bar above it said "4 questions waiting" and the header
					   said 3. The chip now says what it counts ("2 asks"); the surfaces
					   that hold the whole ask list state questions, which is §5.0's own
					   unit for the bar.

					   THE ARIA LABEL IS NOT "waiting" (M1, asks-open widening): the
					   field's set now includes a timed-out-but-answerable ask, and §5's
					   header rule is explicit that an outstanding ask "is not a
					   'waiting for you' state" — the agent keeps working. So the label
					   uses this client's own word for the set (`outstandingAsks`,
					   `lib/asks.ts`) rather than the reading the design forbids. */
					<span
						className="shrink-0 font-mono text-mono-sm text-accent"
						aria-label={
							s.asks_open === 1 ? "1 ask outstanding" : `${s.asks_open} asks outstanding`
						}
					>
						{s.asks_open} ask{s.asks_open === 1 ? "" : "s"}
					</span>
				) : null}
			</div>
			<div className="flex items-baseline gap-2">
				<span className="min-w-0 truncate font-mono text-mono-sm text-ink-dim">
					{cwdText}
					{/* D1 (round-1 design review): a second line with NOTHING on it
					    rendered 44.00px against its neighbours' 50.89px — a 6.89px
					    density break for every row below it. No live or durable row
					    was measured rendering that shape (both carry
					    `cwd`/`model_label`), so this is the structural fix at no
					    cost: the line always carries at least one blank, whose line
					    box is the height of a full line at whatever type scale the
					    reader has chosen. */}
					{cwdText === "" && s.model_label === "" ? "\u00a0" : null}
				</span>
				{/* D1 (mobile UX batch): the model label YIELDS like the cwd does.
				    `shrink-0` made it unable to shrink and unable to ellipsize, so a
				    long label ran to the viewport edge — no ellipsis, unreachable by
				    touch (a row drag is a tap), and `main` itself became horizontally
				    scrollable. `min-w-0 truncate` lets both spans share the row; the
				    row can never exceed it. */}
				<span className="ml-auto min-w-0 truncate font-mono text-mono-sm text-ink-dim">
					{s.model_label}
				</span>
			</div>
		</button>
	);
}

function ThemePicker({
	open,
	onClose,
}: {
	open: boolean;
	onClose: () => void;
}) {
	const [current, setCurrent] = useState(getTheme);
	return (
		<Sheet open={open} onClose={onClose} title="theme">
			<div className="flex flex-col p-2">
				{THEMES.map((t) => (
					<button
						key={t.id}
						type="button"
						onClick={() => {
							applyTheme(t.id);
							setCurrent(t.id);
						}}
						className="flex min-h-11 items-center gap-2 rounded-sm px-2 text-left active:bg-surface"
					>
						<span
							className={cn(
								"w-4 shrink-0 font-mono text-mono-sm",
								t.id === current ? "text-accent" : "text-ink-disabled",
							)}
							aria-hidden
						>
							{t.id === current ? "✓" : ""}
						</span>
						<span className="min-w-0 flex-1">
							<span className="block truncate text-body">
								{t.name}
							</span>
							<span className="block truncate text-meta text-ink-dim">
								{t.description}
							</span>
						</span>
					</button>
				))}
			</div>
		</Sheet>
	);
}

/* The daemon's own words for a refused pin come from `lib/pin-refusal` — shared
   with the session view so one refusal cannot grow two sentences (batch 2, U2). */

/* ------------------------------------------------------------------ */
/* The frame hand-off and the FLIP settle                              */
/* ------------------------------------------------------------------ */

/** A card's top in LAYOUT space: `offsetTop` summed up the offsetParent chain.

    WHY NOT `getBoundingClientRect`. A rect is where the card is PAINTED, so it
    includes the CSS transform the settle itself is running — on a commit
    inside the 180ms window, re-measuring the rect read the settle's own
    in-flight transform as movement (the mirrored jump this change removes) —
    and it moves with the reader's scroll, which must not read as a layout
    change either. Layout offsets cannot see either: measured on the jitter
    rig, a 173px scroll moved a card's rect (397.2 → 224.2) and left its
    offset chain at 397.0. */
function layoutTop(el: HTMLElement): number {
	let top = 0;
	let node: HTMLElement | null = el;
	while (node) {
		top += node.offsetTop;
		node = node.offsetParent as HTMLElement | null;
	}
	return top;
}

/** The card's CURRENT translateY in px, mid-transition value included (the
    computed matrix interpolates while a settle runs). 0 when nothing is set,
    and in non-DOM environments, where DOM geometry does not exist. */
function translateYOf(el: HTMLElement): number {
	if (typeof DOMMatrixReadOnly === "undefined") return 0;
	const transform = getComputedStyle(el).transform;
	if (!transform || transform === "none") return 0;
	try {
		return new DOMMatrixReadOnly(transform).f;
	} catch {
		return 0;
	}
}

/** The settle's own transition, in one place: the invert, a touch's resume and
    the shared duration/easing can never drift apart. */
const SETTLE_TRANSITION =
	"transform var(--transition-duration-base, 180ms) var(--ease-out-quart, ease-out)";

/** Plays a card from its current pose — the settle's inversion, or the pose a
    touch froze it at — back to its layout slot. The transition is stripped on
    `transitionend` so a later settle starts from a clean element, and the
    handler checks the TARGET because `transitionend` BUBBLES: the `new` word's
    opacity fade inside the card also ends, and acting on that event would
    strip the transform transition mid-settle. */
function releaseCard(el: HTMLButtonElement): void {
	el.style.transition = SETTLE_TRANSITION;
	el.style.transform = "";
	const done = (event: TransitionEvent) => {
		if (event.target !== el) return;
		el.style.transition = "";
		el.removeEventListener("transitionend", done);
	};
	el.addEventListener("transitionend", done);
}

/** Writes ONE settle on one card: invert the move with no transition, force
    the inverted position to commit as a style, then play it back to zero. */
function settleCard(el: HTMLButtonElement, dy: number): void {
	el.style.transition = "none";
	el.style.transform = `translateY(${dy}px)`;
	/* Force the inverted position to commit as a style before the
	   transition property returns, or the browser collapses both
	   writes and the card jumps straight to its new slot. */
	void el.offsetHeight;
	releaseCard(el);
}

/** The rows the list PAINTS: every store frame lands in a one-slot buffer,
    applied on the next animation frame — at most one application per frame —
    and NOT AT ALL while a pointer is down on the list.

    WHY BOTH PROPERTIES (both operator-reported, both measured on this
    change's own rig):

    * A frame applied between `pointerdown` and `pointerup` reorders the DOM
      under the finger; the row slides out, and the tap's synthesised click
      then resolves to whatever is underneath — the container — so the tap
      opens nothing. Measured on the rig: a reorder plus two frames landing
      mid-touch moved the tapped row 51px and `location.hash` stayed `#/`;
      with the buffer held it does not move during the touch and the tap
      navigates. The buffer is applied on the NEXT ANIMATION FRAME after the
      release, not inside the pointerup handler: a reorder applied in that
      handler re-renders before the browser dispatches the tap's click,
      which puts the moved target back in the click's path one event later.

    * The daemon pushes a list frame per projection update — ~24-30/s while a
      runtime streams (30.4/s measured over 60s on this rig) — each a new
      array, so applying them the instant they arrive re-rendered (and, before
      the settle's own fix, re-settled a moving row) several times between two
      painted frames. One application per animation frame coalesces a burst
      into one.

    A finger that lifts where the list never hears it must not freeze the
    list forever: the release listens on `window` for `pointerup` /
    `pointercancel`, and on `blur` / `visibilitychange` for a browser that
    takes the gesture away (a call, a tab switch) without a cancel. Pointer
    ids are tracked in a Set so the hold ends when the LAST finger lifts.

    THE HOLD COVERS MOTION, NOT JUST FRAMES (UX round 1, U32). The buffer
    stops moves a NEW frame would start; a settle already gliding when the
    finger lands is still a moving target under it — measured on the rig,
    presses landing while a multi-slot settle glided opened the WRONG session
    (the pressed row travelled 16.9-196.0px under a 120-300ms press, on every
    try; 0 of 10 wrong on a settled list), and the tap is right once the row
    cannot move at all. `onHoldChange` is the screen's cue: on the first
    finger down it pins every mid-settle card where it currently paints, and
    the release plays each its one remaining glide. The screen's settle
    effect pins too, for a settle that would otherwise START mid-hold — one
    invariant under both: a row cannot move under a finger, whatever started
    the move. */
function usePaintedRows(
	sessions: SessionSummary[],
	onHoldChange: (held: boolean) => void,
): {
	rows: SessionSummary[];
	onListPointerDown: (event: { pointerId: number }) => void;
} {
	const [rows, setRows] = useState(sessions);
	const latest = useRef(sessions);
	latest.current = sessions;
	const pointers = useRef(new Set<number>());
	const frame = useRef<number | null>(null);
	/* Read through a ref: the callback closes over element state by identity,
	   and the window listeners below are mounted once. */
	const holdChange = useRef(onHoldChange);
	holdChange.current = onHoldChange;

	/* One scheduled application at a time. A hold BLOCKS the apply instead of
	   rescheduling it: the release is what schedules, so everything that
	   arrived during a touch collapses to one application on the frame after
	   the lift. */
	const schedule = () => {
		if (pointers.current.size > 0 || frame.current !== null) return;
		frame.current = requestAnimationFrame(() => {
			frame.current = null;
			if (pointers.current.size > 0) return;
			setRows(latest.current);
		});
	};

	useEffect(() => {
		if (sessions !== rows) schedule();
	});

	useEffect(() => {
		const release = (event: PointerEvent) => {
			if (!pointers.current.delete(event.pointerId)) return;
			if (pointers.current.size === 0) {
				holdChange.current(false);
				schedule();
			}
		};
		const releaseAll = () => {
			if (pointers.current.size === 0) return;
			pointers.current.clear();
			holdChange.current(false);
			schedule();
		};
		window.addEventListener("pointerup", release);
		window.addEventListener("pointercancel", release);
		window.addEventListener("blur", releaseAll);
		document.addEventListener("visibilitychange", releaseAll);
		return () => {
			window.removeEventListener("pointerup", release);
			window.removeEventListener("pointercancel", release);
			window.removeEventListener("blur", releaseAll);
			document.removeEventListener("visibilitychange", releaseAll);
			if (frame.current !== null) cancelAnimationFrame(frame.current);
			frame.current = null;
			pointers.current.clear();
		};
	}, []);

	/* Stable across renders (round-1 review, N2): the handler reaches
	   everything through refs, so `<main>`'s prop identity never churns. */
	const onListPointerDown = useCallback((event: { pointerId: number }) => {
		/* The FIRST finger down pauses the list — frames and motion both;
		   later fingers join the same hold. */
		if (pointers.current.size === 0) holdChange.current(true);
		pointers.current.add(event.pointerId);
	}, []);

	return { rows, onListPointerDown };
}

export function SessionListScreen() {
	const { sessions, connected } = useSessions();
	/* FLIP settle state: per session id, the card ELEMENT measured last time and
	   its top in LAYOUT space (see `layoutTop`). The element is part of the
	   record because a pin LIFT remounts a card under another section — a new
	   element must appear in place, never glide across the screen. */
	const cardRefs = useRef(new Map<string, HTMLButtonElement>());
	const prevCards = useRef(
		new Map<string, { el: HTMLButtonElement; top: number }>(),
	);
	/* Cards a touch has PAUSED (id → the pose they were pinned at), so the
	   release plays exactly those the rest of the way. */
	const frozenSettles = useRef(new Map<string, number>());
	/* Whether a finger is down RIGHT NOW. The settle effect reads it: a move
	   that lands mid-hold is PINNED, never glided (see `freezeSettles`). */
	const holdActive = useRef(false);

	/* A FINGER PAUSES THE LIST'S MOTION, NOT JUST ITS FRAMES (UX round 1,
	   U32). The frame hand-off stops moves a new frame would start; a settle
	   already gliding when the finger lands is still a moving target under it —
	   measured on the rig, presses landing while a multi-slot settle glided
	   opened the WRONG session (the pressed row travelled 16.9-196.0px under a
	   120-300ms press, on every try; 0 of 10 wrong on a settled list). So the
	   first finger down pins every mid-settle card where it currently paints,
	   and the release gives each its one remaining glide. */
	const freezeSettles = () => {
		for (const [id, el] of cardRefs.current) {
			const ty = translateYOf(el);
			if (ty === 0) continue; // at rest: nothing to pause
			el.style.transition = "none";
			el.style.transform = `translateY(${ty}px)`;
			frozenSettles.current.set(id, ty);
		}
	};
	const resumeSettles = () => {
		if (frozenSettles.current.size === 0) return;
		const frozen = [...frozenSettles.current.keys()];
		frozenSettles.current.clear();
		for (const id of frozen) {
			const el = cardRefs.current.get(id);
			if (el?.isConnected) releaseCard(el);
		}
	};
	const handleHoldChange = useCallback((held: boolean) => {
		holdActive.current = held;
		if (held) freezeSettles();
		else resumeSettles();
	}, []);

	/* THE ROWS BELOW RENDER `rows`, NOT `sessions`: the painted list is the
	   hand-off's (see `usePaintedRows`), which is what keeps a touch's rows
	   stationary and a burst to one paint. Logic that ANSWERS a press (the pin
	   sheet's row, the refusal band) still reads live `sessions` — an answer
	   must not be a frame behind. */
	const { rows, onListPointerDown } = usePaintedRows(sessions, handleHoldChange);
	const pinMarks = usePinMarks();
	const [home, setHome] = useState("");
	const [themeOpen, setThemeOpen] = useState(false);
	/* The Projects sheet lives over THIS screen (the design's "reachable from the
	   sessions screen"), next to the other footer entries. Its state is local:
	   nothing on the list changes when the sheet opens, and the sheet re-reads
	   the store on every open, so a project another surface created meanwhile is
	   never missed. */
	const [projectsOpen, setProjectsOpen] = useState(false);
	const [query, setQuery] = useState("");
	/* The row whose pin action sheet is open, or NONE. Held as the id rather than
	   the summary so a list repaint while the sheet is open cannot leave the
	   sheet describing a stale object — the row it names is re-read from
	   `sessions` on every render, so the toggle always acts on current state. */
	const [pinTarget, setPinTarget] = useState<string | null>(null);
	/* THE REFUSAL TO RENDER INSIDE THAT SHEET, and the row it answers for.

	   Held as a pair rather than as one sentence because the answer can land
	   after the reader has moved on: a slow refusal for one row must never paint
	   inside another row's sheet, so the render gate compares ids as well as
	   clearing on open. */
	const [pinRefusal, setPinRefusal] = useState<{
		sessionId: string;
		reason: string;
	} | null>(null);
	/* THE ROWS WITH A PIN REQUEST STILL IN FLIGHT, ONE ENTRY PER REQUEST. The
	   sheet STAYS OPEN across that wait (design round 11, D25), so a row's action
	   has to be dead while ITS OWN request is outstanding. A single screen-wide
	   slot holding "the row last pressed" answered a different question — "was
	   this row the last press?" — so a press on one row left another row's
	   re-opened sheet live, offering a verb and a request that contradicted the
	   one already on its way (review round 12, MAJOR 1). */
	const [pinPending, setPinPending] = useState<ReadonlySet<string>>(() => new Set());
	const markPinPending = (sessionId: string, pending: boolean) =>
		setPinPending((current) => {
			const next = new Set(current);
			if (pending) next.add(sessionId);
			else next.delete(sessionId);
			return next;
		});
	/* WHAT THE SHEET'S LIVE REGION SAYS, held apart from the text it is derived
	   from so the region can be mounted EMPTY and filled on the commit after (see
	   the effect below). Opening the sheet clears it, so what is written is always
	   this sheet's own answer and never the last one's. */
	const [pinNotice, setPinNotice] = useState("");
	/* The pin action itself, so a wait that ends with the sheet still open can give
	   focus back to the control the reader pressed. */
	const pinActionRef = useRef<HTMLButtonElement>(null);
	const wasPinBusy = useRef(false);
	const visible = rows.filter((session) =>
		`${session.conversation_name} ${session.session_id} ${session.cwd}`
			.toLowerCase()
			.includes(query.toLowerCase()),
	);
	/* THE SIDEBAR'S SECTION ORDER, top to bottom: pinned, then the shared
	   active/previous partition, filtered through whatever query is typed. A
	   pinned row appears ONLY in ★ Pinned (not also in Active/Previous), matching
	   the sidebar, where a pin lifts the row out of the section it ranked into.
	   The RANKING is untouched — each group keeps the daemon's order, so pinning
	   never reorders a section and the FLIP settle below has nothing to animate
	   when a row merely joins the pinned list at the top.

	   THE PARTITION IS THE DAEMON'S, NEVER THE READER'S PRESS. `sessions`
	   carries confirmed pins only; a press the server has not answered for is a
	   MARK (`pinMarks`), which draws the ★ and nothing else. Splitting them is
	   the fix for the rows that moved under the reader: this partition is what
	   REORDERS the list, and the browser answers a reorder under the reader with
	   a scroll adjustment of its own — +88.0px on a successful pin, −51.0px at
	   100% / −101.5px at 200% on a refused one (QA Q13/D18). Rows now move once,
	   when the daemon confirms; a refusal reorders nothing because nothing ever
	   moved. */
	const pinned = visible.filter((session) => session.pinned);
	const rest = visible.filter((session) => !session.pinned);
	const active = rest.filter((session) => session.section === "active");
	const previous = rest.filter((session) => session.section === "previous");
	const pinRow = pinTarget
		? sessions.find((session) => session.session_id === pinTarget) ?? null
		: null;
	/* Busy is about THIS row's action, not about any request: a press on one row
	   never disables another row's sheet. */
	const pinBusy = pinRow !== null && pinPending.has(pinRow.session_id);
	const refusalFor = pinRefusal && pinRow && pinRefusal.sessionId === pinRow.session_id
		? pinRefusal.reason
		: null;
	/* THE DAEMON'S FLAG, WHICH IS WHAT THE ACTION TALKS ABOUT. Not `renderedPin`:
	   the ★ records what the reader asked for, and an unanswered request is not a
	   state — a sheet opened while this row's request is in flight must offer
	   neither a verb nor an intent derived from it (review round 12, MAJOR 1).
	   While that request IS in flight the verb claims nothing at all and simply
	   says what is happening, so no label can precede the daemon's agreement. */
	const pinConfirmed = Boolean(pinRow?.pinned);
	const pinActionLabel = pinBusy
		? "Saving…"
		: pinConfirmed
			? "Unpin from the top"
			: "Pin to the top";
	const pinRefusalText = refusalFor
		? `Could not save the pin: ${clampPinReason(refusalFor)}`
		: null;
	const pinNoticeText = pinRefusalText ?? (pinBusy ? "Saving…" : "");

	/* The pin to RENDER for a row: the reader's unanswered mark over the daemon's
	   confirmed flag. This drives the ★ ON THE CARD AND NOTHING ELSE — not the
	   sheet's verb and not what a press sends. Those read `session.pinned`
	   (`pinConfirmed` above), because a mark is the evidence of a request and not
	   an answer: a sheet re-opened mid-flight would otherwise claim the state the
	   reader is still waiting to hear about, and offer to invert it. */
	const renderedPin = (session: SessionSummary) =>
		pinMarks.get(session.session_id) ?? Boolean(session.pinned);

	/* The ★ shows what the reader asked for; only the daemon may move the row.
	   A failed POST clears the mark, and the ★ falls back with it.

	   THE SHEET STAYS OPEN UNTIL THE ANSWER LANDS (design round 11, D25), and
	   that is the whole point of this shape. Closing on the press left a refused
	   pin with no reason anywhere: the ★ appeared at +2.2…5.0ms and cleared at
	   +5.3…20.5ms — an existence window of ~3ms, a fifth of a 60Hz frame, so the
	   reader often saw nothing at all, and on a slow answer saw a
	   confirmation-shaped mark stay up for the whole wait (306ms, 1516ms, 46.9s
	   measured) and then be withdrawn with no explanation. The refusal reason is
	   rendered IN FLOW inside this sheet — never as an overlay: the reason the
	   band was split out of this change is that anything floating above the list
	   covers the search field. */
	const togglePin = async (sessionId: string, pinnedNext: boolean) => {
		applySessionPin(sessionId, pinnedNext);
		/* A retry in the same sheet starts from no answer, not the last one. */
		setPinRefusal(null);
		markPinPending(sessionId, true);
		try {
			const saved = await setSessionPin(sessionId, pinnedNext);
			/* The route answers with the state it READ BACK, which is the daemon's
			   answer and not ours. A 200 that disagrees means the row was not pinned
			   (a folder-less session answers 409, but a 200 reporting the old value
			   is the same fact), and the mark has to go now: `settlePinMarks` only
			   retires a mark a later frame AGREES with, so this one would never
			   settle and the ★ would sit on a row the daemon never pinned. */
			if (saved.pinned !== pinnedNext) clearSessionPinMark(sessionId);
			/* Only a sheet still showing THIS row closes: the answer can land after
			   the reader dismissed it and opened another row's, and closing that one
			   would answer a press nobody made. */
			setPinTarget((current) => (current === sessionId ? null : current));
		} catch (error) {
			/* A refusal takes the MARK back with it: the row never moved, because
			   only a confirmed list frame reorders this screen. The reason the
			   daemon gave is kept for the sheet to say, in the sheet's own layout,
			   so the press that failed is the press that explains itself. */
			clearSessionPinMark(sessionId);
			setPinRefusal({
				sessionId,
				reason: pinRefusalReason(error),
			});
		} finally {
			markPinPending(sessionId, false);
		}
	};

	useEffect(() => retainSessionListStream(), []);

	/* THE LIVE REGION IS FILLED AFTER THE SHEET HAS MOUNTED, never with it: WebKit
	   AT takes a region's mount as its baseline, so text that arrives inside an
	   already-populated container is not announced — and a re-opened sheet arrives
	   exactly that way, its wait already true. The sheet clears the text as it
	   opens, so the fill is always this sheet's own answer. */
	useEffect(() => {
		setPinNotice(pinNoticeText);
	}, [pinNoticeText]);

	/* FOCUS COMES BACK TO THE ACTION WHEN THE WAIT ENDS. `disabled` blurs a
	   control in a real browser, which drops the reader out of the sheet's column
	   mid-wait and leaves the Sheet's trap with nothing to hold but the first and
	   last element; a refusal is the case the sheet now stays open for, so the
	   action is where the reader was and takes focus back. Only a busy→idle
	   transition, so opening a sheet never steals focus from its own controls. */
	useEffect(() => {
		if (wasPinBusy.current && !pinBusy) pinActionRef.current?.focus();
		wasPinBusy.current = pinBusy;
	}, [pinBusy]);

	/* One predicate for the pin hint, used by BOTH the height class and
	   ``aria-hidden`` — two spellings of one condition is how a control ends up
	   painted one way and read out another (review round 3, NIT 1). Gated on what
	   is VISIBLE (D5) and on the CONFIRMED pins the PAINTED rows carry (D6;
	   round-1 review N1: the source is the hand-off's `rows`, up to a frame
	   behind the store — the same flag, a different source than the comment here
	   once claimed): the caption names a row the reader can see, and a search
	   that merely hides the pinned rows must not bring it back. It is CONFIRMED
	   pins only, so a press the daemon has not answered for leaves the caption
	   up: the ★ Pinned section it points at does not exist until the pin is
	   confirmed, and retiring the caption for a mark alone would take away the
	   only thing explaining the gesture, with nothing to replace it. */
	const showPinHint = visible.length > 0 && !rows.some((session) => session.pinned);

	/* One card factory for all three sections, so a section cannot forget the FLIP
	   ref or the long-press handler — the bug a fourth copy of this markup would
	   eventually grow. */
	const renderCard = (s: SessionSummary) => (
		<SessionCard
			key={s.session_id}
			s={s}
			home={home}
			pinned={renderedPin(s)}
			onLongPress={() => {
				setPinTarget(s.session_id);
				/* Opening a sheet clears the last refusal AND its text: they belonged
				   to the press before, and a stale reason under a fresh action is a
				   lie — the region starts empty whatever it will say next. */
				setPinRefusal(null);
				setPinNotice("");
			}}
			ref={(el) => {
				if (el) cardRefs.current.set(s.session_id, el);
				else cardRefs.current.delete(s.session_id);
			}}
		/>
	);

	/* FLIP settle for reorders (spec §3): a card never teleports under a
	   thumb mid-scroll. After each commit, measure every card's top in LAYOUT
	   space (`layoutTop` — where neither the settle's own in-flight transform
	   nor the reader's scroll can be seen), and where the LAYOUT moved, invert
	   the move with no transition, force a style flush, then play it back to
	   zero (`settleCard`).

	   A commit that moved nothing leaves any in-flight settle ALONE — that
	   guard is this change's core fix. The old code re-measured each card with
	   `getBoundingClientRect` (which INCLUDES the settle's own transform), so a
	   commit inside the 180ms window read the mid-flight offset as movement
	   and wrote it back as a new settle, mirroring the card across its slot
	   (sign-alternating per commit — at the measured ~24-30Hz frame cadence a
	   settle's mirrored writes amplified to millions of pixels within
	   seconds).

	   A card whose layout moves again WHILE settling continues from where it
	   currently paints — its previous layout top plus whatever transform is
	   still in flight — so one real move reads as one settle, never a restart
	   from stale coordinates.

	   New cards — and a card that REMOUNTS because a pin lifted it into
	   another section (React mounts a new element under the new heading) —
	   appear in place, exactly where the reader last saw them. Scroll offset is
	   untouched: only transforms animate, and scroll anchoring stays on.
	   Implemented with `transition`, never `animation`, so the global
	   prefers-reduced-motion block caps the settle to instant for free. Runs
	   synchronously before paint (useLayoutEffect) so the inverted frame is
	   what the user would have seen anyway — the pre-reorder layout. */
	useLayoutEffect(() => {
		const nextCards = new Map<string, { el: HTMLButtonElement; top: number }>();
		for (const [id, el] of cardRefs.current) {
			const top = layoutTop(el);
			nextCards.set(id, { el, top });
			const prev = prevCards.current.get(id);
			/* No previous measurement (a new card), or a NEW ELEMENT under the
			   same id (a section move): appear in place. */
			if (!prev || prev.el !== el) continue;
			/* This commit moved nothing for this card: leave any in-flight
			   settle alone. */
			if (prev.top === top) continue;
			const dy = prev.top + translateYOf(el) - top;
			if (dy === 0) continue;
			/* A finger is down right now: PIN this move where it is instead of
			   gliding it — the row stays put for the whole hold and the release
			   plays it home. This is `freezeSettles`' contract for a settle
			   that would otherwise START mid-touch (the frame that caused this
			   move arrived before the press, but its commit landed after it). */
			if (holdActive.current) {
				el.style.transition = "none";
				el.style.transform = `translateY(${dy}px)`;
				frozenSettles.current.set(id, dy);
				continue;
			}
			settleCard(el, dy);
		}
		prevCards.current = nextCards;
	});
	useEffect(() => {
		getDirectories()
			.then((d) => setHome(d.home))
			.catch(() => {
				/* Home is cosmetic (path shortening); the list works without it. */
			});
	}, []);

	return (
		<div className="relative mx-auto flex h-dvh w-full max-w-[var(--lo-column-max,28rem)] flex-col">
			<header className="flex items-center gap-2 px-3 pt-[max(env(safe-area-inset-top),0.75rem)] pb-2">
				<img
					src={MARK_DATA_URI}
					alt=""
					width={20}
					height={20}
				/>
				<h1 className="text-meta font-medium tracking-[0.18em] text-ink">
					local operator
				</h1>
				{/* OFFLINE, QUIETLY (mobile UX batch 2, U11). `connected === false` on
				    this screen used to change nothing once rows were on it: the list
				    kept painting a frozen frame as though it were live. The chip is
				    gated on holding data — an empty list already says `connecting…`
				    in its placeholder — and it clears itself the moment a frame
				    lands, because `connected` flips back on the SSE's own open. */}
				{sessions.length > 0 && !connected ? (
					<span role="status" className="ml-auto shrink-0 text-meta text-ink-dim">
						reconnecting…
					</span>
				) : null}
			</header>
			{/* BROWSER SCROLL ANCHORING STAYS ON (QA round 4, Q6). An earlier round
			    opted the list out (`overflow-anchor: none`), and with the opt-out every
			    unrelated list change moved a scrolled reader's rows: a new live session
			    +76px, a session resuming +51px, another client's pin below the fold
			    +45.5px at 100% and +91px at 200%, all 0px with anchoring on.

			    The wrapper carries the column's `min-h-0` and keeps the scroller's
			    height off the header and footer, so `<main>` inside it stays the thing
			    that scrolls. It is no longer `relative`: the only absolutely positioned
			    child it had was the refusal band, which moved to its own PR. */}
			<div className="flex min-h-0 flex-1 flex-col">
				<main
					onPointerDown={onListPointerDown}
					className="flex flex-1 flex-col overflow-y-auto px-1 pb-2"
				>
					<input
						value={query}
						onChange={(event) => setQuery(event.target.value)}
						placeholder="Search conversations…"
						className="mx-2 mb-2 min-h-11 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim"
					/>
					{/* THE GESTURE'S DISCOVERER, on the surface that owns the gesture (design
					    round 1, D2). The session view's ☆ is one tap away and does the same
					    thing, but a reader has to already be in a conversation to find it, so
					    it cannot teach the list's own long-press.

					    GATED ON WHAT IS ON SCREEN, not on the store — the whole point of the
					    caption is that it names a row the reader can see. Keying on
					    ``sessions`` (unfiltered) put "touch and hold a row to pin it" directly
					    above "no matching conversations" for a query that matched nothing, and
					    brought it back whenever a search hid the pinned rows (design round 2,
					    D5/D6). ``visible.length > 0`` is the honest condition; the pinned test
					    reads the same painted `rows` as the section split above (round-1
					    review N1), so a search that hides a pin does not re-show the hint.

					    IT COLLAPSES RATHER THAN VANISHES, which is D8: removing the node
					    outright snapped the whole list up ~23px at the exact moment the first
					    pin landed. The wrapper is always mounted and animates its height to
					    zero, so the list settles instead of jumping. `prefers-reduced-motion`
					    caps it to instant for free (the global block), which is the right
					    fallback — the point is not the motion, it is that nothing snaps.

					    THE COLLAPSE IS CONTENT-AGNOSTIC, and that is not incidental. An
					    earlier version capped `max-height` at a fixed 2rem — sized for ONE
					    line of this caption at the default type scale. A caption that WRAPS
					    to two lines (a longer localized string, a narrower container) is then
					    taller than the cap, and `overflow-hidden` clips the second line away:
					    the discoverer silently disappears for exactly the readers who need
					    the label most. (The cap itself scales with the root font, so a
					    large-text one-liner still fits — the failure is the WRAP, not the
					    zoom.) The `0fr`/`1fr` grid trick measures the content itself, so the
					    caption is fully painted at any string length and collapses to zero
					    with no magic number to keep in sync with the type scale. */}
					<div
						className={cn(
							"grid transition-[grid-template-rows] duration-200 ease-out",
							showPinHint ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
						)}
						aria-hidden={showPinHint ? undefined : true}
					>
						<div className="overflow-hidden">
							<p className="mx-2 mb-2 text-meta text-ink-dim">
								touch and hold a row to pin it
							</p>
						</div>
					</div>
					{rows.length === 0 ? (
						<div className="flex flex-1 flex-col items-center justify-center gap-2 px-6 text-center">
							<p className="text-body text-ink-muted">
								{connected
									? "no sessions running"
									: "connecting…"}
							</p>
							<p className="text-body-sm text-ink-dim">
								start one below, or from the TUI on your machine
							</p>
						</div>
					) : (
						<div className="flex flex-col gap-3">
							{/* AN EMPTY SECTION COSTS NO HEADING — the sidebar's own rule
							    (`_display_rows`: "An empty section contributes no header"), and the
							    ★ Pinned section above already followed it while these two did not.
							    Newly reachable because of pinning: a pin LIFTS a row out of its
							    ranked section, so pinning every row (or a search that matches none)
							    painted two bare headings with nothing under them (design round 1,
							    D1). */}
							{pinned.length > 0 ? (
								<section>
									<h2 className="px-2 py-1 text-meta font-medium text-ink-muted">★ Pinned</h2>
									{pinned.map(renderCard)}
								</section>
							) : null}
							{active.length > 0 ? (
								<section>
									<h2 className="px-2 py-1 text-meta font-medium text-ink-muted">Active Sessions</h2>
									{active.map(renderCard)}
								</section>
							) : null}
							{previous.length > 0 ? (
								<section>
									<h2 className="px-2 py-1 text-meta font-medium text-ink-muted">Previous Sessions</h2>
									{previous.map(renderCard)}
								</section>
							) : null}
							{/* The case where EVERY section is empty: a query that matched none, or
							    every row pinned away — pinned rows are not empty, so this arm is the
							    one where the screen would otherwise be a wordless void. */}
							{pinned.length + active.length + previous.length === 0 ? (
								<p className="px-2 py-4 text-center text-body-sm text-ink-dim">
									no matching conversations
								</p>
							) : null}
						</div>
					)}
				</main>
			</div>
			<footer className="flex items-center gap-2 border-t border-hairline px-3 py-2 pb-[max(env(safe-area-inset-bottom),0.5rem)]">
				<button
					type="button"
					onClick={() => navigate("/new")}
					className="flex min-h-11 flex-1 items-center justify-center rounded-md border border-control bg-surface text-body-sm font-medium text-ink select-none active:bg-elevated"
				>
					new session
				</button>
				{/* U10 (mobile UX batch): the entry point the file's own contract
				    describes. `#/past` — searchable, resumable history — was reachable
				    only by knowing its hash; nothing in the app linked to it, so on a
				    phone the feature was dead. The footer is where the other
				    top-level entries live. */}
				<button
					type="button"
					onClick={() => navigate("/past")}
					className="flex min-h-11 items-center justify-center rounded-md border border-control bg-surface px-3 text-body-sm text-ink-muted select-none active:bg-elevated"
				>
					past
				</button>
				<button
					type="button"
					onClick={() => setProjectsOpen(true)}
					className="flex min-h-11 items-center justify-center rounded-md border border-control bg-surface px-3 text-body-sm text-ink-muted select-none active:bg-elevated"
				>
					projects
				</button>
				<button
					type="button"
					onClick={() => setThemeOpen(true)}
					aria-label="choose theme"
					className="flex min-h-11 min-w-11 items-center justify-center rounded-md border border-control bg-surface text-ink-muted select-none active:bg-elevated"
				>
					◐
				</button>
				{/* WIDE VIEW (issue #1870): the same control the session screen's
				    header carries — one component, so the label and the pressed state
				    cannot drift between the two places a reader can reach it. Lives
				    here beside the theme because it is the same kind of preference
				    (per-phone, persisted, not content). */}
				<WideViewButton />
			</footer>
			<ThemePicker open={themeOpen} onClose={() => setThemeOpen(false)} />
			<ProjectsSheet open={projectsOpen} onClose={() => setProjectsOpen(false)} />
			{/* THE PIN ACTION SHEET. Long-press opened it, so it is where the gesture's
			    meaning is spelled out rather than left to be discovered — the row shows
			    a ★ once pinned, and this sheet is how a reader learns the gesture that
			    put it there. ONE primary action and nothing else: a menu of one is a
			    better fit for a phone than an inline control on every row, which would
			    cost the title the width it truncates against. */}
			<Sheet
				open={pinRow !== null}
				onClose={() => setPinTarget(null)}
				title={pinRow?.conversation_name || "untitled"}
			>
				<div className="flex flex-col p-2">
					<button
						ref={pinActionRef}
						type="button"
						disabled={pinBusy}
						/* `disabled` is silent on its own: it says the control is dead and
						   nothing about the wait that killed it, which no assistive tech can
						   see either. */
						aria-busy={pinBusy ? true : undefined}
						onClick={() =>
							pinRow && void togglePin(pinRow.session_id, !pinConfirmed)
						}
						className="flex min-h-11 items-center gap-2 rounded-sm px-2 text-left text-body active:bg-surface disabled:opacity-50"
					>
						<span className="w-4 shrink-0 text-accent" aria-hidden>
							★
						</span>
						{/* The verb names the state the press will SET, and it is read from the
						    DAEMON's flag rather than from the ★ on the row: the ★ records a
						    request the daemon has not answered for, so a sheet opened while
						    that request is in flight would otherwise claim the opposite state
						    and offer to invert it (review round 12, MAJOR 1). While a request
						    for THIS row is outstanding the verb claims nothing at all. */}
						{pinActionLabel}
					</button>
					{/* THE REFUSAL, IN FLOW INSIDE THE SHEET. A paragraph in the sheet's own
					    column, not an overlay: it takes layout space, so it cannot cover
					    another control, and nothing on the list scrolls or reorders to
					    make room for it. `role="alert"` is what announces it — inside a
					    dialog that is in flow, which is fine, unlike an alert floating over
					    the list behind it. The wording is the app's own (`Could not save the
					    pin: <daemon's reason>`), the same shape the withdrawn band used. */}
					{/* THE SHEET'S ONE LIVE REGION, MOUNTED EMPTY AND FILLED AFTER. It is a
					    sibling of the action in the sheet's own column, so it takes layout
					    space and cannot be drawn over the control above it; anything
					    floating (an overlay above the list, a positioned box) fails that,
					    and anything floating above the list covers the search field. The
					    text lands on the commit AFTER the sheet mounts (the `pinNotice`
					    effect), because a container that arrives already populated is not
					    announced — the shape a re-opened sheet has, its wait already true.
					    `break-words` and the clamp in `clampPinReason` keep a long daemon
					    message inside the column. */}
					<p
						role="alert"
						className={cn(
							"px-2 pb-1 text-meta break-words",
							pinRefusalText ? "text-danger" : "text-ink-muted",
						)}
					>
						{pinNotice}
					</p>
				</div>
			</Sheet>
		</div>
	);
}

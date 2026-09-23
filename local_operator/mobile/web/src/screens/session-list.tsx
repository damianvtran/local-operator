/**
 * Session list (`#/`) — the phone's home. One card per live session, kept
 * current by the list SSE; footer row with new session, past sessions, and
 * the theme picker.
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
	useEffect,
	useLayoutEffect,
	useRef,
	useState,
	type Ref,
} from "react";
import { getDirectories, setSessionPin } from "../api";
import { Sheet } from "../components/ui/sheet";
import { Spinner } from "../components/spinner";
import { navigate } from "../router";
import {
	applySessionPin,
	retainSessionListStream,
	useSessions,
} from "../store";
import { applyTheme, getTheme, THEMES } from "../theme";
import { shortenHome } from "../lib/format";
import { MARK_DATA_URI } from "../lib/mark";
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
	onLongPress,
	ref,
}: {
	s: SessionSummary;
	home: string;
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
	const pendingLabel =
		s.pending_kind === "approval"
			? "approval"
			: s.pending_kind === "ask"
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
			className="flex w-full flex-col gap-0.5 rounded-md px-2 py-1.5 text-left select-none active:bg-elevated"
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
				{/* A PINNED ROW CARRIES ITS ★, and it rides the RIGHT cluster rather than
				    the state slot. The TUI made exactly this call (`session_sidebar.py`
				    `_special_mark`): a ★ in the state column made the sessions a user
				    cares about most the only ones that could not report being blocked,
				    broken or finished, because the pin is the DURABLE fact and the state
				    glyph is the volatile one — so the pin moves, not the state. The ★ is
				    also the shape the ★ Pinned heading uses, so the mark and its section
				    cannot disagree about what it means. */}
				{s.pinned ? (
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
			</div>
			<div className="flex items-baseline gap-2">
				<span className="min-w-0 truncate font-mono text-mono-sm text-ink-dim">
					{home ? shortenHome(s.cwd, home) : s.cwd}
				</span>
				<span className="ml-auto shrink-0 font-mono text-mono-sm text-ink-dim">
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
						className="flex min-h-8 items-center gap-2 rounded-sm px-2 text-left active:bg-surface"
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

export function SessionListScreen() {
	const { sessions, connected } = useSessions();
	const [home, setHome] = useState("");
	const [themeOpen, setThemeOpen] = useState(false);
	const [query, setQuery] = useState("");
	/* The row whose pin action sheet is open, or NONE. Held as the id rather than
	   the summary so a list repaint while the sheet is open cannot leave the
	   sheet describing a stale object — the row it names is re-read from
	   `sessions` on every render, so the toggle always acts on current state. */
	const [pinTarget, setPinTarget] = useState<string | null>(null);
	const [pinError, setPinError] = useState("");
	/* FLIP settle state: card DOM by session id, plus each card's content
	   coordinate from the previous commit. */
	const mainRef = useRef<HTMLElement>(null);
	const cardRefs = useRef(new Map<string, HTMLButtonElement>());
	const prevTops = useRef(new Map<string, number>());
	const visible = sessions.filter((session) =>
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
	   when a row merely joins the pinned list at the top. */
	const pinned = visible.filter((session) => session.pinned);
	const rest = visible.filter((session) => !session.pinned);
	const active = rest.filter((session) => session.section === "active");
	const previous = rest.filter((session) => session.section === "previous");
	const pinRow = pinTarget
		? sessions.find((session) => session.session_id === pinTarget) ?? null
		: null;

	/* Optimistic, then confirmed: the row moves the instant the user acts, and
	   the daemon's next list repaint (which `set_pins` already woke) is the
	   authority. A failed POST restores the truth via that same repaint. */
	const togglePin = async (sessionId: string, pinned: boolean) => {
		setPinError("");
		applySessionPin(sessionId, pinned);
		setPinTarget(null);
		try {
			await setSessionPin(sessionId, pinned);
		} catch (e) {
			/* Surface it rather than swallow: an unreachable daemon must not read as
			   "the pin took". The next repaint corrects the row either way. */
			setPinError(String((e as Error).message ?? e));
		}
	};

	useEffect(() => retainSessionListStream(), []);

	/* One predicate for the pin hint, used by BOTH the height class and
	   ``aria-hidden`` — two spellings of one condition is how a control ends up
	   painted one way and read out another (review round 3, NIT 1). Gated on what
	   is VISIBLE (D5) and on the STORE's pins (D6): the caption names a row the
	   reader can see, and a search that merely hides the pinned rows must not
	   bring it back. */
	const showPinHint = visible.length > 0 && !sessions.some((session) => session.pinned);

	/* One card factory for all three sections, so a section cannot forget the FLIP
	   ref or the long-press handler — the bug a fourth copy of this markup would
	   eventually grow. */
	const renderCard = (s: SessionSummary) => (
		<SessionCard
			key={s.session_id}
			s={s}
			home={home}
			onLongPress={() => setPinTarget(s.session_id)}
			ref={(el) => {
				if (el) cardRefs.current.set(s.session_id, el);
				else cardRefs.current.delete(s.session_id);
			}}
		/>
	);

	/* FLIP settle for reorders (spec §3): a card never teleports under a
	   thumb mid-scroll. After each commit, measure every card's position in
	   the scroll content (`rect.top - main.rect.top + scrollTop`, so a user
	   scroll between commits never reads as movement), and where a card moved,
	   apply the inverse translateY with no transition, force a style flush,
	   then play it back to zero with a transform transition. Scroll offset is
	   untouched — only transforms animate. Implemented with `transition`,
	   never `animation`, so the global prefers-reduced-motion block caps the
	   settle to instant for free. Runs synchronously before paint
	   (useLayoutEffect) so the inverted frame is what the user would have
	   seen anyway — the pre-reorder layout. */
	useLayoutEffect(() => {
		const main = mainRef.current;
		const origin = main
			? main.getBoundingClientRect().top - main.scrollTop
			: 0;
		const nextTops = new Map<string, number>();
		for (const [id, el] of cardRefs.current) {
			nextTops.set(id, el.getBoundingClientRect().top - origin);
		}
		for (const [id, el] of cardRefs.current) {
			const prev = prevTops.current.get(id);
			const next = nextTops.get(id);
			/* New cards have no old position and simply appear in place. */
			if (prev === undefined || next === undefined) continue;
			const dy = prev - next;
			if (dy === 0) continue;
			el.style.transition = "none";
			el.style.transform = `translateY(${dy}px)`;
			/* Force the inverted position to commit as a style before the
			   transition property returns, or the browser collapses both
			   writes and the card jumps straight to its new slot. */
			void el.offsetHeight;
			el.style.transition =
				"transform var(--transition-duration-base, 180ms) var(--ease-out-quart, ease-out)";
			el.style.transform = "";
			const done = (event: TransitionEvent) => {
				/* transitionend BUBBLES: the `new` word's opacity fade inside the
				   card also ends, and acting on that event would strip the
				   transform transition mid-settle. */
				if (event.target !== el) return;
				el.style.transition = "";
				el.removeEventListener("transitionend", done);
			};
			el.addEventListener("transitionend", done);
		}
		prevTops.current = nextTops;
	});
	useEffect(() => {
		getDirectories()
			.then((d) => setHome(d.home))
			.catch(() => {
				/* Home is cosmetic (path shortening); the list works without it. */
			});
	}, []);

	return (
		<div className="relative mx-auto flex h-dvh w-full max-w-md flex-col">
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
			</header>
			<main
				ref={mainRef}
				className="flex flex-1 flex-col overflow-y-auto px-1 pb-2"
			>
				<input
					value={query}
					onChange={(event) => setQuery(event.target.value)}
					placeholder="Search conversations…"
					className="mx-2 mb-2 min-h-10 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim"
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
				    reads the STORE so a search that hides a pin does not re-show the hint.

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
				{sessions.length === 0 ? (
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
						{pinError ? (
							<p role="alert" className="px-2 text-meta text-danger">
								Could not save the pin: {pinError}
							</p>
						) : null}
					</div>
				)}
			</main>
			<footer className="flex items-center gap-2 border-t border-hairline px-3 py-2 pb-[max(env(safe-area-inset-bottom),0.5rem)]">
				<button
					type="button"
					onClick={() => navigate("/new")}
					className="flex min-h-11 flex-1 items-center justify-center rounded-md border border-control bg-surface text-body-sm font-medium text-ink select-none active:bg-elevated"
				>
					new session
				</button>
				<button
					type="button"
					onClick={() => setThemeOpen(true)}
					aria-label="choose theme"
					className="flex min-h-11 min-w-11 items-center justify-center rounded-md border border-control bg-surface text-ink-muted select-none active:bg-elevated"
				>
					◐
				</button>
			</footer>
			<ThemePicker open={themeOpen} onClose={() => setThemeOpen(false)} />
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
						type="button"
						onClick={() =>
							pinRow && void togglePin(pinRow.session_id, !pinRow.pinned)
						}
						className="flex min-h-11 items-center gap-2 rounded-sm px-2 text-left text-body active:bg-surface"
					>
						<span className="w-4 shrink-0 text-accent" aria-hidden>
							★
						</span>
						{/* The verb names the state it will SET, so a second look at the same
						    row reads the outcome rather than a description of the store. */}
						{pinRow?.pinned ? "Unpin from the top" : "Pin to the top"}
					</button>
				</div>
			</Sheet>
		</div>
	);
}

/**
 * Session list (`#/`) — the phone's home. One card per live session, kept
 * current by the list SSE; footer row with new session, past sessions, and
 * the theme picker.
 *
 * Visual contract: a streaming session shimmers its name (the row itself is
 * the indicator — no spinner); a session waiting on the user carries the
 * danger dot and a word ("approval" / "question"), because that is the one
 * card that needs a decision (branding §7).
 */
import {
	useEffect,
	useLayoutEffect,
	useRef,
	useState,
	type Ref,
} from "react";
import { getDirectories } from "../api";
import { Sheet } from "../components/ui/sheet";
import { Spinner } from "../components/spinner";
import { navigate } from "../router";
import { retainSessionListStream, useSessions } from "../store";
import { applyTheme, getTheme, THEMES } from "../theme";
import { shortenHome } from "../lib/format";
import { MARK_DATA_URI } from "../lib/mark";
import type { SessionSummary } from "../types";
import { cn } from "../lib/cn";

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
	ref,
}: {
	s: SessionSummary;
	home: string;
	/* FLIP anchor: the list measures every card before/after a reorder so it
	   can settle it into its new slot instead of teleporting it. */
	ref?: Ref<HTMLButtonElement>;
}) {
	const pendingLabel =
		s.pending_kind === "approval"
			? "approval"
			: s.pending_kind === "ask"
				? "question"
				: null;
	/* Render ladder: NEEDS DECISION > WORKING > NEW/UNREAD > IDLE — the same
	   order as the daemon's sort, so what is loudest is also what is
	   highest. Flags coexist in data; exactly one state renders. A session
	   blocked on a decision must be opened anyway, so the unread mark would
	   add noise; a streaming session is drawing the eye already, and "new"
	   marks COMPLETED unviewed activity, never in-flight work. */
	const decision = Boolean(s.needs_attention && pendingLabel);
	const unread = Boolean(s.unseen) && !decision && !s.streaming;
	return (
		<button
			ref={ref}
			type="button"
			onClick={() => navigate(`/s/${encodeURIComponent(s.session_id)}`)}
			className="flex w-full flex-col gap-0.5 rounded-md px-2 py-1.5 text-left select-none active:bg-elevated"
		>
			<div className="flex items-center gap-2">
				{/* ONE reserved indicator slot serving all four states, so every
				    title starts at the same x forever — indicators change colour,
				    never geometry.

				    The slot is sized to the LARGEST occupant (the 12px spinner)
				    and the 6px dot is centred inside it. Rendering the spinner
				    BESIDE the slot, as this did before, defeated the whole point:
				    a streaming row paid slot + gap + spinner and its title sat
				    ~19.5px right of every other row's, so the reserved-slot
				    promise held for three states and broke on the fourth — the
				    one that changes most often. Spinner itself is untouched; it
				    is a shared component and its own size is correct. */}
				<span
					className="flex size-3 shrink-0 items-center justify-center"
					aria-hidden={s.streaming ? undefined : true}
				>
					{s.streaming ? (
						/* The obvious in-progress mark beside the title: a small
						   loading wheel, not just the text sweep — the sweep alone
						   was too subtle to catch at a glance. */
						<Spinner />
					) : (
						<span
							className={cn(
								"inline-block size-1.5 rounded-full",
								decision
									? "lo-pulse bg-danger"
									: unread
										? "bg-accent"
										: "bg-transparent",
							)}
						/>
					)}
				</span>
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
				{/* State word rides BEFORE the count chips in the right cluster
				    (spec §1): `new` truncates the title only, row height never
				    changes. */}
				<NewMark visible={unread} />
				{s.subagents_running > 0 ? (
					/* ⟳ and ☐ render as tofu boxes on phones whose system font lacks
					   those codepoints. Text marks survive every font. */
					<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
						{s.subagents_running} agent{s.subagents_running === 1 ? "" : "s"}
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
	const active = visible.filter((session) => session.section === "active");
	const previous = visible.filter((session) => session.section === "previous");

	useEffect(() => retainSessionListStream(), []);

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
						<section>
							<h2 className="px-2 py-1 text-meta font-medium text-ink-muted">Active Sessions</h2>
							{active.map((s) => (
								<SessionCard
									key={s.session_id}
									s={s}
									home={home}
									ref={(el) => {
										if (el) cardRefs.current.set(s.session_id, el);
										else cardRefs.current.delete(s.session_id);
									}}
								/>
							))}
						</section>
						<section>
							<h2 className="px-2 py-1 text-meta font-medium text-ink-muted">Previous Sessions</h2>
							{previous.map((s) => (
								<SessionCard
									key={s.session_id}
									s={s}
									home={home}
									ref={(el) => {
										if (el) cardRefs.current.set(s.session_id, el);
										else cardRefs.current.delete(s.session_id);
									}}
								/>
							))}
						</section>
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
		</div>
	);
}

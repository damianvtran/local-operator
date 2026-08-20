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
import { useEffect, useState } from "react";
import { getDirectories } from "../api";
import { Sheet } from "../components/ui/sheet";
import { navigate } from "../router";
import { retainSessionListStream, useSessions } from "../store";
import { applyTheme, getTheme, THEMES } from "../theme";
import { shortenHome } from "../lib/format";
import { MARK_DATA_URI } from "../lib/mark";
import type { SessionSummary } from "../types";
import { cn } from "../lib/cn";

function SessionCard({ s, home }: { s: SessionSummary; home: string }) {
	const pendingLabel =
		s.pending_kind === "approval"
			? "approval"
			: s.pending_kind === "ask"
				? "question"
				: null;
	return (
		<button
			type="button"
			onClick={() => navigate(`/s/${s.pid}`)}
			className={cn(
				"flex w-full flex-col gap-0.5 rounded-md px-2 py-1.5 text-left select-none active:bg-elevated",
				s.ended && "text-ink-disabled",
			)}
		>
			<div className="flex items-center gap-2">
				{s.needs_attention && pendingLabel ? (
					<span
						className="lo-pulse inline-block size-1.5 shrink-0 rounded-full bg-danger"
						aria-hidden
					/>
				) : null}
				<span
					className={cn(
						"min-w-0 flex-1 truncate text-body-sm font-medium",
						s.streaming && !s.ended && "lo-shimmer",
					)}
				>
					{s.conversation_name || "untitled"}
				</span>
				{s.needs_attention && pendingLabel ? (
					<span className="shrink-0 text-meta text-danger">
						{pendingLabel}
					</span>
				) : null}
				{s.subagents_running > 0 ? (
					<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
						⟳ {s.subagents_running}
					</span>
				) : null}
				{s.todos_open ? (
					<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
						☐ {s.todos_open}
					</span>
				) : null}
				{s.ended ? (
					<span className="shrink-0 text-meta text-ink-dim">ended</span>
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

	useEffect(() => retainSessionListStream(), []);
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
			<main className="flex flex-1 flex-col px-1 pb-2">
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
					<div className="flex flex-col">
						{sessions.map((s) => (
							<SessionCard key={s.pid} s={s} home={home} />
						))}
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
					onClick={() => navigate("/past")}
					className="flex min-h-11 items-center justify-center rounded-md border border-control bg-surface px-4 text-body-sm text-ink select-none active:bg-elevated"
				>
					past
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

/**
 * Past sessions sheet (`#/past`): searchable, resumable history.
 *
 * Two upgrades over the copy-the-id v1:
 *
 * - **Resume is a button.** A tap reopens the session as a NEW live session
 *   (the daemon spawns a child that resumes the transcript, the same
 *   `--resume` mechanism the CLI uses) and the router takes the phone
 *   straight to it — open, and ready to take a command. No clipboard, no
 *   slash command.
 * - **Search covers what was SAID**, not only the name. The query runs
 *   through the same cached conversation index the TUI's /resume picker uses,
 *   so a distinctive phrase from inside a conversation finds it even when the
 *   name is "untitled". Rows that matched only on their body are marked, or
 *   the hit would read as arbitrary.
 */
import { useEffect, useRef, useState } from "react";
import { resumeSession, searchSessions } from "../api";
import { formatRelative } from "../lib/format";
import { navigate } from "../router";
import type { PastSession } from "../types";

export function PastSessionsScreen() {
	const [query, setQuery] = useState("");
	const [sessions, setSessions] = useState<PastSession[] | null>(null);
	const [error, setError] = useState("");
	/* The session id currently being resumed, so its button shows progress and
	   a double-tap can't spawn two children. */
	const [resumingId, setResumingId] = useState("");
	const [resumeError, setResumeError] = useState("");
	/* Debounce the search: one in-flight query, fired after the user pauses. */
	const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const seqRef = useRef(0);

	useEffect(() => {
		/* A stale response must not overwrite a newer query's results, so each
		   run is sequenced and only the latest applies. */
		const seq = ++seqRef.current;
		if (timerRef.current) clearTimeout(timerRef.current);
		timerRef.current = setTimeout(() => {
			searchSessions(query)
				.then((r) => {
					if (seq === seqRef.current) setSessions(r.sessions);
				})
				.catch((e) => {
					if (seq === seqRef.current)
						setError(String((e as Error).message ?? e));
				});
		}, 180);
		return () => {
			if (timerRef.current) clearTimeout(timerRef.current);
		};
	}, [query]);

	const resume = async (id: string) => {
		if (resumingId) return;
		setResumingId(id);
		setResumeError("");
		try {
			const r = await resumeSession(id);
			navigate(`/s/${r.pid}`);
		} catch (e) {
			setResumeError(String((e as Error).message ?? e));
			setResumingId("");
		}
	};

	return (
		<div className="mx-auto flex min-h-full w-full max-w-md flex-col">
			<header className="flex items-center gap-2 px-2 pt-[max(env(safe-area-inset-top),0.75rem)] pb-2">
				<button
					type="button"
					onClick={() => navigate("/")}
					aria-label="back"
					className="flex min-h-11 min-w-11 items-center justify-center rounded-sm text-ink-muted active:bg-elevated"
				>
					‹
				</button>
				<h1 className="text-heading">past sessions</h1>
			</header>

			<div className="px-2 pb-2">
				<input
					value={query}
					onChange={(e) => setQuery(e.target.value)}
					placeholder="search names and conversations…"
					spellCheck={false}
					autoCapitalize="off"
					autoCorrect="off"
					className="min-h-11 w-full rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim"
				/>
			</div>

			<main className="flex flex-1 flex-col gap-0.5 px-2 pb-3">
				{resumeError ? (
					<p className="px-2 pb-1 text-body-sm text-danger">
						{resumeError}
					</p>
				) : null}
				{error ? (
					<p className="text-body-sm text-danger">{error}</p>
				) : sessions === null ? (
					<p className="text-body-sm text-ink-dim">loading…</p>
				) : sessions.length === 0 ? (
					<p className="text-body-sm text-ink-dim">
						{query ? "no matches" : "no past sessions yet"}
					</p>
				) : (
					sessions.map((s) => (
						<div
							key={s.id}
							className="flex items-center gap-2 rounded-sm px-2 py-1 active:bg-elevated"
						>
							<div className="min-w-0 flex-1">
								<span className="block truncate text-body">
									{/* A PREFIX, dim, ahead of the name — the same
									   mark and the same position the TUI's /resume
									   picker uses, because it is one fact on two
									   surfaces. Ahead of the name specifically so
									   `truncate` eats the tail of a long inherited
									   title rather than the tag: a fresh fork and its
									   parent are otherwise identical rows, and the
									   long descriptive titles are exactly the ones a
									   suffix would lose the mark on. The inherited
									   title is kept beside it because it is the only
									   thing saying what this branched from. */}
									{s.forked ? (
										<span className="mr-1.5 text-meta text-ink-dim">
											[fork]
										</span>
									) : null}
									{s.name || "untitled"}
									{s.body_match ? (
										<span className="ml-1.5 text-meta text-info">
											in conversation
										</span>
									) : null}
								</span>
								<span className="block truncate font-mono text-mono-sm text-ink-dim">
									{s.id}
								</span>
							</div>
							{/* The timestamp gets its OWN shrink-0 slot: appended to the
							   truncating id it was always the part clipped away, and
							   "which session was this" is exactly what it answers. */}
							<span className="shrink-0 text-meta text-ink-dim">
								{formatRelative(s.mtime)}
							</span>
							<button
								type="button"
								onClick={() => void resume(s.id)}
								disabled={resumingId !== ""}
								className="min-h-9 shrink-0 rounded-sm border border-control bg-surface px-3 text-body-sm text-ink active:bg-accent-wash disabled:opacity-50"
							>
								{resumingId === s.id ? "opening…" : "resume"}
							</button>
						</div>
					))
				)}
			</main>
		</div>
	);
}

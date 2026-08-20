/**
 * Past sessions sheet (`#/past`): resumable history. Resuming needs a live
 * session to attach to, which this screen does not have — so for v1 a tap
 * copies the id and shows the hint to resume from a live session's composer
 * with `/resume <id>`.
 */
import { useEffect, useState } from "react";
import { getPastSessions } from "../api";
import { formatRelative } from "../lib/format";
import { navigate } from "../router";
import type { PastSession } from "../types";

export function PastSessionsScreen() {
	const [sessions, setSessions] = useState<PastSession[] | null>(null);
	const [copiedId, setCopiedId] = useState("");
	const [error, setError] = useState("");

	useEffect(() => {
		getPastSessions()
			.then((r) => setSessions(r.sessions))
			.catch((e) => setError(String((e as Error).message ?? e)));
	}, []);

	const copy = async (id: string) => {
		setCopiedId(id);
		try {
			await navigator.clipboard.writeText(id);
		} catch {
			/* The hint row still shows the id for manual copy. */
		}
	};

	return (
		<div className="mx-auto flex min-h-full w-full max-w-md flex-col">
			<header className="flex items-center gap-2 px-2 pt-[max(env(safe-area-inset-top),0.75rem)] pb-3">
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
			<main className="flex flex-1 flex-col gap-1 px-4 pb-4">
				{error ? (
					<p className="text-body-sm text-danger">{error}</p>
				) : sessions === null ? (
					<p className="text-body-sm text-ink-dim">loading…</p>
				) : sessions.length === 0 ? (
					<p className="text-body-sm text-ink-dim">
						no past sessions yet
					</p>
				) : (
					sessions.map((s) => (
						<div key={s.id}>
							<button
								type="button"
								onClick={() => copy(s.id)}
								className="flex min-h-11 w-full items-center gap-2 rounded-sm px-2 text-left active:bg-elevated"
							>
								<span className="min-w-0 flex-1">
									<span className="block truncate text-body">
										{s.name || "untitled"}
									</span>
									<span className="block truncate font-mono text-mono-sm text-ink-dim">
										{s.id}
									</span>
								</span>
								<span className="shrink-0 text-meta text-ink-dim">
									{formatRelative(s.mtime)}
								</span>
							</button>
							{copiedId === s.id ? (
								<p className="px-2 pb-1 text-meta text-info">
									id copied — resume from a live session's
									composer: /resume {s.id}
								</p>
							) : null}
						</div>
					))
				)}
			</main>
		</div>
	);
}

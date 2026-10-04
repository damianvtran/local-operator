/**
 * The composer's working-directory chip, and the sheet it opens.
 *
 * WHY A CHIP AND NOT A SCREEN. The new-session screen that used to ask for a
 * working directory is gone (one tap starts a session and lands in it, and the
 * daemon resolves the directory). That leaves the OTHER half of the operator's
 * ask — "the user can change it from the composer view if they want to" — so
 * the composer carries the current directory as a chip and offers the change in
 * place, without a route, without losing the draft, and without ever leaving
 * the conversation.
 *
 * THE SESSION IDENTITY IS PRESERVED BY THE DAEMON, not by this component: the
 * route retires the unused runtime and starts a successor under the SAME id, so
 * everything this screen owns — the draft in the composer, the transcript, the
 * stream, the route — is untouched. That is why a successful move simply closes
 * the sheet: the chip repaints from the session's own stream.
 *
 * WHAT THE CHIP SHOWS IS THE SESSION'S REAL cwd. It reads ``projection.cwd``,
 * which is the runtime's own publication, never a local echo of what the user
 * picked; ``optimisticCwd`` in the composer only bridges the second or two
 * between a successful move and the successor's first projection, and yields to
 * the projection the moment they agree.
 */
import { useEffect, useMemo, useState } from "react";
import { changeDirectory, getDirectories } from "../api";
import { cn } from "../lib/cn";
import { shortenHome } from "../lib/format";
import type { Directories } from "../types";
import { Sheet } from "./ui/sheet";

/** One choosable directory in the sheet: its path and the word that names it. */
interface DirectoryRow {
	path: string;
	tag: "current" | "home" | "tmp" | "recent";
}

/**
 * A path, spelled the way the user knows it: ``~``-relative when it is under
 * home, absolute otherwise. One helper for the chip and the rows so the two can
 * never disagree about what a directory is called.
 */
function displayPath(path: string, home: string): string {
	return path ? shortenHome(path, home) : "";
}

/** The working-directory sheet: pick a directory, or type one. */
export function DirectorySheet({
	open,
	onClose,
	sessionId,
	cwd,
	home,
	onMoved,
}: {
	open: boolean;
	onClose: () => void;
	/** The session whose directory may change. The id survives the move. */
	sessionId: string;
	/** The session's current directory — the row the sheet marks as "current". */
	cwd: string;
	/** The owner's home, for ``~`` display. Empty until the fetch lands. */
	home: string;
	/** Called with the directory the daemon accepted, so the composer can show
	    it before the successor's projection arrives. */
	onMoved: (cwd: string) => void;
}) {
	const [dirs, setDirs] = useState<Directories | null>(null);
	const [typed, setTyped] = useState("");
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState("");

	/* RECENTS ARE FETCHED ON OPEN, never cached across opens: the whole point of
	   that list is "where you have been working lately", and a stale copy would
	   offer the reader a directory they stopped using. The chip's own fetch
	   (below) covers only ``home``, which does not move. */
	useEffect(() => {
		if (!open) return;
		let cancelled = false;
		getDirectories()
			.then((d) => {
				if (!cancelled) setDirs(d);
			})
			.catch(() => {
				/* The sheet still works with the session's own cwd and the free-text
				   field; a catalogue failure must not blank the surface. */
			});
		return () => {
			cancelled = true;
		};
	}, [open]);

	/* A typed path is the reader's OWN input, so it is cleared when the sheet
	   closes — the next open starts from the session's real state rather than
	   from a half-typed path that was abandoned. */
	useEffect(() => {
		if (!open) {
			setTyped("");
			setError("");
		}
	}, [open]);

	const rows = useMemo(() => {
		const out: DirectoryRow[] = [];
		const seen = new Set<string>();
		const push = (path: string, tag: DirectoryRow["tag"]) => {
			if (!path || seen.has(path)) return;
			seen.add(path);
			out.push({ path, tag });
		};
		push(cwd, "current");
		push(dirs?.home ?? home, "home");
		if (dirs?.tmp) push(dirs.tmp, "tmp");
		for (const recent of dirs?.recent ?? []) push(recent, "recent");
		return out;
	}, [cwd, dirs, home]);

	const commit = async (path: string) => {
		const target = path.trim();
		if (!target || busy) return;
		setBusy(true);
		setError("");
		try {
			await changeDirectory(sessionId, target);
			onMoved(target);
			onClose();
		} catch (e) {
			/* THE SHEET STAYS OPEN ON A REFUSAL, showing the daemon's own sentence
			   (the phone renders ``error`` verbatim by contract). Closing on a
			   refusal would take the explanation away with the sheet and leave the
			   reader looking at an unchanged chip with no idea why. */
			setError(String((e as Error).message ?? e));
		} finally {
			setBusy(false);
		}
	};

	return (
		<Sheet open={open} onClose={onClose} title="working directory">
			<div className="flex flex-col gap-2 px-1 pb-2">
				{rows.map(({ path, tag }) => (
					<button
						key={`${tag}:${path}`}
						type="button"
						onClick={() => void commit(path)}
						disabled={busy || tag === "current"}
						aria-current={tag === "current" ? "true" : undefined}
						className={cn(
							"flex min-h-11 items-center gap-2 rounded-sm border px-3 text-left active:bg-elevated disabled:opacity-60",
							tag === "current" ? "border-accent bg-accent-wash" : "border-control bg-surface",
						)}
					>
						<span className="min-w-0 flex-1 truncate font-mono text-mono-sm text-ink">
							{displayPath(path, dirs?.home ?? home)}
						</span>
						{/* TEXT, never a glyph: a tick renders as tofu on phones whose
						    system font lacks the codepoint (the list's own rule for its
						    state marks), and this row is the one that must read as a
						    state rather than as another choice. */}
						<span className="shrink-0 text-meta text-ink-dim">{tag}</span>
					</button>
				))}

				{/* THE FREE-TEXT FALLBACK, for a directory none of the rows names —
				    the same affordance the deleted picker had, kept because the daemon
				    admits any directory under home or tmp and the rows cannot list
				    them all. */}
				<div className="flex items-center gap-2">
					<input
						value={typed}
						onChange={(e) => setTyped(e.target.value)}
						placeholder="or type another path…"
						spellCheck={false}
						autoCapitalize="off"
						autoCorrect="off"
						className="min-h-11 min-w-0 flex-1 rounded-sm border border-hairline bg-transparent px-3 font-mono text-mono-sm text-ink outline-none placeholder:text-ink-dim focus:border-control"
					/>
					<button
						type="button"
						onClick={() => void commit(typed)}
						disabled={busy || !typed.trim()}
						className="flex min-h-11 shrink-0 items-center justify-center rounded-sm border border-control bg-surface px-3 text-body-sm text-ink active:bg-elevated disabled:text-ink-disabled"
					>
						{busy ? "moving…" : "change"}
					</button>
				</div>

				{error ? (
					<p role="alert" aria-live="assertive" className="text-body-sm text-danger">
						{error}
					</p>
				) : null}
			</div>
		</Sheet>
	);
}

/**
 * The composer's working-directory chip: shows the session's directory and opens
 * the sheet above it.
 *
 * The chip lives in the composer cluster because that is where the reader
 * already is when the question arises, and because the directory is an
 * attribute of the conversation they are composing into, not a setting.
 */
export function WorkingDirectoryChip({
	sessionId,
	cwd,
	onMoved,
}: {
	sessionId: string;
	/** The session's real cwd (``projection.cwd``), empty until published. */
	cwd: string;
	onMoved: (cwd: string) => void;
}) {
	const [open, setOpen] = useState(false);
	const [home, setHome] = useState("");

	/* ``home`` is fetched once per mount and only for DISPLAY: while it is in
	   flight the chip falls back to the raw absolute path, which is the honest
	   rendering of "we do not know where home is yet" — an unshortened path, not
	   a missing one. */
	useEffect(() => {
		let cancelled = false;
		getDirectories()
			.then((d) => {
				if (!cancelled) setHome(d.home);
			})
			.catch(() => {
				/* Path shortening is cosmetic; the raw path is the fallback. */
			});
		return () => {
			cancelled = true;
		};
	}, []);

	const label = cwd ? displayPath(cwd, home) : "…";

	return (
		<div className="flex items-center">
			<button
				type="button"
				onClick={() => setOpen(true)}
				aria-label={`working directory: ${cwd || "unknown"}`}
				className="flex min-h-8 max-w-full items-center gap-1.5 rounded-sm px-1.5 text-mono-sm text-ink-muted active:bg-elevated"
			>
				{/* A folder glyph drawn inline: this package carries no icon
				    dependency (the same reason the list's marks are text), and a
				    16px svg costs no font lookup. */}
				<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
					<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
				</svg>
				<span className="min-w-0 truncate">{label}</span>
			</button>
			<DirectorySheet
				open={open}
				onClose={() => setOpen(false)}
				sessionId={sessionId}
				cwd={cwd}
				home={home}
				onMoved={onMoved}
			/>
		</div>
	);
}

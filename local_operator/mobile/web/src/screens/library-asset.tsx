/**
 * Library asset detail (`#/library/a/:assetId`).
 *
 * A full screen rather than a sheet, because the route is deep-linkable and the
 * composer at the bottom is a real destination: the phone can return here from
 * a session, and an in-app link can name one asset. The screen renders what the
 * relay knows, previews by kind, and keeps the one composer this surface has.
 *
 * PREVIEW HONESTY (design §6.3 / recon §5):
 * - image, audio, video → the bytes route (Range-capable, inline, nosniff);
 * - markdown / text / code → fetched text, the markdown renderer for `.md`;
 * - html → a fetched `data:` document in a `sandbox=""` frame (the page's CSP
 *   only allows `frame-src data:` and the relay also stamps `CSP: sandbox`);
 * - PDF / PPTX / spreadsheets / other → a genuine open/download row. No frame
 *   is promised that the CSP and `Content-Disposition` forbid.
 *
 * The composer sends the asset's identity as the first message, then the
 * reader's words, records the asset↔session link and navigates to the session.
 * If the send cannot be proven, the same text lands in that session's draft —
 * the app's existing convention — so nothing typed is lost.
 */
import { useEffect, useRef, useState } from "react";
import {
	getLibraryAsset,
	libraryBytesUrl,
	linkLibraryChat,
	sendCommand,
	startSession,
} from "../api";
import { MetadataSheet } from "../components/library/metadata-sheet";
import { Button } from "../components/ui/button";
import { Markdown } from "../components/markdown";
import { columnCap } from "../lib/column";
import { cn } from "../lib/cn";
import { assetPrompt, formatBytes, kindLabel, originLabel } from "../lib/library";
import { formatRelative } from "../lib/format";
import { navigate, navigateUp } from "../router";
import { setDraft } from "../store";
import type { LibraryAsset } from "../types";

const TEXT_PREVIEW_LIMIT = 32_000;
const HTML_PREVIEW_LIMIT = 512_000;

type TextState = { state: "loading" } | { state: "error" } | { state: "ready"; text: string };

function useAssetText(asset: LibraryAsset | null, limit: number): TextState {
	const [value, setValue] = useState<TextState>({ state: "loading" });
	useEffect(() => {
		if (!asset) return;
		let live = true;
		const controller = new AbortController();
		setValue({ state: "loading" });
		fetch(libraryBytesUrl(asset.id), {
			credentials: "include",
			signal: controller.signal,
		})
			.then(async (response) => {
				if (!response.ok) throw new Error(String(response.status));
				const raw = await response.text();
				if (!live) return;
				setValue({ state: "ready", text: raw.slice(0, limit) });
			})
			.catch(() => {
				if (live) setValue({ state: "error" });
			});
		return () => {
			live = false;
			controller.abort();
		};
	}, [asset]);
	return value;
}

function Preview({ asset }: { asset: LibraryAsset }) {
	const text = useAssetText(
		["text", "markdown", "code", "html"].includes(asset.kind) ? asset : null,
		asset.kind === "html" ? HTML_PREVIEW_LIMIT : TEXT_PREVIEW_LIMIT,
	);

	if (!asset.exists) {
		return (
			<div className="rounded-md border border-hairline bg-sunken p-4 text-center text-body-sm text-ink-muted">
				The file this entry names is no longer on this machine.
			</div>
		);
	}

	if (asset.kind === "image") {
		return (
			<img
				src={libraryBytesUrl(asset.id)}
				alt={asset.name}
				className="w-full rounded-md bg-sunken object-contain"
				style={columnCap(0.55)}
			/>
		);
	}

	if (asset.kind === "video") {
		return (
			<video
				controls
				preload="metadata"
				src={libraryBytesUrl(asset.id)}
				className="w-full rounded-md bg-sunken"
				style={columnCap(0.55)}
			/>
		);
	}

	if (asset.kind === "audio") {
		return (
			<audio
				controls
				preload="metadata"
				src={libraryBytesUrl(asset.id)}
				className="w-full"
			/>
		);
	}

	if (asset.kind === "markdown" || asset.kind === "text" || asset.kind === "code") {
		if (text.state === "loading") return <p className="text-body-sm text-ink-dim">loading preview…</p>;
		if (text.state === "error") return <p className="text-body-sm text-danger">Couldn&apos;t load this file.</p>;
		return asset.kind === "markdown" ? (
			<div className="rounded-md border border-hairline bg-surface p-3">
				<Markdown text={text.text} />
			</div>
		) : (
			<pre className="lo-scroll max-h-[50vh] overflow-auto rounded-md bg-sunken p-3 font-mono text-mono-sm whitespace-pre-wrap text-ink">
				{text.text}
			</pre>
		);
	}

	if (asset.kind === "html") {
		if (text.state === "loading") return <p className="text-body-sm text-ink-dim">loading preview…</p>;
		if (text.state === "error") return <p className="text-body-sm text-danger">Couldn&apos;t load this file.</p>;
		const frame = `data:text/html;charset=utf-8,${encodeURIComponent(text.text.slice(0, HTML_PREVIEW_LIMIT))}`;
		return (
			<iframe
				title={`${asset.name} preview`}
				sandbox=""
				src={frame}
				className="h-[50vh] w-full rounded-md border border-hairline bg-surface"
			/>
		);
	}

	/* Everything else is an honest fallback, not a failed preview. */
	const note =
		asset.kind === "pdf"
			? "PDF previews aren't rendered on this relay. Open or download it below."
			: asset.kind === "pptx"
				? "Slide rendering isn't available on this phone. Download it or open it on desktop."
				: "This kind has no in-app preview. Open or download it below.";
	return (
		<div className="flex flex-col items-center gap-2 rounded-md border border-hairline bg-sunken p-4 text-center">
			<span className="font-mono text-mono text-ink-muted">{kindLabel(asset.kind)}</span>
			<p className="text-body-sm text-ink-muted">{note}</p>
			<div className="flex gap-2">
				<a
					href={libraryBytesUrl(asset.id)}
					target="_blank"
					rel="noreferrer"
					className="inline-flex min-h-11 items-center rounded-md border border-control bg-surface px-3 text-body-sm text-ink active:bg-elevated"
				>
					open
				</a>
				<a
					href={libraryBytesUrl(asset.id, true)}
					className="inline-flex min-h-11 items-center rounded-md border border-control bg-surface px-3 text-body-sm text-ink active:bg-elevated"
				>
					download
				</a>
			</div>
		</div>
	);
}

function commandId(): string {
	if (typeof globalThis.crypto?.randomUUID === "function") return globalThis.crypto.randomUUID();
	return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function LibraryAssetScreen({ assetId }: { assetId: string }) {
	const [asset, setAsset] = useState<LibraryAsset | null>(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState("");
	const [editOpen, setEditOpen] = useState(false);
	const [draft, setDraftText] = useState("");
	const [sending, setSending] = useState(false);
	const [notice, setNotice] = useState("");
	const composerRef = useRef<HTMLTextAreaElement>(null);

	const load = async (id: string) => {
		setLoading(true);
		try {
			setAsset(await getLibraryAsset(id));
			setError("");
		} catch (e) {
			setError(String((e as Error).message ?? e));
		} finally {
			setLoading(false);
		}
	};

	useEffect(() => {
		void load(assetId);
	}, [assetId]);

	const ask = async () => {
		const question = draft.trim();
		if (!asset || !question || sending) return;
		setSending(true);
		setNotice("");
		let sessionId = asset.chat?.session_id ?? "";
		try {
			if (!sessionId) {
				sessionId = (await startSession()).session_id;
			}
			await sendCommand(sessionId, {
				op: "prompt",
				command_id: commandId(),
				text: assetPrompt(asset, question),
			});
			if (!asset.chat) {
				/* The link is recorded only after a REAL session exists; a failure
				   here must not lose the conversation the reader is heading to. */
				await linkLibraryChat(asset.id, sessionId).catch(() => undefined);
			}
			navigate(`/s/${encodeURIComponent(sessionId)}`);
		} catch (e) {
			if (sessionId) {
				/* The composer's own convention: a send that cannot be proven leaves
				   the words in that session's draft rather than dropping them. */
				setDraft(sessionId, assetPrompt(asset, question));
				navigate(`/s/${encodeURIComponent(sessionId)}`);
			} else {
				setNotice(String((e as Error).message ?? e));
				setSending(false);
			}
		}
	};

	if (loading) {
		return (
			<div className="mx-auto flex h-dvh w-full max-w-[var(--lo-column-max,28rem)] flex-col px-3 pt-[max(env(safe-area-inset-top),0.75rem)]">
				<p className="text-body-sm text-ink-dim">loading asset…</p>
			</div>
		);
	}

	if (error || !asset) {
		return (
			<div className="mx-auto flex h-dvh w-full max-w-[var(--lo-column-max,28rem)] flex-col overflow-hidden">
				<header className="flex items-center gap-2 px-2 pt-[max(env(safe-area-inset-top),0.75rem)] pb-2">
					<button
						type="button"
						onClick={() => navigateUp("/library")}
						aria-label="back to library"
						className="flex min-h-11 min-w-11 items-center justify-center rounded-sm text-ink-muted active:bg-elevated"
					>
						‹
					</button>
					<h1 className="text-heading text-ink">asset</h1>
				</header>
				<div className="flex flex-col items-center gap-2 px-3 py-10">
					<p role="alert" className="text-center text-body-sm text-danger">
						{error || "Asset not found."}
					</p>
					<Button variant="outline" onClick={() => void load(assetId)}>
						Try again
					</Button>
				</div>
			</div>
		);
	}

	return (
		<div className="mx-auto flex h-dvh w-full max-w-[var(--lo-column-max,28rem)] flex-col overflow-hidden">
			<header className="flex items-center gap-1 px-2 pt-[max(env(safe-area-inset-top),0.75rem)] pb-2">
				<button
					type="button"
					onClick={() => navigateUp("/library")}
					aria-label="back to library"
					className="flex min-h-11 min-w-11 items-center justify-center rounded-sm text-ink-muted active:bg-elevated"
				>
					‹
				</button>
				<h1 className="min-w-0 flex-1 truncate px-1 text-heading text-ink">{asset.name}</h1>
				<button
					type="button"
					onClick={() => setEditOpen(true)}
					className="flex min-h-11 items-center rounded-md border border-control bg-surface px-2 text-meta text-ink-muted active:bg-elevated"
				>
					edit
				</button>
			</header>

			<main className="lo-scroll flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-3 pb-3">
				<Preview asset={asset} />

				<section className="flex flex-wrap items-center gap-1">
					<span className="rounded-xs bg-elevated px-1 text-meta text-ink-muted">
						{originLabel(asset.origin_group)}
					</span>
					<span className="text-meta text-ink-dim">
						{kindLabel(asset.kind)}
						{asset.size_bytes ? ` · ${formatBytes(asset.size_bytes)}` : ""}
					</span>
					{asset.highlighted ? <span className="text-meta text-accent">★ highlighted</span> : null}
					{asset.favorite ? <span className="text-meta text-accent">♥ favorite</span> : null}
					{asset.organized ? (
						<span className="text-meta text-ink-dim">
							organized {formatRelative(asset.organized.at)}
						</span>
					) : (
						<span className="text-meta text-ink-dim">unorganized</span>
					)}
				</section>

				{asset.description ? (
					<p className="text-body-sm text-ink">{asset.description}</p>
				) : null}

				{asset.ref.display_path ? (
					<p className="truncate font-mono text-mono-sm text-ink-dim">{asset.ref.display_path}</p>
				) : null}

				{asset.tags.length ? (
					<div className="flex flex-wrap gap-1">
						{asset.tags.map((tag) => (
							<span
								key={tag.id}
								className="rounded-xs bg-elevated px-1.5 py-0.5 text-meta text-ink-muted"
							>
								{tag.name}
							</span>
						))}
					</div>
				) : null}

				{asset.occurrences.length ? (
					<section className="flex flex-col gap-1">
						<h2 className="text-meta text-ink-muted">source conversations</h2>
						{asset.occurrences.map((occurrence) => (
							<button
								key={`${occurrence.session_id}:${occurrence.child_id ?? ""}`}
								type="button"
								onClick={() => navigate(`/s/${encodeURIComponent(occurrence.session_id)}`)}
								className="flex min-h-11 items-center justify-between gap-2 rounded-sm border border-hairline bg-surface px-2 text-left active:bg-elevated"
							>
								<span className="min-w-0 flex-1 truncate text-body-sm text-ink">
									{occurrence.session_title || occurrence.session_id}
								</span>
								<span className="text-meta text-ink-dim">{occurrence.origin}</span>
							</button>
						))}
					</section>
				) : null}

				{asset.chat ? (
					<button
						type="button"
						onClick={() => navigate(`/s/${encodeURIComponent(asset.chat?.session_id ?? "")}`)}
						className="flex min-h-11 items-center gap-2 rounded-sm border border-info-border bg-info-wash px-2 text-left text-body-sm text-info active:bg-elevated"
					>
						<span aria-hidden>↳</span>
						<span className="min-w-0 flex-1 truncate">
							chat: {asset.chat.title || asset.chat.session_id}
						</span>
					</button>
				) : null}

				{notice ? (
					<p role="alert" className="text-body-sm text-danger">
						{notice}
					</p>
				) : null}
			</main>

			{/* Borderless by contract: no divider, no panel fill, safe-area aware.
			    The textarea grows to three rows, then scrolls. */}
			<div className="shrink-0 bg-canvas px-3 pb-[max(env(safe-area-inset-bottom),0.5rem)] pt-1">
				<div className="flex items-end gap-2">
					<textarea
						ref={composerRef}
						value={draft}
						onChange={(event) => setDraftText(event.target.value)}
						rows={1}
						placeholder="ask about this asset…"
						aria-label="ask about this asset"
						className="max-h-[66px] min-h-11 flex-1 resize-none border-0 bg-transparent px-1 py-2 text-body text-ink outline-none placeholder:text-ink-dim"
					/>
					<button
						type="button"
						onClick={() => void ask()}
						disabled={sending || !draft.trim()}
						aria-busy={sending}
						className={cn(
							"flex min-h-11 min-w-11 items-center justify-center rounded-md text-body-sm font-medium",
							sending || !draft.trim()
								? "bg-sunken text-ink-disabled"
								: "bg-accent text-on-accent active:bg-accent-active",
						)}
					>
						{sending ? "…" : "ask"}
					</button>
				</div>
			</div>

			<MetadataSheet
				open={editOpen}
				asset={asset}
				onClose={() => setEditOpen(false)}
				onSaved={(fresh) => setAsset(fresh)}
			/>
		</div>
	);
}

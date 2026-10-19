/**
 * The block under a final answer: file callouts, the quiet state line and any
 * generated components (turn-supplements memo §2.8, lane R).
 *
 * THE BODY-ONLY RULE (memo §2.8; operator directive 2026-10-10): the block
 * renders NO header, label, title, badge or separator. It starts directly with
 * its content, seamlessly under the answer's text; spacing and rhythm alone
 * carry the seam. The one caption allowed anywhere in the block is the frame's
 * own `Source: …` line, and the component's `title` is its name for tooling
 * and accessibility (the frame's `title` attribute), never painted chrome.
 *
 * THE STATES are `lib/supplement.ts`'s `supplementDisposition` — the frozen
 * reader rule every surface ports — and this component is only its renderer:
 *   - `preparing` / `files_preparing` — the row's callouts, then
 *     `◌ Preparing highlights… · Adjust… · Cancel`; the pair is drawn together
 *     or not at all, both visible, both ≥ 44 px (D10 — there is no hover on a
 *     phone, so a control that appears on hover would not exist);
 *   - `block` — files, then components, then nothing else;
 *   - `cancelled_retry` / `failed_retry` — the one quiet line with `· Retry`
 *     (`supplement_restart`, the runtime's own arm);
 *   - `nothing` — renders nothing at all.
 *
 * THE OPS RIDE THE SAME SESSION COMMAND ROUTE every other mutation uses
 * (`api.ts`'s supplement helpers → `POST /api/sessions/{id}/command`), scoped
 * by the session in the URL and the anchor+job pair in the frame. The relay
 * forwards them to the live runtime, whose receipt is the answer's `detail`
 * (`cancelled` / `steering` / `restarting` / the neutral `already finished`);
 * this component never paints a receipt as an outcome — the ROW's next state
 * is the outcome, exactly as the desktop's receipt model says. A REFUSAL
 * (no live session: 409; an older owner or a bad shape: 422; a transport
 * failure: 502/504) is the one thing the reader must see, painted under the
 * line in the surface's refusal voice (`lib/refusal.ts`, the same
 * `Could not <act>: <the daemon's sentence>` hygiene the ask card uses) and
 * cleared by the next press.
 *
 * WHAT THIS FILE DOES NOT DO. No component document is interpreted; no file
 * is listed or read beyond the row's own recorded callouts (the sensitivity
 * predicate is the server's, re-applied by the relay route — this surface
 * renders what the row carries and fetches only by the row's own index).
 */
import { memo, useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { Disclosure } from "./ui/disclosure";
import { SupplementFrame } from "./supplement-frame";
import {
	getSupplementDocument,
	getSupplementFile,
	supplementCancel,
	supplementRestart,
	supplementSteer,
} from "../api";
import { refusalText } from "../lib/refusal";
import {
	COPY_ADJUST,
	COPY_CANCEL,
	COPY_CANCELLED,
	COPY_FAILED,
	COPY_PREPARING,
	COPY_RETRY,
	COPY_SHOW_FEWER,
	MORE_MAX,
	STEER_TEXT_MAX_CHARS,
	causePhrase,
	displayText,
	fileRowParts,
	filesSummaryOf,
	frameBudgetPx,
	moreText,
	planComponentMounts,
	showAllText,
	supplementDisposition,
	supplementFileClass,
	type SupplementBeat,
	type SupplementComponent,
	type SupplementFile,
} from "../lib/supplement";

/**
 * How much of a text preview renders. The route caps a served file at 10 MB,
 * and a phone cannot lay out ten megabytes of `<pre>`; the tail is cut here
 * and the cut is SAID (a `first N characters` caption) — never silently, so a
 * reader never mistakes the end of a preview for the end of a file.
 */
const PREVIEW_TEXT_MAX_CHARS = 32000;

/** The three op leads, one spelling per act (`refusalText`'s hygiene: the lead
    names what the reader attempted, then the daemon's own sentence). */
const OP_LEAD = {
	cancel: "Could not cancel",
	steer: "Could not adjust",
	restart: "Could not retry",
} as const;

export const SupplementBlock = memo(function SupplementBlock({
	beat,
	sessionId,
	jobLive,
}: {
	beat: SupplementBeat;
	/** The conversation's session id — the ops' and the fetch routes' scope. */
	sessionId: string;
	/** `beatLive(beat, projection.pid)` — computed by the transcript from the
	    projection's own process identity; the stale-row rule's input. */
	jobLive: boolean;
}) {
	const disposition = supplementDisposition(beat, jobLive);
	const [adjusting, setAdjusting] = useState(false);
	const [draft, setDraft] = useState("");
	const [busy, setBusy] = useState(false);
	const [opError, setOpError] = useState("");
	/*
	 * The rendered heights of frames already on screen, by digest: written by
	 * the frames' own resize beats, read by the budget's accrual. ONLY rendered
	 * heights arrive here (the frame reports after its first resize beat, never
	 * its reserved estimate): `planComponentMounts` holds the walk at any frame
	 * that has not reported, so a reserved value standing in for a measurement
	 * would defeat the accrual it exists for.
	 */
	const [measured, setMeasured] = useState<ReadonlyMap<string, number>>(() => new Map());
	/*
	 * The sticky auto-mount count: a frame that ever mounted stays mounted, so
	 * a later measurement that makes the accrual STOP fitting (a document that
	 * overran its hint) never unmounts it under the reader. The state only
	 * grows, and `Math.max` against the live suggestion means the suggestion's
	 * own growth (a smaller-than-hinted first frame freeing budget) mounts the
	 * next frame in the same commit.
	 */
	const [sticky, setSticky] = useState(0);

	const sendOp = useCallback(
		async (op: "cancel" | "restart" | "steer", text = ""): Promise<void> => {
			setBusy(true);
			setOpError("");
			try {
				if (op === "cancel") {
					await supplementCancel(sessionId, beat.anchor, beat.job);
				} else if (op === "restart") {
					await supplementRestart(sessionId, beat.anchor, beat.job);
				} else {
					await supplementSteer(sessionId, beat.anchor, beat.job, text);
				}
			} catch (error) {
				/* The refusal is composed, never echoed raw: the lead names the
				   press, the rest is the daemon's own sentence (which a phone
				   user needs — "session not connected" and "update the owner"
				   ask for different next steps). */
				setOpError(refusalText(OP_LEAD[op], error));
			} finally {
				setBusy(false);
			}
		},
		[beat.anchor, beat.job, sessionId],
	);

	const steer = () => {
		const text = draft.trim();
		if (!text) return;
		setAdjusting(false);
		setDraft("");
		void sendOp("steer", text);
	};

	const noteHeight = useCallback((digest: string, height: number) => {
		setMeasured((prev) => {
			if (prev.get(digest) === height) return prev;
			const next = new Map(prev);
			next.set(digest, height);
			return next;
		});
	}, []);

	/*
	 * The aggregate frame budget (§2.8): frames mount automatically only within
	 * a cumulative min(480px, 40vh) of block height, accrued from each frame's
	 * RENDERED height; the rest sit behind ONE disclosure that expands in place
	 * on an explicit action. The stickiness below is the other half — an
	 * already-mounted frame stays even when a late measurement says the accrual
	 * no longer fits.
	 */
	const budgetPx = frameBudgetPx(typeof window === "undefined" ? 900 : window.innerHeight);
	const { visible } = planComponentMounts(beat.components, budgetPx, measured);
	useEffect(() => {
		setSticky((count) => Math.max(count, visible.length));
	}, [visible.length]);
	const autoCount = Math.min(beat.components.length, Math.max(sticky, visible.length));
	/*
	 * The head mounts automatically; the tail is the disclosure's. Splitting
	 * ONCE (rather than a `shown`/`hidden` pair) is deliberate: an earlier
	 * draft computed a head and a tail independently and the open state then
	 * rendered the tail twice — once through the head it had swallowed and once
	 * through the disclosure's children.
	 */
	const head = beat.components.slice(0, autoCount);
	const tail = beat.components.slice(autoCount);

	if (disposition === "nothing") return null;

	const files = beat.files;
	const moreCount = beat.filesMore;
	const components = beat.components;
	const totalFiles = files.length + moreCount;

	/*
	 * The one quiet line for the arms that carry one (§2.8's copy table). The
	 * dot is the phone's own `lo-shimmer-bar` pulse over a standing sunken
	 * fill — the same motion vocabulary the transcript skeleton uses, and it
	 * parks (never vanishes) under reduced motion.
	 */
	const preparing =
		disposition === "preparing" || disposition === "files_preparing";
	const cancelled =
		disposition === "cancelled_retry" || disposition === "files_cancelled_retry";
	const failed = disposition === "failed_retry" || disposition === "files_failed_retry";
	// The typed cause names the wall the retry would meet (identical words to
	// the TUI, memo §2.8's one word list); rows without one — or with a token
	// this build does not know — keep the bare sentence.
	const causeLine = failed ? causePhrase(beat.cause) : "";
	const stateLine = preparing || cancelled || failed ? (
		<div
			className="flex flex-wrap items-center gap-x-2 gap-y-1 text-meta"
			data-supplement-line=""
		>
			{preparing && (
				<>
					<span
						aria-hidden="true"
						className="relative inline-block size-3 overflow-hidden rounded-full bg-sunken"
					>
						<span className="lo-shimmer-bar absolute inset-0 bg-elevated/70" />
					</span>
					<span className="text-ink-muted">{COPY_PREPARING}</span>
				</>
			)}
			{cancelled && <span className="text-ink-muted">{COPY_CANCELLED}</span>}
			{failed && (
				<span className="text-ink-muted">
					{COPY_FAILED}
					{causeLine ? ` — ${causeLine}` : ""}
				</span>
			)}
			{/* The action pair is drawn TOGETHER or not at all (D10), both
			    visible, each a ≥ 44 px target: a phone has no hover, so a
			    control that only appears on hover would not exist. */}
			{preparing && (
				<>
					<span aria-hidden="true" className="text-ink-dim">
						·
					</span>
					{!adjusting && (
						<button
							type="button"
							className="min-h-11 px-1 text-ink-dim"
							data-supplement-adjust=""
							onClick={() => setAdjusting(true)}
						>
							{COPY_ADJUST}
						</button>
					)}
					<button
						type="button"
						className="min-h-11 px-1 text-ink-dim disabled:opacity-50"
						data-supplement-cancel=""
						disabled={busy || beat.state === "cancelling"}
						onClick={() => void sendOp("cancel")}
					>
						{COPY_CANCEL}
					</button>
				</>
			)}
			{(cancelled || failed) && (
				<>
					<span aria-hidden="true" className="text-ink-dim">
						·
					</span>
					{/* `px-2`, not the sibling `px-1`: the word alone measured 38 px
					    at this size, under the surface's own 44 px touch rule
					    (D1/U1) — the extra step is the target, not decoration. */}
					<button
						type="button"
						className="min-h-11 px-2 text-ink-dim disabled:opacity-50"
						data-supplement-retry=""
						disabled={busy}
						onClick={() => void sendOp("restart")}
					>
						{COPY_RETRY}
					</button>
				</>
			)}
		</div>
	) : null;

	const adjustLine = adjusting ? (
		<div className="flex items-center gap-2" data-supplement-adjust-form="">
			<input
				// eslint-disable-next-line jsx-a11y/no-autofocus -- the field does
				// NOT exist until an explicit `Adjust…` press opened it — focusing
				// it is the whole meaning of that press, and there is no other
				// content to steal focus from.
				autoFocus={true}
				enterKeyHint="send"
				className="min-h-11 min-w-0 flex-1 rounded-sm border border-control bg-canvas px-2 text-body-sm text-ink"
				maxLength={STEER_TEXT_MAX_CHARS}
				placeholder="Adjust the graphic…"
				value={draft}
				onChange={(event) => setDraft(event.target.value)}
				onKeyDown={(event) => {
					if (event.key === "Enter") steer();
					if (event.key === "Escape") {
						setAdjusting(false);
						setDraft("");
					}
				}}
			/>
			<button
				type="button"
				className="min-h-11 px-2 text-ink-dim disabled:opacity-50"
				data-supplement-steer-send=""
				disabled={busy || !draft.trim()}
				onClick={steer}
			>
				Send
			</button>
		</div>
	) : null;

	/* A refusal the reader must see; cleared by the next press. */
	const errorLine = opError ? (
		<p className="text-meta text-danger" data-supplement-op-error="">
			{opError}
		</p>
	) : null;

	/*
	 * The files half first (memo §0). The ≤ 2 files one-liner applies when no
	 * COMPONENT rides the row and the whole known set is two files or fewer
	 * (the TUI's own rule, `total_files <= 2 and not components_present`), and
	 * it serves the preparing arm's callouts too — they are the same callouts a
	 * moment earlier. The names come from the model's own builder so the
	 * one-liner and the expanded block can never disagree.
	 */
	let filesSection: ReactNode = null;
	if (files.length > 0 || moreCount > 0) {
		if (totalFiles <= 2 && components.length === 0) {
			const summary = filesSummaryOf(beat);
			filesSection =
				summary === null ? null : (
					<div className="text-meta text-ink-muted" data-supplement-files-summary="">
						{summary}
					</div>
				);
		} else {
			filesSection = (
				<div className="flex flex-col" data-supplement-files="">
					{files.map((file, index) => (
						<CalloutRow
							key={`${file.path}:${file.name}`}
							file={file}
							index={index}
							job={beat.job}
							sessionId={sessionId}
						/>
					))}
					{moreCount > 0 && (
						/* The disclosure TOGGLES its words with the state, the desktop's
						   grammar exactly (`N more files` ⇄ `Show fewer`, UI #971;
						   UX round 1, U2): the chevron says the state, the label says
						   the action. The tail's rows carry NO preview affordance: the
						   file route addresses `files[i]` by index and the stored tail
						   has none, so the honest row is name-only. */
						<Disclosure
							className="text-meta"
							header={(open) => (
								<span className="text-ink-dim" data-supplement-more="">
									{open ? COPY_SHOW_FEWER : moreText(moreCount)}
								</span>
							)}
						>
							<div className="flex flex-col gap-1 pb-1">
								{beat.more.slice(0, MORE_MAX).map((item) => (
									<div key={item} className="flex items-center gap-2 text-meta">
										<span className="min-w-0 flex-1 truncate text-ink-muted">
											{displayText(item)}
										</span>
									</div>
								))}
							</div>
						</Disclosure>
					)}
				</div>
			);
		}
	}

	/* The components, only on a settled block. The disclosure mounts the tail's
	   frames only while it is open — an explicit action, which is the memo's
	   own condition for mounting beyond the budget. */
	const componentsSection =
		disposition === "block" && components.length > 0 ? (
			<div className="flex flex-col gap-2" data-supplement-components="">
				{head.map((component) => (
					<FrameSlot
						key={component.attachment}
						component={component}
						sessionId={sessionId}
						onRetry={() => void sendOp("restart")}
						onHeight={noteHeight}
					/>
				))}
				{tail.length > 0 && (
					/* The same toggle grammar as the files disclosure above
					   (`Show all N graphics` ⇄ `Show fewer`). */
					<Disclosure
						className="text-meta"
						header={(open) => (
							<span className="text-ink-dim" data-supplement-show-all="">
								{open ? COPY_SHOW_FEWER : showAllText(tail.length)}
							</span>
						)}
					>
						<div className="flex flex-col gap-2 pb-1">
							{tail.map((component) => (
								<FrameSlot
									key={component.attachment}
									component={component}
									sessionId={sessionId}
									onRetry={() => void sendOp("restart")}
									onHeight={noteHeight}
								/>
							))}
						</div>
					</Disclosure>
				)}
			</div>
		) : null;

	return (
		<div
			className="flex flex-col gap-1.5"
			data-supplement-block={disposition}
			data-supplement-record={beat.anchor}
		>
			{filesSection}
			{stateLine}
			{adjustLine}
			{errorLine}
			{componentsSection}
		</div>
	);
});

/* ---------------------------------------------------------------------------
 * One callout row, with the type-appropriate preview (§2.7/§3.4 N5)
 * ------------------------------------------------------------------------- */

type FilePreview =
	| { kind: "loading" }
	| { kind: "text"; text: string; truncated: boolean }
	| { kind: "image"; url: string }
	| { kind: "refused"; message: string };

function CalloutRow({
	file,
	index,
	job,
	sessionId,
}: {
	file: SupplementFile;
	index: number;
	job: string;
	sessionId: string;
}) {
	const { name, path, size } = fileRowParts(file);
	const fileClass = supplementFileClass(file.path);
	const previewable = fileClass === "text" || fileClass === "image";
	const [open, setOpen] = useState(false);
	const [preview, setPreview] = useState<FilePreview | null>(null);

	/* The object URL is this component's to revoke — a second holder would be
	   a revoke this row cannot see coming. */
	useEffect(() => {
		if (preview?.kind !== "image") return;
		const url = preview.url;
		return () => URL.revokeObjectURL(url);
	}, [preview]);

	const toggle = async () => {
		if (!previewable) return;
		if (open) {
			setOpen(false);
			return;
		}
		setOpen(true);
		if (preview !== null) return;
		setPreview({ kind: "loading" });
		try {
			const { blob } = await getSupplementFile(sessionId, job, index);
			if (fileClass === "image") {
				setPreview({ kind: "image", url: URL.createObjectURL(blob) });
			} else {
				const text = await blob.text();
				setPreview({
					kind: "text",
					text: text.slice(0, PREVIEW_TEXT_MAX_CHARS),
					truncated: text.length > PREVIEW_TEXT_MAX_CHARS,
				});
			}
		} catch (error) {
			setPreview({ kind: "refused", message: refusalText("Could not preview", error) });
		}
	};

	return (
		<div className="flex flex-col gap-1" data-supplement-file="">
			<div className="flex min-h-11 items-center gap-2">
				{previewable ? (
					/* THE ROW'S OWN 44 px IS THE TARGET (D1/U1): with `min-h-11` the
					   button answers a tap anywhere in the row's visible strip, not
					   just its 17 px text line — measured before the fix: the four
					   buttons were 326-332×17 inside a 44 px row and the padding
					   did not respond. At REST a previewable row deliberately looks
					   like a refused or inert one (D4): the desktop's file lines
					   carry no resting cue either (hover steps there, `active:`
					   steps here), and a chevron would promise a disclosure the
					   tail rows cannot honour. */
					<button
						type="button"
						className="min-h-11 min-w-0 flex-1 truncate text-left text-meta text-ink active:text-ink-dim"
						aria-expanded={open}
						onClick={() => void toggle()}
					>
						{name || displayText(file.path)}
						{path && name ? <span className="ml-2 text-ink-muted">{path}</span> : null}
					</button>
				) : (
					<span className="min-w-0 flex-1 truncate text-meta text-ink">
						{name || displayText(file.path)}
						{path && name ? <span className="ml-2 text-ink-muted">{path}</span> : null}
					</span>
				)}
				{size ? <span className="shrink-0 text-meta text-ink-dim">{size}</span> : null}
			</div>
			{open && preview !== null && (
				<div data-supplement-preview="" className="min-w-0">
					{preview.kind === "loading" && (
						<div aria-hidden="true" className="lo-shimmer-bar h-16 rounded-sm bg-sunken" />
					)}
					{preview.kind === "text" && (
						<>
							<pre className="max-h-64 overflow-auto rounded-sm bg-sunken px-2 py-1.5 font-mono text-mono-sm break-words whitespace-pre-wrap">
								{preview.text}
							</pre>
							{preview.truncated && (
								<span className="text-meta text-ink-dim">
									first {PREVIEW_TEXT_MAX_CHARS.toLocaleString()} characters
								</span>
							)}
						</>
					)}
					{preview.kind === "image" && (
						<img
							src={preview.url}
							alt={name || displayText(file.path)}
							className="max-h-64 max-w-full rounded-sm border border-hairline"
						/>
					)}
					{preview.kind === "refused" && (
						<span className="text-meta text-danger">{preview.message}</span>
					)}
				</div>
			)}
		</div>
	);
}

/* ---------------------------------------------------------------------------
 * One component: the document fetch, then the sandboxed frame
 * ------------------------------------------------------------------------- */

function FrameSlot({
	component,
	sessionId,
	onRetry,
	onHeight,
}: {
	component: SupplementComponent;
	sessionId: string;
	onRetry?: () => void;
	onHeight: (digest: string, height: number) => void;
}) {
	const anchorRef = useRef<HTMLDivElement | null>(null);
	/*
	 * THE §2.11 MOUNT GATE: the document is fetched only once the slot is at
	 * (or near) the viewport — the memo's "a component frame mounts only when
	 * its answer row is in the viewport", implemented exactly as the desktop
	 * slot does (`supplement-row.tsx::SupplementFrameSlot`, same 200 px
	 * margin). Without it, opening a long session fetched every supplemented
	 * turn's document at render, including turns the reader never approached
	 * (UX round 1, U3). `near` only ever turns ON — an approached slot stays
	 * mounted, matching the budget's own stickiness.
	 */
	const [near, setNear] = useState(false);
	useEffect(() => {
		const element = anchorRef.current;
		if (element === null) return;
		if (typeof IntersectionObserver === "undefined") {
			/* No observer (older harness, a test environment without one): the
			   gate has nothing to hold on, so the slot mounts as before. */
			setNear(true);
			return;
		}
		const observer = new IntersectionObserver(
			(entries) => {
				for (const entry of entries) {
					if (entry.isIntersecting) setNear(true);
				}
			},
			{ rootMargin: "200px 0px" },
		);
		observer.observe(element);
		return () => observer.disconnect();
	}, []);

	const [html, setHtml] = useState<string | null>(null);
	const [failed, setFailed] = useState(false);
	const onHeightRef = useRef(onHeight);
	onHeightRef.current = onHeight;

	useEffect(() => {
		if (!near) return;
		let cancelled = false;
		setHtml(null);
		setFailed(false);
		getSupplementDocument(sessionId, component.attachment)
			.then((answer) => {
				if (!cancelled) setHtml(answer.html);
			})
			.catch(() => {
				/* The document could not be fetched (a pruned blob, a refusal,
				   a malformed body): the frame paints the frame-level fallback
				   line instead of holding its reserved box forever. */
				if (!cancelled) setFailed(true);
			});
		return () => {
			cancelled = true;
		};
	}, [near, sessionId, component.attachment]);

	return (
		<div ref={anchorRef} data-supplement-slot="">
			<SupplementFrame
				component={component}
				documentHtml={html}
				documentFailed={failed}
				onRetry={onRetry}
				onHeightChange={(height) => onHeightRef.current(component.attachment, height)}
			/>
		</div>
	);
}

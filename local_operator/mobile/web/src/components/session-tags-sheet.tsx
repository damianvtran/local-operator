/**
 * Conversation tags sheet — the phone's one ASSIGNMENT surface.
 *
 * WHAT THIS IS. The shared registry (`GET /api/tags`) as checkboxes, the
 * conversation's current assignment checked, one primary radio among the
 * selected rows, and a two-step clear. It WRITES NOTHING ELSE: category
 * identity (create/rename) stays with the desktop app and agent callers, and a
 * second implementation of it here is how two surfaces start disagreeing. The
 * catalogue is re-read on every open, and the sheet keeps no local store — the
 * relay's assignment is the only one.
 *
 * DESIRED STATE. Every interaction sends the COMPLETE `{tags, primary}` the
 * route takes (`setSessionTags`), not a delta: a dropped or retried response
 * lands on the same assignment, and the clear is just `[]` + `null`. The pure
 * rules (cap, fallback, label budget) live in `lib/session-tags.ts`.
 *
 * RACES. A tap paints a DRAFT at once; a write's read-back paints only if no
 * newer authoritative list frame arrived meanwhile, because the frame is newer
 * than the read-back and painting over it would move the sheet backwards.
 * A newer frame always clears the draft (it is the store's answer, and the
 * relay wakes the list on success anyway). A refusal reverts to the newest
 * frame and says the relay's own sentence — never silently.
 *
 * PENDING IS PER ROW, ONE WRITER. The rows whose membership the in-flight write
 * changes carry the pending mark; every control is disabled while it is in
 * flight, so two writes can never interleave into a state neither requested.
 */
import { useEffect, useId, useRef, useState, type RefObject } from "react";
import { getTags, setSessionTags } from "../api";
import { refusalText } from "../lib/refusal";
import {
	assignmentFrom,
	atTagCap,
	catalogueRows,
	choosePrimary,
	clearedAssignment,
	MAX_ASSIGNED_TAGS,
	toggleAssignment,
	type TagAssignment,
} from "../lib/session-tags";
import { cn } from "../lib/cn";
import { messageRuntime } from "../store";
import type { TagCatalogueEntry, TagRef } from "../types";
import { Sheet } from "./ui/sheet";

export function SessionTagsSheet({
	open,
	onClose,
	sessionId,
	assigned,
	primary,
	revision,
	returnFocusRef,
}: {
	open: boolean;
	onClose: () => void;
	sessionId: string;
	/** The last authoritative frame's assignment for this conversation. */
	assigned: readonly TagRef[];
	primary: string | null;
	/** Monotonic count of authoritative list frames the parent has applied.
	    The sheet compares it at write start and at read-back; a change means a
	    NEWER truth exists, so the older answer must not paint. */
	revision: number;
	/** The row that opened the sheet, so Done/Escape returns focus to it rather
	    than to the pin sheet's button, which unmounts in the same commit. */
	returnFocusRef?: RefObject<HTMLElement | null>;
}) {
	const [catalogue, setCatalogue] = useState<TagCatalogueEntry[] | null>(null);
	const [loadError, setLoadError] = useState("");
	/* Bumping this re-runs the catalogue effect — the retry control. */
	const [reload, setReload] = useState(0);
	const [draft, setDraft] = useState<TagAssignment | null>(null);
	const [writeError, setWriteError] = useState("");
	const [pending, setPending] = useState<{ ids: readonly string[] } | null>(null);
	const [confirmClear, setConfirmClear] = useState(false);
	const limitHintId = useId();
	/* Latest revision, readable from callbacks that captured an older render. */
	const revisionRef = useRef(revision);
	const seenRevision = useRef(revision);
	/* Catalogue reads are sequenced: a slow first read must not overwrite the
	   retry's answer (the same rule the past-sessions search follows). */
	const readSeq = useRef(0);
	/* Writes are sequenced for the same reason: an older promise settling after
	   a newer write started must not clear the newer write's pending mark. */
	const writeSeq = useRef(0);
	/* The row this sheet is CURRENTLY writing for. `writeSeq` is bumped whenever
	   the target changes (a close/reopen, or the parent pointing at another
	   row), and both the read-back and the refusal check it: a slow answer for
	   row A must never paint A's assignment — or A's error — into B's sheet. */
	const targetRef = useRef(sessionId);

	useEffect(() => {
		/* A NEW TARGET INVALIDATES EVERYTHING IN FLIGHT. The sequence bump makes
		   both callbacks of the old write no-ops; clearing the speculative state
		   keeps a closed or re-opened sheet from showing the last row's draft,
		   pending mark or error before its own first frame arrives. */
		targetRef.current = sessionId;
		writeSeq.current++;
		setDraft(null);
		setPending(null);
		setWriteError("");
		/* `open` belongs here too: closing and re-opening the SAME row is also a
		   new sheet whose state must not be inherited from the previous one. */
	}, [sessionId, open]);

	useEffect(() => {
		if (!open) return;
		const seq = ++readSeq.current;
		setCatalogue(null);
		setLoadError("");
		setDraft(null);
		setWriteError("");
		setPending(null);
		setConfirmClear(false);
		/* Opening snapshots the current frame, so a write started immediately
		   after the open compares against the right revision. */
		revisionRef.current = revision;
		seenRevision.current = revision;
		getTags()
			.then((answer) => {
				if (seq === readSeq.current) setCatalogue(answer.tags);
			})
			.catch((error) => {
				if (seq !== readSeq.current) return;
				/* `[]` so the render ladder cannot read an unresolved catalogue as
				   an empty one; the error branch is checked first anyway. */
				setCatalogue([]);
				setLoadError(String((error as Error).message ?? error));
			});
		// `revision` is read for the snapshot only: a revision change must not
		// re-read the catalogue.
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [open, reload]);

	useEffect(() => {
		if (revision === seenRevision.current) return;
		seenRevision.current = revision;
		revisionRef.current = revision;
		if (!open) return;
		/* The newer frame is the store's answer: drop the speculative draft and
		   whatever write was advertising itself as pending. */
		setDraft(null);
		setPending(null);
	}, [revision, open]);

	const current = draft ?? assignmentFrom(assigned, primary);
	const rows = catalogue === null ? [] : catalogueRows(catalogue, assigned, current);
	const capped = atTagCap(current);

	const submit = (next: TagAssignment, ids: readonly string[]) => {
		const seq = ++writeSeq.current;
		const target = targetRef.current;
		const startedAt = revisionRef.current;
		setWriteError("");
		setPending({ ids });
		setSessionTags(sessionId, next.tags, next.primary)
			.then((answer) => {
				if (seq !== writeSeq.current) return;
				/* The answer must be about the row this sheet is showing. A relay
				   that answers for the row it was asked about (the normal case) is
				   still checked here, so a retargeted sheet cannot paint A's
				   assignment into B even if the sequence guard were missed. */
				if (targetRef.current !== target) return;
				if (answer.session_id !== target) return;
				/* A frame newer than this write started already painted the store's
				   answer; this read-back is older than it. */
				if (revisionRef.current !== startedAt) return;
				setDraft(assignmentFrom(answer.tags, answer.primary));
			})
			.catch((error) => {
				if (seq !== writeSeq.current) return;
				/* A refusal belongs to the row that asked. After a close/reopen
				   it must be dropped, not shown under the new row's checkboxes. */
				if (targetRef.current !== target) return;
				/* Revert to the newest authoritative frame, then say why. */
				setDraft(null);
				setWriteError(
					refusalText(messageRuntime.t("relay.session.tags.save_error"), error),
				);
			})
			.finally(() => {
				if (seq === writeSeq.current) setPending(null);
			});
	};

	const toggle = (id: string) => {
		if (pending) return;
		const next = toggleAssignment(current, id);
		if (next === current) return; // the cap refused it; the row is disabled anyway
		setDraft(next);
		submit(next, [id]);
	};

	const makePrimary = (id: string) => {
		if (pending) return;
		const next = choosePrimary(current, id);
		if (next === current) return;
		setDraft(next);
		submit(next, [id]);
	};

	const clearAll = () => {
		if (pending) return;
		const next = clearedAssignment();
		const ids = current.tags;
		setConfirmClear(false);
		setDraft(next);
		submit(next, ids);
	};

	/* ONE PRE-MOUNTED LIVE REGION. A region created already populated is not
	   announced (the same rule the save-error alert follows below), and one
	   status node per row would put eight empty regions in the accessibility
	   tree. This region exists before either state changes, then reports the
	   loading or saving sentence as it becomes true. */
	const announced = pending !== null
		? messageRuntime.t("relay.session.tags.saving")
		: catalogue === null && !loadError
			? messageRuntime.t("relay.session.tags.loading")
			: "";

	return (
		<Sheet
			open={open}
			onClose={onClose}
			title={messageRuntime.t("relay.session.tags.sheet_title")}
			returnFocusRef={returnFocusRef}
		>
			<div className="flex flex-col p-2">
				<p className="sr-only" role="status" aria-live="polite">{announced}</p>
				{loadError ? (
					<div className="flex flex-col items-start gap-2 px-2 py-1">
						<p role="alert" className="text-body-sm text-danger">
							{messageRuntime.t("relay.session.tags.load_error")}
						</p>
						<button
							type="button"
							onClick={() => setReload((n) => n + 1)}
							className="min-h-11 rounded-sm border border-control bg-surface px-3 text-body-sm text-ink active:bg-accent-wash"
						>
							{messageRuntime.t("relay.session.tags.retry")}
						</button>
					</div>
				) : catalogue === null ? (
					<p className="px-2 py-1 text-body-sm text-ink-dim">
						{messageRuntime.t("relay.session.tags.loading")}
					</p>
				) : rows.length === 0 ? (
					<div className="px-2 py-1">
						<p className="text-body-sm text-ink-dim">
							{messageRuntime.t("relay.session.tags.empty")}
						</p>
						<p className="text-meta text-ink-dim">
							{messageRuntime.t("relay.session.tags.empty_hint")}
						</p>
					</div>
				) : (
					<>
						{/* The cap's REASON, on screen AND as the disabled rows'
						    description: a disabled checkbox with no explanation is a
						    control the reader cannot get past. */}
						{capped ? (
							<p
								id={limitHintId}
								className="px-2 pb-1 text-meta text-ink-dim"
							>
								{messageRuntime.t("relay.session.tags.limit_hint", {
									count: String(MAX_ASSIGNED_TAGS),
								})}
							</p>
						) : null}
						{current.tags.length > 0 ? (
							/* THE PRIMARY COLUMN'S LEGEND. The radios sit in a column of their
							   own, and without a header the isolated control reads as an
							   unexplained dot beside a row. It mirrors the row geometry exactly
							   (flex-1 label, w-11 radio column, min-w-14 status column) so the
							   header stays over the controls it names; the status cell may grow
							   under a localized saving label, and each radio keeps its own
							   `make {name} the primary tag` label, so this legend is visual
							   scaffolding and stays out of the accessibility tree. */
							<div
								className="flex items-center gap-2 px-2 pb-1"
								aria-hidden
							>
								<span className="min-w-0 flex-1" />
								<span className="flex w-11 shrink-0 justify-center text-meta text-ink-dim">
									{messageRuntime.t("relay.session.tags.primary")}
								</span>
								<span className="min-w-14 shrink-0" />
							</div>
						) : null}
						<ul className="flex flex-col">
							{rows.map((row) => {
								const blocked = !row.selected && capped;
								const rowPending = pending?.ids.includes(row.id) ?? false;
								return (
									<li key={row.id} className="flex items-center gap-2">
										<label
											className={cn(
												"flex min-h-11 min-w-0 flex-1 items-center gap-2 rounded-sm px-2",
												blocked ? "opacity-50" : "active:bg-surface",
											)}
											aria-busy={rowPending ? true : undefined}
										>
											<input
												type="checkbox"
												checked={row.selected}
												disabled={pending !== null || blocked}
												aria-describedby={blocked ? limitHintId : undefined}
												onChange={() => toggle(row.id)}
												className="size-4 shrink-0 accent-[var(--color-accent)]"
											/>
											<span className="min-w-0 truncate text-body">
												{row.name}
											</span>
											{row.missing ? (
												<span className="shrink-0 text-meta text-ink-dim">
													{messageRuntime.t("relay.session.tags.missing")}
												</span>
											) : null}
										</label>
										{/* Primary is a radio among the SELECTED rows only:
										    the relay refuses a primary outside `tags`, so
										    offering it on an unchecked row would build the
										    refusal into the surface. */}
										{/* The radio's own 16px disc is the target's visual size, but its
										    BOX is a full 44×44 touch target: a bare 16px radio is a
										    control a thumb misses, and the second column is where
										    those misses land beside a neighbour. The box is a
										    sibling of the checkbox label, never inside it — nesting
										    would make one press ambiguous between checking the row and
										    making it primary. */}
										{row.selected ? (
											<label className="flex size-11 shrink-0 items-center justify-center">
												<input
													type="radio"
													name={`session-tags-primary-${sessionId}`}
													checked={row.primary}
													disabled={pending !== null}
													aria-label={messageRuntime.t(
														"relay.session.tags.make_primary",
														{ name: row.name },
													)}
													onChange={() => makePrimary(row.id)}
													className="size-4 accent-[var(--color-accent)]"
												/>
											</label>
										) : (
											<span className="size-11 shrink-0" aria-hidden />
										)}
										{/* The pending label is LOCALIZED, so its width is not English's:
										    "enregistrement…" (fr) measures ~89px at text-meta, Hindi and
										    Russian ~80px, Spanish ~70px against English's 46px. A fixed
										    56px cell spills left across the 44px radio column; min-w-14 is
										    a floor and whitespace-nowrap lets the cell take its max-content
										    width while the flex-1 tag label absorbs the difference. */}
										<span className="min-w-14 shrink-0 whitespace-nowrap text-right text-meta text-ink-dim">
											{rowPending
												? messageRuntime.t("relay.session.tags.saving")
												: ""}
										</span>
									</li>
								);
							})}
						</ul>
					</>
				)}
				<div className="flex items-center gap-2 px-2 pt-2">
					{current.tags.length > 0 ? (
						confirmClear ? (
							<>
								<button
									type="button"
									onClick={clearAll}
									disabled={pending !== null}
									className="min-h-11 rounded-sm border border-control bg-surface px-3 text-body-sm text-danger active:bg-accent-wash disabled:opacity-50"
								>
									{messageRuntime.t("relay.session.tags.remove_all_confirm")}
								</button>
								<button
									type="button"
									onClick={() => setConfirmClear(false)}
									className="min-h-11 rounded-sm px-3 text-body-sm text-ink-muted active:bg-surface"
								>
									{messageRuntime.t("relay.session.tags.cancel")}
								</button>
							</>
						) : (
							<button
								type="button"
								onClick={() => setConfirmClear(true)}
								disabled={pending !== null}
								className="min-h-11 rounded-sm px-3 text-body-sm text-ink-muted active:bg-surface disabled:opacity-50"
							>
								{messageRuntime.t("relay.session.tags.remove_all")}
							</button>
						)
					) : null}
					<button
						type="button"
						onClick={onClose}
						className="ml-auto min-h-11 rounded-sm border border-control bg-surface px-4 text-body-sm text-ink active:bg-accent-wash"
					>
						{messageRuntime.t("relay.session.tags.done")}
					</button>
				</div>
				{/* THE SHEET'S LIVE REGION, mounted empty and filled on the commit
				    after a refusal — a container that arrives populated is not
				    announced. The lead is ours; the rest is the relay's own sentence
				    (`lib/refusal`), which is written for this reader. */}
				<p role="alert" className="px-2 pb-1 text-body-sm break-words text-danger">
					{writeError}
				</p>
			</div>
		</Sheet>
	);
}

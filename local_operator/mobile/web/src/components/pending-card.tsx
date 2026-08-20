/**
 * Pending request card — pinned above the composer, accent-bordered, the
 * most prominent element on screen (branding §7: a question for the user is
 * the only thing that needs a decision, and it must be unmissable).
 *
 * Approval: tool name + detail + Approve/Deny (+ remember). Ask: the
 * question plus options as tap targets, or a free-text input when the
 * daemon offered no options.
 */
import { useState } from "react";
import { sendCommand } from "../api";
import { cn } from "../lib/cn";
import type { PendingRequest } from "../types";

export function PendingCard({
	pid,
	pending,
}: {
	pid: number;
	pending: PendingRequest;
}) {
	const [remember, setRemember] = useState(false);
	const [freeText, setFreeText] = useState("");
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState("");

	const answer = async (fn: () => Promise<unknown>) => {
		if (busy) return;
		setBusy(true);
		setError("");
		try {
			await fn();
		} catch (e) {
			setError(String((e as Error).message ?? e));
			setBusy(false);
		}
		/* On success the next projection repaint clears `pending`, which
		   unmounts this card — no local reset needed. */
	};

	const approve = (approved: boolean) =>
		answer(() =>
			sendCommand(pid, {
				op: "approval_answer",
				request_id: pending.request_id,
				approved,
				remember,
			}),
		);

	const answerAsk = (value: string) =>
		answer(() =>
			sendCommand(pid, {
				op: "ask_answer",
				request_id: pending.request_id,
				value,
			}),
		);

	return (
		<div className="border-accent bg-accent-wash mx-3 flex flex-col gap-3 rounded-lg border p-3">
			<div className="flex flex-col gap-1">
				<span className="text-meta text-accent">
					{pending.kind === "approval"
						? "approval needed"
						: "question"}
				</span>
				<span className="text-heading">{pending.title}</span>
				{pending.detail ? (
					<p className="text-body-sm text-ink-muted whitespace-pre-wrap">
						{pending.detail}
					</p>
				) : null}
			</div>

			{pending.kind === "approval" ? (
				<>
					<label className="flex min-h-11 items-center gap-2 text-body-sm text-ink-muted select-none">
						<input
							type="checkbox"
							checked={remember}
							onChange={(e) => setRemember(e.target.checked)}
							className="size-4 accent-accent"
						/>
						remember this choice
					</label>
					<div className="flex gap-2">
						<button
							type="button"
							disabled={busy}
							onClick={() => approve(true)}
							className="flex min-h-11 flex-1 items-center justify-center rounded-sm bg-accent text-body-sm font-medium text-on-accent active:bg-accent-active"
						>
							approve
						</button>
						<button
							type="button"
							disabled={busy}
							onClick={() => approve(false)}
							className="flex min-h-11 flex-1 items-center justify-center rounded-sm border border-danger-border bg-danger-wash text-body-sm text-danger active:opacity-80"
						>
							deny
						</button>
					</div>
				</>
			) : pending.options.length > 0 ? (
				<div className="flex flex-col gap-1">
					{pending.options.map((opt) => (
						<button
							key={opt}
							type="button"
							disabled={busy}
							onClick={() => answerAsk(opt)}
							className={cn(
								"flex min-h-11 items-center rounded-sm border border-control bg-surface px-3 text-left text-body-sm text-ink active:bg-elevated",
							)}
						>
							{opt}
						</button>
					))}
				</div>
			) : (
				<div className="flex gap-2">
					<input
						value={freeText}
						onChange={(e) => setFreeText(e.target.value)}
						placeholder="your answer"
						className="min-h-11 min-w-0 flex-1 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim"
					/>
					<button
						type="button"
						disabled={busy || !freeText.trim()}
						onClick={() => answerAsk(freeText.trim())}
						className="flex min-h-11 items-center justify-center rounded-sm bg-accent px-4 text-body-sm font-medium text-on-accent active:bg-accent-active"
					>
						send
					</button>
				</div>
			)}

			{error ? <p className="text-body-sm text-danger">{error}</p> : null}
		</div>
	);
}

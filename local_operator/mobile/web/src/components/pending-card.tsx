/**
 * Pending request card — pinned above the composer, accent-bordered, the
 * most prominent element on screen (branding §7: a question for the user is
 * the only thing that needs a decision, and it must be unmissable).
 *
 * Approval: tool name + detail + Approve/Deny (+ remember). Ask: the
 * question plus options as tap targets (each with its consequence line, U3),
 * or a paste field — masked when the ask is a secret credential (D1/U2) —
 * when the daemon offered no options. A multi-question ask shows a
 * "Question N of M" header and re-renders the next question after each answer
 * (U1); the card keys off `request_id`+`question_index` so React remounts the
 * input between questions rather than carrying a stale draft forward.
 */
import { useState } from "react";
import { sendCommand } from "../api";
import { cn } from "../lib/cn";
import type { PendingRequest } from "../types";

/** Turn a raw command error into copy a phone user can act on. The daemon and
    registrant speak in developer terms (HTTP status strings, "session not
    connected"); a person staring at a phone needs the human version (U4/U7). */
function humanizeError(message: string): string {
	const m = message.toLowerCase();
	if (m.includes("already answered")) {
		/* Stale tap: the question settled on another surface first (U4). */
		return "Already answered — this question was settled on the terminal.";
	}
	if (m.includes("no longer waiting")) {
		return "Already answered — this question is no longer waiting.";
	}
	if (m.includes("not connected") || m.includes("409")) {
		return "The terminal session went away — reopen it to answer.";
	}
	if (m.includes("did not answer") || m.includes("504")) {
		return "The session didn’t respond in time — try again.";
	}
	return message;
}

export function PendingCard({
	pid,
	pending,
	count = 1,
}: {
	pid: number;
	pending: PendingRequest;
	/** Total requests waiting, including this one. A parallel tool batch can
	    open several approvals at once; when >1 the card shows a "1 of N" badge
	    so the user knows more follow, and answering this one reveals the next
	    on the repaint. */
	count?: number;
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
			setError(humanizeError(String((e as Error).message ?? e)));
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

	/* A multi-question ask advances one question at a time (U1): show which
	   question this is so the user knows the card is not the whole prompt. */
	const multiQuestion = pending.question_total > 1;

	return (
		<div className="border-accent bg-accent-wash mx-2 flex flex-col gap-2 rounded-md border p-2.5">
			<div className="flex flex-col gap-0.5">
				<span className="flex items-center justify-between text-meta text-accent">
					<span>
						{pending.kind === "approval"
							? "approval needed"
							: pending.secret
								? "secret requested"
								: "question"}
					</span>
					{multiQuestion ? (
						<span className="font-mono text-mono-sm text-ink-dim">
							Question {pending.question_index + 1} of{" "}
							{pending.question_total}
						</span>
					) : count > 1 ? (
						<span className="font-mono text-mono-sm text-ink-dim">
							1 of {count}
						</span>
					) : null}
				</span>
				<span className="text-body font-medium">{pending.title}</span>
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
							className="flex min-h-11 flex-1 items-center justify-center rounded-sm bg-accent text-body-sm font-medium text-on-accent active:bg-accent-active disabled:opacity-50"
						>
							{busy ? "…" : "approve"}
						</button>
						<button
							type="button"
							disabled={busy}
							onClick={() => approve(false)}
							className="flex min-h-11 flex-1 items-center justify-center rounded-sm border border-danger-border bg-danger-wash text-body-sm text-danger active:bg-danger-wash disabled:opacity-50"
						>
							{busy ? "…" : "deny"}
						</button>
					</div>
				</>
			) : pending.options.length > 0 ? (
				<div className="flex flex-col gap-2">
					{pending.options.map((opt) => (
						<button
							key={opt.label}
							type="button"
							disabled={busy}
							onClick={() => answerAsk(opt.label)}
							/* Accent-tinted left edge + elevated fill so an option
							   reads as a tap target, not a static label or a text
							   field (D2). Disabled dims (D3/D4). */
							className={cn(
								"flex min-h-11 flex-col justify-center rounded-sm border border-l-2 border-control border-l-accent bg-elevated px-3 py-2 text-left active:bg-accent-wash disabled:opacity-50",
							)}
						>
							<span className="text-body-sm font-medium text-ink">
								{opt.label}
							</span>
							{opt.description ? (
								<span className="text-body-sm text-ink-muted">
									{opt.description}
								</span>
							) : null}
						</button>
					))}
				</div>
			) : (
				<div className="flex flex-col gap-1">
					<div className="flex gap-2">
						<input
							/* Remount the field between questions of a multi-part
							   ask so a typed draft never carries forward (U1). */
							key={`${pending.request_id}:${pending.question_index}`}
							value={freeText}
							onChange={(e) => setFreeText(e.target.value)}
							/* Secret asks are credentials: mask the value on screen
							   and suppress the keyboard's learn/suggest so a token is
							   not shoulder-surfable or captured (D1/U2). */
							type={pending.secret ? "password" : "text"}
							autoComplete={pending.secret ? "off" : undefined}
							autoCapitalize={pending.secret ? "none" : undefined}
							autoCorrect={pending.secret ? "off" : undefined}
							spellCheck={pending.secret ? false : undefined}
							placeholder={pending.secret ? "paste secret" : "your answer"}
							className="min-h-11 min-w-0 flex-1 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim"
						/>
						<button
							type="button"
							disabled={busy || !freeText.trim()}
							onClick={() => answerAsk(freeText.trim())}
							className="flex min-h-11 items-center justify-center rounded-sm bg-accent px-4 text-body-sm font-medium text-on-accent active:bg-accent-active disabled:opacity-50"
						>
							{busy ? "…" : "send"}
						</button>
					</div>
					{pending.secret ? (
						<p className="text-meta text-ink-dim">
							secret — sent directly, not shown in the transcript
						</p>
					) : null}
				</div>
			)}

			{error ? <p className="text-body-sm text-danger">{error}</p> : null}
		</div>
	);
}

/**
 * Ask response row — the transcript card for a queued ask SETTLING
 * (`ask_response`) and for its deadline passing (`ask_timeout`), design §4/§5.
 *
 * WHY A CARD AND NOT A NOTICE LINE. Both kinds land in the same place the
 * notices do, and both are one line at rest — but they are the only rows in the
 * transcript that carry the structured Q&A that produced them, and the design's
 * rule is that no surface re-derives Q&A from the sentence (`asks/render.py`).
 * So the disclosure shows the questions and the answers the runtime recorded,
 * which is also what makes a LATE answer readable: the reader can see what they
 * answered and what the agent had already been told.
 *
 * ONE LINE AT REST (§7.4), and the line is the runtime's own — `entry.text` is
 * the shared `harness/rows.py` copy ("Answered — delivering (ask …)", "Timed out
 * after 42m — the agent moved on; you can still answer (ask …)"), the same
 * sentence the TUI prints. Re-authoring it here would be a second wording of one
 * rule, which is exactly what that shared function exists to prevent.
 *
 * THE DISCLOSURE IS THE RECORD. `ask_response` expands to the question/answer
 * pairs; `ask_timeout` expands to the notice the MODEL was given, labelled as
 * such — "Proceed without it: use your recommended option" is an instruction to
 * the agent, not to the reader, and a user who saw it unlabelled would read it
 * as advice to themselves.
 */
import { useState } from "react";
import { cn } from "../lib/cn";
import type { TranscriptEntry } from "../types";

const GLYPH: Record<string, string> = {
	answered: "✓",
	late: "✓",
	declined: "✕",
	timed_out: "!",
};

export function AskRow({ entry }: { entry: TranscriptEntry }) {
	const [open, setOpen] = useState(false);
	const timeout = entry.kind === "ask_timeout";
	const status = String(entry.details.status || (timeout ? "timed_out" : "answered"));
	const pairs = pairAnswers(entry);
	const detail = timeout ? String(entry.details.text || "") : "";
	const hasDetails = pairs.length > 0 || detail !== "";
	const tone =
		entry.details.severity === "warning"
			? "text-warning"
			: status === "declined"
				? "text-ink-muted"
				: "text-success";

	return (
		<div
			data-testid="ask-row"
			data-ask-id={entry.details.ask_id}
			data-ask-status={status}
			className="rounded-sm bg-surface px-1.5"
		>
			<button
				type="button"
				onClick={() => hasDetails && setOpen(!open)}
				className="flex min-h-11 w-full items-center gap-1.5 text-left select-none"
			>
				<span aria-hidden className={cn("w-4 shrink-0 text-center font-mono text-mono-sm", tone)}>
					{GLYPH[status] ?? "•"}
				</span>
				<span className="min-w-0 flex-1 text-body-sm text-ink-muted">{entry.text}</span>
				{hasDetails ? (
					<span aria-hidden className="shrink-0 text-ink-dim">
						{open ? "▾" : "▸"}
					</span>
				) : null}
			</button>
			{open ? (
				<div className="flex flex-col gap-1.5 pb-1.5 pl-5">
					{pairs.length > 0 ? (
						<dl className="flex flex-col gap-1.5">
							{pairs.map((pair, index) => (
								<div key={index} className="flex flex-col">
									<dt className="text-body-sm text-ink-muted">{pair.question}</dt>
									<dd className="text-body-sm text-ink">{pair.answer || "—"}</dd>
								</div>
							))}
						</dl>
					) : null}
					{detail ? (
						<div className="flex flex-col gap-1">
							<span className="text-meta text-ink-dim">what the agent was told</span>
							<p className="text-body-sm text-ink-muted whitespace-pre-wrap break-words">
								{detail}
							</p>
						</div>
					) : null}
				</div>
			) : null}
		</div>
	);
}

/** The row's structured Q&A, as `ask_response` details carry it.
 *
 *  The answer map holds a secret answer as the KEY the runtime stored, never the
 *  value — so a secret's row shows the key name, which is the strongest honest
 *  statement this surface can make about it. */
function pairAnswers(entry: TranscriptEntry): { question: string; answer: string }[] {
	const questions = Array.isArray(entry.details.questions) ? entry.details.questions : [];
	const answers = entry.details.answers ?? {};
	return questions.map((question) => {
		const chosen = answers[String(question?.id || "")];
		const labels = Array.isArray(chosen) ? chosen.map((value) => String(value)) : [];
		return {
			question: String(question?.question || ""),
			answer: labels
				.map((label) => {
					const option = (question?.options ?? []).find((opt) => opt.label === label);
					return option?.description ? `${label} — ${option.description}` : label;
				})
				.join(", "),
		};
	});
}

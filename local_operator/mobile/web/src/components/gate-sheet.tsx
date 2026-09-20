/**
 * Gate sheet — the session view's `/approvals` surface (stage D of issue #1310).
 *
 * WHY THIS EXISTS ON THE PHONE AT ALL, and what it replaces. Before the redesign,
 * a phone could not loosen a session's approval gate in ANY session: the authority
 * was the spawn capability and the relay never has one, so `/approvals auto` was
 * refused everywhere on this surface and the refusal copy sent the reader to "the
 * window that started this session" — a window a phone is not and cannot become.
 * The redesign makes authority a device signature, so the phone can do it, and this
 * is the control that reaches it.
 *
 * TWO DIRECTIONS, AND ONLY ONE OF THEM IS SIGNED:
 *
 *   * TIGHTENING (`ask`) is ordinary — it makes the session stricter — so it needs
 *     no proof and works from any surface, including one whose device is not
 *     paired. Refusing it here would wall a phone off from the safe direction.
 *   * LOOSENING (`auto`) is authority-increasing, so it asks the runtime for a
 *     per-action challenge and signs it. The signature is what costs a human
 *     gesture on the MACHINE's key, or on this phone's non-extractable one.
 *
 * THE `slash_result` OP, not `slash`. `slash` is the off-terminal subset
 * (`/goal`, `/compact`) and answers `/approvals` with "terminal-only here"; the
 * routed op is the one the runtime's authority seam was built for and the one the
 * desktop backend and the TUI's attached pane already use. Taking the same road
 * rather than a second one is the point.
 */
import { useState } from "react";
import { requestOperatorChallenge, sendCommandWithProof } from "../api";
import { Button } from "./ui/button";
import { Sheet } from "./ui/sheet";
import { NotPairedError, operatorFieldsFor } from "../lib/operator-device";
import { PairPromptSheet } from "../screens/pair";

type Mode = "ask" | "auto";

export function GateSheet({
	open,
	onClose,
	sessionId,
}: {
	open: boolean;
	onClose: () => void;
	sessionId: string;
}) {
	const [busy, setBusy] = useState<Mode | null>(null);
	const [error, setError] = useState("");
	const [notice, setNotice] = useState("");
	const [pairPrompt, setPairPrompt] = useState(false);

	async function choose(mode: Mode) {
		if (busy) return;
		setBusy(mode);
		setError("");
		setNotice("");
		try {
			if (mode === "ask") {
				/* Ordinary: no proof, no prompt, and it works from a phone whose
				   device is not paired — which is why it is the branch that is tried
				   first and never gated behind pairing. */
				await sendCommandWithProof(sessionId, {
					op: "slash_result",
					command: "approvals",
					args: "ask",
					images: [],
				});
				setNotice("keeping approvals at ask in this session");
				return;
			}
			await loosen();
		} catch (failure) {
			setError(humanizeGateError(failure));
		} finally {
			setBusy(null);
		}
	}

	async function loosen() {
		try {
			await sendLoosened();
			setNotice("this session's gate is now auto — gated tools run without asking");
			onClose();
		} catch (failure) {
			if (failure instanceof NotPairedError) {
				/* The remedy the refusal copy names, one tap away rather than a
				   sentence telling the reader to go and find a terminal. */
				setPairPrompt(true);
				setError(
					"This phone is not paired with that machine yet — pair it once and it can give this consent itself.",
				);
				return;
			}
			setError(humanizeGateError(failure));
		}
	}

	async function sendLoosened() {
		const challenge = await requestOperatorChallenge(sessionId, {
			action: "loosen",
			request_id: "",
		});
		const fields = await operatorFieldsFor({
			action: "loosen",
			sessionId,
			requestId: "",
			challenge: challenge.challenge,
		});
		await sendCommandWithProof(sessionId, {
			op: "slash_result",
			command: "approvals",
			args: "auto",
			images: [],
			...fields,
		});
	}

	return (
		<>
			<Sheet open={open} onClose={onClose} title="Approvals in this session">
				<div className="flex flex-col gap-3 p-3">
					<p className="text-body-sm text-ink-muted">
						Ask parks every gated tool call until you answer its card. Auto lets them run
						without asking for the rest of this session.
					</p>
					<Button disabled={busy !== null} onClick={() => void choose("ask")}>
						{busy === "ask" ? "…" : "keep asking (ask)"}
					</Button>
					<Button disabled={busy !== null} onClick={() => void choose("auto")}>
						{busy === "auto" ? "…" : "run without asking (auto)"}
					</Button>
					{notice ? <p className="text-body-sm text-ink-muted">{notice}</p> : null}
					{error ? <p className="text-body-sm text-danger">{error}</p> : null}
				</div>
			</Sheet>
			<PairPromptSheet open={pairPrompt} onClose={() => setPairPrompt(false)} />
		</>
	);
}

/** The daemon's and the runtime's words, turned into something a phone can act on.
    The refusal sentences come from the runtime and are already written for this
    reader (design round 2 U8), so they pass through rather than being re-worded
    here — a second copy of one refusal is how two surfaces end up describing the
    same rule differently. */
export function humanizeGateError(error: unknown): string {
	const message = String((error as Error)?.message ?? error);
	if (message.includes("504") || message.toLowerCase().includes("did not answer")) {
		return "The session didn’t respond in time — try again.";
	}
	if (message.includes("409") || message.toLowerCase().includes("not connected")) {
		return "The session went away — reopen it from the session list.";
	}
	return message;
}

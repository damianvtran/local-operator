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
import {
	HttpError,
	isAuthorityRefusalCode,
	requestOperatorChallenge,
	sendCommandWithProof,
} from "../api";
import { Button } from "./ui/button";
import { Sheet } from "./ui/sheet";
import { NotPairedError, operatorFieldsFor, storedCertificate } from "../lib/operator-device";
import { PairPromptSheet } from "../screens/pair";

type Mode = "ask" | "auto";

export function GateSheet({
	open,
	onClose,
	sessionId,
	onReceipt,
}: {
	open: boolean;
	onClose: () => void;
	sessionId: string;
	/** Where a SUCCESS report goes so it outlives this sheet.

	    WHY IT IS A PROP AND NOT `notice` (design round 6, D2). The success path used
	    to `setNotice(...)` and then `onClose()` in the same commit — and the notice
	    is rendered INSIDE the `Sheet`, which returns null when closed — so the
	    receipt was set and unmounted before a frame was painted. Measured: the panel
	    gone from the DOM, the paragraph list empty, the header still reading "needs
	    you", and the user told nothing about whether the tap worked. The sheet is the
	    wrong owner for a receipt about something that already happened; the surface
	    it closes back onto is the right one. */
	onReceipt: (text: string) => void;
}) {
	const [busy, setBusy] = useState<Mode | null>(null);
	const [error, setError] = useState("");
	const [notice, setNotice] = useState("");
	const [pairPrompt, setPairPrompt] = useState(false);
	/* Read ONCE per render, synchronously, from storage this phone already has:
	   the point is to say what will be refused BEFORE the tap rather than after
	   (design round 6, D4). No async probe — `ask` must stay usable while this is
	   unknown, and a device that is paired but whose certificate the machine has
	   revoked still passes this check, which is why it HEADS THE DECISION rather
	   than replaces the refusal. */
	const paired = storedCertificate() !== null;

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
			// OUT FIRST, THEN CLOSE: the receipt has to exist somewhere that is not
			// this sheet before the sheet unmounts (design round 6, D2). The
			// tightening path keeps its in-sheet notice because it does NOT close,
			// which is why that one never had this bug.
			onReceipt("this session's gate is now auto — gated tools run without asking");
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
					{!paired ? (
						/* SAID BEFORE THE TAP, not after (design round 6, D4). Both buttons
						   rendered identically — same size, same weight, no state — while
						   only one of them can fail here, and the refusal and the pairing
						   prompt arrived only once it had been pressed. `ask` works unpaired
						   by design (the gate sheet's own comment), so the asymmetry is the
						   honest thing to state; `storedCertificate()` is a synchronous read
						   this component already has. */
						<p className="text-meta text-ink-muted">
							This phone is not paired with that machine, so <strong>run without asking</strong>{" "}
							will ask you to pair it first. Keeping approvals at <strong>ask</strong> never needs
							a pairing.
						</p>
					) : null}
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
	if (error instanceof HttpError && isAuthorityRefusalCode(error.code)) {
		if (error.code === "operator_authority_unconfigured") {
			/* THE RUNTIME'S OWN COPY, and nothing added: on this host it already
			   names `lop operator install`, which is the command that unlocks both
			   remedies it mentions. Appending a second copy is how two surfaces end
			   up describing one rule differently (UX round 6, U1/U2). */
			return error.message;
		}
		if (error.code === "operator_authority_required") {
			/* THE READER IS ALREADY ON THE PAIRED PHONE (design round 6, D5). The
			   runtime's sentence ends "authorise it from this machine (Touch ID) or
			   from your paired phone" — and this surface IS the paired phone, having
			   just used it. Passing the sentence through unchanged is right (the phone
			   must not reword a refusal it does not own), but the one step that is
			   actually left is knowable here and nowhere else: the machine refused a
			   signature from a device that believed it was paired. */
			return `${error.message} (This phone's signature was refused — on that machine, \`lop operator devices\` shows whether this device is revoked or expired.)`;
		}
	}
	const message = String((error as Error)?.message ?? error);
	if (message.includes("504") || message.toLowerCase().includes("did not answer")) {
		return "The session didn’t respond in time — try again.";
	}
	if (message.includes("409") || message.toLowerCase().includes("not connected")) {
		return "The session went away — reopen it from the session list.";
	}
	return message;
}

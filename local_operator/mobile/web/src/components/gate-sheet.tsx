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
 *   * LOOSENING (`auto`, and the global `default auto`) is authority-increasing,
 *     so it asks the runtime for a per-action challenge and signs it. The
 *     signature is what costs a human gesture on the MACHINE's key, or on this
 *     phone's non-extractable one.
 *
 * THE THIRD CHOICE IS A SCOPE, NOT A THIRD DIRECTION. `default auto` loosens the
 * machine's GLOBAL default — every session that follows it — rather than this
 * one. It shares `auto`'s direction and therefore its proof, so it must be
 * visually subordinate to the two session-scoped choices (this is the scope the
 * user is least likely to mean by accident, and the one whose blast radius a
 * glance has to make legible); it is a `quiet` button for that reason, not a
 * second primary. Its receipt is the runtime's own account of what it applied
 * (`lib/approval-policy`), because a phone cannot see the set of sessions a
 * global write reached and must not claim one.
 *
 * THE `slash_result` OP, not `slash`. `slash` is the off-terminal subset
 * (`/goal`, `/compact`) and answers `/approvals` with "terminal-only here"; the
 * routed op is the one the runtime's authority seam was built for and the one the
 * desktop backend and the TUI's attached pane already use. Taking the same road
 * rather than a second one is the point.
 */
import { useState } from "react";
import { HttpError, isAuthorityRefusalCode, sendCommandWithProof } from "../api";
import { Button } from "./ui/button";
import { Sheet } from "./ui/sheet";
import { NotPairedError, storedCertificate } from "../lib/operator-device";
import {
	approvalPolicyLine,
	globalDefaultReceipt,
	setApprovalsPolicy,
} from "../lib/approval-policy";
import { messageRuntime } from "../store";
import { PairPromptSheet } from "../screens/pair";
import type { ApprovalMode, ApprovalProvenance } from "../types";

type Mode = "ask" | "auto" | "default auto";

export function GateSheet({
	open,
	onClose,
	sessionId,
	onReceipt,
	approvalMode,
	approvalProvenance,
	approvalGlobalMode,
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
	/** The mode IN FORCE for this conversation and where it came from, straight
	    off the projection's pending approval (`types.PendingRequest`). Both are
	    optional because an owner older than these fields sends neither, and the
	    sheet then omits the line rather than guessing one — the sheet's copy
	    already explains the two directions, and a state line is only worth a row
	    when it is the runtime's answer. */
	approvalMode?: ApprovalMode | "";
	approvalProvenance?: ApprovalProvenance | "";
	/** The machine default as the runtime resolved it, so a session override can
	    name the scope it overrides (U4). Absent on an older owner, in which case
	    `approvalPolicyLine` drops that clause rather than negating this
	    conversation's mode into a claim about every other one. */
	approvalGlobalMode?: ApprovalMode | "";
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
	const policyLine = approvalPolicyLine(
		{
			approval_mode: approvalMode,
			approval_provenance: approvalProvenance,
			approval_global_mode: approvalGlobalMode,
		},
		messageRuntime.t,
	);

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
				setNotice(messageRuntime.t("relay.asks.gate.notice_ask"));
				/* AND THE LOOSENING RECEIPT GOES (UX round 8, U8-3). It lives on the
				   header, above this sheet, and it reports a STATE ("the gate is now
				   auto — gated tools run without asking"); a tighten from this very
				   sheet leaves it asserting something the gate no longer is, one frame
				   after the sheet says otherwise. Clearing it here is what makes the
				   round-6 comment's claim ("the next gesture clears it") true, rather
				   than a claim about code that did not exist. */
				onReceipt("");
				return;
			}
			await loosen(mode);
		} catch (failure) {
			setError(humanizeGateError(failure));
		} finally {
			setBusy(null);
		}
	}

	async function loosen(mode: Exclude<Mode, "ask">) {
		try {
			const receipt = await setApprovalsPolicy(sessionId, mode);
			if (receipt.refused) {
				/* A ROUTED SLASH OUTCOME CAN COMPLETE WITHOUT DOING WHAT IT WAS
				   ASKED (the daemon reports `refused` beside `ok` for exactly that).
				   Closing on it would put a receipt on the header describing a
				   change the runtime says did not happen, so the refusal stays in
				   THIS sheet, in the runtime's own words, with the choices still
				   there for the reader to take another way. */
				setError(receipt.detail.trim() || messageRuntime.t("relay.approvals.gate.refused_nothing_changed"));
				return;
			}
			// OUT FIRST, THEN CLOSE: the receipt has to exist somewhere that is not
			// this sheet before the sheet unmounts (design round 6, D2). The
			// tightening path keeps its in-sheet notice because it does NOT close,
			// which is why that one never had this bug.
			onReceipt(
				mode === "auto"
					? messageRuntime.t("relay.asks.gate.notice_auto")
					: globalDefaultReceipt(receipt, messageRuntime.t),
			);
			onClose();
		} catch (failure) {
			if (failure instanceof NotPairedError) {
				/* The remedy the refusal copy names, one tap away rather than a
				   sentence telling the reader to go and find a terminal. */
				setPairPrompt(true);
				setError(messageRuntime.t("relay.asks.gate.not_paired_error"));
				return;
			}
			setError(humanizeGateError(failure));
		}
	}

	return (
		<>
			<Sheet open={open} onClose={onClose} title={messageRuntime.t("relay.asks.gate.title")}>
				<div className="flex flex-col gap-3 p-3">
					{policyLine ? (
						/* THE STATE, BEFORE THE ACTIONS. Every other line here says what a
						   choice WOULD do; this one says what the gate IS right now, in the
						   runtime's terms, so the reader can tell an override from the
						   machine default before pressing either. Omitted when the owner
						   sends no provenance — see the prop's note. */
						<p className="text-meta text-ink-muted">{policyLine}</p>
					) : null}
					<p className="text-body-sm text-ink-muted">
						{messageRuntime.t("relay.asks.gate.body")}{" "}
						{messageRuntime.t("relay.approvals.gate.body_scope")}
					</p>
					<Button disabled={busy !== null} onClick={() => void choose("ask")}>
						{busy === "ask" ? "…" : messageRuntime.t("relay.asks.gate.keep_asking")}
					</Button>
					<Button disabled={busy !== null} onClick={() => void choose("auto")}>
						{busy === "auto" ? "…" : messageRuntime.t("relay.asks.gate.run_without")}
					</Button>
					{/* THE SCOPE-CHANGING CHOICE, and the reason it is `quiet`: the two
					    buttons above act on the session in front of the reader, this one
					    acts on every session that follows the default. Same direction
					    (`auto`), different blast radius — so it is subordinate in weight
					    while still a full-height target of its own, and its sentence says
					    which scope it is BEFORE the tap rather than in a receipt after
					    it. */}
					<Button
						variant="quiet"
						disabled={busy !== null}
						onClick={() => void choose("default auto")}
					>
						{busy === "default auto" ? "…" : messageRuntime.t("relay.approvals.gate.default_auto")}
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
							{messageRuntime.t("relay.asks.gate.unpaired_before")}{" "}
							<strong>{messageRuntime.t("relay.asks.gate.unpaired_run_label")}</strong>{" "}
							{messageRuntime.t("relay.asks.gate.unpaired_middle")}{" "}
							<strong>{messageRuntime.t("relay.asks.gate.unpaired_ask_label")}</strong>{" "}
							{messageRuntime.t("relay.asks.gate.unpaired_after")}{" "}
							{messageRuntime.t("relay.approvals.gate.unpaired_default_auto")}
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
			return messageRuntime.t("relay.asks.gate.operator_refused", { message: error.message });
		}
	}
	const message = String((error as Error)?.message ?? error);
	if (message.includes("504") || message.toLowerCase().includes("did not answer")) {
		return messageRuntime.t("relay.asks.error.session_timeout");
	}
	if (message.includes("409") || message.toLowerCase().includes("not connected")) {
		return messageRuntime.t("relay.asks.error.session_gone_list");
	}
	return message;
}

/**
 * The one road to `/approvals` from the phone.
 *
 * TWO SURFACES NOW TAKE IT, so the sequence lives here once. The gate sheet is a
 * deliberate visit ("change this session's gate"); the pending card's two links
 * change the same policy in the middle of a decision and must NOT answer the card
 * they sit on. Both send the identical frame family, and the proof protocol is not
 * decoration on top of it: the runtime mints a per-action challenge, this phone
 * signs it with a non-extractable key, and the signature is what carries the
 * authority. A second copy of that sequence is how one surface ends up signing for
 * the wrong action, or skipping the challenge behind a button that says it needs
 * one — and the failure would read as "the phone's consent stopped working" on
 * whichever surface drifted. So the sequence is here; the callers differ only in
 * which SCOPE they ask for and what they do with the answer.
 *
 * WHICH SCOPES SIGN, and why the asymmetry is reproduced rather than re-decided:
 *
 *   * `ask` — tighten this session. Ordinary, needs no proof.
 *   * `auto` — loosen this session. Authority-increasing: `loosen` challenge.
 *   * `default auto` — loosen the GLOBAL default, which applies to every session
 *     following it. Also authority-increasing, and it takes the SAME challenge
 *     action (`loosen`) and the same proof fields. The signature is bound to
 *     that action and to the request id (empty on both phone scopes), NOT to
 *     the scope: before its challenge is consumed, a signature minted for one
 *     of these two commands would be accepted on the other. That is deliberate
 *     — both are the same authority-increasing loosening gesture, and the
 *     challenge is single-use — but it means the signature proves the operator
 *     consented to A loosening, not WHICH scope was chosen. The scope is the
 *     frame's claim, and the receipt reports what the runtime actually did.
 *   * `default ask` — tighten globally, no proof. Not offered by any phone
 *     surface today; it is left out of the union on purpose so a caller cannot
 *     land an unproven global write behind a typo.
 */
import { requestOperatorChallenge, sendCommandWithProof, type CommandReceipt } from "../api";
import type { Translator } from "../i18n/keys.gen";
import type { ApprovalMode, ApprovalProvenance } from "../types";
import { operatorFieldsFor } from "./operator-device";

/** The scopes a phone surface may ask for, in the runtime's spelling. */
export type ApprovalsPolicy = "ask" | "auto" | "default auto";

/** Whether this scope can only be taken with this phone's signature.

    Exported because the surfaces say what will be refused BEFORE the tap
    (the unpaired note), and that note has to cover exactly the choices this
    predicate answers true for — one list, not two. */
export function isAuthorityIncreasing(
	policy: ApprovalsPolicy,
): policy is Exclude<ApprovalsPolicy, "ask"> {
	return policy !== "ask";
}

/**
 * Send one `/approvals` scope and hand back the runtime's receipt.
 *
 * `NotPairedError` from the signing step is left to propagate: both callers
 * already turn it into the pairing prompt rather than a sentence telling the
 * reader to go and find a terminal, and a signature this build cannot produce is
 * a refusal the runtime never saw.
 */
export function setApprovalsPolicy(
	sessionId: string,
	policy: ApprovalsPolicy,
): Promise<CommandReceipt> {
	if (!isAuthorityIncreasing(policy)) {
		return sendCommandWithProof(sessionId, {
			op: "slash_result",
			command: "approvals",
			args: policy,
			images: [],
		});
	}
	return signAndSend(sessionId, policy);
}

async function signAndSend(
	sessionId: string,
	policy: Exclude<ApprovalsPolicy, "ask">,
): Promise<CommandReceipt> {
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
	return sendCommandWithProof(sessionId, {
		op: "slash_result",
		command: "approvals",
		args: policy,
		images: [],
		...fields,
	});
}

/**
 * What the header says after a GLOBAL default write, given the runtime's receipt.
 *
 * THE RUNTIME'S SENTENCE WINS WHEN IT HAS ONE. `default auto` applies to the set
 * of sessions that follow the default, and that set is the runtime's to count —
 * a phone that composed its own "all sessions are now auto" would be claiming a
 * scope it cannot see, and the one case that matters (a runtime that applied it
 * to fewer sessions than the phone assumed) is exactly where the claim would be
 * wrong. So a non-empty `detail` is passed through unchanged, the same rule the
 * refusal copy follows on this surface.
 *
 * THE FALLBACK NAMES THE DEFAULT, NOT "ALL SESSIONS". It is used only when the
 * receipt carries no prose, and it deliberately does not say "all sessions": a
 * conversation with its own choice keeps it (the runtime's own receipt says so),
 * so "all" would be a claim neither the runtime nor this phone can make.
 */
export function globalDefaultReceipt(receipt: CommandReceipt, t: Translator): string {
	const said = receipt.detail.trim();
	if (said) return said;
	return t("relay.approvals.policy.default_auto_receipt");
}

/**
 * The effective mode and its provenance, as ONE line for the phone.
 *
 * WHY THE CARD AND THE SHEET BOTH NEED IT. The phone can change the gate from
 * two places, and until this line existed neither said what the gate IS: a
 * reader who had loosened a session yesterday, or whose machine default an
 * agent changed, saw only the two handles and no state. The line is the
 * runtime's answer (the projection carries `approval_mode`/`approval_provenance`
 * on the pending approval), not the phone's memory of its own last gesture — a
 * gate moved from the terminal must not be described by this screen's receipts.
 *
 * THE SCOPE WORDS ARE THE TUI CARD'S, deliberately: the same machine, two
 * surfaces, one vocabulary. A session override states the mode it overrides
 * ("global Auto"), because that is the fact a reader needs before deciding
 * which handle to press.
 *
 * THE GLOBAL HALF IS READ, NEVER DERIVED. Until UX round 2 (U4) this function
 * negated the session mode to name the machine default — so a fresh session with
 * global=ask and its own `ask` override rendered "this session overrides global
 * Auto": a claim about every other conversation, invented from a mode that says
 * nothing about them. The runtime now stamps `approval_global_mode` on the same
 * payload, and when it is absent (an older owner) the clause is DROPPED — half a
 * true sentence beats a whole false one.
 *
 * EMPTY MEANS SAY NOTHING. These fields are absent on a runtime that predates
 * them, and half a statement ("Approvals: Ask.") is still a true one — but a
 * mode with no provenance must not invent a scope, so the fallback names the
 * mode only.
 *
 * THE WORDS COME FROM `relay.approvals.policy.*`, and the translator is an
 * ARGUMENT rather than an import — the same purity rule `lib/asks.ts` follows:
 * this module can be exercised without a rendered React tree or a store. The
 * mode words are keys of their own (`mode_ask`/`mode_auto`) because a locale
 * may translate "Ask" and "Auto" differently from the words the sentence
 * templates wrap them in, and the templates take them as placeholders so word
 * order stays the translator's to choose.
 */
export function approvalPolicyLine(
	input: {
		approval_mode?: ApprovalMode | "";
		approval_provenance?: ApprovalProvenance | "";
		approval_global_mode?: ApprovalMode | "";
	},
	t: Translator,
): string {
	const mode =
		input.approval_mode === "auto"
			? t("relay.approvals.policy.mode_auto")
			: input.approval_mode === "ask"
				? t("relay.approvals.policy.mode_ask")
				: "";
	if (!mode) return "";
	const provenance = input.approval_provenance;
	if (provenance === "session") {
		const global =
			input.approval_global_mode === "auto"
				? t("relay.approvals.policy.mode_auto")
				: input.approval_global_mode === "ask"
					? t("relay.approvals.policy.mode_ask")
					: "";
		/* No global mode, no clause: see the block comment above. */
		return global
			? t("relay.approvals.policy.session_override_global", { mode, global })
			: t("relay.approvals.policy.session_override", { mode });
	}
	if (provenance === "pin") return t("relay.approvals.policy.this_run", { mode });
	if (provenance === "global") return t("relay.approvals.policy.global_default", { mode });
	/* A newer owner naming a scope this build does not know: state the mode,
	 * claim no origin. Guessing "global default" here would be the one wording
	 * this line exists to prevent. */
	return t("relay.approvals.policy.mode_only", { mode });
}

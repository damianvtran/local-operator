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
 *     action (`loosen`) and the same proof fields: the runtime decides what a
 *     signature is accepted for from the frame it is deciding, so a signature
 *     minted for one scope cannot be presented as the other.
 *   * `default ask` — tighten globally, no proof. Not offered by any phone
 *     surface today; it is left out of the union on purpose so a caller cannot
 *     land an unproven global write behind a typo.
 */
import { requestOperatorChallenge, sendCommandWithProof, type CommandReceipt } from "../api";
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
 * session with its own per-session override is not changed by a default write,
 * so "all" would be a claim the runtime never made.
 */
export function globalDefaultReceipt(receipt: CommandReceipt): string {
	const said = receipt.detail.trim();
	if (said) return said;
	return "the default is now auto — sessions that follow it run gated tools without asking";
}

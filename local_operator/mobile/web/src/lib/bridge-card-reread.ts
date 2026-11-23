/**
 * When an on-screen Team Bridge card body asks again — the re-read policy.
 *
 * WHY A CARD KEEPS ASKING. The server answers `pending` from the moment the
 * card is posted until the receiving session takes the body at its next turn,
 * and that hand-off is often long. Measured on the operator's machine (n=4005
 * receipts, post -> hand-off): p50 45 s, p90 21 min, p99 100 min, max 8 h. A
 * panel that read once and kept the first answer said "Waiting to be delivered"
 * forever (the reported bug: a received message whose body never appeared,
 * even hours after delivery, until the conversation was reopened).
 *
 * So while the panel is mounted and its answer is not final, it re-reads on a
 * backoff — 5, 10, 20, 30 s, then every 60 s — with NO overall give-up: any cap
 * below the observed 8 h maximum reintroduces the bug for exactly the slow
 * tail. Only mounted cards poll, and a `pending`/`held` re-read is a LOCAL read
 * in core (resolved from the receiver store before any hub backfill); only
 * misses and offline reach the hub, at most once a minute per card. Unmount or
 * a final answer stops the loop, and returning to the app (visibility) asks at
 * once. Nothing is cached past a mount, so revocation stays live.
 *
 * The TUI (core `tui/bridge_card_body.py`) and the desktop (local-operator-ui)
 * apply the same table.
 */
import type { BridgeCardBody } from "./bridge-card-body";

/** Delays (ms) before each successive re-read; past the end, `REREAD_STEADY_MS`. */
export const REREAD_BACKOFF_MS: readonly number[] = [5_000, 10_000, 20_000, 30_000];
export const REREAD_STEADY_MS = 60_000;

/** Milliseconds to wait before re-read number `attempt` (0-based). */
export function rereadDelay(attempt: number): number {
	return attempt >= 0 && attempt < REREAD_BACKOFF_MS.length ? REREAD_BACKOFF_MS[attempt] : REREAD_STEADY_MS;
}

/** Whether an answer can no longer change by waiting.
 *
 *  Non-final: no answer (a failed or malformed read), any item `pending`, any
 *  item `held` (a transient hold clears when screening runs; a review hold can
 *  still be released or refused), and `unavailable` because Radient was
 *  offline. Everything else is final — admitted, refused, not delivered, the
 *  other unavailable reasons, and any state the parser could not place (it
 *  reads as reasonless `unavailable`), so an unknown state fails quiet instead
 *  of polling forever. */
export function bridgeAnswerIsFinal(answer: BridgeCardBody | null): boolean {
	if (!answer) return false;
	return !answer.items.some(
		(item) =>
			item.state === "pending" ||
			item.state === "held" ||
			(item.state === "unavailable" && item.reason === "offline"),
	);
}

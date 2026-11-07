/**
 * Pure derivation for the `/usage` sheet: grouping, ordering, tone and the
 * binding-window summary. React-free and copy-free so it can be unit tested
 * without a DOM, and so the sheet component holds layout only.
 *
 * WHY DERIVE AT ALL. The daemon ships `tone`, `remaining_fraction`,
 * `account_index`/`account_count`, `state` and `provider_name` since
 * `POST /api/usage/refresh` landed. The SPA is served by the daemon it talks
 * to, but a phone can hold a page loaded before the daemon was updated (or
 * the reverse, a cached bundle against a daemon rolled back), so a field may
 * be missing. Every derived field here therefore prefers the daemon's value
 * and falls back to the same rule the daemon uses (`UsageAmount.status()`:
 * >=1 exhausted, >=0.85 warning), so an older daemon paints the same sheet
 * instead of an empty one; one without the refresh route is the sheet's
 * `unsupported` phase. Python stays the authority whenever it speaks.
 *
 * WHAT IS NOT DERIVED. Paid eligibility (`payable`) and exhaustion of the
 * shared plan stay the daemon's verdicts, exactly as before the redesign; the
 * phone never infers eligibility from a positive wallet.
 */
import type {
	PhoneUsageLimit,
	PhoneUsageReport,
	PhoneUsageResetOutcome,
	PhoneUsageTone,
} from "../api";

export type UsageRowKind = "included" | "family" | "paid";

export type UsageRow = {
	limit: PhoneUsageLimit;
	kind: UsageRowKind;
	tone: PhoneUsageTone;
	/** 0..1 remaining, or null when unmeasurable (rendered as a dotted rule). */
	remaining: number | null;
};

export type UsageAccount = {
	report: PhoneUsageReport;
	/** 1-based label number; the card shows "Account N" only when count > 1. */
	number: number;
	count: number;
	state: NonNullable<PhoneUsageReport["state"]>;
	reauth: boolean;
	rows: UsageRow[];
	/** The window that gates this account first: the card header's summary. */
	binding: UsageRow | null;
	/** Shared included plan exhausted while eligible paid allowance continues. */
	continues: boolean;
	/** A family cap is exhausted and no paid allowance supersedes family caps. */
	familyRestricted: boolean;
};

export type UsageProvider = {
	provider: string;
	name: string;
	accounts: UsageAccount[];
	/** 0 reauth, 1 exhausted, 2 warning, 3 healthy: attention sorts first. */
	attention: number;
};

const WINDOW_ORDER: Record<PhoneUsageLimit["window"], number> = {
	"5h": 0, "7d": 1, "1mo": 2, lifetime: 3, unknown: 4,
};
const KIND_ORDER: Record<UsageRowKind, number> = { included: 0, family: 1, paid: 2 };
const TONE_RANK: Record<PhoneUsageTone, number> = { exhausted: 0, warning: 1, ok: 2, unknown: 3 };

const finite = (value: number | null | undefined): value is number =>
	typeof value === "number" && Number.isFinite(value);

/** Consumed fraction, mirroring Python's `UsageAmount.fraction()`. */
export function usedFraction(amount: PhoneUsageLimit["amount"]): number | null {
	if (finite(amount.used_fraction)) return amount.used_fraction;
	if (finite(amount.used) && finite(amount.limit) && amount.limit > 0) return amount.used / amount.limit;
	if (finite(amount.used) && finite(amount.remaining) && amount.used + amount.remaining > 0) {
		return amount.used / (amount.used + amount.remaining);
	}
	return null;
}

export function remainingFraction(limit: PhoneUsageLimit): number | null {
	if (limit.remaining_fraction !== undefined) return finite(limit.remaining_fraction) ? limit.remaining_fraction : null;
	const used = usedFraction(limit.amount);
	return used === null ? null : Math.min(1, Math.max(0, 1 - used));
}

export function limitTone(limit: PhoneUsageLimit): PhoneUsageTone {
	if (limit.tone) return limit.tone;
	const used = usedFraction(limit.amount);
	if (used === null) return "unknown";
	return used >= 1 ? "exhausted" : used >= 0.85 ? "warning" : "ok";
}

export function rowKind(limit: PhoneUsageLimit): UsageRowKind {
	return limit.paid_fallback ? "paid" : limit.family_cap ? "family" : "included";
}

/** Included windows shortest first, then family caps by window, then paid. */
export function orderRows(limits: PhoneUsageLimit[]): UsageRow[] {
	return limits
		.map((limit, index) => ({ limit, index, kind: rowKind(limit), tone: limitTone(limit), remaining: remainingFraction(limit) }))
		.sort((a, b) => KIND_ORDER[a.kind] - KIND_ORDER[b.kind]
			|| WINDOW_ORDER[a.limit.window] - WINDOW_ORDER[b.limit.window]
			|| a.index - b.index)
		.map(({ limit, kind, tone, remaining }) => ({ limit, kind, tone, remaining }));
}

/** The desktop's "binding" idea: the row closest to refusing work for the
 *  WHOLE account. Included windows only — a spent family cap stops one model
 *  family, not the account, so it must not paint the card "Exhausted" (it has
 *  its own row and the family_caps note). Family caps bind only when the
 *  account reports nothing else. Worst tone wins, then least remaining;
 *  unknown never outranks a measured row. Paid rows never bind. */
export function bindingRow(rows: UsageRow[]): UsageRow | null {
	const usable = rows.filter((row) => row.kind !== "paid" && !row.limit.unlimited);
	const included = usable.filter((row) => row.kind === "included");
	const gating = included.length ? included : usable;
	let best: UsageRow | null = null;
	for (const row of gating) {
		if (!best) { best = row; continue; }
		const byTone = TONE_RANK[row.tone] - TONE_RANK[best.tone];
		if (byTone < 0 || (byTone === 0 && (row.remaining ?? 2) < (best.remaining ?? 2))) best = row;
	}
	return best;
}

function reportState(report: PhoneUsageReport): NonNullable<PhoneUsageReport["state"]> {
	if (report.state) return report.state;
	if (report.credential_invalid) return "reauth_required";
	return report.unavailable || !report.limits.length ? "unavailable" : "available";
}

/** Group by provider, number accounts by login order, sort providers. */
export function buildUsageModel(reports: PhoneUsageReport[], collator: Intl.Collator): UsageProvider[] {
	const groups = new Map<string, PhoneUsageReport[]>();
	for (const report of reports) {
		const list = groups.get(report.provider) ?? [];
		list.push(report);
		groups.set(report.provider, list);
	}
	const providers: UsageProvider[] = [];
	for (const [provider, list] of groups) {
		// Never re-sort accounts by health: "Account 2" must always be the
		// same login. The daemon's login-order index wins; arrival order is the
		// fallback for an older daemon.
		const ordered = list
			.map((report, arrival) => ({ report, key: report.account_index ?? arrival }))
			.sort((a, b) => a.key - b.key)
			.map(({ report }) => report);
		const accounts = ordered.map((report, position): UsageAccount => {
			const rows = orderRows(report.limits);
			const state = reportState(report);
			const paidContinues = rows.some((row) => row.limit.payable);
			const sharedSpent = rows.some((row) => row.kind === "included" && row.limit.shared && row.limit.exhausted);
			return {
				report,
				number: position + 1,
				count: report.account_count ?? ordered.length,
				state,
				reauth: state === "reauth_required" || report.credential_invalid === true,
				rows,
				binding: bindingRow(rows),
				continues: paidContinues && sharedSpent,
				familyRestricted: rows.some((row) => row.kind === "family" && row.limit.exhausted)
					&& !rows.some((row) => row.limit.payable && row.limit.supersedes_tiers),
			};
		});
		let attention = 3;
		for (const account of accounts) {
			if (account.reauth) attention = Math.min(attention, 0);
			else if (account.binding?.tone === "exhausted") attention = Math.min(attention, 1);
			else if (account.binding?.tone === "warning") attention = Math.min(attention, 2);
		}
		providers.push({ provider, name: ordered[0]?.provider_name || provider, accounts, attention });
	}
	return providers.sort((a, b) => a.attention - b.attention || collator.compare(a.name, b.name));
}

/** Two-unit countdown parts (the desktop's `formatCountdown` shape). */
export type Countdown = { unit: "days"; days: number; hours: number }
	| { unit: "day"; days: number }
	| { unit: "hours"; hours: number; minutes: number }
	| { unit: "hour"; hours: number }
	| { unit: "minutes"; minutes: number }
	| { unit: "now" };

export function countdown(resetsAtMs: number, nowMs: number): Countdown {
	const remaining = resetsAtMs - nowMs;
	if (!(remaining > 0)) return { unit: "now" };
	const totalMinutes = Math.max(1, Math.ceil(remaining / 60_000));
	const days = Math.floor(totalMinutes / 1440);
	const hours = Math.floor((totalMinutes % 1440) / 60);
	const minutes = totalMinutes % 60;
	// Two units at most, and a zero second unit is dropped ("3d", "2h"), the
	// desktop's formatCountdown shape.
	if (days > 0) return hours > 0 ? { unit: "days", days, hours } : { unit: "day", days };
	if (hours > 0) return minutes > 0 ? { unit: "hours", hours, minutes } : { unit: "hour", hours };
	return { unit: "minutes", minutes };
}

export type RefreshPhase =
	| { kind: "idle" }
	| { kind: "refreshing" }
	| { kind: "ok"; at: number | null }
	| { kind: "failed" }
	| { kind: "timeout" }
	/** `seconds` is the wait at ENTRY: the one value announced to assistive
	    tech, so the live region does not re-speak every countdown tick. */
	| { kind: "throttled"; until: number; seconds: number }
	/** The daemon predates `POST /api/usage/refresh` (404/405): cached data is
	    all there is, which is not a failure, so no banner and no button. */
	| { kind: "unsupported" };

/** Map a refresh answer to the status line's phase. `nowMs` anchors throttle.
 *  A `joined` answer (this request rode a refresh already in flight) is `ok`:
 *  the caller got fresh numbers, not a refusal. */
export function refreshPhase(
	refresh: { status: "ok" | "throttled" | "failed" | "timeout"; retry_after_ms: number | null; joined?: boolean } | undefined,
	refreshedAt: number | null | undefined,
	nowMs: number,
): RefreshPhase {
	if (!refresh) return { kind: "idle" };
	if (refresh.status === "throttled") {
		const wait = Math.max(0, refresh.retry_after_ms ?? 0);
		return { kind: "throttled", until: nowMs + wait, seconds: Math.max(1, Math.ceil(wait / 1000)) };
	}
	if (refresh.status === "failed") return { kind: "failed" };
	if (refresh.status === "timeout") return { kind: "timeout" };
	return { kind: "ok", at: refreshedAt ?? null };
}

/* -- reset credits ----------------------------------------------------------
 *
 * WHAT THE PHONE MAY SPEND AND HOW IT NAMES IT. A reset spends something that
 * cannot be refunded, on ONE login the operator confirms. The phone still never
 * sees that login's name: it sends back the `account_index`/`account_count`
 * pair the daemon itself displayed plus, when the daemon published one, the
 * opaque `credential_ref` of the row it showed. So the target here carries the
 * DAEMON's numbering, never the card's rendered position -- those diverge the
 * moment an older daemon omits `account_index`, and a request built from the
 * wrong one could spend on a sibling.
 */

/** One spendable reset: what the card offers and what a request must carry. */
export type ResetTarget = {
	provider: string;
	/** 0-based index the projection sent -- the daemon's own numbering. */
	index: number;
	/** Accounts the phone displayed for this provider; the daemon re-checks it. */
	count: number;
	/** Spendable credits, always >= 1 (the badge's number). */
	available: number;
	/** The daemon's pointer at the one login this card showed, or null when it
	 *  published none (an older daemon, or a row it could not attribute). Sent
	 *  with the spend so a login churn cannot move the credit. */
	ref: string | null;
};

/** Spendable reset credits on a report, mirroring Python's exact-int rule.
 *
 * A JSON boolean is `typeof "boolean"`, a float is not `Number.isInteger`, and a
 * negative count is not a count. `null`/absent means the vendor reported
 * nothing trustworthy -- "not reported" and "none left" are both unspendable,
 * and neither may reach a button that spends. */
export function resetsAvailable(report: PhoneUsageReport): number {
	const count = report.resets_available;
	if (typeof count !== "number" || !Number.isInteger(count) || count <= 0) return 0;
	return count;
}

/** The account a reset would spend on, or null when none may be offered.
 *
 * NO FALLBACK NUMBERING (R6). A target built from the card's rendered position
 * is a different address from the one the daemon displayed, and one that
 * silently spends on whichever login sits there -- so a report without the
 * daemon's own `account_index`/`account_count` offers nothing at all. */
export function resetTarget(account: UsageAccount): ResetTarget | null {
	const available = resetsAvailable(account.report);
	if (available === 0) return null;
	const index = account.report.account_index;
	const count = account.report.account_count;
	if (typeof index !== "number" || !Number.isInteger(index) || index < 0) return null;
	if (typeof count !== "number" || !Number.isInteger(count) || count < 1) return null;
	const ref = account.report.credential_ref;
	return {
		provider: account.report.provider,
		index,
		count,
		available,
		ref: typeof ref === "string" && ref !== "" ? ref : null,
	};
}

/** A card's identity across re-renders: provider plus its displayed number. */
export function accountKey(account: UsageAccount): string {
	return `${account.report.provider}#${account.number}`;
}

/** ONE reset interaction for the whole sheet, so a second card's tap cancels
 *  the first card's confirmation instead of stacking two irreversible prompts.
 *
 *  The TARGET is snapshotted into the flow at confirm time. The card re-derives
 *  its offer from the latest numbers, and spending the LAST credit repaints it
 *  with zero -- a panel that read the live target would unmount exactly when
 *  its outcome arrived (D1/Q2). Only the BADGE depends on a positive count. */
export type ResetFlow =
	| { kind: "idle" }
	| { kind: "confirm"; key: string; target: ResetTarget }
	| { kind: "busy"; key: string; target: ResetTarget; redeemRequestId: string }
	| {
			kind: "result";
			key: string;
			target: ResetTarget;
			outcome: PhoneUsageResetOutcome;
			windowsReset: number;
			/** Kept only for an UNKNOWN outcome, where a retry must reuse it. */
			redeemRequestId: string | null;
			/** The daemon refused before spending anything; nothing was attempted. */
			refused?: boolean;
			/** The spend's own re-fetch did not produce current numbers (R5). */
			numbersStale?: boolean;
	  };

export const RESET_FLOW_IDLE: ResetFlow = { kind: "idle" };

/** Close a confirmation (or a result) without spending anything. */
export function resetFlowCancel(): ResetFlow {
	return { kind: "idle" };
}

export function resetFlowConfirm(key: string, target: ResetTarget): ResetFlow {
	return { kind: "confirm", key, target };
}

export function resetFlowStart(key: string, redeemRequestId: string, target: ResetTarget): ResetFlow {
	return { kind: "busy", key, target, redeemRequestId };
}

/** Settle a busy flow with the outcome the daemon reported.
 *
 * A late answer for a flow that is no longer busy is IGNORED: the operator
 * moved on (or a refusal already landed), and painting an outcome into a card
 * they had left would state it as if they had just asked. Only a definitive
 * outcome clears the redeem id -- `failed` keeps it so a retry is idempotent
 * and cannot spend a second credit. `refused` is definitive: nothing was sent.
 *
 * A settled `result` also stays on screen whatever the credit count is now, so
 * spending the last credit still says so. */
export function resetFlowSettle(
	flow: ResetFlow,
	outcome: PhoneUsageResetOutcome,
	windowsReset: number,
	numbersStale = false,
): ResetFlow {
	if (flow.kind !== "busy") return flow;
	const unknown = outcome === "failed";
	return {
		kind: "result",
		key: flow.key,
		target: flow.target,
		outcome,
		windowsReset: Number.isInteger(windowsReset) && windowsReset > 0 ? windowsReset : 0,
		redeemRequestId: unknown ? flow.redeemRequestId : null,
		...(outcome === "refused" ? { refused: true } : {}),
		...(numbersStale ? { numbersStale: true } : {}),
	};
}

/** A refusal the daemon answered with an HTTP status (accounts changed, already
 *  in flight, no reset product): nothing was spent, so the id is dropped and a
 *  retry mints a new one. */
export function resetFlowRefused(flow: ResetFlow): ResetFlow {
	return resetFlowSettle(flow, "refused", 0);
}

/* -- the pending attempt ----------------------------------------------------
 *
 * WHY NOT IN THE FLOW. `ResetFlow` is the one card being interacted with right
 * now: another card's tap replaces it, Cancel clears it, and closing the sheet
 * resets it. The IDEMPOTENCY ID must outlive all of that, because a retry after
 * an unconfirmed outcome that minted a NEW id is exactly how a second credit is
 * spent. So the id lives in its own per-account map, owned by the sheet. */
export type ResetAttempt = {
	redeemRequestId: string;
	/** A request for this account is outstanding right now. */
	busy: boolean;
};
export type ResetAttempts = Readonly<Record<string, ResetAttempt>>;

export const RESET_ATTEMPTS_EMPTY: ResetAttempts = {};

/** The id this attempt must send: a pending one, or a fresh one. */
export function resetAttemptStart(
	attempts: ResetAttempts,
	key: string,
	mint: () => string,
): { attempts: ResetAttempts; redeemRequestId: string } {
	const pending = attempts[key];
	const redeemRequestId = pending?.redeemRequestId ?? mint();
	return { attempts: { ...attempts, [key]: { redeemRequestId, busy: true } }, redeemRequestId };
}

/** The outcome landed. A definitive one forgets the attempt; an UNKNOWN one
 *  keeps its id so a retry can be idempotent. */
export function resetAttemptSettled(
	attempts: ResetAttempts,
	key: string,
	outcome: PhoneUsageResetOutcome,
): ResetAttempts {
	const entry = attempts[key];
	if (!entry) return attempts;
	const next = { ...attempts };
	if (outcome === "failed") next[key] = { redeemRequestId: entry.redeemRequestId, busy: false };
	else delete next[key];
	return next;
}

/** The request finished, whatever it answered: `busy` tracks the request, not
 *  the outcome, so a late answer cannot leave a card looking in flight. */
export function resetAttemptDone(attempts: ResetAttempts, key: string): ResetAttempts {
	const entry = attempts[key];
	if (!entry?.busy) return attempts;
	return { ...attempts, [key]: { redeemRequestId: entry.redeemRequestId, busy: false } };
}

/** Any spend outstanding in this sheet: while one is, every badge is inert, so
 *  a second card cannot replace the flow that is showing the first one's
 *  outcome (U1/R2). */
export function resetAttemptBusy(attempts: ResetAttempts): boolean {
	return Object.values(attempts).some((entry) => entry.busy);
}

/** The status line after a reset's own re-fetch.
 *
 * A reset is not a refresh the operator asked for. Its throttle or failure must
 * never sit beside the outcome sentence -- "Reset applied" next to "Couldn't
 * refresh" reads as a contradiction -- and the cooldown note a throttle would
 * print is noise for a fetch the spend itself caused (R5/Q6). Both collapse to
 * the neutral freshness line, and the panel says the numbers are old instead. */
export function resetRefreshPhase(
	refresh: { status: "ok" | "throttled" | "failed" | "timeout" } | undefined,
	refreshedAt: number | null | undefined,
): { phase: RefreshPhase; numbersStale: boolean } {
	if (!refresh) return { phase: { kind: "idle" }, numbersStale: false };
	if (refresh.status === "ok") return { phase: { kind: "ok", at: refreshedAt ?? null }, numbersStale: false };
	return { phase: { kind: "idle" }, numbersStale: true };
}

/** The catalogue key for one outcome. `reset` is the only one that varies with
 *  a count, so callers pass `windowsReset` to it alone. */
export type ResetOutcomeKey =
	| "relay.usage_reset.outcome_reset"
	| "relay.usage_reset.outcome_already_redeemed"
	| "relay.usage_reset.outcome_no_credit"
	| "relay.usage_reset.outcome_nothing_to_reset"
	| "relay.usage_reset.outcome_failed"
	| "relay.usage_reset.refused";

export function resetOutcomeKey(outcome: PhoneUsageResetOutcome): ResetOutcomeKey {
	switch (outcome) {
		case "reset": return "relay.usage_reset.outcome_reset";
		case "already_redeemed": return "relay.usage_reset.outcome_already_redeemed";
		case "no_credit": return "relay.usage_reset.outcome_no_credit";
		case "nothing_to_reset": return "relay.usage_reset.outcome_nothing_to_reset";
		case "refused": return "relay.usage_reset.refused";
		default: return "relay.usage_reset.outcome_failed";
	}
}

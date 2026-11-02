/**
 * Pure derivation for the `/usage` sheet: grouping, ordering, tone and the
 * binding-window summary. React-free and copy-free so it can be unit tested
 * without a DOM, and so the sheet component holds layout only.
 *
 * WHY DERIVE AT ALL. The daemon ships `tone`, `remaining_fraction`,
 * `account_index`/`account_count`, `state` and `provider_name` since
 * `POST /api/usage/refresh` landed, but a phone can be paired to an OLDER
 * daemon that omits them. Every derived field here therefore prefers the
 * daemon's value and falls back to the same rule the daemon uses
 * (`UsageAmount.status()`: >=1 exhausted, >=0.85 warning), so an old daemon
 * paints the same sheet instead of an empty one. Python stays the authority
 * whenever it speaks.
 *
 * WHAT IS NOT DERIVED. Paid eligibility (`payable`) and exhaustion of the
 * shared plan stay the daemon's verdicts, exactly as before the redesign; the
 * phone never infers eligibility from a positive wallet.
 */
import type { PhoneUsageLimit, PhoneUsageReport, PhoneUsageTone } from "../api";

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
	| { kind: "throttled"; until: number };

/** Map a refresh answer to the status line's phase. `nowMs` anchors throttle. */
export function refreshPhase(
	refresh: { status: "ok" | "throttled" | "failed" | "timeout"; retry_after_ms: number | null } | undefined,
	refreshedAt: number | null | undefined,
	nowMs: number,
): RefreshPhase {
	if (!refresh) return { kind: "idle" };
	if (refresh.status === "throttled") return { kind: "throttled", until: nowMs + Math.max(0, refresh.retry_after_ms ?? 0) };
	if (refresh.status === "failed") return { kind: "failed" };
	if (refresh.status === "timeout") return { kind: "timeout" };
	return { kind: "ok", at: refreshedAt ?? null };
}

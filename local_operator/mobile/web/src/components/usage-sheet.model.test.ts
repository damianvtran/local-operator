import { describe, expect, it } from "vitest";
import type { PhoneUsageLimit, PhoneUsageReport } from "../api";
import { bindingRow, buildUsageModel, countdown, limitTone, orderRows, refreshPhase, remainingFraction } from "./usage-sheet.model";

const limit = (over: Partial<PhoneUsageLimit> = {}): PhoneUsageLimit => ({
	amount: { used: null, limit: null, remaining: null, used_fraction: 0.2, unit: "percent" },
	paid_fallback: false, available: null, unlimited: false, supersedes_tiers: false,
	family_cap: false, shared: true, window: "5h", family: "unknown",
	resets_at_ms: null, payable: false, exhausted: false, ...over,
});
const report = (over: Partial<PhoneUsageReport> = {}): PhoneUsageReport => ({
	provider: "openai", unavailable: false, fetched_at: 1, limits: [limit()], ...over,
});
const collator = new Intl.Collator("en");
const used = (fraction: number | null) => ({ amount: { used: null, limit: null, remaining: null, used_fraction: fraction, unit: "percent" } });

describe("tone and remaining", () => {
	it("prefers daemon fields and mirrors UsageAmount.status() without them", () => {
		expect(limitTone(limit({ tone: "exhausted", ...used(0.1) }))).toBe("exhausted");
		expect(limitTone(limit(used(0.84)))).toBe("ok");
		expect(limitTone(limit(used(0.85)))).toBe("warning");
		expect(limitTone(limit(used(1)))).toBe("exhausted");
		expect(limitTone(limit(used(null)))).toBe("unknown");
		expect(remainingFraction(limit({ remaining_fraction: 0.4, ...used(0.9) }))).toBe(0.4);
		expect(remainingFraction(limit(used(1.04)))).toBe(0);
		expect(remainingFraction(limit({ remaining_fraction: null }))).toBeNull();
		expect(remainingFraction(limit({ amount: { used: 30, limit: 120, remaining: null, used_fraction: null, unit: "credits" } }))).toBe(0.75);
	});
});

describe("row ordering", () => {
	it("included shortest first, then family caps by window, then paid", () => {
		const rows = orderRows([
			limit({ paid_fallback: true, shared: false }),
			limit({ family_cap: true, shared: false, window: "7d", family: "fable" }),
			limit({ window: "7d" }),
			limit({ family_cap: true, shared: false, window: "5h", family: "opus" }),
			limit({ window: "5h" }),
		]);
		expect(rows.map((row) => `${row.kind}:${row.limit.window}`)).toEqual([
			"included:5h", "included:7d", "family:5h", "family:7d", "paid:5h",
		]);
	});

	it("the binding window is the worst tone, then least remaining, never paid", () => {
		const rows = orderRows([
			limit(used(0.5)), limit({ window: "7d", ...used(0.9) }),
			limit({ paid_fallback: true, shared: false, ...used(1) }),
		]);
		expect(bindingRow(rows)?.limit.window).toBe("7d");
		expect(bindingRow(orderRows([limit(used(null)), limit({ window: "7d", ...used(0.1) })]))?.limit.window).toBe("7d");
		// A spent family cap never makes the whole account read "Exhausted".
		const capped = orderRows([limit(used(0.1)), limit({ family_cap: true, shared: false, family: "fable", ...used(1) })]);
		expect(bindingRow(capped)?.kind).toBe("included");
		// ...unless the account reports only family caps.
		expect(bindingRow(orderRows([limit({ family_cap: true, shared: false, ...used(1) })]))?.kind).toBe("family");
	});
});

describe("provider model", () => {
	it("numbers accounts by login index, never by health, and sorts attention first", () => {
		const model = buildUsageModel([
			report({ provider: "zai", provider_name: "Z.AI" }),
			report({ provider: "anthropic", provider_name: "Anthropic", account_index: 2, account_count: 3 }),
			report({ provider: "anthropic", provider_name: "Anthropic", account_index: 0, account_count: 3, limits: [limit(used(1))] }),
			report({ provider: "anthropic", provider_name: "Anthropic", account_index: 1, account_count: 3 }),
			report({ provider: "deepseek", provider_name: "DeepSeek" }),
			report({ provider: "kimi", provider_name: "Kimi", state: "reauth_required", credential_invalid: true }),
		], collator);
		expect(model.map((p) => p.name)).toEqual(["Kimi", "Anthropic", "DeepSeek", "Z.AI"]);
		const anthropic = model[1]!;
		expect(anthropic.accounts.map((a) => [a.number, a.count, a.report.account_index])).toEqual([[1, 3, 0], [2, 3, 1], [3, 3, 2]]);
		expect(anthropic.accounts[0]!.binding?.tone).toBe("exhausted");
	});

	it("falls back to arrival order and the provider id on an older daemon", () => {
		const [only] = buildUsageModel([report(), report({ unavailable: true })], collator);
		expect(only!.name).toBe("openai");
		expect(only!.accounts.map((a) => [a.number, a.count, a.state])).toEqual([[1, 2, "available"], [2, 2, "unavailable"]]);
	});

	it("continuation needs explicit shared exhaustion plus payable credit", () => {
		const exhausted = limit({ ...used(1), exhausted: true });
		const wallet = limit({ paid_fallback: true, shared: false, available: true, payable: true });
		const cap = limit({ family_cap: true, shared: false, ...used(1), exhausted: true, family: "opus" });
		const [continues] = buildUsageModel([report({ limits: [exhausted, wallet] })], collator);
		expect(continues!.accounts[0]!.continues).toBe(true);
		const [familyOnly] = buildUsageModel([report({ limits: [limit(), cap, wallet] })], collator);
		expect(familyOnly!.accounts[0]!.continues).toBe(false);
		expect(familyOnly!.accounts[0]!.familyRestricted).toBe(true);
		const [superseded] = buildUsageModel([report({ limits: [cap, { ...wallet, supersedes_tiers: true }] })], collator);
		expect(superseded!.accounts[0]!.familyRestricted).toBe(false);
	});
});

describe("countdown and refresh phase", () => {
	it("uses the desktop's two-unit shape", () => {
		const now = 1_000_000;
		expect(countdown(now + (2 * 1440 + 4 * 60) * 60_000, now)).toEqual({ unit: "days", days: 2, hours: 4 });
		expect(countdown(now + (14 * 60 + 5) * 60_000, now)).toEqual({ unit: "hours", hours: 14, minutes: 5 });
		expect(countdown(now + 42 * 60_000, now)).toEqual({ unit: "minutes", minutes: 42 });
		expect(countdown(now + 1, now)).toEqual({ unit: "minutes", minutes: 1 });
		expect(countdown(now + 3 * 1440 * 60_000, now)).toEqual({ unit: "day", days: 3 });
		expect(countdown(now + 2 * 60 * 60_000, now)).toEqual({ unit: "hour", hours: 2 });
		expect(countdown(now, now)).toEqual({ unit: "now" });
	});

	it("maps each refresh status, anchoring throttle to now", () => {
		expect(refreshPhase(undefined, null, 0)).toEqual({ kind: "idle" });
		expect(refreshPhase({ status: "throttled", retry_after_ms: 4000 }, null, 10)).toEqual({ kind: "throttled", until: 4010 });
		expect(refreshPhase({ status: "failed", retry_after_ms: null }, 5, 0)).toEqual({ kind: "failed" });
		expect(refreshPhase({ status: "timeout", retry_after_ms: null }, 5, 0)).toEqual({ kind: "timeout" });
		expect(refreshPhase({ status: "ok", retry_after_ms: null }, 5, 0)).toEqual({ kind: "ok", at: 5 });
	});
});

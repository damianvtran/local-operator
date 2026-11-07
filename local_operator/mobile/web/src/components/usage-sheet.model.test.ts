import { describe, expect, it } from "vitest";
import type { PhoneUsageLimit, PhoneUsageReport } from "../api";
import {
	accountKey, bindingRow, buildUsageModel, countdown, limitTone, orderRows, refreshPhase,
	remainingFraction, resetFlowCancel, resetFlowConfirm, resetFlowRefused, resetFlowSettle,
	resetFlowStart, resetOutcomeKey, resetRequestId, resetTarget, resetsAvailable,
} from "./usage-sheet.model";

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
		// `seconds` is fixed at entry: the one value the live region announces.
		expect(refreshPhase({ status: "throttled", retry_after_ms: 4000 }, null, 10)).toEqual({ kind: "throttled", until: 4010, seconds: 4 });
		expect(refreshPhase({ status: "failed", retry_after_ms: null }, 5, 0)).toEqual({ kind: "failed" });
		expect(refreshPhase({ status: "timeout", retry_after_ms: null }, 5, 0)).toEqual({ kind: "timeout" });
		expect(refreshPhase({ status: "ok", retry_after_ms: null }, 5, 0)).toEqual({ kind: "ok", at: 5 });
		// A joined answer rode a refresh already in flight: it is ok, not a refusal.
		expect(refreshPhase({ status: "ok", retry_after_ms: null, joined: true }, 5, 0)).toEqual({ kind: "ok", at: 5 });
	});
});

describe("reset credits", () => {
	const account = (over: Partial<PhoneUsageReport> = {}) => {
		const model = buildUsageModel([report(over)], collator);
		return model[0].accounts[0];
	};

	it("offers only a plain positive integer, never a boolean or a float", () => {
		expect(resetsAvailable(report({ resets_available: 2 }))).toBe(2);
		expect(resetsAvailable(report({ resets_available: 0 }))).toBe(0);
		expect(resetsAvailable(report({ resets_available: null }))).toBe(0);
		expect(resetsAvailable(report({}))).toBe(0);
		// A wire bug must not become a spendable button: Python's exact-int rule.
		expect(resetsAvailable(report({ resets_available: -1 }))).toBe(0);
		expect(resetsAvailable(report({ resets_available: 1.5 }))).toBe(0);
		expect(resetsAvailable(report({ resets_available: true as unknown as number }))).toBe(0);
	});

	it("carries the DAEMON's numbering, not the card's position", () => {
		const second = account({ account_index: 1, account_count: 2, resets_available: 3 });
		expect(resetTarget(second)).toEqual({ provider: "openai", index: 1, count: 2, available: 3 });
		// No account_index (an older daemon) falls back to the rendered number.
		const fallback = account({ resets_available: 1 });
		expect(resetTarget(fallback)).toEqual({ provider: "openai", index: 0, count: 1, available: 1 });
		expect(resetTarget(account({ resets_available: 0 }))).toBeNull();
	});

	it("names a card stably and opens one confirmation at a time", () => {
		const two = buildUsageModel([
			report({ account_index: 0, account_count: 2, resets_available: 1 }),
			report({ account_index: 1, account_count: 2, resets_available: 1 }),
		], collator)[0].accounts;
		const [first, second] = two;
		expect(accountKey(first)).not.toBe(accountKey(second));
		const opened = resetFlowConfirm(accountKey(first));
		expect(opened).toEqual({ kind: "confirm", key: accountKey(first) });
		// Opening the second card's confirmation replaces the first: the state is
		// one object, so two irreversible prompts cannot stack.
		expect(resetFlowConfirm(accountKey(second))).toEqual({ kind: "confirm", key: accountKey(second) });
		expect(resetFlowCancel()).toEqual({ kind: "idle" });
	});

	it("settles only a busy flow and clears the id on a definitive outcome", () => {
		const busy = resetFlowStart("k", "id-1");
		expect(resetFlowSettle(busy, "reset", 2)).toEqual({
			kind: "result", key: "k", outcome: "reset", windowsReset: 2, redeemRequestId: null,
		});
		// `failed` keeps the id so the retry is idempotent.
		expect(resetFlowSettle(busy, "failed", 0)).toEqual({
			kind: "result", key: "k", outcome: "failed", windowsReset: 0, redeemRequestId: "id-1",
		});
		// A negative/garbled window count is not a count.
		expect(resetFlowSettle(resetFlowStart("k", "i"), "reset", -3)).toMatchObject({ windowsReset: 0 });
		// A late answer for a flow the operator already left is ignored.
		expect(resetFlowSettle({ kind: "idle" }, "reset", 1)).toEqual({ kind: "idle" });
		expect(resetFlowSettle(resetFlowConfirm("k"), "reset", 1)).toEqual({ kind: "confirm", key: "k" });
	});

	it("reuses the id after an unconfirmed outcome only", () => {
		const mint = () => "fresh";
		const failed = resetFlowSettle(resetFlowStart("k", "id-1"), "failed", 0);
		expect(resetRequestId(failed, "k", mint)).toBe("id-1");
		// A REFUSAL spent nothing, so the next attempt gets a new id.
		expect(resetRequestId(resetFlowRefused(resetFlowStart("k", "id-1")), "k", mint)).toBe("fresh");
		// So does a definitive outcome, and another card's result.
		expect(resetRequestId(resetFlowSettle(resetFlowStart("k", "id-1"), "reset", 1), "k", mint)).toBe("fresh");
		expect(resetRequestId(failed, "other", mint)).toBe("fresh");
		expect(resetRequestId({ kind: "idle" }, "k", mint)).toBe("fresh");
	});

	it("maps every outcome to its own copy key, unknown outcomes to failed", () => {
		expect(resetOutcomeKey("reset")).toBe("relay.usage_reset.outcome_reset");
		expect(resetOutcomeKey("already_redeemed")).toBe("relay.usage_reset.outcome_already_redeemed");
		expect(resetOutcomeKey("no_credit")).toBe("relay.usage_reset.outcome_no_credit");
		expect(resetOutcomeKey("nothing_to_reset")).toBe("relay.usage_reset.outcome_nothing_to_reset");
		expect(resetOutcomeKey("failed")).toBe("relay.usage_reset.outcome_failed");
		const keys = new Set(
			(["reset", "already_redeemed", "no_credit", "nothing_to_reset", "failed"] as const)
				.map((outcome) => resetOutcomeKey(outcome)),
		);
		expect(keys.size).toBe(5);
	});
});

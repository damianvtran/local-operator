/**
 * `/usage` — the phone's provider usage sheet.
 *
 * FLOW. Opening paints the cached `GET /api/usage` straight away (it never
 * writes and answers older daemons too), then asks `POST /api/usage/refresh`
 * with `force:false`, which the daemon serves from its 5-minute cache unless
 * the data is stale. The header's refresh button posts `force:true`; the
 * daemon floors that at one per 15 s and answers `throttled` with the wait,
 * which this sheet counts down with the button inert. A refresh that fails
 * or times out still answers with cached data: the cards NEVER blank while
 * refreshing or after a failure, and no spinner covers them.
 *
 * LAYOUT CONSTRAINTS (designer spec, round 0). Borderless: cards are the
 * `surface` step on the `elevated` panel, bars a `sunken` track with no stroke.
 * Bars show REMAINING, start-aligned so RTL flips them. Colour is never the
 * only signal — every tone carries a word, exhausted also a glyph. Unknown
 * amounts draw a dotted rule, never an empty track (an empty track reads as
 * exhausted). Labels truncate, numbers do not, so 320px never scrolls
 * sideways. Accounts of one provider are "Account N" in login order, never
 * re-sorted by health and never labelled with an email.
 *
 * Ordering, tone and binding-window derivation live in `usage-sheet.model.ts`
 * (pure, unit-tested); this file is layout and copy only.
 *
 * RESET CREDITS. A card whose account reports spendable reset credits carries
 * ONE text affordance at the right end of its header line (`Reset · N`), so no
 * row is added when there are none and none is pushed down when there are.
 * Tapping it turns that card into an inline confirmation -- no nested modal,
 * because the confirmation must name the account whose numbers sit behind it.
 * Only one card can confirm at a time (one flow object for the sheet), the
 * default focus is Cancel rather than the button that spends, and the phone
 * sends the ACCOUNT INDEX the daemon displayed: it never learns a login name.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getUsage, HttpError, isUsageResetRefusal, refreshUsage, resetUsage, type PhoneUsageEnvelope, type PhoneUsageLimit, type PhoneUsageResetOutcome, type PhoneUsageTone } from "../api";
import { Sheet } from "./ui/sheet";
import { Button } from "./ui/button";
import { ChromeIcon } from "./ui/chrome-icon";
import { Spinner } from "./spinner";
import { messageRuntime, useLocaleSnapshot } from "../store";
import { cn } from "../lib/cn";
import {
	accountKey, buildUsageModel, countdown, refreshPhase, resetFlowCancel, resetFlowConfirm,
	resetFlowRefused, resetFlowSettle, resetFlowStart, resetOutcomeKey, resetRequestId, resetTarget,
	RESET_FLOW_IDLE,
	type RefreshPhase, type ResetFlow, type ResetTarget, type UsageAccount, type UsageRow,
} from "./usage-sheet.model";

const t = messageRuntime.t;

const TONE_TEXT: Record<PhoneUsageTone, string> = {
	ok: "text-success", warning: "text-warning", exhausted: "text-danger", unknown: "text-ink-dim",
};
const TONE_FILL: Record<PhoneUsageTone, string> = {
	ok: "bg-success", warning: "bg-warning", exhausted: "bg-danger", unknown: "bg-ink-dim",
};

function toneWord(tone: PhoneUsageTone): string {
	return tone === "ok" ? t("relay.shell.usage.tone_ok")
		: tone === "warning" ? t("relay.shell.usage.tone_warning")
			: tone === "exhausted" ? t("relay.shell.usage.tone_exhausted")
				: t("relay.shell.usage.tone_unknown");
}

function windowWord(window: PhoneUsageLimit["window"]): string {
	return window === "5h" ? t("relay.shell.usage.window_five")
		: window === "7d" ? t("relay.shell.usage.window_week")
			: window === "1mo" ? t("relay.shell.usage.window_month")
				: window === "lifetime" ? t("relay.shell.usage.window_lifetime")
					: t("relay.shell.usage.window_unknown");
}

function familyWord(family: PhoneUsageLimit["family"]): string {
	return family === "opus" ? t("relay.shell.usage.family_opus")
		: family === "sonnet" ? t("relay.shell.usage.family_sonnet")
			: family === "fable" ? t("relay.shell.usage.family_fable")
				: family === "zread" ? t("relay.shell.usage.family_zread")
					: t("relay.shell.usage.family_unknown");
}

function rowLabel(row: UsageRow): string {
	if (row.kind === "family") return t("relay.shell.usage.row_label_family", { family: familyWord(row.limit.family), window: windowWord(row.limit.window) });
	if (row.kind === "paid") return t("relay.shell.usage.paid");
	return windowWord(row.limit.window);
}

/** The amount column: the runtime's ICU formatter owns locale digits/units. */
function amountText(row: UsageRow): string {
	const limit = row.limit;
	const amount = limit.amount;
	// "Unlimited paid allowance" names a wallet; an included window that is
	// unlimited is just unlimited (design round 1, D4).
	if (limit.unlimited) return row.kind === "paid" ? t("relay.shell.usage.unlimited") : t("relay.shell.usage.unlimited_generic");
	const measured = amount.remaining ?? amount.used;
	const unit = measured !== null && Number.isFinite(measured)
		? amount.unit === "credits" ? t("relay.shell.usage.unit_credits", { count: measured })
			: amount.unit === "usd" ? t("relay.shell.usage.unit_usd")
				: amount.unit === "tokens" ? t("relay.shell.usage.unit_tokens", { count: measured })
					: amount.unit === "requests" ? t("relay.shell.usage.unit_requests", { count: measured }) : null
		: null;
	const remaining = amount.remaining !== null;
	if (measured !== null && Number.isFinite(measured)) {
		if (amount.unit === "percent") return t(remaining ? "relay.shell.usage.remaining_percent" : "relay.shell.usage.used_percent", { value: measured / 100 });
		if (unit !== null) return t(remaining ? "relay.shell.usage.remaining" : "relay.shell.usage.used", { value: measured, unit });
		return t(remaining ? "relay.shell.usage.remaining_unknown_unit" : "relay.shell.usage.used_unknown_unit", { value: measured });
	}
	// Fraction-only windows read as REMAINING from the same value the bar
	// draws (the daemon's clamped remaining_fraction when it sent one), so the
	// number and the fill can never disagree.
	if (row.remaining !== null) return t("relay.shell.usage.remaining_percent", { value: row.remaining });
	return t("relay.shell.usage.amount_unknown");
}

function countdownText(resetsAtMs: number, nowMs: number): string {
	const parts = countdown(resetsAtMs, nowMs);
	if (parts.unit === "now") return t("relay.shell.usage.reset_now");
	const value = parts.unit === "days" ? t("relay.shell.usage.countdown_days", { days: parts.days, hours: parts.hours })
		: parts.unit === "day" ? t("relay.shell.usage.countdown_day", { days: parts.days })
			: parts.unit === "hours" ? t("relay.shell.usage.countdown_hours", { hours: parts.hours, minutes: parts.minutes })
				: parts.unit === "hour" ? t("relay.shell.usage.countdown_hour", { hours: parts.hours })
					: t("relay.shell.usage.countdown_minutes", { minutes: parts.minutes });
	return t("relay.shell.usage.resets_in", { countdown: value });
}

const validTime = (value: number | null | undefined): value is number =>
	typeof value === "number" && Number.isFinite(new Date(value).getTime());

/** A timestamp as short as it can be without lying: time only today, no year
 *  within this year. ICU's `{value, date, medium}` always prints the year, and
 *  the message subset Python shares has no skeletons, so the runtime's own
 *  `Intl.DateTimeFormat` options pick the parts and the catalogue only places
 *  the result (`{when}`). `locale` is the locale of the copy on screen. */
function whenText(at: number, now: number, locale: string): string {
	const date = new Date(at);
	const today = new Date(now);
	const sameYear = date.getFullYear() === today.getFullYear();
	const sameDay = sameYear && date.getMonth() === today.getMonth() && date.getDate() === today.getDate();
	const options: Intl.DateTimeFormatOptions = sameDay ? { hour: "numeric", minute: "2-digit" }
		: { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", ...(sameYear ? {} : { year: "numeric" }) };
	try { return new Intl.DateTimeFormat(locale, options).format(date); } catch { return new Intl.DateTimeFormat(undefined, options).format(date); }
}

/** Relative by default; a tap shows the exact time. Local to its row. The
 *  dotted underline is the affordance (plain dim text read as a label, UX
 *  round 1, U8); the accessible name IS the shown text, and `aria-pressed`
 *  carries which form is shown. */
function ResetToggle({ at, now, locale }: { at: number; now: number; locale: string }) {
	const [exact, setExact] = useState(false);
	return <button type="button" onClick={() => setExact((value) => !value)} aria-pressed={exact}
		className="-my-3 inline-flex min-h-11 shrink-0 items-center whitespace-nowrap text-ink-dim underline decoration-ink-dim decoration-dotted underline-offset-4">
		{exact ? t("relay.shell.usage.reset", { when: whenText(at, now, locale) }) : countdownText(at, now)}
	</button>;
}

function Bar({ row, dim, valueText }: { row: UsageRow; dim: boolean; valueText: string }) {
	if (row.limit.unlimited) return null;
	if (row.remaining === null || row.tone === "unknown") {
		// No track: an empty track would read as exhausted. Dotted rule only.
		return <div role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuetext={valueText}
			className="my-[3px] border-t border-dotted border-ink-dim" />;
	}
	const percent = Math.round(row.remaining * 100);
	return <div role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent} aria-valuetext={valueText}
		className="h-1.5 w-full overflow-hidden rounded-xs bg-sunken">
		{row.remaining > 0 && <div
			className={cn("h-full rounded-xs motion-safe:transition-[width] motion-safe:duration-fast", dim ? "bg-ink-dim" : TONE_FILL[row.tone])}
			// 2px floor so a sliver of headroom is still visible at 1%.
			style={{ width: `max(2px, ${row.remaining * 100}%)` }} />}
	</div>;
}

function LimitRow({ row, dim, now, locale }: { row: UsageRow; dim: boolean; now: number; locale: string }) {
	const label = rowLabel(row);
	const value = amountText(row);
	// Python's tone "unknown" means "no fraction to judge against", not "nothing
	// reported": a paid wallet with a known balance has no limit to be within
	// (design D1 / UX U5), and an unknown amount already says "Amount not
	// reported" in the amount column (D3). Either way the row states it once,
	// so the tone word is shown only for a measured tone.
	const tone = row.limit.unlimited || row.tone === "unknown" ? "" : toneWord(row.tone);
	const reset = validTime(row.limit.resets_at_ms) ? row.limit.resets_at_ms : null;
	const resetText = reset !== null ? countdownText(reset, now) : null;
	const a11y = tone
		? resetText !== null ? t("relay.shell.usage.row_a11y_reset", { label, value, tone, reset: resetText }) : t("relay.shell.usage.row_a11y", { label, value, tone })
		: resetText !== null ? t("relay.shell.usage.row_a11y_bare_reset", { label, value, reset: resetText }) : t("relay.shell.usage.row_a11y_bare", { label, value });
	return <div className={cn("flex min-w-0 flex-col gap-1", row.kind === "family" && "ps-3")}>
		<div className="flex min-w-0 items-baseline justify-between gap-3">
			<span className={cn("min-w-0 truncate text-body-sm", row.kind === "family" ? "text-ink-dim" : "text-ink-muted")}>{label}</span>
			<span className="shrink-0 font-mono text-mono-sm tabular-nums text-ink-muted">{value}</span>
		</div>
		<Bar row={row} dim={dim} valueText={a11y} />
		{(tone || reset !== null) && <div className="flex min-w-0 items-center justify-between gap-3 text-meta">
			<span className={cn("min-w-0 truncate", row.tone === "exhausted" && !dim ? "text-danger" : "text-ink-dim")}>{tone}</span>
			{reset !== null && <ResetToggle at={reset} now={now} locale={locale} />}
		</div>}
		{row.kind === "paid" && row.limit.available === false && <p className="text-meta text-danger"><span aria-hidden>✕ </span>{t("relay.shell.usage.blocked")}</p>}
		{row.kind === "paid" && row.limit.available === null && <p className="text-meta text-ink-dim">{t("relay.shell.usage.eligibility_unknown")}</p>}
	</div>;
}

function Note({ glyph, tone, children }: { glyph: string; tone: string; children: string }) {
	return <p className={cn("flex gap-1.5 text-meta", tone)}><span aria-hidden>{glyph}</span><span>{children}</span></p>;
}

/** One outcome's sentence. Only `reset` varies with a count, so only it takes
 *  one: a `windows_reset` of 0 still says the reset was APPLIED rather than
 *  claiming a window was cleared. */
function outcomeText(outcome: PhoneUsageResetOutcome, windowsReset: number): string {
	const key = resetOutcomeKey(outcome);
	return key === "relay.usage_reset.outcome_reset"
		? t(key, { count: windowsReset })
		: t(key);
}

/** ONE account's reset interaction, in place of the badge it was opened from.
 *
 * INLINE, NOT A NESTED MODAL: the confirmation has to name the account it will
 * spend on, and the card already is that account. A sheet over a sheet would
 * hide the numbers the operator is confirming about.
 *
 * FOCUS: Cancel is the default focus and the side-effecting button is not,
 * because a stray Enter on a freshly opened confirmation would spend a credit
 * that cannot be refunded. `autoFocus` on a stable element inside an existing
 * card also keeps the card's own controls reachable. */
function ResetPanel({ flow, target, name, busy, onCancel, onConfirm }: {
	flow: Extract<ResetFlow, { kind: "confirm" | "busy" | "result" }>;
	target: ResetTarget;
	name: string;
	busy: boolean;
	onCancel: () => void;
	onConfirm: () => void;
}) {
	const heading = t("relay.usage_reset.confirm_title", {
		provider: name, index: String(target.index + 1), count: String(target.count),
	});
	// INERT, NOT DISABLED (the refresh button's rule, UX round 1 U1): native
	// `disabled` drops keyboard focus to <body>, which would strand a keyboard
	// user inside a confirmation whose buttons no longer take focus.
	const inert = busy;
	return <div role="group" aria-label={heading} aria-busy={busy || undefined}
		className="flex flex-col gap-2 rounded-sm bg-sunken px-2.5 py-2">
		<p className="min-w-0 text-meta font-medium text-ink">{heading}</p>
		{flow.kind === "result"
			? <div className="flex flex-col gap-2">
				<p aria-live="polite" className={cn("text-meta",
					flow.refused || flow.outcome === "failed" ? "text-warning" : "text-ink-muted")}>
					{flow.refused ? t("relay.usage_reset.refused") : outcomeText(flow.outcome, flow.windowsReset)}
				</p>
				{/* An unconfirmed outcome is the one that offers a retry: it reuses
				    the request id, so the vendor answers `already_redeemed` instead of
				    spending a second credit. */}
				{flow.outcome === "failed" && !flow.refused && <div className="flex flex-wrap items-center justify-end gap-2">
					<Button variant="tonal" onClick={onConfirm}>{t("relay.usage_reset.confirm")}</Button>
				</div>}
			</div>
			: <p className="text-meta text-ink-muted">{busy ? t("relay.usage_reset.busy") : t("relay.usage_reset.confirm_prompt", { count: target.available })}</p>}
		{flow.kind !== "result" && <div className="flex flex-wrap items-center justify-end gap-2">
			<Button variant="ghost" autoFocus aria-disabled={inert || undefined}
				className={cn(inert && "!text-ink-disabled")}
				onClick={() => { if (!inert) onCancel(); }}>
				{t("relay.usage_reset.cancel")}
			</Button>
			<Button variant="tonal" aria-disabled={inert || undefined} aria-busy={busy || undefined}
				className={cn(inert && "!text-ink-disabled")}
				onClick={() => { if (!inert) onConfirm(); }}>
				{t("relay.usage_reset.confirm")}
			</Button>
		</div>}
	</div>;
}

function AccountCard({ account, name, now, locale, flow, onResetOpen, onResetCancel, onResetConfirm }: {
	account: UsageAccount;
	name: string;
	now: number;
	locale: string;
	flow: ResetFlow;
	onResetOpen: (key: string) => void;
	onResetCancel: () => void;
	onResetConfirm: (key: string, target: ResetTarget) => void;
}) {
	const binding = account.binding;
	// Reauth and unavailable rows show last-known numbers, so their fills go
	// dim: the colour must not claim a current reading it does not have.
	const dim = account.reauth || account.state === "unavailable";
	const percent = binding?.remaining != null ? Math.round(binding.remaining * 100) : null;
	const key = accountKey(account);
	const target = resetTarget(account);
	// A dead grant cannot authenticate a spend, so its card offers none: the
	// attempt would come back unconfirmed when nothing was ever sent.
	const offer = target !== null && !account.reauth;
	const mine = flow.kind !== "idle" && flow.key === key ? flow : null;
	return <div className="flex flex-col gap-3 rounded-md bg-surface px-3 py-2.5">
		<div className="flex min-w-0 items-start justify-between gap-3">
			<span className="min-w-0 truncate text-meta text-ink-muted">
				{account.count > 1 ? t("relay.shell.usage.account", { index: account.number }) : null}
			</span>
			<div className="flex min-w-0 shrink-0 items-center gap-2">
				{/* The ONE reset affordance: the same header line the card already
				    draws, so no row is added when absent and no limit row is pushed
				    down when present. `-my-3` keeps the 44px tap target from growing
				    that row; the label beside it truncates, this never does. */}
				{offer && (mine === null || mine.kind === "result") && <button type="button"
					className="-my-3 inline-flex min-h-11 shrink-0 items-center whitespace-nowrap text-meta font-medium text-accent underline decoration-accent decoration-dotted underline-offset-4"
					onClick={() => onResetOpen(key)}>
					{t("relay.usage_reset.badge", { count: String(target.available) })}
				</button>}
				{account.reauth
					? <span className="shrink-0 text-meta font-medium text-warning">{t("relay.shell.usage.reauth_short")}</span>
					: binding && <div className="flex shrink-0 flex-col items-end">
						<span className={cn("text-meta font-medium", dim ? "text-ink-dim" : TONE_TEXT[binding.tone])}>
							{binding.tone === "exhausted" && <span aria-hidden>✕ </span>}
							{toneWord(binding.tone)}
							{percent !== null && <span className="ms-1.5 font-mono text-mono-sm tabular-nums">{t("relay.shell.usage.remaining_percent", { value: percent / 100 })}</span>}
						</span>
					</div>}
			</div>
		</div>
		{offer && mine !== null && <ResetPanel flow={mine} target={target} name={name}
			busy={mine.kind === "busy"} onCancel={onResetCancel}
			onConfirm={() => onResetConfirm(key, target)} />}
		{account.rows.map((row, index) => <LimitRow key={index} row={row} dim={dim} now={now} locale={locale} />)}
		{(account.reauth || account.state === "unavailable" || account.continues || account.familyRestricted) && <div className="flex flex-col gap-1">
			{account.reauth && <Note glyph="↻" tone="text-warning">{t("relay.shell.usage.reauth")}</Note>}
			{!account.reauth && account.state === "unavailable" && <Note glyph="⚠" tone="text-ink-dim">{t("relay.shell.usage.last_known")}</Note>}
			{account.continues && <Note glyph="→" tone="text-ink-muted">{t("relay.shell.usage.continue")}</Note>}
			{account.familyRestricted && <Note glyph="⚠" tone="text-warning">{t("relay.shell.usage.family_caps")}</Note>}
		</div>}
	</div>;
}

function Skeleton() {
	return <div aria-hidden className="flex flex-col gap-2">
		{[0, 1].map((key) => <div key={key} className="flex flex-col gap-3 rounded-md bg-surface px-3 py-2.5">
			<div className="h-3 w-24 rounded-xs bg-sunken" />
			<div className="h-1.5 w-full rounded-xs bg-sunken" />
			<div className="h-1.5 w-full rounded-xs bg-sunken" />
		</div>)}
	</div>;
}

type Status = { text: string; tone: string; spoken?: string };

/** The status line under the title. `updated` is the "Updated …" sentence the
 *  line falls back to whenever nothing transient is happening (UX round 1,
 *  U7), so freshness never disappears after a throttle. While throttled the
 *  visible countdown ticks but `spoken` — the only part the live region reads
 *  — holds the wait at entry, so it is announced once, not every second (U3). */
function statusLine(phase: RefreshPhase, throttled: boolean, now: number, updated: string | null): Status | null {
	switch (phase.kind) {
		case "refreshing": return { text: t("relay.shell.usage.refreshing"), tone: "text-ink-dim" };
		case "failed": return { text: t("relay.shell.usage.refresh_failed"), tone: "text-warning" };
		case "timeout": return { text: t("relay.shell.usage.refresh_timeout"), tone: "text-warning" };
		case "throttled": {
			if (!throttled) {
				// Cooldown over: say so once, and show freshness again.
				return { text: updated ?? "", tone: "text-ink-dim", spoken: [updated, t("relay.shell.usage.refresh_ready")].filter(Boolean).join(" ") };
			}
			const seconds = Math.max(1, Math.ceil((phase.until - now) / 1000));
			return {
				text: t("relay.shell.usage.throttled", { seconds }),
				tone: "text-ink-dim",
				spoken: t("relay.shell.usage.throttled", { seconds: phase.seconds }),
			};
		}
		default: return updated === null ? null : { text: updated, tone: "text-ink-dim" };
	}
}

export function UsageSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
	// Same-locale bundle changes also repaint; backend preference alone may be
	// unrendered while the owned copy (and its formatter locale) falls back to EN.
	const locale = useLocaleSnapshot();
	const [data, setData] = useState<PhoneUsageEnvelope | null>(null);
	const [error, setError] = useState(false);
	const [phase, setPhase] = useState<RefreshPhase>({ kind: "idle" });
	// The last time a refresh in this sheet actually reached vendors. A later
	// throttled or failed answer carries `refreshed_at: null` (it fetched
	// nothing), and the status line must not lose the earlier "Updated" (U7).
	const [freshAt, setFreshAt] = useState<number | null>(null);
	const [now, setNow] = useState(() => Date.now());
	// A late answer from a closed/reopened sheet must not paint over a newer one.
	const generation = useRef(0);
	// ONE reset interaction for the whole sheet: opening a second card's
	// confirmation replaces the first, so two irreversible prompts can never be
	// open at once.
	const [flow, setFlow] = useState<ResetFlow>(RESET_FLOW_IDLE);

	const refresh = useCallback((force: boolean) => {
		const mine = generation.current;
		setPhase({ kind: "refreshing" });
		refreshUsage(force).then((value) => {
			if (generation.current !== mine) return;
			setData(value);
			setError(false);
			setNow(Date.now());
			const next = refreshPhase(value.refresh, value.refreshed_at, Date.now());
			if (next.kind === "ok" && next.at !== null) setFreshAt(next.at);
			setPhase(next);
		}).catch((reason: unknown) => {
			if (generation.current !== mine) return;
			// A daemon without the refresh route (404, or 405 from a catch-all
			// that only knows GET) is not failing: cached data is all it has
			// (review round 1, R5). Anything else keeps the cards and reports it.
			if (reason instanceof HttpError && (reason.status === 404 || reason.status === 405)) setPhase({ kind: "unsupported" });
			else setPhase({ kind: "failed" });
		});
	}, []);

	const load = useCallback(() => {
		const mine = ++generation.current;
		setError(false);
		setPhase({ kind: "idle" });
		getUsage().then((value) => {
			if (generation.current !== mine) return;
			setData(value);
		}).catch(() => {
			if (generation.current !== mine) return;
			setError(true);
		}).finally(() => {
			if (generation.current === mine) refresh(false);
		});
	}, [refresh]);

	useEffect(() => {
		if (!open) { generation.current += 1; return; }
		setData(null);
		setFreshAt(null);
		setFlow(RESET_FLOW_IDLE);
		load();
	}, [open, load]);

	// One clock for countdowns: per second only while throttled (the button
	// re-enables on time), per minute otherwise for "resets in".
	const throttled = phase.kind === "throttled" && phase.until > now;
	useEffect(() => {
		if (!open) return;
		const id = setInterval(() => setNow(Date.now()), throttled ? 1000 : 60_000);
		return () => clearInterval(id);
	}, [open, throttled]);

	const collator = useMemo(() => new Intl.Collator(locale.uiLocale), [locale.uiLocale]);
	const providers = useMemo(() => data ? buildUsageModel(data.reports, collator) : null, [data, collator]);

	/** Spend one credit on the account the operator confirmed.
	 *
	 * The id comes from the flow, not from this call: after an UNCONFIRMED
	 * outcome (:func:`resetRequestId`) the retry reuses the same id, and the
	 * vendor then answers `already_redeemed` instead of spending a second
	 * credit. The sheet's generation guards the late answer, so a closed and
	 * reopened sheet never paints an outcome into a surface the operator left. */
	const confirmReset = useCallback((key: string, target: ResetTarget) => {
		const mine = generation.current;
		const redeemRequestId = resetRequestId(flow, key, () => crypto.randomUUID());
		setFlow(resetFlowStart(key, redeemRequestId));
		resetUsage({
			provider: target.provider,
			accountIndex: target.index,
			accountCount: target.count,
			redeemRequestId,
		}).then((answer) => {
			if (generation.current !== mine) return;
			setNow(Date.now());
			// The answer carries the envelope the daemon could re-read. When it
			// carries none -- or an EMPTY one, which is what a dropped cache row
			// plus a failed re-fetch looks like -- the numbers on screen stay:
			// they are the last thing the daemon measured, and blanking them
			// would hide the account the outcome is about.
			if (answer.reports && answer.reports.length > 0) {
				setData(answer as PhoneUsageEnvelope);
				const next = refreshPhase(answer.refresh, answer.refreshed_at, Date.now());
				if (next.kind === "ok" && next.at !== null) setFreshAt(next.at);
				setPhase(next);
			}
			setFlow((current) => resetFlowSettle(current, answer.reset.outcome, answer.reset.windows_reset));
		}).catch((reason: unknown) => {
			if (generation.current !== mine) return;
			setNow(Date.now());
			// A typed refusal spent NOTHING (the daemon answered before reaching
			// the vendor), so its id is dropped. Anything else may have spent a
			// credit: the flow keeps the id and shows the unconfirmed copy, whose
			// retry is idempotent.
			if (isUsageResetRefusal(reason)) setFlow((current) => resetFlowRefused(current));
			else setFlow((current) => resetFlowSettle(current, "failed", 0));
		});
	}, [flow]);
	const refreshing = phase.kind === "refreshing";
	const unsupported = phase.kind === "unsupported";
	// The freshness sentence the status line returns to. A refresh's
	// `refreshed_at` is what THAT refresh fetched; when it fetched nothing (an
	// unforced refresh served from a fresh cache, or a plain GET) the oldest
	// report shown is the honest time. `whenText` adds the date when not today.
	const updatedAt = freshAt ?? data?.refreshed_at ?? data?.as_of ?? data?.cached_at ?? null;
	const updated = data !== null && validTime(updatedAt)
		? [t("relay.shell.usage.updated", { when: whenText(updatedAt, now, locale.uiLocale) }), data.may_be_stale ? t("relay.shell.usage.stale") : null].filter(Boolean).join(" ")
		: null;
	// "Showing last known usage" is only true when something is shown: with no
	// data the body's loading/error state speaks, not the status line.
	const status = data === null && phase.kind !== "refreshing" ? null : statusLine(phase, throttled, now, updated);
	const asOf = data?.as_of ?? data?.refreshed_at ?? data?.cached_at ?? null;
	// INERT, NOT DISABLED (UX round 1, U1): a native `disabled` on the focused
	// button drops keyboard focus to <body> on every tap. `aria-disabled` keeps
	// focus on the sheet's main action and the click is ignored instead.
	const inert = refreshing || throttled || (data === null && !error);

	const action = unsupported ? null : <Button variant="icon" aria-label={t("relay.shell.usage.refresh")}
		aria-disabled={inert || undefined} aria-busy={refreshing || undefined}
		// The icon variant already supplies muted ink. Override its cascade
		// explicitly while inert; adding a normal disabled token loses to the
		// variant's later CSS rule (D8/Q5). Same important idiom as its !px-0.
		className={inert ? "!text-ink-disabled" : undefined}
		onClick={() => { if (inert) return; if (error && data === null) load(); else refresh(true); }}>
		{refreshing ? <Spinner className="size-4" /> : <ChromeIcon name="refresh" />}
	</Button>;

	const loadingText = data === null && !error ? t("relay.shell.usage.loading") : "";
	// ONE live region. While throttled, the ticking countdown is visual only
	// (aria-hidden) and the region reads the `spoken` sentence, fixed at entry.
	const detail = <div aria-live="polite" className={cn("min-h-5 px-3 pb-1 text-meta", status?.tone ?? "text-ink-dim")}>
		{status?.spoken !== undefined
			? <><span className="sr-only">{status.spoken}</span><span aria-hidden>{status.text}</span></>
			: status?.text ?? loadingText}
	</div>;

	return <Sheet open={open} onClose={onClose} title={t("relay.shell.usage.title")} headerAction={action} headerDetail={detail}>
		<div className="flex flex-col gap-4 px-3 pt-2 pb-4">
			{data === null && error ? <div className="flex flex-col items-center gap-3 py-6 text-center">
				<p role="alert" className="text-body-sm text-danger">{t("relay.shell.usage.unavailable")}</p>
				<Button variant="tonal" onClick={load}>{t("relay.shell.usage.retry")}</Button>
			</div> : providers === null ? <Skeleton /> : providers.length === 0
				? <p className="text-body-sm text-ink-dim">{t("relay.shell.usage.empty")}</p>
				: providers.map((provider) => <section key={provider.provider} className="flex flex-col gap-2">
					<h3 className="min-w-0 truncate text-body-sm font-medium text-ink">{provider.name}</h3>
					{provider.accounts.map((account) => <AccountCard key={account.number} account={account}
						name={provider.name} now={now} locale={locale.uiLocale} flow={flow}
						onResetOpen={(key) => setFlow(resetFlowConfirm(key))}
						onResetCancel={() => setFlow(resetFlowCancel())}
						onResetConfirm={confirmReset} />)}
					{/* "Account 2" names no login on purpose (no emails cross the
					    tunnel); this says how to find it on the computer (U6). */}
					{provider.accounts.some((account) => account.count > 1) && <p className="text-meta text-ink-dim">{t("relay.shell.usage.accounts_order")}</p>}
				</section>)}
			{data !== null && validTime(asOf) && <p className="text-meta text-ink-dim">
				{t("relay.shell.usage.as_of", { value: asOf })}
			</p>}
		</div>
	</Sheet>;
}

/**
 * `/usage` — the phone's provider usage sheet.
 *
 * FLOW. Opening paints the cached `GET /api/usage` straight away (it never
 * writes and answers older daemons too), then asks `POST /api/usage/refresh`
 * with `force:false`, which the daemon serves from its 5-minute cache unless
 * the data is stale. The header's refresh button posts `force:true`; the
 * daemon floors that at one per 15 s and answers `throttled` with the wait,
 * which this sheet counts down with the button disabled. A refresh that fails
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
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getUsage, refreshUsage, type PhoneUsageEnvelope, type PhoneUsageLimit, type PhoneUsageTone } from "../api";
import { Sheet } from "./ui/sheet";
import { Button } from "./ui/button";
import { ChromeIcon } from "./ui/chrome-icon";
import { Spinner } from "./spinner";
import { messageRuntime, useLocaleSnapshot } from "../store";
import { cn } from "../lib/cn";
import {
	buildUsageModel, countdown, refreshPhase,
	type RefreshPhase, type UsageAccount, type UsageRow,
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
	if (limit.unlimited) return t("relay.shell.usage.unlimited");
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
		: parts.unit === "hours" ? t("relay.shell.usage.countdown_hours", { hours: parts.hours, minutes: parts.minutes })
			: t("relay.shell.usage.countdown_minutes", { minutes: parts.minutes });
	return t("relay.shell.usage.resets_in", { countdown: value });
}

const validTime = (value: number | null | undefined): value is number =>
	typeof value === "number" && Number.isFinite(new Date(value).getTime());

/** Relative by default; a tap shows the exact time. Local to its row. */
function ResetToggle({ at, now }: { at: number; now: number }) {
	const [exact, setExact] = useState(false);
	const full = t("relay.shell.usage.reset", { value: at });
	return <button type="button" onClick={() => setExact((value) => !value)} aria-label={full}
		className="-my-3 inline-flex min-h-11 shrink-0 items-center whitespace-nowrap text-ink-dim">
		{exact ? full : countdownText(at, now)}
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

function LimitRow({ row, dim, now }: { row: UsageRow; dim: boolean; now: number }) {
	const label = rowLabel(row);
	const value = amountText(row);
	const tone = row.limit.unlimited ? "" : toneWord(row.tone);
	const reset = validTime(row.limit.resets_at_ms) ? row.limit.resets_at_ms : null;
	const a11y = reset !== null
		? t("relay.shell.usage.row_a11y_reset", { label, value, tone, reset: countdownText(reset, now) })
		: t("relay.shell.usage.row_a11y", { label, value, tone });
	return <div className={cn("flex min-w-0 flex-col gap-1", row.kind === "family" && "ps-3")}>
		<div className="flex min-w-0 items-baseline justify-between gap-3">
			<span className={cn("min-w-0 truncate text-body-sm", row.kind === "family" ? "text-ink-dim" : "text-ink-muted")}>{label}</span>
			<span className="shrink-0 font-mono text-mono-sm tabular-nums text-ink-muted">{value}</span>
		</div>
		<Bar row={row} dim={dim} valueText={a11y} />
		{(tone || reset !== null) && <div className="flex min-w-0 items-center justify-between gap-3 text-meta">
			<span className={cn("min-w-0 truncate", row.tone === "exhausted" && !dim ? "text-danger" : "text-ink-dim")}>{tone}</span>
			{reset !== null && <ResetToggle at={reset} now={now} />}
		</div>}
		{row.kind === "paid" && row.limit.available === false && <p className="text-meta text-danger"><span aria-hidden>✕ </span>{t("relay.shell.usage.blocked")}</p>}
		{row.kind === "paid" && row.limit.available === null && <p className="text-meta text-ink-dim">{t("relay.shell.usage.eligibility_unknown")}</p>}
	</div>;
}

function Note({ glyph, tone, children }: { glyph: string; tone: string; children: string }) {
	return <p className={cn("flex gap-1.5 text-meta", tone)}><span aria-hidden>{glyph}</span><span>{children}</span></p>;
}

function AccountCard({ account, now }: { account: UsageAccount; now: number }) {
	const binding = account.binding;
	// Reauth and unavailable rows show last-known numbers, so their fills go
	// dim: the colour must not claim a current reading it does not have.
	const dim = account.reauth || account.state === "unavailable";
	const percent = binding?.remaining != null ? Math.round(binding.remaining * 100) : null;
	return <div className="flex flex-col gap-3 rounded-md bg-surface px-3 py-2.5">
		<div className="flex min-w-0 items-start justify-between gap-3">
			<span className="min-w-0 truncate text-meta text-ink-muted">
				{account.count > 1 ? t("relay.shell.usage.account", { index: account.number }) : null}
			</span>
			{account.reauth
				? <span className="shrink-0 text-meta font-medium text-warning">{t("relay.shell.usage.reauth_short")}</span>
				: binding && <div className="flex shrink-0 flex-col items-end">
					<span className={cn("text-meta font-medium", TONE_TEXT[binding.tone])}>
						{binding.tone === "exhausted" && <span aria-hidden>✕ </span>}
						{toneWord(binding.tone)}
						{percent !== null && <span className="ms-1.5 font-mono text-mono-sm tabular-nums">{t("relay.shell.usage.remaining_percent", { value: percent / 100 })}</span>}
					</span>
				</div>}
		</div>
		{account.rows.map((row, index) => <LimitRow key={index} row={row} dim={dim} now={now} />)}
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

function statusLine(phase: RefreshPhase, now: number): { text: string; tone: string } | null {
	switch (phase.kind) {
		case "refreshing": return { text: t("relay.shell.usage.refreshing"), tone: "text-ink-dim" };
		case "ok": return phase.at !== null && validTime(phase.at) ? { text: t("relay.shell.usage.updated", { value: phase.at }), tone: "text-ink-dim" } : null;
		case "failed": return { text: t("relay.shell.usage.refresh_failed"), tone: "text-warning" };
		case "timeout": return { text: t("relay.shell.usage.refresh_timeout"), tone: "text-warning" };
		case "throttled": {
			const seconds = Math.max(1, Math.ceil((phase.until - now) / 1000));
			return { text: t("relay.shell.usage.throttled", { seconds }), tone: "text-ink-dim" };
		}
		default: return null;
	}
}

export function UsageSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
	// Same-locale bundle changes also repaint; backend preference alone may be
	// unrendered while the owned copy (and its formatter locale) falls back to EN.
	const locale = useLocaleSnapshot();
	const [data, setData] = useState<PhoneUsageEnvelope | null>(null);
	const [error, setError] = useState(false);
	const [phase, setPhase] = useState<RefreshPhase>({ kind: "idle" });
	const [now, setNow] = useState(() => Date.now());
	// A late answer from a closed/reopened sheet must not paint over a newer one.
	const generation = useRef(0);

	const refresh = useCallback((force: boolean) => {
		const mine = generation.current;
		setPhase({ kind: "refreshing" });
		refreshUsage(force).then((value) => {
			if (generation.current !== mine) return;
			setData(value);
			setError(false);
			setNow(Date.now());
			setPhase(refreshPhase(value.refresh, value.refreshed_at, Date.now()));
		}).catch(() => {
			if (generation.current !== mine) return;
			// Keep whatever is on screen; only the status line reports it.
			setPhase({ kind: "failed" });
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
	const refreshing = phase.kind === "refreshing";
	const status = statusLine(throttled ? phase : phase.kind === "throttled" ? { kind: "idle" } : phase, now);
	const asOf = data?.refreshed_at ?? data?.cached_at ?? null;

	const action = <Button variant="icon" aria-label={t("relay.shell.usage.refresh")}
		disabled={refreshing || throttled || (data === null && !error)} aria-busy={refreshing || undefined}
		onClick={() => error && data === null ? load() : refresh(true)}>
		{refreshing ? <Spinner className="size-4" /> : <ChromeIcon name="refresh" />}
	</Button>;

	return <Sheet open={open} onClose={onClose} title={t("relay.shell.usage.title")} headerAction={action}>
		<p aria-live="polite" className={cn("min-h-5 truncate px-3 text-meta", status?.tone ?? "text-ink-dim")}>
			{status?.text ?? (data === null && !error ? t("relay.shell.usage.loading") : "")}
		</p>
		<div className="flex flex-col gap-4 px-3 pt-2 pb-4">
			{data === null && error ? <div className="flex flex-col items-center gap-3 py-6 text-center">
				<p role="alert" className="text-body-sm text-danger">{t("relay.shell.usage.unavailable")}</p>
				<Button variant="tonal" onClick={load}>{t("relay.shell.usage.retry")}</Button>
			</div> : providers === null ? <Skeleton /> : providers.length === 0
				? <p className="text-body-sm text-ink-dim">{t("relay.shell.usage.empty")}</p>
				: providers.map((provider) => <section key={provider.provider} className="flex flex-col gap-2">
					<h3 className="min-w-0 truncate text-body-sm font-medium text-ink">{provider.name}</h3>
					{provider.accounts.map((account) => <AccountCard key={account.number} account={account} now={now} />)}
				</section>)}
			{data !== null && validTime(asOf) && <p className="text-meta text-ink-dim">
				{t("relay.shell.usage.as_of", { value: asOf })}
				{data.may_be_stale && <> {t("relay.shell.usage.stale")}</>}
			</p>}
		</div>
	</Sheet>;
}

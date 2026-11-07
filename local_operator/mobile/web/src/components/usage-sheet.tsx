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
 * sends the ACCOUNT INDEX plus the opaque REF the daemon published: it never
 * learns a login name, and the alias the daemon publishes keeps the spend on the
 * login it confirmed even if the login list moves underneath it.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type MutableRefObject } from "react";
import { getUsage, HttpError, isUsageResetInFlight, isUsageResetRefusal, refreshUsage, resetUsage, type PhoneUsageEnvelope, type PhoneUsageLimit, type PhoneUsageResetOutcome, type PhoneUsageTone } from "../api";
import { Sheet } from "./ui/sheet";
import { Button } from "./ui/button";
import { ChromeIcon } from "./ui/chrome-icon";
import { Spinner } from "./spinner";
import { messageRuntime, useLocaleSnapshot } from "../store";
import { cn } from "../lib/cn";
import {
	accountKey, buildUsageModel, countdown, refreshPhase, resetAttemptBusy, resetAttemptDone,
	resetAttemptSettled, resetAttemptStart, resetFlowCancel, resetFlowConfirm,
	resetFlowSettle, resetFlowStart, resetFlowLate, resetOutcomeKey, resetRefreshPhase, resetTarget,
	RESET_ATTEMPTS_EMPTY, RESET_FLOW_IDLE,
	type RefreshPhase, type ResetAttempts, type ResetFlow, type ResetTarget, type UsageAccount, type UsageRow,
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

/** Settle one card's flow with an answer that names it.
 *
 * A BUSY flow settles normally. An IDLE one means the sheet was closed and
 * reopened while the spend was running: the reopen threw the flow away, so the
 * outcome is rebuilt from the attempt's own key and target instead of being
 * dropped (U10). Any other flow -- another card's confirmation -- is left
 * alone, which is what `resetFlowSettle` already does. */
function settleResetFlow(
	current: ResetFlow,
	key: string,
	target: ResetTarget,
	redeemRequestId: string,
	outcome: PhoneUsageResetOutcome,
	windowsReset: number,
	numbersStale: boolean,
): ResetFlow {
	return current.kind === "idle"
		? resetFlowLate(key, target, redeemRequestId, outcome, windowsReset, numbersStale)
		: resetFlowSettle(current, outcome, windowsReset, numbersStale);
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
 * card also keeps the card's own controls reachable. An OUTCOME moves focus to
 * the sentence itself (`tabIndex={-1}`), because the buttons that had it are
 * gone by then and focus would otherwise fall to `<body>` (U2).
 *
 * IT OUTLIVES THE CREDIT COUNT. `flow.target` is the snapshot taken at confirm
 * time, so spending the LAST credit -- which repaints the card with 0 and is
 * the most common real outcome -- still shows "Reset applied" instead of
 * unmounting the panel the instant it has something to say (D1/Q2).
 */
function ResetPanel({ flow, name, busy, onCancel, onConfirm }: {
	flow: Extract<ResetFlow, { kind: "confirm" | "busy" | "result" }>;
	name: string;
	busy: boolean;
	onCancel: () => void;
	onConfirm: () => void;
}) {
	const target = flow.target;
	// A single-account provider is just its provider name: "Account 1 of 1"
	// names nothing the operator does not already know (U9/D7).
	const heading = target.count > 1
		? t("relay.usage_reset.confirm_title", {
			provider: name, index: String(target.index + 1), count: String(target.count),
		})
		: name;
	// INERT, NOT DISABLED (the refresh button's rule, UX round 1 U1): native
	// `disabled` drops keyboard focus to <body>, which would strand a keyboard
	// user inside a confirmation whose buttons no longer take focus.
	const inert = busy;
	const resultRef = useRef<HTMLParagraphElement>(null);
	// The busy state and the outcome share ONE live region, mounted from the
	// moment anything is pending: a region inserted already holding its text is
	// unreliably announced (U2).
	useEffect(() => {
		if (flow.kind === "result") resultRef.current?.focus();
	}, [flow.kind]);
	return <div role="group" aria-label={heading} aria-busy={busy || undefined}
		className="flex flex-col gap-2 rounded-sm bg-sunken px-2.5 py-2">
		{/* D11: with ONE account the title repeated the provider heading 40px
		    above it and named nothing new, so the group's accessible name keeps
		    the provider and the line goes. Balanced wrapping stops a long provider
		    name from leaving the total ("из 3") alone on line two (D12). */}
		{target.count > 1 && <p className="min-w-0 text-meta font-medium text-ink [text-wrap:balance]">{heading}</p>}
		<div aria-live="polite" className="flex flex-col gap-1">
			{flow.kind === "result"
				? <p ref={resultRef} tabIndex={-1} className={cn("min-w-0 text-meta outline-none",
					flow.refused || flow.outcome === "failed" ? "text-warning" : "text-ink-muted")}>
					{flow.refused ? t("relay.usage_reset.refused") : outcomeText(flow.outcome, flow.windowsReset)}
				</p>
				// The spend's own re-fetch decides this, not the status line: the
				// sheet's freshness banner must never appear beside an outcome
				// (R5), so the panel says the numbers are the old ones instead.
				: <p className={cn("min-w-0", busy ? "text-meta text-ink-muted" : "text-body-sm text-ink")}>
					{busy ? t("relay.usage_reset.busy") : t("relay.usage_reset.confirm_prompt", { count: target.available })}
				</p>}
			{flow.kind === "result" && flow.numbersStale && <p className="min-w-0 text-meta text-ink-dim">
				{t("relay.shell.usage.refresh_failed")}
			</p>}
		</div>
		{flow.kind === "result"
			? <div className="flex flex-wrap items-center justify-end gap-2">
				{/* The primary action keeps the RIGHT edge it has in the confirmation
				    row (D9): the thumb had learned that the action is last, and the
				    dismissive control was suddenly sitting there instead. */}
				<Button variant="ghost" onClick={onCancel}>{t("relay.chrome.snackbar.dismiss")}</Button>
				{/* No "try again" on a refusal or a definitive answer: neither can be
				    improved by repeating the request as it stands, and an UNKNOWN
				    outcome's retry is the only one that is idempotent. */}
				{flow.outcome === "failed" && !flow.refused && <Button variant="primary" onClick={onConfirm}>
					{t("relay.shell.usage.retry")}
				</Button>}
			</div>
			: <div className="flex flex-wrap items-center justify-end gap-2">
				<Button variant="ghost" autoFocus aria-disabled={inert || undefined}
					className={cn(inert && "!text-ink-disabled")}
					onClick={() => { if (!inert) onCancel(); }}>
					{t("relay.usage_reset.cancel")}
				</Button>
				{/* PRIMARY, not tonal: `tonal`'s fill barely separates from the
				    panel behind it, which left colour alone to tell the button
				    that SPENDS from the one that does not (D4). WHILE INERT it goes
				    BACK to the neutral tonal fill: keeping the accent made a control
				    that takes no clicks read as the brightest thing on the panel,
				    with a 2.02:1 label on it (D10). "Resetting…" beside it already
				    says what is happening. */}
				<Button variant={inert ? "tonal" : "primary"} aria-disabled={inert || undefined} aria-busy={busy || undefined}
					className={cn(inert && "!text-ink-disabled")}
					onClick={() => { if (!inert) onConfirm(); }}>
					{t("relay.usage_reset.confirm")}
				</Button>
			</div>}
	</div>;
}

function AccountCard({ account, name, now, locale, flow, attempts, busy, exhausted, escapeDismiss, onResetOpen, onResetCancel, onResetConfirm }: {
	account: UsageAccount;
	name: string;
	now: number;
	locale: string;
	flow: ResetFlow;
	attempts: ResetAttempts;
	busy: boolean;
	/** The daemon said this account has no credit left and has not repainted a
	 *  count since (U6): the card stops offering, rather than inviting a tap
	 *  that can only come back `no_credit` again. */
	exhausted: boolean;
	/** Where this card publishes "how to dismiss the panel I am showing". The
	 *  sheet's Escape handler calls it, because only the card knows which badge
	 *  to put focus back on (U12). */
	escapeDismiss: MutableRefObject<(() => void) | null>;
	onResetOpen: (key: string, target: ResetTarget) => void;
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
	const offer = target !== null && !account.reauth && !exhausted ? target : null;
	const badgeRef = useRef<HTMLButtonElement>(null);
	const cardRef = useRef<HTMLDivElement>(null);
	const attempt = attempts[key];
	const live = flow.kind !== "idle" && flow.key === key ? flow : null;
	// Re-entering the sheet while a spend is still outstanding shows the same
	// in-flight state rather than a fresh affordance that would mint a new id.
	const reentered = live === null && offer !== null && attempt?.busy
		? { kind: "busy" as const, key, target: offer, redeemRequestId: attempt.redeemRequestId }
		: null;
	const mine = live ?? reentered;
	// Cancel/dismiss returns focus to the badge it came from; with the badge
	// gone (the last credit was just spent) the card itself takes it, so a
	// keyboard user is never dropped to <body> (U2). The focus call is DEFERRED
	// one frame: the panel replaced the badge in the tree, so the badge does not
	// exist until this state update has painted -- focusing synchronously landed
	// on the card and left a keyboard user one stop short of where they were
	// (U12).
	const dismiss = () => {
		onResetCancel();
		requestAnimationFrame(() => (badgeRef.current ?? cardRef.current)?.focus());
	};
	// Escape asks the card, not the sheet: the panel is the card's, and only it
	// holds the badge focus must return to. A BUSY panel is not dismissible --
	// a request is running and there is nothing to cancel -- so Escape keeps its
	// old meaning there and closes the sheet (U8/U11/U12).
	escapeDismiss.current = mine !== null && mine.kind !== "busy" ? dismiss : null;
	return <div ref={cardRef} tabIndex={-1} className="flex flex-col gap-3 rounded-sm bg-surface px-3 py-2.5 outline-none">
		{/* THE HEADER LINE WRAPS, IN TWO ATOMIC UNITS (D2/U5/D8). The right unit
		    -- the tone word plus the percent -- never splits, so a wrap can only
		    move it whole to its own line. The label and the badge share the other
		    unit and stay on one line: at 320px in ru the badge used to wrap AWAY
		    from the account it belongs to and sit alone at the right edge, which
		    read as detached. The label is still the only thing telling two cards
		    apart, so it keeps its number and truncates rather than the badge. */}
		<div className="flex min-w-0 flex-wrap items-center justify-between gap-x-3 gap-y-1">
			<div className="flex min-w-0 items-center gap-2">
				<span className="min-w-0 truncate text-meta text-ink-muted">
					{account.count > 1 ? t("relay.shell.usage.account", { index: account.number }) : null}
				</span>
				{/* The ONE reset affordance: the same header line the card already
				    draws, so no row is added when absent and no limit row is pushed
				    down when present. `-my-[13.3px]` cancels exactly half the 44px tap
				    target beyond the 17.4px line, so the header does not grow by the
				    2.6px `-my-3` left over (D6); `min-w-11` keeps the target 44px wide
				    in hi/zh-CN, whose badge text is ~42px. A solid underline: dotted
				    reads as a definition or a tooltip, not a button (D5/D7). */}
				{offer !== null && (mine === null || mine.kind === "result") && <button type="button" ref={badgeRef}
					aria-label={`${t("relay.usage_reset.badge", { count: String(offer.available) })} — ${offer.count > 1
						? t("relay.usage_reset.confirm_title", { provider: name, index: String(account.number), count: String(offer.count) })
						: name}`}
					aria-disabled={busy || undefined}
					className={cn("-my-[13.3px] inline-flex min-h-11 min-w-11 shrink-0 items-center justify-center whitespace-nowrap text-meta font-medium text-accent underline decoration-accent underline-offset-4",
						busy && "!text-ink-disabled")}
					onClick={() => { if (!busy) onResetOpen(key, offer); }}>
					{t("relay.usage_reset.badge", { count: String(offer.available) })}
				</button>}
			</div>
			<div className="ms-auto flex shrink-0 items-center justify-end gap-x-1.5 whitespace-nowrap">
				{account.reauth
					? <span className="text-meta font-medium text-warning">{t("relay.shell.usage.reauth_short")}</span>
					: binding && <>
						<span className={cn("text-meta font-medium", dim ? "text-ink-dim" : TONE_TEXT[binding.tone])}>
							{binding.tone === "exhausted" && <span aria-hidden>✕ </span>}
							{toneWord(binding.tone)}
						</span>
						{percent !== null && <span className="shrink-0 font-mono text-mono-sm tabular-nums text-ink-muted">{t("relay.shell.usage.remaining_percent", { value: percent / 100 })}</span>}
					</>}
			</div>
		</div>
		{mine !== null && <ResetPanel flow={mine} name={name}
			busy={mine.kind === "busy"} onCancel={dismiss}
			onConfirm={() => onResetConfirm(key, mine.target)} />}
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
	// The inline panel registers its own dismiss here (see AccountCard): Escape
	// has to run the CARD's dismissal, not a sheet-level one, or focus lands on
	// <body> instead of the badge the panel came from (U12).
	const escapeDismiss = useRef<(() => void) | null>(null);
	// ONE reset interaction for the whole sheet: opening a second card's
	// confirmation replaces the first, so two irreversible prompts can never be
	// open at once.
	const [flow, setFlow] = useState<ResetFlow>(RESET_FLOW_IDLE);
	// The idempotency ids, keyed by account. They live HERE rather than in the
	// flow because the flow is thrown away on every gesture that is not "send
	// that card's confirmation again" -- and a retry that minted a new id is
	// exactly how one tap spends two credits (U1/R2). Deliberately not reset
	// when the sheet reopens: the spend may still be outstanding.
	const [attempts, setAttempts] = useState<ResetAttempts>(RESET_ATTEMPTS_EMPTY);
	// Accounts the daemon answered `no_credit` for, since the last numbers it
	// sent. An outcome without an envelope leaves the old count on screen, and
	// a badge reading "Reset · 2" beside "No reset credits are left." invites a
	// tap that cannot work (U6).
	const [exhausted, setExhausted] = useState<ReadonlySet<string>>(() => new Set());

	const refresh = useCallback((force: boolean) => {
		const mine = generation.current;
		setPhase({ kind: "refreshing" });
		refreshUsage(force).then((value) => {
			if (generation.current !== mine) return;
			setData(value);
			setError(false);
			setNow(Date.now());
			// The daemon's own count supersedes anything this sheet remembered.
			setExhausted(new Set());
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
			setExhausted(new Set());
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
	// Any spend outstanding? While one is, every badge is inert so a second card
	// cannot replace the flow that is showing the first one's outcome (U1/R2).
	const attemptsBusy = resetAttemptBusy(attempts);

	/** Re-read the numbers without letting the status line speak.
	 *
	 * Used after a refusal, where the panel's own sentence is the report: the
	 * count moved, so the list must be re-read (U3), but "Just refreshed — you
	 * can refresh again in 12 seconds" beside it is noise, and a cooldown banner
	 * would outlive the refusal it is not about (Q6/R5). */
	const quietRefresh = useCallback(() => {
		const mine = generation.current;
		refreshUsage(false).then((value) => {
			if (generation.current !== mine) return;
			setData(value);
			setError(false);
			setExhausted(new Set());
			setNow(Date.now());
		}).catch(() => { /* the refusal stands without the numbers */ });
	}, []);

	/** Spend one credit on the account the operator confirmed.
	 *
	 * The id comes from the pending ATTEMPT, not from the flow (U1/R2): after an
	 * UNCONFIRMED outcome the retry reuses the same id, and the vendor then
	 * answers `already_redeemed` instead of spending a second credit -- even if
	 * the sheet was closed and reopened, or another card's badge was tapped,
	 * in between.
	 *
	 * THE ANSWER IS SETTLED BY ACCOUNT, NOT BY SHEET GENERATION (U10). The
	 * attempt outlives a close and reopen -- that is the whole point of it -- so
	 * gating the answer on the sheet's generation meant a spend that landed
	 * afterwards showed NOTHING: the busy panel vanished, the old count came
	 * back, and the id was kept even after a definitive `reset`, so the next
	 * deliberate tap replayed it into "already applied". The attempt, the id and
	 * the outcome are therefore settled whatever is on screen, and the returned
	 * numbers are adopted because they are the newest the daemon has. `load()`
	 * keeps its own generation guard: a late GET must still not repaint a sheet
	 * the operator left. */
	const confirmReset = useCallback((key: string, target: ResetTarget) => {
		const started = resetAttemptStart(attempts, key, () => crypto.randomUUID());
		setAttempts(started.attempts);
		setFlow(resetFlowStart(key, started.redeemRequestId, target));
		resetUsage({
			provider: target.provider,
			accountIndex: target.index,
			accountCount: target.count,
			redeemRequestId: started.redeemRequestId,
			credentialRef: target.ref,
		}).then((answer) => {
			setNow(Date.now());
			// A reset's own re-fetch is not a refresh the operator asked for: a
			// throttle or a failure collapses to the neutral freshness line and
			// the panel says the numbers are old, so "Reset applied" never sits
			// beside "Couldn't refresh" (R5) and the spend does not print a
			// cooldown (Q6). The numbers are still adopted when the daemon could
			// re-read them; an EMPTY answer (dropped cache row plus a failed
			// re-fetch) keeps what is on screen, which is the last thing the
			// daemon measured.
			const { phase: next, numbersStale } = resetRefreshPhase(answer.refresh, answer.refreshed_at);
			if (answer.reports && answer.reports.length > 0) {
				setData(answer as PhoneUsageEnvelope);
				// The daemon's own count supersedes the local `no_credit` verdict.
				setExhausted(new Set());
				if (next.kind === "ok" && next.at !== null) setFreshAt(next.at);
			}
			setPhase(next);
			setAttempts((current) => resetAttemptSettled(current, key, answer.reset.outcome));
			if (answer.reset.outcome === "no_credit") {
				setExhausted((current) => new Set(current).add(key));
			}
			setFlow((current) => settleResetFlow(current, key, target, started.redeemRequestId,
				answer.reset.outcome, answer.reset.windows_reset, numbersStale));
		}).catch((reason: unknown) => {
			setNow(Date.now());
			// `in_flight` IS NOT A REFUSAL (R7). The daemon's single-flight claim
			// is keyed by account, not by redeem id, so this 409 can mean an
			// EARLIER request for the same account is still spending. Nothing here
			// may claim "nothing was spent", and the id must survive: dropping it
			// mints a fresh id on the next tap, which the vendor treats as a second
			// credit. So it settles as the UNKNOWN outcome (id kept, retry
			// idempotent) and the copy says the spend could not be confirmed.
			if (isUsageResetInFlight(reason)) {
				setAttempts((current) => resetAttemptSettled(current, key, "failed"));
				setFlow((current) => settleResetFlow(current, key, target, started.redeemRequestId, "failed", 0, false));
				return;
			}
			// A typed refusal spent NOTHING (the daemon answered before reaching
			// the vendor), so its id is dropped and the list is re-read -- the
			// count it refused on is the thing that moved (U3). Anything else may
			// have spent a credit: the flow keeps the id and shows the
			// unconfirmed copy, whose retry is idempotent.
			if (isUsageResetRefusal(reason)) {
				setAttempts((current) => resetAttemptSettled(current, key, "refused"));
				setFlow((current) => settleResetFlow(current, key, target, started.redeemRequestId, "refused", 0, false));
				quietRefresh();
			} else {
				setFlow((current) => settleResetFlow(current, key, target, started.redeemRequestId, "failed", 0, false));
			}
		}).finally(() => {
			// The request is over, whatever it answered: a card must not stay
			// "in flight" because its answer arrived after the sheet moved on.
			setAttempts((current) => resetAttemptDone(current, key));
		});
	}, [attempts, quietRefresh]);

	/** Escape dismisses the inline panel before the sheet (U8), through the CARD's
	 *  own dismiss so focus returns to the badge it came from (U12). False when no
	 *  dismissible panel is open, which lets the modal layer close the sheet --
	 *  the same answer a busy panel gives (a request is running, nothing to
	 *  cancel). The ✕, a backdrop tap and the drag go straight to `onClose`: a
	 *  visible close control must close the sheet in one gesture (U11). */
	const dismissPanelOnEscape = useCallback((): boolean => {
		const dismiss = escapeDismiss.current;
		if (dismiss === null) return false;
		dismiss();
		return true;
	}, []);
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

	return <Sheet open={open} onClose={onClose} onEscape={dismissPanelOnEscape} title={t("relay.shell.usage.title")} headerAction={action} headerDetail={detail}>
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
						attempts={attempts} busy={attemptsBusy} exhausted={exhausted.has(accountKey(account))}
						escapeDismiss={escapeDismiss}
						onResetOpen={(key, target) => setFlow(resetFlowConfirm(key, target))}
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

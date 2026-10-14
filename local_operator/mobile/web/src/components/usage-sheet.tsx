import { useEffect, useState } from "react";
import { getUsage, type PhoneUsageReport } from "../api";
import { Sheet } from "./ui/sheet";
import { messageRuntime, useLocaleSnapshot } from "../store";

/** Read-only cached facts: eligibility is computed by the backend, not funding. */
export function UsageSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
	// Same-locale bundle changes also repaint; backend preference alone may be
	// unrendered while the owned copy (and its formatter locale) falls back to EN.
	useLocaleSnapshot();
	const [reports, setReports] = useState<PhoneUsageReport[] | null>(null);
	const [error, setError] = useState(false);
	const [freshness, setFreshness] = useState<{ cached_at?: number | null; may_be_stale?: boolean }>({});
	useEffect(() => {
		if (!open) return;
		let active = true;
		setReports(null);
		setError(false);
		setFreshness({});
		getUsage().then((value) => { if (active) { setReports(value.reports); setFreshness(value); } })
			.catch(() => { if (active) setError(true); });
		return () => { active = false; };
	}, [open]);
	return <Sheet open={open} onClose={onClose} title={messageRuntime.t("relay.shell.usage.title")}>
		<div className="flex flex-col gap-3 p-3">
			<p className="text-ink">{messageRuntime.t("relay.shell.usage.cached")}
				{freshness.cached_at != null && <> {messageRuntime.t("relay.shell.usage.as_of", { value: freshness.cached_at })}</>}
				{freshness.may_be_stale && <> {messageRuntime.t("relay.shell.usage.stale")}</>}
			</p>
			{error ? <p role="alert">{messageRuntime.t("relay.shell.usage.unavailable")}</p> : reports === null
				? <p role="status">{messageRuntime.t("relay.shell.usage.loading")}</p> : !reports.length
					? <p>{messageRuntime.t("relay.shell.usage.empty")}</p> : reports.map((report, index) => {
						const paid = report.limits.filter((limit) => limit.payable);
						const exhausted = report.limits.some((limit) => !limit.paid_fallback && limit.shared === true && !limit.family_cap && limit.exhausted);
						const restricted = report.limits.some((limit) => limit.family_cap && limit.exhausted) &&
							!paid.some((limit) => limit.supersedes_tiers);
						return <section key={`${report.provider}-${index}`} className="flex flex-col gap-2">
							<h3>{report.provider}</h3>
							{report.unavailable && <p>{messageRuntime.t("relay.shell.usage.last_known")}</p>}
							{paid.length > 0 && exhausted && <p>{messageRuntime.t("relay.shell.usage.continue")}</p>}
							{restricted && <p>{messageRuntime.t("relay.shell.usage.family_caps")}</p>}
							{report.limits.map((limit, row) => {
								const amount = limit.amount;
								// Reuse the runtime's ICU formatter, whose locale is the selected
								// catalogue's, not the browser's or an unrendered backend preference.
								// Only the provider vocabulary is mapped; future/unknown units retain
								// the bare measured amount without inventing currency or leaking tokens.
								const unit = amount.unit === "credits" ? messageRuntime.t("relay.shell.usage.unit_credits")
									: amount.unit === "usd" ? messageRuntime.t("relay.shell.usage.unit_usd")
										: amount.unit === "tokens" ? messageRuntime.t("relay.shell.usage.unit_tokens")
											: amount.unit === "requests" ? messageRuntime.t("relay.shell.usage.unit_requests") : null;
								const measured = amount.remaining ?? amount.used;
								const remaining = amount.remaining !== null;
								const value = measured !== null && Number.isFinite(measured)
									? amount.unit === "percent"
										? messageRuntime.t(remaining ? "relay.shell.usage.remaining_percent" : "relay.shell.usage.used_percent", { value: measured / 100 })
										: unit !== null
											? messageRuntime.t(remaining ? "relay.shell.usage.remaining" : "relay.shell.usage.used", { value: measured, unit })
											: messageRuntime.t(remaining ? "relay.shell.usage.remaining_unknown_unit" : "relay.shell.usage.used_unknown_unit", { value: measured })
									: amount.used_fraction !== null && Number.isFinite(amount.used_fraction)
										? messageRuntime.t("relay.shell.usage.fraction", { value: amount.used_fraction }) : messageRuntime.t("relay.shell.usage.amount_unknown");
								const label = limit.paid_fallback ? messageRuntime.t("relay.shell.usage.paid") : limit.family_cap ? messageRuntime.t("relay.shell.usage.family") : messageRuntime.t("relay.shell.usage.included");
								const window = limit.window === "5h" ? messageRuntime.t("relay.shell.usage.window_five") : limit.window === "7d" ? messageRuntime.t("relay.shell.usage.window_week") : limit.window === "1mo" ? messageRuntime.t("relay.shell.usage.window_month") : limit.window === "lifetime" ? messageRuntime.t("relay.shell.usage.window_lifetime") : messageRuntime.t("relay.shell.usage.window_unknown");
								// Translate the whole row so locales own field order and punctuation.
								// Separate variants omit unknown/non-family identifiers without leaving
								// empty parentheses; even an unknown window has a translated label.
								const summary = limit.family_cap && limit.family !== "unknown"
									? messageRuntime.t("relay.shell.usage.row_family", { label, family: limit.family, window, value })
									: messageRuntime.t("relay.shell.usage.row", { label, window, value });
								return <div key={row}>
									<p>{summary}</p>
									{limit.resets_at_ms !== null && Number.isFinite(new Date(limit.resets_at_ms).getTime()) && <p>{messageRuntime.t("relay.shell.usage.reset", { value: limit.resets_at_ms })}</p>}
									{limit.paid_fallback && limit.available === false && <p>{messageRuntime.t("relay.shell.usage.blocked")}</p>}
									{limit.paid_fallback && limit.available === null && <p>{messageRuntime.t("relay.shell.usage.eligibility_unknown")}</p>}
									{limit.paid_fallback && limit.available === true && limit.unlimited && <p>{messageRuntime.t("relay.shell.usage.unlimited")}</p>}
								</div>;
							})}
						</section>;
					})}
		</div>
	</Sheet>;
}

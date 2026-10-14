import { useEffect, useState } from "react";
import { getUsage, type PhoneUsageReport } from "../api";
import { Sheet } from "./ui/sheet";
import { messageRuntime, useLanguage } from "../store";

/** Read-only cached facts: eligibility is computed by the backend, not funding. */
export function UsageSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
	useLanguage();
	const [reports, setReports] = useState<PhoneUsageReport[] | null>(null);
	const [error, setError] = useState(false);
	useEffect(() => {
		if (!open) return;
		let active = true;
		setReports(null);
		setError(false);
		getUsage().then((value) => { if (active) setReports(value.reports); })
			.catch(() => { if (active) setError(true); });
		return () => { active = false; };
	}, [open]);
	return <Sheet open={open} onClose={onClose} title={messageRuntime.t("relay.shell.usage.title")}>
		<div className="flex flex-col gap-3 p-3">
			<p className="text-ink-muted">{messageRuntime.t("relay.shell.usage.cached")}</p>
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
								// No invented cap, percentage or currency conversion for credits.
								const value = amount.remaining !== null ? messageRuntime.t("relay.shell.usage.remaining", { value: String(amount.remaining), unit: amount.unit })
									: amount.used !== null ? messageRuntime.t("relay.shell.usage.used", { value: String(amount.used), unit: amount.unit })
										: amount.used_fraction !== null ? messageRuntime.t("relay.shell.usage.fraction", { value: String(amount.used_fraction * 100) }) : messageRuntime.t("relay.shell.usage.amount_unknown");
								return <div key={row}>
									<p>{limit.paid_fallback ? messageRuntime.t("relay.shell.usage.paid") : limit.family_cap ? messageRuntime.t("relay.shell.usage.family") : messageRuntime.t("relay.shell.usage.included")}
										{limit.family_cap && limit.family !== "unknown" ? ` (${limit.family})` : ""}
										{` · ${limit.window === "5h" ? messageRuntime.t("relay.shell.usage.window_five") : limit.window === "7d" ? messageRuntime.t("relay.shell.usage.window_week") : limit.window === "1mo" ? messageRuntime.t("relay.shell.usage.window_month") : limit.window === "lifetime" ? messageRuntime.t("relay.shell.usage.window_lifetime") : messageRuntime.t("relay.shell.usage.window_unknown")}`}: {value}</p>
									{limit.resets_at_ms !== null && Number.isFinite(new Date(limit.resets_at_ms).getTime()) && <p>{messageRuntime.t("relay.shell.usage.reset", { value: new Date(limit.resets_at_ms).toISOString() })}</p>}
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

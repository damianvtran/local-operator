/** Selected-computer policy editor: metadata and effective state come from the
 * authenticated relay, never device-local routing preferences. Unmount/epoch
 * guards prevent a late read/save from painting a different navigation state.
 */
import { useEffect, useRef, useState } from "react";
import { HttpError, getAccountRouting, setAccountRouting, isAccountRoutingPayload, type AccountRoutingPayload } from "../api";
import { navigateUp } from "../router";
import { Button } from "../components/ui/button";
import { messageRuntime } from "../store";

const copy = (key: "title" | "back" | "loading" | "save" | "saving" | "saved" | "retry" | "unavailable" | "error" | "rejected" | "load_error" | "scope") =>
	messageRuntime.t(`relay.account_policy.${key}`);

export function AccountRoutingScreen() {
	const [payload, setPayload] = useState<AccountRoutingPayload | null>(null);
	const [draft, setDraft] = useState("");
	const [loading, setLoading] = useState(true);
	const [saving, setSaving] = useState(false);
	const [error, setError] = useState("");
	const [saved, setSaved] = useState(false);
	const [revision, setRevision] = useState(0);
	const epoch = useRef(0);
	useEffect(() => {
		const current = ++epoch.current;
		setLoading(true);
		setError("");
		getAccountRouting().then((result) => {
			if (epoch.current !== current) return;
			if (!isAccountRoutingPayload(result)) {
				setError(copy("unavailable"));
				return;
			}
			setPayload(result);
			setDraft(result.setting.value);
		}).catch((failure: unknown) => {
			if (epoch.current !== current) return;
			setError(copy(failure instanceof HttpError && failure.status === 404 ? "unavailable" : "load_error"));
		}).finally(() => {
			if (epoch.current === current) setLoading(false);
		});
		return () => { epoch.current++; };
	}, [revision]);

	async function save() {
		if (!payload || saving) return;
		const current = epoch.current;
		setSaving(true);
		setSaved(false);
		setError("");
		try {
			const result = await setAccountRouting(draft);
			if (epoch.current !== current) return;
			if (!isAccountRoutingPayload(result) || result.setting.value !== draft) throw new Error("incompatible save receipt");
			setPayload(result);
			setDraft(result.setting.value);
			setSaved(true);
		} catch (failure: unknown) {
			if (epoch.current === current) setError(failure instanceof HttpError && failure.code === "setting_write_failed" ? failure.message : copy(failure instanceof HttpError && failure.status >= 400 && failure.status < 500 ? "rejected" : "error"));
		} finally {
			if (epoch.current === current) setSaving(false);
		}
	}

	return <main className="relative mx-auto flex h-dvh w-full max-w-[var(--lo-column-max,28rem)] flex-col bg-surface text-ink" data-testid="account-routing-screen">
		<header className="flex shrink-0 items-center gap-3 border-b border-hairline px-3 pt-[max(env(safe-area-inset-top),0.75rem)] pb-3">
			<Button variant="quiet" onClick={() => navigateUp("/")}>{copy("back")}</Button>
			<h1 className="text-body font-medium">{copy("title")}</h1>
		</header>
		<div className="min-h-0 flex-1 overflow-y-auto p-4 space-y-4">
			<p className="text-ink-muted">{copy("scope")}</p>
			{loading && <p role="status">{copy("loading")}</p>}
			{error && <p role="alert" className="text-danger">{error}</p>}
			{!loading && !payload && <Button onClick={() => setRevision(value => value + 1)}>{copy("retry")}</Button>}
			{!loading && payload && <form className="space-y-4" onSubmit={(event) => { event.preventDefault(); void save(); }}>
				<label htmlFor="account-policy" className="block font-medium">{payload.setting.label}</label>
				<p id="account-policy-help" className="text-ink-muted">{payload.setting.help}</p>
				<select id="account-policy" aria-describedby="account-policy-help" data-testid="account-policy-select" className="min-h-11 w-full rounded-sm border border-control bg-surface px-3" value={draft} disabled={saving} onChange={(event) => { setDraft(event.target.value); setSaved(false); }}>
					{payload.setting.choices.map(choice => <option key={choice.value} value={choice.value}>{choice.label}</option>)}
				</select>
				<p className="text-ink-muted">{payload.setting.choices.find(choice => choice.value === draft)?.description}</p>
				<Button type="submit" variant="primary" disabled={saving || draft === payload.setting.value}>{copy(saving ? "saving" : "save")}</Button>
				{saved && <p role="status">{copy("saved")}</p>}
			</form>}
		</div>
	</main>;
}

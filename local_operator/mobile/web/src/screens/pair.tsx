/**
 * Pair this phone — the `#/pair` screen and the sheet an approval opens in place.
 *
 * WHY THIS CHANGED (design v2 A1/A2, Ben's report). The phone used to need a code
 * printed by a terminal command on the computer; a person who tapped Approve on an
 * unpaired phone was told to go and run it, a dead end for anyone who does not
 * live in a terminal. Now the phone FILES a request (`lib/pair-flow.ts`) and the
 * person approves it on the computer, whose own password is the authority. The
 * old code path survives behind "Using a terminal?" — the only place on these
 * screens a command may appear (D16, pinned by `pair-flow.no-terminal.test.tsx`).
 *
 * THE KEY NEVER LEAVES THE PHONE. It is the WebCrypto non-extractable P-256 key
 * in IndexedDB (`lib/operator-device.ts`); only its public point is sent. No
 * fallback for a browser without `crypto.subtle`: a weaker key wearing this
 * flow's copy would be a different design.
 *
 * IN-PLACE, AND NEVER AUTO-SUBMITTING (D15/U14). An approval tapped on an unpaired
 * phone opens `PairingSheetHost`'s sheet titled "Pair this phone to approve from
 * here". The host lives at the app root, not inside the card, because the card
 * unmounts the moment its request is answered elsewhere — and the person still
 * needs to finish pairing and be told the request is gone. After `paired` the
 * sheet says "tap Approve again": approving is the person's gesture, and a
 * pairing that silently approved would turn one consent into two.
 */
import { useEffect, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { claimPairingCode, HttpError, pairingStatus } from "../api";
import { Button } from "../components/ui/button";
import { Disclosure } from "../components/ui/disclosure";
import { Sheet } from "../components/ui/sheet";
import {
	clearDeviceKeyLost,
	deviceIdFor,
	deviceKeyWasLost,
	encodePoint,
	loadOrCreateDeviceKey,
	storeCertificate,
} from "../lib/operator-device";
import {
	AVAILABILITY_POLL_MS,
	checkAvailability,
	clearPendingPair,
	completePairing,
	fileRequest,
	formatCountdown,
	isStandalone,
	NOT_SEEN_HINT_MS,
	pollStatus,
	rekeyAfterRevocation,
	requestPersistentStorage,
	savedPendingPair,
	sasFor,
	STATUS_POLL_MS,
	TERMINAL_REASONS,
	withdrawRequest,
	type PendingPair,
} from "../lib/pair-flow";
import { isolate } from "../lib/bidi";
import { groupSas } from "../lib/pair-sas";
import { navigate } from "../router";
import { messageRuntime, useLocaleSnapshot } from "../store";

/** What the sheet was opened to let the person do next. `approve` is a pending
    card's Approve; `choose` is a gate-sheet choice that needs a paired phone. */
export type PairPurpose = "approve" | "choose";

/** A fact to state before the intro: the phone was removed (U15), or the
    computer refused its signature. The key-lost fact (U16) is read from storage. */
export type PairNotice = "revoked" | "refused";

type Ended = "denied" | "expired" | "withdrawn" | "strikes";

type Step =
	| { kind: "intro" }
	| { kind: "checking" }
	| { kind: "unavailable" }
	| { kind: "unsupported" }
	| { kind: "filing" }
	| { kind: "waiting"; pending: PendingPair; sas: string; reason: string; since: number }
	| { kind: "ended"; why: Ended }
	| { kind: "paired"; authorityReady: boolean }
	| { kind: "busy_same"; requestId: string; resumable: boolean }
	| { kind: "busy_other" }
	| { kind: "rate_limited"; until: number | null }
	| { kind: "revoked" }
	/* `code` is the refusal's machine code (or HTTP status) shown small for
	   support; empty for a transport failure, which has none. */
	| { kind: "error"; message: string; code: string };

const ENDED_BY_REASON: Record<string, Ended> = {
	denied_by_user: "denied",
	expired: "expired",
	withdrawn: "withdrawn",
	strikes_exhausted: "strikes",
};

/** The machine-facing name the computer shows beside the request. Editable only
    on the terminal path (the computer's card shows platform and account too). */
function defaultDeviceName(): string {
	const platform = typeof navigator === "undefined" ? "" : navigator.platform || "";
	/* A device identifier, not UI copy: "iPhone" is a product name, and the
	   computer shows this string verbatim beside the request, like a hostname. */
	const model = platform.includes("iPhone") || platform.includes("iPad") ? "iPhone" : "Phone";
	return `${model} (${new Date().toISOString().slice(0, 10)})`;
}

/**
 * The sentence for a failure the flow does not model as its own step.
 *
 * TRANSPORT VS REFUSAL (QA Q48-2, UX U1). Every failure used to read "Couldn't
 * reach your computer. Check the connection…", including a 422 the computer
 * answered within milliseconds — so a person toggled Wi-Fi and tapped Try again
 * into the same refusal forever. An `HttpError` means the computer ANSWERED, so
 * it gets the refusal sentence (try again, or update the computer); the network
 * sentence is kept for failures where no answer arrived (fetch's TypeError, an
 * abort, a timeout).
 */
function humanizeFailure(error: unknown): string {
	const message = String((error as Error)?.message ?? error);
	if (message.toLowerCase().includes("subtle")) return messageRuntime.t("relay.pairing.error.no_crypto");
	if (error instanceof HttpError) return messageRuntime.t("relay.pairing.error.refused");
	return messageRuntime.t("relay.pairing.error.network");
}

/** The refusal's code for support, or its HTTP status when it carried none. */
function supportCode(error: unknown): string {
	if (!(error instanceof HttpError)) return "";
	return error.code || String(error.status);
}

function errorStep(error: unknown): Step {
	return { kind: "error", message: humanizeFailure(error), code: supportCode(error) };
}

/** Minutes left, rounded up: the rate-limit announcement changes only when
    this does (U5c), so a screen reader hears it at most once a minute. */
function minuteBucket(seconds: number): number {
	return Math.max(0, Math.ceil(seconds / 60));
}

/** One clock for the countdowns, ticking only while a screen shows one. */
function useNow(active: boolean): number {
	const [now, setNow] = useState(() => Date.now());
	useEffect(() => {
		if (!active) return;
		setNow(Date.now());
		const timer = setInterval(() => setNow(Date.now()), 1000);
		return () => clearInterval(timer);
	}, [active]);
	return now;
}

export function PairFlow({
	purpose,
	actionGone = false,
	notice,
	onDismiss,
}: {
	purpose?: PairPurpose;
	/** The action that opened the sheet was answered or expired meanwhile. */
	actionGone?: boolean;
	notice?: PairNotice;
	/** "Not now", "Done" and "Back to the request" all hand control back. */
	onDismiss: () => void;
}) {
	useLocaleSnapshot();
	const [step, setStep] = useState<Step>(() =>
		notice === "revoked" ? { kind: "revoked" } : { kind: "intro" },
	);
	const alive = useRef(true);
	useEffect(() => {
		alive.current = true;
		return () => {
			alive.current = false;
		};
	}, []);
	const set = (next: Step) => {
		if (alive.current) setStep(next);
	};

	/* U5a: each step REPLACES the sheet body, so the button the person just
	   pressed unmounts and focus would fall to <body> inside the modal, leaving a
	   screen-reader user to hunt for where they are. Move focus to the new step's
	   lead line (`data-step-heading`) on every step change after the first — on
	   open, the Sheet's own focus on its close button stays the entry point. */
	const bodyRef = useRef<HTMLDivElement>(null);
	const firstStep = useRef(true);
	useEffect(() => {
		if (firstStep.current) {
			firstStep.current = false;
			return;
		}
		bodyRef.current?.querySelector<HTMLElement>("[data-step-heading]")?.focus();
	}, [step.kind]);

	/* A request filed before a reload is resumed, not orphaned: the computer is
	   still showing it, and filing a second one would be refused as busy. */
	useEffect(() => {
		const saved = savedPendingPair();
		if (saved) void resume(saved);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	/* `reportFailure`: a resume the person ASKED for (Try again after a failure)
	   says why it failed and keeps the saved request; the silent resume on open
	   drops an unreadable request instead of greeting the person with an error. */
	async function resume(saved: PendingPair, reportFailure = false) {
		try {
			const status = await pollStatus(saved.requestId);
			if (TERMINAL_REASONS.has(status.reason)) {
				await settle(saved, status.reason);
				return;
			}
			set({
				kind: "waiting",
				pending: saved,
				sas: await sasFor(saved),
				reason: status.reason,
				since: Date.now(),
			});
		} catch (error) {
			if (reportFailure) {
				set(errorStep(error));
				return;
			}
			/* An unreadable request is not worth a scary error on open: drop it
			   and let the person start again from the intro. */
			clearPendingPair();
		}
	}

	/* U2: "Try again" after a failure RESUMES a request this phone still holds
	   rather than filing a new one. The common case is the certificate fetch
	   failing after the computer already admitted the phone (`settle` keeps the
	   saved request until `completePairing` succeeds): filing again would send
	   the person back to the computer for a second approval and password for a
	   pairing that already happened. Files only when nothing is pending. */
	function retry() {
		const saved = savedPendingPair();
		if (saved) void resume(saved, true);
		else void start();
	}

	async function settle(pending: PendingPair, reason: string) {
		if (reason === "paired") {
			try {
				const done = await completePairing(pending.deviceId);
				set({ kind: "paired", authorityReady: done.authorityReady });
			} catch (error) {
				set(errorStep(error));
			}
			return;
		}
		clearPendingPair();
		set({ kind: "ended", why: ENDED_BY_REASON[reason] ?? "expired" });
	}

	async function start() {
		set({ kind: "checking" });
		try {
			const availability = await checkAvailability();
			if (availability === "unsupported") return set({ kind: "unsupported" });
			if (availability === "unavailable") return set({ kind: "unavailable" });
			await file();
		} catch (error) {
			set(errorStep(error));
		}
	}

	async function file() {
		set({ kind: "filing" });
		try {
			const outcome = await fileRequest(defaultDeviceName());
			switch (outcome.kind) {
				case "filed":
					return set({
						kind: "waiting",
						pending: outcome.pending,
						sas: await sasFor(outcome.pending),
						reason: "waiting",
						since: Date.now(),
					});
				case "busy_same":
					return set({
						kind: "busy_same",
						requestId: outcome.requestId,
						resumable: savedPendingPair()?.requestId === outcome.requestId,
					});
				case "busy_other":
					return set({ kind: "busy_other" });
				case "rate_limited":
					return set({
						kind: "rate_limited",
						until: outcome.retryAfterS === null ? null : Date.now() + outcome.retryAfterS * 1000,
					});
				case "revoked":
					return set({ kind: "revoked" });
				case "unavailable":
					return set({ kind: "unavailable" });
			}
		} catch (error) {
			set(errorStep(error));
		}
	}

	/* D15: while the not-available screen is visible, re-check every 5 s and
	   file by itself once phone approvals are turned on — the person already
	   asked to pair, so making them tap again after fixing the computer is a
	   second chore for one intent. */
	useEffect(() => {
		if (step.kind !== "unavailable") return;
		let stopped = false;
		const timer = setInterval(async () => {
			try {
				if ((await checkAvailability()) === "available" && !stopped) {
					stopped = true;
					clearInterval(timer);
					await file();
				}
			} catch {
				/* A missed re-check is retried on the next tick. */
			}
		}, AVAILABILITY_POLL_MS);
		return () => {
			stopped = true;
			clearInterval(timer);
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [step.kind]);

	/* The status poll, 2 s while the waiting screen is up (UI cadence only). */
	const waitingId = step.kind === "waiting" ? step.pending.requestId : "";
	useEffect(() => {
		if (!waitingId) return;
		let stopped = false;
		const timer = setInterval(async () => {
			const current = savedPendingPair();
			if (!current || current.requestId !== waitingId) return;
			try {
				const status = await pollStatus(waitingId);
				if (stopped) return;
				if (TERMINAL_REASONS.has(status.reason)) {
					stopped = true;
					clearInterval(timer);
					await settle(current, status.reason);
					return;
				}
				setStep((prev) =>
					prev.kind === "waiting" && prev.pending.requestId === waitingId
						? { ...prev, reason: status.reason }
						: prev,
				);
			} catch {
				/* Transient: the next tick asks again. */
			}
		}, STATUS_POLL_MS);
		return () => {
			stopped = true;
			clearInterval(timer);
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [waitingId]);

	const [cancelNote, setCancelNote] = useState("");
	async function cancel(requestId: string, thenFile: boolean) {
		setCancelNote("");
		try {
			const outcome = await withdrawRequest(requestId);
			if (outcome === "in_progress") {
				setCancelNote(messageRuntime.t("relay.pairing.withdraw_in_progress"));
				return;
			}
			if (thenFile) await file();
			else set({ kind: "ended", why: "withdrawn" });
		} catch {
			setCancelNote(messageRuntime.t("relay.pairing.withdraw_failed"));
		}
	}

	const ticking = step.kind === "waiting" || (step.kind === "rate_limited" && step.until !== null);
	const now = useNow(ticking);
	const terminalPath = (
		<TerminalDisclosure onPaired={(authorityReady) => set({ kind: "paired", authorityReady })} />
	);
	/* N3: every step that is not mid-request offers the same text exit, so
	   leaving never depends on reading ✕ as "leave". */
	const notNow = (
		<Button variant="ghost" onClick={onDismiss}>
			{messageRuntime.t("relay.pairing.intro.not_now")}
		</Button>
	);

	/* U5b/U5c: ONE polite live region that exists for the sheet's whole life and
	   whose TEXT changes. A `role="status"` node that mounts already holding its
	   text is often not announced at all, so the step lines below are plain text
	   (focus moves to each step's lead line instead, above) and this region
	   carries what changes WITHOUT a step change: the code appearing, the computer
	   confirming, a retryable failure, and the rate limit. The rate-limit
	   countdown on screen ticks every second, but the announcement is pinned to
	   the minute (`minuteBucket`), so it is spoken at most once a minute and once
	   more when Try again unlocks. */
	const rateLeft = step.kind === "rate_limited" && step.until !== null ? (step.until - now) / 1000 : 0;
	const rateBucket = step.kind === "rate_limited" && step.until !== null ? minuteBucket(rateLeft) : -1;
	const [rateSaid, setRateSaid] = useState({ bucket: -1, text: "" });
	if (rateBucket !== rateSaid.bucket) {
		setRateSaid({
			bucket: rateBucket,
			text:
				rateBucket < 0
					? ""
					: rateBucket === 0
						? messageRuntime.t("relay.pairing.rate_limited_over")
						: messageRuntime.t("relay.pairing.rate_limited", { countdown: formatCountdown(rateLeft) }),
		});
	}
	const stepAnnouncement =
		step.kind === "waiting"
			? step.reason === "signing" || step.reason === "admin_prompt_open"
				? messageRuntime.t("relay.pairing.signing")
				: step.reason === "failed_retryable"
					? messageRuntime.t("relay.pairing.failed_retryable")
					: messageRuntime.t("relay.pairing.code.aria", { digits: step.sas.split("").join(" ") })
			: step.kind === "rate_limited"
				? step.until === null
					? messageRuntime.t("relay.pairing.rate_limited_later")
					: rateSaid.text
				: "";
	/* A cancel that did not go through is said where the person is looking AND
	   announced, since it changes nothing else on screen. */
	const announcement = cancelNote || stepAnnouncement;

	return (
		<div ref={bodyRef}>
			<p className="sr-only" role="status" aria-live="polite" data-pair-live="">
				{announcement}
			</p>
			{renderStep()}
		</div>
	);

	function renderStep(): ReactNode {
		switch (step.kind) {
			case "intro": {
				const lead =
					notice === "refused"
						? messageRuntime.t("relay.pairing.refused_signature")
						: deviceKeyWasLost()
							? messageRuntime.t("relay.pairing.key_lost")
							: "";
				return (
					<FlowBody>
						{lead ? <p className="text-body-sm text-danger">{lead}</p> : null}
						<Lead className="text-body-sm text-ink-muted">{messageRuntime.t("relay.pairing.intro.body")}</Lead>
						<div className="flex flex-col gap-2">
							<Button variant="primary" onClick={() => void start()}>
								{messageRuntime.t("relay.pairing.intro.pair")}
							</Button>
							{notNow}
						</div>
						{terminalPath}
					</FlowBody>
				);
			}
			case "checking":
			case "filing":
				return (
					<FlowBody>
						<Lead className="text-body-sm text-ink-muted">
							{step.kind === "checking"
								? messageRuntime.t("relay.pairing.checking")
								: messageRuntime.t("relay.pairing.filing")}
						</Lead>
					</FlowBody>
				);
			case "unavailable":
				return (
					<FlowBody>
						<Lead className="text-body-sm">{messageRuntime.t("relay.pairing.unavailable.body")}</Lead>
						<p className="text-body-sm text-ink-muted">{messageRuntime.t("relay.pairing.unavailable.waiting")}</p>
						{notNow}
						<TerminalDisclosure approvalsTip />
					</FlowBody>
				);
			case "unsupported":
				return (
					<FlowBody>
						<Lead className="text-body-sm">{messageRuntime.t("relay.pairing.unsupported.body")}</Lead>
						<Button onClick={() => void start()}>{messageRuntime.t("relay.pairing.retry")}</Button>
						{notNow}
						{terminalPath}
					</FlowBody>
				);
			case "waiting": {
				const confirming = step.reason === "signing" || step.reason === "admin_prompt_open";
				const remaining = step.pending.expiresAt ? step.pending.expiresAt - now / 1000 : 0;
				return (
					<FlowBody>
						<Lead as="h2" className="text-body font-medium">
							{messageRuntime.t("relay.pairing.waiting.title")}
						</Lead>
						<SasCode code={step.sas} />
						{confirming ? (
							<p className="text-body-sm">{messageRuntime.t("relay.pairing.signing")}</p>
						) : (
							<>
								{step.reason === "failed_retryable" ? (
									<p className="text-body-sm text-danger">{messageRuntime.t("relay.pairing.failed_retryable")}</p>
								) : null}
								<p className="text-body-sm">
									{step.pending.mode === "confirm"
										? messageRuntime.t("relay.pairing.waiting.confirm_body")
										: messageRuntime.t("relay.pairing.waiting.code_body")}
								</p>
								<p className="text-body-sm text-ink-muted">{messageRuntime.t("relay.pairing.waiting.password")}</p>
								{/* The countdown hides while the computer is confirming: the
								    window bounds the DECISION, not a decision already made (U11). */}
								{remaining > 0 ? (
									<p className="text-meta text-ink-muted tabular-nums">
										{messageRuntime.t("relay.pairing.waiting.expires", {
											countdown: formatCountdown(remaining),
										})}
									</p>
								) : null}
								{now - step.since >= NOT_SEEN_HINT_MS ? (
									<p className="text-meta text-ink-muted">{messageRuntime.t("relay.pairing.waiting.not_seen")}</p>
								) : null}
							</>
						)}
						{cancelNote ? <p className="text-body-sm text-danger">{cancelNote}</p> : null}
						<Button variant="ghost" onClick={() => void cancel(step.pending.requestId, false)}>
							{messageRuntime.t("relay.pairing.waiting.cancel")}
						</Button>
					</FlowBody>
				);
			}
			case "ended": {
				const key = {
					denied: "relay.pairing.denied",
					expired: "relay.pairing.expired",
					withdrawn: "relay.pairing.withdrawn",
					strikes: "relay.pairing.strikes_exhausted",
				} as const;
				return (
					<FlowBody>
						<Lead className="text-body-sm">{messageRuntime.t(key[step.why])}</Lead>
						<Button variant="primary" onClick={() => void start()}>
							{messageRuntime.t("relay.pairing.pair_again")}
						</Button>
						{notNow}
					</FlowBody>
				);
			}
			case "paired":
				return (
					<PairedBody
						authorityReady={step.authorityReady}
						purpose={purpose}
						actionGone={actionGone}
						onDismiss={onDismiss}
					/>
				);
			case "busy_same":
				return (
					<FlowBody>
						<Lead className="text-body-sm">{messageRuntime.t("relay.pairing.busy.same_key")}</Lead>
						{step.resumable ? (
							<Button
								variant="primary"
								onClick={() => {
									const saved = savedPendingPair();
									if (saved) void resume(saved);
								}}
							>
								{messageRuntime.t("relay.pairing.busy.resume")}
							</Button>
						) : null}
						<Button onClick={() => void cancel(step.requestId, true)}>
							{messageRuntime.t("relay.pairing.busy.cancel_earlier")}
						</Button>
						{cancelNote ? <p className="text-body-sm text-danger">{cancelNote}</p> : null}
						{notNow}
					</FlowBody>
				);
			case "busy_other":
				return (
					<FlowBody>
						<Lead className="text-body-sm">{messageRuntime.t("relay.pairing.busy.other")}</Lead>
						{/* U3: Safari and a Home Screen web app keep separate storage on
						    iOS, so the same phone pairing from both holds two different
						    keys. The computer cannot tell that apart from a second phone
						    (any "same phone" signal would be a fingerprint disclosed to a
						    different key, which this refusal must not do), so the hint is
						    always shown, hedged with "might". */}
						<p className="text-body-sm text-ink-muted">{messageRuntime.t("relay.pairing.busy.maybe_this_phone")}</p>
						<Button onClick={() => void start()}>{messageRuntime.t("relay.pairing.retry")}</Button>
						{notNow}
					</FlowBody>
				);
			case "rate_limited":
				return (
					<FlowBody>
						<Lead className="text-body-sm tabular-nums">
							{step.until === null
								? messageRuntime.t("relay.pairing.rate_limited_later")
								: messageRuntime.t("relay.pairing.rate_limited", { countdown: formatCountdown(rateLeft) })}
						</Lead>
						<Button disabled={rateLeft > 0} onClick={() => void start()}>
							{messageRuntime.t("relay.pairing.retry")}
						</Button>
						{notNow}
					</FlowBody>
				);
			case "revoked":
				return (
					<FlowBody>
						<Lead className="text-body-sm">{messageRuntime.t("relay.pairing.revoked")}</Lead>
						<Button variant="primary" onClick={() => void start()}>
							{messageRuntime.t("relay.pairing.pair_again")}
						</Button>
						{notNow}
					</FlowBody>
				);
			case "error":
				return (
					<FlowBody>
						<Lead className="text-body-sm text-danger">{step.message}</Lead>
						{step.code ? (
							<p className="text-meta text-ink-muted">
								{messageRuntime.t("relay.pairing.error.support_code", { code: isolate(step.code) })}
							</p>
						) : null}
						<Button onClick={retry}>{messageRuntime.t("relay.pairing.retry")}</Button>
						{notNow}
						{terminalPath}
					</FlowBody>
				);
		}
	}
}

function FlowBody({ children }: { children: ReactNode }) {
	return <div className="flex flex-col gap-3">{children}</div>;
}

/** A step's lead line: where focus lands when the step appears (U5a).
    `tabIndex={-1}` makes it focusable by script only, never a tab stop. */
function Lead({ as = "p", className, children }: { as?: "p" | "h2"; className: string; children: ReactNode }) {
	const Tag = as;
	return (
		<Tag data-step-heading="" tabIndex={-1} className={`${className} outline-none`}>
			{children}
		</Tag>
	);
}

/** The six digits, grouped as the computer draws them, read digit by digit by a
    screen reader (a "four hundred eighty-two thousand…" reading cannot be compared). */
function SasCode({ code }: { code: string }) {
	return (
		<p
			/* D6: one step above the sheet title (text-display, 28 px): the code is
			   the thing the person compares with a second screen, so it is the
			   hero of this step. */
			className="rounded-sm bg-sunken px-3 py-2 text-center font-mono text-display tracking-widest tabular-nums"
			dir="ltr"
			aria-label={messageRuntime.t("relay.pairing.code.aria", { digits: code.split("").join(" ") })}
		>
			{groupSas(code)}
		</p>
	);
}

function PairedBody({
	authorityReady,
	purpose,
	actionGone,
	onDismiss,
}: {
	authorityReady: boolean;
	purpose?: PairPurpose;
	actionGone: boolean;
	onDismiss: () => void;
}) {
	/* Back to the action, never through it (D15): the sheet names the next tap
	   and closes; the approval itself stays the person's gesture. */
	const returnLine =
		purpose === "approve"
			? messageRuntime.t(actionGone ? "relay.pairing.return.gone" : "relay.pairing.return.approve")
			: purpose === "choose"
				? messageRuntime.t("relay.pairing.return.choose")
				: "";
	const backLabel =
		purpose && !actionGone
			? messageRuntime.t("relay.pairing.return.back")
			: messageRuntime.t("relay.pairing.paired.done");
	return (
		<FlowBody>
			<Lead className="text-body font-medium">{messageRuntime.t("relay.pairing.paired.title")}</Lead>
			<p className="text-body-sm text-ink-muted">
				{authorityReady
					? messageRuntime.t("relay.pairing.paired.body")
					: messageRuntime.t("relay.pairing.paired.not_ready")}
			</p>
			{returnLine ? <p className="text-body-sm">{returnLine}</p> : null}
			{isStandalone() ? null : (
				<p className="text-meta text-ink-muted">{messageRuntime.t("relay.pairing.paired.home_screen")}</p>
			)}
			<Button variant="primary" onClick={onDismiss}>
				{backLabel}
			</Button>
		</FlowBody>
	);
}

/** Commands, never translated, and rendered ONLY inside the disclosure below
    (D16): the shell command that prints a pairing code, and the terminal app's
    slash command that turns phone approvals on (U17). */
const PAIR_SHELL_COMMAND = "lop pair";
const APPROVALS_SLASH_COMMAND = "/mobile approvals on";

/** How often the terminal path asks whether a claimed code was approved, and for
    how long. Unchanged from the code-entry screen this path used to be. */
const CODE_POLL_MS = 1500;
const CODE_POLL_TIMEOUT_MS = 10 * 60 * 1000;

/**
 * "Using a terminal?" — the ONLY place on these screens a command may appear
 * (D16/U19). `data-pair-terminal` is the boundary the no-terminal test scans
 * around. Holds the old code-entry pairing (a daemon from before terminal-free
 * pairing supports only that) and, on the not-available screen, the terminal
 * app's way to turn phone approvals on (U17).
 */
function TerminalDisclosure({
	approvalsTip = false,
	onPaired,
}: {
	approvalsTip?: boolean;
	onPaired?: (authorityReady: boolean) => void;
}) {
	return (
		<div data-pair-terminal="">
			<Disclosure
				header={<span className="text-body-sm text-ink-muted">{messageRuntime.t("relay.pairing.terminal.header")}</span>}
			>
				<div className="flex flex-col gap-3 pt-2">
					{approvalsTip ? (
						<p className="text-body-sm text-ink-muted">
							{/* ISOLATED (i18n audit, ur): a command's leading `/` is a neutral
							   character, so inside right-to-left copy it detaches to the far end
							   of the run unless the command is its own isolate (lib/bidi.ts). */}
							<WithCommand messageKey="relay.pairing.terminal.turn_on_approving" command={APPROVALS_SLASH_COMMAND} />
						</p>
					) : null}
					{onPaired ? <CodeEntry onPaired={onPaired} /> : null}
				</div>
			</Disclosure>
		</div>
	);
}

/** Splits a message on this private-use marker to put an element where its
    `{command}` placeholder was; never a character a catalogue can contain. */
const COMMAND_SLOT = "\uE000";

/**
 * A disclosure sentence with its `{command}` drawn as a command (D7): the
 * monospace inline-code run the transcript uses for code, so `lop pair` reads as
 * something to type rather than as words in the sentence. The catalogue keeps
 * the whole sentence (translators move the placeholder where their grammar puts
 * it); only the render splits it. The command stays its own bidi isolate inside
 * the element, as the ur audit required (lib/bidi.ts).
 */
function WithCommand({
	messageKey,
	command,
}: {
	messageKey: "relay.pairing.terminal.code_intro" | "relay.pairing.terminal.turn_on_approving";
	command: string;
}) {
	const [before, after = ""] = messageRuntime.t(messageKey, { command: COMMAND_SLOT }).split(COMMAND_SLOT);
	return (
		<>
			{before}
			<code dir="ltr" className="rounded-xs bg-sunken px-0.5 font-mono text-mono-sm text-ink">
				{isolate(command)}
			</code>
			{after}
		</>
	);
}

/** The one sentence a pre-code daemon's `/api/pair` sent for a revoked key (403). */
const LEGACY_REVOKED_SENTENCE = "this device has been revoked";

/**
 * Whether a refusal says THIS phone's key is revoked — the only case that may run
 * `rekeyAfterRevocation`, which deletes the device key and certificate.
 *
 * EXACT MATCHES ONLY (round-1 R5). This used to fire on any message containing
 * "revoked", so on an already-paired phone an unrelated refusal whose prose named a
 * revocation (an anchor's, the operator key's) silently unpaired it. Now: the typed
 * `pair_device_revoked` code, or a 403 carrying the exact legacy sentence an older
 * daemon sent before the code existed.
 */
export function isDeviceRevokedRefusal(error: unknown): boolean {
	if (!(error instanceof HttpError)) return false;
	if (error.code === "pair_device_revoked") return true;
	return error.code === "" && error.status === 403 && error.message === LEGACY_REVOKED_SENTENCE;
}

function CodeEntry({ onPaired }: { onPaired: (authorityReady: boolean) => void }) {
	const [code, setCode] = useState("");
	const [name, setName] = useState(() => defaultDeviceName());
	const [state, setState] = useState<"idle" | "waiting" | "failed">("idle");
	const [message, setMessage] = useState("");
	const cancelled = useRef(false);
	useEffect(() => {
		cancelled.current = false;
		return () => {
			cancelled.current = true;
		};
	}, []);

	async function claim() {
		setState("idle");
		setMessage("");
		try {
			const key = await loadOrCreateDeviceKey();
			const deviceId = await deviceIdFor(key.point);
			await claimPairingCode({
				code: code.trim(),
				spki: encodePoint(key.point),
				name: name.trim() || defaultDeviceName(),
			});
			setState("waiting");
			const deadline = Date.now() + CODE_POLL_TIMEOUT_MS;
			while (!cancelled.current && Date.now() < deadline) {
				await new Promise((resolve) => setTimeout(resolve, CODE_POLL_MS));
				if (cancelled.current) return;
				const answer = await pairingStatus(deviceId);
				if (answer.paired && answer.certificate) {
					/* PUBLIC data — kept outside the storage sign-out clears, so a
					   sign-out does not silently unhook a paired phone. */
					storeCertificate(answer.certificate, answer.operator_key_id ?? deviceId);
					clearDeviceKeyLost();
					await requestPersistentStorage();
					onPaired(answer.authority_ready !== false);
					return;
				}
			}
			if (!cancelled.current) {
				setState("failed");
				setMessage(messageRuntime.t("relay.pairing.terminal.no_answer"));
			}
		} catch (error) {
			setState("failed");
			if (isDeviceRevokedRefusal(error)) {
				await rekeyAfterRevocation();
				setMessage(messageRuntime.t("relay.pairing.revoked"));
			} else if (String((error as Error)?.message ?? error).includes("not valid")) {
				setMessage(messageRuntime.t("relay.pairing.terminal.invalid_code"));
			} else {
				setMessage(humanizeFailure(error));
			}
		}
	}

	return (
		<>
			<p className="text-body-sm text-ink-muted">
				<WithCommand messageKey="relay.pairing.terminal.code_intro" command={PAIR_SHELL_COMMAND} />
			</p>
			<label className="flex flex-col gap-1 text-body-sm">
				<span className="text-ink-muted">{messageRuntime.t("relay.pairing.terminal.code_label")}</span>
				<input
					dir="ltr"
					value={code}
					onChange={(event) => setCode(event.target.value)}
					autoCapitalize="none"
					autoCorrect="off"
					spellCheck={false}
					placeholder={messageRuntime.t("relay.pairing.terminal.code_placeholder")}
					className="min-h-11 rounded-sm border border-control bg-surface px-3 font-mono text-body text-ink outline-none placeholder:text-ink-dim"
				/>
			</label>
			<label className="flex flex-col gap-1 text-body-sm">
				<span className="text-ink-muted">{messageRuntime.t("relay.pairing.terminal.name_label")}</span>
				<input
					value={name}
					onChange={(event) => setName(event.target.value)}
					className="min-h-11 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none"
				/>
			</label>
			<Button disabled={state === "waiting" || code.trim().length === 0} onClick={() => void claim()}>
				{state === "waiting" ? "…" : messageRuntime.t("relay.pairing.terminal.submit")}
			</Button>
			{state === "waiting" ? (
				<p role="status" className="text-body-sm text-ink-muted">
					{messageRuntime.t("relay.pairing.terminal.waiting", { name })}
				</p>
			) : null}
			{state === "failed" && message ? <p className="text-body-sm text-danger">{message}</p> : null}
		</>
	);
}

/** The full-screen route (`#/pair`): the same flow with a header. */
export function PairScreen() {
	useLocaleSnapshot();
	return (
		<div className="mx-auto flex h-dvh w-full max-w-[520px] flex-col gap-3 overflow-y-auto px-3 pb-[max(env(safe-area-inset-bottom),1.5rem)] pt-[max(env(safe-area-inset-top),0.5rem)]">
			<header className="flex items-center gap-2">
				<button
					type="button"
					onClick={() => navigate("/")}
					aria-label={messageRuntime.t("relay.pairing.back")}
					className="flex min-h-11 min-w-11 items-center justify-center rounded-sm text-ink-muted active:bg-elevated"
				>
					<span aria-hidden dir="ltr" className="lo-directional">‹</span>
				</button>
				<h1 className="min-w-0 flex-1 truncate text-body font-medium">
					{messageRuntime.t("relay.pairing.screen_title")}
				</h1>
			</header>
			<h2 className="text-body font-medium">{messageRuntime.t("relay.pairing.title")}</h2>
			<PairFlow onDismiss={() => navigate("/")} />
		</div>
	);
}

/** The sheet itself; `PairingSheetHost` decides when it shows. */
export function PairPromptSheet({
	open,
	onClose,
	purpose,
	actionGone = false,
	notice,
}: {
	open: boolean;
	onClose: () => void;
	purpose?: PairPurpose;
	actionGone?: boolean;
	notice?: PairNotice;
}) {
	useLocaleSnapshot();
	return (
		<Sheet
			open={open}
			onClose={onClose}
			title={
				purpose
					? messageRuntime.t("relay.pairing.title_for_action")
					: messageRuntime.t("relay.pairing.title")
			}
		>
			<div className="p-3">
				{open ? (
					<PairFlow purpose={purpose} actionGone={actionGone} notice={notice} onDismiss={onClose} />
				) : null}
			</div>
		</Sheet>
	);
}

// ---------------------------------------------------------------------------
// The app-root host: one pairing sheet that outlives the card that opened it
// ---------------------------------------------------------------------------

type SheetState = {
	purpose: PairPurpose;
	/** The action the sheet returns to; `markPairingActionGone` flips `gone`. */
	actionId?: string;
	notice?: PairNotice;
	gone: boolean;
	/** Bumped per open so a re-open starts the flow fresh. */
	serial: number;
};

let sheetState: SheetState | null = null;
let serial = 0;
const sheetListeners = new Set<() => void>();

function emitSheet(next: SheetState | null) {
	sheetState = next;
	for (const listener of sheetListeners) listener();
}

/** Open the pairing sheet in place (D15). The caller's action is NOT retried. */
export function openPairingSheet(input: { purpose: PairPurpose; actionId?: string; notice?: PairNotice }) {
	serial += 1;
	emitSheet({ ...input, gone: false, serial });
}

/** The action the open sheet would return to was answered or expired. */
export function markPairingActionGone(actionId: string) {
	if (sheetState && sheetState.actionId === actionId && !sheetState.gone) {
		emitSheet({ ...sheetState, gone: true });
	}
}

export function closePairingSheet() {
	if (sheetState !== null) emitSheet(null);
}

function subscribeSheet(listener: () => void) {
	sheetListeners.add(listener);
	return () => {
		sheetListeners.delete(listener);
	};
}

/** Mounted once at the app root (see `app.tsx`). */
export function PairingSheetHost() {
	const state = useSyncExternalStore(subscribeSheet, () => sheetState, () => null);
	return (
		<PairPromptSheet
			key={state?.serial ?? 0}
			open={state !== null}
			onClose={closePairingSheet}
			purpose={state?.purpose}
			actionGone={state?.gone ?? false}
			notice={state?.notice}
		/>
	);
}

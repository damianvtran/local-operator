/**
 * Pair this phone (stage D of issue #1310) — `#/pair`.
 *
 * WHY A SCREEN AND NOT A BUTTON IN SETTINGS. Pairing is the one moment the
 * operator authorises THIS device, and it needs two facts from two places at once:
 * a code minted on the machine by `lop pair`, and a key generated here. A screen
 * can hold both plus the waiting state between them, which is what the flow
 * actually is.
 *
 * THE ORDER OF THE TWO STEPS IS THE SECURITY SHAPE, and it is deliberate:
 *
 *   1. generate the key HERE, non-extractable, and never send it;
 *   2. send the PUBLIC point with the code;
 *   3. poll until the operator's own gesture on the machine has produced the
 *      certificate, then keep it.
 *
 * Nothing this screen sends can be turned into a signature by the machine, and the
 * code alone produces a PENDING request that `lop pair` shows and the operator still
 * has to approve. A stolen code is therefore not a stolen device — it is a request
 * the operator reads the name of before answering.
 *
 * NO WEB-CRYPTO FALLBACK. A browser without `crypto.subtle` gets the refusal rather
 * than a weaker scheme: the whole point of the device tier is that the private half
 * cannot be exported, and an implementation that exported it so it could "still
 * work" would be a different design wearing this one's copy.
 */
import { useEffect, useRef, useState } from "react";
import { claimPairingCode, pairingStatus } from "../api";
import { Button } from "../components/ui/button";
import { Sheet } from "../components/ui/sheet";
import {
	DEVICE_SCOPES,
	deviceIdFor,
	encodePoint,
	loadOrCreateDeviceKey,
	storeCertificate,
} from "../lib/operator-device";
import { navigate } from "../router";

/** How often the screen asks whether the operator has answered. Chosen against the
    operator's own rhythm rather than a network budget: they are looking at a
    terminal on another device and typing a name, which takes seconds. */
const POLL_INTERVAL_MS = 1500;

/** How long the screen waits before it stops asking and says so. Long enough for a
    careful reader, short enough that a forgotten tab does not poll forever. */
const POLL_TIMEOUT_MS = 10 * 60 * 1000;

type Status =
	| { kind: "idle" }
	| { kind: "waiting"; deviceId: string }
	| { kind: "paired"; deviceId: string; name: string }
	| { kind: "failed"; message: string };

export function PairScreen() {
	const [code, setCode] = useState("");
	const [name, setName] = useState(() => defaultDeviceName());
	const [status, setStatus] = useState<Status>({ kind: "idle" });
	const cancelled = useRef(false);

	useEffect(() => {
		cancelled.current = false;
		return () => {
			cancelled.current = true;
		};
	}, []);

	async function claim() {
		setStatus({ kind: "idle" });
		try {
			const key = await loadOrCreateDeviceKey();
			const deviceId = await deviceIdFor(key.point);
			await claimPairingCode({
				code: code.trim(),
				spki: encodePoint(key.point),
				name: name.trim() || defaultDeviceName(),
			});
			setStatus({ kind: "waiting", deviceId });
			const deadline = Date.now() + POLL_TIMEOUT_MS;
			while (!cancelled.current && Date.now() < deadline) {
				await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
				if (cancelled.current) return;
				const answer = await pairingStatus(deviceId);
				if (answer.paired && answer.certificate) {
					/* The certificate is PUBLIC data — a statement over this phone's
					   public point plus the operator's signature — so it is kept in
					   ordinary storage rather than the private storage the sign-out
					   path clears. Losing it would silently unhook a paired phone
					   while the refusal copy went on telling the reader to use it. */
					storeCertificate(answer.certificate, answer.operator_key_id ?? deviceId);
					setStatus({
						kind: "paired",
						deviceId,
						name: answer.name || name.trim(),
					});
					return;
				}
			}
			if (!cancelled.current) {
				setStatus({
					kind: "failed",
					message:
						"No answer from the machine. Run `lop pair` there and approve this phone, then try again.",
				});
			}
		} catch (error) {
			setStatus({ kind: "failed", message: humanizePairingError(error) });
		}
	}

	return (
		<div className="mx-auto flex h-dvh w-full max-w-[520px] flex-col gap-3 overflow-y-auto px-3 pb-6 pt-[max(env(safe-area-inset-top),0.5rem)]">
			<header className="flex items-center gap-2">
				<button
					type="button"
					onClick={() => navigate("/")}
					aria-label="back to sessions"
					className="flex min-h-8 min-w-8 items-center justify-center rounded-sm text-ink-muted active:bg-elevated"
				>
					‹
				</button>
				<h1 className="min-w-0 flex-1 truncate text-body font-medium">Pair this phone</h1>
			</header>

			<p className="text-body-sm text-ink-muted">
				On the machine that runs your sessions, run <code>lop pair</code> and enter the code it
				prints. It authorises THIS phone once, and the phone&rsquo;s key never leaves it.
			</p>

			<label className="flex flex-col gap-1 text-body-sm">
				<span className="text-ink-muted">pairing code</span>
				<input
					value={code}
					onChange={(event) => setCode(event.target.value)}
					autoCapitalize="none"
					autoCorrect="off"
					spellCheck={false}
					placeholder="paste the code"
					className="min-h-11 rounded-sm border border-control bg-surface px-3 font-mono text-body text-ink outline-none placeholder:text-ink-dim"
				/>
			</label>

			<label className="flex flex-col gap-1 text-body-sm">
				<span className="text-ink-muted">what to call this device</span>
				<input
					value={name}
					onChange={(event) => setName(event.target.value)}
					className="min-h-11 rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none"
				/>
			</label>

			<Button
				disabled={status.kind === "waiting" || code.trim().length === 0}
				onClick={() => void claim()}
			>
				{status.kind === "waiting" ? "waiting for the machine…" : "pair"}
			</Button>

			{status.kind === "waiting" ? (
				<p className="text-body-sm text-ink-muted">
					Sent. The machine is showing a request naming &ldquo;{name}&rdquo; — approve it there.
				</p>
			) : null}

			{status.kind === "paired" ? (
				<div className="flex flex-col gap-1 rounded-sm border border-hairline bg-elevated p-3">
					<p className="text-body-sm">
						Paired as <strong>{status.name}</strong>.
					</p>
					<p className="text-meta text-ink-muted">
						This phone can now approve parked tool calls and loosen a session&rsquo;s approval
						gate ({DEVICE_SCOPES.join(", ")}). Revoke it on the machine with{" "}
						<code>lop operator devices --revoke {status.deviceId}</code>.
					</p>
				</div>
			) : null}

			{status.kind === "failed" ? (
				<p className="text-body-sm text-danger">{status.message}</p>
			) : null}
		</div>
	);
}

function defaultDeviceName(): string {
	/* A name the operator will RECOGNISE in `lop pair`'s confirmation, which is the
	   only thing standing between a request and a signed certificate. The user agent
	   is a guess and is labelled as one by the field being editable. */
	const platform = typeof navigator === "undefined" ? "" : navigator.platform || "";
	const label = platform.includes("iPhone") || platform.includes("iPad") ? "iPhone" : "phone";
	return `${label} (${new Date().toISOString().slice(0, 10)})`;
}

function humanizePairingError(error: unknown): string {
	const message = String((error as Error)?.message ?? error);
	if (message.includes("not valid")) {
		return "That code is not valid — run `lop pair` on the machine again for a fresh one.";
	}
	if (message.includes("revoked")) {
		return "This device has been revoked on the machine. Pair it again from `lop pair`.";
	}
	if (message.toLowerCase().includes("subtle")) {
		return "This browser cannot hold a device key securely, so pairing is unavailable here.";
	}
	return message;
}

/** The sheet the session view opens to offer the pairing screen without leaving the
    session: a phone that cannot sign should be one tap from being able to, because
    the refusal copy names the phone as a remedy. */
export function PairPromptSheet({
	open,
	onClose,
}: {
	open: boolean;
	onClose: () => void;
}) {
	return (
		<Sheet open={open} onClose={onClose} title="Pair this phone">
			<div className="flex flex-col gap-3 p-3">
				<p className="text-body-sm text-ink-muted">
					Approving a parked tool call and loosening a session&rsquo;s approval gate need the
					operator&rsquo;s own consent. Pair this phone once and it can give that consent from
					here.
				</p>
				<Button
					onClick={() => {
						onClose();
						navigate("/pair");
					}}
				>
					pair this phone
				</Button>
			</div>
		</Sheet>
	);
}

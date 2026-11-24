/**
 * The ask id a reader can MATCH (Ben, 2026-11-23: the relay showed no ask id
 * anywhere, so a reader could not find the `a-745e` an agent quoted).
 *
 * `AskIdFoot` is the quiet id at the foot of a card or a settled row: mono,
 * muted, at the reading-end edge, with a copy door. It is a real button at the
 * 44 px touch floor because the row it sits in already owns its own press — a
 * copy affordance nested in the row's button is invalid HTML and unreachable.
 *
 * THE ID IS NOT THE STATE'S ONLY HOME, deliberately: the chip in the agent's
 * sentence (added with the jump) says open/answered/withdrawn and opens the
 * asks sheet; this foot only lets the reader copy what the agent wrote.
 */
import { useEffect, useRef, useState } from "react";
import { messageRuntime } from "../store";

/** How long "Copied" stays up. Feedback timing only: nothing waits on it, and
 *  the card below is re-read on its own schedule anyway. */
const COPIED_MS = 2000;

/**
 * Copy one string, or select its node so the platform's own copy gesture can
 * finish the job. A relay page can be served over plain HTTP on a LAN address,
 * where `navigator.clipboard` is absent (not a secure context) — so "copy"
 * failing is an ordinary state here rather than an exotic one, and it must
 * leave the reader something they can act on instead of a dead tap.
 */
async function copyOrSelect(text: string, node: HTMLElement | null): Promise<boolean> {
	try {
		await navigator.clipboard.writeText(text);
		return true;
	} catch {
		const selection = window.getSelection();
		if (node && selection) {
			const range = document.createRange();
			range.selectNodeContents(node);
			selection.removeAllRanges();
			selection.addRange(range);
		}
		return false;
	}
}

export function AskIdFoot({
	askId,
	className,
}: {
	askId: string;
	/** Placement only; the foot owns its own size, ink and feedback. */
	className?: string;
}) {
	const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
	const idNode = useRef<HTMLSpanElement>(null);
	useEffect(() => {
		if (state !== "copied") return;
		const timer = window.setTimeout(() => setState("idle"), COPIED_MS);
		return () => window.clearTimeout(timer);
	}, [state]);
	const copy = async () => {
		setState((await copyOrSelect(askId, idNode.current)) ? "copied" : "failed");
	};
	return (
		<>
			<button
				type="button"
				data-testid="ask-id-foot"
				data-ask-id-copy={askId}
				aria-label={messageRuntime.t("relay.asks.id.copy_label", { id: askId })}
				onClick={() => void copy()}
				className={className === undefined
					? "flex min-h-11 items-center justify-end gap-1 self-end rounded-sm px-1 font-mono text-mono-sm text-ink-muted active:bg-surface"
					: `flex min-h-11 items-center justify-end gap-1 rounded-sm px-1 font-mono text-mono-sm text-ink-muted active:bg-surface ${className}`}
			>
				<span ref={idNode} dir="ltr">{askId}</span>
				{/* The mark is decorative: the button's own accessible name says what
				    the tap does, and `Copied` is only a glance's confirmation. */}
				<span aria-hidden="true" className="text-ink-dim">
					{state === "copied" ? messageRuntime.t("relay.asks.id.copied") : "⧉"}
				</span>
			</button>
			{/* A LIVE REGION, not a toast and not a second `role="status"`: the relay
			    has no toast layer, and the card already carries one status line that
			    tests and readers count. `aria-live` on an always-mounted (but empty)
			    span announces the change when it lands; the visible mark cannot, since
			    it is aria-hidden and its glyph says nothing. The named form is used
			    because several ask ids can sit on one screen. */}
			<span aria-live="polite" className="sr-only">
				{state === "copied" ? messageRuntime.t("relay.asks.id.copied_named", { id: askId }) : ""}
			</span>
			{state === "failed" ? (
				<p role="status" className="text-meta text-ink-dim">
					{messageRuntime.t("relay.asks.id.copy_failed")}
				</p>
			) : null}
		</>
	);
}

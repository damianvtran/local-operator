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
import { createContext, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { askChipOpen, askChipWord } from "../lib/ask-chip";
import type { AskChipFact } from "../lib/ask-chip";
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

/**
 * What a chip needs from its screen: the ids THIS session can answer for, and
 * the one action a chip performs (open the asks sheet at that ask).
 *
 * A CONTEXT, not props: the chip is drawn by the markdown renderer inside a
 * transcript row, several component layers below the screen that owns the ask
 * state and the sheet. Threading a map and a callback through every row would
 * put ask state in the signature of components that have no other reason to
 * know what an ask is. `null` means "this screen draws no chips", and the
 * renderer is what asks for chips explicitly (`Markdown askChips`), so the
 * context alone never turns an id into a control on a surface that did not opt
 * in.
 */
export interface AskChipHost {
	facts: ReadonlyMap<string, AskChipFact>;
	open: (id: string) => void;
}

const AskChipHostContext = createContext<AskChipHost | null>(null);

export function AskChipHostProvider({
	value,
	children,
}: {
	value: AskChipHost;
	children: ReactNode;
}) {
	return <AskChipHostContext.Provider value={value}>{children}</AskChipHostContext.Provider>;
}

export function useAskChipHost(): AskChipHost | null {
	return useContext(AskChipHostContext);
}

/**
 * An ask id quoted in an agent's prose.
 *
 * THE CHIP IS A BUTTON, because its whole point is that it opens the ask it
 * names. Its target is its own line box: an inline control inside running
 * prose cannot be 44 px tall without breaking the sentence into blocks, and the
 * card foot below is the 44 px way to copy the same id (the native app reached
 * the same split for the same reason).
 *
 * AN ID THIS SESSION CANNOT ANSWER FOR IS NOT A CONTROL: another conversation's
 * id and an aged-out one look identical from here, and a chip that opened the
 * wrong question's sheet would be worse than plain text. It is drawn as mono
 * with a dotted underline — the one signal that this is an id and not a word —
 * and does nothing, per the decided behaviour: unmatched ids stay plain.
 */
export function AskIdChip({ id }: { id: string }) {
	const host = useAskChipHost();
	const fact = host === null ? undefined : host.facts.get(id);
	const word = fact === undefined ? null : askChipWord(fact.status);
	if (host === null || fact === undefined || word === null) {
		return (
			<span
				dir="ltr"
				data-testid="ask-id-plain"
				className="font-mono text-mono-sm text-ink-muted underline decoration-dotted underline-offset-2"
			>
				{id}
			</span>
		);
	}
	const state = word.kind === "key" ? messageRuntime.t(word.key) : word.text;
	const open = askChipOpen(fact.status);
	return (
		<button
			type="button"
			data-testid="ask-chip"
			data-ask-id={id}
			data-ask-status={fact.status}
			aria-label={
				fact.question === ""
					? messageRuntime.t("relay.asks.chip.label_bare", { id, state })
					: messageRuntime.t("relay.asks.chip.label", { id, state, question: fact.question })
			}
			onClick={() => host.open(id)}
			className="mx-0.5 inline-flex min-h-6 items-center gap-1 rounded-xs bg-sunken px-1 align-baseline font-mono text-mono-sm text-ink active:bg-elevated focus-visible:underline"
		>
			{/* The dot is a reading aid, never the state: the word beside it says
			    the same thing, so the chip survives greyscale and colour blindness. */}
			<span aria-hidden="true" className={open ? "text-accent-active" : "text-ink-dim"}>●</span>
			<span dir="ltr">{id}</span>
			<span aria-hidden="true" className="text-ink-dim">·</span>
			<span>{state}</span>
		</button>
	);
}

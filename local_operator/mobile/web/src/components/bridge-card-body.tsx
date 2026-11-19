/**
 * The body under an inbound Team Bridge card on the phone relay.
 *
 * WHAT IT ADDS. The card's row is still the receipt line the relay projects
 * ("<sender> · <conversation>"), drawn by the transcript's notice row. When
 * the relay advertises `features.team_bridge_card_body`, this panel sits
 * under it and reads the card's body once per mount (retries on a
 * not-yet-written row, below)
 * (`GET /api/sessions/{sid}/bridge-cards/{entry}/body`, `lib/bridge-card-body.ts`
 * for the contract and its constraints), then shows, per message:
 *  - an admitted body as markdown, under the "External message · untrusted
 *    content" frame, folded behind a worded Show more / Show less when long;
 *  - the server's own word for every other state (held, waiting for
 *    screening, refused, pending, not delivered, deleted, offline, …);
 *  - for a coalesced card, at most 20 messages and "N more messages in this
 *    card" for the rest.
 *
 * WHY NOTHING ON AN OLDER RELAY. Without the capability the panel renders
 * nothing and asks nothing, so the card is byte-for-byte today's receipt line:
 * an old relay has no route to ask, and a client-made "unavailable" there would
 * be a new claim the server never made.
 *
 * A failed read (network, 404 for a row this relay cannot answer, a malformed
 * frame) reads as the generic "Message body unavailable" — never "offline",
 * which is the SERVER saying it could not reach Radient, and never a
 * moderation outcome.
 *
 * NOT YET, NOT FAILED (review M1). On an idle or steered session the live
 * receipt is emitted BEFORE the journal row is written (core `session.py`
 * `_deliver_peer_message`: `_emit_peer_receipt`, then `_prompt_messages` takes
 * the turn lock and persists). A panel mounted on that receipt can ask before
 * the row exists and get `404 {"error": "no such bridge card"}`. The replayed
 * row keeps the same id, so the row never remounts: treating that 404 as a
 * final answer would pin a false "unavailable" on the newest card. That ONE
 * answer (matched on its body, so "unknown session" and any other 404 stay
 * failures) is re-asked after `NOT_YET_RETRY_MS` while Loading shows; only
 * when the row is still missing does the card say unavailable.
 *
 * NOTHING TO RESOLVE (review m1). A card from before receipts carried message
 * ids answers `{items: [], more: 0}` (core `card_body.py`: nothing to resolve).
 * The server made no claim about a body there, so the panel steps aside and
 * the card is its receipt line alone — the TUI twin's behaviour — instead of
 * stamping "unavailable" on every old card in history. A 404 that outlives the
 * retries is different: the reader was shown a framed Loading for a card the
 * relay then could not find, so it ends in the generic unavailable line the
 * desktop also uses for a failed read.
 */
import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { getBridgeCardBody, HttpError } from "../api";
import { CONTENT_RESIZED_EVENT } from "../lib/content-resized";
import { MESSAGE_PRESSED_EVENT } from "../lib/message-fold";
import {
	bridgeBodyPreview,
	bridgeBodyPreviewText,
	bridgeCardBodyEnabled,
	bridgeCardBodyOf,
	bridgeItemCopyKey,
	type BridgeCardBody,
	type BridgeCardBodyItem,
} from "../lib/bridge-card-body";
import { messageRuntime, useCapabilities } from "../store";
import { Markdown } from "./markdown";

type Load = { phase: "loading" } | { phase: "ready"; answer: BridgeCardBody } | { phase: "failed" };

/** Delays before each re-ask of a card whose row is not written yet. The
    window is receipt emit → turn lock → journal append; QA's live runs
    (#4278 QA round 1, n=3) saw the persisted row answer 200 about 0.45 s
    after delivery, so two re-asks reaching ~2 s cover it with margin while
    keeping a genuinely missing row's Loading short. */
export const NOT_YET_RETRY_MS: readonly number[] = [500, 1500];

/** The route's answer for an entry it has no `team_bridge_message` row for —
    exactly the not-yet-journaled case. Keyed on the body, not the status:
    `404 unknown session` is a different, final answer. */
function cardNotYetWritten(error: unknown): boolean {
	return error instanceof HttpError && error.status === 404 && error.message === "no such bridge card";
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
	return new Promise((resolve) => {
		const timer = setTimeout(resolve, ms);
		signal.addEventListener("abort", () => {
			clearTimeout(timer);
			resolve();
		}, { once: true });
	});
}

/** True for a notice row that is an inbound Team Bridge delivery. The live arm
    also carries the type under `details.custom_kind` (`note_nonpeer_delivery`). */
export function isBridgeCardNotice(entry: { custom_type?: string; details?: { custom_kind?: string } }): boolean {
	return entry.custom_type === "team_bridge_message" || entry.details?.custom_kind === "team_bridge_message";
}

export function BridgeCardBodyPanel({ pid, entryId }: { pid: string; entryId: string }) {
	const enabled = bridgeCardBodyEnabled(useCapabilities());
	if (!enabled) return null;
	return <BridgeCardBodyRead pid={pid} entryId={entryId} />;
}

function BridgeCardBodyRead({ pid, entryId }: { pid: string; entryId: string }) {
	const [load, setLoad] = useState<Load>({ phase: "loading" });
	const box = useRef<HTMLDivElement>(null);

	useEffect(() => {
		/* Read on every mount, keep no copy (revocation stays live). The abort
		   keeps an unmounted row from writing a late answer into state. */
		const controller = new AbortController();
		const { signal } = controller;
		setLoad({ phase: "loading" });
		void (async () => {
			for (let attempt = 0; ; attempt += 1) {
				try {
					const payload = await getBridgeCardBody(pid, entryId, signal);
					if (signal.aborted) return;
					const answer = bridgeCardBodyOf(payload);
					setLoad(answer ? { phase: "ready", answer } : { phase: "failed" });
					return;
				} catch (error) {
					if (signal.aborted) return;
					/* Still Loading: the row may simply not be written yet (M1). The
					   abort wakes the wait, so an unmounted row stops here. */
					if (cardNotYetWritten(error) && attempt < NOT_YET_RETRY_MS.length) {
						await sleep(NOT_YET_RETRY_MS[attempt], signal);
						if (signal.aborted) return;
						continue;
					}
					setLoad({ phase: "failed" });
					return;
				}
			}
		})();
		return () => controller.abort();
	}, [pid, entryId]);

	/* The panel changes height when the answer lands: tell the transcript, the
	   way a folding user message does, so its follow/anchor logic holds still. */
	const painted = useRef(false);
	useLayoutEffect(() => {
		if (!painted.current) {
			painted.current = true;
			return;
		}
		box.current?.dispatchEvent(new CustomEvent(CONTENT_RESIZED_EVENT, { bubbles: true }));
	}, [load]);

	/* Nothing to resolve (a pre-receipt card): no panel, the receipt line
	   stands alone. Dropped-id items are counted in `more`, so a card whose
	   every item was unattributable still keeps its panel and says so.
	   The empty box stays mounted (hidden, zero height) only so the Loading
	   frame's disappearance still reaches the transcript as a resize. */
	if (load.phase === "ready" && load.answer.items.length === 0 && load.answer.more === 0) {
		return <div ref={box} data-bridge-card-body="" data-state="empty" hidden />;
	}

	let content;
	if (load.phase === "loading") {
		content = <p className="text-meta text-ink-dim">{messageRuntime.t("relay.transcript.bridge.loading")}</p>;
	} else if (load.phase === "failed") {
		content = <p className="text-meta text-ink-dim">{messageRuntime.t("relay.transcript.bridge.unavailable")}</p>;
	} else {
		const { items, more } = load.answer;
		content = (
			<>
				{items.map((item, index) => (
					<BridgeItem key={`${item.messageId}:${index}`} item={item} toggleId={`bridge:${entryId}:${index}`} />
				))}
				{more > 0 ? (
					<p className="text-meta text-ink-dim">{messageRuntime.t("relay.transcript.bridge.more_messages", { count: more })}</p>
				) : null}
			</>
		);
	}

	return (
		/* The provenance frame: a bordered region whose first line names the
		   content as external and untrusted, so a teammate's words can never read
		   as the app's or the agent's own. `ml-[22px]` lines the frame up with the
		   receipt text after the notice row's 16px glyph column + gap. `mb-2.5`
		   (10px) is the gap under the card: clearly more than the next receipt's
		   4px `mt-1` to its own card, so each receipt line reads as the head of
		   the body below it, not the foot of the one above (design D5). */
		<div
			ref={box}
			data-bridge-card-body=""
			data-state={load.phase}
			aria-busy={load.phase === "loading" || undefined}
			className="ml-[22px] mt-1 mb-2.5 flex min-w-0 flex-col gap-2 rounded-md border border-control px-3 py-2"
		>
			<p className="text-meta text-ink-dim">{messageRuntime.t("relay.transcript.bridge.provenance")}</p>
			{content}
		</div>
	);
}

function BridgeItem({ item, toggleId }: { item: BridgeCardBodyItem; toggleId: string }) {
	if (item.state === "admitted" && item.body) return <AdmittedBody body={item.body} toggleId={toggleId} />;
	return (
		<p data-bridge-item-state={item.state} className="text-meta text-ink-muted">
			{messageRuntime.t(bridgeItemCopyKey(item))}
		</p>
	);
}

/* `toggleId` names this body to the transcript's place-holding fold recipe
   (`lib/message-fold.ts`): unique per card item, so the scroller can find the
   body's top (opening) or the control (closing) by it. */
function AdmittedBody({ body, toggleId }: { body: string; toggleId: string }) {
	const [open, setOpen] = useState(false);
	const id = useId();
	const box = useRef<HTMLDivElement>(null);
	const preview = bridgeBodyPreview(body);
	const shown = open ? body : bridgeBodyPreviewText(preview);

	/* The body changed height after it painted: say so, after the press below
	   told the transcript which control to hold — the same pair the user
	   message fold sends (`user-text.tsx`), so Show less lands under the thumb
	   instead of ~1000px below the card on a long body (design D3). */
	const painted = useRef(false);
	useLayoutEffect(() => {
		if (!painted.current) {
			painted.current = true;
			return;
		}
		box.current?.dispatchEvent(new CustomEvent(CONTENT_RESIZED_EVENT, { bubbles: true }));
	}, [open]);

	return (
		<div data-bridge-item-state="admitted" className="flex min-w-0 flex-col gap-1">
			<div ref={box} id={id} data-message-id={toggleId} className="text-body min-w-0">
				<Markdown text={shown} />
			</div>
			{preview.truncated ? (
				/* Worded, not a bare chevron: the control opens someone else's
				   message, and the word is the only cue that more exists. 44px tall
				   target per the relay guidelines. */
				<button
					type="button"
					aria-expanded={open}
					aria-controls={id}
					data-message-toggle={toggleId}
					onClick={(event) => {
						/* Before the flip, so the transcript reads the control's top
						   while it is still under the finger. */
						event.currentTarget.dispatchEvent(
							new CustomEvent(MESSAGE_PRESSED_EVENT, { bubbles: true, detail: { id: toggleId } }),
						);
						setOpen((v) => !v);
					}}
					className="min-h-11 self-start text-meta text-accent"
				>
					{messageRuntime.t(open ? "relay.transcript.message.show_less" : "relay.transcript.message.show_more")}
				</button>
			) : null}
		</div>
	);
}


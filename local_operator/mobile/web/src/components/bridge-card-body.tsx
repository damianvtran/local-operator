/**
 * The body under an inbound Team Bridge card on the phone relay.
 *
 * WHAT IT ADDS. The card's row is still the receipt line the relay projects
 * ("<sender> · <conversation>"), drawn by the transcript's notice row. When
 * the relay advertises `features.team_bridge_card_body`, this panel sits
 * under it and reads the card's body once per mount
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
 */
import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { getBridgeCardBody } from "../api";
import { CONTENT_RESIZED_EVENT } from "../lib/content-resized";
import {
	bridgeBodyPreview,
	bridgeCardBodyEnabled,
	bridgeCardBodyOf,
	bridgeItemCopyKey,
	type BridgeCardBody,
	type BridgeCardBodyItem,
} from "../lib/bridge-card-body";
import { messageRuntime, useCapabilities } from "../store";
import { Markdown } from "./markdown";

type Load = { phase: "loading" } | { phase: "ready"; answer: BridgeCardBody } | { phase: "failed" };

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
		setLoad({ phase: "loading" });
		getBridgeCardBody(pid, entryId, controller.signal).then(
			(payload) => {
				if (controller.signal.aborted) return;
				const answer = bridgeCardBodyOf(payload);
				setLoad(answer ? { phase: "ready", answer } : { phase: "failed" });
			},
			() => {
				if (!controller.signal.aborted) setLoad({ phase: "failed" });
			},
		);
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

	let content;
	if (load.phase === "loading") {
		content = <p className="text-meta text-ink-dim">{messageRuntime.t("relay.transcript.bridge.loading")}</p>;
	} else if (load.phase === "failed" || load.answer.items.length === 0) {
		content = <p className="text-meta text-ink-dim">{messageRuntime.t("relay.transcript.bridge.unavailable")}</p>;
	} else {
		const { items, more } = load.answer;
		content = (
			<>
				{items.map((item, index) => (
					<BridgeItem key={`${item.messageId}:${index}`} item={item} />
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
		   receipt text after the notice row's 16px glyph column + gap. */
		<div
			ref={box}
			data-bridge-card-body=""
			data-state={load.phase}
			aria-busy={load.phase === "loading" || undefined}
			className="ml-[22px] mt-1 flex min-w-0 flex-col gap-2 rounded-md border border-control px-3 py-2"
		>
			<p className="text-meta text-ink-dim">{messageRuntime.t("relay.transcript.bridge.provenance")}</p>
			{content}
		</div>
	);
}

function BridgeItem({ item }: { item: BridgeCardBodyItem }) {
	if (item.state === "admitted" && item.body) return <AdmittedBody body={item.body} />;
	return (
		<p data-bridge-item-state={item.state} className="text-meta text-ink-muted">
			{messageRuntime.t(bridgeItemCopyKey(item))}
		</p>
	);
}

function AdmittedBody({ body }: { body: string }) {
	const [open, setOpen] = useState(false);
	const id = useId();
	const preview = bridgeBodyPreview(body);
	const shown = open || !preview.truncated ? body : `${preview.text}…`;
	return (
		<div data-bridge-item-state="admitted" className="flex min-w-0 flex-col gap-1">
			<div id={id} className="text-body min-w-0">
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
					onClick={() => setOpen((v) => !v)}
					className="min-h-11 self-start text-meta text-accent"
				>
					{messageRuntime.t(open ? "relay.transcript.message.show_less" : "relay.transcript.message.show_more")}
				</button>
			) : null}
		</div>
	);
}


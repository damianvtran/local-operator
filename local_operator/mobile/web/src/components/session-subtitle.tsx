/** One quiet liveness line replaces a working band and an outage overlay.
 * `connected` already includes the connectivity owner's grace: no second
 * timer hides an outage here. Age is local time since that fact went false,
 * not a claim about an unexposed server timestamp. Only the transition is a
 * live announcement; five-second age ticks must not interrupt reading. */
import { contentDir } from "../lib/content-direction";
import { useEffect, useRef, useState } from "react";
import { blockingPending } from "../lib/asks";
import { formatElapsed } from "../lib/format";
import { sessionHealth } from "../lib/session-health";
import { messageRuntime } from "../store";
import type { SessionProjection } from "../types";
import { ChromeIcon } from "./ui/chrome-icon";

export function SessionSubtitle({ projection, connected, temporary = false }: {
	projection: SessionProjection;
	connected: boolean;
	/** A temporary chat: the line reads `Temporary · <status>` behind the dashed
	    mark, on the line that already exists, so the mark costs no height. */
	temporary?: boolean;
}) {
	const lostAt = useRef<number | null>(connected ? null : Date.now());
	const [now, setNow] = useState(Date.now());
	const [elapsed, setElapsed] = useState(projection.activity_started_s ?? 0);
	useEffect(() => {
		lostAt.current = connected ? null : Date.now();
		setNow(Date.now());
		if (connected) return;
		const timer = setInterval(() => setNow(Date.now()), 5000);
		return () => clearInterval(timer);
	}, [connected]);
	useEffect(() => {
		setElapsed(projection.activity_started_s ?? 0);
		if (!projection.streaming) return;
		const timer = setInterval(() => setElapsed(value => value + 1), 1000);
		return () => clearInterval(timer);
	}, [projection.streaming, projection.activity_started_s]);
	/* THE FIRST RUNGS ARE THE STATUS DOT'S (`lib/session-health.ts`), so the
	   dot and the words beside it cannot disagree — a stale heartbeat reads
	   "Not answering" here too, and a disk rebuild with no running owner reads
	   "Ended" rather than "Idle". */
	const health = sessionHealth(projection, connected);
	let label: string;
	if (health === "ended") label = messageRuntime.t("relay.chrome.session.subtitle.ended");
	else if (health === "not_answering") label = messageRuntime.t("relay.chrome.session.subtitle.not_answering");
	else if (health === "reconnecting") {
		const seconds = Math.max(1, Math.floor((now - (lostAt.current ?? now)) / 1000));
		label = seconds < 60 ? messageRuntime.t("relay.chrome.link.since_seconds", { count: seconds })
			: messageRuntime.t("relay.chrome.link.since_minutes", { count: Math.floor(seconds / 60) });
	} else if (blockingPending(projection.pending, projection.asks)) label = messageRuntime.t("relay.chrome.session.subtitle.waiting");
	else if (projection.streaming) label = projection.activity_started_s === null
		? messageRuntime.t("relay.chrome.session.subtitle.working_plain")
		: messageRuntime.t("relay.chrome.session.subtitle.working", { elapsed: formatElapsed(elapsed) });
	else label = messageRuntime.t("relay.chrome.session.subtitle.idle");
	/* THE MARK AND THE WORD LEAD, so a narrow header truncates the status's
	   tail ("Working · 12s" goes first) and "Temporary" is the last to go. */
	if (temporary) label = messageRuntime.t("relay.session.temporary_subtitle", { status: label });
	return <>
		{/* No dot of its own: the header's status control draws the ONE dot
		    beside this line (the link-down mark used to live here). */}
		{temporary ? (
			<span data-temporary="" className="flex min-w-0 items-center gap-1 text-meta text-ink-dim">
				<ChromeIcon name="temporary" size={12} />
				<span dir={contentDir(label)} className="block min-w-0 truncate">{label}</span>
			</span>
		) : <span dir={contentDir(label)} className="block truncate text-meta text-ink-dim">{label}</span>}
	</>;
}

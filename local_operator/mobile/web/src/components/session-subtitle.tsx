/** One quiet liveness line replaces a working band and an outage overlay.
 * `connected` already includes the connectivity owner's grace: no second
 * timer hides an outage here. Age is local time since that fact went false,
 * not a claim about an unexposed server timestamp. Only the transition is a
 * live announcement; five-second age ticks must not interrupt reading. */
import { useEffect, useRef, useState } from "react";
import { formatElapsed } from "../lib/format";
import { messageRuntime } from "../store";
import type { SessionProjection } from "../types";

export function SessionSubtitle({ projection, connected }: { projection: SessionProjection; connected: boolean }) {
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
	let label: string;
	if (!connected) {
		const seconds = Math.max(1, Math.floor((now - (lostAt.current ?? now)) / 1000));
		label = seconds < 60 ? messageRuntime.t("relay.chrome.link.since_seconds", { count: seconds })
			: messageRuntime.t("relay.chrome.link.since_minutes", { count: Math.floor(seconds / 60) });
	} else if (projection.ended) label = messageRuntime.t("relay.chrome.session.subtitle.ended");
	else if (projection.degraded) label = messageRuntime.t("relay.chrome.session.subtitle.not_answering");
	else if (projection.pending) label = messageRuntime.t("relay.chrome.session.subtitle.waiting");
	else if (projection.streaming) label = projection.activity_started_s === null
		? messageRuntime.t("relay.chrome.session.subtitle.working_plain")
		: messageRuntime.t("relay.chrome.session.subtitle.working", { elapsed: formatElapsed(elapsed) });
	else label = messageRuntime.t("relay.chrome.session.subtitle.idle");
	return <>
		<span className="block truncate text-meta text-ink-dim">{!connected ? <span aria-hidden className="me-1 inline-block size-1.5 rounded-full bg-warning" /> : null}{label}</span>
		<span className="sr-only" role="status">{!connected ? messageRuntime.t("relay.chrome.link.reconnecting") : ""}</span>
	</>;
}

/**
 * Hash-based routing, hand-rolled. Routes: `#/`, `#/new`, `#/past`,
 * `#/s/:sessionId`, `#/s/:sessionId/a/:jobId`. A hash router is the right
 * shape here because the daemon
 * serves a single static bundle and no server-side route table exists.
 */
import { useEffect, useState } from "react";

export type Route =
	| { name: "list" }
	| { name: "new" }
	| { name: "past" }
	| { name: "session"; sessionId: string; jobId?: string };

export function parseHash(hash: string): Route {
	const path = hash.replace(/^#/, "") || "/";
	if (path === "/new") return { name: "new" };
	if (path === "/past") return { name: "past" };
	const agent = path.match(/^\/s\/([^/]+)\/a\/([^/]+)$/);
	if (agent) {
		return {
			name: "session",
			sessionId: decodeURIComponent(agent[1]),
			jobId: decodeURIComponent(agent[2]),
		};
	}
	const session = path.match(/^\/s\/([^/]+)$/);
	if (session) return { name: "session", sessionId: decodeURIComponent(session[1]) };
	return { name: "list" };
}

const ROUTE_STATE_KEY = "loMobileRoute";

export function navigate(to: string, options: { replace?: boolean } = {}): void {
	const method = options.replace ? "replaceState" : "pushState";
	/* Mark entries created inside the SPA. The detail header can then use true
	   chronological Back for an in-app visit without sending a direct/deep-link
	   visit to an unrelated page that happened to precede Local Operator. */
	history[method]({ ...history.state, [ROUTE_STATE_KEY]: true }, "", `#${to}`);
	window.dispatchEvent(new PopStateEvent("popstate", { state: history.state }));
}

export function navigateUp(fallback: string): void {
	if (history.state?.[ROUTE_STATE_KEY]) {
		history.back();
		return;
	}
	/* A direct route has no trustworthy in-app predecessor. Replace it with its
	   hierarchy parent so the primary Back control always stays in the app. */
	navigate(fallback, { replace: true });
}

export function useRoute(): Route {
	const [route, setRoute] = useState<Route>(() => parseHash(location.hash));
	useEffect(() => {
		const onChange = () => setRoute(parseHash(location.hash));
		window.addEventListener("hashchange", onChange);
		window.addEventListener("popstate", onChange);
		return () => {
			window.removeEventListener("hashchange", onChange);
			window.removeEventListener("popstate", onChange);
		};
	}, []);
	return route;
}

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

const ROUTE_STATE_KEY = "loMobileHasInAppPredecessor";

export function navigate(
	to: string,
	options: { replace?: boolean; hasInAppPredecessor?: boolean } = {},
): void {
	const method = options.replace ? "replaceState" : "pushState";
	/* A pushed route has a chronological in-app predecessor. A replacement must
	   opt into that claim: hierarchy fallback replaces an untrusted external
	   predecessor and therefore must remain false at every recursive level. */
	const hasInAppPredecessor = options.hasInAppPredecessor ?? !options.replace;
	history[method]({ ...history.state, [ROUTE_STATE_KEY]: hasInAppPredecessor }, "", `#${to}`);
	window.dispatchEvent(new PopStateEvent("popstate", { state: history.state }));
}

export function navigateUp(fallback: string): void {
	if (history.state?.[ROUTE_STATE_KEY]) {
		history.back();
		return;
	}
	/* A direct route has no trustworthy in-app predecessor. Replacing it with a
	   hierarchy parent preserves that fact so repeated Back keeps climbing to
	   root instead of eventually escaping Local Operator. */
	navigate(fallback, { replace: true, hasInAppPredecessor: false });
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

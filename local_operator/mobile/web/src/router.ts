/**
 * Hash-based routing, hand-rolled. Routes: `#/`, `#/new`, `#/past`,
 * `#/s/:pid`. A hash router is the right shape here because the daemon
 * serves a single static bundle and no server-side route table exists.
 */
import { useEffect, useState } from "react";

export type Route =
	| { name: "list" }
	| { name: "new" }
	| { name: "past" }
	| { name: "session"; pid: number };

export function parseHash(hash: string): Route {
	const path = hash.replace(/^#/, "") || "/";
	if (path === "/new") return { name: "new" };
	if (path === "/past") return { name: "past" };
	const session = path.match(/^\/s\/(\d+)/);
	if (session) return { name: "session", pid: Number(session[1]) };
	return { name: "list" };
}

export function navigate(to: string): void {
	location.hash = to;
}

export function useRoute(): Route {
	const [route, setRoute] = useState<Route>(() => parseHash(location.hash));
	useEffect(() => {
		const onChange = () => setRoute(parseHash(location.hash));
		window.addEventListener("hashchange", onChange);
		return () => window.removeEventListener("hashchange", onChange);
	}, []);
	return route;
}

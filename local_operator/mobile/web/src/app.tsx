/**
 * App root: hash-routed screens over the shared store. The theme is
 * initialised before first paint (in main.tsx) so there is no flash of the
 * wrong ramp.
 */
import { useEffect } from "react";
import { useRoute } from "./router";
import { PairScreen } from "./screens/pair";
import { PastSessionsScreen } from "./screens/past-sessions";
import { SessionListScreen } from "./screens/session-list";
import { SessionScreen } from "./screens/session-view";
import { retainConfigStream } from "./store";

export function App() {
	const route = useRoute();
	/* App-level, once: the config stream carries the machine's resolved
	   language for EVERY screen (a session view shows it as much as the list),
	   so it is retained at the root rather than per screen. */
	useEffect(() => retainConfigStream(), []);
	switch (route.name) {
		case "past":
			return <PastSessionsScreen />;
		case "pair":
			return <PairScreen />;
		case "session":
			return (
				<SessionScreen
					key={`${route.sessionId}:${route.jobId ?? "root"}`}
					sessionId={route.sessionId}
					jobId={route.jobId}
				/>
			);
		default:
			return <SessionListScreen />;
	}
}

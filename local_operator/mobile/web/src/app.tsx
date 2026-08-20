/**
 * App root: hash-routed screens over the shared store. The theme is
 * initialised before first paint (in main.tsx) so there is no flash of the
 * wrong ramp.
 */
import { useRoute } from "./router";
import { NewSessionScreen } from "./screens/new-session";
import { PastSessionsScreen } from "./screens/past-sessions";
import { SessionListScreen } from "./screens/session-list";
import { SessionScreen } from "./screens/session-view";

export function App() {
	const route = useRoute();
	switch (route.name) {
		case "new":
			return <NewSessionScreen />;
		case "past":
			return <PastSessionsScreen />;
		case "session":
			return <SessionScreen key={route.pid} pid={route.pid} />;
		default:
			return <SessionListScreen />;
	}
}

/**
 * App root: hash-routed screens over the shared store. The theme is
 * initialised before first paint (in main.tsx) so there is no flash of the
 * wrong ramp.
 */
import { useEffect } from "react";
import { useRoute } from "./router";
import { PairingSheetHost, PairScreen } from "./screens/pair";
import { reconcileDeviceKey } from "./lib/operator-device";
import { AccountRoutingScreen } from "./screens/account-routing";
import { LibraryScreen } from "./screens/library";
import { LibraryAssetScreen } from "./screens/library-asset";
import { NotificationDetailScreen } from "./screens/notification-detail";
import { NotificationsScreen } from "./screens/notifications";
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
	/* U16, on load: a certificate whose key the browser evicted is a pairing that
	   can no longer sign. Clearing it now makes every surface read "not paired"
	   (and offer pairing) instead of signing with a fresh key and being refused. */
	useEffect(() => {
		void reconcileDeviceKey().catch(() => undefined);
	}, []);
	/* The pairing sheet lives here, above every screen, so an approval card that
	   unmounts mid-pairing (its request answered elsewhere) does not take the
	   sheet with it (D15). */
	return (
		<>
			<Screen route={route} />
			<PairingSheetHost />
		</>
	);
}

function Screen({ route }: { route: ReturnType<typeof useRoute> }) {
	switch (route.name) {
		case "account-routing":
			return <AccountRoutingScreen />;
		case "past":
			return <PastSessionsScreen />;
		case "library":
			return <LibraryScreen />;
		case "library-asset":
			return <LibraryAssetScreen key={route.assetId} assetId={route.assetId} />;
		case "notifications":
			return <NotificationsScreen />;
		case "notification":
			return <NotificationDetailScreen key={route.itemId} itemId={route.itemId} />;
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

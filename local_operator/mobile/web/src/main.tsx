import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./app";
import { initWideView } from "./lib/viewport";
import { initTheme } from "./theme";
import { initializeLocaleMetadata, resolveInitialLocale } from "./store";
import "./styles/index.css";

/* Apply the persisted theme before first paint — the attribute is what
   every --lo-* variable keys off, so a late set would flash the default
   ramp for users who picked another theme. */
initTheme();
/* The persisted wide view, for the same reason: the meta has to be right before the
   first layout or the page paints once at the default width and reflows. */
initWideView();
/* The copy's language and the document's `lang`/`dir` are resolved BEFORE the
   first content paint (i18n review round 1, QA Q-2): the backend's resolved
   language arrives on the config stream's baseline frame, and painting before
   it left a cold deep link showing English copy under Urdu metadata. What is
   painted is still decided by the audited owned bundle — `resolveInitialLocale`
   only tells the store which backend preference to test against, and a missing
   or stalled stream keeps the English default rather than a device guess. */
void resolveInitialLocale().then(() => {
	initializeLocaleMetadata();
	createRoot(document.getElementById("root")!).render(
		<StrictMode>
			<App />
		</StrictMode>,
	);
});

/* Note: private-storage cleanup on sign-out is NOT wired here. The SPA never
   renders a logout control, and the login screen is a separate server-rendered
   page the SPA is unmounted for — so a document click listener for a logout
   anchor would be dead code (U2). Cleanup lives where it is actually reachable:
   the login page's own inline clear script (daemon.py `_LOGIN_HTML`) and the
   api.ts 401 handler, both of which clear the scoped storage programmatically
   without relying on the `Clear-Site-Data` header WebKit may ignore. */


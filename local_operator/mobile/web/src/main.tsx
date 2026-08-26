import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./app";
import { initTheme } from "./theme";
import "./styles/index.css";

/* Apply the persisted theme before first paint — the attribute is what
   every --lo-* variable keys off, so a late set would flash the default
   ramp for users who picked another theme. */
initTheme();

/* Note: private-storage cleanup on sign-out is NOT wired here. The SPA never
   renders a logout control, and the login screen is a separate server-rendered
   page the SPA is unmounted for — so a document click listener for a logout
   anchor would be dead code (U2). Cleanup lives where it is actually reachable:
   the login page's own inline clear script (daemon.py `_LOGIN_HTML`) and the
   api.ts 401 handler, both of which clear the scoped storage programmatically
   without relying on the `Clear-Site-Data` header WebKit may ignore. */

createRoot(document.getElementById("root")!).render(
	<StrictMode>
		<App />
	</StrictMode>,
);

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./app";
import { initTheme } from "./theme";
import "./styles/index.css";

/* Apply the persisted theme before first paint — the attribute is what
   every --lo-* variable keys off, so a late set would flash the default
   ramp for users who picked another theme. */
initTheme();

createRoot(document.getElementById("root")!).render(
	<StrictMode>
		<App />
	</StrictMode>,
);

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./app";
import { initTheme } from "./theme";
import "./styles/index.css";
import { clearPrivateSessionStorage } from "./private-storage";

/* Apply the persisted theme before first paint — the attribute is what
   every --lo-* variable keys off, so a late set would flash the default
   ramp for users who picked another theme. */
initTheme();

document.addEventListener("click", (event) => {
	const target = event.target;
	if (target instanceof Element && target.closest('a[href="/logout"]')) {
		clearPrivateSessionStorage();
	}
});

createRoot(document.getElementById("root")!).render(
	<StrictMode>
		<App />
	</StrictMode>,
);

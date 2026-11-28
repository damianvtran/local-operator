import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

/**
 * Production build of the streaming benchmark page (`src/bench/streaming.html`).
 *
 * WHY A SEPARATE PRODUCTION BUILD: the dev server runs React's development
 * build, which is several times slower per commit and would measure the dev
 * build's overhead, not the relay's. `react-dom/client` is aliased to
 * `react-dom/profiling` so `<Profiler>` still reports commit durations in a
 * minified production bundle. The alias applies to baseline and variant alike.
 *
 * Usage (from this directory):
 *   BENCH_OUT=<dir> node_modules/.bin/vite build -c vite.bench.config.ts
 *   python3 -m http.server <port> -d <dir>     # then open /src/bench/streaming.html
 * Not wired into `build`, `check` or the shipped bundle.
 */
export default defineConfig({
	base: "./",
	plugins: [react(), tailwindcss()],
	resolve: { alias: [{ find: /^react-dom\/client$/, replacement: "react-dom/profiling" }] },
	build: {
		outDir: path.resolve(process.env.BENCH_OUT ?? "dist-bench"),
		emptyOutDir: true,
		sourcemap: false,
		target: "es2022",
		// `pele/lazy` (mermaid diagrams) is dynamic-imported and the synthetic stream
		// has no diagram fence, so it is never loaded; externalising it keeps the
		// bench buildable from a node_modules tree that predates the pele dep.
		rollupOptions: { input: path.resolve("src/bench/streaming.html"), external: ["pele/lazy"] },
	},
});

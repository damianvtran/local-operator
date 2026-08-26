import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Relative base: the daemon serves the bundle from its own root, and a
// tunnel/identity-proxy may mount it under a path — absolute asset URLs
// would break there, relative ones never do.
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "dist",
    sourcemap: false,
    target: "es2022",
    chunkSizeWarningLimit: 600,
  },
  server: {
    // Dev mode proxies the API at a running daemon so the phone UI can be
    // developed against live sessions: `lop mobile serve` on 4098.
    proxy: {
      "/api": "http://127.0.0.1:4098",
      "/healthz": "http://127.0.0.1:4098",
      // The auth gate lives on the daemon: without these, dev-mode browsers
      // can never log in (the SPA has no login page of its own) and the
      // 401 → reload loop never lands anywhere useful.
      "/login": "http://127.0.0.1:4098",
      "/logout": "http://127.0.0.1:4098",
    },
  },
	test: {
		// happy-dom does not survive vitest's global copy on all Node versions
		// (its localStorage getter is non-enumerable). src/test-setup.ts
		// re-attaches real Storage so the suite runs on any supported Node.
		setupFiles: ["./src/test-setup.ts"],
	},
});

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

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
    // developed against live sessions: `lop mobile serve` on 4097.
    proxy: {
      "/api": "http://127.0.0.1:4097",
      "/healthz": "http://127.0.0.1:4097",
    },
  },
});

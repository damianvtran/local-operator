#!/usr/bin/env node
/* Serves real popup modules in a normal browser tab. No browser is launched and
 * no extension APIs, profile, daemon or credentials are accessed. The iframe is
 * a DOM fixture, NOT Chrome's native circular popup viewport or OS launch time.
 * BEFORE is immutable git content; AFTER is rebuilt per request for review. */
import { createServer } from "node:http";
import { execFileSync } from "node:child_process";
import { mkdtemp, open, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { build } from "esbuild";

for (const key of Object.keys(process.env)) if (/^(CMUX_|LOP_)/.test(key)) delete process.env[key];
const root = resolve(import.meta.dirname, "..");
const base = process.argv[2] || "fb2525c1a738f57fb7b2970965f6316517aaab85";
const scratch = await mkdtemp(join(tmpdir(), "lop-popup-performance-"));
const archive = await open(join(scratch, "base.tar"), "w");
try {
  execFileSync("git", ["archive", base, "extension/src"], { cwd: resolve(root, ".."), stdio: ["ignore", archive.fd, "inherit"] });
} finally {
  await archive.close();
}
execFileSync("tar", ["-xf", join(scratch, "base.tar"), "-C", scratch]);
const results = new Map();
const server = createServer(async (req, res) => {
  // Review reloads must execute the current worktree, never a cached bundle.
  res.setHeader("Cache-Control", "no-store");
  try {
    const url = new URL(req.url, "http://localhost");
    if (url.pathname === "/metrics" && req.method === "POST") {
      let body = ""; for await (const chunk of req) body += chunk;
      results.set(url.searchParams.get("key"), JSON.parse(body)); res.end("ok"); return;
    }
    if (url.pathname === "/metrics") { res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify(Object.fromEntries(results), null, 2)); return; }
    const variant = url.searchParams.get("variant") === "before" ? "before" : "after";
    const requestedState = url.searchParams.get("state");
    const state = ["healthy", "slow", "unreachable", "standby", "consent", "stalled-storage"].includes(requestedState) ? requestedState : "healthy";
    const src = variant === "before" ? join(scratch, "extension/src") : join(root, "src");
    if (url.pathname === "/stub.mjs") { res.setHeader("Content-Type", "text/javascript"); res.end(await readFile(join(root, "tests/fixtures/popup-performance.mjs"))); return; }
    if (url.pathname.endsWith("popup.js")) {
      const output = await build({ entryPoints: [join(src, "popup/popup.ts")], bundle: true, format: "esm", target: "chrome116", write: false });
      res.setHeader("Content-Type", "text/javascript"); res.end(output.outputFiles[0].text); return;
    }
    if (url.pathname.endsWith("popup.css") || url.pathname.endsWith("first-paint.js")) {
      res.setHeader("Content-Type", url.pathname.endsWith("css") ? "text/css" : "text/javascript"); res.end(await readFile(join(src, "popup", url.pathname.split("/").pop()))); return;
    }
    if (url.pathname === "/frame") {
      let html = await readFile(join(src, "popup/popup.html"), "utf8");
      const config = { state, delay: Number(url.searchParams.get("delay") || 0), large: url.searchParams.get("large") === "1" };
      html = html.replace('<script src="first-paint.js"></script>', `<script>localStorage.clear(); ${url.searchParams.get("warm") === "1" ? 'localStorage.setItem("lop:pin-hint", "148px");' : ''}</script><script src="first-paint.js?variant=${variant}"></script>`);
      html = html.replace('href="popup.css"', `href="popup.css?variant=${variant}"`);
      html = html.replace(/<script type="module" src="popup.js"><\/script>/, `<script type="module">
import { installPopupPerformance } from '/stub.mjs';
const fixture = installPopupPerformance(${JSON.stringify(config)});
const report = () => {
 const visible = [...document.querySelectorAll('.state')].filter(n => !n.classList.contains('hidden')).map(n => n.id);
 const populated = visible.length && !visible.includes('pending');
 if (populated && fixture.metrics.firstPopulatedMs == null) fixture.metrics.firstPopulatedMs = performance.now() - fixture.start;
 const card = document.getElementById('card');
 parent.postMessage({ ...fixture.metrics, visible, geometry: { card: card.getBoundingClientRect().toJSON(), scrollHeight: document.documentElement.scrollHeight, clientHeight: document.documentElement.clientHeight, bodyHeight: document.body.getBoundingClientRect().height } }, location.origin);
};
new MutationObserver(records => { fixture.metrics.mutations += records.length; report(); }).observe(document.body, { subtree: true, attributes: true, childList: true, characterData: true });
window.addEventListener('message', e => { if(e.origin === location.origin && e.data === 'burst') for(let i=0;i<30;i++) fixture.emit({connState:{newValue:'connected'}}); });
window.addEventListener('unhandledrejection', e => { fixture.metrics.error = String(e.reason); report(); });
await import('/popup.js?variant=${variant}'); report(); setInterval(report, 500);
</script>`);
      res.setHeader("Content-Type", "text/html"); res.end(html); return;
    }
    res.setHeader("Content-Type", "text/html");
    const query = url.searchParams.toString();
    res.end(`<!doctype html><title>Popup performance fixture</title><style>body{font:14px system-ui;margin:24px;display:flex;gap:24px;background:#eee}iframe{width:300px;height:600px;border:0}pre{white-space:pre-wrap;max-width:750px;font-size:12px}button{padding:8px}</style><div><h2>${variant.toUpperCase()} / ${state}</h2><iframe src="/frame?${query.replaceAll("&", "&amp;")}"></iframe></div><div><p>Shipped popup DOM / synthetic Chrome storage and health. Not a native popup or OS launch benchmark.</p><button id="burst">30 storage events</button><pre id="metrics">Loading…</pre></div><script>const key=${JSON.stringify(query)};document.getElementById('burst').onclick=()=>document.querySelector('iframe').contentWindow.postMessage('burst',location.origin);addEventListener('message',e=>{if(e.origin!==location.origin)return;document.getElementById('metrics').textContent=JSON.stringify(e.data,null,2);fetch('/metrics?key='+encodeURIComponent(key),{method:'POST',body:JSON.stringify(e.data)});});</script>`);
  } catch (error) { res.statusCode = 500; res.end(String(error)); }
});
server.listen(Number(process.argv[3] || 0), "127.0.0.1", () => console.log(`http://127.0.0.1:${server.address().port}/?variant=before&state=healthy&delay=100&large=1\nBEFORE ${base}; AFTER ${root}; GET /metrics for captured browser measurements`));
process.on("SIGTERM", () => server.close(async () => { await rm(scratch, { recursive: true, force: true }); process.exit(0); }));

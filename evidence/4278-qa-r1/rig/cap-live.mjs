#!/usr/bin/env node
/* QA r1 #4278 M1 live-delivery repro: phone viewing an IDLE live session while a Team Bridge
   message is delivered into it. node cap-live.mjs --cdp <port> --base <relay> --out <dir> --mid <id> --ready <file>
   Password from env QA_PW. The delivery itself is fired by the caller once <ready> exists. */
import { existsSync, writeFileSync } from "node:fs";
import { join } from "node:path";
const flag = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i > 0 ? process.argv[i + 1] : d; };
const CDP = flag("cdp"), BASE = flag("base"), OUT = flag("out"), MID = flag("mid"), READY = flag("ready"), THEME = flag("theme", "localOperatorLight");
const SID = "abcdef012345"; const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const ver = await (await fetch(`http://127.0.0.1:${CDP}/json/version`)).json();
const ws = new WebSocket(ver.webSocketDebuggerUrl); await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
let nextId = 1; const pending = new Map(); const listeners = [];
ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { const { r, j } = pending.get(m.id); pending.delete(m.id); m.error ? j(new Error(JSON.stringify(m.error))) : r(m.result); } else listeners.forEach((f) => f(m)); };
const send = (method, params = {}, sessionId) => new Promise((r, j) => { const id = nextId++; pending.set(id, { r, j }); ws.send(JSON.stringify({ id, method, params, sessionId })); });
const { targetId } = await send("Target.createTarget", { url: "about:blank" });
const { sessionId: S } = await send("Target.attachToTarget", { targetId, flatten: true });
const P = (m, p) => send(m, p, S);
await P("Page.enable"); await P("Runtime.enable"); await P("Network.enable"); await P("Network.clearBrowserCookies");
await P("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
const t0 = Date.now(); const log = [];
listeners.push((m) => { if (m.sessionId !== S) return; const p = m.params || {};
	if (m.method === "Network.requestWillBeSent" && p.request.url.includes("/bridge-cards/")) log.push({ ms: Date.now() - t0, ev: "request", id: p.requestId, url: p.request.url.replace(BASE, "") });
	if (m.method === "Network.responseReceived" && p.response.url.includes("/bridge-cards/")) log.push({ ms: Date.now() - t0, ev: "response", id: p.requestId, status: p.response.status, cache: p.response.headers["cache-control"] });
	if (m.method === "Network.loadingFailed") log.push({ ms: Date.now() - t0, ev: "failed", id: p.requestId, err: p.errorText });
});
const evaluate = async (e) => { const r = await P("Runtime.evaluate", { expression: e, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails)); return r.result.value; };
const shot = async (n) => { const r = await P("Page.captureScreenshot", { format: "png" }); writeFileSync(join(OUT, `${n}.png`), Buffer.from(r.data, "base64")); console.log("frame", n); };
await P("Page.navigate", { url: `${BASE}/login` }); await pause(1500);
await evaluate(`(() => { const i = document.querySelector('input[type=password],input[name=password]'); i.value = ${JSON.stringify(process.env.QA_PW)}; i.form.submit(); })()`);
await pause(2500);
await evaluate(`localStorage.setItem('lo-mobile-theme', ${JSON.stringify(THEME)})`);
await P("Page.navigate", { url: `${BASE}/#/s/${SID}` }); await pause(1000); await P("Page.reload", {}); await pause(6000);
for (let k = 0; k < 4; k++) { const n = await evaluate("(() => { const bs = [...document.querySelectorAll('button[data-condense-header][aria-expanded=false]')]; bs.forEach((b) => b.click()); return bs.length; })()"); log.push({ ms: Date.now() - t0, ev: "folds-opened", n }); if (!n) break; await pause(1200); }
await pause(3000);
log.push({ ms: Date.now() - t0, ev: "panels-before-delivery", n: await evaluate("document.querySelectorAll('[data-bridge-card-body]').length") });
log.push({ ms: Date.now() - t0, ev: "phone-viewing-ready" });
writeFileSync(READY, String(Date.now()));
// Wait for the live card's body request, then observe 25 s more for any re-read.
const end = Date.now() + 60000;
while (Date.now() < end && !log.some((e) => e.ev === "request" && /\/bridge-cards\/peer-/.test(e.url))) await pause(100);
log.push({ ms: Date.now() - t0, ev: "first-live-request-seen-or-timeout" });
await pause(25000);
const panel = await evaluate(`(() => { const rows = [...document.querySelectorAll('[data-bridge-card-body]')]; const p = rows[rows.length - 1]; if (!p) return null; p.scrollIntoView({ block: 'center' }); return { state: p.getAttribute('data-state'), text: p.innerText.slice(0, 200), count: rows.length }; })()`);
await pause(600); await shot("m1-live-after-25s");
log.push({ ms: Date.now() - t0, ev: "panel", panel });
// Re-read the same row server-side now that the journal should hold it.
const liveUrl = (log.find((e) => e.ev === 'request' && /\/bridge-cards\/peer-/.test(e.url)) || {}).url || `/api/sessions/${SID}/bridge-cards/none/body`;
const later = await evaluate(`fetch('${liveUrl}').then(async r => ({ status: r.status, body: (await r.text()).slice(0, 200) }))`);
log.push({ ms: Date.now() - t0, ev: "manual-reread", later });
// Remount (leave and come back): does a fresh mount recover?
await P("Page.navigate", { url: `${BASE}/#/` }); await pause(1500); await P("Page.navigate", { url: `${BASE}/#/s/${SID}` }); await pause(5000);
const panel2 = await evaluate(`(() => { const rows = [...document.querySelectorAll('[data-bridge-card-body]')]; const p = rows[rows.length - 1]; if (!p) return null; p.scrollIntoView({ block: 'center' }); return { state: p.getAttribute('data-state'), text: p.innerText.slice(0, 200) }; })()`);
await pause(600); await shot("m1-live-after-remount");
log.push({ ms: Date.now() - t0, ev: "panel-after-remount", panel2 });
writeFileSync(join(OUT, "m1-live-timeline.json"), JSON.stringify(log, null, 1));
console.log(JSON.stringify(log, null, 1));
await send("Target.closeTarget", { targetId }); ws.close(); process.exit(0);

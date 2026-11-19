#!/usr/bin/env node
/*
 * QA r1 core #4278 — phone web relay capture over CDP (headless Chrome the caller started).
 *   node cap.mjs --cdp <port> --base http://127.0.0.1:<relay> --out <dir> --prefix after
 *     [--themes localOperatorLight,localOperatorDark] [--mode full|frames0|nocap]
 * Password from env QA_PW (never argv). Writes PNGs + <prefix>-net.json + <prefix>-geom.json.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const flag = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i > 0 ? process.argv[i + 1] : d; };
const CDP = flag("cdp"); const BASE = flag("base"); const OUT = flag("out"); const PREFIX = flag("prefix", "after");
const THEMES = flag("themes", "localOperatorLight,localOperatorDark").split(",");
const MODE = flag("mode", "full");
const SID = "abcdef012345";
mkdirSync(OUT, { recursive: true });
const pause = (ms) => new Promise((r) => setTimeout(r, ms));

const ver = await (await fetch(`http://127.0.0.1:${CDP}/json/version`)).json();
const ws = new WebSocket(ver.webSocketDebuggerUrl);
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
let nextId = 1; const pending = new Map(); const listeners = [];
ws.onmessage = (ev) => {
	const m = JSON.parse(ev.data);
	if (m.id && pending.has(m.id)) { const { r, j } = pending.get(m.id); pending.delete(m.id); m.error ? j(new Error(JSON.stringify(m.error))) : r(m.result); }
	else listeners.forEach((f) => f(m));
};
const send = (method, params = {}, sessionId) => new Promise((r, j) => { const id = nextId++; pending.set(id, { r, j }); ws.send(JSON.stringify({ id, method, params, sessionId })); });

const { targetId } = await send("Target.createTarget", { url: "about:blank" });
const { sessionId: S } = await send("Target.attachToTarget", { targetId, flatten: true });
const P = (m, p) => send(m, p, S);
await P("Page.enable"); await P("Runtime.enable"); await P("Network.enable");
await P("Network.clearBrowserCookies"); await P("Network.setCacheDisabled", { cacheDisabled: false });
await P("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
await P("Emulation.setTouchEmulationEnabled", { enabled: true });

// Network log: every request + its ExtraInfo (real Cookie header) + response status/headers.
const net = new Map(); let mountEpoch = "init";
listeners.push((m) => {
	if (m.sessionId !== S) return;
	const p = m.params || {};
	const rec = (id) => { if (!net.has(id)) net.set(id, { id, epoch: mountEpoch }); return net.get(id); };
	if (m.method === "Network.requestWillBeSent") Object.assign(rec(p.requestId), { url: p.request.url, method: p.request.method, t: p.timestamp });
	if (m.method === "Network.requestWillBeSentExtraInfo") rec(p.requestId).cookieSent = Boolean(p.headers?.Cookie || p.headers?.cookie);
	if (m.method === "Network.responseReceived") Object.assign(rec(p.requestId), { status: p.response.status, cacheControl: p.response.headers["cache-control"] ?? p.response.headers["Cache-Control"] ?? null, fromDiskCache: p.response.fromDiskCache });
	if (m.method === "Network.loadingFailed") Object.assign(rec(p.requestId), { failed: p.errorText });
});
const bodyReqs = () => [...net.values()].filter((r) => r.url && r.url.includes("/bridge-cards/"));

const evaluate = async (expr) => { const r = await P("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails)); return r.result.value; };
const navigate = async (url) => { await P("Page.navigate", { url }); await pause(1500); };
const waitFor = async (expr, ms = 15000) => { const end = Date.now() + ms; while (Date.now() < end) { if (await evaluate(expr)) return true; await pause(200); } return false; };
const shot = async (name) => { const r = await P("Page.captureScreenshot", { format: "png" }); writeFileSync(join(OUT, `${name}.png`), Buffer.from(r.data, "base64")); console.log("frame", name); };

const GEOM = `(() => { const R = (e) => { const r = e.getBoundingClientRect(); return { y: Math.round(r.y), h: Math.round(r.height), w: Math.round(r.width) }; };
  return { theme: document.documentElement.dataset.theme, lang: document.documentElement.lang, docW: document.documentElement.scrollWidth, vw: innerWidth,
    panels: [...document.querySelectorAll('[data-bridge-card-body]')].map((p) => ({ state: p.getAttribute('data-state'), ...R(p), scrollW: p.scrollWidth, clientW: p.clientWidth,
      text: p.innerText.slice(0, 220), toggle: p.querySelector('button') ? { text: p.querySelector('button').innerText, expanded: p.querySelector('button').getAttribute('aria-expanded') } : null,
      links: [...p.querySelectorAll('a')].map((a) => a.getAttribute('href')), rawHtml: p.querySelectorAll('script,iframe,img').length })),
    receipts: [...document.querySelectorAll('*')].filter((e) => e.children.length === 0 && /·/.test(e.textContent) && /Ola|bcv|Team|·/.test(e.textContent)).slice(0, 12).map((e) => e.textContent.slice(0, 80)) }; })()`;

// 1) Unauthenticated read of the body route from the page origin, before login: expect 401.
await navigate(`${BASE}/login`);
const pre = await evaluate(`fetch('/api/sessions/${SID}/bridge-cards/e_admit/body').then(async r => ({ status: r.status, body: (await r.text()).slice(0, 120) }))`);
console.log("unauth", JSON.stringify(pre));
await shot(`${PREFIX}-00-login`);

// 2) Real form login.
const ok = await evaluate(`(() => { const i = document.querySelector('input[type=password],input[name=password]'); if (!i) return false; i.value = ${JSON.stringify(process.env.QA_PW)}; i.form.submit(); return true; })()`);
if (!ok) throw new Error("no password field");
await pause(2500);
const cookies = await P("Network.getCookies", { urls: [BASE] });
console.log("cookies", cookies.cookies.map((c) => ({ name: c.name, httpOnly: c.httpOnly, sameSite: c.sameSite })));

const geoms = {};
const openSession = async (reload = false) => { mountEpoch = `mount-${Date.now()}`; await navigate(`${BASE}/#/s/${SID}`); if (reload) { await P("Page.reload", { ignoreCache: true }); await pause(1500); } await waitFor(`document.body.innerText.includes('Watch for Team Bridge') || document.body.innerText.includes('Team Bridge')`, 20000); await pause(2500); let opened = 0; for (let k = 0; k < 4; k++) { const n = await evaluate("(() => { const bs = [...document.querySelectorAll('button[data-condense-header][aria-expanded=false]')]; bs.forEach((b) => b.click()); return bs.length; })()"); opened += n; if (!n) break; await pause(1200); } console.log('folds opened', opened); await pause(2000); };

for (const theme of THEMES) {
	const tn = theme.includes("Light") ? "light" : "dark";
	await evaluate(`localStorage.setItem('lo-mobile-theme', ${JSON.stringify(theme)})`);
	await openSession(true);
	geoms[`${tn}-mount`] = await evaluate(GEOM);
	await evaluate(`document.querySelector('[data-lo-transcript-content], main')?.scrollTo?.(0,0)`);
	if (MODE !== "full") {
		// before / nocap / zh: a tour of the transcript at the phone viewport.
		await evaluate(`(document.querySelector('[data-bridge-card-body]') || [...document.querySelectorAll('*')].find(e => e.children.length===0 && e.textContent.includes('Watch for Team Bridge')))?.scrollIntoView({block:'start'})`);
		await pause(600); await shot(`${PREFIX}-${tn}-01-top`);
		for (let k = 2; k <= 6; k++) { await evaluate(`(() => { let s = document.scrollingElement; const all = [...document.querySelectorAll('*')].filter(e => /(auto|scroll)/.test(getComputedStyle(e).overflowY) && e.scrollHeight > e.clientHeight + 10); if (all.length) s = all.sort((a,b)=>b.clientHeight-a.clientHeight)[0]; s.scrollBy(0, 640); })()`); await pause(500); await shot(`${PREFIX}-${tn}-0${k}-scroll`); }
		continue;
	}
	const names = ["01-admitted-short", "02-long-collapsed", "04-held-awaiting-review", "05-held-transient", "06-refused", "07-deleted", "08-offline", "09-coalesced"];
	const count = await evaluate(`document.querySelectorAll('[data-bridge-card-body]').length`);
	console.log(theme, "panels", count);
	for (let i = 0; i < count && i < names.length; i++) {
		await evaluate(`document.querySelectorAll('[data-bridge-card-body]')[${i}].scrollIntoView({block:'center'})`);
		await pause(500); await shot(`${PREFIX}-${tn}-${names[i]}`);
		if (i === 1) {
			await evaluate(`document.querySelectorAll('[data-bridge-card-body]')[1].querySelector('button')?.click()`);
			await pause(700);
			geoms[`${tn}-long-expanded`] = await evaluate(`(() => { const p = document.querySelectorAll('[data-bridge-card-body]')[1]; const b = p.querySelector('button'); return { h: Math.round(p.getBoundingClientRect().height), toggle: b && b.innerText, expanded: b && b.getAttribute('aria-expanded'), tail: p.innerText.slice(-160) }; })()`);
			await evaluate(`document.querySelectorAll('[data-bridge-card-body]')[1].scrollIntoView({block:'start'})`); await pause(400);
			await shot(`${PREFIX}-${tn}-03a-long-expanded-top`);
			await evaluate(`document.querySelectorAll('[data-bridge-card-body]')[1].querySelector('button').scrollIntoView({block:'end'})`); await pause(400);
			await shot(`${PREFIX}-${tn}-03b-long-expanded-end`);
			await evaluate(`document.querySelectorAll('[data-bridge-card-body]')[1].querySelector('button')?.click()`); await pause(400);
		}
	}
	// Coalesced tail ("N more messages").
	await evaluate(`(() => { const ps = document.querySelectorAll('[data-bridge-card-body]'); ps[ps.length-1]?.lastElementChild?.scrollIntoView({block:'end'}); })()`);
	await pause(500); await shot(`${PREFIX}-${tn}-10-coalesced-end`);
}

// 3) Remount check: leave to the list and come back; each mount must refetch.
if (MODE === "full" || MODE === "nocap") {
	mountEpoch = "list"; await navigate(`${BASE}/#/`); await pause(1500);
	await openSession();
	geoms["remount"] = await evaluate(GEOM);
}
// 4) Logout then read: 401 after the cookie is gone.
await evaluate(`fetch('/logout', { method: 'POST' }).catch(() => null)`);
await P("Network.clearBrowserCookies");
const post = await evaluate(`fetch('/api/sessions/${SID}/bridge-cards/e_admit/body').then(async r => ({ status: r.status }))`);
console.log("after-clear-cookies", JSON.stringify(post));

const reqs = bodyReqs();
writeFileSync(join(OUT, `${PREFIX}-net.json`), JSON.stringify({ unauthBeforeLogin: pre, afterClearCookies: post, bodyRequests: reqs }, null, 1));
writeFileSync(join(OUT, `${PREFIX}-geom.json`), JSON.stringify(geoms, null, 1));
const byEpoch = {}; for (const r of reqs) { const k = r.epoch; byEpoch[k] = byEpoch[k] || {}; const e = r.url.split("/bridge-cards/")[1]?.split("/")[0]; byEpoch[k][e] = (byEpoch[k][e] || 0) + 1; }
console.log("bodyRequests", reqs.length, JSON.stringify(byEpoch));
console.log("cookieSent", reqs.map((r) => r.cookieSent).filter((x) => x !== true).length === 0, "cache", [...new Set(reqs.map((r) => r.cacheControl))], "status", [...new Set(reqs.map((r) => r.status))]);
await send("Target.closeTarget", { targetId });
ws.close();
process.exit(0);

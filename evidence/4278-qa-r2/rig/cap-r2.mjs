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
// ---- QA r2 #4278 (delta): M1 not-yet-written retry, abort, unknown-session, D3, D5, N1, empty.
const EPOCH = process.env.QA_EPOCH_FILE, UNK = process.env.QA_UNK_FILE;
const { unlinkSync, existsSync } = await import("node:fs");
const setEpoch = (e) => { writeFileSync(EPOCH, e); mountEpoch = e; };
const out = {};
const SCROLLER = `(() => { const all = [...document.querySelectorAll('*')].filter(e => /(auto|scroll)/.test(getComputedStyle(e).overflowY) && e.scrollHeight > e.clientHeight + 10); return all.sort((a,b)=>b.clientHeight-a.clientHeight)[0] || document.scrollingElement; })()`;
const panelsState = `[...document.querySelectorAll('[data-bridge-card-body]')].map(p => p.getAttribute('data-state'))`;
const openFolds = async () => { let opened = 0; for (let k = 0; k < 4; k++) { const n = await evaluate("(() => { const bs = [...document.querySelectorAll('button[data-condense-header][aria-expanded=false]')]; bs.forEach((b) => b.click()); return bs.length; })()"); opened += n; if (!n) break; await pause(300); } return opened; };
const loadSession = async () => { await navigate(`${BASE}/#/`); await navigate(`${BASE}/#/s/${SID}`); await P("Page.reload", { ignoreCache: true }); await pause(1500); await waitFor(`document.body.innerText.includes('Watch for Team Bridge')`, 20000); await pause(2000); };
// In-page recorder: every 25 ms, the panel states by DOM index (card order).
const startRec = `(() => { window.__rec = []; const t0 = performance.now(); let last = ''; window.__recT = setInterval(() => { const s = JSON.stringify(${panelsState}); if (s !== last) { last = s; window.__rec.push({ ms: Math.round(performance.now() - t0), states: JSON.parse(s) }); } }, 25); return true; })()`;
const stopRec = `(() => { clearInterval(window.__recT); return window.__rec; })()`;
const reqsFor = (ep) => bodyReqs().filter((r) => r.epoch === ep).map((r) => ({ entry: r.url.split("/bridge-cards/")[1].split("/")[0], status: r.status, failed: r.failed, t: r.t }));
const tally = (rs) => rs.reduce((a, r) => ((a[r.entry] = (a[r.entry] || []).concat(r.status ?? r.failed)), a), {});

await navigate(`${BASE}/login`);
await evaluate(`(() => { const i = document.querySelector('input[type=password],input[name=password]'); i.value = ${JSON.stringify(process.env.QA_PW)}; i.form.submit(); return true; })()`);
await pause(2500);
await evaluate(`localStorage.setItem('lo-mobile-theme', 'localOperatorLight')`);

// A) M1: e_admit late x1, e_long late x2, e_held never written (relay QA_LATE). Loading -> body.
setEpoch("m1a"); await loadSession();
out.m1a_panelsBeforeOpen = await evaluate(`document.querySelectorAll('[data-bridge-card-body]').length`);
await evaluate(startRec);
const tOpen = Date.now(); out.m1a_folds = await openFolds();
await pause(Math.max(0, 250 - (Date.now() - tOpen)));
await evaluate(`document.querySelectorAll('[data-bridge-card-body]')[0]?.scrollIntoView({block:'start'})`);
await shot("m1-a-t0250-loading"); out.m1a_t250 = await evaluate(panelsState);
await pause(Math.max(0, 1100 - (Date.now() - tOpen))); await shot("m1-a-t1100-admit-body-long-loading"); out.m1a_t1100 = await evaluate(panelsState);
await pause(Math.max(0, 4500 - (Date.now() - tOpen))); await shot("m1-a-t4500-settled"); out.m1a_t4500 = await evaluate(panelsState);
await evaluate(`document.querySelectorAll('[data-bridge-card-body]')[2]?.scrollIntoView({block:'center'})`); await pause(300); await shot("m1-a-held-missing-unavailable");
out.m1a_rec = await evaluate(stopRec);
out.m1a_reqs = reqsFor("m1a"); out.m1a_tally = tally(out.m1a_reqs);
out.m1a_texts = await evaluate(`[...document.querySelectorAll('[data-bridge-card-body]')].slice(0,3).map(p => p.innerText.slice(0,90))`);

// B) Abort: navigate away right after the first not-yet answer; no re-ask may follow.
setEpoch("m1b"); await loadSession();
await openFolds();
const end = Date.now() + 5000;
while (Date.now() < end && !bodyReqs().some((r) => r.epoch === "m1b" && /e_held\//.test(r.url) && r.status)) await pause(20);
const tAway = Date.now(); await P("Page.navigate", { url: `${BASE}/#/` }); out.m1b_awayAfterFirst404ms = Date.now() - tAway;
await pause(4000);
out.m1b_reqs = reqsFor("m1b"); out.m1b_tally = tally(out.m1b_reqs);
out.m1b_panelsAfterAway = await evaluate(`document.querySelectorAll('[data-bridge-card-body]').length`);
await shot("m1-b-after-navigate-away");

// C) Unknown session: transcript loaded, then the route's own session check refuses; one read each.
setEpoch("m1c"); await loadSession();
writeFileSync(UNK, "on");
await evaluate(startRec); await openFolds(); await pause(4000);
out.m1c_rec = await evaluate(stopRec);
out.m1c_reqs = reqsFor("m1c"); out.m1c_tally = tally(out.m1c_reqs);
out.m1c_states = await evaluate(panelsState);
await evaluate(`document.querySelectorAll('[data-bridge-card-body]')[0]?.scrollIntoView({block:'start'})`); await pause(300); await shot("m1-c-unknown-session-unavailable");
unlinkSync(UNK);

// D) Layout: D5 gaps, N1 ellipsis, empty answer, D3 Show less.
setEpoch("lay"); await loadSession(); await openFolds(); await pause(4000);
out.lay_states = await evaluate(panelsState);
out.d5 = await evaluate(`(() => [...document.querySelectorAll('[data-bridge-card-body]')].map((p, i) => { const r = p.getBoundingClientRect(); let row = p; while (row.parentElement && !row.nextElementSibling) row = row.parentElement; let nx = p.nextElementSibling ? p.nextElementSibling : row.nextElementSibling; while (nx && nx.getBoundingClientRect().height === 0) nx = nx.nextElementSibling; const n = nx ? nx.getBoundingClientRect() : null; return { i, state: p.getAttribute('data-state'), hidden: p.hidden, h: Math.round(r.height), marginBottom: getComputedStyle(p).marginBottom, marginTop: getComputedStyle(p).marginTop, bottom: +r.bottom.toFixed(1), nextTop: n ? +n.top.toFixed(1) : null, gap: n ? +(n.top - r.bottom).toFixed(1) : null, nextText: nx ? nx.innerText.slice(0, 50) : null }; }))()`);
await evaluate(`document.querySelectorAll('[data-bridge-card-body]')[0].scrollIntoView({block:'start'})`); await evaluate(`${SCROLLER}.scrollBy(0, -60)`); await pause(400); await shot("d5-gap-light");
// N1
out.n1 = await evaluate(`(() => [8, 9].map((i) => { const p = document.querySelectorAll('[data-bridge-card-body]')[i]; const b = p.querySelector('[data-message-id]'); const t = b.innerText.trimEnd(); return { i, toggle: p.querySelector('button')?.innerText, tail: t.slice(-40), endsWithEllipsis: t.endsWith('…'), doubled: /[.!?]…$/.test(t) }; }))()`);
for (const [i, n] of [[8, "n1-sentence-cut-no-ellipsis"], [9, "n1-negative-midsentence-ellipsis"]]) { await evaluate(`document.querySelectorAll('[data-bridge-card-body]')[${i}].querySelector('button').scrollIntoView({block:'center'})`); await pause(400); await shot(n); }
// Empty answer: e_empty (index 10) — hidden, zero height, no frame text.
out.empty = await evaluate(`(() => { const ps = document.querySelectorAll('[data-bridge-card-body]'); const p = ps[10]; const r = p.getBoundingClientRect(); return { count: ps.length, state: p.getAttribute('data-state'), hidden: p.hidden, h: r.height, display: getComputedStyle(p).display, text: p.innerText, visiblePanelsAfterN1neg: [...ps].slice(10).filter(x => x.getBoundingClientRect().height > 0).length }; })()`);
await evaluate(`${SCROLLER}.scrollTo(0, ${SCROLLER}.scrollHeight)`); await pause(500); await shot("empty-card-receipt-only-end");
// D3: open the long card, put its Show less near the bottom of the viewport, tap it.
const ctr = async (sel) => evaluate(`(() => { const e = ${sel}; const r = e.getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; })()`);
const tap = async ({ x, y }) => { await P("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ x, y }] }); await P("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] }); };
const LONGBTN = `document.querySelectorAll('[data-bridge-card-body]')[1].querySelector('button[data-message-toggle]')`;
await evaluate(`${LONGBTN}.scrollIntoView({block:'center'})`); await pause(400);
await tap(await ctr(LONGBTN)); await pause(900);
out.d3_afterShowMore = await evaluate(`${LONGBTN}.getAttribute('aria-expanded')`);
await evaluate(`(() => { const b = ${LONGBTN}; const s = ${SCROLLER}; s.scrollBy(0, b.getBoundingClientRect().bottom - (innerHeight - 220)); })()`); await pause(600);
const D3 = `(() => { const s = ${SCROLLER}; const b = ${LONGBTN}; const p = document.querySelectorAll('[data-bridge-card-body]')[1]; const br = b.getBoundingClientRect(), pr = p.getBoundingClientRect(); return { scrollTop: Math.round(s.scrollTop), scrollHeight: s.scrollHeight, clientHeight: s.clientHeight, btnTop: Math.round(br.top), btnText: b.innerText, expanded: b.getAttribute('aria-expanded'), cardTop: Math.round(pr.top), cardBottom: Math.round(pr.bottom), cardInView: pr.bottom > 0 && pr.top < innerHeight, btnInView: br.bottom > 0 && br.top < innerHeight, vh: innerHeight }; })()`;
out.d3_before = await evaluate(D3); await shot("d3-1-before-tap-show-less");
await tap(await ctr(LONGBTN));
await evaluate(`new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))`);
out.d3_after_frame = await evaluate(D3); await shot("d3-2-after-tap-next-frame");
await pause(800); out.d3_after_settled = await evaluate(D3); await shot("d3-3-after-tap-settled");

writeFileSync(join(OUT, "r2-results.json"), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
await send("Target.closeTarget", { targetId }); ws.close(); process.exit(0);

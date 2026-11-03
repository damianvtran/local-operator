/* Real-path E2E for CLI-first pairing.
 *
 * Real daemon (uvicorn child, isolated HOME/config root, OS-picked port), the
 * BUILT extension (dist copy with the daemon port rewritten so it can never dial
 * the operator's 4099), loaded via CDP Extensions.loadUnpacked into a throwaway
 * headless Chrome (--use-mock-keychain, unique profile, port 0), and the REAL
 * `python -m local_operator.cli browser pair`. Typing is real CDP input
 * (Input.insertText + Enter), not value assignment.
 *
 * Usage: node pair-e2e.mjs <repo> <outdir>
 * Reaps: daemon process group + Chrome process group by exact pgid, then sweeps
 * the unique profile prefix and asserts zero survivors.
 */
import { spawn, spawnSync } from "node:child_process";
import { cp, mkdir, readFile, readdir, writeFile, rm } from "node:fs/promises";
import { existsSync, statSync, readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { setTimeout as sleep } from "node:timers/promises";
import net from "node:net";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const REPO = resolve(process.argv[2]);
const OUT = resolve(process.argv[3]);
const PY = join(REPO, ".venv/bin/python");
const log = (...a) => console.log(new Date().toISOString().slice(11, 23), ...a);
const report = { steps: [], frames: {} };
const t0 = Date.now();
const step = (name, data) => {
  const row = { t_s: +((Date.now() - t0) / 1000).toFixed(2), name, ...data };
  report.steps.push(row);
  log(name, JSON.stringify(data ?? {}).slice(0, 400));
};

function freePort() {
  return new Promise((res) => {
    const s = net.createServer();
    s.listen(0, "127.0.0.1", () => {
      const p = s.address().port;
      s.close(() => res(p));
    });
  });
}

const ROOT = mkdtempSync(join(tmpdir(), "lo-pair-e2e-"));
const CONFIG = join(ROOT, ".local-operator");
const env = Object.fromEntries(
  Object.entries(process.env).filter(([k]) => !/^(CMUX_|LOP_|LOCAL_OPERATOR_)/.test(k)),
);
Object.assign(env, {
  HOME: ROOT,
  LOCAL_OPERATOR_CONFIG_DIR: CONFIG,
  NO_COLOR: "1",
  LOCAL_OPERATOR_LANGUAGE: "en",
  GIT_CONFIG_SYSTEM: "/dev/null",
  GIT_TERMINAL_PROMPT: "0",
});

const children = [];
function reap() {
  for (const pid of children) {
    try {
      process.kill(-pid, "SIGTERM");
    } catch {}
  }
}
process.on("exit", reap);

function cli(...args) {
  const started = Date.now();
  const r = spawnSync(PY, ["-m", "local_operator.cli", "browser", ...args], {
    cwd: ROOT,
    env,
    encoding: "utf8",
    timeout: 60_000,
  });
  return { rc: r.status, out: r.stdout, err: r.stderr, ms: Date.now() - started };
}

function pending() {
  const file = join(CONFIG, "run/browser/pairing-pending.json");
  if (!existsSync(file)) return {};
  return JSON.parse(readFileSync(file, "utf8")).pending ?? {};
}
function expireAll() {
  const file = join(CONFIG, "run/browser/pairing-pending.json");
  if (!existsSync(file)) return;
  const saved = JSON.parse(readFileSync(file, "utf8"));
  for (const entry of Object.values(saved.pending ?? {})) entry.expires_at = Date.now() / 1000 - 1;
  writeFileSync(file, JSON.stringify(saved));
}

async function main() {
  await mkdir(OUT, { recursive: true });
  const daemonPort = await freePort();
  // --- daemon
  const daemon = spawn(
    PY,
    [
      "-c",
      "import sys,uvicorn\nfrom local_operator.browser_bridge.daemon import create_app\np=int(sys.argv[1])\nuvicorn.run(create_app(p),host='127.0.0.1',port=p,log_level='warning')",
      String(daemonPort),
    ],
    { cwd: ROOT, env, stdio: ["ignore", "ignore", "pipe"], detached: true },
  );
  children.push(daemon.pid);
  for (let i = 0; i < 300; i++) {
    try {
      const h = await (await fetch(`http://127.0.0.1:${daemonPort}/health`)).json();
      if (h.status === "ok") break;
    } catch {}
    await sleep(100);
  }
  step("daemon up", { port: daemonPort, pid: daemon.pid });

  // --- CLI with NO extension at all
  const none = cli("pair");
  step("pair: no extension connected", { rc: none.rc, out: none.out.trim() });

  // --- isolated dist
  const dist = join(ROOT, "dist");
  await cp(join(REPO, "extension/dist"), dist, { recursive: true });
  const offenders = [];
  const walk = async (dir) => {
    for (const item of await readdir(dir, { withFileTypes: true })) {
      const full = join(dir, item.name);
      if (item.isDirectory()) await walk(full);
      else if (item.name.endsWith(".js")) {
        const src = await readFile(full, "utf8");
        if (src.includes("4099")) {
          await writeFile(full, src.split("4099").join(String(daemonPort)));
          if ((await readFile(full, "utf8")).includes("4099")) offenders.push(full);
        }
      }
    }
  };
  await walk(dist);
  if (offenders.length) throw new Error(`still dials 4099: ${offenders}`);

  // --- chrome
  const profile = join(ROOT, "chrome-profile");
  const chrome = spawn(
    CHROME,
    [
      "--headless=new",
      `--user-data-dir=${profile}`,
      "--remote-debugging-port=0",
      "--use-mock-keychain",
      "--password-store=basic",
      "--no-first-run",
      "--no-default-browser-check",
      "about:blank",
    ],
    { stdio: "ignore", detached: true },
  );
  children.push(chrome.pid);
  const portFile = join(profile, "DevToolsActivePort");
  for (let i = 0; i < 300 && !(existsSync(portFile) && statSync(portFile).size > 0); i++) await sleep(100);
  const cdpPort = readFileSync(portFile, "utf8").split("\n")[0].trim();
  const version = await (await fetch(`http://127.0.0.1:${cdpPort}/json/version`)).json();
  const browser = await connect(version.webSocketDebuggerUrl);
  report.chrome = version.Browser;
  const loaded = await browser.send("Extensions.loadUnpacked", { path: dist });
  const id = loaded.id;
  report.extensionId = id;
  step("extension loaded", { id, chrome: version.Browser });

  // Wait for the worker to dial and get a code minted at its handshake.
  for (let i = 0; i < 200 && !pending()[id]; i++) await sleep(100);
  step("extension connected unpaired", { pending: Object.keys(pending()) });
  // Loading unpacked fires onInstalled(reason=install) -> the pairing tab opens.
  const installTab = await findPairTab(cdpPort, id);
  step("install opened pairing tab", { opened: Boolean(installTab) });
  if (installTab) await closeTarget(browser, installTab.id);

  // --- COLD STATE: code expired, no tab, no popup.
  expireAll();
  await sleep(300);
  const coldStatus = cli("status");
  step("cold state status", { out: coldStatus.out.trim().split("\n").filter((l) => /extension|paired|pair/.test(l)) });

  const first = cli("pair");
  const code1 = (first.out.match(/\b(\d{6})\b/) || [])[1];
  step("pair from cold state", { rc: first.rc, ms: first.ms, out: first.out.trim(), code: code1 });
  const again = cli("pair");
  const code1b = (again.out.match(/\b(\d{6})\b/) || [])[1];
  step("pair run twice: same code", { same: code1 === code1b, code1, code1b });

  // The show_pairing nudge opened the tab?
  let tab = null;
  for (let i = 0; i < 50 && !tab; i++) {
    tab = await findPairTab(cdpPort, id);
    if (!tab) await sleep(100);
  }
  step("pairing tab opened by lop browser pair", { opened: Boolean(tab), url: tab?.url });
  if (!tab) throw new Error("no pairing tab");
  const page = await attach(browser, tab.id);
  await page.send("Emulation.setDeviceMetricsOverride", { width: 1000, height: 760, deviceScaleFactor: 2, mobile: false });
  await page.send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] });
  await sleep(700);
  await shoot(page, "tab-opened-empty");
  await page.eval(`(() => { const i = document.getElementById("pair-code"); i.focus(); return 1; })()`);
  await page.send("Input.insertText", { text: "4821" });
  await sleep(200);
  await shoot(page, "tab-typing");
  await page.eval(`(() => { const i = document.getElementById("pair-code"); i.select(); return 1; })()`);

  // Wrong code x4 (under the cap), then a 5th wrong -> rotation.
  const wrong = code1 === "000000" ? "111111" : "000000";
  for (let n = 1; n <= 5; n++) {
    await typeCode(page, wrong);
    await sleep(600);
    const err = await page.eval(`document.getElementById("pair-error").textContent`);
    const rotated = pending()[id]?.code !== code1;
    step(`wrong code #${n}`, { error: err, attempts: pending()[id]?.attempts, rotated });
    if (n === 1) await shoot(page, "tab-wrong-code");
    if (n === 5) await shoot(page, "tab-locked");
  }
  // The old code is now dead.
  await typeCode(page, code1);
  await sleep(600);
  step("old code after lockout", {
    error: await page.eval(`document.getElementById("pair-error").textContent`),
  });

  // Expired code: re-run pair (new code), expire it, type it.
  const fresh = cli("pair");
  const code2 = (fresh.out.match(/\b(\d{6})\b/) || [])[1];
  step("pair after lockout", { code: code2, differs_from_dead: code2 !== code1 });
  expireAll();
  await typeCode(page, code2);
  await sleep(600);
  step("expired code", { error: await page.eval(`document.getElementById("pair-error").textContent`) });
  await shoot(page, "tab-expired");

  // Focus loss: switch to another tab and back; the field and its digits stay.
  const third = cli("pair");
  const code3 = (third.out.match(/\b(\d{6})\b/) || [])[1];
  await page.eval(`(() => { const i = document.getElementById("pair-code"); i.focus(); i.select(); return 1; })()`);
  await page.send("Input.insertText", { text: code3.slice(0, 3) });
  const other = await browser.send("Target.createTarget", { url: "about:blank" });
  await browser.send("Target.activateTarget", { targetId: other.targetId });
  await sleep(500);
  await browser.send("Target.activateTarget", { targetId: tab.id });
  await sleep(500);
  const kept = await page.eval(`document.getElementById("pair-code").value`);
  step("focus loss keeps typed digits", { typedBefore: code3.slice(0, 3), valueAfter: kept });
  await page.eval(`(() => { const i = document.getElementById("pair-code"); i.focus(); i.setSelectionRange(3,3); return 1; })()`);
  await page.send("Input.insertText", { text: code3.slice(3) });
  const userStart = Date.now();
  await pressEnter(page);
  for (let i = 0; i < 50; i++) {
    const shown = await page.eval(`!document.getElementById("paired-state").classList.contains("hidden")`);
    if (shown) break;
    await sleep(100);
  }
  step("correct code -> Paired", {
    ms: Date.now() - userStart,
    title: await page.eval(`document.getElementById("paired-title").textContent`),
  });
  await shoot(page, "tab-paired");
  // Wait for the worker to reconnect with its new token.
  let status = "";
  for (let i = 0; i < 60; i++) {
    status = cli("status").out;
    if (/paired:\s+yes/.test(status)) break;
    await sleep(500);
  }
  step("status after pairing", { lines: status.trim().split("\n").filter((l) => /connected|paired/.test(l)) });
  const paired = cli("pair");
  step("pair when already paired", { rc: paired.rc, out: paired.out.trim() });

  // --- States that need a stub (named as such): loading + unreachable.
  for (const [name, stub] of [
    ["tab-loading", `window.fetch = () => new Promise(() => {});`],
    ["tab-unreachable", `window.fetch = () => Promise.reject(new TypeError("Failed to fetch"));`],
    ["tab-form-nocode", null],
  ]) {
    const { targetId } = await browser.send("Target.createTarget", { url: "about:blank" });
    const p = await attach(browser, targetId);
    await p.send("Emulation.setDeviceMetricsOverride", { width: 1000, height: 760, deviceScaleFactor: 2, mobile: false });
    await p.send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] });
    if (stub) await p.send("Page.addScriptToEvaluateOnNewDocument", { source: stub });
    else
      await p.send("Page.addScriptToEvaluateOnNewDocument", {
        source: `window.fetch = () => Promise.resolve({ ok: true, json: async () => ({ paired: false, authorized_extension_ids: [] }) });`,
      });
    await p.send("Page.navigate", { url: `chrome-extension://${id}/pair/pair.html` });
    await sleep(900);
    await shoot(p, name);
  }
  // Dark mode variant of the code-entry state.
  {
    const { targetId } = await browser.send("Target.createTarget", { url: "about:blank" });
    const p = await attach(browser, targetId);
    await p.send("Emulation.setDeviceMetricsOverride", { width: 1000, height: 760, deviceScaleFactor: 2, mobile: false });
    await p.send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "dark" }] });
    await p.send("Page.addScriptToEvaluateOnNewDocument", {
      source: `window.fetch = () => Promise.resolve({ ok: true, json: async () => ({ paired: false, authorized_extension_ids: [] }) });`,
    });
    await p.send("Page.navigate", { url: `chrome-extension://${id}/pair/pair.html` });
    await sleep(900);
    await shoot(p, "tab-form-dark");
    // Narrow viewport (a phone-width window / split screen).
    await p.send("Emulation.setDeviceMetricsOverride", { width: 380, height: 760, deviceScaleFactor: 2, mobile: false });
    await sleep(300);
    await shoot(p, "tab-form-narrow-dark");
  }
  // --- The popup's pairing card, as a page at the popup's 300px (unpaired).
  {
    const { targetId } = await browser.send("Target.createTarget", { url: "about:blank" });
    const p = await attach(browser, targetId);
    await p.send("Emulation.setDeviceMetricsOverride", { width: 300, height: 600, deviceScaleFactor: 2, mobile: false });
    await p.send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] });
    await p.send("Page.addScriptToEvaluateOnNewDocument", {
      source: `window.fetch = () => Promise.resolve({ ok: true, json: async () => ({ paired: false, extension_connected: true, protocol_version: 1, authorized_extension_ids: [] }) }); try { localStorage.clear(); } catch {}`,
    });
    // First paint: health never answers, so the card is the #pending placeholder
    // at the pinned height. Then the settled form. Equal heights = no resize.
    await p.send("Page.addScriptToEvaluateOnNewDocument", { source: "" });
    const first = await (async () => {
      const { targetId: t2 } = await browser.send("Target.createTarget", { url: "about:blank" });
      const q = await attach(browser, t2);
      await q.send("Emulation.setDeviceMetricsOverride", { width: 300, height: 600, deviceScaleFactor: 2, mobile: false });
      await q.send("Page.addScriptToEvaluateOnNewDocument", { source: "window.fetch = () => new Promise(() => {}); try { localStorage.clear(); } catch {}" });
      await q.send("Page.navigate", { url: `chrome-extension://${id}/popup/popup.html` });
      await sleep(800);
      await shoot(q, "popup-first-paint");
      return q.eval(`document.getElementById("card").getBoundingClientRect().height`);
    })();
    await p.send("Page.navigate", { url: `chrome-extension://${id}/popup/popup.html` });
    await sleep(1000);
    await shoot(p, "popup-pairing");
    const settled = await p.eval(`document.getElementById("card").getBoundingClientRect().height`);
    step("popup first paint vs settled pairing card", { first, settled, resize_px: +(settled - first).toFixed(2) });
    await p.eval(`(() => { const e = document.getElementById("pair-error"); e.textContent = ${JSON.stringify("Too many wrong tries. Ask Local Operator to \"pair my browser\" for a new code.")}; e.classList.remove("hidden"); return 1; })()`);
    await sleep(200);
    await shoot(p, "popup-pairing-error");
  }
  await writeFile(join(OUT, "report.json"), JSON.stringify(report, null, 2));
}

async function findPairTab(cdpPort, id) {
  const list = await (await fetch(`http://127.0.0.1:${cdpPort}/json/list`)).json();
  return list.find((t) => t.type === "page" && t.url.startsWith(`chrome-extension://${id}/pair/pair.html`)) ?? null;
}
async function closeTarget(browser, targetId) {
  await browser.send("Target.closeTarget", { targetId });
}
async function attach(browser, targetId) {
  const { sessionId } = await browser.send("Target.attachToTarget", { targetId, flatten: true });
  const page = browser.session(sessionId, targetId);
  await page.send("Page.enable");
  await page.send("Runtime.enable");
  return page;
}
async function typeCode(page, code) {
  await page.eval(`(() => { const i = document.getElementById("pair-code"); i.focus(); i.select(); return 1; })()`);
  await page.send("Input.insertText", { text: code });
  await pressEnter(page);
}
async function pressEnter(page) {
  for (const type of ["keyDown", "keyUp"])
    await page.send("Input.dispatchKeyEvent", {
      type,
      key: "Enter",
      code: "Enter",
      windowsVirtualKeyCode: 13,
      nativeVirtualKeyCode: 13,
      ...(type === "keyDown" ? { text: "\r" } : {}),
    });
}
async function shoot(page, name) {
  const geometry = await page.eval(`JSON.stringify((() => {
    const r = (el) => { if (!el) return null; const b = el.getBoundingClientRect(); return { x: +b.x.toFixed(1), y: +b.y.toFixed(1), w: +b.width.toFixed(1), h: +b.height.toFixed(1) }; };
    const visible = [...document.querySelectorAll("section.state")].filter((s) => !s.classList.contains("hidden")).map((s) => s.id);
    const err = document.getElementById("pair-error");
    return {
      viewport: [innerWidth, innerHeight],
      docHeight: document.documentElement.scrollHeight,
      scrollbar: document.documentElement.scrollHeight > innerHeight,
      visible,
      card: r(document.getElementById("card")),
      input: r(document.getElementById("pair-code")),
      button: r(document.querySelector("#pair-form button[type=submit]")),
      error: err ? { ...r(err), text: err.textContent, hidden: err.classList.contains("hidden") } : null,
      focused: document.activeElement?.id || document.activeElement?.tagName,
      pairingSection: (() => { const s = document.getElementById("pairing"); return s ? +s.getBoundingClientRect().height.toFixed(2) : null; })(),
      pendingMin: (() => { const s = document.getElementById("pending"); return s ? getComputedStyle(s).minHeight : null; })(),
    };
  })())`);
  const shot = await page.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
  await writeFile(join(OUT, `${name}.png`), Buffer.from(shot.data, "base64"));
  report.frames[name] = JSON.parse(geometry);
  log("frame", name, geometry);
}

function connect(wsUrl) {
  return new Promise((resolveConn, rejectConn) => {
    const ws = new WebSocket(wsUrl);
    let next = 1;
    const waiting = new Map();
    ws.addEventListener("message", (event) => {
      const msg = JSON.parse(String(event.data));
      const entry = waiting.get(msg.id);
      if (!entry) return;
      waiting.delete(msg.id);
      if (msg.error) entry.reject(new Error(JSON.stringify(msg.error)));
      else entry.resolve(msg.result);
    });
    ws.addEventListener("error", () => rejectConn(new Error("CDP socket failed")));
    const rawSend = (method, params, sessionId) =>
      new Promise((res, rej) => {
        const id = next++;
        waiting.set(id, { resolve: res, reject: rej });
        ws.send(JSON.stringify({ id, method, params: params ?? {}, ...(sessionId ? { sessionId } : {}) }));
      });
    ws.addEventListener("open", () =>
      resolveConn({
        send: (method, params) => rawSend(method, params),
        session: (sessionId, targetId) => ({
          targetId,
          send: (method, params) => rawSend(method, params, sessionId),
          eval: async (expression) => {
            const r = await rawSend("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true }, sessionId);
            if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
            return r.result.value;
          },
        }),
      }),
    );
  });
}

try {
  await main();
} catch (error) {
  console.error("E2E FAILED:", error);
  await writeFile(join(OUT, "report.json"), JSON.stringify({ ...report, error: String(error) }, null, 2));
  process.exitCode = 1;
} finally {
  reap();
  await sleep(1500);
  spawnSync("pkill", ["-f", ROOT], { stdio: "ignore" });
  await sleep(500);
  const left = (spawnSync("pgrep", ["-f", ROOT], { encoding: "utf8" }).stdout ?? "").trim();
  console.log("survivors:", left ? left.split("\n").length : 0);
  if (!left) await rm(ROOT, { recursive: true, force: true });
}

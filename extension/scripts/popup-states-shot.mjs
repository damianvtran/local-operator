/* Capture the popup's cards out of a REAL Chrome, and measure them.
 *
 * Why a script in the tree rather than a throwaway: the `#pending` first-paint
 * pin (popup.css, first-paint.js, PIN_BY_STATE in popup.ts) is a set of MEASURED
 * pixel values, and a stale pin is a visible reflow on every popup open. The
 * only way to keep them honest is to re-measure the card at 300px width in the
 * browser that actually lays it out — an eyeballed number is how the 167.8px
 * reflow of design round 3 survived review. So the measurement and the frames
 * come from one run, against `extension/dist`.
 *
 * Every Chrome flag here is AGENTS.md §"Capturing a browser surface: headless
 * Chrome, never a window": headless so it cannot steal the operator's focus, a
 * unique mktemp profile so it can never touch the operator's own Chrome or its
 * paired store extension, and a port Chrome picks (read back from
 * DevToolsActivePort) because a fixed port on this many-session machine drives
 * ANOTHER agent's browser. The extension is loaded over CDP's
 * Extensions.loadUnpacked, never --load-extension, which branded Chrome 137+
 * silently ignores.
 *
 * Usage: node scripts/popup-states-shot.mjs <dist-dir> <out-dir>
 */
import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

// Node 22+ ships a global WebSocket, so the CDP client below needs no
// dependency — deliberately, since this repo keeps the extension devDeps to
// esbuild + typescript and a capture script is not a reason to grow them.

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const [, , distArg, outArg] = process.argv;
const DIST = resolve(distArg ?? "dist");
const OUT = resolve(outArg ?? "/tmp/lo-popup-shots");

/** The states this popup can paint that the change touches, each expressed as
 * the /health payload that produces it. The popup reads /health on every
 * render, so overriding `fetch` on the page is the whole fixture — no daemon,
 * and therefore no chance of touching the operator's real one. */
const STATES = {
  // The paired, working browser: the card a healthy user sees.
  connected: { paired: true, extension_connected: true, protocol_version: 1 },
  // The daemon is unreachable. `fetch` rejects, which is what a dead daemon
  // looks like from the popup, and renders the Retry card.
  disconnected: null,
  // The subject of this change: paired, socket attached, worker mute.
  unresponsive: {
    paired: true,
    extension_connected: false,
    extension_unresponsive: true,
    link_attached: true,
    protocol_version: 1,
  },
};

async function main() {
  await mkdir(OUT, { recursive: true });
  const profile = `/tmp/lo-popup-shot.${process.pid}.${Date.now()}`;
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
  // Own the process group so a harness killed mid-run leaves addressable
  // survivors rather than a browser reparented to PPID 1 holding the profile.
  const pgid = chrome.pid;
  const teardown = () => {
    try {
      process.kill(-pgid, "SIGTERM");
    } catch {
      /* already gone */
    }
  };
  process.on("exit", teardown);

  try {
    const portFile = join(profile, "DevToolsActivePort");
    // Chrome writes this file ASYNCHRONOUSLY; an immediate read gets nothing
    // and the connect fails intermittently, which reads as a flaky harness.
    for (let i = 0; i < 200 && !(existsSync(portFile) && statSync(portFile).size > 0); i++) {
      await sleep(100);
    }
    const port = (await readFile(portFile, "utf8")).split("\n")[0].trim();
    const version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
    const browser = await connect(version.webSocketDebuggerUrl);

    const loaded = await browser.send("Extensions.loadUnpacked", { path: DIST });
    const id = loaded.id;
    const report = { chrome: version.Browser, extensionId: id, states: {} };

    for (const [name, health] of Object.entries(STATES)) {
      const page = await openPage(browser, `chrome-extension://${id}/popup/popup.html`, health);
      // 300x600 is the popup's real geometry, and it MUST come from the CDP
      // override: --window-size clamps to a 500px floor on Chrome 152, so a
      // frame sized by the flag is not evidence of anything.
      await page.send("Emulation.setDeviceMetricsOverride", {
        width: 300,
        height: 600,
        deviceScaleFactor: 2,
        mobile: false,
      });
      await sleep(600);
      const measured = await page.eval(`(() => {
        const card = document.getElementById("card");
        const shown = [...document.querySelectorAll("section.state")].find(
          (s) => !s.classList.contains("hidden"),
        );
        const pending = document.getElementById("pending");
        return JSON.stringify({
          shown: shown ? shown.id : null,
          cardHeight: card ? card.getBoundingClientRect().height : null,
          pendingPin: pending ? getComputedStyle(pending).minHeight : null,
          pinHint: localStorage.getItem("lop:pin-hint"),
          reloadOffered: !!document.getElementById("reload-extension") &&
            !!shown && shown.contains(document.getElementById("reload-extension")),
        });
      })()`);
      const shot = await page.send("Page.captureScreenshot", { format: "png" });
      const file = join(OUT, `${name}.png`);
      await writeFile(file, Buffer.from(shot.data, "base64"));
      report.states[name] = { ...JSON.parse(measured), file };
      await browser.send("Target.closeTarget", { targetId: page.targetId });
    }

    console.log(JSON.stringify(report, null, 2));
  } finally {
    teardown();
    await sleep(2000);
    console.error(`profile=${profile}`);
  }
}

/** Open the popup as a page with /health stubbed BEFORE any script runs, so the
 * first render already sees the state under test. */
async function openPage(browser, url, health) {
  const { targetId } = await browser.send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await browser.send("Target.attachToTarget", { targetId, flatten: true });
  const page = browser.session(sessionId, targetId);
  await page.send("Page.enable");
  await page.send("Runtime.enable");
  const stub =
    health === null
      ? `window.fetch = () => Promise.reject(new TypeError("Failed to fetch"));`
      : `window.fetch = () => Promise.resolve({ ok: true, json: async () => (${JSON.stringify(health)}) });`;
  // addScriptToEvaluateOnNewDocument runs before the page's own scripts, which
  // is what makes this the state of the FIRST paint rather than a later render.
  await page.send("Page.addScriptToEvaluateOnNewDocument", {
    source: `${stub}\ntry { localStorage.clear(); } catch {}`,
  });
  await page.send("Page.navigate", { url });
  await sleep(800);
  return page;
}

function connect(wsUrl) {
  return new Promise((resolveConn, rejectConn) => {
    const ws = new WebSocket(wsUrl);
    let next = 1;
    const pending = new Map();
    ws.addEventListener("message", (event) => {
      const msg = JSON.parse(String(event.data));
      const entry = pending.get(msg.id);
      if (!entry) return;
      pending.delete(msg.id);
      if (msg.error) entry.reject(new Error(JSON.stringify(msg.error)));
      else entry.resolve(msg.result);
    });
    ws.addEventListener("error", () => rejectConn(new Error("CDP socket failed")));
    const rawSend = (method, params, sessionId) =>
      new Promise((res, rej) => {
        const id = next++;
        pending.set(id, { resolve: res, reject: rej });
        ws.send(JSON.stringify({ id, method, params: params ?? {}, ...(sessionId ? { sessionId } : {}) }));
      });
    ws.addEventListener("open", () =>
      resolveConn({
        send: (method, params) => rawSend(method, params),
        session: (sessionId, targetId) => ({
          targetId,
          send: (method, params) => rawSend(method, params, sessionId),
          eval: async (expression) => {
            const r = await rawSend("Runtime.evaluate", { expression, returnByValue: true }, sessionId);
            if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
            return r.result.value;
          },
        }),
      }),
    );
  });
}

await main();

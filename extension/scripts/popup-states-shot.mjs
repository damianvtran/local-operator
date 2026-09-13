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
import { spawn, spawnSync } from "node:child_process";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { existsSync, rmSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

// Node 22+ ships a global WebSocket, so the CDP client below needs no
// dependency — deliberately, since this repo keeps the extension devDeps to
// esbuild + typescript and a capture script is not a reason to grow them.

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const [, , distArg, outArg] = process.argv;
const DIST = resolve(distArg ?? "dist");
const OUT = resolve(outArg ?? "/tmp/lo-popup-shots");

/* THE HARNESS MUST NOT DIAL THE OPERATOR'S DAEMON.
 *
 * The worker cold-dials `DEFAULT_PORT` (4099) with no token and no pairing
 * gate, so a load-unpacked build on a throwaway profile reaches the operator's
 * REAL daemon on loopback. This is not theoretical: review runs of this very
 * harness put +9 accepted `/extension` connections on the operator's daemon
 * against an idle baseline of 0, and one run evicted their live link before it
 * self-healed. The blast radius happened to be bounded only because the daemon
 * closes an unknown `extension_id` with 4004 before its later-connection-wins
 * eviction — a harness that ever ran with a PAIRED id would evict for real.
 *
 * esbuild INLINES the constant into every bundle, so seeding storage after the
 * extension loads is too late: the cold-start dial has already gone out. The
 * port is therefore rewritten in the BUILT artifacts before Chrome ever sees
 * them, and this script refuses to run against a dist that still carries 4099.
 * 4499 has no listener, so a dial fails closed instead of finding a stranger.
 */
const REAL_DAEMON_PORT = "4099";
const HARNESS_PORT = "4499";
const PATCHED_FILES = ["worker.js", "popup/popup.js", "options/options.js"];

/** Rewrite the daemon port inside the built bundles, then verify no executable
 * artifact still names the real one. Refuses rather than repairs on a surprise:
 * a harness that "fixes" an unexpected tree is how an unnoticed dial happens. */
async function isolateFromOperatorDaemon(dist) {
  for (const relative of PATCHED_FILES) {
    const file = join(dist, relative);
    if (!existsSync(file)) throw new Error(`refusing to load ${dist}: ${relative} is missing`);
    const source = await readFile(file, "utf8");
    await writeFile(file, source.split(REAL_DAEMON_PORT).join(HARNESS_PORT));
  }
  // Verify against the tree that will actually be loaded, not against the
  // writes above: a bundle that gained a second copy of the constant, or a new
  // entry point nobody added to PATCHED_FILES, must fail the run rather than
  // pass it silently. `.js.map` is excluded because it is never executed.
  const offenders = [];
  const walk = async (dir) => {
    for (const item of await readdir(dir, { withFileTypes: true })) {
      const full = join(dir, item.name);
      if (item.isDirectory()) await walk(full);
      else if (item.name.endsWith(".js") && (await readFile(full, "utf8")).includes(REAL_DAEMON_PORT)) {
        offenders.push(full);
      }
    }
  };
  await walk(dist);
  if (offenders.length) {
    throw new Error(
      `refusing to load a build that still dials ${REAL_DAEMON_PORT}: ${offenders.join(", ")}`,
    );
  }
  // Belt and braces: if something IS listening on the harness port, this
  // profile's extension would pair with it. Never proceed into a stranger.
  try {
    await fetch(`http://127.0.0.1:${HARNESS_PORT}/health`, {
      signal: AbortSignal.timeout(500),
    });
    throw new Error(`refusing to run: something is listening on ${HARNESS_PORT}`);
  } catch (error) {
    if (String(error?.message ?? "").startsWith("refusing")) throw error;
    // Connection refused is the expected and required outcome.
  }
}

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
  // The INCIDENT SHAPE, and the reason the inline banner exists: the worker is
  // wedged while a decision is still queued. renderOnce() paints the consent
  // card and returns before the unresponsive branch, so this is the state where
  // #unresponsive — and its Reload button — is unreachable.
  "origin-wedged": {
    paired: true,
    extension_connected: false,
    extension_unresponsive: true,
    link_attached: true,
    pending_origin: "https://contested.example",
    protocol_version: 1,
  },
  // The same pending decision against a HEALTHY worker: the banner must be
  // absent here, or the remedy is being offered to a worker that does not need
  // it. The control for the state above.
  "origin-healthy": {
    paired: true,
    extension_connected: true,
    pending_origin: "https://contested.example",
    protocol_version: 1,
  },
  // Paired, and deliberately receiving no commands because ANOTHER authorised
  // install holds the wheel. The secondary install's card, and the reason this
  // repo can have two builds installed at once; the shape is the store build's
  // when a locally loaded build is driving.
  //
  // The two fields below are the HEALTHY values as the real wire carries them
  // (silence measured at 1.6 ms against a 20 s ping interval), so this frame is
  // the reference the two qualified frames underneath must NOT be confused with:
  // it is the frame the earlier design and UX rounds approved byte-for-byte, and
  // `standby-promise` is rewritten per render, so it is the one most at risk
  // from U9.
  standby: {
    paired: true,
    extension_connected: true,
    protocol_version: 1,
    driver_extension_id: "omibaecbjdhgbbcedbnnnmjpmopfheof",
    driver_label: "Chrome 0.1.10",
    authorized_extension_ids: [
      "omibaecbjdhgbbcedbnnnmjpmopfheof",
      "jbadjeaodkoboanppmpjiifpconegdcj",
    ],
    standby_extension_ids: ["jbadjeaodkoboanppmpjiifpconegdcj"],
    link_attached: true,
    link_silent_s: 0.0016,
    takeover_within_s: null,
  },
  // U9, the EARLY window: the wheel is attached and merely silent. 41.2 s is
  // UX's own measurement at t+35.3 s of their run, where `takeover_within_s` is
  // still null and the card used to make the unqualified promise.
  "standby-silent": {
    paired: true,
    extension_connected: true,
    protocol_version: 1,
    driver_extension_id: "omibaecbjdhgbbcedbnnnmjpmopfheof",
    driver_label: "Chrome 0.1.10",
    authorized_extension_ids: [
      "omibaecbjdhgbbcedbnnnmjpmopfheof",
      "jbadjeaodkoboanppmpjiifpconegdcj",
    ],
    standby_extension_ids: ["jbadjeaodkoboanppmpjiifpconegdcj"],
    link_attached: true,
    link_silent_s: 41.2,
    takeover_within_s: null,
  },
  // U9, COMMITTED: silence past the deadline, so the daemon is counting down to
  // the severance (UX's t+44.3 s capture — the frame that was byte-identical to
  // `standby` before this change).
  "standby-countdown": {
    paired: true,
    extension_connected: false,
    extension_unresponsive: true,
    protocol_version: 1,
    driver_extension_id: "omibaecbjdhgbbcedbnnnmjpmopfheof",
    driver_label: "Chrome 0.1.10",
    authorized_extension_ids: [
      "omibaecbjdhgbbcedbnnnmjpmopfheof",
      "jbadjeaodkoboanppmpjiifpconegdcj",
    ],
    standby_extension_ids: ["jbadjeaodkoboanppmpjiifpconegdcj"],
    link_attached: true,
    link_silent_s: 50.3,
    takeover_within_s: 19.7,
  },
  // U10: the wheel was severed and NOBODY took it — QA's §6.2 payload, verbatim.
  // Both installs are authorised, neither is named as driving, and the wedge
  // latch is up, so no surface can say WHICH worker went mute.
  severed: {
    paired: true,
    extension_connected: false,
    extension_unresponsive: true,
    link_attached: false,
    protocol_version: 1,
    driver_extension_id: "",
    takeover_within_s: null,
    standby_extension_ids: [],
    authorized_extension_ids: [
      "jbadjeaodkoboanppmpjiifpconegdcj",
      "omibaecbjdhgbbcedbnnnmjpmopfheof",
    ],
  },
};

/** Session-storage fixtures, keyed like STATES.
 *
 * `connState` is what the worker writes from the daemon's role statement, and it
 * lives in `chrome.storage.session` — so unlike /health it cannot be stubbed
 * before the page loads. It is set and the page reloaded, which is also what a
 * real user's second open looks like for a durable role.
 */
const SESSION_FIXTURES = {
  standby: { connState: "standby" },
  "standby-silent": { connState: "standby" },
  "standby-countdown": { connState: "standby" },
};

/** The id the fixtures above were WRITTEN with, as the popup's OWN identity.
 *
 * It is a placeholder, not a fact about this build: `manifest.dev.json` pins a
 * dev build's id to a keypair, and the keypair's id is what
 * `Extensions.loadUnpacked` reports. A fixture that names a different id than
 * the popup is running under makes EVERY self-scoped state unreachable —
 * `selfAuthorized` is false, so the popup falls through to the pairing form and
 * captures a PAIRING CODE CARD for `standby`, `standby-silent`,
 * `standby-countdown` and `severed` alike, all byte-identical. That is not a
 * hypothetical: it is what this script did before the substitution below, and
 * the frames looked plausible enough to file. So the loaded id is substituted
 * into every payload, and the report repeats it, rather than the fixtures
 * assuming a literal. */
const SELF_ID_PLACEHOLDER = "jbadjeaodkoboanppmpjiifpconegdcj";
/** Set once the extension is loaded; used by openPage's stub. */
let SELF_ID = SELF_ID_PLACEHOLDER;

async function main() {
  await mkdir(OUT, { recursive: true });
  // BEFORE Chrome exists, let alone before the extension loads: the cold-start
  // dial goes out within milliseconds of the worker being instantiated.
  await isolateFromOperatorDaemon(DIST);
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
    SELF_ID = id;
    const report = { chrome: version.Browser, extensionId: id, states: {} };

    for (const [name, health] of Object.entries(STATES)) {
      const page = await openPage(
        browser,
        `chrome-extension://${id}/popup/popup.html`,
        health,
        SESSION_FIXTURES[name],
      );
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
        const wedge = document.getElementById("origin-wedge");
        const wedgeReload = document.getElementById("origin-wedge-reload");
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
          // The inline remedy on the consent card: visible means a real rect,
          // not merely a node in the document — the D3 finding was precisely a
          // control that existed while occupying 0px in a hidden section.
          wedgeBannerVisible: !!wedge && !wedge.classList.contains("hidden") &&
            wedge.getBoundingClientRect().height > 0,
          wedgeReloadVisible: !!wedgeReload && wedgeReload.getBoundingClientRect().height > 0,
          tone: getComputedStyle(document.getElementById("card")).getPropertyValue("--tone").trim(),
          // Clipping is what design D2 measured: content whose bottom edge sits
          // past the scroll container's visible height is off-screen.
          bodyScrollHeight: document.querySelector(".body")?.scrollHeight ?? null,
          bodyClientHeight: document.querySelector(".body")?.clientHeight ?? null,
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
    sweepProfile(profile);
    console.error(`profile=${profile}`);
  }
}

/** Kill whatever still holds this capture's profile, then remove the directory.
 *
 * SIGTERM to the process group is not enough on its own: AGENTS.md §6 measured
 * 0-5 helpers surviving it across 11 runs on Chrome 152 with no stable pattern by
 * timing, which is exactly why one run "looks" clean. So the sweep is
 * `pkill -f <this run's mktemp prefix>` followed by a pgrep ASSERTION, and the
 * directory is only removed once the count is 0 — scoped to that unique prefix,
 * so it can never touch another session's Chrome or the operator's.
 *
 * The assertion is the load-bearing half. Without it this harness leaked one
 * profile per run (measured: four from this session's four captures, among ten
 * others under /tmp from earlier ones), and a `rm -rf` under a live helper is the
 * kind of failure that reports success: the directory reappears as soon as that
 * helper writes again.
 */
function sweepProfile(profile) {
  spawnSync("pkill", ["-f", profile], { stdio: "ignore" });
  const survivors = spawnSync("pgrep", ["-f", profile], { encoding: "utf8" });
  const count = (survivors.stdout ?? "").trim().split("\n").filter(Boolean).length;
  if (count > 0) {
    console.error(`NOT removed, ${count} process(es) still hold it: ${profile}`);
    return false;
  }
  rmSync(profile, { recursive: true, force: true });
  return true;
}

/** Open the popup as a page with /health stubbed BEFORE any script runs, so the
 * first render already sees the state under test. */
async function openPage(browser, url, health, session) {
  const { targetId } = await browser.send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await browser.send("Target.attachToTarget", { targetId, flatten: true });
  const page = browser.session(sessionId, targetId);
  await page.send("Page.enable");
  await page.send("Runtime.enable");
  const stub =
    health === null
      ? `window.fetch = () => Promise.reject(new TypeError("Failed to fetch"));`
      : `window.fetch = () => Promise.resolve({ ok: true, json: async () => (${JSON.stringify(
          health,
        )
          .split(SELF_ID_PLACEHOLDER)
          .join(SELF_ID)}) });`;
  // addScriptToEvaluateOnNewDocument runs before the page's own scripts, which
  // is what makes this the state of the FIRST paint rather than a later render.
  await page.send("Page.addScriptToEvaluateOnNewDocument", {
    source: `${stub}\ntry { localStorage.clear(); } catch {}`,
  });
  await page.send("Page.navigate", { url });
  await sleep(800);
  if (session) {
    await page.eval(`chrome.storage.session.set(${JSON.stringify(session)}).then(() => "ok")`);
    await page.send("Page.reload");
    await sleep(800);
  }
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

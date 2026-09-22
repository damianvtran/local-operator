#!/usr/bin/env node
/* Real HTTP/WS daemon + shipped worker/storage/popup modules. Chrome is an
 * isolated API fixture, NOT an installed extension/browser. Run from extension:
 * node scripts/popup-performance-live.mjs [absolute-python-with-project-deps]
 * Every child and the JS module host lose inherited LOP_/CMUX_ environment; all
 * daemon identity/config state is disposable, including its synthetic tokens. */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve, join } from "node:path";
import { pathToFileURL } from "node:url";
import { build } from "esbuild";
import { installWorkerPerformance } from "../tests/fixtures/worker-performance.mjs";
import { installPopupDom } from "../tests/fixtures/popup-dom.mjs";

for (const key of Object.keys(process.env)) if (/^(CMUX_|LOP_)/.test(key)) delete process.env[key];
const root = resolve(import.meta.dirname, "../..");
const python = process.argv[2] || join(root, ".venv/bin/python");
const home = await mkdtemp(join(tmpdir(), "lop-popup-live-"));
process.env.HOME = home;
process.env.XDG_CONFIG_HOME = join(home, "config");
const id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", secondId = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const token = "synthetic-fixture-only-not-a-credential";
const source = `
import hashlib, os, socket
from pathlib import Path
import uvicorn
from local_operator.browser_bridge.daemon import add_identity, create_app
root = Path(os.environ['HOME']) / 'bridge-root'
for identity in ['${id}', '${secondId}']:
    add_identity(root, identity, hashlib.sha256(b'${token}').hexdigest(), label='Synthetic fixture', driver_id='${id}')
sock = socket.socket()
sock.bind(('127.0.0.1', 0))
sock.listen()
port = sock.getsockname()[1]
print(port, flush=True)
uvicorn.Server(uvicorn.Config(create_app(port, root), log_level='error')).run(sockets=[sock])
`;
const daemon = spawn(python, ["-u", "-c", source], { cwd: root, env: { ...process.env, PYTHONPATH: root }, stdio: ["ignore", "pipe", "inherit"] });
const sockets = [];
const nativeSocket = globalThis.WebSocket;
let sequence = 0;
async function bundle(text) {
  const outfile = join(home, `module-${sequence++}.mjs`);
  await build({ stdin: { contents: text, resolveDir: join(root, "extension") }, bundle: true, platform: "node", format: "esm", outfile });
  return import(pathToFileURL(outfile));
}
const bounded = (promise, label) => {
  let timer;
  return Promise.race([promise, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`Timed out: ${label}`)), 15000); })]).finally(() => clearTimeout(timer));
};
try {
  const port = Number(String((await bounded(once(daemon.stdout, "data"), "daemon port"))[0]).trim());
  assert.ok(port > 0);
  const base = `http://127.0.0.1:${port}`;
  // Readiness is a real response barrier, not a performance assertion.
  for (let attempt = 0; ; attempt++) {
    try { if ((await fetch(`${base}/health`)).ok) break; } catch {}
    if (attempt > 100) throw new Error("daemon did not become ready");
    await new Promise(resolve => setTimeout(resolve, 20));
  }
  let connected;
  const ready = new Promise(resolve => { connected = resolve; });
  const fixture = installWorkerPerformance({ port, id, token, onState: (_, changes) => { if (changes.connState === "connected") connected(); } });
  globalThis.WebSocket = class extends nativeSocket {
    constructor(url) { super(url, { headers: { Origin: `chrome-extension://${id}` } }); sockets.push(this); }
  };
  const worker = await bundle('import "./src/worker.ts"; export * from "./src/origins.ts";');
  await bounded(ready, "worker hello_ack");
  const health = await (await fetch(`${base}/health`)).json();
  assert.equal(health.extension_connected, true);
  assert.equal(health.driver_extension_id, id);
  console.log(JSON.stringify({ step: "real worker hello/ack + HTTP health", connected: true, driver: "synthetic first", startupReads: fixture.metrics.reads.length, startupWrites: fixture.metrics.writes.length, startupActionCalls: fixture.metrics.actions.length }));

  const state = JSON.parse(await readFile(join(home, "bridge-root/run/browser/bridge.json"), "utf8"));
  const headers = { "Content-Type": "application/json", "X-Bridge-Key": state.session_key };
  for (const [name, requestHeaders, body] of [["unauthorized", { "Content-Type": "application/json" }, {}], ["invalid-input", headers, {}]]) {
    const response = await fetch(`${base}/rpc`, { method: "POST", headers: requestHeaders, body: JSON.stringify(body) });
    const result = await response.json();
    assert.ok(response.status >= 400);
    console.log(JSON.stringify({ step: name, status: response.status, response: result }));
  }
  let painted;
  const firstPaint = new Promise(resolve => { painted = resolve; });
  let dom;
  dom = installPopupDom(await readFile(join(root, "extension/src/popup/popup.html"), "utf8"), () => { if (dom.visible().includes("connected")) painted(); });
  const beforeReads = fixture.metrics.reads.length;
  await bundle('import "./src/popup/popup.ts";');
  await bounded(firstPaint, "real-health popup");
  assert.deepEqual(dom.visible(), ["connected"]);
  console.log(JSON.stringify({ step: "shipped popup + real HTTP health", visible: dom.visible(), reads: fixture.metrics.reads.slice(beforeReads) }));

  const entry = await worker.raiseAccessRequest(new URL("https://example.test"), "synthetic-owner");
  const response = await bounded(new Promise(resolve => {
    for (const listener of fixture.listeners.message) listener({ event: "origin_decision", origin: entry.origin, entryId: entry.entryId, decision: "once" }, {}, resolve);
  }), "consent ACK");
  assert.equal(response.applied, true);
  assert.equal(fixture.session.accessQueue.length, 0);
  assert.equal(Object.values(fixture.session.onceGrants).length, 1);
  const wrongOwner = await worker.consumeOnceGrant(new URL("https://example.test"), "different-owner");
  const rightOwner = await worker.consumeOnceGrant(new URL("https://example.test"), "synthetic-owner");
  const repeated = await worker.consumeOnceGrant(new URL("https://example.test"), "synthetic-owner");
  assert.deepEqual([wrongOwner, rightOwner, repeated], [false, true, false]);
  console.log(JSON.stringify({ step: "worker runtime consent + durable grant", response, queueLength: fixture.session.accessQueue.length, wrongOwner, rightOwner, repeated }));

  const standby = new nativeSocket(`ws://127.0.0.1:${port}/extension`, { headers: { Origin: `chrome-extension://${secondId}` } });
  sockets.push(standby);
  const ack = new Promise(resolve => standby.addEventListener("message", event => resolve(JSON.parse(event.data)), { once: true }));
  await bounded(once(standby, "open"), "second socket");
  standby.send(JSON.stringify({ event: "hello", proto: 1, extension_version: "0.1.20", token, browser: "Synthetic fixture" }));
  const secondAck = await bounded(ack, "second identity ACK");
  assert.equal(secondAck.role, "standby");
  console.log(JSON.stringify({ step: "second real identity", paired: secondAck.paired, role: secondAck.role }));
} finally {
  for (const socket of sockets) { socket.onclose = null; socket.onerror = null; socket.close(); }
  daemon.kill("SIGTERM");
  await once(daemon, "exit");
  await rm(home, { recursive: true, force: true });
}

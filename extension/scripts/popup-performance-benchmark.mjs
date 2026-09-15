#!/usr/bin/env node
/* Controlled Node timings are supporting evidence, not Chrome startup timing.
 * Run from extension/, optionally pass a source-root containing popup/. */
import { build } from "esbuild";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { installPopupDom } from "../tests/fixtures/popup-dom.mjs";
import { installPopupPerformance } from "../tests/fixtures/popup-performance.mjs";
for (const key of Object.keys(process.env)) if (/^(CMUX_|LOP_)/.test(key)) delete process.env[key];
const source = resolve(process.argv[2] || "src");
const output = await build({ entryPoints: [resolve(source, "popup/popup.ts")], bundle: true, format: "esm", platform: "node", write: false });
const html = await readFile(resolve(source, "popup/popup.html"), "utf8");
for (const state of ["healthy", "slow", "unreachable", "standby", "consent"]) {
  for (const warm of [false, true]) {
    let finish;
    const painted = new Promise(resolve => { finish = resolve; });
    let host;
    const fixture = installPopupPerformance({ state, delay: 20, large: true });
    host = installPopupDom(html, () => {
      fixture.metrics.mutations++;
      const visible = host.visible();
      if (visible.length === 1 && visible[0] !== "pending") finish(visible[0]);
    });
    if (warm) localStorage.setItem("lop:pin-hint", "148px");
    await import(`data:text/javascript;base64,${Buffer.from(output.outputFiles[0].text + `\n// ${state}-${warm}`).toString("base64")}`);
    const timer = setTimeout(() => { throw new Error("Popup did not populate"); }, 15000);
    const visible = await painted; clearTimeout(timer);
    const m = fixture.metrics;
    console.log(JSON.stringify({ state, warm, visible, firstPopulatedMs: +(performance.now() - fixture.start).toFixed(1), reads: m.reads.length, bytes: m.reads.reduce((n, r) => n + (r.bytes || 0), 0), healthCalls: m.health.length, mutations: m.mutations, readKeys: m.reads.map(r => r.keys) }));
  }
}

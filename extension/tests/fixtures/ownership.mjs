import { build } from "esbuild";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

// Actual navigation/state/ownership modules, disposable Chrome/CDP transport.
// No browser process, live bridge, install, config, or operator tab is touched.
export async function fixture() {
  const dir = await mkdtemp(join(tmpdir(), "lop-ownership-"));
  let store = {}; let next = 100;
  const tabs = new Map(); const removed = [];
  const faults = { navigation: false, remove: false, attach: false, capture: false };
  globalThis.ownershipFaults = faults;
  globalThis.chrome = {
    storage: { session: {
      get: async () => structuredClone(store),
      set: async value => { Object.assign(store, structuredClone(value)); },
    } },
    tabs: {
      create: async options => { const tab = { id: next++, ...options }; tabs.set(tab.id, tab); return tab; },
      get: async id => { if (!tabs.has(id)) throw Error(`No tab with id: ${id}`); return tabs.get(id); },
      update: async (id, options) => { if (faults.navigation) throw Error("net::ERR_CONNECTION_REFUSED"); Object.assign(tabs.get(id), options); return tabs.get(id); },
      remove: async id => { if (faults.remove) throw Error("policy denied removal"); removed.push(id); tabs.delete(id); },
    },
  };
  const entry = join(dir, "entry.mjs");
  await writeFile(entry, `export * from ${JSON.stringify(resolve("src/commands/nav.ts"))}; export * from ${JSON.stringify(resolve("src/ownership.ts"))};`);
  const mocks = {
    cdp: `import {getSurfaces,removeSurface} from ${JSON.stringify(resolve("src/state.ts"))};
      export class BridgeCommandError extends Error {constructor(code,message,data={}){super(message);this.code=code;this.data=data}}
      export const isStalled=()=>false;
      export const attach=async()=>{if(globalThis.ownershipFaults.attach)throw Error('attach failed')},detach=async()=>{},cdp=async()=>({result:{value:JSON.stringify({url:'https://example.test/',title:'fixture'})}}),pruneSurface=async(t)=>removeSurface(t),requireSurface=async(t)=>{const s=(await getSurfaces())[t];if(!s)throw new BridgeCommandError('tab_closed','gone');await chrome.tabs.get(s.tabId);return s};`,
    "log-capture": `export const dropLogCapture=()=>{},startLogCapture=async()=>{if(globalThis.ownershipFaults.capture)throw Error('capture failed')};`,
    origins: `export const safeHttpUrl=v=>new URL(v),ensureTopLevelAccess=async()=>({allowed:true}),askOrigin=async()=>true,withOriginGate=async(t,r,fn)=>fn();`,
    // The real helper and its constants are pass-throughs in this harness: the
    // modules under test still import them by NAME, so the fixture has to
    // export them or esbuild fails the build ("No matching export in
    // fixture:settle"). Handing back the op unchanged keeps the harness's
    // fault injection (which rejects) exactly as it was.
    settle: `export const settle=async()=>{};export const CHROME_API_DEADLINE_MS=5000;export const deadline=(op)=>op;`,
    "tab-groups": `export const reconcileTabGroup=async()=>{};`,
  };
  const outfile = join(dir, "bundle.mjs");
  await build({entryPoints:[entry],bundle:true,platform:"node",format:"esm",outfile,plugins:[{name:"isolated-transport",setup(b){
    b.onResolve({filter:/^\.\.?\/(cdp|log-capture|origins|settle|tab-groups)$/},a=>({path:a.path.split('/').at(-1),namespace:"fixture"}));
    b.onLoad({filter:/.*/,namespace:"fixture"},a=>({contents:mocks[a.path],loader:"js",resolveDir:resolve(".")}));
  }}]});
  let loaded = await import(pathToFileURL(outfile));
  const owner = {owner_proof:"a".repeat(40),requester:"session:synthetic",owner_generation:"g1",allocation_id:"allocation-1",url:"https://example.test/"};
  const call = (method, params=owner) => loaded.withOwnership(method,params,()=>loaded[method]?.(params,"synthetic-request") ?? Promise.resolve({}),loaded.close);
  return {call,loaded,owner,tabs,removed,faults,store:()=>store,
    restart:async()=>{loaded=await import(pathToFileURL(outfile)+`?restart=${Date.now()}`);},
    resetStorage:()=>{store={};},close:()=>rm(dir,{recursive:true,force:true})};
}

import http from "node:http";
import { fixture } from "./ownership.mjs";

// Test-only HTTP peer for the REAL Python BridgeClient. It dispatches through
// the bundled extension ownership/nav modules, but every Chrome API is in-memory.
const f = await fixture();
let oldExtension = false;
let refuseAttach = false;
let calls;
const server = http.createServer(async (request, response) => {
  if (request.headers["x-bridge-key"] !== process.env.BROWSER_FIXTURE_KEY) {
    response.writeHead(401).end("unauthorized"); return;
  }
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : {};
  response.setHeader("content-type", "application/json");
  if (request.url === "/fixture") {
    Object.assign(f.faults, body.faults ?? {});
    if (typeof body.oldExtension === "boolean") oldExtension = body.oldExtension;
    if (body.restartWorker) await f.restart();
    if (body.resetBrowser) f.resetStorage();
    if (typeof body.refuseAttach === "boolean") refuseAttach = body.refuseAttach;
    if (body.captureCalls) calls = [];
    response.end(JSON.stringify({tabs:f.tabs.size,removed:f.removed.length,calls})); return;
  }
  try {
    calls?.push({method: body.method, has_tab: Boolean(body.params?.tab)});
    if (oldExtension && body.method.startsWith("owner_")) throw {code:"internal",message:`unknown method ${body.method}`};
    // This peer omits the page-action bodies. Feed the same typed reply as
    // the worker regression, retiring only the capability, not the fake tab.
    const surfaces = f.store().surfaces;
    const pin = body.params?.tab;
    if (refuseAttach && ["read", "click"].includes(body.method) && pin && Object.hasOwn(surfaces ?? {}, pin)) {
      delete surfaces[pin];
      throw {code:"internal",message:"Chrome refused debugger attachment to this tab",data:{undrivable_tab:"different_extension",surface_retired:true}};
    }
    const result = await f.call(body.method, body.params);
    response.end(JSON.stringify({id:body.id,ok:true,result}));
  } catch (error) {
    response.end(JSON.stringify({id:body.id,ok:false,error:{code:error.code ?? "internal",message:error.message ?? String(error),data:error.data ?? {}}}));
  }
});
server.listen(0,"127.0.0.1",()=>console.log(JSON.stringify({port:server.address().port})));
process.on("SIGTERM",()=>server.close(async()=>{await f.close();process.exit(0);}));

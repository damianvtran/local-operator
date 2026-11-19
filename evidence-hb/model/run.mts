import { readFileSync, readdirSync } from "node:fs";
import { bridgeCardBodyOf } from "./bridge-receipt-model.ts";
const dir = "/Users/damian/.local-operator/sessions/3d299fb1b279/scratchpad/hb/real";
for (const f of readdirSync(dir).sort()) {
  const m = bridgeCardBodyOf(JSON.parse(readFileSync(dir + "/" + f, "utf8")));
  const items = (m?.items ?? []).map((i: any) => i.state + (i.reason ? "/" + i.reason : "") + (i.transient ? "/transient" : "") + (i.body ? "/body" + i.body.length : ""));
  console.log(f.replace(".json", "").padEnd(10), "items=" + (m?.items?.length ?? "null"), "more=" + m?.more, JSON.stringify([...new Set(items)]));
}

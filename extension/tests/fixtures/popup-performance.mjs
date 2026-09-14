/* Synthetic storage and health only: the caller executes the shipped popup.
 * IPC byte counts are JSON payload sizes, not Chrome transport measurements.
 * Barriers let tests prove ordering without machine-speed assertions. */
export const FIXTURE_ID = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
export function installPopupPerformance({ state = "healthy", delay = 0, large = false, gate, healthFetch } = {}) {
  const metrics = { reads: [], writes: [], health: [], messages: [], mutations: 0 };
  const session = {
    connState: "connected", surfaces: { synthetic: { tabId: 1, nonce: "fixture", epoch: 1 } },
    ...(large ? { refs: { synthetic: Object.fromEntries(Array.from({ length: 10000 }, (_, i) => [`e${i}`, { backendDOMNodeId: i, role: "link", name: "Synthetic reference ".repeat(8) }])) } } : {}),
  };
  if (state === "consent") session.accessQueue = [{ entryId: "fixture-entry", origin: "https://example.com", displayAuthority: "example.com", requester: "fixture-owner", kind: "async", requestedAt: Date.now(), expiresAt: Date.now() + 600000, sequence: 1, broad: { scope: "domain", key: "example.com" } }];
  const local = { port: 4099, allowAllSites: false, ...(large ? { origins: Object.fromEntries(Array.from({ length: 10000 }, (_, i) => [`https://site${i}.example`, "allow"])) } : {}) };
  const listeners = [];
  const emit = (changes, area = "session") => listeners.forEach(fn => fn(changes, area));
  const wait = () => delay ? new Promise(resolve => setTimeout(resolve, delay)) : Promise.resolve();
  const area = (name, values) => ({
    async get(keys) {
      const read = { area: name, keys, start: performance.now() - start };
      metrics.reads.push(read);
      if (gate) await gate(read);
      if (state === "stalled-storage") await new Promise(() => {});
      await wait();
      const result = Object.fromEntries((keys == null ? Object.keys(values) : Array.isArray(keys) ? keys : [keys]).filter(k => k in values).map(k => [k, values[k]]));
      const wire = JSON.stringify(result);
      read.bytes = new TextEncoder().encode(wire).length;
      read.end = performance.now() - start;
      return JSON.parse(wire);
    },
    async set(changes) {
      metrics.writes.push({ area: name, keys: Object.keys(changes) });
      const events = Object.fromEntries(Object.entries(changes).map(([k, newValue]) => [k, { oldValue: values[k], newValue }]));
      Object.assign(values, changes);
      emit(events, name);
    },
  });
  globalThis.chrome = {
    storage: { local: area("local", local), session: area("session", session), onChanged: { addListener: fn => listeners.push(fn) } },
    runtime: { id: FIXTURE_ID, getManifest: () => ({ version: "fixture" }), openOptionsPage() {}, reload() { throw new Error("Fixture cannot reload an extension"); }, async sendMessage(message) { metrics.messages.push(message); return { applied: false }; } },
    tabs: { query: async () => [] },
  };
  const health = { paired: true, extension_connected: true, protocol_version: 1, authorized_extension_ids: [FIXTURE_ID], driver_extension_id: FIXTURE_ID, surface_url: "https://example.com/fixture", surface_title: "Synthetic fixture" };
  if (state === "standby") Object.assign(health, { driver_extension_id: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", driver_label: "Synthetic second browser", standby_extension_ids: [FIXTURE_ID] });
  globalThis.fetch = async (...args) => {
    const call = { start: performance.now() - start }; metrics.health.push(call);
    await wait();
    if (state === "slow") await new Promise(resolve => setTimeout(resolve, 300));
    if (state === "unreachable") throw new Error("Synthetic daemon unavailable");
    const response = healthFetch ? await healthFetch(...args) : { ok: true, json: async () => health };
    call.end = performance.now() - start;
    return response;
  };
  // Fixture construction (especially the large ref map) is not popup work.
  // Start after provisioning it, before the caller imports the shipped module.
  const start = performance.now();
  return { metrics, start, session, local, emit, health };
}

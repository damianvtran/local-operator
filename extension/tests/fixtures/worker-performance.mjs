/* Synthetic Chrome host for the shipped worker; transport may be a real local
 * WebSocket. No tabs, credentials or installed extension are accessed. */
export function installWorkerPerformance({ port, id, token, onState = () => {}, action = async () => {} }) {
  const session = {}, local = { port, token };
  const listeners = {};
  const metrics = { reads: [], writes: [], actions: [] };
  const event = name => ({ addListener(fn) { (listeners[name] ||= []).push(fn); } });
  const area = (name, values) => ({
    async get(keys) { const result = Object.fromEntries((Array.isArray(keys) ? keys : [keys]).filter(k => k in values).map(k => [k, values[k]])); metrics.reads.push({ area: name, keys, bytes: JSON.stringify(result).length }); return structuredClone(result); },
    async set(changes) { const old = { ...values }; Object.assign(values, structuredClone(changes)); metrics.writes.push({ area: name, keys: Object.keys(changes) }); for (const fn of listeners.storage || []) fn(Object.fromEntries(Object.entries(changes).map(([k, newValue]) => [k, { oldValue: old[k], newValue }])), name); onState(name, changes); },
    async remove(keys) { for (const key of Array.isArray(keys) ? keys : [keys]) delete values[key]; },
  });
  globalThis.chrome = {
    storage: { session: area("session", session), local: area("local", local), onChanged: event("storage") },
    alarms: { create() {}, clear: async () => true, onAlarm: event("alarm") },
    action: Object.fromEntries(["setBadgeBackgroundColor", "setBadgeTextColor", "setBadgeText", "setTitle"].map(name => [name, async args => { metrics.actions.push({ name, args }); await action(name, args); }])),
    debugger: { attach: async () => {}, detach: async () => {}, sendCommand: async () => ({}), onEvent: event("debugger"), onDetach: event("detach") },
    scripting: { executeScript: async () => [] },
    tabs: { query: async () => [], get: async () => { throw new Error("No real tabs in fixture"); }, remove: async () => {}, onRemoved: event("tabRemoved"), onReplaced: event("tabReplaced"), onUpdated: event("tabUpdated") },
    windows: { get: async () => ({ id: 1 }), getCurrent: async () => ({ id: 1 }) },
    notifications: { create: async () => {}, clear: async () => {}, onClicked: event("notification") },
    runtime: { id, getManifest: () => ({ version: "0.1.16" }), getURL: path => `chrome-extension://${id}/${path}`, sendMessage: async () => {}, onStartup: event("startup"), onInstalled: event("installed"), onMessage: event("message"), openOptionsPage() {} },
  };
  return { session, local, metrics, listeners };
}

/* Re-export shim: the module itself now lives in `driver/origin-policy.ts`.
 *
 * WHY A SHIM RATHER THAN REPOINTING THE IMPORTERS: this module has six import
 * sites in the extension (`access-grants`, `origins`, the options page, the
 * popup and its origin flow, and the queue), and every one of them is a line in
 * the diff of a change whose whole point is that it alters no behaviour. The
 * modules with one or two call sites were repointed instead, so `driver/` holds
 * the logic and this file is only a path.
 *
 * The `driver/` directory is the host-free set a second host vendors whole;
 * `export *` rather than a named list so a new export there cannot silently
 * disappear from this path. Do NOT add logic here — anything that touches the
 * `chrome` API belongs on this side of the boundary, outside `driver/`, and
 * would also break the invariant `tests/driver-host-free.test.mjs` enforces. */
export * from "./driver/origin-policy";

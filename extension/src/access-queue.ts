/* Re-export shim: the module itself now lives in `driver/access-queue.ts`.
 *
 * WHY A SHIM RATHER THAN REPOINTING THE IMPORTERS: eight import sites — the
 * approval store, `state`, `origins`, the popup, the popup's origin flow, the
 * access command and `access-flow` — and repointing them would put eight
 * review-invisible lines into a diff whose point is that it changes nothing.
 * Modules with one or two call sites were repointed instead, so `driver/` holds
 * the logic and this file is only a path.
 *
 * The `driver/` directory is the host-free set a second host vendors whole;
 * `export *` rather than a named list so a new export there cannot silently
 * disappear from this path. Do NOT add logic here — anything that touches the
 * `chrome` API belongs on this side of the boundary, outside `driver/`, and
 * would also break the invariant `tests/driver-host-free.test.mjs` enforces. */
export * from "./driver/access-queue";

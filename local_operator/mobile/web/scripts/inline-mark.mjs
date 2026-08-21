/**
 * Regenerate src/lib/mark.ts from public/mark.png.
 *
 * The header mark is inlined as a data URI so it needs no request: over an
 * identity-proxied tunnel (Cloudflare Access) a /mark.png fetch is itself
 * gated, so the <img> got a 302 HTML body and rendered the broken-image
 * glyph. Run after replacing public/mark.png. The matching daemon-side copy
 * (local_operator/mobile/daemon.py::_mark_data_uri) reads the same PNG from
 * mobile/static/, so both surfaces stay the one image.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const png = readFileSync(join(root, "public", "mark.png"));
const b64 = png.toString("base64");
const out = `/** The LO mark as a data URI — inlined so it needs no request. Over an
 * identity-proxied tunnel (Cloudflare Access) a /mark.png fetch is itself
 * gated, so the header <img> got a 302 HTML body and showed the broken-image
 * glyph. Inline renders behind Access and pre-auth alike. Generated from
 * public/mark.png by scripts/inline-mark.mjs; do not hand-edit the string.
 */
export const MARK_DATA_URI = "data:image/png;base64,${b64}";
`;
writeFileSync(join(root, "src", "lib", "mark.ts"), out);
console.log(`src/lib/mark.ts regenerated (${png.length} bytes)`);

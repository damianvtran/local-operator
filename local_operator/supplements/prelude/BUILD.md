# Prelude build note

`prelude.css` / `prelude.js` are the **minified build** that `supplements/document.py` injects
into every component and that `tests/unit/supplements/test_prelude.py` pins. `prelude.src.css` /
`prelude.src.js` are the hand-maintained sources (git only; the wheel ships the minified pair).

Rebuild with **esbuild 0.28.2** (other versions emit different bytes: a 1-byte `@media`
spacing difference in the CSS and a 2-byte difference in the JS were measured, and the
size budget has 31 B of gzip headroom):

    npx --yes esbuild@0.28.2 prelude.src.css --minify --outfile=prelude.css
    npx --yes esbuild@0.28.2 prelude.src.js  --minify --outfile=prelude.js

Then bump `PRELUDE_VERSION` in `supplements/document.py` and update the pinned digest in
`tests/unit/supplements/test_prelude.py`; the digest test fails until you do.

**Size method, one for every figure:** `cat prelude.css prelude.js | gzip -9 | wc -c`.

| build | CSS | JS | raw | gzip -9 |
|---|---|---|---|---|
| spike, round-2 label layout (memo App. C) | 2,044 | 8,180 | 10,224 | 4,538 |
| C0, with the per-frame nonce echo (memo §4.1 S-R4) | 2,044 | 8,259 | 10,303 | 4,577 |

Caps (memo §2.6): ≤ 11 KB raw (11,264 B) / ≤ 4.5 KB gzip (4,608 B). The C0 pair leaves
961 B raw / 31 B gzip. Any addition re-measures and, at this margin, trims.

The App. B accent guard (+412 B gzip) is deliberately NOT in the build; it lands with the
surface that needs it, under the size test.

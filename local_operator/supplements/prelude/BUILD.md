# Prelude build note

`prelude.css` / `prelude.js` are the **minified build** that `supplements/document.py` injects
into every component and that `tests/unit/supplements/test_prelude.py` pins. `prelude.src.css` /
`prelude.src.js` are the hand-maintained sources (git only; the wheel ships the minified pair).

Rebuild with **esbuild 0.28.2** (other versions emit different bytes: a 1-byte `@media`
spacing difference in the CSS and a 2-byte difference in the JS were measured, and the
size budget has 1 B of gzip headroom):

    npx --yes esbuild@0.28.2 prelude.src.css --minify --outfile=prelude.css
    npx --yes esbuild@0.28.2 prelude.src.js  --minify --outfile=prelude.js

Then bump `PRELUDE_VERSION` in `supplements/document.py` and update the pinned digest in
`tests/unit/supplements/test_prelude.py`; the digest test fails until you do.

**Size method, one for every figure:** `cat prelude.css prelude.js | gzip -9 | wc -c`.

| build | CSS | JS | raw | gzip -9 |
|---|---|---|---|---|
| spike, round-2 label layout (memo App. C) | 2,044 | 8,180 | 10,224 | 4,538 |
| C0, with the per-frame nonce echo (memo §4.1 S-R4) | 2,044 | 8,259 | 10,303 | 4,577 |
| C0 round 1: pre-nonce `error` held and flushed (QA Q-2), nonce > 64 refused not truncated (R4) | 2,044 | 8,202 | 10,246 | 4,607 |

Caps (memo §2.6): ≤ 11 KB raw (11,264 B) / ≤ 4.5 KB gzip (4,608 B). The C0 pair leaves
1,018 B raw / **1 B gzip**. The round-1 additions were paid for by trims with no behaviour
change: one error-post helper for three call sites, destructuring the message event, one
`"use strict"` instead of two, and a space out of the fallback media query. Stripping the
nonce from `LO.theme` measured 4,611 B and was NOT taken (it would also hide nothing: any
component script can add its own `message` listener).

**The cap is not final for native (lane N).** The memo's §4.1 Native row needs the prelude to
post through `window.ReactNativeWebView.postMessage` when present, listen on `document` as
well as `window`, and accept the iOS host push whose `event.source` is `null` (the
`e.source!==parent` check drops it today). None of that fits in 1 B. Lane N either injects a
native-only shim at assembly time without touching this shared pair, or revisits the cap in its
own PR with measured numbers.

The App. B accent guard (+412 B gzip) is deliberately NOT in the build; it lands with the
surface that needs it, under the size test.

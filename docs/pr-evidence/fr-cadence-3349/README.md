# French cadence #3349 — retained QA evidence

These 12 comparison PNGs and original matrix reports are copied byte-for-byte from independent QA session `b65f5adc1dfb`; `manifest.json` binds each portable path to its original relative source path, SHA-256 and revisions. This retention is not a new capture, audit or visual sign-off.

## Before source-fit

`pre-fit/compare-{dark,light}-{60x20,80x24,150x40}.png` show English fallback at `029c0ec7ade8be8bdb6625a5816b4e23fc0a2902` on the left and original French at `dcdb0b280553861b72b225695497810d201ef890` on the right. Original French label/help were 36/206 cells; these are deliberately retained regression evidence, **not the fixed result**. See `qa-native-supplement.md` for original interaction matrix/geometry.

## Fixed source-fit

`fixed/selected-comparison-{60x20,80x24,150x40}.png` compare English / original French / compact French, with dark/light rows. `fixed/editor-comparison-{60x20,80x24,150x40}.png` show compact French editor, rejection and correction direct neighbors in both themes. Fixed source revision is `005d1eed1dbb1de84db78398ab188ec3df9b821d`; its French catalogue SHA-256 is `42333772890dd59733f0bcff715c9d608ff1f7c069c8dbb025ee3b2f0548ed63`. See `qa-round2.md` for exact matrix, geometry and first/settled results. Label/help are 25/133 cells; full wide help and label fit, while pre-existing narrow single-line help truncation remains disclosed.

The later `4472b530665c4ab40c5f3a2884f25fe2c25d48b9` change is test-only; runtime/catalogue/evidence bytes were verified identical to the fixed revision. Current provenance-only retention does not change those inputs or reset suites. Original reports contain historical model claims: **apply `qa-provenance-override-3349.json`**, authored by QA from completed-message usage metadata, rather than treating requested runtime headers as served-model evidence. Immutable original reports are not rewritten.

All source captures use the synthetic cadence fixture described in those reports, not operator sessions. Strict credential/private-key patterns over retained binary/text bytes returned zero matches (counts in manifest); this is not an OCR guarantee. No dependencies, full raw frame archive or live configuration are retained here.

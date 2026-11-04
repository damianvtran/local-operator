# Actual host-refusal error presentation — integration capture

The CSS-loaded `OperatorApp` was run with `run_test(size=(100,32))`. The real
`AttachedSession.cold(...)._ensure_bound()` generated the refused-host exception;
`FakeSession` carried that exception through the real TUI prompt-error handler.
Only the host-request seam and memory admission were doubled. No runtime was
spawned and no live session was accessed. `capture.py.txt` is the executed rig.

Frames: English, long French, Urdu, and Urdu with `LOP_I18N=0` forced English
fallback. The resolver's actual exception text and CSS path are in each receipt.
First and settled SVGs have identical rendered text in all four cases. Native
PNG rasterizations of settled frames were opened and visually inspected.

At 100x32, active screen content/virtual size is 98x30; neither screen scrollbar
appeared. English and French wrap over three rows without clipping; the failure
is visible above the composer. The fallback frame shows English guidance.

**Limitation, not a native RTL sign-off:** the Urdu raster is visibly degraded
(unjoined letters / terminal ordering); this is the existing documented TUI
stance, not browser-quality Arabic shaping or mirrored layout. Error text has no
embedded bidi control characters. This capture establishes the actual degraded
presentation and English kill-switch fallback, not new bidi isolation or native
RTL support. No peer-owned app/AttachedSession code was changed to improve it.
The screenshot does not certify a translation; the two independent semantic
audits per locale are retained separately under `i18n/audits/evidence/`.

Only trailing whitespace on unpainted SVG lines was normalized for the repository whitespace gate; captured text, geometry and PNG frames are unchanged.

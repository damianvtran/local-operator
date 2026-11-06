# Spanish SDK startup refusals — staged translation only (4 keys)

Locale es · namespace wire.errors.sdk_startup · immutable core revision
0e7cde33ca8008d561e74f4aed774aa5f76ff7d5.

## What this is

Four idiomatic Spanish catalogue values for `wire.errors.sdk_startup`, staged as
a candidate file `wire.errors.sdk_startup.json`. Nothing was installed,
generated, committed, activated or ledgered.

## Provenance

- Translator/session/model, fallback interpretation and honour-system both:
  `translator-record.json` and `served-model-evidence.json`.
- Producer/caller map and the render-boundary analysis: `caller-proof.md`,
  derived from the pinned blobs, with the retained architect assessment
  (`architect-assessment.md`) named as a separate source.
- Immutable inputs: `en.json`, `context.json`, `es-style.md`, `glossary.md`,
  `HOWTO.md`, `RFC.md`, producer snapshots, sibling `ru/vi/ur/hi` values.

## Mechanical evidence

`check_staged.py` runs under `scripts/run_bounded.py --memory-mb 256
--timeout 90` with inherited `CMUX_*` stripped; output is `mechanical.log`
(exit 0). It parses with the pinned core ICU runtime (`runtime.py`,
`format.py`), checks exact key set/order, placeholder name/type parity,
technical literals, English-fragment guards, terminal-punctuation parity and
absence of bidi controls. Eleven planted defects are rejected, including a
planted English fragment, a translated `mode='own'` literal and an injected
bidi character. A pinned-runtime render check shows `{error}`, `{params}` and
the repr'd strings pass through byte-exactly, so they are not re-parsed.

## Two caveats a reader must carry forward

1. **Ship-set coupling.** `generate.py::_shipped_locales` requires every
   namespace a locale CARRIES to have a passed, fresh ledger row. Committing
   this file without its independent audit row would drop `es` from the
   generated shipped set — a locale-wide English regression. The catalogue file
   and its audit row must land together.
2. **Embedded English cause.** `attach_hosting_unconfigured` keeps `{error}`
   opaque by its own context note, so the factory's English
   "Hosting platform is not configured." still appears inside the Spanish
   sentence until a display-specific owned key exists in every locale. That is
   a caller/producer change, out of scope here and not claimed as done.

## Not run

Linguistic audit, product/UI rendering, pseudo-localization, RTL rendering,
SDK/runtime integration, the full catalogue validator, coverage/union gates,
scanner, tests, build, install and any native run.

No self-audit: independent readers must audit these exact candidate bytes.
This is a translator packet, not a tracked audit record.

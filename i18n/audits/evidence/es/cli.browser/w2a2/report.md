# Spanish W2 independent audit — cli.browser — w2a2

**Verdict: PASSED.** 64 keys read in English and Spanish, with all 64 English context entries. No blocker, major, or minor findings; 2 nit(s).

- Audited revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d` (`feat/i18n-es-wire-runtime`).
- Target is untracked; on-disk sha256: `5c44bbf8afb79aeddf704464621b3858aed8bc51e3435af5061227aa6e0182e2`.
- English sha256: `a9841deed81927762716c7592a592bb4e3f069d16f6ba0def6becb755b728321`; context sha256: `1e5f1dfeaa5d828302831cba4134605a1a7ff46b5c8ac7af7fcd913da95cbaee`.
- Independence: authored and repaired none of the copy; no repository writes beyond this evidence directory.

## Verified, not assumed

- Read every key against English meaning, context role/tone/notes, neutral Spanish idiom, tú address, infinitive action labels, sentence case, glossary, and product-name policy.
- Parsed every message with the repository ICU runtime and compared argument sets, plural selectors, and AST select branch labels. No plural messages occur here.
- Compared flags, backtick spans, ANSI escapes, Rich tags, and argparse printf tokens per key. No key in these five namespaces actually has a `%(... )s` or `%%` token.
- Confirmed authored newline counts match. Wrapped browser cleanup/status messages are self-contained copy; layout remains with the caller.
- No `maxCells` declaration exists in these sidecars. No clipping finding is inferred merely from Spanish being longer.
- The real `build_cli_parser()` rendered `Iniciar el daemon`, `Detener el daemon`, and `Reiniciar el daemon`; the select labels remain `start`/`stop`/`restart` and an unknown value falls through to `other`.
- The JSON Schema example renders as `{'required_sections': [...]}` through the real parser.
- Byte-identical English is legitimate where present: `supervisor` is also Spanish, `no` is also Spanish, and `Error: {sentence}` preserves the required prefix around opaque provider/recovery prose.

## Findings

### es-w2-B01 — nit

- Key: `cli.browser.installed_line`
- Current: `instalado:          {value}`
- Problem: Current text is 'instalado:          {value}'. The value starts at zero-based character column 20, while the aligned browser status block uses column 21. This is a one-cell visual misalignment, not clipping; the value remains visible and no maxCells budget is declared.
- Proposed: Use 'instalado:           {value}' (one additional ASCII space).

### es-w2-B02 — nit

- Key: `cli.browser.daemon_healthy`
- Current: `daemon en buen estado: {value}`
- Problem: Current text is 'daemon en buen estado: {value}'. The value starts at zero-based character column 23, while the aligned browser status block uses column 21. This is a two-cell visual misalignment, not clipping; the value remains visible and no maxCells budget is declared.
- Proposed: Use 'daemon sano:           {value}' to preserve the meaning and start the value at column 21.


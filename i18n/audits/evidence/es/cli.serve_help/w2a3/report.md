# Spanish audit — cli.serve_help — round 1 (w2a3)

- **Verdict: failed** — 0 blocker, 1 major, 0 minor, 0 nit
- Keys read: 5 English, 5 Spanish, 5 context entries
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d`; branch `feat/i18n-es-wire-runtime`
- Target is UNTRACKED; on-disk sha256: `71ff3f34bab59a2908341f5aabd01a3c4dd723c61f55bab4c1d30d701d789842`
- English sha256: `bd0b51271edc67c23978ca23688a9af1ef94a5623b0e86a5e3bbb418a5b243f0`; context sha256: `d569b1252c7965cc07456c6dba086187237c609a6d9f3021a206a07bf731c770`
- Auditor: translation-auditor / es-w2-auditor-c, session 422494cdb264, model **kimi/k3.1-pro** (self-reported; fallback from anthropic/claude-sonnet-5-5 via openai/gpt-6-astra after quota HTTP 429 notices)
- Independence: no translation or repair authored; only this evidence directory written.

## Verdict reason

Read all 5 keys. 0 blocker, 1 major, 0 minor, 0 nit; any blocker or major fails the namespace. All parser-derived machine checks pass and every behavioural promise not named in a finding survives.

## What was verified rather than assumed

1. Read every English value, Spanish value and context entry side by side; judged accuracy, idiom, tú/infinitive register, behavioural promises, product-name policy and glossary consistency.
2. Used the repository runtime.py parser, not a regex or visual brace comparison, for message_arguments, plural_selectors and select-label signatures. Exact key sets and parser-derived argument tuples match; zero plural arguments occur.
3. Checked flags, backticked commands, choice tokens, identifiers, numeric literals and argparse printf tokens against the actual strings. No required source token was dropped or renamed.
4. No maxCells is declared in this namespace. Width was reviewed without a declared clipping budget; no rendered-fit or terminal-layout pass is claimed.
5. Spanish is LTR, so RTL-specific checks are not applicable. No value displayed as a redaction marker; the catalogue bytes themselves were read.
6. Derived daemon drift by enumerating tracked es JSON with git ls-files: committed relay.asks has three 'daemon' values; no committed es value contains 'demonio'. Sibling cli.browser_help also uses 'daemon'.
7. Accepted 'compilación' for build: it matches the existing es catalogue and accurately distinguishes builds from versions; no build/compilación drift finding.

## Findings

### es-w2-C04 — MAJOR

- Keys: cli.serve_help.listener_fd
- Exact current text and evidence: Glossary drift: this namespace uses 'demonio'/'demonios', while the committed, shipped es relay.asks catalogue uses 'daemon' in error.reach_failed, error.no_reason and error.read_timeout; the W2 sibling cli.browser_help also uses 'daemon'. Neither the style guide nor glossary authorizes a daemon/demonio split. Current assigned values: cli.serve_help.listener_fd: Adoptar un socket de escucha ya asociado en lugar de asociar uno. Lo establece un demonio que reemplaza su propia imagen de proceso con una nueva compilación (`server/reload`), para que el puerto nunca quede sin servicio. No es un flag para el usuario.
- Proposed remediation: Standardize on 'daemon'/'daemons' in these keys (and coordinate the adjacent W2 browser keys so the surface stays consistent). Keep all surrounding grammar and ICU bodies otherwise unchanged.

## Mechanical result

Final checker: **0 mechanical failures**. Parser-derived argument, plural and select checks passed; required source tokens survived. Synthetic negative controls detected translated select labels, renamed arguments, dropped printf tokens and invalid Spanish plural categories.

No `maxCells` is declared; no rendered-fit test is claimed. No repo-wide tests were run, per delegation constraints.

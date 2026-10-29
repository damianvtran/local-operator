# Spanish audit — cli.update_help — round 1 (w2a3)

- **Verdict: failed** — 0 blocker, 1 major, 1 minor, 0 nit
- Keys read: 6 English, 6 Spanish, 6 context entries
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d`; branch `feat/i18n-es-wire-runtime`
- Target is UNTRACKED; on-disk sha256: `c4d2302ef3a4b10c36b234fff33e80905ae4855ba8734445a174d9a948537190`
- English sha256: `78da39b393ee77900255ddfdf92a17220a6eedbecba340e7d925a2fbdf01abf9`; context sha256: `46dfc07c168a78805b437a4561dc0a8f47b005a48176b424b5aca47373cdaa5f`
- Auditor: translation-auditor / es-w2-auditor-c, session 422494cdb264, model **kimi/k3.1-pro** (self-reported; fallback from anthropic/claude-sonnet-5-5 via openai/gpt-6-astra after quota HTTP 429 notices)
- Independence: no translation or repair authored; only this evidence directory written.

## Verdict reason

Read all 6 keys. 0 blocker, 1 major, 1 minor, 0 nit; any blocker or major fails the namespace. All parser-derived machine checks pass and every behavioural promise not named in a finding survives.

## What was verified rather than assumed

1. Read every English value, Spanish value and context entry side by side; judged accuracy, idiom, tú/infinitive register, behavioural promises, product-name policy and glossary consistency.
2. Used the repository runtime.py parser, not a regex or visual brace comparison, for message_arguments, plural_selectors and select-label signatures. Exact key sets and parser-derived argument tuples match; zero plural arguments occur.
3. Checked flags, backticked commands, choice tokens, identifiers, numeric literals and argparse printf tokens against the actual strings. No required source token was dropped or renamed.
4. No maxCells is declared in this namespace. Width was reviewed without a declared clipping budget; no rendered-fit or terminal-layout pass is claimed.
5. Spanish is LTR, so RTL-specific checks are not applicable. No value displayed as a redaction marker; the catalogue bytes themselves were read.
6. Derived daemon drift by enumerating tracked es JSON with git ls-files: committed relay.asks has three 'daemon' values; no committed es value contains 'demonio'. Sibling cli.browser_help also uses 'daemon'.
7. Accepted 'compilación' for build: it matches the existing es catalogue and accurately distinguishes builds from versions; no build/compilación drift finding.
8. Verified the PyPI-versus-lop-update distinction, published macOS wheel exception, downgrade restriction, supervised-daemon exception and manual mesh-roll path against the English and sidecars.

## Findings

### es-w2-C06 — MAJOR

- Keys: cli.update_help.no_services
- Exact current text and evidence: Glossary drift: this namespace uses 'demonio'/'demonios', while the committed, shipped es relay.asks catalogue uses 'daemon' in error.reach_failed, error.no_reason and error.read_timeout; the W2 sibling cli.browser_help also uses 'daemon'. Neither the style guide nor glossary authorizes a daemon/demonio split. Current assigned values: cli.update_help.no_services: No pasar los demonios `lop serve` a la nueva compilación; los demonios supervisados sí se reparan. Usar solo cuando algo más vaya a iniciar esos demonios, p. ej., un script que controle su lanzamiento.
- Proposed remediation: Standardize on 'daemon'/'daemons' in these keys (and coordinate the adjacent W2 browser keys so the surface stays consistent). Keep all surrounding grammar and ICU bodies otherwise unchanged.

### es-w2-C07 — MINOR

- Keys: cli.update_help.from_snapshot
- Exact current text and evidence: Current typography: 'de PyPI —la única compilación que incluye el agente de claves de macOS—; cualquier'. The Spanish guide requires an em dash with spaces (' — '). The published-wheel/key-agent behavioural distinction itself is accurate.
- Proposed remediation: Use 'de PyPI — la única compilación que incluye el agente de claves de macOS —; cualquier', preserving Spanish's spaced em dashes and no space before the semicolon. Alternatively restructure to 'de PyPI, la única compilación que incluye el agente de claves de macOS; cualquier' to avoid the parenthetical dash/semicolon junction.

## Mechanical result

Final checker: **0 mechanical failures**. Parser-derived argument, plural and select checks passed; required source tokens survived. Synthetic negative controls detected translated select labels, renamed arguments, dropped printf tokens and invalid Spanish plural categories.

No `maxCells` is declared; no rendered-fit test is claimed. No repo-wide tests were run, per delegation constraints.

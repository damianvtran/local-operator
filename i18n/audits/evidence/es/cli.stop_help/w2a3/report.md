# Spanish audit — cli.stop_help — round 1 (w2a3)

- **Verdict: passed** — 0 blocker, 0 major, 0 minor, 0 nit
- Keys read: 9 English, 9 Spanish, 9 context entries
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d`; branch `feat/i18n-es-wire-runtime`
- Target is UNTRACKED; on-disk sha256: `5007e8644710979b686f02e3ef1dddb03766c2aab7c5a78b66be88e34e3ab3b2`
- English sha256: `ab8ae153e8749888e2426614c06d1ba672c70bad33d2a44bed46c00004f33d28`; context sha256: `1f94c5fc9b839aa459fa989f3480be38641251c8e581d436703642c3f474105d`
- Auditor: translation-auditor / es-w2-auditor-c, session 422494cdb264, model **kimi/k3.1-pro** (self-reported; fallback from anthropic/claude-sonnet-5-5 via openai/gpt-6-astra after quota HTTP 429 notices)
- Independence: no translation or repair authored; only this evidence directory written.

## Verdict reason

Read all 9 keys. No semantic, register, glossary, machine-syntax or width defect found. All parser-derived machine checks pass and every behavioural promise not named in a finding survives.

## What was verified rather than assumed

1. Read every English value, Spanish value and context entry side by side; judged accuracy, idiom, tú/infinitive register, behavioural promises, product-name policy and glossary consistency.
2. Used the repository runtime.py parser, not a regex or visual brace comparison, for message_arguments, plural_selectors and select-label signatures. Exact key sets and parser-derived argument tuples match; zero plural arguments occur.
3. Checked flags, backticked commands, choice tokens, identifiers, numeric literals and argparse printf tokens against the actual strings. No required source token was dropped or renamed.
4. No maxCells is declared in this namespace. Width was reviewed without a declared clipping budget; no rendered-fit or terminal-layout pass is claimed.
5. Spanish is LTR, so RTL-specific checks are not applicable. No value displayed as a redaction marker; the catalogue bytes themselves were read.
6. Accepted 'compilación' for build: it matches the existing es catalogue and accurately distinguishes builds from versions; no build/compilación drift finding.

## Findings

Clean: all 9 keys read; no findings.

## Mechanical result

Final checker: **0 mechanical failures**. Parser-derived argument, plural and select checks passed; required source tokens survived. Synthetic negative controls detected translated select labels, renamed arguments, dropped printf tokens and invalid Spanish plural categories.

No `maxCells` is declared; no rendered-fit test is claimed. No repo-wide tests were run, per delegation constraints.

# Spanish audit — cli.hub_help — round 1 (w2a3)

- **Verdict: failed** — 1 blocker, 0 major, 0 minor, 0 nit
- Keys read: 12 English, 12 Spanish, 12 context entries
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d`; branch `feat/i18n-es-wire-runtime`
- Target is UNTRACKED; on-disk sha256: `b28d2b692f21374217d5236241eb50c0d14873f7b82587e65debbe3e9697642e`
- English sha256: `df519de9acb3cdd99fa5ae4cc0c5753b090c43b79f8870c543258c0f4fcdfec6`; context sha256: `5b948945d34b6fc0b08c0de5739f65dc86069c75796dfa2b48506f461aca3a36`
- Auditor: translation-auditor / es-w2-auditor-c, session 422494cdb264, model **kimi/k3.1-pro** (self-reported; fallback from anthropic/claude-sonnet-5-5 via openai/gpt-6-astra after quota HTTP 429 notices)
- Independence: no translation or repair authored; only this evidence directory written.

## Verdict reason

Read all 12 keys. 1 blocker, 0 major, 0 minor, 0 nit; any blocker or major fails the namespace. All parser-derived machine checks pass and every behavioural promise not named in a finding survives.

## What was verified rather than assumed

1. Read every English value, Spanish value and context entry side by side; judged accuracy, idiom, tú/infinitive register, behavioural promises, product-name policy and glossary consistency.
2. Used the repository runtime.py parser, not a regex or visual brace comparison, for message_arguments, plural_selectors and select-label signatures. Exact key sets and parser-derived argument tuples match; zero plural arguments occur.
3. Checked flags, backticked commands, choice tokens, identifiers, numeric literals and argparse printf tokens against the actual strings. No required source token was dropped or renamed.
4. No maxCells is declared in this namespace. Width was reviewed without a declared clipping budget; no rendered-fit or terminal-layout pass is claimed.
5. Spanish is LTR, so RTL-specific checks are not applicable. No value displayed as a redaction marker; the catalogue bytes themselves were read.

## Findings

### es-w2-C01 — BLOCKER

- Keys: cli.hub_help.hub
- Exact current text and evidence: Current: 'Actualizaciones de Agent Hub'. English: 'Agent Hub update status'. The translation changes the command from reporting update status to the updates themselves; the sidecar identifies this as the Agent Hub command summary.
- Proposed remediation: Use 'Estado de las actualizaciones de Agent Hub'.

## Mechanical result

Final checker: **0 mechanical failures**. Parser-derived argument, plural and select checks passed; required source tokens survived. Synthetic negative controls detected translated select labels, renamed arguments, dropped printf tokens and invalid Spanish plural categories.

No `maxCells` is declared; no rendered-fit test is claimed. No repo-wide tests were run, per delegation constraints.

# Spanish audit — cli.mobile_help — round 1 (w2a3)

- **Verdict: failed** — 0 blocker, 1 major, 0 minor, 0 nit
- Keys read: 15 English, 15 Spanish, 15 context entries
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d`; branch `feat/i18n-es-wire-runtime`
- Target is UNTRACKED; on-disk sha256: `133b59107b2004d69333cd7e295b8e49cf45482ecbc22ea33ce50950aa1e3d9a`
- English sha256: `88681bae09adf93216e3e3b990652f7f57654a0438f6bc8c3cc5dd9ebb6eb947`; context sha256: `26054791e1413cfba7d777561c410c8de6f47a76a1d5c3a8081fc8b2077c84e1`
- Auditor: translation-auditor / es-w2-auditor-c, session 422494cdb264, model **kimi/k3.1-pro** (self-reported; fallback from anthropic/claude-sonnet-5-5 via openai/gpt-6-astra after quota HTTP 429 notices)
- Independence: no translation or repair authored; only this evidence directory written.

## Verdict reason

Read all 15 keys. 0 blocker, 1 major, 0 minor, 0 nit; any blocker or major fails the namespace. All parser-derived machine checks pass and every behavioural promise not named in a finding survives.

## What was verified rather than assumed

1. Read every English value, Spanish value and context entry side by side; judged accuracy, idiom, tú/infinitive register, behavioural promises, product-name policy and glossary consistency.
2. Used the repository runtime.py parser, not a regex or visual brace comparison, for message_arguments, plural_selectors and select-label signatures. Exact key sets and parser-derived argument tuples match; zero plural arguments occur.
3. Checked flags, backticked commands, choice tokens, identifiers, numeric literals and argparse printf tokens against the actual strings. No required source token was dropped or renamed.
4. No maxCells is declared in this namespace. Width was reviewed without a declared clipping budget; no rendered-fit or terminal-layout pass is claimed.
5. Spanish is LTR, so RTL-specific checks are not applicable. No value displayed as a redaction marker; the catalogue bytes themselves were read.
6. Verified cli.mobile_help.daemon_action select labels remain exactly start/stop/restart/other. Repository renderer output: start => 'Iniciar el demonio', stop => 'Detener el demonio', restart => 'Reiniciar el demonio', unknown pause => 'pause el demonio'.
7. Verified the 'gate' meaning in mobile/install.py: gate_closed probes /api/sessions and requires HTTP 401, i.e. an authentication gate. 'control de acceso' is accurate and is not a finding.
8. Derived daemon drift by enumerating tracked es JSON with git ls-files: committed relay.asks has three 'daemon' values; no committed es value contains 'demonio'. Sibling cli.browser_help also uses 'daemon'.

## Findings

### es-w2-C03 — MAJOR

- Keys: cli.mobile_help.install, cli.mobile_help.status, cli.mobile_help.daemon_action, cli.mobile_help.logs, cli.mobile_help.devices_port, cli.mobile_help.serve
- Exact current text and evidence: Glossary drift: this namespace uses 'demonio'/'demonios', while the committed, shipped es relay.asks catalogue uses 'daemon' in error.reach_failed, error.no_reason and error.read_timeout; the W2 sibling cli.browser_help also uses 'daemon'. Neither the style guide nor glossary authorizes a daemon/demonio split. Current assigned values: cli.mobile_help.install: Instalar el demonio, la contraseña y el LaunchAgent | cli.mobile_help.status: Estado del demonio, control de acceso y sesiones activas | cli.mobile_help.daemon_action: {action, select, start{Iniciar} stop{Detener} restart{Reiniciar} other{{action}}} el demonio | cli.mobile_help.logs: Mostrar el final de los registros del demonio y de los runtimes | cli.mobile_help.devices_port: Puerto del demonio móvil (por defecto, el del propio demonio) | cli.mobile_help.serve: Ejecutar el demonio (en primer plano)
- Proposed remediation: Standardize on 'daemon'/'daemons' in these keys (and coordinate the adjacent W2 browser keys so the surface stays consistent). Keep all surrounding grammar and ICU bodies otherwise unchanged.

## Mechanical result

Final checker: **0 mechanical failures**. Parser-derived argument, plural and select checks passed; required source tokens survived. Synthetic negative controls detected translated select labels, renamed arguments, dropped printf tokens and invalid Spanish plural categories.

No `maxCells` is declared; no rendered-fit test is claimed. No repo-wide tests were run, per delegation constraints.

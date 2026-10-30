# Spanish audit — cli.services_help — round 1 (w2a3)

- **Verdict: failed** — 0 blocker, 1 major, 0 minor, 0 nit
- Keys read: 9 English, 9 Spanish, 9 context entries
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d`; branch `feat/i18n-es-wire-runtime`
- Target is UNTRACKED; on-disk sha256: `008f3dfe5a93443c0de8d2ca0e07b208cb696693fc9850aa04ce00168c06009d`
- English sha256: `76aca39fb3bfe46be1d31fbfde2176ee525b7ac650961104894a894f71751935`; context sha256: `cc6104f97fbb0faac19232252063cdb7effaa2ffb4bbe2d4b65c2c749a8fe1e1`
- Auditor: translation-auditor / es-w2-auditor-c, session 422494cdb264, model **kimi/k3.1-pro** (self-reported; fallback from anthropic/claude-sonnet-5-5 via openai/gpt-6-astra after quota HTTP 429 notices)
- Independence: no translation or repair authored; only this evidence directory written.

## Verdict reason

Read all 9 keys. 0 blocker, 1 major, 0 minor, 0 nit; any blocker or major fails the namespace. All parser-derived machine checks pass and every behavioural promise not named in a finding survives.

## What was verified rather than assumed

1. Read every English value, Spanish value and context entry side by side; judged accuracy, idiom, tú/infinitive register, behavioural promises, product-name policy and glossary consistency.
2. Used the repository runtime.py parser, not a regex or visual brace comparison, for message_arguments, plural_selectors and select-label signatures. Exact key sets and parser-derived argument tuples match; zero plural arguments occur.
3. Checked flags, backticked commands, choice tokens, identifiers, numeric literals and argparse printf tokens against the actual strings. No required source token was dropped or renamed.
4. No maxCells is declared in this namespace. Width was reviewed without a declared clipping budget; no rendered-fit or terminal-layout pass is claimed.
5. Spanish is LTR, so RTL-specific checks are not applicable. No value displayed as a redaction marker; the catalogue bytes themselves were read.
6. Derived daemon drift by enumerating tracked es JSON with git ls-files: committed relay.asks has three 'daemon' values; no committed es value contains 'demonio'. Sibling cli.browser_help also uses 'daemon'.
7. Accepted 'compilación' for build: it matches the existing es catalogue and accurately distinguishes builds from versions; no build/compilación drift finding.

## Findings

### es-w2-C05 — MAJOR

- Keys: cli.services_help.services, cli.services_help.services_description, cli.services_help.status_description, cli.services_help.wait, cli.services_help.reclaim, cli.services_help.reclaim_description, cli.services_help.pid
- Exact current text and evidence: Glossary drift: this namespace uses 'demonio'/'demonios', while the committed, shipped es relay.asks catalogue uses 'daemon' in error.reach_failed, error.no_reason and error.read_timeout; the W2 sibling cli.browser_help also uses 'daemon'. Neither the style guide nor glossary authorizes a daemon/demonio split. Current assigned values: cli.services_help.services: Pasar a la compilación actual los servicios de esta máquina que no son runtimes (demonios serve y supervisados), sin detener ninguna conversación activa | cli.services_help.services_description: Todo lo que local_operator ejecuta en esta máquina y no es una conversación. Un “servicio” es un demonio `lop serve` o un LaunchAgent supervisado: el relé móvil, el puente del navegador, el túnel y el agente de despertares. Los runtimes —los procesos que contienen tus conversaciones— nunca se detienen: “restart” recarga un demonio serve sin cambiar de proceso, conservando su pid y su socket. | cli.services_help.status_description: Solo lectura. Indica la compilación de la instalación y, para cada demonio `lop serve`, la que está SIRVIENDO: ambos lados de cada comparación, porque la deriva habitual en esta máquina es una recompilación de la misma versión, y la versión por sí sola no revela que un demonio esté desactualizado. | cli.services_help.wait: Cuánto esperar a que un demonio solicitado vuelva con la nueva compilación (predeterminado: {wait_seconds}s; debe ser positivo). Si no vuelve, se informa de ello, sigue sirviendo la compilación que cargó y el próximo `lop services restart` lo reintenta. | cli.services_help.reclaim: Terminar un demonio `lop serve` registrado que no esté sirviendo su dirección (nunca uno que la esté sirviendo) | cli.services_help.reclaim_description: Pedir a UN demonio `lop serve`, identificado por pid, que termine, escalando de SIGTERM a SIGKILL tras un tiempo límite, después de comprobar que el proceso es el demonio serve de este producto y que no es el que sirve la dirección indicada en su registro. `lop services status` lista los demonios a los que se aplica. Nunca es automático: el demonio puede supervisar runtimes de sesión y quien lee un registro no puede comprobar que un sucesor esté listo para ocupar su lugar. | cli.services_help.pid: pid del demonio, exactamente como lo imprime `lop services status`
- Proposed remediation: Standardize on 'daemon'/'daemons' in these keys (and coordinate the adjacent W2 browser keys so the surface stays consistent). Keep all surrounding grammar and ICU bodies otherwise unchanged.

## Mechanical result

Final checker: **0 mechanical failures**. Parser-derived argument, plural and select checks passed; required source tokens survived. Synthetic negative controls detected translated select labels, renamed arguments, dropped printf tokens and invalid Spanish plural categories.

No `maxCells` is declared; no rendered-fit test is claimed. No repo-wide tests were run, per delegation constraints.

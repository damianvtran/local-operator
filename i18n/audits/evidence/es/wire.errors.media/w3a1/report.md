# Audit — es / wire.errors.media / w3a1

- **Verdict: passed (clean)** — 0 blocker, 0 major, 0 minor, 0 nit
- Revision: `66da866475f3e74553196a3167c18c6125141844`; target **untracked**; on-disk sha256 `5d31f81759f001777e8d1e0dbde8b5c60525acb0de1f5e886dd66e8fff21d3b8`
- Keys read: **20** en / 20 es
- Auditor: independent translation-auditor, session `c328736e07e9`; model **kimi/k3.1-pro** (self-reported temporary provider-quota fallback; primary anthropic/claude-sonnet-5-5).

## Verified, not assumed

Every key was read side by side with English. Missing Radient configuration remains missing, not rejected; `{vendor} rechazó tu clave de API. Sustitúyela.` remains a credential refusal with the same replacement remedy; insufficient credit still directs adding credit with the same provider; sign-in failure still directs signing in again in Ajustes. Temporary/current speech unavailability keeps “Inténtalo de nuevo en un momento,” while save/generation/transcription failures remain failures. Upstream no-response, reported-error and HTTP-response cases remain upstream failures with their distinct causes. Unknown provider/model and missing temporary audio remain absence/recognition errors, not permission or size/type errors. No refusal or failure reads as success and no remedy overpromises.

The real `local_operator.i18n.runtime` parser accepted all 20 targets. Ordered keys, all 15 source ICU arguments, plural selectors and select-label shapes match; flags, code spans, HTTP and product literals survive. `{provider}/{model}` and `{vendor}` render with synthetic values intact. No plural branches occur, hence no Spanish `many` case is needed and no `few`/`zero`/`two` appears. `media_refusal` is exactly `{error}` on both sides: correct pure passthrough, not a missing translation. Exact commands/output are in `audit.log`; the reproducible read-only probe is `checks.py.txt`.

No English context sidecar exists (`wire.errors.media.context.json` absent). Register was therefore checked against the shipped Spanish error copy named in the delegation, not an invented contract. Tú imperatives, sentence case, proveedor/modelo, credenciales/clave de API and Ajustes are consistent.

Length was measured rather than treated as a hidden budget. The largest growth is `temp_audio_missing`, 44 → 65 characters (1.477×); it remains concise and is not a clipping finding without `maxCells`.

## Findings

None. All 20 keys are clean.

## Reader limitation and scope

An unrelated zh-CN `speech_no_credential` value displayed as `[redacted]` during broad source searching; it was not judged from that display or used as evidence. The sanctioned `read_public_copy` route returned the tracked English source exactly: “Sign in to Radient in Settings to enable speaking aloud.” It cannot read these untracked Spanish targets or absent sidecars, and no workaround was used.

No catalogue or programme file was repaired. Only evidence was written. No native review, rendered visual fit, release eligibility or repo-wide test result is claimed.

# PR #4270 — design round-2 TUI frames (evidence only, not for merge)

Captured from PR head `8fedf4fcff10` by driving the real `OperatorApp` `/login bedrock`
path under `run_test(size=(110, 34))`, exported with `scripts.visual_capture.save_capture`
and rasterized with `rsvg-convert` (880x578, 8x17 px cells).

Isolation: `env -i PATH=<fake-bin>:/usr/bin:/bin`, `LOCAL_OPERATOR_PATH_PRIMED=1`,
`TERM=xterm-256color`, HOME/TMPDIR under `/tmp/bra2` (removed afterwards); no CMUX_*/LOP_*/AWS_*
inherited. AWS config/credentials/SSO cache are `tests/unit/providers/bedrock/awsfixtures.py`
fixtures (or absent / malformed per frame). The real `aws` CLI was unreachable; frame 01 runs a
fake `aws` shell script that prints the device URL + code and sleeps.

The Bedrock check is the real `bedrock.check.probe` (httpx + auth) with `LOP_BEDROCK_ENDPOINT_URL`
pointed at: an in-rig stub answering `401 UnrecognizedClientException` (04) or
`403 AccessDeniedException` (05) on `GET /inference-profiles`, and a closed loopback port (06).
`scripts/bedrock_fake_server.py` always answers that GET with 200, so it has no rejected/denied
mode for the control-plane probe.

| Frame | State |
|---|---|
| 01-sso-device-code.png | AWS SSO → expired `corp-sso` → Sign in now: device URL + user code shown, waiting |
| 02-no-aws-method-chooser.png | No `~/.aws`: method chooser with empty profile/SSO descriptions |
| 03-no-aws-sso-configure-hint.png | No `~/.aws`, picked AWS SSO: "Run aws configure sso" warning, chooser re-shown |
| 04-check-rejected.png | API key check rejected (401): login failed, nothing saved |
| 05-check-access-denied-saved-unverified.png | Check 403 (cannot list models): warning, saved unverified |
| 06-check-unverified-network.png | Endpoint unreachable: warning, saved without verification |
| 07-unreadable-aws-config.png | Malformed `~/.aws/config`: "could not be read" warning + chooser |
| 08-secret-access-key-masked.png | Access key flow: secret access key typed, masked |
| 09-session-token-prompt.png | Session token (optional) prompt, empty |
| 10-session-token-masked.png | Session token typed, masked |

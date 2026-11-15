# R10 native-layout preflight repair — source-only

Base 95c13d9ad8bf97809fd4eb6ea31f1873c60b990e. Review scope: base..new head,
R10 exact launcher layout and direct identity integration. This single commit
carries the fix and evidence; parent handoff supplies its exact SHA.
The prior execution GO was RELEASED. No admission/runtime/paid trial is authorized
by this patch; new independent review and NEW desk GO are required.

## Static facts and implementation

Outer /Applications/Docker.app: CFBundleIdentifier com.docker.docker,
CFBundleExecutable com.docker.backend, version 4.43.1. Nested Desktop.app:
identifier com.electron.dockerdesktop, executable Docker Desktop, same version.
The actual nested executable exists and local signature verification passed for
both bundles, using an Apple anchor plus Docker Team 9BNSXJN65R. The exact
commands/results, selected plist hashes and identities are recorded in
 desktop-layout-provenance.json. An initial static -R argument syntax error is
retained there; corrected inline requirement syntax starts with '='. No failure
was hidden, no Docker executable was invoked, and no bundle was modified.

The explicit nested path is the default. The old flat Docker and Docker Desktop
paths remain allowed ONLY when their own outer plist, executable and trusted
signature match. No discovery/fallback, symlink alias, arbitrary path or launchd/
open-based launcher is added. Canonical outer/inner/plist/executable paths are
required; escaping CFBundleExecutable values and mixed bundle versions fail.
Runtime preflight records this static layout provenance into its eventual receipt.

Common ownership logic already recognizes the outer bundle prefix, which includes
the nested UI. PID/start/exe comparisons and Popen handle ownership are unchanged.
The existing choose_helper requires a genuinely new recognized helper, a live
exact owner and descendant or exact approved raw-file-holder association. Tests
exercise the nested owner with that existing function and reject a preexisting
helper. Installed recognized virtualization/qemu-arm64 files exist statically.
No helper basename is added: com.docker.krun remains unqualified, not guessed in.

## Evidence, limits and preservation

R10-RED.txt records missing layout validation; R10-GREEN.txt records targeted
static-layout and direct lifecycle/auth/image/root/supervisor regressions.
Fixture binaries are inert text and never executed. Positive tests cover the
explicit nested and both flat layouts; negatives cover outer/inner plist escapes,
outer/inner/executable symlinks, arbitrary paths and untrusted signer metadata.
Actual installed bundles passed the SAME validation function via read-only
codesign calls, not Docker startup. Public signature metadata is not a historical
VM cap, readiness, ownership or native-success receipt.

The original stopped preflight stays unchanged in registered HOLD, SHA256
 dbbf242301eee3c777d9721e7d459d4da72b3e1725476eef66e900f83064ecf2.
layout-preflight-stop-public.json selects technical fields only; home paths and
private process/configuration data are omitted. Existing source observations,
wheel and receipts are preserved; no raw private payload is committed.

Remaining native gates: UI entry must actually bring up a daemon within the
reviewed bounded readiness path; owner must stay attributable/live; selected VM
runner must satisfy unchanged helper association; info/RSS/swap/disk/container
caps and exact container/image/owned-Desktop cleanup need real proof. A nested
path does NOT prove backend activation or safe reparenting. Failures stop without
live patches, alternative helper guesses or automatic retries.

# Agent review remediation — R6/R7, one coordinated source-only pass

Reviewed base: 1163494896d65281b17c8deae1596e9c3da207ac.
Reviewer: full-scope review session 401077fb590a; findings relayed by parent.
Commit identity: this note ships in the unified remediation; parent handoff gives
one exact SHA for BOTH IDs. Remediation review scope is reviewed base..new head.
Neither finding is rejected or deferred. Optional pull is NOT signed off: NOPULL
remains the current disposition until a new explicit clean review/desk authority.

## R6 MAJOR — fixed in this commit

Actual NativeOps.stop_owned_resolver and command() previously re-entered the
expired WORK deadline while cleanup had a valid independent budget. The RED
reproduction mocks the adapter's child and CLI transport, not the stop method:
SIGTERM/SIGKILL occurred and ZERO CID recovery calls followed the monitor error.

Cleanup now has separate physical-resource checks and its existing independent
60 s envelope. Work-monitor callbacks cannot cancel a cleanup CLI operation.
The stop path catches and records monitor/client failures, ALWAYS enters exact
known-CID recovery in finally, attempts absence verification even when remove
transport fails, and raises an unverified/failure result after those attempts.
No exception is treated as successful retirement. Missing/invalid CID never
causes broad cleanup. An exhausted cleanup budget cannot spawn further commands.

Before granting client grace, the envelope reserves 5 s for reap, two existing
15 s CID calls and 15 s Desktop retirement; this is budget allocation, not a
measured latency claim. If those operations cannot finish in the remaining budget,
receipt/exit remain unverified. Source tests exercise the actual adapter methods
for expired work/valid cleanup, adequate mocked disk, monitor failure, removal
failure, daemon-unavailable response, invalid/absent CID and exhausted budget.

## R7 MAJOR — fixed in this commit, source proof not native credential proof

Docker CLI source is pinned at
`fdd4b8e5da335e1d0d83d29af5045f57280cd694`; hashes/URLs are in the new append-only
metadata receipt referenced by docker-anonymous-source.json. Relevant paths:

* cli/config/config.go: LoadDefaultConfigFile calls DetectDefaultStore when
  !ContainsAuth(). An empty directory or `auths:{}` is NOT sufficient.
* cli/config/configfile/file.go: ContainsAuth is true when the auth map has at
  least one entry. GetCredentialsStore selects file storage absent configured
  helpers/stores; DOCKER_AUTH_CONFIG can override it and must not be inherited.
* cli/config/credentials/default_store_darwin.go: default is osxkeychain.
* cli/config/credentials/file_store.go: an empty auth record returns no secret;
  an unmatched registry also returns empty AuthConfig rather than invoking a helper.

Both owned CLI configurations now explicitly contain:

```json
{"auths":{"https://index.docker.io/v1/":{}},"credsStore":"","credHelpers":{}}
```

The NON-empty map suppresses detection without supplying credentials. CLI and
resolver-child environments are constructed from a whitelist, not copied from
the operator environment; auth/proxy/token overrides are excluded. Git metadata
queries additionally disable system/global credential configuration. Desktop
itself retains its normal environment: this is a CLI mitigation, not a claim
that opaque Desktop application internals never use platform APIs.

The regression creates ONLY a test-owned executable named like the macOS helper.
A Python driver following the pinned public source selection rule invokes that
sentinel for old empty config (negative control), but actual adapter config plus
actual NativeOps.command does NOT invoke it and excludes a synthetic auth-env
sentinel. It never runs Docker, the actual osxkeychain helper, Go compilation or
any keychain API. This is NOT presented as a native Docker CLI smoke test.

## Evidence and unchanged gates

R6-R7-RED.txt preserves the original failures (home prefix redacted, terminal
trailing whitespace normalized). R6-R7-GREEN.txt records current bounded tests,
the real default entrypoint's refusal to run and static verifier/refusal output.
R6-R7-CHECK-ATTEMPT1.txt preserves a recording-shell timeout AFTER all 26 tests
passed; only the interrupted incomplete-gate command was repeated, not the tests.
Its fresh expected exit 1 is appended to GREEN. No production lease was changed.
The validated 17,389,450-byte wheel and its staging receipt are unchanged; no
artifact was installed/executed or downloaded again. All 26 media inputs remain
unacquired. No Desktop/Docker/start/pull/build/resolver/model/credential operation
occurred; only source/public metadata reads and owned Python test subprocesses.
No fresh slot or GO is inferred, and no freeze/runtime qualification is claimed.

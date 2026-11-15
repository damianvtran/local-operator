# Prospective meter privacy preservation — source-only candidate

Base: `9dee883473441bc1db300abb975cd11297c6ef01`. Scope is ONLY
`experimental/meter_proxy.py`, its focused ASGI regression test, and prospective
source/evidence metadata. No account, routing default, model, source dataset,
rate card, historical receipt, runtime lifecycle or native resource policy was
changed. The prior data-flow audit describes its pinned old source accurately;
this candidate fixes its concrete provider-field overwrite, not its remaining
qualification gaps.

## Correctness boundary

The actual request guard validates a provider object before key lookup, body
capture, reservation or upstream dispatch, and revalidates the final prepared
body. Supported explicit fields are preserved verbatim:

- `zdr`: exact JSON boolean `true` OR `false`; no truthiness coercion.
- `data_collection`: exact string `allow` OR `deny`; no case/unknown fallback.

Missing fields remain missing. No default ZDR or no-training policy is imposed.
Explicit null/non-object preferences, malformed types/values and unsupported
provider fields fail closed with a generic refusal. Unsupported filters such as
`only` are rejected rather than silently discarded: the benchmark cannot promise
those caller constraints while owning the destination. Previously tolerated
benchmark-owned `order`, `allow_fallbacks` and `max_price` inputs remain ignored
and replaced with authoritative Anthropic-only/no-fallback/priced bounds. They
cannot weaken the final route, fallback prohibition or price ceiling. Other
request shaping and pricing rules are unchanged.

This does not assert that `zdr=false` disables an account-enforced policy, or that
`zdr=true` finds an eligible endpoint. A provider refusal must remain a refusal;
no flag stripping/retry or fallback has been added. Current first-party
Anthropic/ZDR compatibility, account settings, Radient decisions/recommendations,
MCP destinations and financial17 exposure remain separately unqualified.

## Actual-path evidence, not an assignment-only probe

`experimental/test_meter_privacy.py` reuses the established AdapterTests fixture
helpers without inheriting/collecting their unrelated test suite. Requests enter
the real Starlette ASGI app and run its real guard/proxy/preparation/settlement
path. Only outbound HTTP transport is replaced by `httpx.MockTransport`; all
keys/prices/responses are synthetic fixture data. No sockets, external HTTP,
credential access, app/VM/helper startup, install, model or paid call occurs.

RED (`PROVIDER-PRIVACY-RED-v1.txt`): an actual prepared upstream request lost the
explicit `zdr=true`, yielding `None is not True`. GREEN
(`PROVIDER-PRIVACY-GREEN-v1.txt`): six focused tests pass in normal and `-O`
existing repo Python3.12.13, covering:

- Both strict flags reaching the prepared upstream body.
- Individual true/false and allow/deny flags plus mixed combinations, with no
  missing-field default insertion.
- Authoritative route/no-fallback/max-price despite conflicting caller values.
- Exact no-policy baseline provider object.
- Sixteen malformed/unknown preference cases: local402, ZERO upstream calls.
- Existing tier/output/modality/server-tool/unknown-extension refusals, also
  ZERO upstream calls. No downstream privacy policy is claimed from these tests.

Commands (run from `benchmarks/harness_cost/experimental`):

```sh
~/local-operator/.venv/bin/python -B -m unittest -v test_meter_privacy
~/local-operator/.venv/bin/python -O -B -m unittest -v test_meter_privacy
```

The retained RED traceback uses `~` instead of an operator home path; this is
privacy sanitization of source-only test evidence, not alteration of paid or
historical receipts. Source-only/macOS ASGI proof is NOT Linux runtime evidence.

## Truthful generation and historical drift

`PROSPECTIVE-METER-PRIVACY-v1.json` pins the old historical meter digest, the new
meter/test digests, previous source head and unchanged owned-launch/resolver
file digests. The old meter bytes were read from that Git object and independently
matched the retained historical binding. No historical expected hash was edited.
The reconstruction SHA256 manifest includes this NEW generation and its evidence.

Consequently `verify_static.py` correctly exits1 with EXACTLY:

`historical source changed: experimental/coding/../meter_proxy.py`

The full actual result is retained as `PROVIDER-PRIVACY-STATIC-v1.json`; its
privacy indicators are empty and all other historical/source checks remain
unchanged. This is an intentional, explicit CURRENT-vs-HISTORICAL mismatch,
NOT a green historical check. The historical verifier and its recorded expected
hashes were not weakened to accept a changed source file. Prior kit manifests,
preregistration/closure hashes, account attestations and rate cards are NOT
rewritten or re-labeled current. Do not package/build/dispatch this source under
an old reviewed manifest or frozen runtime attestation.

## Explicit pending admission-validator gap

This focused commit does NOT add a build/preflight generation validator or wire
future preflight to select the new generation. The JSON is a prospective source
receipt, NOT an executable admission gate. The default OLD generation is
INELIGIBLE for future admission; automated selection/enforcement is still a
blocking follow-on, not claimed implemented here. Counting one expected drift
in a diagnostic is NOT a substitute for that validator.

Before any future build/admission, a separately reviewed fail-closed path must
explicitly select the new generation, check new meter AND test hashes, verify the
exact old Git object, and verify the unchanged remainder against the approved
base/manifest. ONLY the declared product-source delta (meter + new test) is
permitted. Regression tampering of the new meter, new tests, unrelated runtime
bytes and restricted unchanged source must EACH reject. Wire this validator
into actual build/preflight, not an ad-hoc wrapper suppressing a historical
error. This extends build admission and is deliberately deferred rather than
expanding the narrow meter correctness patch or inventing readiness.

Required next gate: independent delta review, prospective generation/closure
binding to the resulting committed source, and fresh applicable build/wire/runtime
qualification before any future paid freeze/dispatch. Existing preflight must
continue rejecting old closure hashes against these changed bytes. No old
runtime qualification is reusable (this reconstruction was never qualified).
R16 launch/identity bytes and resolver pins remain unchanged; native budget2/3,
operator resource ask `a-9509` / relay `a-1ca7` OPEN. No fresh execution GO exists.

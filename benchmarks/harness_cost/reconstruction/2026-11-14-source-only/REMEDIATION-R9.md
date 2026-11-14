# R9 preflight-gap remediation — source-only unified image/swap amendment

Base: 6ea66a73f5766db25ce47c0dafe804053005cb89 (round 8 clean).
R9 labels the execution-preflight gap, not an already executed native attempt.
This note ships in the one unified remediation; parent handoff names its SHA.
Review scope is base..new head, R9 and direct image/swap/lifecycle integrations.
ALL predecessor execution GO/slots/signoff expire on this patch. New independent
review and NEW desk GO are required; no admission flags are fabricated/reused.

## Exact ownership, ordered retirement and truthful failure

The actual NativeOps adapter snapshots full image/config IDs, layer DiffIDs,
build-cache IDs/activity and container IDs/image use before any optional pull.
CLI templates select metadata, never image/container Config.Env. Baseline is
immutable. The only pull is the already-pinned linux/amd64 carrier, once; no
Docker build exists, so no new build-cache record IDs are expected. The receipt
compares before/after IDs; this is not a claim that no other actor ever built.

Cleanup order in actual run_lifecycle is:

1. Recover the exact owned resolver/container with the existing bounded policy.
2. Attempt exact image retirement, including after a container-cleanup failure.
3. Attempt owned Desktop quit only when existing ownership/foreign-work checks
   and the new asset-conflict checks permit it.
4. Record asset snapshots/claim/result/errors plus final resources; mandatory
   failures stay failed/unverified and propagate, never resolved success.

Only a carrier content ID absent from baseline, matching the pinned reference
and platform, with no other new images/tags/references/container users, can be
claimed. Removal is `image rm <exact-ID>`: NO force, prune, layer/cache removal,
shared-daemon takeover or deletion of preexisting IDs. Shared layer DiffIDs are
recorded and preserved through baseline image references. Logical ID retirement
is NOT physical Docker.raw shrinkage or a full content-store garbage collection
claim. Unreferenced compressed blobs are not exhaustively enumerable here.

Preexisting carrier IDs are preserved even if a digest pull adds a reference.
Failed pulls that nevertheless materialize a proven complete new carrier can
retire that exact ID. Failed partial pulls without a proven image identity leave
partial-blob retirement explicitly UNVERIFIED: no broad cleanup is invented.
Unexpected cache/new images, active cache/build work, foreign containers or new
references preserve the unowned objects and prevent unsafe Desktop shutdown.
Unsupported/incomplete inventories also fail closed; they never authorize pull.
After a removal transport failure the adapter still attempts bounded after-state
inspection, but a mandatory command failure is NOT success. Existing cleanup
budget remains authoritative; exhaustion/unknown ownership can leave cleanup
unverified. There is no retry/alternate deletion path or new cleanup timeout.

Native ownership receipts can reveal private image references: keep future raw
receipts in registered HOLD, do not auto-commit them. Only inert test inventories
are published now. The preexisting host/cache/other work is never pruned.

## Swap guard and units

Desk's new limit is exactly 12,000,000,000 bytes (decimal), NOT 12 GiB.
`sysctl vm.swapusage` K/M/G values are interpreted as binary units using Decimal.
The parser ceilings the displayed conversion; the guard additionally includes
half the last displayed unit to conservatively cover text-format rounding.
Unknown/missing units fail closed, never default to zero. PRE-startup, during
work/cleanup and final samples retain raw text, nominal byte ceiling, precision
upper bound, limit and phase. Final failed sampling/over-limit evidence is written
into the receipt before raising failure. The existing command/cleanup bounds are
reused, not relaxed or replaced by a new arbitrary sysctl timeout.

The original stopped preflight's 11738.12M is already over 12 GB decimal. Its
historical truncated-byte conversion is retained unchanged; R9 uses ceiling and
formatted-precision bounds, not a rewritten observation. A desk may clarify units
only in a future explicit admission; no silent reinterpretation weakens the guard.
Original HOLD receipt remains byte-identical, SHA256
013d4ddb41f383cf810d8d2e0dab01617cbfdf2e90f6c8e0c7e0dc53c123a007.
preflight-stop-public.json selects technical fields only (no home paths, process
identities, credentials or private payloads) and binds the original checksum.

## Static API proof and source-only regressions

Public Docker CLI source is pinned at
fdd4b8e5da335e1d0d83d29af5045f57280cd694. New immutable raw-source hashes/URLs are
in metadata-receipt-20261114T204051175032Z.jsonl, retaining the existing license.
`cli/command/system/df.go` and `formatter/disk_usage.go` expose verbose BuildCache
contexts to non-table templates without ID truncation. `formatter/buildcache.go`
exposes InUse as a string and marks active IDs with '*'; the adapter validates and
records that state, refusing active foreign builders before pull/deletion/quit.
Unsupported native CLI versions will stop, not fall back to weaker inventories.

R9-RED.txt records missing cleanup-before-quit and swap contracts at baseline
(home prefixes redacted; terminal trailing whitespace normalized).
R9-GREEN.txt records targeted tests against actual NativeOps snapshot, pull,
image cleanup, quit and receipt methods using inert CLI transports. The actual
lifecycle ordering test executes actual adapter retirement/quit methods, not just
an events-only fake. Cases cover preexisting/shared images, materialized and
unmaterialized partial pulls, foreign users/cache/tags/images, exhausted budgets,
cleanup failures, signal-style unwinding, swap conversion/precision and final
failed-resource receipts. Existing direct R6/R7/R8 regression files are retained.

No Desktop/Docker CLI/actual credential helper/keychain, native resolver, model,
install, build or artifact extraction ran. Tests use inert transports/source
fixtures; they are NOT native image/swap/VM qualification. The validated wheel
and original preflight receipt remain untouched. No all-arm runtime readiness,
freeze, previous trial recovery or paid success is claimed.

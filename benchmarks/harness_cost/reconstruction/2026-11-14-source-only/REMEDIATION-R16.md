# R16 — stable process identity, not ordinary scheduler state

Base d68f2cf76e1e21fec6d75e15ebd270b5a7c1244a. Round14 MAJOR R16 is fixed
in one focused source-only commit. Retest R16; scope base..new head. Exact SHA
is in the handoff. No launcher/ownership/resource-policy refactor is included.

## Change

DarwinIdentity.record compares PID, parent/group/UID, microsecond birth and
executable path across the snapshot closure. Ordinary scheduler status changes
(SRUN2 <-> SSLEEP3, stop/resume) and runtime flags/nice/file-count changes are NOT
PID replacement. The returned state reflects the second read's classification.

Exit/zombie classification remains explicit rather than discarded with status:
- live <-> zombie transitions fail closed for ordinary records;
- the held creator's supported live -> zombie transition remains bound to its
  exact PID/birth and retained creation authority;
- an already-zombie held creator must remain zombie with the same PID/birth;
  an impossible resurrection/replacement is rejected;
- SID is read again after the closure and must match, so a real session change
  does not get mistaken for harmless scheduler noise.

No unknown-history reset, signature bypass, ancestor/session adoption, cap change,
or new launch behavior is introduced. NativeOps and resolver-pins bytes are
unchanged from the prior head, including 4.3GB/2CPU gates and UI acknowledgement.

## Evidence

R16-RED.txt reproduces the reported error through the ACTUAL
NativeOps.observe_owned -> snapshot -> DarwinIdentity.record dispatch with inert
kernel/process transports. Identical stable identity and only 2/3 scheduler
variation triggered 'process identity changed during snapshot closure'. Home
prefix is redacted; no failure evidence is rewritten as passing.

R16-GREEN.txt records 30 targeted tests: actual adapter population for 2->3,
3->2, 2->4, 4->3 and runtime flag changes; actual adapter plus direct-record
negative controls for PID/birth/path/SID changes and exit classification; retained
creator-zombie hold/reap and direct-session lifecycle regressions. Tests do NOT
preset successful identity results. All app/kernel transports are inert; no new
binding/native probe or app/helper/VM/Docker startup occurred.

Source-only static/privacy verification and proposal-only command are included
in the same GREEN artifact. Prior native attempt1/2 receipts remain immutable.
Native budget stays 2/3 consumed; optional pull NO and no attempt3 GO.

Operator resource-policy ask a-9509 (relay a-1ca7) remains PENDING with the parent.
This commit neither answers it nor changes the existing 4.3GB/2CPU policy to
8.2GB/14CPU. Fresh review + the actual policy decision + NEW GO are separate gates.

# Agent review remediation — R4/R5, one coordinated source-only pass

Reviewed base: 6423561e6cf24ea19ce342b815a3532462d0f541.
Findings relayed by parent. Commit identity: this note ships in the unified
remediation commit; parent handoff supplies its exact SHA for BOTH IDs.
Independent remediation review remains required. No runtime/VM success claimed.

* R4 MAJOR — fixed in this commit. SIGTERM/SIGINT raise a termination exception
  through finally; further signals are ignored during bounded owned cleanup.
  Signal delivery is blocked around Popen ownership assignment to close that
  race. Only the recorded client process group and recorded container ID are
  stopped; daemon-unavailable is not mistaken for verified container absence.
  Independent PID1 alarm at 840 s bounds container life after host/client loss;
  it kills the resolver's process group and reaps with os.waitpid, deliberately
  NOT reentrant Popen.wait from a handler interrupting that wait (stub exercise
  caught that deadlock before dispatch). The outer lease remains 900 s.
* R5 MAJOR — fixed in this commit. Both outer and inner scripts explicitly reject
  sys.flags.optimize/PYTHONOPTIMIZE before effects. Mandatory existing assertions
  therefore cannot be stripped into a permissive execution. Normal, -O, -OO and
  PYTHONOPTIMIZE controls are exercised by source-only subprocess tests.

R4-R5-RED.txt records the original leaked stub client, missing independent alarm
and optimization bypass. Tests stop/reap only exact owned PIDs, including failed
reproduction cleanup. R4-R5-GREEN.txt records the remediated stub signal/timeout,
independent alarm, optimization and disk-fence checks. No Docker command is used
by these tests; fake container rm/inspect commands operate only on scratch files.

Desk constraints were batched into the same packet: conditional 17:00–18:15 UTC
slot on 2026-11-14, NO GO now; fresh >=30 GiB + fleet <=26,000 MB + memory GO;
VM caps <=4 GB /2 CPUs verified BEFORE start; no alternate blocked-settings read
or config write. Allocated Docker.raw blocks*512 and filesystem free bytes are
recorded baseline/during/after; >5 GiB growth or free <25 GiB stops owned work.
Root HOLD is registered but does not itself authorize deletion or runtime work.
VM lifecycle belongs to the separately admitted session-start owner, who must
quit only Docker it started without harming other work; resolver never starts
or quits a shared daemon. Settings-cap evidence remains inaccessible/unresolved.

RED evidence is redacted only to replace the operator home prefix with ~.
Only source and own stub processes ran. No Docker/VM start, settings access,
network runtime, resolver, install, build, service, paid/model call or credential
access occurred. The 26 media inputs remain unacquired. No freeze or admission.

# Bounded runner execution and cleanup are different contracts

`cleanup-required` remains the default. A backend unable to construct authoritative
coverage refuses with 125 **before spawning**. This implementation has no
Darwin authoritative constructor; observing a complete-looking PID list is not
coverage. Linux keeps the strict `/proc` backend. This is a loss of Darwin strict
gate availability, not a claim that those gates have been cleaned up.

`--contract=command-status` requests the command's normalized status. It never
certifies group cleanup (`cleanup_status` is always `unknown`, `admit_cleanup`
rejects every command-status receipt), but it does **enforce** a positive memory
grant, inherited or auto, with the same `memory_guard.Guard(pgid, budget)` the
strict runner and the bash tool use. A grant still admits only tightening:
zero/widening refuse before spawn (`memory-policy-refused`).

## Reserved-leader signal authority (architect ruling, option B)

The child runs with `start_new_session=True`, so its group id equals its pid.
While that leader is **unreaped** the kernel cannot reuse the number, so
`killpg(proc.pid)` reaches only our group. The supervisor therefore observes the
leader without consuming it — `os.waitid(WNOWAIT)` where CPython exposes it
(`terminal_observer=waitid-wnowait`), else Darwin's `sysctl(KERN_PROC_PID)`
zombie flag (`darwin-sysctl-zombie`, which is how Python 3.12 on macOS runs) —
and every signal (deadline TERM/KILL, forwarded signal, memory KILL, leader-exit
sweep) precedes the single consuming `os.waitpid`. None follows it. `Popen.poll`
is never used because it maps ECHILD to status 0. An unreadable observer is
never terminal: after 2 s the group is SIGKILLed, reaped, and the run reports
126 `supervision-failure`. The leader-exit sweep consults the fork-free group
listing only to decide whether to signal (unknown means signal); it is never
cleanup evidence. Darwin answers EPERM rather than ESRCH to `killpg` when the
zombie leader is the group's only member; that reading is accepted only when the
group listing agrees.

Refusals that remain: `memory-policy-refused` and `unsupported-boundary` (no
`os.killpg`, no non-consuming observer, or a non-default SIGCHLD disposition).
Escaped (`setsid`) descendants, uninterruptible sleep and scheduler delay still
rule out hard wall-time or RAM promises.

## Runtime capability and real-path coverage

`test_run_bounded_contract_cli.py` executes the CLI with isolated HOME/config,
checks actual completion JSON and verifies pre-spawn side effects are absent.
Child-outcome cells run on Python 3.12 and 3.13 (the observer differs) and
check real 0, 7 and SIGTERM child outcomes plus actual budget, NODE_OPTIONS and
notification-gate propagation, a real memory breach killing an allocating
grandchild (137), and a leader-exit sweep of a left-behind sleeper. Existing CI's Ubuntu
`test` matrix includes Python 3.13 on main (3.12 on pull requests); the normal
unit-test shard selection includes these tests. The 3.12/3.13/3.14
`packaging-smoke` matrix is **not** a unit-test/CLI execution claim.

## Proof-dependent admission

The completion receipt is a fresh `run-bounded-completion/v1` sibling of the
legacy identity records. Its `cleanup_scope` is always `owned-process-group`;
`descendant_cleanup` is always `unknown`. Only the exact terminal verdict accepted
by an authoritative owner before irreversible revocation can set
`cleanup_status=proven-empty`. A wait, signal, ESRCH, empty residual list or zero
child status cannot supply that proof. Direct-only mode forcibly retains unknown.

`admit_cleanup` requires the caller's expected fresh invocation challenge and
publication directory, strict contract, published final path, proof and successful
command/wrapper statuses. The filesystem is trusted and has no competing writer;
this is not authenticated publication against hostile writers. Pending inodes
are retained. Publication uses fsync and non-overwriting hard links but promises
neither directory crash durability nor bounded filesystem latency. These limits
are distinct from the bounded legacy best-effort identity writer.

## Source integration safeguards

Delivery is integrated Python, not an executable JSON patch manifest. The POC's
`integration_notes[0]` described an environment policy "move". Correction: this was
**two independent text edits** (define a helper, replace the inline block), with
nothing inherently linking them. The historical scratch manifest is preserved as
provenance and is not a build input.

The ordered exact-anchor-once applier is replaced by a tested integrated-source
invariant: exactly one `_child_environment` definition, both execution paths call
it, and NODE_OPTIONS policy exists nowhere else. Real behavior tests exercise
resolved grants, the 6144 MiB heap floor, last-wins heap flags, quoted options,
unrelated flags and notification defaults without stubbing the builder.

The existing `flake8 .` CI gate and scoped `make check-changed` lint include F821.
The contract suite also runs Pyflakes over the integrated runner, contract,
ownership module and contract tests. Its negative control removes the builder's
name and requires **both** unresolved call sites to be detected. A manifest note
is not used as evidence that a definition exists.

## Consumer audit (2026-11-02, repository source only)

Searches cover the repository's Python (including files over 1 MB), shell,
workflow, Makefile and documentation for identity/completion schemas, record
paths, verdicts and wrapper invocation. Identified readers:

| Reader | What it currently consumes | Cleanup-admission limitation |
| --- | --- | --- |
| `tests/unit/scripts/test_run_bounded_run_record.py` | spawn/exit schema, status, actors, verdict | Legacy identity tests, not production resource release |
| `tests/unit/scripts/test_run_bounded_linux_spawn_load.py` | run ID, exit code, leader/PGID identity | No fresh completion admission |
| `tests/unit/scripts/test_run_bounded_incident_shape.py` | reservation/proof actors plus PID checks | No fresh completion admission; does not prove incident cause |
| `tests/unit/scripts/test_owned_process_group_reservation.py` | proof actors/residuals in exit identity record | Synthetic assertions, not a production release policy |
| `tests/manual/run_bounded_owned_canary.py` | VERIFIED and child-ready identity barriers | Test protocol, not a terminal cleanup certificate |
| `tests/unit/scripts/test_run_bounded_contract.py` | completion receipt and `admit_cleanup` | Explicit positive/negative admission controls |
| `tests/unit/scripts/test_run_bounded_contract_cli.py` | actual CLI completion JSON | Command/cleanup separation and grant refusal |

Known status-only callers — Makefile's type-check gate, `scripts/ci_scope.py`'s
`JOB_COMMANDS["type-check"]`, the AGENTS.md gate spellings and
`scripts/bench_send_admission_ab.sh` — explicitly pass `--contract=command-status`:
they consume a return code, not a cleanup certificate, and the default refuses on
Darwin before spawn. CI's serial-control wrapper (`ci.yml`) stays on the strict
default because it runs on Linux, where the strict backend exists.
`test_ci_hygiene.py` pins those spellings. They do not become proof-dependent
consumers merely because they inspect a return code. No production call to `admit_cleanup` was found outside the new module/tests.

**Unidentifiable:** the three organizational/downstream lanes claimed by the old
`_ownership_record` comment are unnamed. Lane 1, lane 2 and lane 3 cannot be mapped
to source consumers or admission rules. The source comment is corrected rather
than presenting this reader inventory as those three lanes. No external-session,
external-repository or production cleanup-consumer coverage is claimed.

## Non-claims and merge readiness

This does not solve or diagnose the twelve-runtime incident. Escaped descendants,
hard RAM limits, hard wall-time bounds and a persistent cleanup owner remain out
of scope. Supported backends may still lose authority after spawn and return
unknown cleanup while a child or other group member survives. No caller may
interpret that as clean.

Legacy tests contain assumptions invalidated by pre-spawn refusal and independent
baseline defects. Their exact failure classification and executed same-code
controls must accompany the implementation handoff; passing only the new tests
is not a clean merge gate. Native Linux coverage and independent review/QA remain
required before merge.

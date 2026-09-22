# Process memory guard: a per-command RAM ceiling for `lop`

Status: design + interface contract, 2026-09-21. This file is the contract the
implementation slices code against. Landing it in the same change as the code
keeps the contract reviewable. No version bump rides this doc (standing rule:
`pyproject.toml` stays at the last released version on every branch).

## 1. Why

An agent session runs a shell command — a build, a `pip install`, a job with a
mis-sized batch — and it allocates until the *device* runs out of memory. The
kernel then does not politely fail the command; it picks a victim by its own
heuristic, and on this host the victim is often the `lop` runtime or another
session's process, taking the whole desktop down. The agent never learns it was
the cause: the command was OOM-killed by someone else, or the session died with
it, and the transcript ends with no attributable failure. The retry is identical,
because nothing told the model the command itself was too big.

The fix the operator asked for is the k8s/Docker shape, mapped onto one shell
command: account for the device's memory, give each command a ceiling, and when
the command crosses it **kill the command's whole process group — never the
runtime** — so the tool result can say "your command exceeded the memory budget"
and the model can revise to something that fits. On a 32 GB device that stopgap
is the difference between a slow command and a dead session.

**The OS gives no signal to wait on.** Measured while building this (2026-09-21):
the machine death that motivated the guard left **no jetsam event** in the unified
log — the newest entries `log show` returns are from the day before. So nothing at
the OS layer announces memory trouble a process could react to, which is why the
guard **measures RSS itself** every tick rather than waiting for a kernel
notification that never comes. The design below is built on that measurement, not
on an assumed OS signal.

What already exists, and is reused rather than rebuilt:

- Every bash command is spawned into its **own session and process group**
  (`tools/builtin.py:2728`, `start_new_session=True` via
  `procstate.detached_popen_kwargs()`), and `_kill()` already stops that group
  atomically through `procstate.terminate_process_tree(pid, force=True)`
  (`tools/builtin.py:2925-2935`, `procstate.py:338-400`). A group is exactly the
  unit `memory.oom.group=1` defines, and the kill primitive is already correct.
- The foreground poll loop (`tools/builtin.py:3189-3218`) already wakes at most
  every 250 ms and already owns the timeout and abort branches. A memory sample
  slots in beside the 500 ms `_emit_update()` tick with no new loop.
- A stdlib-only, **no-psutil** memory reader already exists
  (`mobile/resources.py`): `ri_phys_footprint` via `proc_pid_rusage` on macOS
  (`_darwin_footprint_bytes`, L221), `smaps_rollup` Pss on Linux
  (`_linux_pss_bytes`, L291), and a batched `ps` RSS reader (`_parse_ps_rss`,
  L129).
- The "reserve" concept — take a fraction of *available* memory, hold a floor out
  of it so the OS and the other sessions survive — is already implemented and
  titrated in `conftest.py:1156-1212`. This design follows the same pressure-aware
  shape, but keeps command-specific constants so worker-pool tuning cannot silently
  alter a live command's budget.

`psutil` remains deliberately **not** a dependency. Nothing here adds it.

## 2. Scope of this change

In scope:

1. `local_operator/memory_guard.py` — a stdlib-only budget calculator and a
   per-group sampler.
2. The guard's enforcement point in `execute_bash`'s foreground poll loop, and in
   the background-job runner (`_detach_to_job`'s loop, `tools/builtin.py:3059-3067`).
3. Config keys, their `/settings` registry rows, and the consumer-default wiring.
4. The tool-result contract the model reads.

Out of scope (deliberately, and recorded so nobody assumes otherwise):

- **Hard throttling.** The `memory.high` throttle is a kernel reclaim behavior; in
  userspace we cannot stall an allocation, so the soft threshold is an
  *advisory*, not a slowdown. Stated in the docstring, not implied.
- **The eval kernel** (`tools/eval.py:609`), the detached `exec_worker`
  (`exec_mode.py:514`), `resume_click`, the secrets client, and `mcp/auth`. All
  spawn detached groups and all are candidates, but the eval worker in particular
  is a **known heavy corner** (`AGENTS.md` §Environment) whose process group is
  managed by `_close_kernel` on a different lifecycle; folding it in here would
  widen an already structural change. §9 gives the phase-2 seam, which is the
  same `MemoryGuard` class — this doc is what makes that a one-line addition
  rather than a redesign.
- **Windows.** `os.getpgid`/`killpg` do not exist there and `group_reaper` already
  early-returns a no-op on win32 (`group_reaper.py` docstring). The guard degrades
  to disabled on Windows, loudly once, for the same reason.

## 3. Budget model

### 3.1 What is read, platform by platform, stdlib only

| quantity | macOS | Linux |
|---|---|---|
| physical RAM | `os.sysconf("SC_PHYS_PAGES") * SC_PAGE_SIZE` | same |
| **available** (the budget's base) | `vm_stat`: `free + speculative + file-backed`, × page size | `/proc/meminfo` `MemAvailable:` |
| swap total (report + pressure floor) | `sysctl -n vm.swapusage` → `total` | `/proc/meminfo` `SwapTotal:` |
| per-pid honest footprint | `proc_pid_rusage` `ri_phys_footprint` | `smaps_rollup` `Pss` |
| group RSS (cheap aggregate) | `ps -axo pid=,pgid=,rss=` | same |

The physical and available arms are copied verbatim from the probes `conftest.py`
already ships and has been burned by: `_total_memory_mb` (`conftest.py:1006`) and
`_available_memory_mb` (`conftest.py:1027`). Two of those details are load-bearing
and are restated here so they are not re-derived wrongly:

- **macOS available is `free + speculative + file-backed`, not `inactive`.** The
  conftest measurement (`conftest.py:1040-1049`) is the reason: counting
  `inactive` reported 8,137 MB of headroom at the moment the host had 452 MB free
  and 6.1 of 7.2 GB of swap consumed. `file-backed` is the subset `vm_stat` itself
  identifies as clean and droppable, so it needs no invented discount fraction.
- **macOS swap `used` is cumulative and is NOT a pressure term**
  (`conftest.py:1050-1061`): it stays in the swap file until faulted back or
  reboot, so it reads "this host swapped since boot", not "this host is swapping
  now". We read swap `total` (stable) and swap `free` (instantaneous, recovers),
  and never `used`.

Confirmed on this host (2026-09-21): `hw.memsize` = 38,654,705,664 (36 GiB),
`vm.swapusage` total 5,120 MB, available arm ≈ 6.5 GiB, and
`ps -axo pid=,pgid=,rss=` completes in ~60 ms over 739 processes.

### 3.2 The ceiling for one command

The arithmetic keeps the same pressure-sensitive shape as the worker cap, but
uses command-specific values and a physical-RAM backstop. These values are kept
next to this consumer rather than shared with pytest: changing worker-pool tuning
must not silently change the safety budget of a live command.

```text
available_mb = <§3.1 available arm>
total_mb     = <§3.1 physical RAM>

effective_available = available_mb                         # lower under swap pressure
reserve_mb = min(_MEMORY_RESERVE_CAP_MB,                     # 1,024
                 total_mb // _MEMORY_RESERVE_FRACTION)       # // 16
budget_mb = min(_MEMORY_SHARE * effective_available,         # 0.75
                effective_available - reserve_mb,
                _MEMORY_PHYSICAL_CAP_FRACTION * total_mb)    # 0.25
ceiling_mb = max(0, int(budget_mb))                           # auto mode

# Existing command-specific floor applies after arithmetic (64 MB by default).
soft_mb    = int(ceiling_mb * _SOFT_FRACTION)                # 0.8
```

The ceiling is for a single command group; unlike the pytest worker cap it is not
divided by a worker count, and it is **not** a machine-wide aggregate governor.
Concurrent commands independently compute their own limits and their combined
memory use can exceed physical RAM. The available-memory share, reserve, and swap
pressure arm reduce a command's limit when the sampled host is constrained, while
the physical cap bounds a command on an unusually idle host.

**Worked numbers.** On a **36 GiB** host with 3,675 MB available, reserve =
`min(1,024, 36,864//16)` = 1,024 MB; the three terms are 2,756 MB (75% of
available), 2,651 MB (available less reserve), and 9,216 MB (25% physical cap),
so the ceiling is **2,651 MB**. At abundant availability, a 36 GiB host's limit
is capped at **9,216 MB** per command. On an **8 GiB** host with 500 MB available,
the arithmetic result is zero because the 512 MB reserve exceeds available memory;
the 64 MB default floor keeps ordinary commands viable but does not allow a large
job to run unbounded. As availability falls further the computed ceiling remains
pressure-sensitive, except for that deliberately small floor.

### 3.3 Swap, compression, and how they enter

Swap is **not** spendable headroom and is not added to the ceiling. A command that
reaches its RSS ceiling and then pages is precisely the "takes the whole device
down" failure the operator described: the device thrashes long before the kernel
OOM-kills. Counting swap as budget would *raise* the ceiling on the exact host
state we are trying to prevent. Swap enters in one bounded way instead: a
**pressure floor** on the reserve. If free swap (`vm.swapusage` `free` on macOS,
`SwapFree:` on Linux) is below `_SWAP_FLOOR_MB` (256 MB), the effective available
memory is the min of the §3.1 arm and the free-swap headroom, which lowers the
ceiling on a host that has already paged itself into a corner. On this host now
(1,192 MB free) the floor does not bind; the term exists for the day it does.

**Compression (macOS).** The compressor is deliberately not subtracted from
available — pages it occupies are already excluded from free and file-backed
(`conftest.py:1063-1064`). But it is the reason RSS *under-reads* the true
footprint of a command: a compressor-backed page is charged to the process but no
longer resident. That is why the sampler (§4) prefers `ri_phys_footprint`
(compressor-inclusive) over `ps` RSS where it can afford the per-pid read, and
why the hard ceiling carries interpretation margin below the true device limit
(§4.3) rather than being set *at* it.

## 4. Enforcement mechanism

### 4.1 Measuring a group's usage

Two sampling shapes, cheap-first, falling back only when the cheap read cannot
see the group:

- **Fast arm — group RSS via `ps`.** One `ps -axo pid=,pgid=,rss=` (measured
  ~60 ms for 739 processes on this host), filtered in Python to `pgid ==
  spawned_pgid`. Sums RSS in KiB → bytes. This is the whole periodic cost: one
  fork per tick, independent of group size, and it catches grandchildren that
  `sh -c` spawned into the same group. The bash child is a group leader
  (`spawned_pgid = os.getpgid(process.pid)`, `tools/builtin.py:2818`), so
  membership is `pgid == spawned_pgid`.
- **Fidelity arm — per-pid footprint.** When the fast sum is within one tick's
  growth of the soft threshold (the watch band, §4.2), refine with
  `mobile.resources.session_resource_usage(pids, footprint_probe=...)` to get
  `ri_phys_footprint`/Pss per pid. This is used *near the decision*, not every
  tick, because `proc_pid_rusage` is per-pid and the batched `top` dump can cost
  1.3 s on this host (`mobile/resources.py:375-381`). We pass our own runner so a
  guard tick never pays the `top` fallback's unbounded cost.

### 4.2 Two thresholds

- **Soft (`soft_mb`, 0.8 × ceiling) — the `memory.high` analog, advisory only.**
  Crossing it does not slow anything (we cannot throttle in userspace); it emits
  ONE live advisory on the update stream ("memory 2.7/3.3 GB") and arms the
  fidelity arm. Repeated ticks over the soft line do not re-emit.
- **Hard (`ceiling_mb`) — the `memory.max`/`oom.group` analog, a kill.** When the
  measured group usage is `>= ceiling_mb`, kill the *whole group* with the
  existing `_kill()` → `terminate_process_tree(pid, force=True)` path and set a
  `memory_exceeded` flag that the result builder reads. The kill is atomic over
  the group by construction, which is the `memory.oom.group=1` guarantee — no
  partial kill, no leader-only kill leaving children to grow.

### 4.3 Hazards, stated

- **RSS under-reads on macOS** (compression): the fast arm can understate a
  command's true charge. Mitigation: the hard ceiling sits *below* the device
  limit by the reserve, and the fidelity arm tightens the reading as it nears the
  line; a command killed one tick late is still killed well below the device
  cliff because the reserve held headroom back.
- **A fast allocator can outrun a 250 ms poll.** One tick can be one allocation
  burst. The per-command ceiling is the minimum of a responsive 0.75 × available
  term, available less the reserve, and a 0.25 × physical-RAM cap. That reserve
  leaves margin, but a burst can still outrun the guard; this reduces the blast
  radius and does not make allocation safe. Concurrent command groups are not
  aggregated, so their independent ceilings do not guarantee a host-wide bound.
- **Polling must never block the loop.** The `ps` read and the footprint read run
  in `asyncio.to_thread`; the tick is added to the existing `asyncio.wait(...,
  timeout=min(0.25, remaining))` loop, never a new blocking call on the loop
  thread.
- **It must not fight timeout/abort/steering-detach.** The memory tick is a third
  branch beside `timed_out` and `aborted`; when it fires it sets its own flag and
  calls the same `_kill()`, so the drain/reap tail (`tools/builtin.py:3246-3267`)
  runs unchanged. On the steering-`CancelledError` path the command detaches to a
  background job as it does today; the guard's tick is re-installed in the
  background runner (`tools/builtin.py:3059-3067`) so a detached command is still
  bounded.
- **It must not kill a legitimate big long-runner.** The ceiling is a property of
  *the command's group*, not of "the command", and a long-runner that is simply
  large is *still over budget* by the operator's own definition. The escape is
  explicit and configurable (§7), not a silent exemption: auto mode uses the
  computed ceiling; manual mode and the per-call `memory_mb` override are how a
  user says "this one is allowed to be big" — with the same fail-closed default
  that the command is killed rather than being allowed to take the device down.

## 5. The "agent sees it" contract

When the guard fires, the tool result must teach the model what happened and what
to do differently. It rides the **result**, never the byte stream (the stream is
bytes from the child; this is the harness talking — the same rule the
credential-dump advisory follows at `tools/builtin.py:3320-3337`).

- **Placement:** inserted at index 0 of `parts`, exactly where the TIMEOUT line
  goes (`tools/builtin.py:3317-3319`): `MEMORY LIMIT EXCEEDED`. It goes *first*,
  not last, because the tool card keeps the HEAD of at most 40 lines (the lesson
  recorded at `tools/builtin.py:3328-3334`).
- **Status:** `is_error=True` via `_error(...)`, so the call classifies as a
  failure the model must react to — not a success with a note. (The kill already
  yields a non-zero return code; the guard makes it *attributable*.) It is an
  ordinary `_error`, not `_invalid_arguments`: the argument was satisfiable, the
  machine said no.
- **Wording (terse, model-actionable, plain text — the tool card paints Text, so
  no backticks, no markdown):**

  ```text
  MEMORY LIMIT EXCEEDED: this command's process group reached 3.4 GB, over the
  3.3 GB budget for one command on this device (36 GB total, 6.5 GB available).
  The command was killed; the session is fine. Reduce peak memory and retry:
  stream instead of loading all rows, lower the batch size, or process the input
  in chunks. To allow a deliberately large command, pass memory_mb on the bash
  call or raise bash.memory.limit_mb in settings.
  ```

  The numbers are the *measured* group peak and the ceiling; naming them is what
  lets the model size the retry. It must not name a "safe" number it cannot know.
- **Soft advisory (live):** one line on the update stream when the group first
  crosses `soft_mb`, e.g. `memory 2.7/3.3 GB — approaching the command budget`,
  emitted through the existing `_emit_update()` channel so the operator sees the
  pressure before the kill. It is advisory only and never a kill.

## 6. Exemptions and safety

- **Never sample or kill the runtime.** The guard only ever reads the pgid it was
  handed (`spawned_pgid`, captured from the child it spawned) and only ever kills
  that pgid. It never enumerates its own process group, `os.getpid()`, the
  session's pgid, or the guard itself. A guard tick reads a group id it already
  holds; it makes no discovery call that could return the runtime's own group.
- **Never kill on doubt.** A failed sample (no `ps`, unparseable output, a
  vanished pid) yields `None` and the tick is a no-op: the guard only kills on a
  *measured* reading `>= ceiling`. Unknown never kills. This matches the
  fail-closed posture the rest of the repo takes on probes that cannot answer (a
  guard that killed on a failed read would kill every command the moment `ps`
  hiccupped).
- **A vanished group is gone, not huge.** `pgid` no longer present in `ps` output
  → skip, the process already exited and the normal reap path owns it.
- **Disable / override.** `bash.memory.enabled=false` disables the guard for the
  machine; a per-call `memory_mb=0` on the bash tool disables it for that command
  (meaning "no budget", explicitly, at the caller's risk). The default is enabled;
  a config that says nothing gets the protection.
- **The guard cannot kill the wrapper.** It has no artifact of its own to reap;
  the kill goes through `_kill()`, which is the same path timeout and abort use.

## 7. Config surface

Following `AGENTS.md` §"Adding a configuration key": every key below gets a
`Setting` in `SETTINGS` under a declared `Section`, a module-level default
constant next to the reader in `memory_guard.py`, and a mapping in
`_consumer_defaults()` in `tests/unit/test_settings_io.py`. The consumer-default
test fails **by name** if the mapping is missing, which is the enforcement.

**Section.** A new `Section("memory_guard", "Command memory limit", Scope.LIVE, …)`.
LIVE, and in its own section, for the same reason `bash.shell` is LIVE and lives
under `tools`: `execute_bash` reads these through a fresh
`ConfigManager(config_dir())` per call (the `_configured_bash_shell` pattern,
`tools/builtin.py:1994-2006`), so an edit lands on the next command. It must NOT
live in the `shell_environment` section, whose members are deliberately NOT live
because the agent's own shell can lower them; the memory ceiling is a resource
policy the *operator* owns and a loosening by the agent is no privilege
escalation (the kill only ever stops the agent's own command).

**Keys.**

| key | path | kind | default | meaning |
|---|---|---|---|---|
| `bash.memory.enabled` | `("bash","memory","enabled")` | BOOL | `True` | master switch |
| `bash.memory.mode` | `("bash","memory","mode")` | ENUM | `"auto"` | `auto` = derived ceiling; `manual` = use `limit_mb` |
| `bash.memory.limit_mb` | `("bash","memory","limit_mb")` | INT | `0` | ceiling in MB when `mode=manual`; `0` = "use the auto ceiling" |
| `bash.memory.soft_fraction` | `("bash","memory","soft_fraction")` | FLOAT | `0.8` | advisory threshold as a fraction of the ceiling |

`path` is a genuine three-level nested tuple, spelled once as
`BASH_MEMORY_ENABLED_PATH` etc. next to the reader (mirroring `BASH_SHELL_PATH`,
`tools/builtin.py:194`), and the `/settings` row's `path=` mirrors it, pinned by a
test the way `test_bash_shell_row_shares_the_consumer_path`
(`tests/unit/test_settings_io.py:566`) pins `bash.shell`.

**Per-call override.** Add `memory_mb: float | None = None` to `BashParams`
(`tools/builtin.py:2009`). Semantics: `None` = use config; `0` = disable for this
call; `> 0` = explicit ceiling. Its schema description must stay terse (the schema
rides every request — the same note at `tools/builtin.py:2012`).

## 8. Interface contract

### 8.1 New module: `local_operator/memory_guard.py`

Stdlib only. Mirrors the shape of `mobile/resources.py` (injectable runner and
probe) so its tests never fork a real `ps`.

```python
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

#: Auto-mode defaults; separate from pytest's worker-pool tuning because this
#: consumer bounds one command group, not a concurrent pool.
_MEMORY_SHARE = 0.75
_MEMORY_RESERVE_CAP_MB = 1024
_MEMORY_RESERVE_FRACTION = 16
_MEMORY_PHYSICAL_CAP_FRACTION = 0.25
_SOFT_FRACTION = 0.8
_SWAP_FLOOR_MB = 256

BASH_MEMORY_ENABLED_PATH = ("bash", "memory", "enabled")
BASH_MEMORY_MODE_PATH = ("bash", "memory", "mode")
BASH_MEMORY_LIMIT_MB_PATH = ("bash", "memory", "limit_mb")
BASH_MEMORY_SOFT_FRACTION_PATH = ("bash", "memory", "soft_fraction")

BASH_MEMORY_ENABLED_DEFAULT = True
BASH_MEMORY_MODE_DEFAULT = "auto"
BASH_MEMORY_LIMIT_MB_DEFAULT = 0
BASH_MEMORY_SOFT_FRACTION_DEFAULT = 0.8

#: A runner is the same shape mobile.resources uses: argv -> (rc, stdout).
Runner = Callable[[list[str]], "tuple[int, str]"]

@dataclass(frozen=True)
class Budget:
    """The result of one budget computation. All MB unless noted."""
    ceiling_mb: int
    soft_mb: int
    available_mb: int | None
    total_mb: int | None
    reserve_mb: int | None
    source: str            # "auto" | "manual" | "override" | "disabled"
    reason: str            # one line, for the result text and the ledger

def compute_budget(
    *,
    mode: str = BASH_MEMORY_MODE_DEFAULT,
    limit_mb: int = BASH_MEMORY_LIMIT_MB_DEFAULT,
    soft_fraction: float = BASH_MEMORY_SOFT_FRACTION_DEFAULT,
    override_mb: float | None = None,
    enabled: bool = BASH_MEMORY_ENABLED_DEFAULT,
    runner: Runner | None = None,
) -> Budget:
    """Resolve the per-command ceiling from config + host memory.

    Pure except for the injectable ``runner`` (defaults to a subprocess
    runner for ``vm_stat``). Never raises; an unmeasurable host degrades to
    ``source="disabled"`` rather than to a guess.
    """

def group_rss_bytes(
    pgid: int, *, runner: Runner | None = None
) -> int | None:
    """Sum RSS (bytes) of every process in ``pgid`` in ONE ``ps`` pass.

    ``None`` when the group is gone or a probe failed — the caller treats
    an unknown as "do not kill". ``pgid`` is the id the caller spawned; this
    function never discovers a group of its own.
    """

@dataclass
class Sample:
    """One tick's reading of a guarded group."""
    pgid: int
    bytes_used: int | None
    bytes_soft: int
    bytes_hard: int
    over_soft: bool
    over_hard: bool
    #: True once the per-pid fidelity read was spent this tick.
    refined: bool = False

class Guard:
    """Per-command memory guard. One instance per spawned command group.

    Constructed by the tool AFTER the child is spawned and its pgid is
    known (``spawned_pgid``), so the guard is bound to exactly one group
    and can never sample or kill any other.
    """

    def __init__(
        self,
        pgid: int,
        budget: Budget,
        *,
        runner: Runner | None = None,
        footprint_probe: Callable[[int], "int | None"] | None = None,
        tick_s: float = 0.25,
    ) -> None: ...

    @property
    def hard_bytes(self) -> int: ...

    @property
    def soft_bytes(self) -> int: ...

    async def sample(self) -> Sample:
        """Read the group's usage off the event loop (``asyncio.to_thread``).

        Cheap arm first; refines with the per-pid footprint read only when
        the cheap sum is inside the watch band near the soft line.
        """

    def should_kill(self, sample: Sample) -> bool:
        """True iff ``sample.over_hard`` on a MEASURED reading.

        ``None`` usage is never a kill. Pure, so the decision is a unit
        target with no process in sight.
        """

    def soft_notice(self, sample: Sample) -> str | None:
        """The one-shot live advisory line, or ``None``.

        Latched: returns a string once per guard instance, then ``None``,
        so a group sitting over the soft line does not spam the stream.
        """

    def over_budget_message(self, sample: Sample) -> str:
        """The tool-result text from §5. Plain text, no markdown."""
```

### 8.2 The integration seam in `execute_bash`

- **Construction:** after `spawned_pgid = os.getpgid(process.pid)` and the
  `group_reaper.register_group(...)` line (`tools/builtin.py:2816-2819`), build
  `budget = compute_budget(...)` from config (read like `_configured_bash_shell`)
  and `guard = Guard(spawned_pgid, budget, ...)` when
  `spawned_pgid is not None and budget.source != "disabled"`. A command whose
  group id could not be captured runs unguarded (it is Windows-shaped; the guard
  is a POSIX feature).
- **Tick:** inside the existing `while True:` loop (`tools/builtin.py:3190-3218`),
  beside the `next_update` branch, add a `next_mem_sample` gate on the same 250 ms
  cadence. On a tick: `sample = await guard.sample()`; if `guard.should_kill(sample)`:
  set `memory_exceeded = True; _kill(); break` — the existing drain/reap tail runs
  unchanged. Else if a soft notice is due, fold it into `_emit_update()`.
- **Result:** in the result builder (`tools/builtin.py:3317-3338`), when
  `memory_exceeded`, insert the §5 line at index 0 (where TIMEOUT goes) and return
  `_error(...)` rather than `_text(...)`. Move the guard's measured peak into
  `details` so a renderer can show it and compaction can prune on it.
- **Background runner:** the same tick is added to the `while not bg_wait.done():`
  loop (`tools/builtin.py:3059-3067`) so a command that detached via steering or
  `background=True` is still bounded; its result builder
  (`tools/builtin.py:3086-3099`) gets the same line.
- **Tests inject a fake runner** through `runner=`/`footprint_probe=` exactly as
  `mobile/resources.py` already does (`mobile/resources.py:312-345`): a fake that
  returns a sequence of `ps` outputs drives the whole guard with no real process,
  and a fake that returns growing values proves the kill fires. A real-process
  test spawns a real `python -c` that allocates and asserts the group is killed
  and the runtime survives.

## 9. Failure modes and risks

| # | failure mode | mitigation |
|---|---|---|
| F1 | Sampler under-reads on macOS (compression) | prefer `ri_phys_footprint` near the line; ceiling below device limit by the reserve (§4.3) |
| F2 | Fast allocator outruns one poll window or concurrent groups exceed aggregate RAM | Per-group ceiling is `min(0.75 × effective-available, effective-available − reserve, 0.25 × physical-RAM)`; it reduces a group's blast radius but neither catches every burst nor aggregates concurrent groups |
| F3 | Sample blocks the event loop | all probes through `asyncio.to_thread`; tick rides the existing `asyncio.wait` timeout |
| F4 | Guard fights timeout/abort/steering | third branch beside the two existing ones; same `_kill()`; same drain/reap tail; re-installed in the bg runner |
| F5 | Guard kills the runtime or a sibling session | bound to the captured `spawned_pgid`; never discovers a group; never samples `os.getpid()` |
| F6 | Guard kills on a failed read | unknown usage → no kill; `should_kill` requires a measured `over_hard` |
| F7 | Legit big long-runner killed | explicit `memory_mb` override and `mode=manual`; documented, not a silent exemption |
| F8 | `ps` unavailable / non-POSIX | guard degrades to `source="disabled"`; one loud line, then silent |
| F9 | Config unreadable | read failure → constants' defaults (enabled, auto); a command must run when `config.yml` is broken |
| F10 | A group id reused after the group dies | `group_rss_bytes` returns ``None`` for a group absent from the `ps` output and `should_kill` refuses a `None` reading, so a reused pgid is never charged the dead group's memory |
| F11 | Ceiling computed on a host whose probes are all absent | `source="disabled"` with `reason`; no kill, no exception — the pre-guard behaviour |

## 10. Risks to watch during rollout

- **The advisory noise floor.** A `pip install` that legitimately peaks near the
  soft line will emit the live advisory. Watch whether the advisory becomes
  wallpaper; if it does, the fix is raising `soft_fraction`, not silencing it.
- **The kill-vs-detach interaction.** A command killed at the same tick a
  steering interrupt arrives could land on either branch; ensure the flag the
  result reads is set before `_kill()` and that the detach path checks it, so a
  memory-killed command is never reported as "continues in the background".
- **The reserve on a small device.** On a 16 GB or 8 GB device with little
  available memory the budget can drive the ceiling to a number small enough to
  kill ordinary commands. The floor belongs to the implementation to justify with
  a measured small-host number; this doc proposes none, because none was measured.
- **The eval worker stays unguarded** (out of scope, §2) and is the known heavy
  corner. Shipping the bash guard without it is still a large reduction, but the
  next incident may well be an eval cell — the `Guard` class is deliberately
  reusable so phase 2 is wiring, not design.

## 11. Implementation measurements (2026-09-21, filled at build time)

Two numbers this doc left open were measured during the implementation. Recorded
here because they are the evidence behind the cadence and the floor, and the next
reader will otherwise re-derive (or re-guess) them.

**Cadence cost, under real fleet load.** The fast arm is one `ps` per tick. On
this 36 GB host (~719 processes, load average ~12) a full `ps -axo pid=,pgid=,rss=`
measures **31-53 ms** on a quiet read (`min 31.3 / median 39.4 / p90 51.6 / max
53.2` over 12 reads) and **~47 ms mean, 57 ms worst** per read under **8 concurrent
guarded commands** (160 reads over a 6 s window; aggregate ~1.26 of one core for
the whole fleet of eight). At 250 ms per command that is under a fifth of one
command's wall time spent sampling, so **the 250 ms cadence stands** — it does not
itself become fleet pressure. `ps -g PGID` measures ~8x cheaper (median 4.3 ms vs
33.6 ms) *on macOS*, but is **not portable-equivalent** — Linux procps reads `-g`
as an e-group/session selector, not `pgid` — so the implementation keeps the
portable full-table read and filters in Python. Folded into the fidelity arm, the
per-pid footprint read is spent only near the soft line (§4.2).

**Small-device floor.** No 8 GB / 16 GB device was available to measure. The
arithmetic itself is the hazard: an 8 GB device at ~1 GB available resolves to
`min(512, 1024-1024) = 0 MB`, which would kill every command on its first tick.
The implementation therefore floors the auto ceiling at `_MIN_CEILING_MB = 64 MB`,
a **judgement, not a calibrated number** (stated as such in the code). It is set
low on purpose: the aim is to keep *ordinary* commands alive on a pressured small
host — measured here, a `git status` peaks at ~3 MB, a shell pipeline at ~4 MB and
a trivial `python3 -c` at ~15 MB of interpreter, so a 64 MB ceiling clears all of
them — not to license a big job, which asks for `memory_mb=` or `mode=manual`.

64 MB is the **default** floor, not the only one: `compute_budget(floor_mb=...)`
lets a caller whose command is not an ordinary one name its own, which is how
§12's build path prices a package-manager child. The floor is a parameter rather
than a number a second consumer re-derives, because the reserve arithmetic has one
owner and the floor is the only part of it that is a judgement about the command.

## 12. Phase 2's first consumer: the mobile bundle build (2026-09-21)

The prediction in §10 was tested the same day, by the second instance of the
incident this guard exists for. `mobile/install.py::_run_build_step` bounded a
build child by **time only** (`_BUILD_STEP_TIMEOUT = 600.0`), and that file records
the incident's rate in two docstrings: *"+100 processes and +5 GB every 25 s until
the host had 0.1 GB free"*, and *"28 processes and 3.2 GB RSS in 8 s"* for the
version probe. The arithmetic is the argument: **0.2 GB/s x 600 s is on the order
of 120 GB** of growth before that time bound could fire, against a 36 GB host. A
bound three orders of magnitude too late cannot protect against the failure it was
added for, and the one place the code RELIED on it is `_pin_mismatch`'s deliberate
fail-open ("the bound protects that case") — so the fail-open was resting on a
bound that does not hold at the recorded rate. This was the incident that took the
operator's machine down, twice in one day.

What landed, and what it says about the seam §10 predicted ("phase 2 is wiring,
not design"):

- The build step samples its group's RSS and kills the **group** on breach, using
  this module's `Guard` and `compute_budget` unchanged in shape. `Guard.sample_sync`
  is the one addition: the install path is synchronous, so there is no loop for
  `sample`'s `asyncio.to_thread` hop to be scheduled on.
- The budget arithmetic gained ONE parameter, `floor_mb` (above), for the measured
  reason in the floor paragraph: 120.8 MB for a package-manager child against 3-15 MB
  for the commands the default floor was sized on.
- The wait had to change shape too, and that is the part worth remembering: a single
  blocking `communicate(timeout=bound)` **cannot** enforce a memory ceiling, because
  nothing reads the group while it blocks. The build step now takes its wait in
  `guard.tick_s` slices with a sample between them, and re-entering `communicate` is
  lossless (documented) and non-duplicating (pinned by a test).
- The fail-open in `_pin_mismatch` is **left alone**, deliberately. It can now rest
  on a bound that holds: a probe that has started resolving the pin grows at the
  measured 0.4 GB/s, so it is stopped at the ceiling a few ticks in — inside the
  probe's own 20 s `_PIN_PROBE_TIMEOUT`, which by itself permitted roughly another
  8 GB before it could fire.
- Discriminating measurement, on a real process group (leader + descendant, against
  a 110 MB ceiling): **guarded, the step ended in 0.728 s with the group killed and
  both pids gone; with the memory budget disabled — the single blocking
  `communicate` the code had before — the same child was still running at 3 s and
  only the clock stopped it.**

§2's exemption list is otherwise unchanged: the eval kernel and the detached
`exec_worker` remain unguarded, and the seam for them is still this `Guard`.


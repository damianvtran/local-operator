# Cross-platform support

`lop` is developed and released from macOS. The unit suite runs on Linux in CI
and the interactive surface is exercised on macOS, so a mechanism that is
silently macOS-only — or silently POSIX-only — passes every gate we have. The
failure this document exists to make visible is not the loud one (an
`ImportError` at startup, which somebody notices); it is the quiet one: a code
path that reports success and does nothing on this OS.

So the rule, stated once and applied in every row below:

> A mechanism may be unsupported on a platform. It may not be unsupported
> **silently**. Either it works, or the user is told which platform they are on,
> what is missing, and how to get the same thing by hand (a foreground command,
> a different supervisor, a different interpreter).

## How the matrix is measured

The instrument is the probe battery: `scripts/xplat_probe.py`, stdlib-only and
self-driving, which runs the same fixed list of mechanism probes on whatever OS
it is pointed at and prints one matrix. `--list` is the authoritative probe
list; `scripts/xplat/README.md` is how to run each piece. The statuses it
prints mean:

| status | means |
| --- | --- |
| `PASS` | the mechanism worked here, on this host, in this run |
| `WARN` | it worked, with a gap the reader should see (a degraded path, a privacy control that does not apply) |
| `SKIP` | the probe could not be attempted, **with its reason** (no `pty` module on Windows; no tunnel configured on this host). A `SKIP` is not a pass and never reads as one |
| `FAIL` | the mechanism does not work here, with what it actually did instead |

Every cell carries its evidence class, because "we support Windows" is not one
fact:

| evidence | what backs the cell |
| --- | --- |
| **reading** | a battery run on a real host of that operating system. The measured numbers are in [Measured baseline](#measured-baseline) |
| **contract** | the behaviour is pinned by code plus unit tests; there is **no** battery reading on that OS from this workspace, so treat it as a design statement, not a measurement |
| **gap** | not implemented; the user gets the refusal quoted in the cell |

## The matrix

| Surface | macOS | Linux | Windows |
| --- | --- | --- | --- |
| Startup and package import | reading: every module imports | reading: same | reading: `497/502` modules import; the other five refuse off POSIX with a named reason rather than breaking. The module-level POSIX imports are gone (`fcntl` and friends are imported inside the POSIX-only function that uses them), so an import-time `ImportError` is not the Windows failure mode |
| `lop serve` (HTTP API) | reading: `/health` 200, second bind refused | reading: same | reading: `/health` 200 and the second bind **refused** — the collision guard uses `SO_EXCLUSIVEADDRUSE` instead of `SO_REUSEADDR`, because on Windows a second bind to the same port with `SO_REUSEADDR` silently succeeds |
| TUI (interactive) | reading: booted in-process **and** driven through a real pty | reading: same | reading, half of it: Textual boots and writes a settled frame (`tui.boot`, 47 KB SVG). The pty probe `SKIP`s here **by construction** — Windows has no `pty` module, ConPTY is a different API — so **do not** read that half as an exercised terminal driver |
| `lop exec` (headless) | reading: runs, and fails without a traceback when it has no provider | reading: same | reading: same |
| `bash` tool | reading: real bash, `/bin/sh` last resort | reading: same | **gap → refusal**: `no bash on this Windows host, and Windows has no /bin/sh. Install Git for Windows …, or point this tool at one you have: lop config edit bash.shell \"<path>\"`. It does **not** silently run the command under `cmd.exe`/PowerShell: this tool advertises bash, and executing another language would be a wrong answer the model cannot see. When bash exists only in Git for Windows' own directories, those are searched, because `which` misses `Git\cmd` |
| Paths, config, sessions | reading: roots resolve, config round-trips, sessions list | reading: same | reading: roots resolve from `USERPROFILE` (15 of them), 241 config options round-trip, sessions list |
| Secret store | reading: round-trip | reading: round-trip | reading, with one **gap** inside it: the default keyfile tier works. The passphrase tier's broker refuses up front — `the secret broker cannot run on win32: peer authentication is not implemented there, so the broker could not tell one caller from another …`. `lop secret status` also states what protects the store at rest, because a Windows store is usable rather than broken and a tier line that reads like a POSIX store's would be a lie |
| Unattended wake service | reading: LaunchAgent plist written | reading: a `systemctl --user` unit whose timer is enabled and active, measured on real systemd 255 (the container image cannot run systemd at all — even a control unit exits 255 under emulation); the before-reading was a `FAIL` ("no supervisor installer for this platform") | reading: the task is registered and started via `schtasks` (no elevation, no new dependency), and `wake.install` reports `supervisor: installed`. Placement across a real logoff/logon cycle is **not** covered by any runner |
| Mobile portal (`lop mobile`) | reading: daemon serves, `/healthz` 200; install via launchd; password in the login Keychain | reading: daemon serves; install writes and would enable a `systemctl --user` unit, but a redirected `HOME` cannot reach the real user manager, so the battery records `SKIP` rather than pretending | reading: install registered the task, started it, and the daemon served `/healthz` 200 with `dist: true`; password in a DPAPI blob keyed to the user. All three arms now carry `LOCAL_OPERATOR_CONFIG_DIR` — without it the daemon read the DEFAULT store and exited 2 |
| Tunnels | reading: `SKIP` — no tunnel configured on that host | reading: same `SKIP` | reading: same `SKIP` (no tunnel configured on the runner). `cloudflared` is resolved from `PATH` (never downloaded by `lop`); the service arm is the same three supervisors, unit-tested on Windows |
| Single-owner locks | reading: a second holder is refused | reading: same | reading: a second holder is refused, through the `msvcrt` lock where `flock` does not exist. Every "one owner" invariant in `lop` rests on this row |
| Orphan-process reaping | reading: reaper active | reading: reaper active | **gap, announced**: `process-group ledgers are POSIX-only … a shell command that outlives a HARD death of this process … is NOT reclaimed at the next startup`. Logged once per process, at the first command, because a safety net that is absent looks exactly like one that works |
| No supervisor at all (Devuan, Alpine, containers, WSL2 without systemd) | n/a | **gap → refusal**, one sentence per daemon: `no supported user service supervisor found (launchctl on macOS, systemctl --user on Linux, Task Scheduler via schtasks on Windows); run <that daemon's foreground command>` | same refusal |
| Notifications / focus / `/fork` | richest: `osascript`, window focus, terminal protocols | terminal-mediated protocols, `notify-send` when the desktop has it | thinner: notification is the terminal's own protocol (a bare BEL cannot carry text), window focus is macOS-only and says so, and no `/fork` terminal backend exists — the fork is still created and durable and the receipt names the command that reaches it, which is why that message is a note rather than a warning |

## Measured baseline

Both halves are the same instrument on the same probes; only the platform and
the revision differ. The **before** readings were re-taken on `09178d07` with
the current battery (a battery that grew probes between the two readings would
make the comparison a lie), and the **after** ones are at the head of this
branch.

| Host | Python | Before | After |
| --- | --- | --- | --- |
| Ubuntu 24.04, arm64, in a container | 3.12.3 | `PASS=19 WARN=2 SKIP=2 FAIL=3` | **`PASS=21 WARN=2 SKIP=3 FAIL=0`** |
| Linux Mint 22, amd64 under emulation | 3.12.3 | same set — the two distros agreed on every probe | **`PASS=21 WARN=2 SKIP=3 FAIL=0`** |
| macOS, arm64 (the release platform, and the one that must not regress) | 3.13.12 | `PASS=21 WARN=2 SKIP=3 FAIL=0` with the pre-fix count of 11 unguarded POSIX attributes | **`PASS=21 WARN=2 SKIP=3 FAIL=0`** |
| Windows Server 2025, AMD64, `windows-latest` runner | 3.12.10 | `PASS=13 WARN=1 SKIP=3 FAIL=9` | **`PASS=21 WARN=2 SKIP=3 FAIL=0`** |

What the before-readings were not passing:

* **Ubuntu and Mint**: `static.posix_attributes` `FAIL` (15 unguarded POSIX
  attributes); `static.posix_imports` `WARN` (two unguarded module-level
  `import fcntl`); `wake.install` `FAIL` — `no supervisor installer for this
  platform`, so a scheduled wake fired only if a session happened to be
  reopened; `mobile.install` `FAIL` — `install needs macOS launchd`, so the
  phone portal could only run in the foreground.
* **Windows**: `import.package` `FAIL` (`496/502` — `os.register_at_fork`
  unguarded in the evidence store, plus the modules that import it);
  `secret.roundtrip` `FAIL` (`module 'socket' has no attribute 'AF_UNIX'`, which
  killed the secret store outright); `serve.health`, `serve.double_bind` and
  `mobile.daemon_serve` all `FAIL` with "no response", i.e. `lop serve` and the
  portal never came up; `config.roundtrip` and `tui.boot` `FAIL` on the
  probe's own cp1252 decoding rather than on the product.
* **macOS** was already green on behaviour; what moved there was the static
  reading, 11 unguarded POSIX attributes to 0.

The two `WARN`s that remain are honest and deliberate, and they are not the same
two everywhere: `static.posix_attributes` reports 0 **fatal** and a count of
*leads* (uses it cannot prove are guarded — a lead is for reading, not for
acting on), and `wake.status` reports what it could not establish — `cannot be
verified for this store` on macOS and Linux, where an isolated `HOME` cannot
reach the real user manager, and `not installed` on Windows, where there is no
registration to find. The three `SKIP`s are: no tunnel configured on the host
(×2), and `mobile.install`, which on macOS is not run at all because the
installer writes to the login keychain an isolated `HOME` does not have, on a
container without `Node >= 22` refuses legibly, and on a redirected `HOME`
declines to enable a unit the real home does not own.

Where the readings come from:

* the Linux rows are the battery's own JSON output, one file per image, built
  from `scripts/xplat/Dockerfile.probe` and run by `scripts/xplat_linux_matrix.sh`;
* the macOS row is a battery run on a development host plus the exported TUI
  frame (`tui.boot` writes an SVG; it is the same frame the visual-validation
  recipe in `AGENTS.md` renders and looks at), because a `boot` that "passed"
  without a look at the frame is exactly the claim this document refuses to
  make;
* the Windows row comes from the `xplat-probe-windows` job in
  `.github/workflows/ci.yml`, which runs the same battery on `windows-latest`
  with PowerShell steps; its artifact is the Windows evidence. The rows above
  that say `reading` for Windows are backed by it; the rows that still say
  `contract` or `gap` are the ones no runner exercises at all (notifications,
  window focus, `/fork`, the orphan reaper's hard-death gap), and they say so.

A note on the Linux Mint row: the Mint image is `amd64`-only and this host is
`arm64`, so its userland runs under emulation. That is fine for the things the
battery measures there (imports, paths, config, the CLI, the locks, the
supervisor's *decision*) and useless for systemd — every unit on that image,
including a control `/bin/echo` unit, exits 255 under emulation. So Mint's
service-manager behaviour is **not** covered by its row; the systemd arm was
proved on real systemd 255 in an Ubuntu container instead, by installing the
unit and observing `systemctl --user` report both the service and its timer
enabled and active.

## What this document does not claim

* **Windows containers are not an option, and not a missing feature.** Docker on
  a macOS or Linux host runs a Linux kernel, and a Windows container image needs
  a Windows kernel underneath it. Windows validation is therefore a real Windows
  runner — never Wine, never emulation, because neither would be evidence about
  Windows.
* **One runner is not every Linux.** CI has exactly one Linux (Ubuntu LTS). The
  mechanisms most likely to differ between distros — a user systemd instance,
  `notify-send`, `xdg-open`, the terminal `lop` was launched from — are why the
  container matrix exists. Mint 22 agreeing with Ubuntu 24.04 is one data point,
  not a proof about Fedora.
* **A `PASS` is about a mechanism, not about a workflow.** The battery proves a
  daemon serves and a frame renders; it does not prove a UI reads well on that
  OS, and it does not exercise the flows that need a provider, a network or a
  logged-in desktop.
* **The unmeasured Windows corners are named, not implied.** Task Scheduler
  placement, DPAPI placement, the process-group reaper and the terminal driver
  are all "contract" or "gap" rows above for the same reason: there is no
  reading for them yet.
* **The `O_BINARY` claim in this document was wrong three times, and what stands
  here is the audit's actual RESULT rather than a summary of it.** The
  `master.key is 33 bytes` failure had a cause -- on Windows `os.open` is the
  CRT's *text* mode unless `O_BINARY` is given, so a `0x0A` handed to `os.write`
  lands as `0x0D 0x0A`. What review round 4 measured, independently, pairing
  every `os.open` in `local_operator/**.py` with the writes in its file and
  resolving one level of helper indirection: **20 write sites, 21 descriptor
  instances, 18 now carrying the flag and 3 not.** No unflagged `os.open`
  receives an `os.write` -- that half of the claim holds.

  Scope the rule to what is true: **every `os.open`-based write site in this
  package ORs the flag in.** It is NOT true of the pipe ends. The three without
  it are the two `os.pipe()` ends in `evaluation/adapters/rpc.py` (wired from
  `supervisor.py`, and `os.pipe()` does not set `_O_BINARY`) and
  `clipboard.py`'s scratch probe, whose fd is a `tempfile.mkstemp` handle that
  ALREADY carries the flag through `tempfile`'s own `_bin_openflags` -- so
  `b'blat'`'s safety is the second reason there, not the first.

  Three earlier versions of this bullet are why the method is stated rather than
  the conclusion: the first named three "frame writers" that are read-only, the
  second said "one site" while three writes went unpatched (one of them
  `withdraw_inbox`), and the third generalised a file-level pairing into a claim
  about descriptors it cannot see.
* **The read side had the same defect, and it was found by review rather than by
  the work that fixed the write side.** `verify.py` hashed artifact bytes read
  under `_READ_FLAGS` and compared the digest to the file NAME, and text mode
  rewrites on the way IN as well: on Windows an artifact containing `0x0D 0x0A`
  was hashed as something other than what is on disk, so it failed its own
  verification. `_READ_FLAGS` in `verify.py` and `store.py` now carries the flag.
  Same defect, opposite direction.
* **One pipe channel is still genuinely unmeasured, and it is the WORSE of the
  two, not the safer.** `evaluation/adapters/rpc.py` frames on `b"\n"`
  (`canonical_line`) and `parse_canonical_line` hard-REJECTS a CR rather than
  tolerating it, so a byte translated by the CRT POISONS that channel instead of
  being ignored. `os.pipe()` does not open `_O_BINARY` (an earlier version of
  this bullet claimed it does -- false in CPython 3.12/3.13/3.14 and on main),
  and CI cannot see the gap: the unit matrix is ubuntu + macos, and the Windows
  jobs run two boundary test files plus `xplat_probe.py`, which has no pipe
  probe. Recorded as an open gap, not repaired here.


## Re-measuring

```sh
# the battery on this host, table + JSON
OUT=${OUT:-/tmp/xplat-matrix}
python scripts/xplat_probe.py --json "$OUT/$(uname -s).json"
# every Linux distro we care about, in containers
OUT="$OUT" scripts/xplat_linux_matrix.sh            # ubuntu:24.04 + linuxmintd/mint22-amd64
OUT="$OUT" scripts/xplat_linux_matrix.sh ubuntu:24.04
# one cross-OS grid from a directory of those JSONs
python scripts/xplat_report.py "$OUT"
```

CI does the Ubuntu and Windows legs on every PR that can affect either
(`xplat-probe-linux`, `xplat-probe-windows`), and uploads the JSON as an
artifact — that artifact, not this page, is the current reading. A `FAIL` in any
of these runs exits non-zero, so a mechanism that stops working on a platform
turns the corresponding leg red rather than quietly degrading.

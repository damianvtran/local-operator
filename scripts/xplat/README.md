# The cross-platform probe battery

Four small pieces, one job: get a *reading* of `lop`'s platform-dependent
mechanisms on an OS that is not the one you are sitting at. The support matrix
they feed is `docs/XPLATFORM.md`; this file is how to run them.

| Piece | What it is |
| --- | --- |
| `scripts/xplat_probe.py` | the battery: one process, a fixed list of mechanism probes, one matrix, `--json` for machine output. Stdlib-only, so it runs in a bare container before anything is installed |
| `scripts/xplat_report.py` | combines several `--json` files into one **cross-OS** grid. The interesting fact is never "probe X failed" but "probe X passed on macOS and failed on Windows" |
| `scripts/xplat_linux_matrix.sh` | builds `scripts/xplat/Dockerfile.probe` per distro and runs the battery inside each — the DISTROS one Ubuntu runner cannot be |
| `.github/workflows/ci.yml` jobs `xplat-probe-linux` / `xplat-probe-windows` | the same battery on a real Ubuntu runner and a real Windows runner, with the JSON uploaded as the run's artifact |

## Running the battery

```sh
python scripts/xplat_probe.py                     # full battery, table on stdout
python scripts/xplat_probe.py --json out.json     # ... and machine-readable output
python scripts/xplat_probe.py --list              # probe names only
python scripts/xplat_probe.py --only tui secret   # probes whose name contains one of these
python scripts/xplat_probe.py --out-dir shots/    # put TUI frames here (e.g. out of a container)
python scripts/xplat_probe.py --keep              # keep the sandbox directory
```

Exit status is **0 when nothing `FAIL`ed, 1 otherwise**, which is what makes it
usable as a gate: the `WARN`/`SKIP` rows print their reason, and a `FAIL` fails
the run.

`--only` matches the probe's own name with underscores (`--only static_posix`,
`--only tui`, `--only secret`); the table prints those names dotted
(`static.posix.imports`). Use the plain words in the table's first column when
you are guessing.

**It never touches your live install.** Every probe runs with `HOME`,
`USERPROFILE`, `LOCAL_OPERATOR_CONFIG_DIR`, the `XDG_*` roots and `TMPDIR`
re-homed into a fresh sandbox under the system temp dir, and inherited `CMUX_*`
variables are stripped — an inherited workspace id is enough to make a TUI boot
rename the operator's real cmux workspaces. `--keep` prints the sandbox path;
otherwise it is removed on exit.

## What the statuses mean

| status | means |
| --- | --- |
| `PASS` | it worked, here, in this run |
| `WARN` | it worked with a gap worth reading (a degraded path, a privacy control that does not apply on this platform) |
| `SKIP` | it could not be attempted **and this line says why** (no `pty` module on Windows; no tunnel configured). Never a silent pass |
| `FAIL` | it does not work here; the detail line says what happened instead |

Read a `SKIP` before you read a count: a row that is `SKIP` on one OS and `PASS`
on another has told you nothing about the platform yet.

## Running the Linux distro matrix

```sh
scripts/xplat_linux_matrix.sh                       # ubuntu:24.04 + linuxmintd/mint22-amd64
scripts/xplat_linux_matrix.sh ubuntu:24.04          # one image
OUT=/tmp/xplat-matrix scripts/xplat_linux_matrix.sh # where the JSON lands
PROBE_ARGS="--only tui serve" scripts/xplat_linux_matrix.sh
```

It builds `scripts/xplat/Dockerfile.probe` with `--build-arg BASE=<image>`, runs
the battery in each container, then prints the combined grid from
`scripts/xplat_report.py` and exits non-zero if any image had a `FAIL`. A `FAIL`
on one distro does not stop the others: the *difference* between distros is the
finding.

Two details of the rig are deliberate, not incidental:

* the probe image uses its own `scripts/xplat/Dockerfile.probe.dockerignore`,
  which replaces the repository root's — the root file excludes `tests/`, and
  the TUI probe imports the app double that lives there;
* the build context is the repository root, because the image installs the
  package from source.

**Windows cannot be done this way, on any host.** Docker on macOS or Linux runs
a Linux kernel, and a Windows container image needs a Windows kernel underneath
it. Windows validation is therefore a real Windows runner (the
`xplat-probe-windows` CI job) — not Wine, not emulation, neither of which would
be evidence about Windows.

## Combining several runs

```sh
python scripts/xplat_report.py /tmp/xplat-matrix             # grid
python scripts/xplat_report.py /tmp/xplat-matrix --detail    # + the rows that are not PASS
```

It labels each column from the run's own host facts (system, release, machine),
so two containers of the same distro stay distinguishable. Exit status is 1 if
any probe `FAIL`ed anywhere, so a shell that wraps it cannot swallow a failure.

## Adding a probe

1. Write `def probe_<family>_<thing>(env: dict[str, str]) -> Result:` beside the
   others, and return `Result("<family>.<thing>", "PASS"|"WARN"|"SKIP"|"FAIL",
   detail)`. The name in the `Result` is what the table and the JSON use; the
   function name is what `--only` matches.
2. Register it in `PROBES`. Nothing else is needed — the harness derives the
   name, times it, catches anything it raises (a raising probe is reported as a
   `FAIL`, never as a dead battery), and records the seconds it took.
3. Use the `env` you are handed for every child. That is the isolation. A probe
   that reads `os.environ` directly is measuring your machine.
4. Prefer `SKIP` with a reason over a `FAIL` for a mechanism this OS genuinely
   does not have — and over `PASS` for anything you did not actually do.
5. If the probe needs a child that keeps running, spawn it with `_spawn_cli`
   (own process group) and stop it with `_terminate`; a daemon left in the
   battery's own group makes the cleanup signal the battery itself.
6. Stdlib only. The battery has to run in a container before the package's
   dependencies exist, and `import.package` measures the package, not its
   tooling.

A probe is worth adding when the answer differs by platform *and* a silent
difference would look like success — an installer that reports "installed" and
installs nothing, a lock that silently does not lock.

## Reading the output honestly

* A green table is not a green workflow: check the `SKIP` rows and the `WARN`
  details, and open the TUI frame `tui.boot` writes if you are claiming the UI
  boots somewhere.
* Numbers in `docs/XPLATFORM.md` carry the date and commit they were read at.
  Re-run the battery rather than quoting them forward.
* One Ubuntu runner is one Linux. If a change touches a desktop mechanism
  (`notify-send`, `xdg-open`, the launching terminal, a user systemd instance),
  run the distro matrix before claiming it works on Linux.

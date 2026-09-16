# Design: install generations — one tree per build, a pointer resolved at exec

Status: implemented (Phase 1). Scope: one PR. No `pyproject.toml` version bump —
the release owner handles that.

## 1. The problem as found in the code

`uv tool install --force` **recreates** `~/.local/share/uv/tools/local-operator`
in place. It does not patch the tree; it deletes and rewrites it, all 4.8k files
and ~136 MB, over a window of seconds — and every process on the machine whose
`sys.path` names that tree is reading files that no longer exist.

Measured on the reporting host, 2026-09-15:

* **36 sessions died with no exit record.** The incident vocabulary already has
  the sentence for it — ``runtime-killed``: "the runtime disappeared without
  exiting cleanly while this turn was running". Nothing recorded a stop, because
  nothing ran: the process was gone between two bytecodes.
* **113 crash reports** in `~/Library/Logs/DiagnosticReports` name the planted
  libpython dylib, clustered inside the install window (19:22:57, 19:23:23-19:23:52,
  and 19:28:0x-19:28:38 as the app respawned). A process **launched** during the
  rewrite dies at load: dyld cannot resolve `@executable_path/../lib/libpython3.12.dylib`
  out of a tree that is being emptied.

The runtime's own self-refresh does not help here, and this is the part that
makes the incident unfixable inside the old shape. `BUILD_CHECK_S`'s docstring
states the design plainly: *"a runtime that is already idle can afford to notice
an update within a few seconds, and a busy one never checks."* The idle gate is
deliberate — a runtime holding a turn must not retire on someone else's
schedule — so the fleet's most valuable processes had, by construction, no
defence at all.

The same rewrite also produced this, which is why the marker and the settle
window exist at all: `~/.local/share/uv/tools/local-operator/.lop-source` was
written by one process while others were still importing the previous build, so
a runtime could compare its boot stamp against a marker that described neither
tree (`docs/design-build-skew.md`).

There is no repair available inside that layout. A tree that is rewritten in
place cannot be handed over from, and the failure mode is not an exception in a
process that is watching — it is a process that is not watching because it is
busy, plus one that dies before it can watch.

## 2. The layout

Each build gets its **own** root. The stable path becomes a **pointer** that is
resolved once per process, at exec:

```text
~/.local/bin/lop ─┐
~/.local/bin/local-operator ─┴─► <stable>/current ─► <stable>/generations/<id>/
                                                          bin/lop -> tools/local-operator/bin/lop
                                                          tools/local-operator/          # the venv (sys.prefix)
                                                          tools/local-operator/.lop-source
                                                          tools/local-operator/uv-receipt.toml
```

`<stable>` is `~/.local/share/lop`, deliberately NOT under `~/.local/share/uv`:
uv owns that tree and rewrites it, and a subdirectory of a tree an installer
recreates is not a stable root.

```text
<stable>/current          the pointer: the ONE mutable artefact of the layout
<stable>/bin/python3      the supervisor shim (see §5)
<stable>/generations/<id> one root per build; <id> is UTC-timestamped
<stable>/generations/<id>.partial  only ever transient — see §3
```

**Nothing a running process holds is ever rewritten.** Its `sys.path` names the
generation it was launched from, so the install that supersedes it is beside it,
not under it. That is the whole fix: the incident needs a handover, and a
handover needs two trees.

**Resolution happens once, in the kernel, at exec.** `~/.local/bin/lop` is a
symlink to `<stable>/current/bin/lop`, whose target is the generation's own
console-script shim — whose shebang names **that generation's** interpreter
absolutely. So the process that starts has a concrete `sys.prefix` and
`sys.path`; it never imports through the mutable `current`.

That last point is not incidental and it is measured. Launching an interpreter
*through* the pointer (`<stable>/current/tools/local-operator/bin/python3`)
reports `sys.prefix` as the pointer path, with `current` in it — CPython detects
a venv from the parent directory of the path it was invoked by, so the venv is
found, and the process then holds a `sys.path` that a later flip redirects. That
is the original failure wearing a new hat, which is why every spawn site in this
change resolves the pointer itself and hands the child a **concrete** path
(`update.current_interpreter`, `launch._spawn_interpreter`).

## 3. Install, flip, prune

### 3.1 One install, six steps

`update.install_into_generation(source=None, *, runner=None, version, commit, ref)`:

1. reserve `generations/<id>` with `os.mkdir` (exclusive: two installs in the
   same second take two names rather than sharing one tree);
2. run uv with `UV_TOOL_DIR=<gen>/tools` and `UV_TOOL_BIN_DIR=<gen>/bin` — the
   whole mechanism, and the reason a fresh generation cannot disturb the
   installed tool tree or `~/.local/bin`;
3. write `.lop-source` into the new venv, so the marker is there **before** the
   generation is visible to any reader;
4. flip the pointer (§3.2);
5. write the stable launchers and the supervisor shim (best-effort);
6. the caller prunes (§3.3).

A failure at 2 or 3 removes the tree and raises. Nothing observable has changed:
the pointer never moved. Steps 5 and 6 are best-effort by design — the build is
already current, and a sandbox that denies a write must not be told the upgrade
failed.

**No staging rename**, and it is worth recording why, because
`<id>.partial` + `os.rename` looks tidier and is wrong here: uv bakes the
installation path into the console-script shims it writes under
`UV_TOOL_BIN_DIR`, so every one of them would name the renamed-away directory and
dangle. Reserve-and-build-in-place keeps uv's own artefacts pointing at paths
that survive.

### 3.2 The flip

```python
staged = pointer.with_name(f"current.tmp-{os.getpid()}")
os.symlink(generation, staged)
os.rename(staged, pointer)          # atomic
```

The staging name is a **sibling** of the pointer, so the rename cannot cross a
filesystem, and `os.rename` over a symlink is the atomic step: there is no
instant at which `current` is absent or names a tree that is not there. What that
buys is that **a running process is never disturbed** — the whole incident — and
that a *spawn* through this layout is resolved to a concrete path before the
child starts (`update.current_interpreter`, `launch._spawn_interpreter`), so no
child ever holds a mutable component.

**The residual, measured rather than argued.** Execing a CONSOLE SCRIPT through
the chain (a person typing `lop`, a launchd unit starting) at the exact instant
of a rename can still fail at startup on this platform: macOS returns `EINVAL`
for a path whose component is replaced underneath the reader, and in the child
that component is its own `sys.path[0]`, which the console script keeps spelled
through `current`. Reproduced by a synthetic rename loop at ~143k flips/second
(3 failures in 120 execs; the runtime-spawn path was clean in the same run,
because it hands over a concrete interpreter). At the rate flips really happen —
once per install, days apart — that window is not reachable, and the failure is
loud (`No such file or directory` from the shell, immediately re-runnable)
rather than a silent death of a live session. Removing it entirely would mean
rewriting every `~/.local/bin` entry on every flip, which trades this for a
many-file non-atomic update; that trade is not worth making for a startup error
nobody has hit, and it is recorded here so a future reader can revisit it with
the number in hand.

Two further measured details, both of which a later "tidy-up" would undo:

* **`current_generation()` reads with `os.readlink`, not `Path.resolve`.**
  readlink returns what the link NAMES; `resolve` walks the chain and can be
  caught between steps. macOS raises `OSError: [Errno 22] Invalid argument` for
  both spellings when the link is replaced underneath the reader (a tight
  rename loop reproduces it — `tests/unit/test_install_generations.py` measures
  it), and the layout's answer to that is not "eliminate it" but "every caller
  has a documented fallback and pruning refuses to delete anything": the
  *transience* is the guarantee, and a pointer left dangling by an
  unlink-then-create would fail the very next read instead.
* **Paths compare through one resolution.** `<stable>` is built from
  `Path.home()`, and on this platform `/tmp` is a symlink to `/private/tmp`, so
  the directory the pointer *named* and the directory pruning *listed* are two
  spellings of one path. Both sides go through `update._real` (a non-raising
  realpath). A set of unresolved paths against resolved ones never matches, and
  the symptom is a prune that silently keeps every generation forever — found by
  this change's own test, not by inspection.

### 3.3 Retention: structural, not a count

`update.prune_generations(keep=DEFAULT_KEEP_GENERATIONS, referenced=…)` keeps a
generation when **any** of these holds:

1. it is the pointer's target;
2. a live or persisted record names its install root — session runtimes
   (`run/mobile`) and the `lop serve` daemon (`run/serve`, whose record has
   carried `prefix` all along) both publish one, and `SessionRecord` gains
   `install_root` for it (additive; `PROTOCOL_VERSION` does not move);
3. it is one of the last `keep` (default **2**) generations nothing references.

Rules 1 and 2 are why this is safe to run on a busy machine; rule 3 is the
margin for a session with no record yet (an engage's first ~1.2 s) and for a
terminal whose record has aged out. Two rather than one because the previous
generation is exactly the one a just-flipped fleet is still reading from.

A generation with **no `.lop-source`** is an install in flight and is skipped —
only age (one hour) calls it debris, which is the shape a `kill -9` mid-install
leaves behind. Nothing else can leave one: every failure path removes its own
tree, and a finished generation always carries a marker.

Deletion is what makes the layout affordable rather than a leak — a generation
is a whole venv — so pruning runs after every successful install **and** on
demand from `lop install prune`, which reports what it removed. Removals are
returned and reported rather than only logged: a silent 136 MB delete is not
something this tool gets to do.

### 3.4 Migration

`lop install migrate` copies the tree the running `lop` imports from into
generation 1 and flips the pointer. **Non-destructive**: the legacy fixed tree
is copied, never moved or deleted, so a machine that has just adopted the layout
still has the install it was running on and can fall back to it by hand; it is
removed only by an explicit prune.

A real copy rather than hardlinks — the cheap shape is wrong here: a hardlinked
generation shares inodes with a tree that `uv tool install --force` is about to
rewrite, and this layout's entire promise is that a generation's bytes are
written once. 136 MB and ~4.8k files is the honest price, paid once per machine.

## 4. What this makes of the old defences

They stay, and their role changes from **safety** to **convergence**:

* **the build watch** (`buildwatch.build_changed`) now compares this process's
  boot sample against the build a fresh `lop` would load — the POINTER's
  generation (`update.disk_build`) — because this process's own tree can no
  longer move. `installed_build()` keeps "this process" semantics, so
  `lop --version` and the record's stamp stay honest about the code in memory;
  the two functions are the distinction the whole change turns on.
* **the settle window** is measured on the *disk* install's marker
  (`buildwatch.disk_marker_prefix` → the pointer's venv), which is the install
  whose freshness is in question. Left reading this process's own marker it
  would report a just-landed install as long settled and disarm itself exactly
  where it still does work.
* **the idle gate** is unchanged and now correct by construction. A busy runtime
  is not in danger of losing its files; it simply keeps running the build it
  loaded until its own work is done, and then retires. **A mixed-generation
  fleet is an accepted steady state** (`docs/design-build-skew.md` §6): new work
  engages on `current`, existing work finishes where it is.
* **the files-gone probe** (`process._files_gone`) and
  **`update.classify_import_failure`** stay as the legacy safety net for the
  shapes that still rewrite in place — pip and pipx installs, a process started
  before the machine migrated, a `lop` running out of the old fixed uv-tool
  tree. For a generation process they answer "no move" by construction, which is
  the truth: a lazy `ImportError` there is a packaging bug, not an install race.

## 5. Supervised daemons

launchd `Program` and systemd `ExecStart` were naming an interpreter inside one
venv — the branded hardlink, with the libpython dylib pin beside it. That is
exactly the path a prune removes, and a unit **re-executes** it on every
restart, which is how 113 processes died at load.

So the four installers (`mobile`, `wakes`, `browser_bridge`, `tunnels`) name a
**stable shim** instead: `<stable>/bin/python3`, a 20-line `sh` script that
resolves `current` and execs that generation's interpreter. It is a script
rather than a symlink because a symlink at that path loses the venv entirely —
CPython decides "am I in a venv" from the parent directory of the path it was
executed through, and a symlink to `<gen>/…/bin/python3` resolves to the base
interpreter with no `site-packages` (verified). The shim prefers the branded
image when it is planted, so `p_comm` still reads `Local Operator`, and falls
back to the interpreter when it is not; `ProgramArguments[0]` — what macOS
Background Task Management names the login item by — is unchanged.

`procname.supervised_image()` answers `None` on a machine with no generation
layout, and every installer then keeps **exactly** the plist it shipped before.
That gate matters: a source checkout must never rewrite the operator's plists,
and a pip/pipx install has no pointer to name.

## 6. CLI

* `lop update` — unchanged for the user, and on the uv-tool layout it now
  installs into a generation and flips. Both front ends share
  `update.perform_upgrade`, so the TUI's `/update` gets the same path from the
  same code.
* `lop update --from-snapshot <dir-or-ref>` — install a local build. A directory
  is installed as it stands (a caller that prepares its own tree keeps doing
  so); a git ref is archived out of the repository the command runs in, and its
  commit is recorded, so two builds of one unchanged version stay
  distinguishable. This is the in-repo half of the out-of-tree `lop-update`
  script's job, and it is deliberately independent of it.
* `lop install prune [--keep N]`, `lop install migrate`, `lop install status`.

pip and pipx **keep today's behaviour**, with the consequence stated: neither
has a directory layout this module can make atomic, so they still rewrite
`site-packages` in place under the running fleet. There is no generation story
for them to route through.

## 7. Not in this PR

The **boundary-latched refresh** — a busy runtime re-checking when its turn
settles, so a fleet converges on the first idle boundary after a flip rather
than at each session's next natural turnover — is Phase 2. Nothing here depends
on it, and nothing here blocks it.

## 8. Evidence

Unit: `tests/unit/test_install_generations.py` (pointer-flip atomicity, observed
under a racing reader and at the instant before the swap; the per-generation uv
environment; pruning's three retention rules plus its fail-closed refusals; the
disk read against the pointer; the shim resolving a flipped pointer; the
migration's copy semantics and its refusal of a source checkout).

E2E, against a real `process.py` in a subprocess under an isolated `HOME`
(`tests/e2e/test_install_generations_e2e.py`): a runtime parked on a real `bash`
tool survives a real install-and-flip with the same pid and a completed turn and
no incident; an unwatched runtime keeps its pid and its heartbeat; a spawn after
a flip runs the generation `current` names, by a concrete path; the stable
launcher resolves the current generation with no rewrite of its own.

Real-path walkthrough (isolated `HOME`, real `uv`, real console scripts — the
runbook QA is asked to repeat at fleet scale):

```sh
uv tool install --force .                                  # a legacy fixed tree
<legacy>/lop install migrate                               # copy + flip, legacy left in place
~/.local/bin/lop --version                                 # through the pointer chain
<generation>/lop update --from-snapshot <dir>              # real uv build into a new generation + flip
~/.local/bin/lop install status                            # pointer, generations, disk build
~/.local/bin/lop install prune                             # removes only superseded trees
```

Observed: the migration copied the tree into generation 1 and flipped the
pointer while leaving the legacy install intact; `lop --version` resolved
through `~/.local/bin/lop → current → generation`; the snapshot install built
55 packages into a new generation, flipped, and pruned the superseded one;
`install status` reported the pointer, the generations and what a new `lop`
would load; the supervisor shim resolved `current` and exec'd a concrete
generation interpreter (verified by `sys.prefix`).

The live-host proof — a real install with ~N live runtimes, the pid set
identical before and after, zero new `DiagnosticReports` — belongs to QA, and
`LOP_INSTALL_ROOT` plus an isolated `HOME` is how to run it without touching the
operator's own sessions.

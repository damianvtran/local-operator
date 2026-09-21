# Endpoint protection: what Local Operator installs, persists and launches

This document exists because behaviour-based endpoint protection (EDR) has
quarantined Local Operator installs under a **persistence** heuristic, deleted
the install, and left the operator with a product that will not start and no
document to hand their security team. It is written for two readers: the
operator who has just lost an install, and the IT or security admin who has to
decide whether to allow this software deliberately rather than repeatedly
re-discover it.

Everything below is either read out of the installers in this repository or
measured on a real macOS install, and each claim says which. Nothing here is an
aspiration; where something is not done yet, §5 says so.

## 1. What this product installs and persists on a user machine

### The four user LaunchAgents

A full install registers up to four per-user `launchd` jobs in
`~/Library/LaunchAgents/`. They are ordinary **user** agents — no root helper,
no `/Library/LaunchDaemons`, no privileged install. Each plist is written by
its own installer module, and each job runs an interpreter image named
`Local Operator` with a different module. Two keys matter, and they do
different jobs: `Program` names
the **image** launchd executes, and `ProgramArguments[0]` is a **role label**,
not a path. The remaining argv elements are the module and its arguments.

| Label | `Program` — the image | `ProgramArguments[0]` — role label | Rest of argv | Log |
| --- | --- | --- | --- | --- |
| `com.local-operator.mobile` | `<lop root>/bin/python3` | `Local Operator [mobile daemon] port=4098` | `-m local_operator.mobile.service --port 4098` | `~/Library/Logs/local-operator/mobile.log` |
| `com.local-operator.browser` | `<lop root>/bin/python3` | `Local Operator [browser bridge] port=4099` | `-m local_operator.browser_bridge.daemon --port 4099` | `~/Library/Logs/local-operator/browser-bridge.log` |
| `com.local-operator.tunnel` | `<lop root>/bin/python3` | `Local Operator [tunnel]` | `-m local_operator.tunnels.service` | `~/.local-operator/tunnel/service.log` |
| `com.local-operator.wakes` | `<lop root>/bin/python3` | `Local Operator [wakes]` | `-m local_operator.wakes.supervisor` | `~/.local-operator/logs/wake-supervisor.log` |

`<lop root>` is the stable install root, `~/.local/share/lop` (`stable_root()`
in `local_operator/update.py`), whose `current` entry names the generation in
use.

The `Program` column is the **shim** `~/.local/share/lop/bin/python3` on every
job, and that is the live value rather than a placeholder: measured here on
2026-09-21, all four `com.local-operator.*` plists carry
`Program = /Users/damian/.local/share/lop/bin/python3` — what
`daemon_image_path()` returns, i.e. `<lop root>/bin/python3`, a POSIX shell
script with no signature of its own. The image an EDR sees is what the shim
execs: the current generation's `<generation>/tools/local-operator/bin/Local
Operator`, a binary named `Local Operator` (§2, §3).

The `<prefix>`/`<install env>` distinction belongs to the **pre-generation
render**, not to the live plists. `launchd_job` names the branded link
`<prefix>/bin/Local Operator` where no shim can be planted — `<prefix>` is the
install prefix of whichever CLI installed the job, `<install env>` the
environment the installing component runs from (the desktop app's Application
Support tree, below) — and, with no branded link either, drops `Program`
entirely and puts the image in `ProgramArguments[0]`. The uv-tool prefix of that
older shape survives in the dated `…wakes.plist.bak-…` record below, in
`ProgramArguments[0]`, with no `Program` key.

The image/argv split is deliberate, not incidental. The four installers all
spread `procname.launchd_job(...)` — callers `mobile/install.py`,
`browser_bridge/install.py`, `tunnels/install.py`, `wakes/install.py` — and
that renderer sets `Program` to the image it will name and `ProgramArguments[0]`
to the role label precisely so that launchd can execute a file named `Local
Operator` while passing the array as argv. It names the shim where the
generation layout is present (`supervised_image()`), and the branded link
otherwise. This snippet is that branch: the shape a machine with no generation
layout renders **when a branded link can be planted** — with none it renders the
argv-only fallback above instead:

```python
return {
    "Program": str(link),
    "ProgramArguments": [branded_argv0(label) if label else BRAND, "-m", module, *args],
}
```

With only `ProgramArguments` set, launchd uses element 0 as **both** the image
and argv[0], which is why the pre-branding shape had the branded image path
there and every daemon collapsed to one indistinguishable `Local Operator` row
in launchd's own listing. An EDR rule or a vendor report keyed on
`ProgramArguments[0]` therefore reads a role label, not a binary; the image is
`Program`.

What each one is for: the **mobile** job serves the phone portal, the
**browser** job serves the browser-bridge daemon that the paired browser
extension talks to, the **tunnel** job maintains the outbound connector to the
remote-access service (it dials out; nothing inbound is opened), and the
**wakes** job is a supervisor that starts a session runtime when a scheduled
wake comes due.

The keys that matter to a behavioural engine, all rendered in the same module
that writes them (`render_plist` in `local_operator/mobile/install.py`,
`local_operator/browser_bridge/install.py`, `local_operator/wakes/install.py`
and the value dict in `local_operator/tunnels/install.py`):

- `RunAtLoad: true` on all four — they start at login without further action.
- `KeepAlive: {SuccessfulExit: false}` on all four — a crashed job is
  restarted by launchd; a job that exits cleanly stays down. For the wakes
  supervisor that is deliberate self-retirement: it exits 0 when there is
  nothing left to supervise.
- `ProcessType: Interactive` on mobile and browser (they hold
  long-lived connections), `ThrottleInterval: 10` on tunnel, and
  `StartInterval: 900` on wakes (a bounded self-heal re-check every 15
  minutes).
- `EnvironmentVariables: {LOCAL_OPERATOR_CONFIG_DIR: <config dir>}` on the
  mobile, tunnel and wakes jobs, so those three are pinned to a specific store
  (the browser bridge's plist carries no such key).

Labels are not always literally the four above: the browser bridge derives its
label from the config root (`label()` in
`local_operator/browser_bridge/install.py`), so a non-default config root
appends a suffix and a plist named accordingly. A scan keyed on the exact four
filenames will miss the extra one; keying on the `com.local-operator.` prefix
does not.

Only one of the four — **the wakes supervisor** — is expected on a minimal
install, and even that one appears only once something asks it to fire. The
other three are written by explicit CLI commands — `lop mobile install`,
`lop browser install`, `lop tunnel install` — and never by the act of
installing the package; `wakes` is different in that it installs and repairs
itself whenever a wake is scheduled (`lop wake create`, or an agent arming a
wake), because there is nothing for a supervisor to do until then. That basis
is the code (`ensure_supervisor_installed` in `local_operator/wakes/install.py`,
reached from the wake-creating path in `local_operator/cli.py`, against the
four explicit `install()` entry points elsewhere), not a reproduced minimal
install: the machine this document was measured on has all four jobs, and no
fresh minimal install was built to confirm the count.

Verified on this machine rather than assumed: reading
`~/Library/LaunchAgents/*.plist` gives exactly the four jobs above, and
`launchctl list` shows them loaded alongside the desktop app's own rows
(`com.local-operator.ShipIt`, `application.com.local-operator.*`), which belong
to the app bundle and its updater rather than to the CLI installer.

### Where the program image lives: the shim, and the prefixes behind it

`Program` is an absolute path to the program image. On a machine with the
generation layout it is the **shim**, `<lop root>/bin/python3`, the same path for
all four jobs — what an EDR sees as the process image is what that shim execs,
the current generation's `<generation>/tools/local-operator/bin/Local
Operator` (§2, §3). The shim is a machine-level artefact, so it does not depend
on how the CLI was installed.

The **install prefix** decides what the pre-generation render names, and it is
what `sys.prefix`, `.lop-source` and the update machinery key on
(`install_kind()` in `local_operator/update.py` sorts uv-tool, pipx and
plain-pip installs; a generation install reports `UV_TOOL`, because its prefix
carries its own `uv-receipt.toml` — which is why the layout has to be named
here):

- **a generation install (what `lop-update` and `lop update` both produce)**:
  `~/.local/bin/lop` is a symlink to `~/.local/share/lop/current/bin/lop`, and
  `current` names one generation root,
  `~/.local/share/lop/generations/<stamp>-<sha-or-version>/`.
  The install prefix (`sys.prefix`) is that root's `tools/local-operator/`, so the
  branded image is `<generation>/tools/local-operator/bin/Local Operator`. Resolve
  the pointer (`readlink ~/.local/share/lop/current`) rather than quoting a
  generation name, which changes at every update; `.lop-source` in that same tree
  records which build it is (`<sha> <ref>` for a git build, `pypi <version>` for a
  wheel).
- **uv tool**: `~/.local/share/uv/tools/local-operator/`, with console scripts in
  that tree's `bin/` — the pre-generation layout, which `uv tool install --force`
  writes in place; a machine that only ever installed through the generation
  layout has no such tree at all.
- **pipx**: `~/.local/pipx/venvs/local-operator/`.
- **plain pip**: the venv prefix or the base interpreter that received the
  package, wherever the user's toolchain put it.

`Local Operator` is **not** a separate binary. It is a hardlink to the Python
interpreter of that environment (`local_operator/procname.py`): the documented
anti-pattern `[sys.executable, "-m", module]` is what makes the OS notify
"python3 is running in the background" on install, so the installer plants a
second name for the interpreter's inode instead. §2 covers what that hardlink
costs us; §5 covers replacing it.

One shape has no `Program` key at all, and it is worth recognising because it
is what a pre-branding plist looks like. When no branded link can be planted,
`launchd_job` falls back to `launchd_program`'s argv alone — element 0 is the
image path, there is no `Program` key, and launchd uses element 0 for both
roles. That is the shape of the stale `…wakes.plist.bak-…` copy discussed
below. A scanner that reads only `ProgramArguments[0]` reads an image path on
such a plist and a role label on a branded one, which is exactly why the field
to read is `Program`.

### What decides the name an endpoint tool sees

Two independent readers answer "what is this process called", and they are
**not** decided by the same thing. Measured on macOS 25.6.0 (2026-09-19, six
calls per pid, stable within each run):

| Reader | Source | Decided by |
| --- | --- | --- |
| Activity Monitor, `proc_name()`, `ps -o ucomm` / `-o comm` | `p_comm` | the **basename of the path used at `execve`** |
| `ps -o args`, `top -o command`, `pgrep -f` | `argv[0]` | whatever the parent passed (world-readable) |
| `proc_pidpath()` — the file path an EDR builds its threat name from | the vnode path of the executable **inode** | **not reliably**: for a hardlinked inode it returns whichever link name the VFS cached |

Measured, on four live processes of this product:

```console
$ ps -o pid=,ucomm=,args=  # abbreviated; full series in the 2026-09-19 audit
pid 42028  ucomm=Local Operator   exec'd through <gen>/bin/Local Operator   proc_pidpath=<uv>/cpython-3.14.3-…/bin/python3.14
pid 19821  ucomm=Local Operator   exec'd through <gen>/bin/Local Operator   proc_pidpath=<uv>/cpython-3.14.3-…/bin/python3.14
pid  1343  ucomm=python3.14       exec'd through <gen>/bin/python3 (symlink) proc_pidpath=<uv>/cpython-3.14.3-…/bin/python3.14
pid 60013  ucomm=python3.14       exec'd through <uv-tool>/bin/python3 (symlink) proc_pidpath=<uv>/python3.14
```

Two consequences an admin needs, and neither is obvious from the plists alone:

- **The exec path is what decides the reported name, so it is what an exclusion
  has to cover.** `p_comm` — the Activity Monitor column, and the axis the
  2026-09-19 incident was about — is fixed by the path the process was executed
  *through*: a symlink is resolved by the kernel (so `…/bin/python3` reports
  `python3.14`), while a hardlink is not (so `…/bin/Local Operator` reports
  `Local Operator`). Both processes above run the same interpreter build.
- **`proc_pidpath()` is NOT a reliable way to name a hardlinked image**, so a
  file-path exclusion written only against `*/bin/Local Operator` is
  incomplete: the branded file and the interpreter it is a second name for are
  one inode, and the path an EDR reports for that inode is whichever link name
  the VFS cached. Write the exclusion for `*/bin/Local Operator` **and** for the
  interpreter paths the same inode answers to (see §4).

**The plant is a hard link, and that has a blast radius worth stating plainly.**
Because the branded `Local Operator` file and the interpreter are one inode,
anything a security tool does to the file — quarantine, ownership change, mode
change, content replacement — reaches the interpreter every venv on the machine
resolves to. Measured on this machine, 2026-09-19 ~23:44 local:

```console
$ ls -la ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3.12
-rw-------  3 _sentinel  _sentinel  49968 …  python3.12
$ ~/local-operator/.venv/bin/python -c pass
/bin/bash: …/.venv/bin/python: Permission denied
```

Both `cpython-3.12.13` and `cpython-3.13.12` were found in that state (mode
`0600`, the agent user's ownership, the interpreter's own byte size) and every
3.12/3.13 virtualenv on the host — the development venvs, whose `bin/python` is
a symlink to exactly that binary — was dead with `Permission denied`. Recovery
is one command per version, and it worked:

```console
$ uv python install --reinstall 3.12.13     # same version, fresh inode, 0755
 Installed Python 3.12.13 in 3.90s
$ uv python install --reinstall 3.13.12
```

**The cause is not proven, and this document will not claim one.** What is
recorded is the state, the timestamp and the recovery. The reading that fits is
that the remediation an EDR applies to a flagged `Local Operator` file reaches
the interpreter through the shared inode; against that reading, the same
`_sentinel`-owned shape had already been measured on five worktree *links*
before this session, and `cpython-3.13.12` was found damaged on a machine where
no process of ours had executed a 3.13 binary at all — so it may be ambient
scanning of the interpreter tree rather than a reaction to an execution. Waking
the same file to a mode our code chose would be the same mistake from the other
side; nothing in `procname.py` writes modes or ownership, and nothing should
(the module docstring records the measurement: a `chmod` on the link once broke
an unrelated worktree's console script, because the mode belongs to the shared
inode).

A future change could reduce this blast radius: planting a per-venv **copy** of
the interpreter is viable now that the mandatory `lib/libpython3.X.dylib`
symlink is planted beside it (the failure the module docstring records for a
copy is the missing dylib, not the copy shape; §5 already measures a detached
copy executing beside a venv-like `lib/`). That would cost ~50 KB per venv and
replace the inode staleness check with a content one, and it is deliberately NOT
done in the change that added this measurement — it is its own decision with its
own measurements.

### The config directory

`~/.local-operator/` unless `LOCAL_OPERATOR_CONFIG_DIR` says otherwise. It holds
the config file, the session and analytics stores, the attachments, the caches,
and an encrypted long-term secret store at `~/.local-operator/secrets/` with a
local broker socket (`broker.sock`) that other parts of the product talk to.
The store's *contents* are deliberately out of scope here; the fact that a
local broker exists and where it lives is not sensitive.

### The desktop app's managed Python environment

When the desktop app (not the CLI) is used, the app provisions its own Python
environment instead of relying on a global `lop`:

- `~/Library/Application Support/Local Operator/managed-python/packaged/runtimes/<id>/`
  — a standalone CPython runtime, copied at provisioning time out of the seed
  shipped inside the notarized app bundle (`seedPath()` plus a `ditto` copy in
  the desktop app's `managed-python.ts`), not fetched at install time.
- `~/Library/Application Support/Local Operator/managed-python/packaged/environments/<id>/`
  — a venv built on top of that runtime, containing the product.
- `~/Library/Application Support/Local Operator/managed-python/packaged/selected-environment.json`
  — which runtime and venv are active, and the backend version they contain.
- `~/Library/Application Support/Local Operator/local-operator-venv` — the
  older, pre-split layout, which machines installed before the change still
  have.

The products in those environments are also named `Local Operator` (same
hardlink mechanism), and the Electron helpers run with
`--user-data-dir=/Users/…/Library/Application Support/Local Operator`. The app's
*backend* is a separate question, and on this machine it is not the Application
Support interpreter: measured on 2026-09-21 with the app bundle running, the live
backend is the app's own child running the **generation** tree's interpreter,
`<generation>/tools/local-operator/bin/python -c "from local_operator.cli import
main; main()" serve --port 1111`, while the environment above is present and
selected (`managed-python/packaged/selected-environment.json`) with no process
observed running it. Running a backend out of Application Support is a normal
Electron/Mac pattern, but it is one of the signals §3 is about.

This is the artefact class whose *signature* differs from the CLI's, which
matters here more than the path does: these images carry the app's Developer ID
and its team, not CPython's anonymous ad-hoc signature. §2 states both classes
and the evidence for each.

### The notifier bundle (built at runtime, inside the config directory)

`~/.local-operator/notifier/LocalOperator.app` is a real, persisted,
compiled-then-executed bundle, and it is the one artefact whose paths sit
**inside the config directory** rather than in an install prefix:

- It is built at runtime with `clang` by `build_bundle` in
  `local_operator/tui/notifier_app/__init__.py`, into
  `bundle_root(config_dir)` = `<config dir>/notifier/LocalOperator.app` — so
  under the default config dir, inside `~/.local-operator/`. The module
  documents why it is not shipped in the wheel's `site-packages`: that tree may
  be read-only, is replaced wholesale on upgrade, and is shared between
  environments.
- It is then ad-hoc signed in place (`codesign --force --sign -`), and its
  bundle identifier is `BUNDLE_ID` in that same module — spelled literally in
  the source, anonymised here as `me.<individual>.localoperator`. This is the
  one artefact whose identifier is personalised rather than product-named —
  every other bundle here is `com.local-operator` — so a rule written against
  the product's bundle identifier does not cover it.
- A `.build-stamp` file beside it makes the build once-only, so the bundle
  persists across sessions rather than being compiled per run.

A freshly written Mach-O that is then executed out of the config directory is
precisely the kind of item §3 is about, which is why it is in §4's exclusion
list as well.

### Installs compare before they rewrite, and restart only when something needs it

Every installer treats an existing plist as a stale artefact rather than as
state to trust, because the plist embeds an **absolute interpreter path** that
an upgrade invalidates. What they do about it changed with this branch:

- mobile, browser bridge and tunnel now compare the rendering with the file on
disk (the tunnel also compares its mode — 0600 is that installer's own) and,
when they are equal, neither write nor reload. The generation shim made the
plist path stable, so a re-install is normally a no-op: it used to rewrite
identical bytes and then `bootout` + `bootstrap` the job, which is two signals
an EDR reads as "Persistence: launchd job / plist file modification".
- When the bytes differ they write and reload exactly as before. A job that is
loaded but not running is repaired with `launchctl kickstart -k` rather than by
rewriting the file, and the reload is also taken when the daemon is not
answering — `health` for mobile and the bridge, the connector's own
`/_lop_tunnel/health` on `127.0.0.1:<gateway_port>` for the tunnel.
- **Which installers verify, and which report launchd's verdict — not the same
thing.** Mobile (a 20 s `_serving` loop) and the browser bridge (a health loop)
re-probe after loading, so their success means a daemon that is answering. The
tunnel does not re-probe: it reports what `launchctl` said, so a successful
`lop tunnel install` means the unit is LOADED. The unanswering state is what
makes it reinstall in the first place, but nothing asks again afterwards, so
"install succeeded" is not evidence the gateway came back.
- Before this branch the write and the `bootout` + `bootstrap` were
unconditional in all three, and a **fresh** install still writes and loads,
which is the state this section describes for an admin reading a newly
installed machine.
- What this does **not** claim: the 2026-09-19 incident's plist-modification
indicators came from `lop update --refresh-daemons`, whose repair path
(`launchd.rewrite_if_stale`) already compared content before writing, and from a
plist that was genuinely stale — a real repair, not an identical write.
- wakes compares the rendered plist with the file it finds; if they differ it
  rewrites and re-bootstraps, and if the plist matches but launchd has the
  label stopped it repairs with `launchctl kickstart -k` rather than leaving a
  dead job that merely prints as loaded.

This is observable on a real upgrade path, with one honest qualification. This
machine has both `~/Library/LaunchAgents/com.local-operator.wakes.plist` and a
`…wakes.plist.bak-…` copy beside it. The backup names
`~/.local/share/uv/tools/local-operator/bin/Local Operator` in
`ProgramArguments[0]` and has no `Program` key; the live plist names the
generation **shim**, `~/.local/share/lop/bin/python3`, in `Program` (§1). Same
label, rewritten program path — which is exactly what an upgrade looks like to
an EDR watching that file.

The qualification: **that backup is an observation on the author's machine, not
a documented upgrade mechanic.** Nothing at this revision writes a `.bak-…`
file — no writer exists in this repository or in the desktop app's sources — so
treat the code above as the mechanism and the backup as a leftover of unknown
provenance. Its value here is narrower and still real: it is a genuine example
of the pre-branding plist shape (the `ProgramArguments[0]`-is-the-image form
described above), which is otherwise only reachable in the fallback branch.

## 2. How the shipped code is signed today (two captures predate the generation layout)

Measured on this machine, not assumed. There are **two artefact classes**, and
they are signed differently. The distinction is load-bearing rather than
pedantic: an exclusion keyed on a certificate behaves differently depending on
which class it names, and §3 and §5 both rest on it.

- The **CLI artefact** — a uv-tool, pipx or plain-pip install — is ad-hoc
  signed by the linker. There is no identity, no team, and nothing to key an
  exclusion on.
- The **app-provisioned artefacts** — anything the desktop app installs under
  `~/Library/Application Support/Local Operator/` — carry the *same Developer ID
  and team* as the app bundle itself.

### Two artefacts sit on the `lop` command path, and only one is signed

An admin who starts from the command they type (`lop …`) and runs `codesign` on
it gets a different answer than the one below, so name both artefacts.

Both this block and Class 1's below are the **pre-generation capture**, kept
verbatim as the record: the uv-tool tree they name no longer exists on this
machine (verified 2026-09-21), so re-running them fails — the `head -1` exits 1
with `No such file or directory`, and the `ls -l` resolves to
`~/.local/share/lop/current/bin/lop` instead. Substitute
`$(readlink ~/.local/share/lop/current)/tools/local-operator` for that tree; the
signing answers do not depend on the path.

```console
$ ls -l "$HOME/.local/bin/lop"
lrwxr-xr-x  /Users/…/.local/bin/lop -> /Users/…/.local/share/uv/tools/local-operator/bin/lop
$ head -1 "$HOME/.local/share/uv/tools/local-operator/bin/lop"
#!/Users/…/.local/share/uv/tools/local-operator/bin/python3

$ codesign -dv "$HOME/.local/bin/lop"
/Users/…/.local/bin/lop: code object is not signed at all        # exit 1
$ spctl -a -v -t exec "$HOME/.local/bin/lop"
/Users/…/.local/bin/lop: rejected
source=no usable signature                                       # exit 3
```

Those are the correct answers for a shell **script**: `bin/lop` is Python source
with a shebang, so there is no Mach-O to sign and nothing for Gatekeeper to
assess. The artefact that carries a signature is the **interpreter image that
shebang names** — the next section. In a vendor report, ``bin/lop`` is best
described as "an unsigned script whose interpreter is ad-hoc signed"; calling
the `lop` path itself "ad-hoc signed" describes a file that has no signature at
all.

### Class 1: the CLI interpreter (the daemon artefact)

Pre-generation capture as above — this tree is gone too; re-measured unchanged
against the current generation's image on 2026-09-21.

```console
$ codesign -dv --verbose=2 "$HOME/.local/share/uv/tools/local-operator/bin/Local Operator"
Executable=/Users/…/.local/share/uv/tools/local-operator/bin/Local Operator
Identifier=-
Format=Mach-O thin (arm64)
CodeDirectory v=20400 size=520 flags=0x20002(adhoc,linker-signed) hashes=13+0 location=embedded
Signature=adhoc
Info.plist=not bound
TeamIdentifier=not set
Sealed Resources=none
Internal requirements=none

$ spctl -a -v -t exec "$HOME/.local/share/uv/tools/local-operator/bin/Local Operator"
/Users/…/.local/share/uv/tools/local-operator/bin/Local Operator: rejected
```

What each line means for an EDR:

- `Signature=adhoc` with `flags=…(adhoc,linker-signed)` — this is not an
  unsigned binary in the "tampered" sense; an ad-hoc signature is what the
  linker emits on Apple silicon, where a signature is mandatory. It is
  nonetheless a signature **about bytes only**: it certifies nothing about who
  produced the code.
- `Identifier=-` and `TeamIdentifier=not set` — there is no signing identifier
  and no team. There is no stable, verifiable subject to key an exclusion on.
- `Info.plist=not bound`, `Sealed Resources=none`, `Internal requirements=none`
  — nothing outside the executable itself is sealed, so the identity cannot
  extend to the Python code the process then imports.
- `spctl … rejected` — Gatekeeper rejects it. An organisation using
  Gatekeeper-based deployment controls will see it as unidentified.

The hardlink does not change any of this, and cannot: the same `codesign -dv`
on the environment's own `bin/python3.14` prints the identical fields and
resolves `Executable=` to
`~/.local/share/uv/python/cpython-3.14.3-macos-aarch64-none/bin/python3.14`,
because a hardlink is a second name for that inode, not a second file. Whatever
the name says, the signature is CPython's.

### Class 2: the app-provisioned interpreters (Application Support)

The images the desktop app installs are **not** in Class 1 — and on this machine
they are not what any plist names either (§1: all four `com.local-operator.*`
plists name the generation shim, whose exec target is the CLI tree). Measured
against the app's own environment image:

```console
$ codesign -dv --verbose=2 "$HOME/Library/Application Support/Local Operator/managed-python/packaged/environments/<id>/bin/Local Operator"
Executable=/Users/…/Library/Application Support/Local Operator/managed-python/packaged/environments/<id>/bin/Local Operator
Identifier=python3
Format=Mach-O thin (arm64)
CodeDirectory v=20500 size=467 flags=0x10000(runtime) hashes=4+7 location=embedded
Signature size=8973
Authority=Developer ID Application: <individual> (SHA2U6KT7V)
Authority=Developer ID Certification Authority
Authority=Apple Root CA
Timestamp=Sep 15, 2026 at 2:50:22 PM
Info.plist=not bound
TeamIdentifier=SHA2U6KT7V
Runtime Version=14.2.0
Sealed Resources=none
Internal requirements count=1 size=168

$ codesign --verify --verbose=2 "…/bin/Local Operator"
…: valid on disk
…: satisfies its Designated Requirement

$ spctl -a -v -t exec "…/bin/Local Operator"
…: rejected (the code is valid but does not seem to be an app)
```

Read the difference from Class 1 field by field, because it is the whole point
of this section:

- `Identifier=python3` and `Internal requirements count=1` — the binary's own
  identity is still CPython's, not the product's. The *name* `Local Operator`
  and the *signature subject* still disagree; only the signature's authority
  changes.
- `flags=0x10000(runtime)` — a hardened runtime, which an ad-hoc linker
  signature does not carry.
- `Authority=Developer ID Application: …` with `TeamIdentifier=SHA2U6KT7V` —
  the same team ID as the app bundle, and `codesign --verify` reports it
  *satisfies its Designated Requirement*. This is not an ad-hoc signature.
- `spctl … rejected` persists, but the reason is now
  `(the code is valid but does not seem to be an app)`. That is Gatekeeper's
  *app* assessment declining a bare Mach-O, which is a different statement from
  "unidentified code", and an admin comparing the two classes should not read
  it as one.

The legacy `~/Library/Application Support/Local Operator/local-operator-venv/bin/Local Operator`
reports the same authority and team. The hardlink mechanic is identical in both
classes — this image's inode is shared with
`…/managed-python/packaged/runtimes/<id>/bin/python3.12` — and that shared inode
is what §5 builds on.

**Provenance, and the edge of what was measured here.** The app does not sign
these images: `managed-python.ts` in the desktop app copies the runtime out of
the notarized bundle with `ditto` and says so in as many words — *"Nested Mach-O
signatures travel with the copy. Never sign, repair or execute the seed here"* —
and it verifies each Mach-O with `codesign --verify --strict` before accepting a
seed, rejecting one with no signed binaries. So the signature is **inherited
from the provisioning copy**, not locally applied. That provenance is read from
that source, not measured here; what was measured on this machine is the
signature *on the artefact*. The two agree in a way worth recording, because
`runtimes/<seed identity>-<uuid>` is deliberately one directory per provisioning
*generation*: a runtime provisioned under an older seed keeps that seed's
signature, and indeed this environment's image differs in signature byte count
and signing timestamp from the seed now inside the app bundle, which is what
that design predicts.

Every `codesign` block in this document is reproduced as captured, with no line
trimmed from what the command printed on the macOS the capture was taken on
(25.6), so a reader can diff it field by field against their own machine. A
newer macOS adds trailing fields these captures predate — on 27.0 both
Developer-ID blocks also print `Total signatures=1` and `Chosen signature=1`,
which the ad-hoc Class 1 image does not — and an app bundle that has been
replaced since the capture moves its `Timestamp` and `Sealed Resources … files=`
count (this machine's bundle now reads `Sep 20, 2026 at 10:14:27 PM` and
`files=1562`, where the block below records the earlier build). The app-bundle
block follows the same rule.

### The desktop app bundle

```console
$ codesign -dv --verbose=2 "/Applications/Local Operator.app"
Executable=/Applications/Local Operator.app/Contents/MacOS/Local Operator
Identifier=com.local-operator
Format=app bundle with Mach-O thin (arm64)
CodeDirectory v=20500 size=446 flags=0x10000(runtime) hashes=3+7 location=embedded
Signature size=8972
Authority=Developer ID Application: <individual> (SHA2U6KT7V)
Authority=Developer ID Certification Authority
Authority=Apple Root CA
Timestamp=Sep 15, 2026 at 9:07:50 PM
Notarization Ticket=stapled
Info.plist entries=32
TeamIdentifier=SHA2U6KT7V
Runtime Version=26.5.0
Sealed Resources version=2 rules=13 files=1840
Internal requirements count=1 size=180

$ spctl -a -v "/Applications/Local Operator.app"
/Applications/Local Operator.app: accepted
source=Notarized Developer ID
```

The app is the artefact an organisation *can* key on: a Developer ID
signature, a Team ID, a hardened runtime and a stapled notarization ticket,
which Gatekeeper accepts. The name printed against every `Authority=` line is a
placeholder (`<individual>`): this document is committed to a public repository,
so the individual's name is not reproduced here, but the **team ID is literal**
and is what an exclusion keys on.

Three honest qualifications.

**First, that certificate is personal, not an organisation's.** It is a
Developer ID issued to an individual, so an exclusion written against it is tied
to one person's Apple account rather than to the company, and it will not
survive a change of maintainer. §5 says what fixes that.

**Second, a certificate exclusion on macOS is not a cheap way to key on it.**
SentinelOne's exclusion-mode documentation is explicit: *"macOS — The
certificate exclusion is a performance focus exclusion. It disables monitoring
of the excluded processes, in addition to suppressing alerts."* So an
organisation that keys on `SHA2U6KT7V` to silence a daemon false positive
silently stops monitoring the whole Electron app, its helpers and the managed
Python environment with it — strictly more than the path-scoped exclusion §4
recommends, and it reaches further than the detection that prompted it. (Those
are vendor-documented semantics, not measurements: SentinelOne is installed on
this machine, but its policy state is root-only and was not escalated to, so no
exclusion could be observed firing — see §4.)

**Third, and the correction that matters most in this section, the app being
signed does not sign the interpreters it provisions — but it does not leave them
unsigned either.** The app-provisioned images under Application Support carry
the app's Developer ID and team (Class 2 above), so a certificate rule *can*
cover the daemons' program image on a desktop-app install. What a rule written
for the app bundle does not cover is the **CLI install path**: the uv-tool, pipx
and plain-pip images are ad-hoc signed with no team at all (Class 1). One
product, two signing states, and the state depends on how the user installed it.

## 3. The behaviour pattern that trips behavioural engines

No single item below is malicious, and several are ordinary macOS practice.
What behavioural engines score is the **combination**, in one product, in one
install:

1. **Persistence.** Four per-user LaunchAgents with `RunAtLoad`, one of them
   (`wakes`) also on a 15-minute interval, all of them restarting on an
   unsuccessful exit. Registering a supervisor daemon *is* persistence; that
   property is inherent to a product that may have to do something while the
   user is not looking. It cannot be hidden — the point is that it should be
   legible and expected.
2. **An updater that installs new code.** The desktop app updates itself
   through Squirrel: it downloads a release and has launchd run the installer
   as the submitted job `com.local-operator.ShipIt`
   (`…/Squirrel.framework/Resources/ShipIt`, visible in `launchctl print`),
   which installs a new bundle into `/Applications`. The app's own backend
   installer additionally writes a shell script into
   `$TMPDIR/local-operator-install-*/` and executes it, and that script builds
   and installs a persistent Python environment under
   `~/Library/Application Support/Local Operator/`. (The desktop app is a
   separate bundle maintained in the `local-operator-ui` repository; the
   script-writing path is `src/main/backend/backend-installer.ts` there.)
3. **A managed-Python environment under Application Support.** The app
   *provisions* one, and a backend launched from it is running out of a location
   that is not a normal toolchain path. On this machine it is not what runs:
   measured on 2026-09-21 with the app bundle running, the live backend is the
   app's own child running the **generation** tree's interpreter,
   `<generation>/tools/local-operator/bin/python … serve --port 1111`, with the
   Application Support environment present and selected
   (`selected-environment.json`) and no process observed running it. What the app
   *supervises* points elsewhere too: the live `wakes` plist names the generation
   shim and the running supervisor is the generation's image, not an Application
   Support one. An installer run compares before it writes —
   `launchd.rewrite_if_stale` rewrites the plist only where the render differs
   from the file on disk, so a re-install that would change nothing changes no
   bytes — but the **reload is not gated on that write**: a file that is already
   current is still reloaded where the daemon is not answering (`kickstart`, and
   the bootstrap reload past it) (§1, "Installs compare before they rewrite").
   This machine shows what those states do to the supervised path: the stale
   `…wakes.plist.bak-…` (a leftover §1 flags as unknown provenance, but on this
   machine it is the old path) names the uv-tool prefix, while the live plist
   names the generation **shim** `~/.local/share/lop/bin/python3`, so a tool that
   watches the plist sees a supervised program being repointed out of the
   toolchain and into the product's own install root — a directory that is itself
   the kind of place persistence hides.
4. **A binary that is not what it is called.** The program the EDR sees is
   named `Local Operator` while its code signature's identity is CPython's,
   because it is a hardlink (§2). This is a genuine **name/identity
   disagreement by construction**, and it holds in both signing classes: the
   identifier is CPython's either way (`Identifier=python3` on an
   app-provisioned image, `Identifier=-` on a CLI one). What differs by class is
   how strong a signal it looks like. On this machine all four
   `com.local-operator.*` plists name the shim, and all four daemons resolve
   through it to the current generation's `…/bin/Local Operator`, whose
   `codesign -dv` prints `flags=…(adhoc,linker-signed)` with
   `TeamIdentifier=not set` — the supervised program here is a binary named one
   thing, signed by nobody. The app-provisioned image under Application Support
   prints `TeamIdentifier=SHA2U6KT7V` and passes `codesign --verify`, so an
   engine that reports *that* image has a mismatch between the file name and the
   signing identifier rather than an unidentified binary — it is the class-2
   artefact (§2), not what a plist on this machine names today. The naming is not
   concealment (it was adopted so Activity Monitor and the login-items list say
   something meaningful instead of `python3`), but an engine cannot tell the two
   motivations apart from the artefact alone.
   Note also what the naming does *not* fix: a name/identifier disagreement we
   control is still a disagreement an engine has to resolve on its own.

An install that persists itself, fetches new code, runs a temp-directory
installer, writes an executable environment into Application Support — a data
root, not an execution origin — and supervises daemons out of its own install
tree is, to a heuristic, indistinguishable from the pattern those heuristics
are designed to catch. That is the finding, and it is why the fix in §4 is an
allow-list rather than an argument.

## 4. The admin recipe

### Allow-list by path, not by hash

Exclusions for this product must be **path-scoped**, and the reason is not
convenience: what the exclusion would have to identify changes on every
release. The plists embed an absolute path to an interpreter that an upgrade
replaces (and an upgrade rewrites the plist when that path changed — a re-install
that would change nothing now writes nothing, but the rewrite case is real), and
the desktop app provisions its managed Python environment under a new path when
it updates. A hash-based exclusion would break at each upgrade and the install
would be quarantined again — which is the failure this document is trying to
stop.

Scope the exclusion to:

- the install prefix — `~/.local/share/lop/**` for the generation layout (the
  `current` pointer and every generation root the launcher resolves through),
  `~/.local/share/uv/tools/local-operator/**` for a pre-generation uv tool
  install, `~/.local/pipx/venvs/local-operator/**` for pipx, or the equivalent
  venv prefix for a plain pip install;
- **the branded image name, and the shared interpreter it is a hardlink of** —
  `*/bin/Local Operator` (which the plant creates in every tree the product runs
  from, including each `lop` generation) **and** the interpreter paths the same
  inode answers to, e.g.
  `~/.local/share/uv/python/cpython-*/bin/python3.*` and the uv-tool
  environment's own `…/bin/python3`. Both halves are needed: the exec path
  decides `p_comm`, while the path an EDR reports for the inode is whichever
  link name its cache holds (§1);
- the plist paths — `~/Library/LaunchAgents/com.local-operator.*` (the glob is
  deliberately `com.local-operator.*` rather than `*.plist`, so it also matches
  a label the browser bridge derives from a non-default config root, §1);
- for desktop-app installs, `/Applications/Local Operator.app/**` and
  `~/Library/Application Support/Local Operator/**` (the managed Python
  environment and the app's data directory);
- the notifier bundle the TUI builds at runtime —
  `~/.local-operator/notifier/LocalOperator.app/**` (§1). It sits **inside the
  config directory** rather than in an install prefix, and its bundle
  identifier is the personalised one, so a rule written only against
  `com.local-operator` patterns does not reach it.

Do **not** disable the persistence detection story globally to make this go
away. That trades a product-specific false positive for blindness to the
technique itself, and would not have been necessary here.

### What this recipe cannot cover, and why

Four shapes run under a `Local Operator` parent and are **not** named by any of
the above — an admin should expect them and exclude by path or ancestry rather
than by the product name:

- **Third-party MCP servers launched through `uv tool`/`uvx`** (measured:
  `uv tool uvx workspace-mcp …` and the interpreter it execs). The launcher is
  another product's binary and execs its own interpreter; our spawn site hands
  it no label, and it could not honour one — the name is decided by the path
  that process uses at `execve` (§1), which is `uv`'s.
- **Framework interpreters** (`/opt/homebrew/…/Python.framework/…`). A branded
  hardlink of one reports `Python`, not the link name, because the framework's
  `bin/python3.x` is a stub that re-executes the real binary inside
  `Resources/Python.app` — the second exec is what the kernel names. An
  exclusion has to cover the framework path itself.
- **Anything an agent's `bash` tool starts** (`pytest`, `python`, a QA rig, a
  build). The child's `p_comm` is fixed at its own `execve` by `bash`, and on
  macOS a hardlink to the SIP-protected `/bin/bash` cannot be executed, so this
  is attributable by ancestry or by path exclusion, not by renaming.
- **The desktop app's `-c` identity probe** (`python3 -c "from
  local_operator.cli import main; main()" serve …`). It is deliberately left
  unbranded: the app launches its backend that way and then asks the SAME `-c`
  string to report `sys.executable`, refusing a backend whose base interpreter
  does not match — a branded answer would break the app's own backend on every
  machine (`local_operator/procname.py`, `is_own_launch`).

### Name the exclusion mode: `Suppress Alerts`, on **all engines**

The exclusion *mode* is the part of this recipe that decides whether it
suppresses a false positive or stops monitoring the software, and the default is
the one this scenario needs. Written out, because it is the safety boundary:

- **`Suppress Alerts` (the default) — use this.** Per SentinelOne's
exclusion-mode documentation: *"When you exclude files or folders with default
path exclusions, Agents monitor the files and processes but do not show alerts
in the Console and do not mitigate detections. This also applies to detections
in threat groups whose root process is in the excluded path or file."* That
second sentence is what makes this recipe work for the case this document
exists for: the detection is **of the install itself**, whose root process is
inside the excluded path, and it is exactly the case a root-process clause
covers. Set the engine scope to **All engines** (the default), not to a single
engine.
- **`Interoperability` / `Performance Focus` — do not use.** They *"reduce
monitoring and mitigation of the excluded items"*; the mode documentation puts
it more bluntly — Interoperability *"reduces the monitoring level on the
excluded processes, in addition to suppressing alerts"* and Performance Focus
*"disables monitoring of the excluded processes"*. Following this section while
selecting one of those trades a false positive for blindness to
`~/.local/share/lop/**`, the app bundle and the managed
Python environment — the opposite of what SECURITY.md promises.
- On **macOS agents 4.6+** the vendor support matrix lists `Interoperability` as
**No** and `Performance Focus` as **Yes**, so on a current agent the only
alternative an admin can pick is the monitoring-disabling one. There is no
"more compatible" middle setting to fall back to here.
- A **certificate** exclusion is the same trap on macOS: *"macOS — The
certificate exclusion is a performance focus exclusion. It disables monitoring
of the excluded processes, in addition to suppressing alerts."* §2 explains why
that matters for anything keyed on the developer certificate.
- If a genuine interoperability problem is diagnosed later, that is a different
conversation with different evidence: SentinelOne's own FAQ recommends
*"consult with SentinelOne Support before using Interoperability or Performance
exclusions"*.

Those mode semantics are **vendor-documented, not measured here**. This host has
a SentinelOne agent installed, but its policy state is root-only and was not
escalated to, so no detection and no exclusion could be observed firing. What is
reproducible locally is §1's inventory and §3's behaviour pattern; the classifier's
verdict on it is not, and this document does not claim it.

The quotations above are what was actually read, and they are worth naming as
sources rather than paraphrasing:

- SentinelOne's exclusion-mode text as reproduced by a SentinelOne partner
  knowledge base: the mode definitions and the per-OS agent support matrix
  quoted above
  (`https://support.guardz.com/en/articles/13429659-path-exclusion-modes-in-detail`),
  and the `Suppress Alerts` root-process clause and the macOS certificate line
  (`https://support.guardz.com/en/articles/13429787-additional-notes-for-exclusion-configuration-via-sentinelone-console`).
  The root-process clause — the one this recipe leans on — has a second
  partner copy
  (`https://support.guardz.com/en/articles/10807055-creating-a-path-exclusion-for-sentinelone`),
  and the partner page whose own source note credits SentinelOne's community
  documentation (its link: community article `000006818`) is
  `https://support.guardz.com/en/articles/10807589-path-exclusions-best-practices-for-sentinelone`.
- SentinelOne's own FAQ, for the "consult with SentinelOne Support before using
  Interoperability or Performance exclusions" line
  (`sentinelone.com/faq`).

The same mode text appears in the SentinelOne console's exclusion-mode help,
which is the first-party copy; it was not reachable from here (the console needs
a tenanted login and the community pages render client-side), so the partner-KB
pages above are the copies this document was written against.

### Filing the false positive with the vendor

Report it as a behavioural false positive with the evidence attached:

- the four plist paths and their contents (or §1's inventory), and the
  `Program` value of each — that key, not `ProgramArguments[0]`, is the field
  that names the image, which matters because the analyst will scan the argv
  array first;
- the install prefix, and the `Local Operator`-named image inside it, called
  out explicitly as a **hardlink to the environment's CPython**, so the
  analyst can see in one step why the name and the signature disagree;
- the paths in `~/Library/Application Support/Local Operator` for a
  desktop-app install, including the fact that the directory is the app's data
  root and what it provisions, not an execution origin — the helper processes
  run from the app bundle and the generation tree;
- **per-release file hashes**, so the vendor can distinguish our releases from
  anything else using the same names.

Nothing has to be invented for that last item: PyPI publishes a SHA-256 for
every uploaded file and exposes it through its JSON API.

```console
$ curl -s https://pypi.org/pypi/local-operator/json | python3 -c 'import json,sys; d=json.load(sys.stdin); v=d["info"]["version"]; [print(f["filename"], f["digests"]["sha256"]) for f in d["releases"][v]]'
local_operator-<version>-py3-none-any.whl <sha256>
local_operator-<version>.tar.gz            <sha256>
```

### After a quarantine has already deleted an install

This is the failure people actually hit, and the order matters. Reinstalling
into a policy that has not changed simply gets the new install quarantined as
well, so step 2 is what makes step 1 stick. If the reinstall is removed anyway,
apply the exclusion first — these exclusions are path-scoped and do not require
the file to exist — and then reinstall.

1. **Reinstall**, so there is something to allow:
   - the `lop` CLI: `lop-update` installs a committed revision into a new
     generation. It takes its ref from its argument and defaults to the **local**
     `main`, and it performs **no** remote comparison, so a checkout whose local
     `main` is stale builds an old commit and installs it: `git fetch origin
     main` and confirm `git rev-parse --short main` equals `git rev-parse --short
     origin/main` first, or name the ref explicitly — `lop-update <sha>`.
   - uv tool: `uv tool install --force local-operator`
   - pipx: `pipx install --force local-operator`
   - desktop app: reinstall the app bundle.
2. **Apply the path-scoped exclusion** from the previous section.
3. **Verify the jobs are loaded and running**, because a quarantined delete
   usually took the plists with it or left them pointing at a path that no
   longer exists:

   ```console
   $ launchctl list | grep com.local-operator
   # nothing here: the jobs are gone and the plists with them
   7480	0	com.local-operator.mobile      # loaded and running
   -	0	com.local-operator.wakes       # loaded, nothing running behind it
   $ lop mobile install && lop browser install   # re-registers and restarts
   $ lop wake status                             # whether the supervisor is live
   ```

   A job that is loaded but stopped prints with a `-` in the PID column rather
   than a number — that is a job launchd knows about and nothing is running
   behind, not a healthy one.

## 5. What would remove the need for exclusions

Two pieces of work, stated as work rather than intent:

- **Sign and notarize with a Developer ID issued to the company** — an
  organisation Apple Developer account rather than the personal certificate in
  use today, so the identity an admin keys an exclusion against belongs to the
  company and outlives any one maintainer. The gap is narrower than "the
  artefacts are unsigned", and worth stating precisely after §2: the desktop
  app bundle is signed and notarized, the app-provisioned interpreter images
  under Application Support carry that same Developer ID and team, but the
  **CLI install path carries no identity signature an admin can key on**
  (Class 1), and where a Developer ID is present it is an individual's, not
  the organisation's.
  A company certificate closes both halves: the CLI artefact gains an identity
  to key on, and the identity an admin keys on becomes one the organisation
  owns. Note the cost before reaching for it as a shortcut, though — on macOS a
  certificate exclusion is a performance-focus exclusion (§2), so it disables
  monitoring of everything it covers rather than only suppressing alerts, and
  it reaches past the detection that prompted it. The path-scoped exclusion
  above stays the narrower answer even once this ships.
- **Ship a small signed launcher instead of the renamed interpreter.** A
  launcher or daemon-runtime binary we compile, sign with that Developer ID and
  notarize would make the process an EDR sees our own code, and would let the
  program image carry our signature and our team identifier. The constraint that
  motivates this is narrower than "a hardlink cannot be signed separately", and
  it is worth spelling out, because the obvious objection is one experiment
  wide: `codesign -f` on a hardlinked name does **not** refuse it — it detaches
  the name into a new, independently signed inode and leaves the interpreter's
  inode untouched. Measured on a copy of the interpreter in a scratch
  directory, so the real install was not touched:

  ```console
  $ ls -li ./orig ./branded        # before: one inode, two names
  1153987269 -rwxr-xr-x@ 2 ./branded
  1153987269 -rwxr-xr-x@ 2 ./orig
  $ codesign -s - -i com.local-operator.test -f ./branded
  ./branded: replacing existing signature
  $ ls -li ./orig ./branded        # after: the signed name is its own inode
  1153987330 -rwxr-xr-x@ 1 ./branded
  1153987269 -rwxr-xr-x@ 1 ./orig
  $ codesign -dv ./orig            # the interpreter's signature is intact
  Executable=/private/tmp/…/orig
  Identifier=-
  Format=Mach-O thin (arm64)
  CodeDirectory v=20400 size=520 flags=0x20002(adhoc,linker-signed) hashes=13+0 location=embedded
  Signature=adhoc
  Info.plist=not bound
  TeamIdentifier=not set
  Sealed Resources=none
  Internal requirements=none
  ```

  The detached copy even runs when a venv-like `lib/` sits beside it, which is
  the layout the install prefix already has (`lib/libpython3.14.dylib` resolves
  through `@rpath`): the same copy executed from a directory without a `lib/`
  beside it fails exactly as `procname.py` documents, with
  `dyld: Library not loaded: @rpath/libpython3.14.dylib`. So the real constraint
  is: **the installer plants a hardlink**, which means whatever signature sits on
  that inode is CPython's, and a separately signed image would be replaced by a
  fresh hardlink at the next install — the signature would not survive the next
  `lop … install`, `lop-update`, or app update. Two further reasons point the
  same way: signing a copy would still be signing *CPython*, so the process an
  EDR sees remains someone else's code with our name on it, and the copy would
  need its own update path every time the interpreter changes. A launcher we
  compile is the version of this that holds.

Until both have shipped, path-scoped exclusions are the supported answer, and
this document is the evidence to justify them.

One further change belongs on this list, and it is about the blast radius rather
than about detection: **plant a per-venv copy of the interpreter instead of a
hardlink to it.** It would not stop an EDR flagging the name, but it would stop
whatever an EDR does to that name from reaching the interpreter every venv on the
machine resolves to through the shared inode (measured, §1: two uv interpreters
left unreadable on 2026-09-19, every 3.12/3.13 venv on the host dead until
reinstalled). The cost is stated with it: ~50 KB per venv, and a
content-staleness check where the inode comparison is today. It is not done
yet, and the reason to note it here rather than only in an issue is that the
copy shape's known failure — `dyld: Library not loaded: @rpath/libpython3.X.dylib`
— is exactly what the mandatory `lib/libpython3.X.dylib` symlink fixes, so the
objection to it has already been measured away.

## 6. Where to find this

- [`../SECURITY.md`](../SECURITY.md) links here from its endpoint-protection
  section — that is the file an admin or security reviewer opens first.
- [`../README.md`](../README.md) links here from its safety model.

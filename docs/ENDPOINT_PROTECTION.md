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
its own installer module, and each job runs the same interpreter image with a
different module:

| Label | `ProgramArguments` | Log |
| --- | --- | --- |
| `com.local-operator.mobile` | `Local Operator -m local_operator.mobile.service --port 4098` | `~/Library/Logs/local-operator/mobile.log` |
| `com.local-operator.browser` | `Local Operator -m local_operator.browser_bridge.daemon --port 4099` | `~/Library/Logs/local-operator/browser-bridge.log` |
| `com.local-operator.tunnel` | `Local Operator -m local_operator.tunnels.service` | `~/.local-operator/tunnel/service.log` |
| `com.local-operator.wakes` | `Local Operator -m local_operator.wakes.supervisor` | `~/.local-operator/logs/wake-supervisor.log` |

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
  tunnel and wakes jobs, so those two are pinned to a specific store.

Labels are not always literally the four above: the browser bridge derives its
label from the config root (`label()` in
`local_operator/browser_bridge/install.py`), so a non-default config root
appends a suffix and a plist named accordingly. A scan keyed on the exact four
filenames will miss the extra one; keying on the `com.local-operator.` prefix
does not.

Only one of the four exists on a minimal install. They are registered on demand
by the CLI — `lop mobile install`, `lop browser install`, `lop tunnel install` —
and the wakes job installs and repairs itself whenever a wake is scheduled
(`lop wake create`, or an agent arming a wake), because there is nothing for a
supervisor to do until something asks it to fire.

Verified on this machine rather than assumed: reading
`~/Library/LaunchAgents/*.plist` gives exactly the four jobs above, and
`launchctl list` shows them loaded alongside the desktop app's own rows
(`com.local-operator.ShipIt`, `application.com.local-operator.*`), which belong
to the app bundle and its updater rather than to the CLI installer.

### Where the program image lives (the install prefix)

`ProgramArguments[0]` is always an absolute path to a binary named
`Local Operator`. It lives in the install prefix, which depends on how the CLI
was installed (`install_kind()` in `local_operator/update.py` distinguishes
them):

- **uv tool** (the documented end-user path): `~/.local/share/uv/tools/local-operator/`,
  with console scripts in that tree's `bin/`.
- **pipx**: `~/.local/pipx/venvs/local-operator/`.
- **plain pip**: the venv prefix or the base interpreter that received the
  package, wherever the user's toolchain put it.

`Local Operator` is **not** a separate binary. It is a hardlink to the Python
interpreter of that environment (`local_operator/procname.py`): macOS names a
non-bundle login item by the basename of `ProgramArguments[0]`, and the
documented anti-pattern `[sys.executable, "-m", module]` is what makes the OS
notify "python3 is running in the background" on install. §2 covers what that
hardlink costs us; §5 covers replacing it.

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
  — a standalone CPython runtime the app downloads.
- `~/Library/Application Support/Local Operator/managed-python/packaged/environments/<id>/`
  — a venv built on top of that runtime, containing the product.
- `~/Library/Application Support/Local Operator/managed-python/packaged/selected-environment.json`
  — which runtime and venv are active, and the backend version they contain.
- `~/Library/Application Support/Local Operator/local-operator-venv` — the
  older, pre-split layout, which machines installed before the change still
  have.

The products in those environments are also named `Local Operator` (same
hardlink mechanism), and the app's backend and helper processes are launched
**from inside `~/Library/Application Support/Local Operator`** — for example
`.../managed-python/packaged/environments/<id>/bin/python -c "from
local_operator.cli import main; main()" serve --port 1111`, and the Electron
helpers with `--user-data-dir=/Users/…/Library/Application Support/Local
Operator`. Executing helper processes out of Application Support is a normal
Electron/Mac pattern, but it is one of the signals §3 is about.

### Installs rewrite and restart their own job at upgrade time

Every installer treats an existing plist as a stale artefact rather than as
state to trust, because the plist embeds an **absolute interpreter path** that
an upgrade invalidates:

- mobile and browser write the plist unconditionally and then
  `launchctl bootout` + `launchctl bootstrap` it, so the old registration is
  replaced rather than left running against a path that no longer exists.
- tunnel does the same (`bootout` then `bootstrap`, tolerating a missing unit).
- wakes compares the rendered plist with the file it finds; if they differ it
  rewrites and re-bootstraps, and if the plist matches but launchd has the
  label stopped it repairs with `launchctl kickstart -k` rather than leaving a
  dead job that merely prints as loaded.

This is observable on a real upgrade path: this machine has both
`~/Library/LaunchAgents/com.local-operator.wakes.plist` and a
`…wakes.plist.bak-…` copy beside it. The backup names
`~/.local/share/uv/tools/local-operator/bin/Local Operator`; the live plist
names the desktop app's managed-python interpreter. Same label, rewritten
program path — which is exactly what an upgrade looks like to an EDR watching
that file.

## 2. How the shipped code is signed today

Measured on this machine, not assumed. The CLI/daemon artefact is ad-hoc
signed by the linker and carries **no identity**; the desktop app is properly
signed and notarized.

### The interpreter the console script runs (the daemon artefact)

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

### The desktop app bundle

```console
$ codesign -dv --verbose=2 "/Applications/Local Operator.app"
Executable=/Applications/Local Operator.app/Contents/MacOS/Local Operator
Identifier=com.local-operator
Format=app bundle with Mach-O thin (arm64)
CodeDirectory v=20500 size=446 flags=0x10000(runtime) hashes=3+7 location=embedded
Authority=Developer ID Application: Damian Tran (SHA2U6KT7V)
Authority=Developer ID Certification Authority
Authority=Apple Root CA
Timestamp=Sep 15, 2026 at 9:07:50 PM
Notarization Ticket=stapled
TeamIdentifier=SHA2U6KT7V
Sealed Resources version=2 rules=13 files=1840

$ spctl -a -v "/Applications/Local Operator.app"
/Applications/Local Operator.app: accepted
source=Notarized Developer ID
```

The app is the artefact an organisation *can* key on: a Developer ID
signature, a Team ID, a hardened runtime and a stapled notarization ticket,
which Gatekeeper accepts.

Two honest qualifications. First, that certificate is a **personal**
Developer ID issued to an individual, not a Developer ID issued to an
organisation — so an exclusion written against it is tied to one person's Apple
account rather than to the company, and it will not survive a change of
maintainer. §5 says what fixes that. Second, the app being signed does not
sign the interpreters it provisions into Application Support: the managed
environment's `Local Operator` is the same ad-hoc-signed hardlink, so a rule
written for the app bundle does not cover the daemons' program image.

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
3. **Execution from inside Application Support.** On a desktop-app install,
   the app's backend and the wakes supervisor both run interpreters living
   under `~/Library/Application Support/Local Operator/…`, not from a normal
   toolchain location. When an installer next runs, it rewrites its own plist
   and re-registers the job (§1), so the path launchd is supervising moves
   inside that directory on upgrade — a change an EDR watching the plist sees
   as a supervised program being repointed.
4. **A binary that is not what it is called.** The program the EDR sees is
   named `Local Operator` while its code signature is an ad-hoc CPython
   signature, because it is a hardlink (§2). This is a **masquerading signal by
   construction**: on this machine the launchd job's `program` is
   `…/bin/Local Operator` and `codesign -dv` on it prints CPython's own
   linker-signed identity with no team. There is no way to make that image
   carry our signature — it is the same inode as the interpreter. The naming is
   not concealment (it was adopted so Activity Monitor and the login-items list
   say something meaningful instead of `python3`), but an engine cannot tell
   the two motivations apart from the artefact alone.

An install that persists itself, fetches new code, runs a temp-directory
installer, writes an executable environment into Application Support and
supervises daemons out of that directory is, to a heuristic, indistinguishable
from the pattern those heuristics are designed to catch. That is the finding,
and it is why the fix in §4 is an allow-list rather than an argument.

## 4. The admin recipe

### Allow-list by path, not by hash

Exclusions for this product must be **path-scoped**, and the reason is not
convenience: what the exclusion would have to identify changes on every
release. The plists are rewritten by every installer run, they embed an
absolute path to an interpreter that an upgrade replaces, and the desktop app
provisions its managed Python environment under a new path when it updates. A
hash-based exclusion would break at each upgrade and the install would be
quarantined again — which is the failure this document is trying to stop.

Scope the exclusion to:

- the install prefix — `~/.local/share/uv/tools/local-operator/**` for a uv
  tool install, `~/.local/pipx/venvs/local-operator/**` for pipx, or the
  equivalent venv prefix for a plain pip install;
- the four plist paths — `~/Library/LaunchAgents/com.local-operator.*.plist`;
- for desktop-app installs, `/Applications/Local Operator.app/**` and
  `~/Library/Application Support/Local Operator/**` (the managed Python
  environment and the data directory the app executes helpers from).

Do **not** disable the persistence detection story globally to make this go
away. That trades a product-specific false positive for blindness to the
technique itself, and would not have been necessary here.

### Filing the false positive with the vendor

Report it as a behavioural false positive with the evidence attached:

- the four plist paths and their contents (or the inventory in §1);
- the install prefix, and the `Local Operator`-named image inside it, called
  out explicitly as a **hardlink to the environment's CPython**, so the
  analyst can see in one step why the name and the signature disagree;
- the paths in `~/Library/Application Support/Local Operator` for a
  desktop-app install, including the fact that helper processes execute from
  there;
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
  company and outlives any one maintainer. The desktop app bundle already
  carries a Developer ID signature and a stapled notarization ticket; the CLI
  and daemon artefacts do not.
- **Ship a small signed launcher instead of the renamed interpreter.** A
  launcher or daemon-runtime binary we compile, sign with that Developer ID and
  notarize would make the process an EDR sees our own code, and would let the
  program image carry our signature and our team identifier. Today's naming
  cannot: the image is a hardlink to the interpreter's inode, so it can never be
  signed separately, and until that changes every install depends on an
  exclusion an admin has to write by hand.

Until both have shipped, path-scoped exclusions are the supported answer, and
this document is the evidence to justify them.

## 6. Where to find this

- [`../SECURITY.md`](../SECURITY.md) links here from its endpoint-protection
  section — that is the file an admin or security reviewer opens first.
- [`../README.md`](../README.md) links here from its safety model.

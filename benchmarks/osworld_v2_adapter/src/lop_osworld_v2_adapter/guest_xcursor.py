"""Apply upstream's X-display leak fix to the guest's action server before an episode.

THE MEASURED CAUSE (2026-10-26). The released 2.1 guest image
(``ami-01017272139e01feb``, built 2026-03-27) ships an action server whose
``/screenshot`` handler opens an X display through ``pyxcursor.Xcursor()`` and
never closes it. Every screenshot leaks one X client connection and one file
descriptor in the server process:

* stock guest, read-only ``/proc/<pid>/fd`` counts: +50 fds for 50
  ``/screenshot`` calls, +102 per 100 on a live episode's guest; ``/platform``
  and ``/terminal`` add nothing;
* at roughly 480-500 leaked connections the X session dies (X clients counted
  over ``ss -x`` fell from 383 to 2 in one 3 s sample), the action server exits,
  and every systemd restart fails at import with ``Can't connect to display
  ":0"`` / ``Invalid MIT-MAGIC-COOKIE-1 key``. The unit's start limit and the
  guest watchdog's no-effect cap then leave nothing listening on :5000, so the
  host sees ``connection refused`` with the instance still ``running``.

That is the ``bridge-wedged`` terminal that ended 80 of 191 sealed OSWorld 2.1
episodes (42%), clustered at 200-300 steps because the harness takes about two
screenshots per step. All three diagnostic shapes are this one death seen from
different moments: a request lost as the server died (``outcome is unknown``),
a request refused after it died (``not_sent``), and a statement that failed
while the display was going away (``input may be partial``).

THE FIX IS UPSTREAM'S, BYTE FOR BYTE. OSWorld fixed exactly this in
xlang-ai/OSWorld PR #500 ("fix xserver connections leak", merge
``8253399dfa8b3c1d5783a73c29a91f11f7c6d7c3``, 2026-05-07), after the image was
built. The PR has two halves:

* ``desktop_env/server/pyxcursor.py`` gains ``close()``, ``__del__``,
  ``__enter__`` and ``__exit__``, so an ``Xcursor`` closes its display when the
  object is collected;
* ``desktop_env/server/main.py`` wraps four ``Display`` users in
  ``with``/``finally``.

Only the first half is applied here, and that is enough. The stock handler
assigns ``cursor_obj = Xcursor()`` to a local, so CPython's reference counting
collects it when the request returns, and the new ``__del__`` closes the
display. Measured on a bare stock guest with only this file replaced: 30
screenshots moved the server's fd count from 15 to 15. ``main.py``'s other
``Display()`` users (``/screen_size``, ``/window_size``, ``/start_recording``)
are not on the per-step path; the harness calls none of them per step. The
main.py refactor is therefore NOT taken -- it would rewrite a 68 KB file the
benchmark ships for no measured gain.

WHAT THIS IS NOT. It changes no task setup, no evaluator, no step or wall cap,
and no observation content: on a static screen a stock screenshot and a patched
screenshot are pixel-identical apart from the top-bar clock minute (measured,
bbox ``(1030, 8, 1038, 18)`` and nothing once that strip is masked). It is an
infrastructure-stability correction to the harness's own environment, and it is
a DEVIATION FROM THE STOCK AMI that every record discloses
(``guest-xcursor-patch.json``, see ``docs/benchmarks/osworld_2/README.md``).

HOW IT LANDS, and why the action server needs no restart. The stock server runs
``app.run(debug=True)``, so werkzeug's stat reloader watches every imported
module and re-execs the serving child when one changes. Replacing the file is
therefore the whole operation: the reloader picks it up (measured 0.8 s) before
the first observation, and no privileged command or ``systemctl`` call is
needed. The file belongs to the ``user`` account the server runs as, so the
write itself is unprivileged. The replacement is atomic (temp file, fsync,
``os.replace``) so the reloader can never import a half-written module.

GATES, ALL FAIL-OPEN AND ALL RECORDED. The patch is applied only when the file
on the guest is the exact stock file (sha256 ``0d8e3071...``), so a different
image, an already-patched guest or an upstream server built after the fix is
left alone and recorded as such. The guest then has to prove the reload: the
serving process for ``/home/user/server/main.py`` must be a new PID. Every
outcome -- applied, skipped, failed -- is written as
``guest-xcursor-patch.json`` beside ``guest-preparation.json`` with
``applied: true|false``, the upstream PR, the merge SHA, the before and after
file digests and the reason. An episode whose patch did not land still runs, on
the stock guest, and says so; there is no silent no-op.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass, field
from importlib import resources
from typing import Callable, Literal, Sequence

from lop_osworld_v2_adapter.guest_disk import CommandResult

#: Where the stock image keeps the action server's cursor helper. Read from the
#: running server's own argv on this AMI (``/usr/bin/python
#: /home/user/server/main.py``); main.py imports it as ``from pyxcursor import
#: Xcursor``, so it resolves from the script's directory.
GUEST_PATH = "/home/user/server/pyxcursor.py"

#: The upstream fix this module applies, by identity. The PR number and merge
#: commit are what a reader needs to audit the deviation against upstream.
UPSTREAM_REPOSITORY = "xlang-ai/OSWorld"
UPSTREAM_PR = 500
UPSTREAM_MERGE_SHA = "8253399dfa8b3c1d5783a73c29a91f11f7c6d7c3"
UPSTREAM_PATH = "desktop_env/server/pyxcursor.py"

#: The file the released image ships (2.1 AMI ``ami-01017272139e01feb``), read
#: back from a bare guest. It is upstream's pre-fix file
#: (``ef43297b`` parent of the fix) plus one trailing blank line; nothing else.
STOCK_SHA256 = "0d8e3071049614b480e417a58a512c277050034bdb9d3bc01a3ba4b0e7f76137"

#: ``desktop_env/server/pyxcursor.py`` at ``UPSTREAM_MERGE_SHA``, vendored
#: byte-for-byte as package data and checked against this digest at load.
PATCHED_SHA256 = "87d09a465bf87c3766ac3b8584f80c3288890e874b45d9a56b811b0a4268e840"

#: The vendored file's name inside this package. Not a ``.py`` name on purpose:
#: it is guest data, never imported here (it needs numpy and libX11), and a
#: module name would put it in front of the linters and type checkers as if it
#: were adapter code. Shipped via ``[tool.setuptools.package-data]``.
VENDORED_RESOURCE = "pyxcursor-8253399d.py.upstream"

#: Evidence file name, beside ``guest-preparation.json`` in the episode cache.
EVIDENCE_FILENAME = "guest-xcursor-patch.json"

#: Budgets. The whole pass is bounded so preparation can never outlive setup:
#: one read, one write, then polling for the reloader's new serving process.
#: The reload measured 0.8 s; ``RELOAD_WAIT_S`` is ~12x that.
COMMAND_TIMEOUT_S = 15.0
RELOAD_WAIT_S = 10.0
RELOAD_POLL_S = 0.5

Status = Literal["applied", "already-patched", "unrecognised", "failed", "unreachable"]

#: Reports the file's digest and the action server's serving PIDs. The serving
#: processes are found by argv (``.../server/main.py``) rather than by unit,
#: because the reloader's child is not the unit's MainPID: werkzeug's parent
#: stays, the child is re-exec'd. Read-only; prints one JSON object.
_INSPECT_SOURCE = r"""
import hashlib, json, os, sys
path = sys.argv[1]
try:
    digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
except OSError as error:
    digest = None
pids = []
for entry in os.listdir('/proc'):
    if not entry.isdigit():
        continue
    try:
        argv = open('/proc/%s/cmdline' % entry, 'rb').read().split(b'\0')
    except OSError:
        continue
    if any(a.endswith(b'/server/main.py') for a in argv):
        pids.append(int(entry))
print(json.dumps({'sha256': digest, 'server_pids': sorted(pids)}))
"""

#: Atomic replacement: write a sibling temp file, fsync, keep the original's
#: mode, rename over. A reloader that stats mid-write must never import a
#: partial module. It refuses unless the file is STILL the stock file, so a
#: race with anything else that touched it cannot be overwritten blindly.
_WRITE_SOURCE = r"""
import base64, hashlib, json, os, sys
path, expected, payload = sys.argv[1], sys.argv[2], base64.b64decode(sys.argv[3])
current = hashlib.sha256(open(path, 'rb').read()).hexdigest()
if current != expected:
    print(json.dumps({'written': False, 'sha256': current}))
    raise SystemExit(0)
tmp = path + '.lop-tmp'
with open(tmp, 'wb') as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(tmp, os.stat(path).st_mode & 0o7777)
os.replace(tmp, path)
print(json.dumps({'written': True,
                  'sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest()}))
"""

GuestCommand = Callable[[Sequence[str], float], CommandResult]
Clock = Callable[[], float]
Sleep = Callable[[float], None]


@dataclass(frozen=True)
class PatchReport:
    """What happened to the guest's cursor helper, in disclosure-ready form."""

    status: Status
    reason: str
    sha256_before: str | None = None
    sha256_after: str | None = None
    server_pids_before: tuple[int, ...] = ()
    server_pids_after: tuple[int, ...] = ()
    reload_s: float | None = None
    steps: tuple[str, ...] = field(default=())

    @property
    def applied(self) -> bool:
        """True only when the patched module is what the server now serves.

        ``already-patched`` counts: the leak fix is in effect either way, and
        the record must say so rather than read as a failure. ``applied`` is
        the field a score report keys its disclosure on.
        """

        return self.status in ("applied", "already-patched")

    def to_json(self) -> dict[str, object]:
        return {
            "schema": "lop-guest-xcursor-patch-v1",
            "applied": self.applied,
            "status": self.status,
            "reason": self.reason,
            "upstream": {
                "repository": UPSTREAM_REPOSITORY,
                "pr": UPSTREAM_PR,
                "merge_sha": UPSTREAM_MERGE_SHA,
                "path": UPSTREAM_PATH,
            },
            "guest_path": GUEST_PATH,
            "stock_sha256": STOCK_SHA256,
            "patched_sha256": PATCHED_SHA256,
            "sha256_before": self.sha256_before,
            "sha256_after": self.sha256_after,
            "server_pids_before": list(self.server_pids_before),
            "server_pids_after": list(self.server_pids_after),
            "reload_s": self.reload_s,
            "steps": list(self.steps),
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_json(), indent=2, sort_keys=True).encode("utf-8")


def patched_source() -> bytes:
    """The vendored upstream file, refused unless it hashes to ``PATCHED_SHA256``.

    A corrupted or hand-edited vendored copy must never reach a guest: the
    whole claim of this module is "upstream's exact bytes".
    """

    data = resources.files(__package__).joinpath(VENDORED_RESOURCE).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PATCHED_SHA256:
        raise ValueError(f"vendored pyxcursor digest {digest} != pinned {PATCHED_SHA256}")
    return data


def _inspect(run: GuestCommand) -> tuple[str | None, tuple[int, ...]]:
    result = run(["python3", "-c", _INSPECT_SOURCE, GUEST_PATH], COMMAND_TIMEOUT_S)
    if result.returncode != 0:
        raise RuntimeError("inspect command failed")
    data = json.loads(result.stdout.strip().splitlines()[-1])
    digest = data.get("sha256")
    pids = data.get("server_pids")
    if not isinstance(pids, list) or not all(type(p) is int for p in pids):
        raise ValueError("inspect returned no pid list")
    return (digest if isinstance(digest, str) else None), tuple(pids)


def apply_xcursor_patch(
    run: GuestCommand,
    *,
    clock: Clock = time.monotonic,
    sleep: Sleep = time.sleep,
) -> PatchReport:
    """Replace the guest's leaking ``pyxcursor.py`` with upstream's fix, or say why not.

    Never raises: every failure is a ``PatchReport`` with ``applied`` False and
    a reason, and the caller runs the episode on the stock guest. Transport
    exception text is never recorded (it can echo the guest address); only
    its type is.
    """

    steps: list[str] = []
    try:
        payload = patched_source()
    except Exception as error:  # noqa: BLE001 - recorded, never raised
        return PatchReport(status="failed", reason=f"vendored file invalid: {type(error).__name__}")
    try:
        before, pids_before = _inspect(run)
        steps.append("inspect")
    except Exception as error:  # noqa: BLE001 - recorded, never raised
        return PatchReport(status="unreachable", reason=f"inspect: {type(error).__name__}")
    if before == PATCHED_SHA256:
        return PatchReport(
            status="already-patched",
            reason="guest already carries the upstream fix",
            sha256_before=before,
            sha256_after=before,
            server_pids_before=pids_before,
            server_pids_after=pids_before,
            steps=tuple(steps),
        )
    if before != STOCK_SHA256:
        # A different image or a locally modified server: replacing a file we
        # have not measured would be a change nobody reviewed.
        return PatchReport(
            status="unrecognised",
            reason="guest file is not the measured stock file; left unchanged",
            sha256_before=before,
            server_pids_before=pids_before,
            steps=tuple(steps),
        )
    if not pids_before:
        return PatchReport(
            status="failed",
            reason="no running action server to reload",
            sha256_before=before,
            steps=tuple(steps),
        )
    try:
        result = run(
            [
                "python3",
                "-c",
                _WRITE_SOURCE,
                GUEST_PATH,
                STOCK_SHA256,
                base64.b64encode(payload).decode("ascii"),
            ],
            COMMAND_TIMEOUT_S,
        )
        steps.append("write")
        written = (
            json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else {}
        )
    except Exception as error:  # noqa: BLE001 - recorded, never raised
        # The outcome of a lost write is unknown, so look before claiming
        # anything: the inspect below reports whatever is really on disk.
        written = {}
        steps.append(f"write-error:{type(error).__name__}")
    started = clock()
    after: str | None = None
    pids_after: tuple[int, ...] = ()
    deadline = started + RELOAD_WAIT_S
    while True:
        try:
            after, pids_after = _inspect(run)
        except Exception:  # noqa: BLE001 - the server is mid-reload; retry
            after, pids_after = None, ()
        reloaded = bool(pids_after) and bool(set(pids_after) - set(pids_before))
        if after == PATCHED_SHA256 and reloaded:
            steps.append("reloaded")
            return PatchReport(
                status="applied",
                reason="upstream fix written; server reloaded it",
                sha256_before=before,
                sha256_after=after,
                server_pids_before=pids_before,
                server_pids_after=pids_after,
                reload_s=round(clock() - started, 3),
                steps=tuple(steps),
            )
        if clock() + RELOAD_POLL_S > deadline:
            break
        sleep(RELOAD_POLL_S)
    if after == PATCHED_SHA256:
        reason = "file written but the server did not reload it within the wait"
    elif written.get("written") is False:
        reason = "guest file changed before the write; left unchanged"
    else:
        reason = "file was not replaced"
    return PatchReport(
        status="failed",
        reason=reason,
        sha256_before=before,
        sha256_after=after,
        server_pids_before=pids_before,
        server_pids_after=pids_after,
        steps=tuple(steps),
    )

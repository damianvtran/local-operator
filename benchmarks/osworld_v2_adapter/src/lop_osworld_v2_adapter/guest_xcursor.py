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
outcome is written as ``guest-xcursor-patch.json`` beside
``guest-preparation.json`` with the tri-state ``applied`` (``true`` patched,
``false`` provably stock, ``null`` write attempted but unconfirmed -- see
``PatchReport.applied``), the status, the upstream PR, the merge SHA, the
before/after file digests and the reason. An episode whose patch did not land
still runs, and its record says which guest it ran on -- or that this is not
known; there is no silent no-op, and "stock" is never claimed after a write
unless the stock digest was observed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
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
#: back from a bare guest. It is upstream's pre-fix file -- the fix merge's
#: parent ``36a6adf44f0255eb0cf2a591470ce06bfc52c8b2`` -- plus one trailing
#: blank line; nothing else.
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
#: The reload measured 0.8 s; ``RELOAD_WAIT_S`` is ~12x that. ``RELOAD_WAIT_S``
#: bounds when polling STOPS, not each poll: the worst case is one inspect
#: (15 s) + one write (15 s) + ~10 s of fast-failing polls + one last inspect
#: that started just before the deadline (15 s), about 55 s, all inside
#: ``reset_start``. The typical pass measured ~1-2 s.
COMMAND_TIMEOUT_S = 15.0
RELOAD_WAIT_S = 10.0
RELOAD_POLL_S = 0.5

Status = Literal[
    "applied", "already-patched", "unrecognised", "failed", "unreachable", "unconfirmed"
]

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
try:
    with open(tmp, 'wb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, os.stat(path).st_mode & 0o7777)
    os.replace(tmp, path)
finally:
    # A failed write (ENOSPC on the ~93%-full image) must not leave the
    # temp file in the server directory; after os.replace it is already gone.
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
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
    # ``None`` for both ``*_after`` fields means NO post-write inspection
    # succeeded: either no write was attempted (``write_attempted`` False, the
    # guest was left as inspected) or every inspection after it failed. It never
    # means "the server is gone" -- an empty PID list from a successful
    # inspection is the only way to say that.
    sha256_after: str | None = None
    server_pids_before: tuple[int, ...] = ()
    server_pids_after: tuple[int, ...] | None = None
    write_attempted: bool = False
    reload_s: float | None = None
    steps: tuple[str, ...] = field(default=())

    @property
    def applied(self) -> bool | None:
        """Whether the episode's guest carries the deviation: a TRI-STATE.

        * ``True`` -- the patched module is what the server serves (``applied``
          proved by a new serving PID with the patched digest, or
          ``already-patched``: the fix is in effect either way).
        * ``False`` -- the guest is the stock guest, PROVABLY: no write was
          attempted, the guest's own pre-write check refused it, or a
          post-write inspection observed the stock digest.
        * ``None`` -- ``unconfirmed``: a write was attempted and its effect was
          never verified, so the guest may serve either module (a reloader or
          a later restart can import a patched file at any point). A score
          reader must treat this as "deviation possible", never as stock.

        ``applied`` is the field a score report keys its disclosure on.
        """

        if self.status == "unconfirmed":
            return None
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
            "server_pids_after": (
                None if self.server_pids_after is None else list(self.server_pids_after)
            ),
            "write_attempted": self.write_attempted,
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

    # By path beside this module, NOT ``importlib.resources``: the worker
    # imports adapter modules through discovery's ``_VerifiedSourceLoader``,
    # which serves hashed source and has no resource reader, so
    # ``resources.files(__package__)`` raises "Can't open orphan path" there
    # (measured on the first validation episode: the patch was recorded
    # ``applied: false`` for exactly that). The file is RECORD-covered, so the
    # package digest attests it, and the pin below re-checks it regardless.
    data = (Path(__file__).resolve().parent / VENDORED_RESOURCE).read_bytes()
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

    Never raises: every failure is a ``PatchReport`` with a status and a reason.
    Before a write is attempted a failure is ``applied`` False (the guest is
    stock, as inspected). AFTER one is attempted, "stock" is claimed only when
    a later inspection observed the stock digest (or the guest's own pre-write
    check refused the write); anything else is ``unconfirmed`` with
    ``applied`` None, because a write whose response was lost -- the reloader
    re-execs the very child that answers ``/execute`` -- may well have landed.
    Transport exception text is never recorded (it can echo the guest
    address); only its type is.
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
            server_pids_before=pids_before,
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
    written: dict[str, object] = {}
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
        steps.append("write" if result.returncode == 0 else f"write-rc:{result.returncode}")
        if result.returncode == 0:
            parsed = json.loads(result.stdout.strip().splitlines()[-1])
            if isinstance(parsed, dict):
                written = parsed
    except Exception as error:  # noqa: BLE001 - recorded, never raised
        # The outcome of a lost write is unknown: the response can be lost
        # because the write SUCCEEDED and the reloader re-exec'd the child
        # serving it. Only the inspections below may decide.
        steps.append(f"write-error:{type(error).__name__}")
    if written.get("written") is False:
        # The guest's own check found a file that is no longer the stock one,
        # and replaced nothing: whatever is there was not put there by us.
        return PatchReport(
            status="failed",
            reason="guest file changed before the write; left unchanged",
            sha256_before=before,
            server_pids_before=pids_before,
            write_attempted=True,
            steps=tuple(steps),
        )
    started = clock()
    # The LAST SUCCESSFUL observation, not the last attempt: a failed inspect
    # proves nothing about the file and must not erase what an earlier one saw.
    observed: tuple[str | None, tuple[int, ...]] | None = None
    deadline = started + RELOAD_WAIT_S
    while True:
        try:
            observed = _inspect(run)
            steps.append("inspect-after")
        except Exception:  # noqa: BLE001 - the server is mid-reload; retry
            steps.append("inspect-after-error")
        if observed is not None:
            after, pids_after = observed
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
                    write_attempted=True,
                    reload_s=round(clock() - started, 3),
                    steps=tuple(steps),
                )
        if clock() + RELOAD_POLL_S > deadline:
            break
        sleep(RELOAD_POLL_S)
    if observed is None:
        return PatchReport(
            status="unconfirmed",
            reason=(
                "write attempted but no later inspection succeeded; the guest may "
                "serve the stock or the patched module"
            ),
            sha256_before=before,
            server_pids_before=pids_before,
            write_attempted=True,
            steps=tuple(steps),
        )
    after, pids_after = observed
    if after == STOCK_SHA256:
        # Observed, not assumed: the stock file is still on disk after the
        # write, so the stock module is the only one any process can import.
        status: Status = "failed"
        reason = "file was not replaced (stock digest observed after the write)"
    elif after == PATCHED_SHA256:
        status = "unconfirmed"
        reason = (
            "patched file is on disk but no reload was observed within the wait; "
            "the server may serve either module until it next reloads"
        )
    else:
        status = "unconfirmed"
        reason = "guest file is neither the stock nor the patched file after the write"
    return PatchReport(
        status=status,
        reason=reason,
        sha256_before=before,
        sha256_after=after,
        server_pids_before=pids_before,
        server_pids_after=pids_after,
        write_attempted=True,
        steps=tuple(steps),
    )

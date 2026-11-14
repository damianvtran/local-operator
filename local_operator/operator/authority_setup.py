"""Turn on phone approvals (operator authority) without a terminal — design v2 §6 (C3).

WHY THIS EXISTS. Approving an action from a phone needs a paired phone, and pairing
needs operator authority installed on this computer: the root-owned anchor the
runtime trusts (``trust.load_anchor``). Until now the only way to install it was a
terminal verb (``lop operator init``/``install``/``setup``), so a person who only
uses the desktop app or the TUI hit a dead end ("run `lop pair`"). This module is
the one function the desktop route (``POST /v1/desktop/operator/setup``) and the
TUI verb (``/mobile approvals on``) both call.

THE TRUST BOUNDARY IS THE OS ADMIN PROMPT, AND ONLY THAT (§6 items 1-3). The desktop
bearer is not an agent boundary (a same-user process can hold it), so a click on a
button is consent UX, not proof a person is there. What makes this safe is that the
one privileged write goes through :func:`elevate.run_elevated` — osascript's admin
dialog, UAC, or a polkit (pkexec) dialog — which a person must answer. So, enforced
HERE and not left to callers:

* there is NO stored-password parameter on this path (the CLI's ``--sudo-secret``
  route is untouched and out of scope: design v2 §10 Q4);
* a process that already has administrator rights is REFUSED, not shortcut — a root
  process could write the anchor with no prompt at all, which is exactly the
  "no human" install this path must never perform. On Windows that means an
  ELEVATED TOKEN (:func:`_privileged`): ``Start-Process -Verb RunAs`` from an
  elevated process shows no UAC consent, so it would be the same no-human install;
* pkexec is resolved from fixed system paths, never ``PATH``
  (:func:`elevate.pkexec_path`), so a planted ``~/.local/bin/pkexec`` cannot pass
  the "a GUI prompt can appear" gate;
* with ``terminal_sudo=False`` (the server), Linux must have pkexec: the ``sh -c
  sudo`` fallback has no TTY there and, under passwordless sudo, succeeds with no
  human. The TUI passes ``terminal_sudo=True`` and runs the prompt under
  ``App.suspend()`` so sudo owns the real terminal.

The worst an agent holding the bearer can do is RAISE an OS prompt a human must
answer — the same exposure ``admit_device`` already has.

THE ANCHOR HAS THE SHAPE ``init`` GIVES IT (``handlers._init``), with
``ADMISSION_LISTED``, not the way ``_setup`` does: every phone paired later still
needs its own admit prompt. And it is IDEMPOTENT: on a host whose anchor is already
usable nothing is created, staged or installed — replacing a working anchor would
invalidate every phone already paired. The install also pins the TARGET as read
before the prompt (absent, or its sha256), so an anchor another setup lands while
this prompt is open is never overwritten (R1-4).

THE BYTES THE HUMAN APPROVES ARE THE BYTES BUILT HERE (R1-1). The staging file is
user-writable, so a same-user process holding the bearer could plant a FIFO there
and feed a read-back its own anchor; pinning the digest of what was READ would then
install the attacker's key behind a genuine prompt. So the anchor is serialised in
memory, its sha256 is the pin the privileged install verifies, the staged file is
written with ``O_NOFOLLOW``/``O_NONBLOCK`` and refused unless it is a regular file
we own whose bytes equal the in-memory bytes — checked again just before the
prompt — and after the install the INSTALLED anchor is read back and must name the
key we created with ``ADMISSION_LISTED``; the receipt records what was read.

THE AUDIT AND STATE FILES ARE EVIDENCE, NOT INTEGRITY PROOF (R1-6). They are 0600
files under the user's config directory, so any same-user process can rewrite or
delete them. They tell a person (and a support reader) what happened; nothing may
treat them as tamper-evident or make a trust decision from them — ``describe_state``
derives "on" from the root-owned anchor, never from the last attempt.

EVERY ATTEMPT IS AUDITED: one JSON line per receipt
(``proposed → consent → generated → installed → verified``) with the attempt id and
the actor (``desktop``/``tui``), appended to ``<config>/operator/authority-setup-audit.jsonl``;
the last attempt is also kept in ``authority-setup.json`` so a refresh can show
"Finish setup" after a cancelled prompt.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from local_operator.i18n.keys import wire_operator_setup as _ws
from local_operator.i18n.messages import Msg

#: Who started an attempt; recorded on every audit row (§6 item 6).
Actor = Literal["desktop", "tui"]
ACTORS: tuple[str, ...] = ("desktop", "tui")

#: Machine outcome codes. Clients branch on these, never on the sentence.
OUTCOME_ON = "on"
OUTCOME_ALREADY_ON = "already_on"
OUTCOME_PRIVILEGED = "privileged_process"
OUTCOME_NO_PASSWORD_PROMPT = "no_password_prompt"
OUTCOME_CANNOT_INSTALL = "cannot_install"
OUTCOME_KEY_FAILED = "key_failed"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_FAILED = "failed"
OUTCOME_NOT_VERIFIED = "not_verified"
OUTCOME_IN_PROGRESS = "in_progress"
#: The TUI's own precheck: this terminal app cannot hand the terminal to a password
#: prompt (a driver that cannot suspend). Distinct from ``no_password_prompt``
#: because the cause is the terminal app, not the computer (QA Q3/Q4).
OUTCOME_TERMINAL_CANNOT_PROMPT = "terminal_cannot_prompt"
#: The staged statement was not exactly the bytes built here (a FIFO, a link, a
#: foreign file, or changed before the prompt): refused with no prompt (R1-1).
OUTCOME_UNSAFE_STAGING = "unsafe_staging"
#: The install refused because the anchor changed while the prompt was open —
#: another setup landed one — so nothing was replaced (R1-4).
OUTCOME_CHANGED_ELSEWHERE = "changed_elsewhere"

#: Outcomes that ended with nothing staged or prompted: the host itself refused.
REFUSALS = frozenset(
    {
        OUTCOME_PRIVILEGED,
        OUTCOME_NO_PASSWORD_PROMPT,
        OUTCOME_CANNOT_INSTALL,
        OUTCOME_IN_PROGRESS,
        OUTCOME_TERMINAL_CANNOT_PROMPT,
    }
)

_MESSAGES: dict[str, Callable[[], Msg]] = {
    OUTCOME_ON: _ws.outcome_on,
    OUTCOME_ALREADY_ON: _ws.outcome_already_on,
    OUTCOME_PRIVILEGED: _ws.refused_privileged,
    OUTCOME_NO_PASSWORD_PROMPT: _ws.refused_no_password_prompt,
    OUTCOME_CANNOT_INSTALL: _ws.refused_cannot_install,
    OUTCOME_KEY_FAILED: _ws.refused_key_failed,
    OUTCOME_CANCELLED: _ws.outcome_cancelled,
    OUTCOME_FAILED: _ws.outcome_failed,
    OUTCOME_NOT_VERIFIED: _ws.outcome_not_verified,
    OUTCOME_IN_PROGRESS: _ws.refused_in_progress,
    OUTCOME_TERMINAL_CANNOT_PROMPT: _ws.refused_terminal_cannot_prompt,
    OUTCOME_UNSAFE_STAGING: _ws.refused_unsafe_staging,
    OUTCOME_CHANGED_ELSEWHERE: _ws.outcome_changed_elsewhere,
}

#: Held as a key REFERENCE, like ``_MESSAGES``, because ``wire.operator_setup`` is a
#: deferred namespace (not in the eager bundle): every caller renders the Msg
#: in-process — the TUI through ``text()``, the desktop route as an envelope whose
#: ``code`` is the machine token ``file_only`` and whose ``text`` is server-rendered
#: — so no client ever resolves this key. The census in
#: ``scripts/i18n/deferred_watch.py`` checks that shape (a bare key call handed on
#: is its failure signal); these callers are recorded in
#: ``local_operator/i18n/deferred_wire_inventory.json``.
_FILE_ONLY_MESSAGE: Callable[[], Msg] = _ws.outcome_file_only

_STATE_FILE = "authority-setup.json"
_AUDIT_FILE = "authority-setup-audit.jsonl"
_LOCK_FILE = "authority-setup.lock"
_RUNNING_FILE = "authority-setup.running"

#: In-process half of the one-attempt-at-a-time rule; the file lock below is the
#: cross-process half (a TUI and a desktop server are separate processes).
_THREAD_LOCK = threading.Lock()


class SetupInProgress(Exception):
    """Another attempt holds the setup lock; a second OS prompt would be confusing."""


def outcome_message(outcome: str) -> Msg:
    """The keyed sentence for ``outcome`` (one of the ``OUTCOME_*`` codes)."""
    return _MESSAGES[outcome]()


def file_only_message() -> Msg:
    """U18's plain statement for a host whose approvals are not presence-gated."""
    return _FILE_ONLY_MESSAGE()


def _operator_dir(root: Path) -> Path:
    path = root / "operator"
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


@contextlib.contextmanager
def _exclusive(root: Path) -> Iterator[None]:
    """Hold the one-attempt lock, or raise :class:`SetupInProgress` at once.

    Non-blocking on purpose: a second click while the OS prompt is open must be
    told "already running", not queue a second prompt behind the first.
    """
    if not _THREAD_LOCK.acquire(blocking=False):
        raise SetupInProgress
    handle = None
    # Only the attempt that WROTE the marker removes it (R1-3): a refused second
    # attempt from another process must not delete the running attempt's marker,
    # or `in_progress` would read "not running" while that prompt is still open.
    wrote_marker = False
    try:
        try:
            import fcntl
        except ImportError:  # pragma: no cover — Windows: the thread lock is the guard
            fcntl = None  # type: ignore[assignment]
        if fcntl is not None:
            # 0600 like every other file this module writes under operator/.
            handle = os.fdopen(
                os.open(_operator_dir(root) / _LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600), "r+"
            )
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                handle = None
                raise SetupInProgress from None
        # The marker is what `in_progress` reads. A status read must not take the
        # lock itself: a poll landing at the instant of a click would make that
        # click fail "already running".
        _write_private(_operator_dir(root) / _RUNNING_FILE, str(os.getpid()), append=False)
        wrote_marker = True
        yield
    finally:
        if wrote_marker:
            with contextlib.suppress(OSError):
                (root / "operator" / _RUNNING_FILE).unlink()
        if handle is not None:
            handle.close()  # closing releases the flock
        _THREAD_LOCK.release()


def in_progress(root: Path) -> bool:
    """Whether an attempt is running (in this process or another) right now.

    Read-only: the in-process lock's state, else the running marker's pid if that
    process is still alive (a crashed attempt's stale marker reads as not running).
    """
    if _THREAD_LOCK.locked():
        return True
    try:
        pid = int((root / "operator" / _RUNNING_FILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # alive, owned by someone else
    return True


def _write_private(path: Path, text: str, *, append: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "a" if append else "w", encoding="utf-8") as stream:
        stream.write(text)


def record_in_progress_refusal(root: Path, actor: str) -> None:
    """Append the audit row for an attempt refused because another one is running.

    §6 item 6 asks for an audit row per attempt (QA Q1). A refused second click is
    recorded as its own one-row attempt, ``proposed`` / ``in_progress``. It touches
    ONLY the audit log: the state file is the RUNNING attempt's, and overwriting it
    would make a refresh describe the refusal instead of the open prompt.
    """
    row = {
        "attempt": uuid.uuid4().hex[:12],
        "actor": actor,
        "step": "proposed",
        "ok": False,
        "code": OUTCOME_IN_PROGRESS,
        "at": time.time(),
    }
    with contextlib.suppress(OSError):
        _write_private(
            _operator_dir(root) / _AUDIT_FILE, json.dumps(row, sort_keys=True) + "\n", append=True
        )


def _windows_token_elevated() -> bool:
    """Whether this Windows process runs with an elevated (administrator) token.

    ``IsUserAnAdmin`` answers for the CURRENT token, which is what matters: an
    elevated token makes ``Start-Process -Verb RunAs`` succeed with no UAC consent
    (R1-2). FAIL CLOSED: if the check cannot run, the answer is "elevated", so the
    attempt is refused rather than risk a prompt-free install.
    """
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return True


def _privileged() -> bool:
    """Whether this process could write the anchor WITHOUT a person's prompt.

    ``handlers._is_privileged`` answers for POSIX (euid 0) and deliberately says
    ``False`` on Windows, because the CLI's print-the-command path depends on that.
    This path needs the stricter question on every platform (§6 item 1), so it adds
    the Windows token check here instead of changing the CLI's predicate.
    """
    from local_operator.operator import trust as trust_module
    from local_operator.operator.handlers import _is_privileged

    if _is_privileged():
        return True
    return trust_module._on_windows() and _windows_token_elevated()


#: ``O_NOFOLLOW`` refuses a symlink as the final component; ``O_NONBLOCK`` keeps a
#: FIFO from blocking the open (the R1-1 attack drains a write into a FIFO). Both
#: are POSIX-only; Windows has neither and relies on the lstat/fstat checks.
_SAFE_OPEN = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)


class _UnsafeStaging(Exception):
    """The staging path is not a regular file we own holding exactly our bytes."""


def _check_ours(fd: int) -> None:
    import stat

    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise _UnsafeStaging("not a regular file")
    geteuid = getattr(os, "geteuid", None)
    if geteuid is not None and info.st_uid != geteuid():
        raise _UnsafeStaging("not owned by this user")


def _stage_exact(path: Path, expected: bytes) -> None:
    """Write ``expected`` to ``path``, refusing anything but a fresh regular file.

    Replaces ``handlers.stage_anchor`` on THIS path only (its ``write_bytes``
    follows a link and blocks on a FIFO); the CLI verbs keep their own staging.
    A pre-existing non-regular entry (FIFO, link, directory) is refused, not
    removed: it is somebody's deliberate act, and the person is told instead.
    """
    import stat

    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            raise _UnsafeStaging("staging path is not a regular file")
    except FileNotFoundError:
        pass
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _SAFE_OPEN, 0o600)
    except OSError as error:  # ELOOP (a link), ENXIO (a FIFO with no reader)
        raise _UnsafeStaging(str(error)) from None
    try:
        _check_ours(fd)
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        view = memoryview(expected)
        while view:
            view = view[os.write(fd, view) :]
    finally:
        os.close(fd)


def _staged_is_exact(path: Path, expected: bytes) -> bool:
    """Whether ``path`` is still a regular file we own whose bytes are ``expected``.

    Asked immediately before the prompt: a swap after this point is caught by the
    privileged install's own hash check, which pins ``sha256(expected)``.
    """
    import stat

    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return False
        fd = os.open(path, os.O_RDONLY | _SAFE_OPEN)
    except OSError:
        return False
    try:
        _check_ours(fd)
        chunks: list[bytes] = []
        remaining = len(expected) + 1  # one byte past: an appended tail must fail
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    except (OSError, _UnsafeStaging):
        return False
    finally:
        os.close(fd)
    return b"".join(chunks) == expected


def _target_pin(target: Path) -> str | None:
    """The ``expected_old_hex`` for ``target`` as it is NOW, or ``None`` if unpinnable.

    ``EXPECT_ABSENT`` for no file, else the sha256 of a regular file's bytes. A
    link, directory or unreadable file at the anchor path cannot be pinned, and
    the caller refuses rather than install over something it did not read.
    """
    import stat

    from local_operator.operator import trust as trust_module

    try:
        if not stat.S_ISREG(os.lstat(target).st_mode):
            return None
    except FileNotFoundError:
        return trust_module.EXPECT_ABSENT
    except OSError:
        return None
    try:
        fd = os.open(target, os.O_RDONLY | _SAFE_OPEN)
    except OSError:
        return None
    digest = hashlib.sha256()
    try:
        while chunk := os.read(fd, 65536):
            digest.update(chunk)
    except OSError:
        return None
    finally:
        os.close(fd)
    return digest.hexdigest()


def read_last_attempt(root: Path) -> dict[str, Any] | None:
    """The last attempt's record, or ``None`` when there is none or it is unreadable."""
    try:
        data = json.loads((root / "operator" / _STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def setup_authority(
    root: Path,
    *,
    actor: Actor,
    admin_route: Literal["gesture"] = "gesture",
    terminal_sudo: bool = False,
    runner: Callable[[list[list[str]]], Any] | None = None,
    can_prompt: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Install operator authority behind ONE OS admin prompt; return the receipts.

    ``admin_route`` exists so the signature says what it does: ``"gesture"`` is the
    only route, and anything else raises. There is deliberately no ``sudo_secret``.

    ``runner`` replaces :func:`elevate.run_elevated` for tests and the §8 E2E stub
    gesture; it receives the same argv list and must return an object with
    ``exit_code``/``cancelled``/``refusal``. The host prechecks (privileged process,
    no password prompt) run BEFORE it, so a stub cannot bypass them.

    ``can_prompt`` is the CALLER'S own "can a prompt reach a person from here"
    question, asked with the host prechecks — before any consent row, key or
    staging (QA Q3). The TUI passes "this driver can hand over the terminal"; it can
    only add a refusal, never skip one.

    Raises :class:`SetupInProgress` when another attempt is running (after auditing
    the refusal, QA Q1). Never raises for a refused or failed attempt: those are
    outcomes, with receipts.
    """
    if admin_route != "gesture":
        raise ValueError("operator authority setup from the desktop/TUI is gesture-only")
    if actor not in ACTORS:
        raise ValueError(f"unknown actor: {actor!r}")
    try:
        # `_run` never raises SetupInProgress itself; only the lock does.
        with _exclusive(root):
            return _run(
                root,
                actor=actor,
                terminal_sudo=terminal_sudo,
                runner=runner,
                can_prompt=can_prompt,
            )
    except SetupInProgress:
        record_in_progress_refusal(root, actor)
        raise


def _run(
    root: Path,
    *,
    actor: str,
    terminal_sudo: bool,
    runner: Callable[[list[list[str]]], Any] | None,
    can_prompt: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    # Late imports: handlers pulls the whole CLI verb module, which a server route
    # should only pay for when someone actually turns approvals on.
    from local_operator.operator import (
        LEVEL_OPERATOR_FILE_ONLY,
        create_key,
        install_commands,
        load_anchor,
        operator_authority_report,
    )
    from local_operator.operator import trust as trust_module
    from local_operator.operator.elevate import (
        PURPOSE_PHONE_APPROVALS,
        REFUSAL_NO_PASSWORD_PROMPT,
        gui_prompt_refusal,
        prompt_refusal,
        run_elevated,
    )
    from local_operator.operator.handlers import _existing_key
    from local_operator.operator.keychain import KeyBackendError
    from local_operator.operator.sign import anchor_for_handle

    attempt = uuid.uuid4().hex[:12]
    started = time.time()
    receipts: list[dict[str, Any]] = []
    audit = _operator_dir(root) / _AUDIT_FILE

    def receipt(step: str, ok: bool, code: str, **data: Any) -> None:
        row: dict[str, Any] = {"step": step, "ok": ok, "code": code, "at": time.time()}
        row.update(data)
        receipts.append(row)
        # Appended as it happens, not at the end: an attempt whose process dies
        # while the OS prompt is open still leaves proposed/consent/generated.
        line = {"attempt": attempt, "actor": actor, **row}
        try:
            _write_private(audit, json.dumps(line, sort_keys=True) + "\n", append=True)
        except OSError:
            pass  # The audit is evidence, never a reason to fail the person's action.

    def finish(outcome: str) -> dict[str, Any]:
        report = operator_authority_report()
        presence_gated = bool(report.get("presence_enforced_by_os"))
        result: dict[str, Any] = {
            "ok": outcome in (OUTCOME_ON, OUTCOME_ALREADY_ON),
            "outcome": outcome,
            "attempt": attempt,
            "actor": actor,
            "started_at": started,
            "finished_at": time.time(),
            "level": report["level"],
            # §6 item 5: a file-only host is TOLD its approvals are not
            # presence-gated, never left to infer a boundary it does not have.
            "presence_gated": presence_gated,
            "file_only": report["level"] == LEVEL_OPERATOR_FILE_ONLY,
            "receipts": receipts,
        }
        try:
            _write_private(
                _operator_dir(root) / _STATE_FILE,
                json.dumps(result, sort_keys=True, default=str),
                append=False,
            )
        except OSError:
            pass
        return result

    receipt("proposed", True, "phone_approvals")

    loaded = load_anchor()
    if loaded.usable:
        # IDEMPOTENT ON AN INSTALLED ANCHOR (§6 item 4). Checked before the privilege
        # refusal so a status re-run from anywhere answers "already on" truthfully.
        receipt("consent", True, OUTCOME_ALREADY_ON)
        receipt("installed", True, OUTCOME_ALREADY_ON)
        receipt("verified", True, OUTCOME_ALREADY_ON)
        return finish(OUTCOME_ALREADY_ON)

    if _privileged():
        # REFUSED, NOT SHORTCUT (§6 item 1): a root process (or an elevated Windows
        # token) would write the anchor with no prompt, i.e. with no human.
        receipt("consent", False, OUTCOME_PRIVILEGED)
        return finish(OUTCOME_PRIVILEGED)

    if trust_module.install_refusal() is not None:
        receipt("consent", False, OUTCOME_CANNOT_INSTALL)
        return finish(OUTCOME_CANNOT_INSTALL)

    # §6 item 2: ask "can a person be prompted HERE" before promising anything or
    # creating a key. The server form excludes terminal sudo (no TTY).
    cannot = prompt_refusal() if terminal_sudo else gui_prompt_refusal()
    if cannot is not None:
        receipt("consent", False, OUTCOME_NO_PASSWORD_PROMPT)
        return finish(OUTCOME_NO_PASSWORD_PROMPT)
    if can_prompt is not None and not can_prompt():
        # The caller's own half of the same question (the TUI: can this driver
        # hand the terminal to sudo?), asked HERE so a refusal leaves no consent
        # row, key or staging behind (QA Q3).
        receipt("consent", False, OUTCOME_TERMINAL_CANNOT_PROMPT)
        return finish(OUTCOME_TERMINAL_CANNOT_PROMPT)
    receipt("consent", True, "gesture")

    try:
        existing = _existing_key(root, "auto")
        handle = existing if existing is not None else create_key(config_root=root)
    except (KeyBackendError, FileExistsError, OSError):
        receipt("generated", False, OUTCOME_KEY_FAILED)
        return finish(OUTCOME_KEY_FAILED)
    # The `init` shape (handlers._init): ADMISSION_LISTED, so a phone counts only
    # once the anchor lists it — each later phone still gets its own admit prompt.
    # Serialised IN MEMORY and pinned from memory (R1-1): never from a read-back of
    # the user-writable staging path, which a FIFO or a swap could answer.
    expected = trust_module.anchor_bytes(
        replace(anchor_for_handle(handle, label=""), admission=trust_module.ADMISSION_LISTED)
    )
    digest = hashlib.sha256(expected).hexdigest()
    staged = trust_module.staging_path(root)
    try:
        _stage_exact(staged, expected)
    except (_UnsafeStaging, OSError):
        receipt("generated", False, OUTCOME_UNSAFE_STAGING, key_id=handle.key_id)
        return finish(OUTCOME_UNSAFE_STAGING)
    receipt(
        "generated", True, "staged", key_id=handle.key_id, backend=handle.backend, sha256=digest
    )

    # Pin the target as it is NOW (R1-4): absent, or the unusable file's digest.
    target = trust_module.anchor_path()
    old = _target_pin(target)
    if old is None:
        receipt("installed", False, OUTCOME_CANNOT_INSTALL)
        return finish(OUTCOME_CANNOT_INSTALL)
    # The last look before a person is asked: still a regular file we own, holding
    # exactly the bytes built above. A swap after this is the install's hash check.
    if not _staged_is_exact(staged, expected):
        receipt("installed", False, OUTCOME_UNSAFE_STAGING)
        return finish(OUTCOME_UNSAFE_STAGING)
    command = install_commands(staged, target, digest, old)
    if runner is not None:
        result = runner(command)
    else:
        result = run_elevated(command, purpose=PURPOSE_PHONE_APPROVALS, terminal_sudo=terminal_sudo)
    if getattr(result, "cancelled", False) or getattr(result, "exit_code", 1) != 0:
        # Nothing staged survives a failed prompt: a statement left behind is what
        # an unrelated later `lop operator install` would quietly apply.
        with contextlib.suppress(OSError):
            staged.unlink()
        if getattr(result, "refusal_code", "") == REFUSAL_NO_PASSWORD_PROMPT or getattr(
            result, "refusal", ""
        ):
            outcome = OUTCOME_NO_PASSWORD_PROMPT
        elif getattr(result, "cancelled", False):
            outcome = OUTCOME_CANCELLED
        elif _target_pin(target) != old:
            # The install's compare-and-swap refused: an anchor landed (or changed)
            # while the prompt was open, and it was left alone (R1-4).
            outcome = OUTCOME_CHANGED_ELSEWHERE
        else:
            outcome = OUTCOME_FAILED
        receipt("installed", False, outcome)
        return finish(outcome)
    receipt("installed", True, "installed", path=str(target))

    # VERIFY WHAT WAS INSTALLED, NOT WHAT WE MEANT TO INSTALL (R1-1): the root-owned
    # anchor is read back through the runtime's own loader and must name the key
    # created above with ADMISSION_LISTED, byte-for-byte the statement built here.
    # The receipt records the values READ, so a tampered anchor cannot read "on".
    loaded = load_anchor()
    installed = loaded.anchor
    try:
        landed = target.read_bytes() == expected
    except OSError:
        landed = False
    if (
        installed is None
        or not loaded.usable
        or not landed
        or installed.key_id != handle.key_id
        or installed.spki != handle.spki
        or installed.admission != trust_module.ADMISSION_LISTED
    ):
        receipt("verified", False, OUTCOME_NOT_VERIFIED)
        return finish(OUTCOME_NOT_VERIFIED)
    receipt(
        "verified", True, "verified", key_id=installed.key_id, admission=installed.admission
    )
    return finish(OUTCOME_ON)


def describe_state(root: Path, *, terminal_sudo: bool = False) -> dict[str, Any]:
    """What a client needs to draw the setup block, without side effects.

    ``state`` is DERIVED from the host, not from the last attempt: ``on`` when the
    anchor is usable; ``finish_setup`` when an attempt was made and did not land
    (the "Finish setup" affordance after a cancelled prompt); ``off`` otherwise.
    ``can_prompt`` answers before the click whether an OS prompt can appear here,
    so a client can show U18's sentence instead of a button that cannot work.
    """
    from local_operator.operator import (
        LEVEL_OPERATOR_FILE_ONLY,
        operator_authority_report,
        operator_authority_unusable,
    )
    from local_operator.operator.elevate import gui_prompt_refusal, prompt_refusal

    report = operator_authority_report()
    last = read_last_attempt(root)
    usable = not operator_authority_unusable()
    if usable:
        state = "on"
    elif last is not None and not last.get("ok"):
        state = "finish_setup"
    else:
        state = "off"
    blocked: str | None = None
    if not usable:
        if _privileged():
            blocked = OUTCOME_PRIVILEGED
        elif (prompt_refusal() if terminal_sudo else gui_prompt_refusal()) is not None:
            blocked = OUTCOME_NO_PASSWORD_PROMPT
    return {
        "state": state,
        "level": report["level"],
        "presence_gated": bool(report.get("presence_enforced_by_os")),
        "file_only": report["level"] == LEVEL_OPERATOR_FILE_ONLY,
        "device_admission": bool(report.get("device_admission")),
        "in_progress": in_progress(root),
        "can_prompt": blocked is None,
        "blocked": blocked,
        "last_attempt": last,
    }


__all__ = [
    "ACTORS",
    "OUTCOME_ALREADY_ON",
    "OUTCOME_CANCELLED",
    "OUTCOME_CANNOT_INSTALL",
    "OUTCOME_CHANGED_ELSEWHERE",
    "OUTCOME_FAILED",
    "OUTCOME_IN_PROGRESS",
    "OUTCOME_KEY_FAILED",
    "OUTCOME_NOT_VERIFIED",
    "OUTCOME_NO_PASSWORD_PROMPT",
    "OUTCOME_ON",
    "OUTCOME_PRIVILEGED",
    "OUTCOME_TERMINAL_CANNOT_PROMPT",
    "OUTCOME_UNSAFE_STAGING",
    "REFUSALS",
    "SetupInProgress",
    "describe_state",
    "file_only_message",
    "in_progress",
    "outcome_message",
    "read_last_attempt",
    "record_in_progress_refusal",
    "setup_authority",
]

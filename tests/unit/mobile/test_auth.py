"""Cookie signing is the whole session layer: rotation must invalidate, a
wrong signature must never verify, and expiry must actually expire."""

from __future__ import annotations

import time

import pytest

from local_operator.mobile.auth import (
    check_password,
    generate_password,
    sign_cookie,
    verify_cookie,
)


def test_cookie_round_trip() -> None:
    password = generate_password()
    cookie = sign_cookie(password)
    assert verify_cookie(cookie, password)


def test_rotation_invalidates_every_cookie() -> None:
    old, new = generate_password(), generate_password()
    cookie = sign_cookie(old)
    assert not verify_cookie(cookie, new)


def test_tampered_signature_never_verifies() -> None:
    password = generate_password()
    expiry, sig = sign_cookie(password).rsplit(".", 1)
    forged = f"{expiry}.{'0' * len(sig)}"
    assert not verify_cookie(forged, password)


def test_tampered_expiry_never_verifies() -> None:
    password = generate_password()
    expiry, sig = sign_cookie(password).rsplit(".", 1)
    forged = f"{int(expiry) + 10**9}.{sig}"
    assert not verify_cookie(forged, password)


def test_expired_cookie_fails() -> None:
    password = generate_password()
    now = time.time()
    cookie = sign_cookie(password, now=now - 40 * 24 * 3600)  # issued 40 days ago
    assert not verify_cookie(cookie, password, now=now)


def test_garbage_values_fail_closed() -> None:
    password = generate_password()
    assert not verify_cookie(None, password)
    assert not verify_cookie("", password)
    assert not verify_cookie("no-dot-here", password)
    assert not verify_cookie("abc.def", password)


def test_check_password_is_exact() -> None:
    assert check_password("hunter2", "hunter2")
    assert not check_password("hunter2 ", "hunter2")
    assert not check_password("", "hunter2")


def test_generated_passwords_are_urlsafe_and_unique() -> None:
    a, b = generate_password(), generate_password()
    assert a != b
    assert all(c.isalnum() or c in "-_" for c in a)


# ---------------------------------------------------------------------------
# The password store (A10/C5)
#
# The defect: every one of the three store functions went through the macOS
# `security` binary. On Linux and Windows `load_password()` answered `None`
# (so `lop mobile serve` exited 2 telling the user to run `lop mobile install`,
# which cannot work there) and `store_password`/`delete_password` raised an
# uncaught `FileNotFoundError` at the CLI.
#
# No test here touches a real keyring, a real DPAPI store, or the operator's
# own Keychain: every subprocess these functions make is faked, and the file
# fallback is redirected to a tmpdir.
# ---------------------------------------------------------------------------


class _FakeRun:
    """Records every subprocess the store makes, and replays a canned result."""

    def __init__(self, code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.code = code
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[tuple[list[str], str | None]] = []

    def __call__(self, argv, **kwargs):  # noqa: ANN001, ANN003, ANN204
        import subprocess

        self.calls.append((list(argv), kwargs.get("input")))
        return subprocess.CompletedProcess(list(argv), self.code, self.stdout, self.stderr)


def _linux(  # noqa: ANN001, ANN202
    monkeypatch, tmp_path, *, secret_tool: bool, code: int = 0, stdout: str = ""
):
    # THE PLATFORM IS PATCHED THROUGH ``auth``'s OWN CONSTANTS, NEVER THROUGH
    # ``auth.sys``/``auth.os``. Those two ARE the ``sys`` and ``os`` modules, so
    # patching them rewrote the process's platform for the duration of every
    # test here — ten of them — and ``pathlib`` reads ``os.name`` at call time:
    # under ``"nt"``, every ``Path(...)`` in the process raises
    # ``UnsupportedOperation``, and every module imported inside the window dies
    # or is left half-initialised and ``NameError``s later in an unrelated test.
    # That is what made this file order-dependent in a batch. See
    # ``local_operator.mobile.auth._PLATFORM`` for the measured detail.
    from local_operator.mobile import auth

    runner = _FakeRun(code, stdout)
    monkeypatch.setattr(auth, "_PLATFORM", "linux")
    monkeypatch.setattr(auth, "_IS_WINDOWS", False)
    monkeypatch.setattr(auth, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(
        auth.shutil, "which", lambda name: f"/usr/bin/{name}" if secret_tool else None
    )
    monkeypatch.setattr(auth.subprocess, "run", runner)
    monkeypatch.delenv("LOP_MOBILE_PASSWORD", raising=False)
    return auth, runner


def test_the_env_override_wins_on_every_platform(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """The documented container/dev path, now first class rather than a workaround."""
    auth, runner = _linux(monkeypatch, tmp_path, secret_tool=True, stdout="from-keyring")
    monkeypatch.setenv("LOP_MOBILE_PASSWORD", "from-env")

    assert auth.load_password() == "from-env"
    assert runner.calls == []


def test_linux_prefers_the_keyring_and_never_puts_the_password_in_argv(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """`ps` reads argv; a portal password must not be there."""
    auth, runner = _linux(monkeypatch, tmp_path, secret_tool=True)

    auth.store_password("s3cret")

    argv, stdin = runner.calls[-1]
    assert argv[0].endswith("secret-tool")
    assert argv[1] == "store"
    assert "s3cret" not in argv, "the password went into argv"
    assert stdin == "s3cret"


def test_linux_falls_back_to_a_private_file_when_libsecret_is_absent(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """THE PRE-FIX CRASH: this used to raise FileNotFoundError('security')."""
    auth, _runner = _linux(monkeypatch, tmp_path, secret_tool=False)

    auth.store_password("s3cret")

    path = tmp_path / "mobile" / "password"
    assert path.exists()
    assert path.read_text().strip() == "s3cret"
    assert (path.stat().st_mode & 0o777) == 0o600, "the fallback must be private"
    assert auth.load_password() == "s3cret"
    assert auth.store_description().startswith("a 0600 file")


def test_a_keyring_that_takes_over_removes_a_stale_plaintext_copy(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """Leaving the file beside the keyring keeps a copy strictly weaker than it."""
    auth, _runner = _linux(monkeypatch, tmp_path, secret_tool=False)
    auth.store_password("s3cret")
    assert (tmp_path / "mobile" / "password").exists()

    auth2, _runner2 = _linux(monkeypatch, tmp_path, secret_tool=True)
    auth2.store_password("rotated")

    assert not (tmp_path / "mobile" / "password").exists()


def test_a_refusing_keyring_is_a_named_refusal_rather_than_a_traceback(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """A headless box has no Secret Service; the answer names the env override."""
    auth, _runner = _linux(monkeypatch, tmp_path, secret_tool=True, code=1)

    with pytest.raises(auth.PasswordStoreUnavailable) as caught:
        auth.store_password("s3cret")

    assert "LOP_MOBILE_PASSWORD" in str(caught.value)


def test_an_unreadable_store_reads_as_no_password_rather_than_raising(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """`load_password` is on the status path: absence is None, never an exception."""
    auth, _runner = _linux(monkeypatch, tmp_path, secret_tool=True, code=2, stdout="")

    assert auth.load_password() is None


def test_windows_round_trips_through_dpapi(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """Windows gets the platform's own store, not a plaintext file.

    The ctypes binding itself cannot be exercised off Windows; what is asserted
    is the contract around it — where the blob lives, that it round-trips, and
    that a failure to reach crypt32 is a NAMED refusal naming the override. The
    double is what makes that decidable here.
    """
    from local_operator.mobile import auth

    monkeypatch.setattr(auth, "_PLATFORM", "win32")
    monkeypatch.setattr(auth, "_IS_WINDOWS", True)
    monkeypatch.setattr(auth, "config_dir", lambda: tmp_path)
    monkeypatch.delenv("LOP_MOBILE_PASSWORD", raising=False)

    def fake_dpapi(protect: bool, data: bytes) -> bytes:
        # Invertible, which is what DPAPI is; the real binding is a ctypes call
        # to crypt32 and cannot run here at all.
        return b"sealed:" + data if protect else data.removeprefix(b"sealed:")

    monkeypatch.setattr(auth, "_dpapi", fake_dpapi)

    auth.store_password("s3cret")

    blob = tmp_path / "mobile" / "password.dpapi"
    assert blob.read_bytes() == b"sealed:s3cret"
    assert auth.load_password() == "s3cret"
    assert auth.store_description().startswith("a DPAPI-protected file")


def test_windows_without_dpapi_refuses_legibly(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    from local_operator.mobile import auth

    monkeypatch.setattr(auth, "_PLATFORM", "win32")
    monkeypatch.setattr(auth, "_IS_WINDOWS", True)
    monkeypatch.setattr(auth, "config_dir", lambda: tmp_path)

    def boom(_protect: bool, _data: bytes) -> bytes:
        raise OSError(5, "Access is denied")

    monkeypatch.setattr(auth, "_dpapi", boom)

    with pytest.raises(auth.PasswordStoreUnavailable) as caught:
        auth.store_password("s3cret")

    assert "LOP_MOBILE_PASSWORD" in str(caught.value)
    assert "Access is denied" in str(caught.value)


def test_newlines_are_still_refused(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """A newline splits the macOS mini-shell command into two."""
    auth, _runner = _linux(monkeypatch, tmp_path, secret_tool=False)

    with pytest.raises(ValueError, match="newlines"):
        auth.store_password("a\nb")


def test_delete_removes_whichever_store_is_in_use(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    auth, _runner = _linux(monkeypatch, tmp_path, secret_tool=False)
    auth.store_password("s3cret")

    auth.delete_password()

    assert not (tmp_path / "mobile" / "password").exists()
    assert auth.load_password() is None

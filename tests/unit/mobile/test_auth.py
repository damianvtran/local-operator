"""Cookie signing is the whole session layer: rotation must invalidate, a
wrong signature must never verify, and expiry must actually expire."""

from __future__ import annotations

import time

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

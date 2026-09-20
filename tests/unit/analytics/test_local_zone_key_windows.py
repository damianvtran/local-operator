"""The day-bucket zone key on Windows (audit D19).

``/etc/localtime`` does not exist there, and ``os.path.realpath`` does not raise
for a missing path — it returns it unchanged — so the POSIX probe silently found
nothing and the DST-sensitive ``tzname()`` fallback became the permanent answer:
the fast path was refused for half of every year, since the name the rollup
compares against changes twice a year.

The registry read is exercised as a real call, by injecting a stand-in
``winreg`` module. It is a registry read and nothing else, so the stand-in
drives the production call shape rather than a copy of it.
"""

from __future__ import annotations

import sys
import types

import pytest

from local_operator.analytics import store as store_module

REGISTRY_ZONE = "Eastern Standard Time"


class _FakeKey:
    def __enter__(self) -> "_FakeKey":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _fake_winreg(*, value: object = REGISTRY_ZONE, raises: OSError | None = None) -> object:
    module = types.ModuleType("winreg")
    module.HKEY_LOCAL_MACHINE = 0x80000002  # type: ignore[attr-defined]
    module.OpenKey = lambda *_args, **_kwargs: _FakeKey()  # type: ignore[attr-defined]

    def _query(*_args: object) -> tuple[object, int]:
        if raises is not None:
            raise raises
        return value, 1

    module.QueryValueEx = _query  # type: ignore[attr-defined]
    return module


def test_the_registry_name_is_preferred_when_it_can_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg())
    assert store_module._windows_zone_key() == REGISTRY_ZONE


def test_an_unreadable_registry_falls_through_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stripped-down image must cost the fast path, never the bucket."""
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg(raises=FileNotFoundError(2, "gone")))
    assert store_module._windows_zone_key() is None


def test_an_empty_registry_value_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg(value=""))
    assert store_module._windows_zone_key() is None


def test_the_zone_key_uses_the_registry_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wiring, with ``os.name`` set only for the call: ``_local_zone_key`` builds no path."""
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg())
    monkeypatch.delenv("TZ", raising=False)
    with monkeypatch.context() as windows:
        windows.setattr(store_module.os, "name", "nt")
        assert store_module._local_zone_key() == REGISTRY_ZONE


def test_the_zone_key_still_prefers_tz_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    """POSIX behaviour unchanged, including the ordering the docstring states."""
    monkeypatch.setenv("TZ", "Elsewhere/Nowhere")
    assert store_module._local_zone_key() == "Elsewhere/Nowhere"

    # With TZ gone the POSIX probe answers: an IANA name where /etc/localtime
    # points into a zoneinfo tree, the abbreviation otherwise. Either is a
    # non-empty string, and neither is where this change is — asserted so the
    # Windows branch cannot have displaced the POSIX one.
    monkeypatch.delenv("TZ", raising=False)
    assert isinstance(store_module._local_zone_key(), str)

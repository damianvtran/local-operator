"""The CI ordering hook must be INERT except for one narrow, stated effect."""

from __future__ import annotations

from types import SimpleNamespace

from tests import ci_order_owned_first as hook


class Item:
    def __init__(self, name):
        self.path = SimpleNamespace(name=name)

    def __repr__(self):
        return self.path.name


def _items(names):
    return [Item(n) for n in names]


def test_it_moves_the_owned_files_to_the_front(monkeypatch):
    monkeypatch.setattr(hook.sys, "platform", "linux")
    items = _items(["test_aaa.py", "test_run_bounded_linux_spawn_load.py", "test_bbb.py"])
    hook.pytest_collection_modifyitems(None, items)
    assert [i.path.name for i in items] == [
        "test_run_bounded_linux_spawn_load.py",
        "test_aaa.py",
        "test_bbb.py",
    ]


def test_it_is_inert_off_linux(monkeypatch):
    monkeypatch.setattr(hook.sys, "platform", "darwin")
    items = _items(["test_aaa.py", "test_run_bounded_linux_spawn_load.py"])
    hook.pytest_collection_modifyitems(None, items)
    assert [i.path.name for i in items] == ["test_aaa.py", "test_run_bounded_linux_spawn_load.py"]


def test_it_never_selects_skips_or_reorders_anything_else(monkeypatch):
    monkeypatch.setattr(hook.sys, "platform", "linux")
    names = ["test_zzz.py", "test_yyy.py", "test_xxx.py"]
    items = _items(names)
    hook.pytest_collection_modifyitems(None, items)
    # No owned item: the list is returned EXACTLY as collected, order included.
    assert [i.path.name for i in items] == names

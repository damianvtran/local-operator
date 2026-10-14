"""Owner generation counters and recipient observation store (synthetic, tmp_path only).

Each test pins one rule from ``docs/design/mesh-credential-lifecycle.md``: the two
counters are independent and forward-only, a sticky ``withdrawn`` survives logout
and re-login, and unprovable persisted state is ``needs_authoritative_baseline``
rather than a silent restart at 1.
"""

from pathlib import Path

import pytest

from local_operator.network.credentials import lifecycle as L
from local_operator.network.types import MeshRefusal

NET = "net-1"
KEY = "mcp:linear"
SCOPE = "acct-a"


@pytest.fixture
def lc(tmp_path: Path) -> L.LifecycleStore:
    return L.LifecycleStore(NET, root=tmp_path)


def _login(lc: L.LifecycleStore, at: float = 1.0, **kw) -> L.CounterRow:
    return lc.record_native(KEY, SCOPE, L.NativeFacts(present=True, row_updated_at=at, **kw))


def test_first_observation_seeds_at_one(lc):
    row = _login(lc)
    assert (row.material_generation, row.login_chain_revision) == (1, 1)
    assert row.state == L.STATE_AVAILABLE


def test_rotation_advances_material_not_chain(lc):
    _login(lc, 1.0)
    row = _login(lc, 2.0)
    assert (row.material_generation, row.login_chain_revision) == (2, 1)
    # Same updated_at again is not a rotation.
    assert _login(lc, 2.0) == row


def test_new_login_advances_chain_by_one(lc):
    _login(lc, 1.0)
    row = _login(lc, 2.0, new_login=True)
    assert row.login_chain_revision == 2
    assert row.material_generation == 2  # one generation, not two, for one event


def test_logout_then_relogin(lc):
    _login(lc, 1.0)
    out = lc.record_native(KEY, SCOPE, L.NativeFacts(present=False))
    assert out.state == L.STATE_ABSENT
    assert (out.material_generation, out.login_chain_revision) == (2, 1)
    back = _login(lc, 3.0)
    assert (back.material_generation, back.login_chain_revision) == (3, 2)


def test_scope_change_and_reenable_advance_chain_only(lc):
    _login(lc, 1.0)
    row = _login(lc, 1.0, scope_changed=True)
    assert (row.material_generation, row.login_chain_revision) == (1, 2)


def test_sticky_withdrawn_survives_logout_relogin_until_reenable(lc):
    _login(lc, 1.0, withdrawn=True)
    assert lc.snapshot(KEY, SCOPE).state == L.STATE_WITHDRAWN
    out = lc.record_native(KEY, SCOPE, L.NativeFacts(present=False))
    assert out.withdrawn and out.state == L.STATE_WITHDRAWN
    back = _login(lc, 3.0)
    assert back.withdrawn and back.state == L.STATE_WITHDRAWN
    cleared = _login(lc, 3.0, authorised_reenable=True)
    assert not cleared.withdrawn
    assert cleared.login_chain_revision == back.login_chain_revision + 1


def test_counters_survive_reload_and_never_move_down(tmp_path):
    a = L.LifecycleStore(NET, root=tmp_path)
    _login(a, 1.0)
    _login(a, 2.0, new_login=True)
    b = L.LifecycleStore(NET, root=tmp_path)
    row = b.snapshot(KEY, SCOPE)
    assert row is not None
    assert (row.material_generation, row.login_chain_revision) == (2, 2)
    prev = row
    for facts in (
        L.NativeFacts(present=False),
        L.NativeFacts(present=True, row_updated_at=0.5),
        L.NativeFacts(present=True, row_updated_at=0.5),
    ):
        nxt = L.derive(prev, KEY, SCOPE, facts)
        assert nxt.material_generation >= prev.material_generation
        assert nxt.login_chain_revision >= prev.login_chain_revision
        prev = nxt


def test_observation_store_refuses_stale_and_replay_applies_newer(tmp_path):
    a = L.LifecycleStore(NET, root=tmp_path)

    def obs(chain, mat):
        return L.Observation("dev-a", KEY, SCOPE, chain, mat)

    assert a.observe(obs(2, 5)).applied
    assert a.observe(obs(2, 5)).reason == L.OBSERVED_REPLAY
    assert a.observe(obs(1, 9)).reason == L.OBSERVED_STALE  # chain went down
    assert a.observe(obs(3, 4)).reason == L.OBSERVED_STALE  # material went down
    # High-water survives reload and still refuses the stale one.
    b = L.LifecycleStore(NET, root=tmp_path)
    assert b.observe(obs(2, 4)).reason == L.OBSERVED_STALE
    res = b.observe(obs(2, 6))
    assert res.applied and (res.login_chain_revision, res.material_generation) == (2, 6)
    # A different owner is a different high-water.
    assert b.observe(L.Observation("dev-b", KEY, SCOPE, 1, 1)).applied
    # A bare huge claimed jump is not adopted.
    assert b.observe(obs(2, 6 + 2**20 + 1)).reason == L.OBSERVED_JUMP


def test_invalid_observation_refused_without_write(lc):
    res = lc.observe(L.Observation("dev-a", KEY, SCOPE, True, 1))  # bool is not an int
    assert res.reason == L.OBSERVED_INVALID
    assert not lc.path.exists()


def test_persisted_file_is_metadata_only_and_private(lc):
    _login(lc)
    lc.observe(L.Observation("dev-a", KEY, SCOPE, 1, 1))
    data = lc.load().to_json()
    assert set(data) == {"schema", "network_id", "counters", "observations"}
    assert set(data["observations"][0]) == {
        "owner",
        "key",
        "account_scope_ref",
        "login_chain_revision",
        "material_generation",
    }
    assert (lc.path.stat().st_mode & 0o777) == 0o600


@pytest.mark.parametrize(
    "content",
    [
        b"not json",
        b"{}",
        b'{"schema": 2, "network_id": "net-1", "counters": [], "observations": []}',
        b'{"schema": 1, "network_id": "other", "counters": [], "observations": []}',
        b'{"schema": 1, "network_id": "net-1", "observations": [], "counters": [{"key": "k",'
        b' "account_scope_ref": "s", "material_generation": 0, "login_chain_revision": 1,'
        b' "present": true, "withdrawn": false, "row_updated_at": null}]}',
    ],
)
def test_reset_or_unprovable_state_needs_authoritative_baseline(lc, content):
    _login(lc)
    original_dir = lc.path.parent
    lc.path.write_bytes(content)
    assert lc.status() == L.NEEDS_AUTHORITATIVE_BASELINE
    with pytest.raises(MeshRefusal) as exc:
        _login(lc, 9.0)
    assert exc.value.code == L.NEEDS_AUTHORITATIVE_BASELINE
    res = lc.observe(L.Observation("dev-a", KEY, SCOPE, 9, 9))
    assert (res.applied, res.reason) == (False, L.NEEDS_AUTHORITATIVE_BASELINE)
    assert lc.path.read_bytes() == content  # left for repair, never re-initialised
    assert lc.path.parent == original_dir


def test_counter_exhaustion_refuses_rather_than_wraps():
    prev = L.CounterRow(KEY, SCOPE, L.MAX_REVISION, 1, True, False, 1.0)
    with pytest.raises(MeshRefusal) as exc:
        L.derive(prev, KEY, SCOPE, L.NativeFacts(present=True, row_updated_at=2.0))
    assert exc.value.code == "lifecycle_counter_exhausted"


# ---------------------------------------------------------------------------
# Writer must never produce a file its own reader rejects (review M1)
# ---------------------------------------------------------------------------


def _fill_observations(lc: L.LifecycleStore, count: int, *, scope_pad: str = "") -> None:
    """Seed ``count`` distinct high-water rows in one write, as repeated observes would."""
    with lc.locked():
        doc = lc.load()
        for i in range(count):
            doc.high_water[("dev-a", KEY, f"{scope_pad}s{i}")] = (1, 1)
        lc._save(doc)


def test_distinct_scope_refs_cannot_make_the_file_unreadable(tmp_path):
    lc = L.LifecycleStore(NET, root=tmp_path)
    _fill_observations(lc, L.MAX_ROWS - 1)
    assert lc.observe(L.Observation("dev-a", KEY, "last-fit", 1, 1)).applied
    before = lc.path.read_bytes()
    # The table is now exactly full: a NEW ident is refused by name, file untouched.
    res = lc.observe(L.Observation("dev-a", KEY, "one-too-many", 1, 1))
    assert (res.applied, res.reason) == (False, L.OBSERVED_TABLE_FULL)
    assert lc.path.read_bytes() == before
    reopened = L.LifecycleStore(NET, root=tmp_path)
    assert reopened.status() == "ok"
    replay = reopened.observe(L.Observation("dev-a", KEY, "last-fit", 1, 1))
    assert replay.reason == L.OBSERVED_REPLAY


def test_long_scope_refs_cannot_exceed_the_file_bound(tmp_path, monkeypatch):
    # WHY a patched ceiling: reaching the real 1 MiB bound takes ~3.5k full
    # load-and-rewrite cycles (~35 s), which alone pushed the heaviest CI shard past
    # its wall ceiling. ``_save`` and ``load`` both read the module constant at call
    # time, so a small ceiling exercises the identical production byte-bound path.
    bound = 8_192
    monkeypatch.setattr(L, "MAX_FILE_BYTES", bound)
    lc = L.LifecycleStore(NET, root=tmp_path)
    pad = "x" * 240
    refused = None
    accepted = 0
    for i in range(L.MAX_ROWS):
        before = lc.path.read_bytes() if lc.path.exists() else None
        res = lc.observe(L.Observation("dev-a", KEY, f"{pad}{i:05d}", 1, 1))
        if not res.applied:
            refused = res
            break
        accepted += 1
    assert refused is not None and refused.reason == L.OBSERVED_TABLE_FULL
    # The bound, not the row cap, stopped it - and only a handful of rows were needed.
    assert 0 < accepted < L.MAX_ROWS and accepted < 100
    # The refusal left the file byte-identical, within the bound, and still loadable.
    assert lc.path.read_bytes() == before
    assert lc.path.stat().st_size <= bound
    assert L.LifecycleStore(NET, root=tmp_path).status() == "ok"
    # An existing ident still moves forward after the refusal.
    res = lc.observe(L.Observation("dev-a", KEY, f"{pad}{0:05d}", 2, 3))
    assert res.applied and (res.login_chain_revision, res.material_generation) == (2, 3)


def test_full_table_still_applies_updates_to_existing_idents(tmp_path):
    lc = L.LifecycleStore(NET, root=tmp_path)
    _fill_observations(lc, L.MAX_ROWS)
    # A withdraw / forward step on an ident that already has a row is never blocked.
    res = lc.observe(L.Observation("dev-a", KEY, "s7", 2, 3))
    assert res.applied and (res.login_chain_revision, res.material_generation) == (2, 3)
    assert L.LifecycleStore(NET, root=tmp_path).status() == "ok"
    # An unchanged-counter restrictive re-apply is a replay (the announce handler
    # still applies it), and writes nothing.
    before = lc.path.read_bytes()
    assert lc.observe(L.Observation("dev-a", KEY, "s7", 2, 3)).reason == L.OBSERVED_REPLAY
    assert lc.path.read_bytes() == before


def test_owner_counters_table_is_capped_too(tmp_path):
    lc = L.LifecycleStore(NET, root=tmp_path)
    with lc.locked():
        doc = lc.load()
        for i in range(L.MAX_ROWS):
            facts = L.NativeFacts(present=True, row_updated_at=1.0)
            doc.rows[(KEY, f"s{i}")] = L.derive(None, KEY, f"s{i}", facts)
        lc._save(doc)
    before = lc.path.read_bytes()
    with pytest.raises(MeshRefusal) as exc:
        lc.record_native(KEY, "brand-new", L.NativeFacts(present=True, row_updated_at=1.0))
    assert exc.value.code == "lifecycle_table_full"
    assert lc.path.read_bytes() == before
    # Existing rows still move (logout of an existing scope is never blocked).
    assert lc.record_native(KEY, "s3", L.NativeFacts(present=False)).state == L.STATE_ABSENT
    assert L.LifecycleStore(NET, root=tmp_path).status() == "ok"

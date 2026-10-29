"""Entity storage is bounded globally, independent of requests and authorities."""

from local_operator.team_bridge.avatar_cache import AvatarCache, Entity


def key(target, incarnation="login-a", scope=("root", "upstream")):
    return (scope, (1, "acc_1", incarnation), target, "0123456789abcdef", 64)


def test_byte_limit_lru_and_replacement():
    cache = AvatarCache(max_entries=3, max_bytes=6)
    entity = Entity(b"abc", "image/png", '"one"')
    cache.put(key("a"), entity)
    cache.put(key("b"), entity)
    assert cache.get(key("a")) == entity
    cache.put(key("c"), entity)
    assert cache.get(key("b")) is None
    assert cache._bytes == 6
    cache.put(key("a"), Entity(b"x", "image/png", '"two"'))
    assert cache._bytes == 4
    cache.put(key("huge"), Entity(b"x" * 7, "image/png", '"huge"'))
    assert cache.get(key("huge")) is None
    assert cache._bytes == 4


def test_observed_authority_change_and_signout_purge_only_scope():
    cache = AvatarCache()
    entity = Entity(b"abc", "image/png", '"one"')
    cache.put(key("a"), entity)
    other = key("b", scope=("other", "upstream"))
    cache.put(other, entity)
    cache.admit(("root", "upstream"), (1, "acc_1", "login-a-again"))
    assert cache.get(key("a")) is None
    assert cache.get(other) == entity
    cache.put(key("a", "login-a-again"), entity)
    cache.admit(("root", "upstream"), None)
    assert cache.get(key("a", "login-a-again")) is None
    assert cache.get(other) == entity


def test_retention_expiry_is_monotonic_and_sweeps_on_put():
    now = [1000.0]
    cache = AvatarCache(max_entries=8, max_bytes=6, ttl_s=60, clock=lambda: now[0])
    entity = Entity(b"abc", "image/png", '"one"')
    cache.put(key("a"), entity)
    now[0] += 59.9
    assert cache.get(key("a")) == entity
    now[0] += 0.1
    assert cache.get(key("a")) is None
    assert cache._bytes == 0
    cache.put(key("b"), entity)
    now[0] += 61
    # An expired entry never holds byte-cap room against a live one.
    cache.put(key("c"), Entity(b"xyzuvw", "image/png", '"two"'))
    assert cache.get(key("c")) is not None
    assert cache._bytes == 6

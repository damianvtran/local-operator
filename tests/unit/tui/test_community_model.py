"""The ``/community`` composer rules (frame T5/T10) and the ``/orgs`` default (Amendment A)."""

from __future__ import annotations

from types import SimpleNamespace

from local_operator.community.service import PostOutcome
from local_operator.tui import community_model as model


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_enter_clears_now_and_a_refusal_brings_the_text_back() -> None:
    composer = model.Composer(draft="hello")
    composer.begin("hello", "k" * 16)
    assert composer.draft == "" and composer.status == "checking"
    assert composer.can_send() is False
    composer.settle(PostOutcome("refused", category="spam_or_promotion", reference="MOD-AAAA-BBBB"))
    assert composer.status == "refused"
    assert composer.draft == "hello", "T5: empty composer gets the text back"


def test_a_new_draft_is_never_overwritten_the_old_text_is_set_aside() -> None:
    composer = model.Composer()
    composer.begin("first", "k" * 16)
    composer.draft = "second"
    composer.settle(PostOutcome("cancelled"))
    assert composer.draft == "second" and composer.set_aside == "first"
    assert composer.restore_set_aside() is False, "never over a live draft"
    composer.draft = ""
    assert composer.restore_set_aside() is True and composer.draft == "first"


def test_rate_limit_keeps_enter_off_until_the_countdown_ends() -> None:
    clock = Clock()
    composer = model.Composer(clock=clock)
    composer.begin("x", "k" * 16)
    composer.settle(PostOutcome("rate_limited", scope="account_burst", retry_after_s=42))
    assert composer.seconds_left() == 42 and composer.can_send() is False
    assert composer.draft == "x"
    clock.now += 42
    assert composer.tick() is True and composer.status == "idle" and composer.can_send()


def test_posted_and_too_late_keep_the_composer_clear() -> None:
    for kind in ("posted", "too_late"):
        composer = model.Composer()
        composer.begin("x", "k" * 16)
        composer.settle(PostOutcome(kind))  # type: ignore[arg-type]
        assert composer.status == kind and composer.draft == ""


def test_a_leading_slash_never_posts_and_double_slash_escapes() -> None:
    composer = model.Composer()
    assert composer.postable_text("/help") is None
    assert composer.postable_text("//help is a command") == "/help is a command"
    assert composer.postable_text("   ") is None


def test_read_only_and_locked_channels_refuse_enter_and_keep_the_draft() -> None:
    composer = model.Composer(draft="keep me")
    welcome = SimpleNamespace(name="welcome-and-rules", locked=False)
    model.composer_for_channel(composer, welcome)
    assert composer.status == "read_only" and not composer.can_send()
    locked = SimpleNamespace(name="help", locked=True)
    model.composer_for_channel(composer, locked)
    assert composer.status == "locked" and composer.draft == "keep me"
    model.composer_for_channel(composer, SimpleNamespace(name="help", locked=False))
    assert composer.status == "idle"


def test_rail_puts_initial_channels_first() -> None:
    names = ["zeta", "help", "general", "welcome-and-rules", "alpha"]
    ordered = model.rail_order([SimpleNamespace(name=n) for n in names])
    assert [c.name for c in ordered] == ["welcome-and-rules", "general", "help", "alpha", "zeta"]


COMMUNITY = SimpleNamespace(id="local-operator", name="Local Operator Community")
TRAN = {"tenant_id": "t1", "tenant_name": "Tran Robotics", "status": "active"}
HOME = {"tenant_id": "t2", "tenant_name": "Minerva", "status": "active", "is_home": True}
INVITE = {"tenant_id": "t3", "tenant_name": "Northwind", "status": "invited"}


def test_groups_never_merge_and_all_is_private_only() -> None:
    orgs, communities = model.org_entries([TRAN, HOME], [COMMUNITY])
    assert [o.kind for o in orgs] == ["all", "private", "private"]
    assert [c.kind for c in communities] == ["community"]
    single, _ = model.org_entries([TRAN], [COMMUNITY])
    assert [o.kind for o in single] == ["private"], "no All row for one org"


def test_default_is_invitation_then_home_then_last_never_the_community() -> None:
    orgs, communities = model.org_entries([TRAN, HOME, INVITE], [COMMUNITY])
    assert model.default_selection(orgs, communities, "local-operator").key == "t3"
    orgs, communities = model.org_entries([TRAN, HOME], [COMMUNITY])
    assert model.default_selection(orgs, communities, "local-operator").key == "t2"
    orgs, communities = model.org_entries([TRAN, {**HOME, "is_home": False}], [COMMUNITY])
    assert model.default_selection(orgs, communities, "t1").key == "t1"
    # A community as the last selection never wins while a private org exists.
    assert model.default_selection(orgs, communities, "local-operator").kind == "all"
    # No private org at all: the community is the answer.
    _, only = model.org_entries([], [COMMUNITY])
    assert model.default_selection([], only, None).key == "local-operator"

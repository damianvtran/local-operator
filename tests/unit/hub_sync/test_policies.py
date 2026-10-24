"""Policy migration, exact-bool validation and real on-disk CAS controls."""

import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator import settings_io
from local_operator.config import ConfigManager, read_config_values
from local_operator.hub_sync.identity import LinkError, LinkIdentity
from local_operator.hub_sync.policies import (
    materialize_policy,
    read_policy,
    update_policy,
)


def identity() -> LinkIdentity:
    return LinkIdentity(
        "agent", "synthetic-local", "https://hub.example/v1", "public", None, "remote"
    )


def test_migration_preserves_legacy_off_once_explicit_policy_wins(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hub", {"auto_update": {"agents": False}})
    link = identity()
    first = materialize_policy(manager, link)
    assert first.policy.subscribe is False and first.policy.auto_push is False
    manager.set_config_value(
        "hub", {**manager.get_config_value("hub"), "auto_update": {"agents": True}}
    )
    assert materialize_policy(manager, link) == first
    another = replace(link, local_id="another")
    assert materialize_policy(manager, another).policy.subscribe is True
    chosen = update_policy(manager, link, expected_revision=first.revision, subscribe=True)
    assert chosen.policy.subscribe is True
    manager.set_config_value(
        "hub", {**manager.get_config_value("hub"), "auto_update": {"agents": False}}
    )
    assert materialize_policy(manager, link) == chosen


def test_deferred_resolution_materializes_captured_hint_not_new_creation_default(
    tmp_path: Path,
) -> None:
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hub", {"auto_update": {"agents": True}})
    link = identity()
    original = materialize_policy(manager, link, legacy_subscribe_hint=False)
    assert original.policy.subscribe is False
    # Neither a subsequently changed creation default nor a later migration
    # hint can rewrite an already explicit per-link setting.
    assert materialize_policy(manager, link, legacy_subscribe_hint=True) == original


def test_fresh_direct_config_edit_changes_digest_and_rejects_stale_write(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path)
    link = identity()
    before = materialize_policy(manager, link)
    document = yaml.safe_load(manager.config_file.read_text())
    document["values"]["hub"]["links"][link.link_id]["subscribe"] = False
    document["values"]["unrelated"] = {"keep": True}
    manager.config_file.write_text(yaml.safe_dump(document))
    actual = manager.config_file.read_bytes()
    with pytest.raises(LinkError) as refusal:
        update_policy(manager, link, expected_revision=before.revision, subscribe=True)
    assert refusal.value.code == "policy-revision-changed"
    assert manager.config_file.read_bytes() == actual
    values = read_config_values(tmp_path)
    assert values is not None
    assert read_policy(values, link).revision != before.revision
    assert values["unrelated"] == {"keep": True}


def test_public_optin_requires_capability_and_explicit_grant(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path)
    link = identity()
    before = materialize_policy(manager, link)
    original = manager.config_file.read_bytes()
    with pytest.raises(LinkError) as refusal:
        update_policy(
            manager,
            link,
            expected_revision=before.revision,
            auto_push=True,
            confirm_public_egress=True,
        )
    assert refusal.value.code == "unsupported-precondition"
    assert manager.config_file.read_bytes() == original
    with pytest.raises(LinkError) as refusal:
        update_policy(
            manager,
            link,
            expected_revision=before.revision,
            auto_push=True,
            can_enable_auto_push=True,
        )
    assert refusal.value.code == "unsafe-publication"
    assert manager.config_file.read_bytes() == original
    enabled = update_policy(
        manager,
        link,
        expected_revision=before.revision,
        auto_push=True,
        subscribe=False,
        can_enable_auto_push=True,
        confirm_public_egress=True,
    )
    assert enabled.policy.to_json() == {"subscribe": False, "auto_push": True}
    assert enabled.public_egress_grant is not None
    values = read_config_values(tmp_path)
    assert values is not None
    assert read_policy(values, link) == enabled
    assert values["hub"]["public_egress_grants"][link.link_id]["link_id"] == link.link_id
    # Turning outbound OFF and changing Subscribe must work while hub capability
    # or credentials are unavailable. Stored ON elsewhere remains a paused fact.
    off = update_policy(manager, link, expected_revision=enabled.revision, auto_push=False)
    assert off.policy.to_json() == {"subscribe": False, "auto_push": False}
    assert off.public_egress_grant == enabled.public_egress_grant


@pytest.mark.parametrize("value", ["true", "false", 0, 1, [], {}])
def test_invalid_policy_never_mutates_unrelated_values(tmp_path: Path, value) -> None:
    manager = ConfigManager(tmp_path)
    link = identity()
    before = materialize_policy(manager, link)
    original = manager.config_file.read_bytes()
    with pytest.raises(LinkError) as refusal:
        update_policy(manager, link, expected_revision=before.revision, subscribe=value)
    assert refusal.value.code == "invalid-link-policy"
    assert manager.config_file.read_bytes() == original


def test_actual_policy_cas_and_language_writer_preserve_both_values(tmp_path: Path) -> None:
    first, second = ConfigManager(tmp_path), ConfigManager(tmp_path)
    link = identity()
    before = materialize_policy(first, link)
    ready, release, started = threading.Event(), threading.Event(), threading.Event()
    real_write = first._write_config
    errors = []

    def paused_write(config, *, drop=None) -> None:
        # `drop` forwarded like the real writer: `set_config_value` passes it as a
        # keyword (deep preservation, PR #2819), and a stub that dropped it would
        # silently turn the write under test into a no-op.
        ready.set()
        assert release.wait(10)
        real_write(config, drop=drop)

    first._write_config = paused_write

    def policy_writer() -> None:
        try:
            update_policy(first, link, expected_revision=before.revision, subscribe=False)
        except BaseException as error:
            errors.append(error)

    def language_writer() -> None:
        started.set()
        try:
            settings_io.write_setting(second, settings_io.BY_KEY["language"], "en")
        except BaseException as error:
            errors.append(error)

    policy = threading.Thread(target=policy_writer)
    language = threading.Thread(target=language_writer)
    policy.start()
    try:
        assert ready.wait(10)
        language.start()
        assert started.wait(10)
    finally:
        release.set()
        policy.join(10)
        if language.ident is not None:
            language.join(10)
    assert not errors and not policy.is_alive() and not language.is_alive()
    values = read_config_values(tmp_path)
    assert values is not None and values["language"] == "en"
    assert read_policy(values, link).policy.subscribe is False
    assert values["hub"]["links"][link.link_id] == {"subscribe": False, "auto_push": False}


def test_unknown_or_mixed_lineage_attestation_cannot_be_overridden_by_confirmation(
    tmp_path: Path,
) -> None:
    manager = ConfigManager(tmp_path)
    link = identity()
    before = materialize_policy(manager, link)
    for reason in ("baseline-unknown", "mixed-lineage"):
        with pytest.raises(LinkError) as refusal:
            update_policy(
                manager,
                link,
                expected_revision=before.revision,
                auto_push=True,
                confirm_public_egress=True,
                ineligible_reason=reason,
            )
        assert refusal.value.code == reason
    assert read_policy(manager.get_config().values, link) == before


def test_public_consent_carve_out_mints_the_first_grant_without_eligibility(
    tmp_path: Path,
) -> None:
    """The cross-surface fix: a public origin with NO grant (reason
    ``unsafe-publication``) mints the grant when consent is given, even though
    ``can_enable_auto_push`` is False — the fact the CLI/desktop PATCH pass.

    Before this, ``update_policy`` raised the ineligible reason BEFORE reaching
    its own consent branch, so the CLI and desktop could never create the first
    public grant; only the TUI's divergent path worked.
    """

    manager = ConfigManager(tmp_path)
    link = identity()
    before = materialize_policy(manager, link)
    original = manager.config_file.read_bytes()
    # WITHOUT consent: still refused, naming unsafe-publication, no grant.
    with pytest.raises(LinkError) as refusal:
        update_policy(
            manager,
            link,
            expected_revision=before.revision,
            auto_push=True,
            can_enable_auto_push=False,
            ineligible_reason="unsafe-publication",
        )
    assert refusal.value.code == "unsafe-publication"
    assert manager.config_file.read_bytes() == original
    # WITH consent: the carve-out mints the grant.
    enabled = update_policy(
        manager,
        link,
        expected_revision=before.revision,
        auto_push=True,
        can_enable_auto_push=False,
        confirm_public_egress=True,
        ineligible_reason="unsafe-publication",
    )
    assert enabled.policy.auto_push is True
    assert enabled.public_egress_grant is not None


@pytest.mark.parametrize(
    "reason", ["unsupported-precondition", "baseline-unknown", "mixed-lineage"]
)
def test_public_consent_carve_out_does_not_bypass_an_earlier_gate(
    tmp_path: Path, reason: str
) -> None:
    """EVERY earlier eligibility gate still refuses even with consent."""

    manager = ConfigManager(tmp_path)
    link = identity()
    before = materialize_policy(manager, link)
    original = manager.config_file.read_bytes()
    with pytest.raises(LinkError) as refusal:
        update_policy(
            manager,
            link,
            expected_revision=before.revision,
            auto_push=True,
            can_enable_auto_push=False,
            confirm_public_egress=True,
            ineligible_reason=reason,
        )
    assert refusal.value.code == reason
    assert manager.config_file.read_bytes() == original


def test_private_link_is_unaffected_by_the_carve_out(tmp_path: Path) -> None:
    """A PRIVATE origin with reason unsafe-publication is still refused."""

    manager = ConfigManager(tmp_path)
    link = replace(identity(), visibility="org", tenant_id="acme")
    before = materialize_policy(manager, link)
    with pytest.raises(LinkError) as refusal:
        update_policy(
            manager,
            link,
            expected_revision=before.revision,
            auto_push=True,
            can_enable_auto_push=False,
            confirm_public_egress=True,
            ineligible_reason="unsafe-publication",
        )
    assert refusal.value.code == "unsafe-publication"


def test_skill_creation_default_follows_the_agents_switch() -> None:
    """Q9: the creation default and the runner ceiling read ONE switch.

    Skills have no kind-wide shared settings key (adding one is a settings/i18n
    catalogue change owned by that lane), so ``creation_policy`` must agree with
    ``HubSyncSettings.auto_for('skill')`` -- both follow hub.auto_update.agents.
    """

    from local_operator.hub_sync.policies import creation_policy
    from local_operator.hub_sync.settings import HubSyncSettings

    off = {"hub": {"auto_update": {"agents": False, "teams": True}}}
    assert creation_policy(off, "skill").subscribe is False
    assert HubSyncSettings.from_values(off).auto_for("skill") is False
    on = {"hub": {"auto_update": {"agents": True, "teams": False}}}
    assert creation_policy(on, "skill").subscribe is True
    # An unset or non-bool spelling is the canonical ON, on both sides.
    unset: dict[str, Any] = {"hub": {}}
    assert creation_policy(unset, "skill").subscribe is True
    odd = {"hub": {"auto_update": {"agents": "yes"}}}
    assert creation_policy(odd, "skill").subscribe is True


def test_revert_removes_the_staged_policy_and_grant_it_reports_removed(tmp_path: Path) -> None:
    """The revert's return value must match the file, not the caller's intent.

    ``revert_policy_if_unchanged`` is the initial-install cleanup: it removes the
    staged entry after a REFUSED or crashed install, and the hub service treats
    whatever is left as authoritative. Under deep preservation an undeclared
    removal is copied straight back off disk — and a surviving
    ``public_egress_grants[link_id]`` satisfies both consent checks, so a grant
    staged by a failed attempt would be re-used instead of minted deliberately by
    the next one. There was no test for this function at all (review round 1, B1).
    """
    from local_operator.hub_sync.identity import LinkPolicy, PublicEgressGrant
    from local_operator.hub_sync.policies import (
        revert_policy_if_unchanged,
        stored_link_policy,
    )

    manager = ConfigManager(tmp_path)
    link = identity()
    # Exactly what the initial install stages: an explicit entry and the public
    # grant that came with it. Seeded directly because `update_policy` refuses an
    # automatic public publication for this synthetic origin — the staging path
    # uses its own consent carve-out, which is the code under test's neighbour.
    manager.set_config_value(
        "hub",
        {
            "links": {link.link_id: LinkPolicy(subscribe=True, auto_push=True).to_json()},
            "public_egress_grants": {link.link_id: PublicEgressGrant(link.link_id).to_json()},
        },
    )
    staged = read_policy(read_config_values(tmp_path) or {}, link)
    seeded = read_config_values(tmp_path) or {}
    assert stored_link_policy(seeded, link.link_id) is not None
    assert seeded["hub"]["public_egress_grants"][link.link_id]["link_id"] == link.link_id

    assert revert_policy_if_unchanged(manager, link, expected_revision=staged.revision) is True

    # Through a FRESH manager, so the assertion cannot be satisfied by the writer's
    # own in-memory snapshot.
    fresh = ConfigManager(tmp_path)
    values = fresh.get_config().values
    assert stored_link_policy(values, link.link_id) is None
    assert link.link_id not in (values.get("hub", {}).get("public_egress_grants") or {})
    assert read_policy(values, link).revision != staged.revision

    # ...and a second call on the now-absent entry reports the truth.
    assert revert_policy_if_unchanged(fresh, link, expected_revision=staged.revision) is False

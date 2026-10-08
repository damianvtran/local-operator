"""Hub-pulled agents: provenance stamping at pull, and the re-fetch verdicts.

The marketplace is faked at the CLIENT boundary (a stub ``RadientClient`` whose
``download_agent_from_marketplace`` writes a real agent archive), so the pull
side runs the real ``import_agent`` + stamping path and the sync side runs the
real comparison — nothing here hand-writes the tags a pull would write. A
credential is never required: the design's degradation (``unavailable`` per
row) is a behaviour under test, not a gap.
"""

from __future__ import annotations

import uuid
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator.agent_profiles import (
    SEED_ORIGIN_PREFIX,
    SeedSyncVerdict,
    install_seed,
)
from local_operator.agent_sync import SyncReport, sync_agent_profiles, sync_payload
from local_operator.agents import (
    HUB_ORIGIN_PREFIX,
    HUB_SHA256_PREFIX,
    NO_HUB_CREDENTIAL_REASON,
    AgentEditFields,
    AgentRegistry,
    hub_fingerprint,
    hub_origin,
    sync_hub_agents,
)


def _fields(**overrides: Any) -> AgentEditFields:
    base: dict[str, Any] = dict(
        name=None,
        description=None,
        tags=None,
        categories=None,
        security_prompt=None,
        hosting=None,
        model=None,
        last_message=None,
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        current_working_directory=None,
    )
    base.update(overrides)
    return AgentEditFields(**base)


def _archive(path: Path, *, name: str, description: str, text: str, tags=None) -> None:
    """Write a real agent archive, shaped like ``export_agent`` produces.

    Built by hand rather than by exporting a registry row: the archive's
    identity is replaced at import anyway, and a template row would leak into
    the registry each test asserts over. The fields are exactly the required
    ones ``AgentData`` validates (``id``, ``name``, ``created_date``,
    ``version``) plus the two the comparison reads.
    """

    agent_yml = yaml.dump(
        {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "created_date": "2026-01-01T00:00:00Z",
            "version": "1.0.0",
            "tags": list(tags or []),
            "categories": [],
        }
    )
    with zipfile.ZipFile(path, "w") as archive_zip:
        archive_zip.writestr("agent.yml", agent_yml)
        archive_zip.writestr("system_prompt.md", text)


class _StubHub:
    """A minimal ``RadientClient``: serves one profile per agent id."""

    def __init__(self, files: dict[str, dict[str, Any]], *, fail: set[str] | None = None):
        self._files = files
        self._fail = fail or set()
        self.requested: list[str] = []

    def download_agent_from_marketplace(
        self, agent_id: str, dest_path: Path, *, with_credential: bool = False
    ) -> None:
        self.requested.append(agent_id)
        if agent_id in self._fail:
            raise RuntimeError(f"simulated hub failure for {agent_id}")
        spec = self._files[agent_id]
        _archive(
            Path(dest_path),
            name=spec["name"],
            description=spec["description"],
            text=spec["text"],
            tags=spec.get("tags"),
        )


def _pull(
    registry: AgentRegistry, *, agent_id: str, name: str, description: str, text: str, tags=None
):
    stub = _StubHub(
        {agent_id: {"name": name, "description": description, "text": text, "tags": tags}}
    )
    outcome = registry.download_agent_from_radient(stub, agent_id)
    imported = outcome.agent
    row = registry.get_agent_by_name(imported.name)
    assert row is not None
    return row


def _tag(row, prefix: str) -> str | None:
    for tag in row.tags:
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return None


def test_a_pull_stamps_the_marketplace_id_and_the_pulled_fingerprint(tmp_path) -> None:
    registry = AgentRegistry(tmp_path)
    row = _pull(
        registry, agent_id="abc-123", name="hunter", description="Tracks things", text="PROMPT v1"
    )

    assert hub_origin(row) == "abc-123"
    assert _tag(row, HUB_SHA256_PREFIX) == hub_fingerprint(
        registry.get_agent_system_prompt(row.id), row.description
    )


def test_a_pull_replaces_a_hub_marker_carried_by_the_archive(tmp_path) -> None:
    """A published archive could name someone else's listing as its source.

    The stamp describes where THIS pull came from, so the archive's own
    ``hub:``/``hub_sha256:`` tags are dropped rather than trusted — otherwise a
    publisher could point the user's next sync at an agent of their choosing.
    """

    registry = AgentRegistry(tmp_path)
    row = _pull(
        registry,
        agent_id="real-id",
        name="hunter",
        description="Tracks things",
        text="PROMPT v1",
        tags=["role", f"{HUB_ORIGIN_PREFIX}someone-else", f"{HUB_SHA256_PREFIX}{'0' * 64}"],
    )

    assert hub_origin(row) == "real-id"
    assert [tag for tag in row.tags if tag.startswith(HUB_ORIGIN_PREFIX)] == ["hub:real-id"]


def test_import_strips_provenance_tags_carried_by_an_archive(tmp_path) -> None:
    """The import choke point refuses ALL provenance, not just the hub pair.

    ``test_a_pull_replaces_a_hub_marker_carried_by_the_archive`` covers the
    pull path, which re-stamps afterwards. This is the plain ``import_agent``
    path (the desktop import route and archive imports), where nothing
    re-stamps: an archive could otherwise plant a ``hub_sha256:`` baseline and
    a ``hub:`` listing for the next no-force sync to act on, or have its row
    adopted by the seed arm (agent review round 1, B1).
    """

    registry = AgentRegistry(tmp_path)
    zip_path = tmp_path / "archive.zip"
    _archive(
        zip_path,
        name="hunter",
        description="Tracks things",
        text="PROMPT v1",
        tags=[
            "role",
            "osint",
            f"{HUB_ORIGIN_PREFIX}evil-listing-1",
            f"{HUB_SHA256_PREFIX}{'a' * 64}",
            "seed:coder",
            "seed_version:9.9.9",
            f"seed_sha256:{'b' * 64}",
        ],
    )

    outcome = registry.import_agent(zip_path)
    imported = outcome.agent
    row = registry.get_agent_by_name(imported.name)
    assert row is not None

    assert list(row.tags) == ["role", "osint"], "only what the author wrote survives import"
    assert hub_origin(row) is None
    assert _tag(row, HUB_SHA256_PREFIX) is None
    assert not any(tag.startswith(SEED_ORIGIN_PREFIX) for tag in row.tags)


def test_a_planted_hub_marker_cannot_make_sync_overwrite_the_row(tmp_path) -> None:
    """The B1 repro: with the strip, the archive's listing is never fetched.

    Before the strip: an archive tagged ``hub:evil``/``hub_sha256:<its own
    text>`` imported as "provably unedited since pull", and the next no-force
    sync fetched the archive-named listing and wrote its text over the row
    (``applied: True``, no force, reproduced by the reviewer). Provenance is
    earned at pull, so the imported row carries no marker and the hub arm has
    nothing to walk.
    """

    registry = AgentRegistry(tmp_path)
    zip_path = tmp_path / "archive.zip"
    _archive(
        zip_path,
        name="hunter",
        description="Tracks things",
        text="ORIGINAL TEXT",
        tags=["role", f"{HUB_ORIGIN_PREFIX}evil-listing-1"],
    )
    outcome = registry.import_agent(zip_path)
    imported = outcome.agent
    row = registry.get_agent_by_name(imported.name)
    assert row is not None

    stub = _StubHub({"evil-listing-1": {"name": "hunter", "description": "d", "text": "PWNED"}})
    verdicts = sync_hub_agents(registry, radient_client=stub, force=False)

    assert stub.requested == [], "a planted marker must not name a listing to fetch"
    assert list(verdicts) == []
    assert registry.get_agent_system_prompt(row.id) == "ORIGINAL TEXT"


def test_a_pull_with_a_nonconforming_id_is_refused_before_any_write(tmp_path) -> None:
    """The id addresses a URL path AND a temp filename: refuse at entry (S-2).

    A non-conforming id used to reach the download and import before the
    provenance stamp dropped it; it is now refused up front -- nothing
    downloaded, nothing imported -- because a crafted value (`../x`, an
    absolute path) could otherwise steer where the archive is written. The
    stamp keeps its own guard as the last gate for any future caller.
    """

    registry = AgentRegistry(tmp_path)

    with pytest.raises(ValueError):
        _pull(registry, agent_id="bad id!", name="hunter", description="d", text="PROMPT v1")

    assert registry.list_agents() == []

    from local_operator.agents import AgentEditFields

    probe = registry.create_agent(AgentEditFields.model_validate({"name": "probe"}))
    stamped = registry._stamp_hub_provenance(probe, "bad id!")
    assert hub_origin(stamped) is None
    assert not any(tag.startswith(HUB_ORIGIN_PREFIX) for tag in stamped.tags)


def test_hub_sync_walks_identical_changed_and_edited_rows(tmp_path) -> None:
    """The verdict table, driven through the real pull path for each setup."""

    registry = AgentRegistry(tmp_path)
    row = _pull(
        registry, agent_id="abc-123", name="hunter", description="Tracks things", text="PROMPT v1"
    )

    def sync(text: str, *, force: bool = False):
        stub = _StubHub(
            {"abc-123": {"name": "hunter", "description": "Tracks things", "text": text}}
        )
        return sync_hub_agents(registry, radient_client=stub, force=force)

    (unchanged,) = sync("PROMPT v1")
    assert unchanged.verdict == "up-to-date"

    (moved,) = sync("PROMPT v2")
    assert moved.verdict == "updated"
    assert moved.applied is True
    assert moved.replaced_instructions == "PROMPT v1"
    assert registry.get_agent_system_prompt(row.id).strip() == "PROMPT v2"

    # A local edit after the update: the re-fetch must refuse without force.
    registry.set_agent_system_prompt(row.id, "MY EDIT")
    (edited,) = sync("PROMPT v3")
    assert edited.verdict == "diverged"
    assert edited.applied is False
    assert registry.get_agent_system_prompt(row.id) == "MY EDIT"

    (forced,) = sync("PROMPT v3", force=True)
    assert forced.verdict == "updated"
    assert forced.applied is True
    assert forced.replaced_instructions == "MY EDIT"
    assert registry.get_agent_system_prompt(row.id).strip() == "PROMPT v3"


def test_a_hub_row_without_a_recorded_baseline_refuses_until_forced(tmp_path) -> None:
    """Same safe direction as legacy seeds: unprovable cleanliness refuses."""

    registry = AgentRegistry(tmp_path)
    row = _pull(registry, agent_id="abc-123", name="hunter", description="d", text="PROMPT v1")
    registry.update_agent(
        row.id,
        _fields(tags=[tag for tag in row.tags if not tag.startswith(HUB_SHA256_PREFIX)]),
    )

    stub = _StubHub({"abc-123": {"name": "hunter", "description": "d", "text": "PROMPT v2"}})
    (verdict,) = sync_hub_agents(registry, radient_client=stub)

    assert verdict.verdict == "diverged"
    assert verdict.applied is False
    assert registry.get_agent_system_prompt(row.id).strip() == "PROMPT v1"


def test_hub_sync_without_a_credential_degrades_per_row(tmp_path) -> None:
    """No credential is never a failure: the rows report why, and nothing raises."""

    registry = AgentRegistry(tmp_path)
    _pull(registry, agent_id="abc-123", name="hunter", description="d", text="PROMPT v1")
    _pull(registry, agent_id="def-456", name="gatherer", description="d", text="PROMPT v1")

    verdicts = sync_hub_agents(registry, radient_client=None)

    assert [verdict.verdict for verdict in verdicts] == ["unavailable", "unavailable"]
    assert all(verdict.reason == NO_HUB_CREDENTIAL_REASON for verdict in verdicts)
    # Ordered by row name (gatherer before hunter), not by pull order: a run's
    # output must not depend on which row the registry happened to list first.
    assert [verdict.hub_id for verdict in verdicts] == ["def-456", "abc-123"]


def test_one_failing_listing_does_not_end_the_run(tmp_path) -> None:
    registry = AgentRegistry(tmp_path)
    _pull(registry, agent_id="broken-id", name="hunter", description="d", text="PROMPT v1")
    _pull(registry, agent_id="fine-id", name="gatherer", description="d", text="PROMPT v1")

    stub = _StubHub(
        {"fine-id": {"name": "gatherer", "description": "d", "text": "PROMPT v1"}},
        fail={"broken-id"},
    )
    verdicts = {verdict.name: verdict for verdict in sync_hub_agents(registry, radient_client=stub)}

    assert verdicts["hunter"].verdict == "unavailable"
    assert "could not re-fetch" in verdicts["hunter"].reason
    assert verdicts["gatherer"].verdict == "up-to-date"


def test_hub_sync_ignores_rows_without_the_marker(tmp_path) -> None:
    registry = AgentRegistry(tmp_path)
    install_seed("reviewer", registry=registry)

    assert sync_hub_agents(registry, radient_client=None) == []


def test_the_coordinator_reports_a_hub_row_once_and_answers_every_requested_name(tmp_path) -> None:
    """One entry per requested name, and a hub row is the HUB verdict only.

    ``hunter`` is a hub row, so the seed arm's "not installed from a packaged
    starter" placeholder is dropped — it is true and useless beside the real
    verdict. ``special`` (a role authored here) and ``ghost`` (a typo) are not
    sync targets at all; an explicit request that silently produced nothing
    would read as success.
    """

    registry = AgentRegistry(tmp_path)
    _pull(registry, agent_id="abc-123", name="hunter", description="d", text="PROMPT v1")
    registry.create_agent(_fields(name="special", description="d", tags=["role"]))

    report = sync_agent_profiles(
        registry, radient_client=None, names=["hunter", "special", "ghost"]
    )

    by_name = {entry.name: entry for entry in report.entries}
    assert sorted(by_name) == ["ghost", "hunter", "special"]
    assert by_name["hunter"].verdict == "unavailable"
    assert by_name["hunter"].hub_id == "abc-123"
    assert "not installed from a packaged starter" in by_name["special"].detail
    assert "no installed role of this name" in by_name["ghost"].detail


def test_the_coordinator_runs_the_seed_arm_while_the_hub_degrades(tmp_path) -> None:
    registry = AgentRegistry(tmp_path)
    install_seed("reviewer", registry=registry)
    _pull(registry, agent_id="abc-123", name="hunter", description="d", text="PROMPT v1")

    report = sync_agent_profiles(registry, radient_client=None)

    assert {entry.name for entry in report.entries} == {"reviewer", "hunter"}
    counts = report.counts()
    assert counts["up-to-date"] == 1 and counts["unavailable"] == 1
    rendered = report.render()
    assert "reviewer: up-to-date" in rendered
    assert "hunter: hub unavailable" in rendered


def test_an_unbumped_update_does_not_render_as_a_no_op() -> None:
    """``1.0.0 -> 1.0.0`` reads as nothing happened (agent review round 1, M1).

    An unbumped body move is a real update; the receipt says what moved
    instead of showing an empty version transition. The verdict itself comes
    from the real classifier in the profiles suite — this pins only the line a
    reader diffs, which is why the verdict is constructed by hand.
    """

    report = SyncReport(
        entries=(
            SeedSyncVerdict(
                name="reviewer",
                verdict="outdated-clean",
                applied=True,
                detail="the packaged starter changed; this copy was unedited",
                installed_version="1.0.0",
                packaged_version="1.0.0",
                replaced_instructions="OLD TEXT",
            ),
        )
    )

    rendered = report.render()
    assert "reviewer: updated to the packaged starter (1.0.0, text moved)" in rendered
    assert "1.0.0 -> 1.0.0" not in rendered


def test_the_payload_carries_every_verdict_field_and_a_kind(tmp_path) -> None:
    registry = AgentRegistry(tmp_path)
    install_seed("reviewer", registry=registry)
    _pull(registry, agent_id="abc-123", name="hunter", description="d", text="PROMPT v1")

    payload = sync_payload(sync_agent_profiles(registry, radient_client=None))

    by_name = {entry["name"]: entry for entry in payload["entries"]}
    assert by_name["reviewer"]["kind"] == "seed"
    assert by_name["hunter"]["kind"] == "hub"
    assert by_name["hunter"]["hub_id"] == "abc-123"
    assert payload["summary"]["up-to-date"] == 1


def test_an_available_update_renders_as_available_not_updated() -> None:
    """A READ-ONLY classification must not claim a write (F2's over-claim trap).

    ``counts()`` used to bucket any ``outdated-clean`` as "updated" and the
    renderer printed "updated to the packaged starter" unconditionally — both
    were true only while a clean row was ALWAYS applied on the spot. Under
    ``--check``/``--dry-run`` it is written nowhere, so both halves must learn
    the distinction, and the applied form keeps its old words.
    """

    def verdict(*, applied: bool) -> SeedSyncVerdict:
        return SeedSyncVerdict(
            name="reviewer",
            verdict="outdated-clean",
            installed_version="1.0.0",
            packaged_version="2.0.0",
            applied=applied,
        )

    available = SyncReport(entries=(verdict(applied=False),))
    text = available.render()
    assert "update available (1.0.0 -> 2.0.0)" in text
    assert "updated to the packaged starter" not in text
    assert available.counts()["available"] == 1
    assert available.counts()["updated"] == 0

    applied = SyncReport(entries=(verdict(applied=True),))
    assert "updated to the packaged starter (1.0.0 -> 2.0.0)" in applied.render()
    assert applied.counts()["updated"] == 1
    assert applied.counts()["available"] == 0

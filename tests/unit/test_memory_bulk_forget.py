"""Filter-based forget: the dry run, the confirmation, and the guards around it.

Every destructive path here is exercised through the shipped human runner
(``memory_cli.run``) rather than a private helper, so the printed preview and the
executed set cannot drift. Facts are personal (never work state) because these
rows live in ``global`` and ``scope_fit`` must admit them.
"""

import json
import time

import pytest

from local_operator.memory import MemoryError, MemoryIdentity, MemoryStore
from local_operator.memory_cli import render_human, run


@pytest.fixture
def rig(tmp_path):
    root = tmp_path / "home" / ".local-operator"
    identity = MemoryIdentity("cli-session", "", "")
    session_dir = root / "sessions" / "cli-session"
    store = MemoryStore(root, identity, session_dir, origin="cli-device")
    return root, identity, session_dir, store


def human(text, root, identity, session_dir=None, prefix="/memory"):
    out, failed = run(text, config=root, identity=identity, session_dir=session_dir, prefix=prefix)
    return render_human(out, failed, prefix), failed


def payload(text, root, identity, session_dir=None, prefix="/memory"):
    out, failed = run(text, config=root, identity=identity, session_dir=session_dir, prefix=prefix)
    assert not failed, out
    return json.loads(out)


def old_fact(store, title, *, days=40, tags=None, verified=False, body="A personal note"):
    fact = {"title": title, "body": body, "created_at": time.time() - days * 86400}
    if tags:
        fact["tags"] = tags
    if verified:
        fact["verified_at"] = time.time() - days * 86400
    return store.put("global", fact)


def statuses(store, items):
    return [store.show("global", item["id"])["status"] for item in items]


def test_dry_run_is_default_deletes_nothing_and_echoes_the_filter(rig):
    root, identity, session_dir, store = rig
    matched = [old_fact(store, f"Old personal note {index}") for index in range(3)]
    keep = store.put("global", {"title": "Fresh personal note", "body": "Dentist number"})

    text, failed = human("forget --scope global --older-than 30d", root, identity, session_dir)

    assert not failed
    assert "Filter: scope global · older than 30d" in text
    assert "3 match(es) of 4 fact(s) in global (75%)" in text
    assert "Nothing was deleted." in text
    assert "To forget exactly these 3 fact(s), run:" in text
    assert statuses(store, matched) == ["active", "active", "active"]
    assert store.show("global", keep["id"])["status"] == "active"


def test_confirmation_deletes_exactly_the_previewed_set(rig):
    root, identity, session_dir, store = rig
    matched = [old_fact(store, f"Old personal note {index}") for index in range(2)]
    fresh = [
        store.put("global", {"title": f"Recent note {index}", "body": "Note"}) for index in range(4)
    ]

    preview = payload(
        "forget --scope global --older-than 30d",
        root,
        identity,
        session_dir,
        prefix="lop memory",
    )
    assert preview["count"] == 2 and preview["total"] == 6
    assert preview["guarded"] is False, "one third of the scope must not need --force"
    assert preview["fingerprint"] and f"--expect {preview['fingerprint']}" in preview["confirm"]

    # The printed command is the confirmation: running it verbatim is the contract.
    assert preview["confirm"].startswith("lop memory forget ")
    confirmed, failed = human(preview["confirm"][len("lop memory ") :], root, identity, session_dir)

    assert not failed, confirmed
    assert "Forgotten 2 fact(s) in global" in confirmed
    assert statuses(store, matched) == ["forgotten", "forgotten"]
    assert statuses(store, fresh) == ["active"] * 4


def test_a_fact_changed_since_the_preview_is_refused_and_nothing_is_deleted(rig):
    root, identity, session_dir, store = rig
    changed, other = [old_fact(store, f"Old personal note {index}") for index in range(2)]
    for index in range(4):
        store.put("global", {"title": f"Recent note {index}", "body": "Note"})

    preview = payload(
        "forget --scope global --older-than 30d",
        root,
        identity,
        session_dir,
        prefix="lop memory",
    )
    assert preview["count"] == 2
    # A human or agent corrects one of the previewed facts before confirmation.
    current = store.show("global", changed["id"])
    store.put(
        "global",
        {"title": "Old personal note 0", "body": "A personal note, corrected"},
        entity=changed["id"],
        expected_version=current["version"],
    )

    refused, failed = human(preview["confirm"][len("lop memory ") :], root, identity, session_dir)

    assert failed
    assert "fingerprint mismatch" in refused
    assert statuses(store, [changed, other]) == ["active", "active"]


def test_committed_revision_is_re_checked_inside_one_transaction(rig):
    root, identity, session_dir, store = rig
    first, second = [old_fact(store, f"Old personal note {index}") for index in range(2)]

    # forget_many walks ids in sorted order: give the LOWER id the valid revision
    # so its tombstone is written before the higher id's stale one fails, and the
    # rollback must undo that first tombstone too.
    good, bad = sorted([first, second], key=lambda item: item["id"])
    with pytest.raises(MemoryError) as refused:
        store.forget_many("global", {good["id"]: good["version"], bad["id"]: "not-the-revision"})
    assert "changed concurrently" in str(refused.value)
    assert statuses(store, [first, second]) == ["active", "active"]


def test_second_confirmed_run_reports_zero_matches_instead_of_failing(rig):
    root, identity, session_dir, store = rig
    matched = [old_fact(store, f"Old personal note {index}") for index in range(2)]
    for index in range(4):
        store.put("global", {"title": f"Recent note {index}", "body": "Note"})
    preview = payload(
        "forget --scope global --older-than 30d",
        root,
        identity,
        session_dir,
        prefix="lop memory",
    )
    command = preview["confirm"][len("lop memory ") :]

    first, failed = human(command, root, identity, session_dir)
    assert not failed and "Forgotten 2 fact(s)" in first
    second, failed = human(command, root, identity, session_dir)
    assert not failed, second
    assert "No facts in global match. Nothing was forgotten." in second
    assert statuses(store, matched) == ["forgotten", "forgotten"]


def test_whole_scope_match_is_refused_without_force_then_allowed_with_it(rig):
    root, identity, session_dir, store = rig
    matched = [old_fact(store, f"Old personal note {index}") for index in range(2)]

    preview = payload(
        "forget --scope global --older-than 30d",
        root,
        identity,
        session_dir,
        prefix="lop memory",
    )
    assert preview["guarded"] is True and preview["total"] == preview["count"] == 2
    # The preview must not defeat its own guard by pre-filling --force.
    assert "--force" not in preview["confirm"]

    refused, failed = human(preview["confirm"][len("lop memory ") :], root, identity, session_dir)
    assert failed
    assert "the whole scope" in refused and "Nothing was forgotten" in refused
    assert statuses(store, matched) == ["active", "active"]

    confirmed, failed = human(
        preview["confirm"][len("lop memory ") :] + " --force", root, identity, session_dir
    )
    assert not failed and "Forgotten 2 fact(s)" in confirmed
    assert statuses(store, matched) == ["forgotten", "forgotten"]


def test_fraction_threshold_refuses_above_half_and_allows_exactly_half(rig):
    root, identity, session_dir, store = rig
    old = [old_fact(store, f"Old personal note {index}") for index in range(4)]
    for index in range(2):
        store.put("global", {"title": f"Recent note {index}", "body": "Note"})

    preview = payload(
        "forget --scope global --older-than 30d",
        root,
        identity,
        session_dir,
        prefix="lop memory",
    )
    # 4 of 6 is above half: the guard must refuse without --force.
    assert preview["count"] == 4 and preview["total"] == 6
    assert preview["guarded"] is True and "--force" not in preview["confirm"]
    refused, failed = human(preview["confirm"][len("lop memory ") :], root, identity, session_dir)
    assert failed and "more than 50% of it" in refused
    assert statuses(store, old) == ["active"] * 4

    # Two old facts forget out of band: 2 of 4 is EXACTLY half, which is not
    # "more than half" and therefore confirms without --force.
    for old_item in old[:2]:
        current = store.show("global", old_item["id"])
        store.put(
            "global",
            {},
            entity=old_item["id"],
            expected_version=current["version"],
            forget=True,
        )
    allowed = payload(
        "forget --scope global --older-than 30d",
        root,
        identity,
        session_dir,
        prefix="lop memory",
    )
    assert allowed["count"] == 2 and allowed["total"] == 4
    assert allowed["guarded"] is False and "--force" not in allowed["confirm"]
    confirmed, failed = human(allowed["confirm"][len("lop memory ") :], root, identity, session_dir)
    assert not failed and "Forgotten 2 fact(s)" in confirmed
    assert statuses(store, old[2:]) == ["forgotten", "forgotten"]


def test_preview_refuses_above_the_bulk_cap(rig):
    root, identity, session_dir, store = rig
    old = [old_fact(store, f"Old personal note {index}") for index in range(201)]
    store.put("global", {"title": "Recent note", "body": "Note"})

    text, failed = human("forget --scope global --older-than 30d", root, identity, session_dir)

    # The preview itself must refuse: printing a confirmation the store rejects
    # (the round-1 MAJOR: 205 matches vs a 200 cap) is never acceptable.
    assert failed
    assert "201 fact(s)" in text and "at most 200" in text
    assert "Narrow the filter" in text and "chunks" in text
    assert statuses(store, old) == ["active"] * 201

    # Exactly the cap still previews, and its command is runnable.
    current = store.show("global", old[0]["id"])
    store.put("global", {}, entity=old[0]["id"], expected_version=current["version"], forget=True)
    preview = payload(
        "forget --scope global --older-than 30d",
        root,
        identity,
        session_dir,
        prefix="lop memory",
    )
    assert preview["count"] == 200
    assert preview["confirm"].startswith("lop memory forget ")


def test_preview_prefixes_are_unique_when_ids_share_a_timestamp(rig, monkeypatch):
    import local_operator.memory as memory

    root, identity, session_dir, store = rig
    ids = iter(
        [
            "0bbbbbbb-1111-7000-8000-000000000001",
            "0bbbbbbb-1111-7000-8000-0000000000a1",
            "0bbbbbbb-1111-7000-8000-000000000002",
            "0bbbbbbb-1111-7000-8000-0000000000b2",
        ]
    )
    monkeypatch.setattr(memory, "_id", lambda: next(ids))
    first = store.put("global", {"title": "Old note one", "body": "A personal note"})
    second = store.put("global", {"title": "Old note two", "body": "Another personal note"})

    text, failed = human(
        "forget --scope global --title-matches '^Old note'",
        root,
        identity,
        session_dir,
        prefix="lop memory",
    )

    assert not failed
    # These ids share their first 35 characters; a fixed 12-character prefix prints
    # the same token twice and leaves a human unable to act on a line.
    assert f"global/{first['id']}" in text and f"global/{second['id']}" in text


def test_catastrophic_filter_regex_is_refused(rig):
    root, identity, session_dir, store = rig
    old_fact(store, "Old personal note")

    text, failed = human(
        "forget --scope global --title-matches '(a+)+$'", root, identity, session_dir
    )
    assert failed
    assert "unbounded repeat" in text and "refused rather than run" in text

    allowed, failed = human(
        "forget --scope global --title-matches '^(Old) personal note$'",
        root,
        identity,
        session_dir,
    )
    assert not failed and "1 match(es)" in allowed


def test_filtered_forget_requires_an_explicit_scope(rig):
    root, identity, session_dir, store = rig
    old_fact(store, "Old personal note")

    text, failed = human("forget --older-than 30d", root, identity, session_dir)

    assert failed
    assert "needs an explicit --scope" in text and "personal bank" in text


def test_tag_unverified_and_title_filters_compose(rig):
    root, identity, session_dir, store = rig
    target = old_fact(store, "Old wifi note", tags=["stale"], body="Wifi code is 4411")
    tagged_but_verified = old_fact(
        store, "Old wifi note two", tags=["stale"], verified=True, body="Wifi code is 4412"
    )
    untagged = old_fact(store, "Old heating note", body="Heating timer is 6am")
    fresh = store.put("global", {"title": "Old-sounding fresh note", "body": "Note"})

    preview = payload(
        "forget --scope global --older-than 30d --unverified --tag stale "
        "--title-matches '^Old wifi' --body-matches 441",
        root,
        identity,
        session_dir,
    )
    assert [row["id"] for row in preview["items"]] == [target["id"]]
    assert "never verified" in preview["filter_text"] and "tag 'stale'" in preview["filter_text"]
    assert statuses(store, [tagged_but_verified, untagged, fresh]) == ["active"] * 3


def test_single_id_and_filter_flags_are_mutually_exclusive(rig):
    root, identity, session_dir, store = rig
    item = old_fact(store, "Old personal note", tags=["stale"])

    text, failed = human(f"forget global/{item['id']} --tag stale", root, identity, session_dir)

    assert failed and "not both" in text
    assert store.show("global", item["id"])["status"] == "active"


def test_bad_duration_and_missing_filters_are_explained(rig):
    root, identity, session_dir, store = rig
    bad, failed = human("forget --scope global --older-than 30", root, identity, session_dir)
    assert failed and "positive duration" in bad
    empty, failed = human("forget --scope global", root, identity, session_dir)
    assert failed and "or filters to preview a bulk forget" in empty

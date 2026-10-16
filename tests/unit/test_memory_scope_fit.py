# Scope-fit guard tests (LOH-25 global-memory hygiene): a deterministic refusal on
# GLOBAL writes whose content is work/lane/progress state, which belongs in session scope.
"""Global writes must refuse work-progress content; personal facts must still store."""

import pytest

from local_operator.memory import MemoryError, MemoryIdentity, MemoryStore, scope_fit

WORK = {
    "title": "Release cut",
    "body": "Merge PR #2463 after CI goes green; next step: cut the release and tag v1.2.3",
}


def _store(tmp_path):
    return MemoryStore(tmp_path, MemoryIdentity("s1", "t1", "a1"))


def test_work_progress_refused_at_global_write(tmp_path):
    """A work-shaped global put is refused BEFORE any DB/index/oplog write."""
    store = _store(tmp_path)
    with pytest.raises(MemoryError) as exc:
        store.put("global", dict(WORK))
    assert "memory://session" in str(exc.value)
    # Refusal precedes every write: no store root, no db, no rows.
    assert not store.root.exists()
    assert store.inventory()["items"] == []


def test_refusal_is_value_free(tmp_path):
    """The offending text and its identifiers are never echoed in the error."""
    store = _store(tmp_path)
    body = "Merge PR #2463 after CI goes green; next step: cut the release"
    with pytest.raises(MemoryError) as exc:
        store.put("global", {"title": "Release cut", "body": body})
    message = str(exc.value)
    assert "#2463" not in message
    assert body not in message


@pytest.mark.parametrize(
    "text",
    [
        "MR !1234 is green; next step: tag v1.2.3",
        "commit 8f3ac21 merged, CI green",
        "PR #2463 is merged; CI green",
        "HOLD pending review on PR #12",
        "the release lane next step is the merge",
        "PR #99 CHANGES",
        "next step: cut the release window",
        "remaining work on the branch",
    ],
)
def test_scope_fit_refuses_work_shapes(text):
    with pytest.raises(MemoryError):
        scope_fit("global", {"title": "x", "body": text, "tags": [], "aliases": [], "subject": ""})


# QA round 1 (M-1) regression set: these were WRONGLY refused by the first matcher.
# A bare `#N`, a bare `next step` or a bare `HOLD` is ordinary personal prose.
@pytest.mark.parametrize(
    "text",
    [
        "Read #1 Foundation, #2 Dune, #3 Hyperion",
        "Office #4 is on the third floor of the Lambton building",
        "My apartment is #7 in the Wakefield complex",
        "My Amazon order #1123 arrives Tuesday",
        "Policy #44120 is the home insurance",
        "I take bus #5 to the market every Saturday",
        "I am up to chapter #12 of the manuscript",
        "Call the clinic on extension #22",
        "My season seat is row 3, seat #18",
        "My gym membership is #88123 and renews in March",
        "Flight NZ #442 lands at 6pm",
        "The invoice #9931 for the heat pump is paid",
        "Room #3 is where the sewing machine lives",
        "Chapter #7 is my favourite in the book",
        "#1 priority is family",
        "The vet is on extension #4",
        "Next step is to let the dough rest for an hour",
        "Recipe: next step is to fold in the flour",
        "Note to self: the next step in the sourdough is the overnight proof",
        "The bank put a HOLD on the cheque for three days",
        "The dry cleaner put a HOLD on my suit until Friday",
        "I keep a HOLD on the parcel at the depot",
    ],
)
def test_ordinary_personal_fact_not_refused(tmp_path, text):
    """NEGATIVE control from QA round 1: personal prose must still store to global."""
    store = _store(tmp_path)
    done = store.put("global", {"title": "Personal note", "body": text, "tags": []})
    assert done["scope"] == "global"


@pytest.mark.parametrize(
    "text",
    [
        "Priya holds the spare key for my flat",
        "I commit to morning workouts",
        "I live at 123 Kowhai Street",
        "Tag Nakamura is the designer",
        "Hold the stretch for thirty seconds",
        "Sha Verma handles the lease",
        "Ci Chen is my accountant",
        "My bank checks were reordered",
        "I build model ships in the garage",
        # Code review round 1: honorific FP and ordinary-prose FPs.
        "Mr Okafor lives at #12",
        "Mrs Patel's flat #4 needs a key",
        "Mr and Mrs Okafor live at #12",
        "Mrs Tran is #1 priority",
        "Merged households with Mr Patel in 2019",
        "I bought a Tag Heuer watch, version 2",
        "I commit 5000000 to savings each year",
        "Check the green bin goes out Tuesday",
    ],
)
def test_personal_words_that_look_workish_store(tmp_path, text):
    store = _store(tmp_path)
    assert store.put("global", {"title": "Note", "body": text, "tags": []})["scope"] == "global"


@pytest.mark.parametrize(
    "fact",
    [
        {
            "title": "Elara Venn",
            "body": "Elara Venn lives in Dunedin; contact elara@example.invalid",
        },
        {"title": "Preferences", "body": "Prefer conventional commit messages"},
        {"title": "Colleagues", "body": "Mira Vale owns the Pergamon contact"},
        {"title": "Location", "body": "Damian is based in Auckland"},
    ],
)
def test_durable_personal_fact_still_stores_to_global(tmp_path, fact):
    """NEGATIVE control: the deliberate personal-helper contract must keep working."""
    store = _store(tmp_path)
    payload = {**fact, "tags": [], "aliases": [], "subject": ""}
    done = store.put("global", payload)
    assert done["scope"] == "global"


def test_explicit_session_path_unaffected(tmp_path):
    """NEGATIVE control: the same work text is legal in session/team/agent scope."""
    store = _store(tmp_path)
    payload = {**WORK, "tags": [], "aliases": [], "subject": ""}
    assert store.put("session", dict(payload))["scope"] == "session"
    assert store.put("team", dict(payload))["scope"] == "team"


def test_forget_never_blocked(tmp_path):
    """Forget is a removal, never refused by the scope-fit guard."""
    store = _store(tmp_path)
    seeded = store.put("global", {"title": "Old", "body": "A personal note", "tags": []})
    store.put(
        "global",
        {},
        entity=seeded["id"],
        expected_version=seeded["version"],
        forget=True,
    )
    assert not any(i["id"] == seeded["id"] for i in store.inventory()["items"])


def test_global_write_via_prepare_fact_is_guarded(tmp_path):
    """prepare_fact alone cannot be used to smuggle work content into global."""
    store = _store(tmp_path)
    with pytest.raises(MemoryError):
        scope_fit("global", store.prepare_fact(dict(WORK)))
    assert not store.root.exists()


def _memory_guide() -> str:
    from local_operator.guides import discover_guides, make_guide_resolver

    guides = {guide.name: guide for guide in discover_guides()}
    body = make_guide_resolver(guides)("guide://memory")
    assert body is not None, "the shipped memory guide must actually resolve"
    return " ".join(body.split())


def test_guide_reworded_for_bounded_scope_fit():
    """The guide must state the bounded, shape-only nature honestly."""
    body = _memory_guide()
    assert "memory://session" in body
    # The old absolute claim is gone; the bounded nature is stated.
    assert "not a content classifier" not in body
    assert "bounded locator filter" in body

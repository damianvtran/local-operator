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
        "checks green",
        "MRs #12 and #13 are waiting on review",
        "MRs rebased onto main",
        "PR rebased onto main",
        "the PR was merged",
        "commit cafebabe merged",
        "sha deadbeef landed",
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
        # Round 3: `checks` in the work-word list reopened the #N false positive;
        # lowercase mr/mrs cannot be told from the honorific.
        "My bank checks #2 and #3 were reordered",
        "The rent checks #2 cleared Friday",
        "I deposited checks #4 and #5",
        "mr smith lives at #12",
        "mrs okafor is at flat #4",
        "mrs tran is #1 priority",
        # Round 4: `PR` is also permanent residency / public relations.
        "My PR application merged into the family one",
        "The PR agency merged with a rival",
        "Our PR consultant squashed the rumour",
        "I merged my notes to main folder",
    ],
)
def test_personal_words_that_look_workish_store(tmp_path, text):
    store = _store(tmp_path)
    assert store.put("global", {"title": "Note", "body": text, "tags": []})["scope"] == "global"


@pytest.mark.parametrize(
    "text,shape",
    [
        # Live global-bank evidence (2026-10-29): five work-shaped writes reached
        # `global` from agents that believed they were storing durable facts, each
        # through ONE missing shape. These are the real texts, so a regression that
        # reopens any of them fails here. The refusal must NAME the shape it matched,
        # never a coarse bucket (QA round 1, Q-4).
        (
            "Saver releases so far: agent-server #233 live (image 20db00fc), #234 live",
            "tracker-ref",
        ),
        ("PR3078 R1 remediation: all source edits done (validate, chain, client)", "tracker-ref"),
        ("Session todo: query Cloudflare MCP for radienthq.com zone id + ruleset", "session-todo"),
        (
            "cloudflare-zone-lookup: next action is mcp__cloudflare_whoami then search zones",
            "next-step",
        ),
        (
            "Do not use memory recall for task progress on this workstream; project record",
            "task-progress",
        ),
    ],
)
def test_live_missed_work_shapes_are_refused(text, shape):
    """The five shapes that reached the live global bank are refused, by name."""
    with pytest.raises(MemoryError) as exc:
        scope_fit("global", {"title": "x", "body": text, "tags": [], "aliases": [], "subject": ""})
    assert shape in str(exc.value)


# The shapes Q-1 and the operator's Q-3 decision keep refusing. `HOLD ... branch` and
# `next step: rebase the branch` were refused on the base and a `branches?`
# pluralisation silently dropped the singular (QA round 1, Q-1), so they are pinned
# here rather than left to the tracker arm, which never covered them.
@pytest.mark.parametrize(
    "text,shape",
    [
        ("HOLD pending review on the branch", "hold-branch"),
        ("next step: rebase the branch", "next-step"),
        # The operator's decision on QA Q-3: a global fact must never carry
        # session-narration framing, so the phrase stays refused even for an errand.
        ("Session todo: dentist", "session-todo"),
    ],
)
def test_shapes_that_stay_refused(text, shape):
    with pytest.raises(MemoryError) as exc:
        scope_fit("global", {"title": "x", "body": text, "tags": [], "aliases": [], "subject": ""})
    assert shape in str(exc.value)


def test_session_todo_refusal_is_actionable_and_the_fact_itself_stores(tmp_path):
    """QA round 1, Q-3: the refusal must say which shape fired AND what to do.

    `scope="session"` alone is not the remedy the writer needs — a personal reminder
    sent to a session bank dies with the conversation. The actionable advice is to
    keep the fact and drop the session framing, so the same content without the
    prefix must store in global.
    """
    store = _store(tmp_path)
    with pytest.raises(MemoryError) as exc:
        store.put(
            "global",
            {"title": "Cloudflare zone", "body": "Session todo: query the zone id", "tags": []},
        )
    message = str(exc.value)
    assert "session-todo" in message
    assert "drop the session framing" in message
    assert "query the zone id" not in message  # still value-free
    stored = store.put(
        "global", {"title": "Cloudflare zone", "body": "query the zone id", "tags": []}
    )
    assert stored["scope"] == "global"


# Negative controls, one class per arm, drawn from ordinary prose and — where the live
# bank has one — from its REAL personal rows (contacts, preferences, environment).
# Round 1 bound each widening: the unspaced ref needs the shape a status line has, the
# plan/progress phrase needs a work subject, and the person-level nouns (`tickets`,
# `issues`, `branches`) are work only beside a work verb or a review state.
@pytest.mark.parametrize(
    "text",
    [
        # Live-bank personal rows: `lanes` in a stored disk-reclaim rule (why `lane`
        # stayed singular in the HOLD arm), and `task progress` / `release status`
        # inside the operator's own boundaries note (why the plan arm needs a subject).
        "never force a shared tool's lock (uv cache prune --force while other lanes "
        "hold the uv cache lock) - report the blocker instead",
        "Global memory is only for durable user/personal facts, preferences, contacts "
        "and environment-not task progress, release status, transient approvals, "
        "debug logs or session resumption.",
        # Plural `PRs` with no tracker ref beside it, from the live bank.
        "UI i18n PRs consolidate under one owner; shared-catalogue merge "
        "order: 1052 actions union.",
        # Two uppercase letters + three digits that are NOT a tracker ref (a JWT alg).
        "assertion = Google ID token (RS256, aud=GOOGLE_VAULT_CLIENT_ID, "
        "iss accounts.google.com).",
        # A bare TODO and a lane label: `session todo` is the phrase, not `todo`.
        "L4 remaining TODO: drawer tabs Asks|Updates|All (D9), transcript flag line "
        "+ in-view auto-read.",
        # A hyphenated key ending in `pr` + one digit: not an unspaced PR ref.
        "never delete or move ~/worktrees/sdk-pr2 (717M).",
        # Review B-1: `PR`/`MR` + digits is also how personal reference numbers are
        # written — `PR` is Philippine Airlines' IATA code, and policy, medical-record,
        # parking-bay and membership numbers take the same form. Each was ACCEPTED on
        # the base and must stay accepted.
        "Flight PR102 Manila to Sydney departs 22:40, seat 14A.",
        "Boarding pass PR1842 - terminal 2, gate 14.",
        "Insurance policy PR90021 renews 1 June, premium by direct debit.",
        "Medical record MR204567 at St Vincent's (Dr Okafor).",
        "I park in bay PR1234 at the office garage.",
        "My gym membership number is MR98765.",
        "The passport application (PR4431) is with the consulate.",
        # The documented bound of the unspaced arm: only a LINE-START ref is a marker.
        "The build I mentioned is covered in PR3078.",
        # Review M-1: a bare plan topic is a noun, not a shape.
        "Our workstreams at the charity are fundraising, events and grants.",
        "I run three workstreams at the school: reading, sports and music.",
        # QA Q-1/Q-2: person-level nouns without work context. Six of these were
        # refused on the head this round fixes and accepted on the base.
        "Heavy snow can hold the branches down until it melts.",
        "The next step is to check the branches for damage after the storm.",
        "Please hold my tickets at the box office until Friday.",
        "There is a HOLD on the concert tickets until payday.",
        "The next step is to collect the tickets from the box office.",
        "next action: book a ticket for the concert on Friday",
        "next action: book tickets for the concert on Friday",
        "Parking tickets: #4412 and #4413 are both paid.",
        "My medical issues are listed in file #7.",
        "The HOA has a hold on all issues of the newsletter until the AGM.",
        # Rust/JSON spec only, and the 2-digit product-code class the arm excludes.
        "I fitted MR16 LED bulbs in the kitchen downlights.",
        # QA round 1's own passing controls, kept so they cannot regress.
        "next action: call the plumber",
        "The next action is to renew the passport before March.",
        "Session notes: the todo list for the trip is on the fridge.",
        "dentist appointment on Thursday at 3pm",
    ],
)
def test_personal_prose_still_stores(tmp_path, text):
    """Every one of these is a durable personal fact and must reach the global bank."""
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

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


# ---------------------------------------------------------------------------
# The two directions, enumerated and COUNTED.
#
# The bias in doubt is STORE THE FACT (see the code comment above `_WORK_TOKEN`): a
# work note that reaches global is caught later by `lop memory triage`, which lists it
# and can move it; a personal fact a shape rule refuses is silent data loss. So every
# false positive from review/QA rounds 1-2 is an accepted case here, every work shape
# is a refused case with the name it reports, and `test_the_bias_is_store_the_fact`
# pins BOTH lengths — a future widening cannot quietly invert the bias.
# ---------------------------------------------------------------------------
PERSONAL_PROSE: tuple[str, ...] = (
    # --- the live bank's own personal rows ------------------------------------
    "never force a shared tool's lock (uv cache prune --force while other lanes "
    "hold the uv cache lock) - report the blocker instead",
    "UI i18n PRs consolidate under one owner; shared-catalogue merge order: 1052 actions union.",
    "assertion = Google ID token (RS256, aud=GOOGLE_VAULT_CLIENT_ID, iss accounts.google.com).",
    "L4 remaining TODO: drawer tabs Asks|Updates|All (D9), transcript flag line "
    "+ in-view auto-read.",
    "never delete or move ~/worktrees/sdk-pr2 (717M).",
    "I fitted MR16 LED bulbs in the kitchen downlights.",
    "The next action is to renew the passport before March.",
    "Session notes: the todo list for the trip is on the fridge.",
    # --- review round 1, B-1: personal reference numbers ----------------------
    # `PR` is Philippine Airlines' IATA code; policy, medical-record, bay and
    # membership numbers take the same letter+digits form.
    "Flight PR102 Manila to Sydney departs 22:40, seat 14A.",
    "Boarding pass PR1842 - terminal 2, gate 14.",
    "Insurance policy PR90021 renews 1 June, premium by direct debit.",
    "Medical record MR204567 at St Vincent's (Dr Okafor).",
    "I park in bay PR1234 at the office garage.",
    "My gym membership number is MR98765.",
    "The passport application (PR4431) is with the consulate.",
    # The documented bound of the unspaced arm: only a LINE-OPENING ref is a marker.
    "The build I mentioned is covered in PR3078.",
    # --- review round 1, M-1: bare plan topics ---------------------------------
    "Our workstreams at the charity are fundraising, events and grants.",
    "I run three workstreams at the school: reading, sports and music.",
    # --- QA round 1, Q-1/Q-2: person-level nouns -------------------------------
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
    # --- review round 2, M-1 / QA round 2, QA2-1: ordinary English verbs ------
    # `push`, `squash` and `cherry-pick` are cooking, prams and fruit.
    "Next step: push the button to confirm the transfer.",
    "The next step is to push the pram to the corner shop.",
    "next step: squash the tomatoes into the passata",
    "Next step: squash the booking for the court on Tuesday.",
    "The next step is to cherry-pick the ripe fruit for the jam.",
    "next step: push the trolley round to the checkout",
    "Next action: squash the aphids on the roses.",
    # --- the operator's own durable preference, from the live bank (01a141b9) ------
    # A prohibition list that MENTIONS the phrase is not a progress report. This row
    # is why `_TASK_PROGRESS` keys on the report shape (`task progress:`, `… on …`).
    "Global memory is only for durable user/personal facts, preferences, contacts and "
    "environment-not task progress, release status, transient approvals, debug logs or "
    "session resumption; those use session/team/agent work banks.",
    "task progress, release status, transient approvals",
    # Review round 2, m-1's third sentence: `against` is not the report shape, so this
    # personal training report stores.
    "task progress against the marathon milestones is on track",
    # --- review round 2, m-1: plan nouns that are also personal vocabulary -----
    "sprint training",
    "marathon milestones",
    "wedding deliverables",
    # --- QA round 2, QA2-2: a ticket HOLD is a box office, not a review state ---
    "There is a HOLD pending review on the concert ticket.",
    "The box office has a hold pending their review of my tickets.",
    # --- QA round 1's own passing controls -----------------------------------
    "next action: call the plumber",
    "dentist appointment on Thursday at 3pm",
)

WORK_SHAPES: tuple[tuple[str, str], ...] = (
    # The five live global-bank rows, verbatim from the real store. Each was written
    # by an agent that believed it was storing a durable fact; each refusal must also
    # NAME the shape, never a coarse bucket (QA round 1, Q-4).
    ("Saver releases so far: agent-server #233 live (image 20db00fc), #234 live", "tracker-ref"),
    ("PR3078 R1 remediation: all source edits done (validate, chain, client)", "tracker-ref"),
    # Review round 2, n-1: the list form an agent writes a session note in.
    ("- PR3078 R1 remediation: done", "tracker-ref"),
    ("1. PR3078 R1 remediation: done", "tracker-ref"),
    ("Session todo: query Cloudflare MCP for radienthq.com zone id", "session-todo"),
    (
        "cloudflare-zone-lookup: next action is mcp__cloudflare_whoami then search zones",
        "next-step",
    ),
    (
        "Do not use memory recall for task progress on this workstream; project record",
        "task-progress",
    ),
    # Operator decision on QA round 1, Q-3: session narration never belongs in global,
    # even for an errand, and the refusal says how to keep the fact.
    ("Session todo: dentist", "session-todo"),
    # Review/QA round 1: the singular work shapes that must keep refusing.
    ("HOLD pending review on the branch", "hold-branch"),
    ("next step: rebase the branch", "next-step"),
    ("the release lane next step is the merge", "next-step"),
    ("next step: cut the release and tag v1.2.3", "release-cut"),
    ("Merge PR #2463 after CI goes green", "tracker-ref"),
    ("the PR was merged", "merge-status"),
    # The DOCUMENTED line-opening bound (guide + `_WORK_REF` comment): a personal
    # identifier that opens a line has the shape of a status line, so it is refused and
    # the writer rephrases. Pinned so the bound cannot drift silently either way.
    ("PR1234 is the bay I use at the office garage.", "tracker-ref"),
    ("MR204567 at St Vincent's (Dr Okafor).", "tracker-ref"),
)

# Sentences that a literal-phrase rule refuses BY OPERATOR DECISION (round 3, item 2:
# "`_TASK_PROGRESS` keys off the literal phrase `task progress` only"). They are
# personal prose, so they are listed and asserted REFUSED rather than dropped: the cost
# is visible here, and reversing the decision is a test change, not an excavation.
DOCUMENTED_PHRASE_COST: tuple[str, ...] = (
    # A report SHAPE, even about a kitchen: `task progress on …` reads as a status line,
    # and the shape (not the subject) is what the arm tests.
    "Task progress on the kitchen renovation is slow.",
    # Review round 2, m-1 measured these as false positives from the noun list. Two of
    # them are `task progress on …` REPORTS, so the report shape refuses them; the
    # third (`against`) is not the shape at all and stores — see PERSONAL_PROSE.
    "Task progress on this week's sprint training is good.",
    "task progress on the wedding deliverables: rings bought",
)


@pytest.mark.parametrize("text", PERSONAL_PROSE)
def test_personal_prose_still_stores(tmp_path, text):
    """Every one of these is a durable personal fact and must reach the global bank."""
    store = _store(tmp_path)
    assert store.put("global", {"title": "Note", "body": text, "tags": []})["scope"] == "global"


@pytest.mark.parametrize("text,shape", WORK_SHAPES)
def test_work_shapes_are_refused_and_named(text, shape):
    with pytest.raises(MemoryError) as exc:
        scope_fit("global", {"title": "x", "body": text, "tags": [], "aliases": [], "subject": ""})
    assert shape in str(exc.value)


@pytest.mark.parametrize("text", DOCUMENTED_PHRASE_COST)
def test_the_cost_of_a_literal_phrase_is_pinned(text):
    """Reports the guard refuses even when the subject is personal.

    `task progress on …` is a status line whatever it is about, so a kitchen renovation
    report is refused and the writer rephrases — while the operator's prohibition list
    in `PERSONAL_PROSE` stores. That pair is the whole boundary.
    """
    with pytest.raises(MemoryError) as exc:
        scope_fit("global", {"title": "x", "body": text, "tags": [], "aliases": [], "subject": ""})
    assert "task-progress" in str(exc.value)


def test_the_bias_is_store_the_fact(tmp_path):
    """Both directions, counted. A widening that inverts the bias fails HERE.

    The numbers are deliberately exact: adding a sentence to either tuple without
    deciding which way it should fall is the mistake this catches.
    """
    assert len(PERSONAL_PROSE) == 45
    assert len(WORK_SHAPES) == 16
    assert len(DOCUMENTED_PHRASE_COST) == 3
    store = _store(tmp_path)
    for index, text in enumerate(PERSONAL_PROSE):
        stored = store.put("global", {"title": f"note {index}", "body": text, "tags": []})
        assert stored["scope"] == "global", text
    for text, _shape in WORK_SHAPES:
        with pytest.raises(MemoryError):
            store.put("global", {"title": "work", "body": text, "tags": []})


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

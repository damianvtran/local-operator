"""The on-demand guide must not turn work longevity into global permission.

The native helper already binds personal facts to global. The regression was its
adjacent guidance: an anonymous profile or a request for future recall could send
ordinary work to that same bank. Exercise the real packaged resolver, not a copy
of the Markdown. These are guidance contracts, not proof a model obeys them;
normal-tool fresh-agent scope-selection pilots remain a separate acceptance gate.
"""


def _memory_guide() -> str:
    # Resolve on demand just as read(guide://memory) does. Keeping this import
    # inside the test also leaves source-only inspection free of product setup.
    from local_operator.guides import discover_guides, make_guide_resolver

    guides = {guide.name: guide for guide in discover_guides()}
    body = make_guide_resolver(guides)("guide://memory")
    assert body is not None, "the shipped memory guide must actually resolve"
    return " ".join(body.split())


def test_anonymous_profile_uses_session_for_work_not_global() -> None:
    body = _memory_guide()

    assert "Anonymous `main` has no agent bank: use session for work, not global." in body
    assert "use session or global instead" not in body


def test_long_lived_work_is_not_a_global_fallback() -> None:
    body = _memory_guide()

    assert "Work state and checkpoints use `memory://session`" in body
    assert "shared work uses `memory://team`" in body
    assert "profile knowledge uses `memory://agent` when authorized" in body
    assert "Long-lived work stays in those scopes, not global." in body
    assert "`memory://global` is only for durable personal/user facts" in body
    assert "future-session request does not turn work state into personal facts" in body
    assert "`memory://global` when intentionally durable" not in body


def test_personal_helper_and_existing_privacy_limits_remain_explicit() -> None:
    body = _memory_guide()

    # The helper's scope contract: session by default, global only on request.
    assert "stores in the **session** bank — the default" in body
    assert 'memory(op="store", scope="global", text="My colleague Elara lives in Dunedin")' in body
    assert "Only those two scope values are accepted" in body
    # LOH-25: the helper gained a bounded SHAPE refusal, so the old absolute
    # "not a content classifier" claim is retired for an honest, bounded one.
    assert "not a content classifier" not in body
    assert "bounded locator filter" in body
    assert "memory://session" in body
    assert "Existing session rows are not promoted or rewritten." in body
    assert "Explicit role allowlists are never widened" in body
    assert "secret NAME references only" in body
    assert "an unlabelled secret can still be stored, journaled and synced" in body


def test_capture_label_and_tombstone_floor_claims_match_the_code() -> None:
    """PR #3086 review r2, m2 + n1.

    The label is the fact's STATUS, so a deliberate edit or move still lists as
    `captured`; the old "(automatic, unreviewed)" gloss contradicted the capture
    paragraph above it. And the eviction floor counts EVERY tombstone the session
    holds — forgets and expiries as well as evictions — so the guide must not read as
    if only evictions are counted.
    """
    body = _memory_guide()

    assert "labelled `captured` (status `candidate`" in body
    assert "(automatic, unreviewed)" not in body
    assert "once the session holds 2,500 tombstones" in body
    assert "its forgets and expiries counted with its evictions" in body


def test_scope_fit_shapes_and_the_session_todo_remedy_are_documented() -> None:
    """Review round 1 n-1/n-2 and QA round 1 Q-3/Q-4.

    The shape list must name what the code actually refuses (a line-opening `PR1234`,
    `session todo`, `task progress`, `next action`), it must not claim the refusal
    says only "the marker", and the `stated` gloss must not re-create the
    contradiction the capture paragraph removed.
    """
    body = _memory_guide()

    assert "a line OPENING with" in body
    assert "naming the shape that" in body
    assert "naming the marker and" not in body
    assert "A `Session todo:` line is refused too" in body
    assert "write the reminder itself, without the session framing" in body
    assert "or `stated` (deliberate)" not in body


def test_the_boundaries_round_3_settled_are_documented() -> None:
    """Review round 2 M-1/m-1/n-2 and QA round 2 QA2-1/QA2-2.

    The guide must not promise a verb as work context (`push`, `squash` are English),
    must not claim a `ticket` HOLD is refused when only `branch` is, and must state the
    two bounds a reader would otherwise only find in a PR thread: a line-OPENING
    personal number, and the literal `task progress` phrase.
    """
    body = _memory_guide()

    assert "a work verb (`rebase`, `squash`, …)" not in body
    assert "unambiguous work token" in body
    assert "or ticket with a review state" not in body
    assert "on a branch with a review state" in body
    assert "`task progress` on a named workstream" not in body
    assert "the phrase `task progress`" in body
    assert "a personal identifier that OPENS a line" in body
    assert "write it mid-sentence" in body

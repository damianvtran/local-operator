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

    assert "Both operations are bound to global; there is **no scope argument**." in body
    # LOH-25: the helper gained a bounded SHAPE refusal, so the old absolute
    # "not a content classifier" claim is retired for an honest, bounded one.
    assert "not a content classifier" not in body
    assert "narrow locator" in body
    assert "memory://session" in body
    assert "Existing session rows are not promoted or rewritten." in body
    assert "Explicit role allowlists are never widened" in body
    assert "secret NAME references only" in body
    assert "an unlabelled secret can still be stored, journaled and synced" in body

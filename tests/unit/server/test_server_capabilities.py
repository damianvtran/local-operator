"""What ``GET /v1/capabilities`` promises, and the one key that is conditional.

The app gates surfaces on this payload, so a key that is advertised but not
honoured — or honoured but not advertised — is a user-visible defect rather than
a documentation slip. ``references`` is the second kind's opposite number: the
renderer shipped the ``@`` picker, the inline chips and the composer tip behind
``features.references``, so the key's ABSENCE is what kept the whole feature dark
until this route published it.

The tests below hold the two facts a future edit can break in one line each: the
key is advertised on a default process, and it is *withheld* by the process that
would not honour it. The second is the one worth spelling out, because
"simplify" here means replacing a conditional with a literal, and that edit looks
harmless in review and is a lie at runtime.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.references import AT_REFERENCES_ENV, at_references_enabled


def _features(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["result"]["features"]


@pytest.mark.asyncio
async def test_a_default_process_advertises_references(test_app_client, monkeypatch):
    """The writer the renderer waits for, present with no configuration.

    Asserted as ``>= 1`` rather than ``== 1`` because that is the comparison the
    client makes (``desktopFeatureState`` reads
    ``(features[feature] ?? 0) >= minimumVersion``): a bump for a future shape
    must keep this test green, and an accidental ``0`` — which a client reads as
    absent — must not.
    """
    monkeypatch.delenv(AT_REFERENCES_ENV, raising=False)
    response = await test_app_client.get("/v1/capabilities")
    assert response.status_code == 200
    features = _features(response.json())
    assert at_references_enabled() is True
    assert features.get("references", 0) >= 1


@pytest.mark.asyncio
async def test_the_kill_switch_withholds_the_key_rather_than_advertising_it(
    test_app_client, monkeypatch
):
    """A process told not to expand must not offer a picker that paints chips.

    ``LOCAL_OPERATOR_AT_REFERENCES`` is read PER CALL precisely so the override
    can be set after the module is imported, and it disables the expansion at
    :meth:`Session.prompt` — the single site every ordinary PROMPT reaches (a
    ``steer`` bypasses it and expands nothing either way, so the switch is not
    what decides that path). A capabilities payload that ignored it would tell
    the app the machine expands mentions, and the user would then be shown a
    reference the model never receives: the message reaches it as the literal
    ``@path`` it was typed as.

    EVERY falsey spelling is exercised, and the list is the predicate's own:
    ``0``, ``false`` and ``no``, compared after ``strip()`` and case-sensitively.
    The first two were the whole of this loop until review, which then showed
    where the gap bites: an inline restatement of the rule that narrowed the
    vocabulary to ``("0", "false")`` agreed with the switch on every case here
    and disagreed on ``"no"`` — so it passed this test and failed the one below.
    The spelling list belongs where the coverage is claimed.
    """
    for spelling in ("0", "false", "no"):
        monkeypatch.setenv(AT_REFERENCES_ENV, spelling)
        assert at_references_enabled() is False
        features = _features((await test_app_client.get("/v1/capabilities")).json())
        assert "references" not in features, spelling


@pytest.mark.asyncio
async def test_the_advertisement_is_read_from_the_switch_and_not_restated(
    test_app_client, monkeypatch
):
    """The payload follows the switch's VALUE, whatever a copy of it would say.

    The failure this guards is not "the wrong value" — it is a SECOND copy of
    the rule that has DRIFTED from it. A route that re-read ``os.environ``
    itself, or that decided membership from a module constant, agrees with the
    expansion today and diverges the first time either the vocabulary or the
    default changes. So the switch is flipped under the payload and the payload
    is required to follow it in the same process.

    WHAT THIS DOES NOT PROVE, because the docstring here claimed it until review
    ran the experiment: it is not a proof of single-reader-ness. A FAITHFUL
    inline copy of the predicate — same variable, same vocabulary, same default
    — passes all three tests in this file; review wrote one into the route and
    got ``3 passed``. What actually fails a restatement is a drifted one, and
    only where the drift is visible in one of the spellings exercised here or
    above (narrowing the vocabulary to ``("0", "false")`` is caught by ``"no"``,
    which is why this loop and the one above both carry it). The code imports
    the predicate; this test holds it to the predicate's behaviour, and the
    comment above the key is where the single-reader rule is stated.
    """
    seen: list[bool] = []
    for spelling in (None, "yes", "no", "1"):
        if spelling is None:
            monkeypatch.delenv(AT_REFERENCES_ENV, raising=False)
        else:
            monkeypatch.setenv(AT_REFERENCES_ENV, spelling)
        enabled = at_references_enabled()
        features = _features((await test_app_client.get("/v1/capabilities")).json())
        assert ("references" in features) is enabled, spelling
        seen.append(enabled)
    # The loop must actually have moved the switch, or it proves nothing.
    assert seen == [True, True, False, True]

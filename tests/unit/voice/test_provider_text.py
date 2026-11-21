"""``redact_provider_text`` / ``secrets_in_play``: provider text is shown and logged.

Every key here is synthetic. The helper exists because the exact-value scrub only
knows a WHOLE key, and a provider may echo a prefix, a masked ``abcd...wxyz`` or a
key with whitespace in it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.redaction_shapes import REDACTION_MARKER
from local_operator.voice.provider_text import redact_provider_text, secrets_in_play

KEY = "xk9Q7m2ZrTb4WcL8vNd3Hs5Yp"
OTHER = "rk-Qw83nPz0Lm41Vt69Zx"


def _leaks(text: str, key: str, run: int = 6) -> bool:
    folded = "".join(ch for ch in text.casefold() if not ch.isspace() and ch != "-")
    flat = "".join(ch for ch in key.casefold() if not ch.isspace() and ch != "-")
    return any(flat[i : i + run] in folded for i in range(len(flat) - run + 1))


@pytest.mark.parametrize(
    "echo",
    [
        KEY,
        KEY[:8],
        KEY[8:15],
        KEY.upper(),
        f"{KEY[:4]}\u2026{KEY[-4:]}",
        f"{KEY[:4]}...{KEY[-4:]}",
        f"{KEY[:4]}****{KEY[-4:]}",
        " ".join(KEY[i : i + 5] for i in range(0, len(KEY), 5)),
        "-".join(KEY[i : i + 5] for i in range(0, len(KEY), 5)),
        "\n".join(KEY[i : i + 7] for i in range(0, len(KEY), 7)),
    ],
)
def test_every_echo_shape_loses_its_six_character_runs(echo: str) -> None:
    out = redact_provider_text(f"Rejected credential {echo} for now.", [KEY])
    assert not _leaks(out, KEY), out
    assert out.startswith("Rejected credential ") and out.endswith(" for now.")
    assert REDACTION_MARKER in out
    assert KEY[:4] + "\u2026" + KEY[-4:] not in out


def test_every_secret_in_play_is_removed_not_only_the_first() -> None:
    out = redact_provider_text(f"a {KEY[:9]} b {OTHER[2:12]} c", [KEY, OTHER])
    assert not _leaks(out, KEY) and not _leaks(out, OTHER)
    assert out.startswith("a ") and out.endswith(" c")


@pytest.mark.parametrize(
    "sentence",
    [
        "Rejected key [redacted]: too many concurrent requests.",
        "Your Radient credit balance is too low for speech. Add credits in the Console.",
        "Too many concurrent requests: your plan allows 2 at once.",
        "The audio could not be decoded as a supported container.",
    ],
)
def test_ordinary_sentences_are_left_alone(sentence: str) -> None:
    assert redact_provider_text(sentence, [KEY, OTHER]) == sentence


def test_a_common_word_is_not_a_secret_unless_it_is_part_of_the_actual_key() -> None:
    # The rule is per secret: "balance" is a 7-letter word, and it is only a leak
    # for a key that really contains it.
    assert redact_provider_text("balance is low", [KEY]) == "balance is low"
    has_word = "zz-balance-77Qk"
    assert "balance" not in redact_provider_text("balance is low", [has_word])


def test_short_secrets_are_skipped_entirely() -> None:
    sentence = "the test key was refused"
    short: list[Any] = ["test", "a", "", None, "el-test"]
    assert redact_provider_text(sentence, short) == sentence


def test_run_threshold_is_six_characters() -> None:
    assert KEY[:5] in redact_provider_text(f"see {KEY[:5]} there", [KEY])
    assert KEY[:6] not in redact_provider_text(f"see {KEY[:6]} there", [KEY])


def test_no_secrets_returns_the_text_unchanged() -> None:
    assert redact_provider_text("nothing to hide", []) == "nothing to hide"
    assert redact_provider_text("", [KEY]) == ""


def test_the_exact_key_scrub_still_applies_first() -> None:
    assert KEY not in redact_provider_text(f"Authorization: Bearer {KEY}", [KEY])


# -- gathering the secrets a request could have used -------------------------------


class _Store:
    def __init__(self, **keys: Any) -> None:
        self.keys = keys
        self.calls: list[str] = []

    async def get_api_key(self, provider: str, session_id: Any = None, **kw: Any) -> Any:
        assert kw.get("read_only") is True  # a lookup here must never move routing
        self.calls.append(provider)
        value = self.keys.get(provider)
        if isinstance(value, Exception):
            raise value
        return value

    async def get_persisted_api_key(self, provider: str, session_id: Any = None, **kw: Any) -> Any:
        self.calls.append(provider)
        value = self.keys.get(provider)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.mark.asyncio
async def test_secrets_in_play_collects_every_stored_key() -> None:
    store = _Store(
        elevenlabs=KEY,
        radient=OTHER,
        groq="groq-Mm2x91ZzKp07",
        **{"openai-key": "sk-oa-0123456789ab"},
    )
    got = await secrets_in_play(
        store,  # type: ignore[arg-type]
        config_dir=Path("."),
        base_url="https://api.radienthq.com/v1",
        session_id="s1",
    )
    assert (
        KEY in got and OTHER in got and "groq-Mm2x91ZzKp07" in got and "sk-oa-0123456789ab" in got
    )


@pytest.mark.asyncio
async def test_secrets_in_play_degrades_when_the_store_is_unreadable() -> None:
    store = _Store(elevenlabs=RuntimeError("database is locked"), radient=OTHER)
    got = await secrets_in_play(
        store,  # type: ignore[arg-type]
        config_dir=Path("."),
        base_url="https://api.radienthq.com/v1",
        session_id="s1",
    )
    assert OTHER in got and KEY not in got  # one unreadable row does not hide the rest


@pytest.mark.asyncio
@pytest.mark.parametrize("store", [None, object()])
async def test_secrets_in_play_never_raises_on_a_missing_or_foreign_store(store: Any) -> None:
    got = await secrets_in_play(
        store, config_dir=None, base_url="https://api.radienthq.com/v1", session_id=None
    )
    assert isinstance(got, list)

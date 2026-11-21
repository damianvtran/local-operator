"""The ONE speech provider the user chose (``speech.provider``).

The user picks ``radient`` (the Radient cloud, billed to their Radient credit)
or ``elevenlabs`` (their own key). There is no ordering and no fallthrough: a
request is served by that provider alone, and when it cannot serve, it fails
with that provider's own reason. Hearing (STT) and speaking (TTS) use the same
provider, and hands-free voice, the Speak button and the voice routes all read
this one setting.

UNSET means "no provider chosen yet". The voice routes refuse it; the Speak
button keeps the whole cascade (see ``server/routes/speech.py``).

The value is read from DISK on every call, not from a ``ConfigManager`` the
server built at start-up: the write normally comes from another process (the
desktop settings page, ``lop config edit``), and this value decides whose
account is billed, so a stale copy is not an option. ``read_config_values`` never
writes, moves or reports anything, so a read is safe on a hot path.

Import-light, and ``settings_io`` is not imported here: the registry row carries
the literal values and ``tests/unit/voice/test_speech_provider.py`` pins them
against these.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from local_operator.stt import AudioPath
from local_operator.tts import VoicePath

SETTING_KEY = "speech.provider"

SpeechProvider = Literal["radient", "elevenlabs"]

#: Closed. A new provider is a new line in each map below AND a registry choice;
#: OpenAI and Groq are deliberately not pinnable.
PROVIDERS: tuple[SpeechProvider, ...] = ("radient", "elevenlabs")

#: The one rung of each cascade a provider may be served by.
PINNED_STT: dict[str, AudioPath] = {
    "radient": AudioPath.PROVIDER_STT_RADIENT,
    "elevenlabs": AudioPath.PROVIDER_STT_ELEVENLABS,
}
PINNED_TTS: dict[str, VoicePath] = {
    "radient": VoicePath.PROVIDER_TTS_RADIENT,
    "elevenlabs": VoicePath.PROVIDER_TTS_ELEVENLABS,
}


def configured_provider(config_dir: Path) -> Optional[SpeechProvider]:
    """The chosen provider, or ``None`` when nothing valid is chosen.

    Anything that is not exactly one of :data:`PROVIDERS` (absent, hand-edited
    to another vendor, the wrong type, a file that cannot be read right now)
    reads as unset. That is the safe direction: unset spends nothing, whereas
    guessing a provider from a bad value could bill the wrong account.
    """
    from local_operator.config import read_config_values

    try:
        values = read_config_values(Path(config_dir))
    except TypeError:  # a config_dir that is not path-like (a test double)
        return None
    section = (values or {}).get("speech")
    raw = section.get("provider") if isinstance(section, dict) else None
    # Spelled out so the return type narrows without a cast.
    if raw == "radient":
        return "radient"
    if raw == "elevenlabs":
        return "elevenlabs"
    return None

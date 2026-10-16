"""The Gemini Live front end's locked configuration, as DATA (session-voice.md R2, R7).

``POST /v1/desktop/voice/live-token`` mints a single-use ephemeral token whose
setup is LOCKED to what this module builds: the model, the system instruction,
the two tools, AUDIO modality, input and output transcription, session
resumption with sliding-window compression, and manual activity detection.
Electron main opens the socket with that token, and because the setup travels
inside the token, nothing main (or anything that steals the token) sends can
add a tool or rewrite the instruction.

WHY THIS IS ITS OWN MODULE. The system instruction and the tool list are the
two values the pending R4 decision (who speaks replies) may change. They are
module-level constants here, built into the request body by one pure function,
so changing them is a data edit with one test that pins the shape, and the
route that sends the body never has to change.

R7, A HARD RULE THAT LIVES HERE: the tool list has no gate tool and no answer
tool. ``relay_to_session`` passes the operator's words on; ``voice_control``
names a client-side playback action. Neither can approve, deny or answer
anything. ``tests/unit/voice/test_live_token.py`` pins the tool set exactly.

Nothing in this module performs I/O, reads a credential or logs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from local_operator.voice.controls import CONTROL_ACTIONS

#: The Live model the token is locked to. "Gemini 3.8 Live" is the default
#: low-latency voice model in Google's docs (fetched 2026-10-16).
LIVE_MODEL = "gemini-3.8-live"

#: The R2 system instruction, verbatim from the architect's record. R4 may
#: change it: edit HERE, and the body test re-pins it.
SYSTEM_INSTRUCTION = (
    "You are the ears of another agent. For each take call exactly one tool. "
    "Never answer, summarise or advise."
)

RELAY_TOOL = "relay_to_session"
CONTROL_TOOL = "voice_control"

#: The two tools, NON_BLOCKING so a pending call never holds the model's turn.
#: The client always answers ``{status}`` with ``scheduling: SILENT`` (R2), so a
#: tool response never makes the model speak.
FUNCTION_DECLARATIONS: tuple[dict[str, Any], ...] = (
    {
        "name": RELAY_TOOL,
        "description": "Pass the user's words on to the agent they are talking to.",
        "behavior": "NON_BLOCKING",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "text": {
                    "type": "STRING",
                    "description": "The user's words exactly as spoken.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": CONTROL_TOOL,
        "description": "Control the voice playback itself; never reaches the agent.",
        "behavior": "NON_BLOCKING",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "enum": list(CONTROL_ACTIONS),
                    "description": "Which playback control the user asked for.",
                },
            },
            "required": ["action"],
        },
    },
)

#: Google's documented defaults, stated explicitly so the token says what it
#: does: one minute to OPEN a session, thirty to use it (R9 mints a fresh
#: token at the thirty-minute expiry).
NEW_SESSION_WINDOW = timedelta(minutes=1)
TOKEN_LIFETIME = timedelta(minutes=30)

#: One session per token. Resuming the session does not count as a use
#: (API reference, ``AuthToken.uses``), so R9's reconnect across the ~10 min
#: connection resets works with ``uses: 1``.
TOKEN_USES = 1


def live_setup() -> dict[str, Any]:
    """The ``BidiGenerateContentSetup`` the token locks, in Google's wire casing."""
    return {
        "model": f"models/{LIVE_MODEL}",
        "generationConfig": {"responseModalities": ["AUDIO"]},
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "tools": [{"functionDeclarations": [dict(item) for item in FUNCTION_DECLARATIONS]}],
        # Push-to-talk (R10): the client sends activityStart/End itself.
        "realtimeInputConfig": {"automaticActivityDetection": {"disabled": True}},
        # R9: the session is disposable but survives the connection resets.
        "sessionResumption": {},
        "contextWindowCompression": {"slidingWindow": {}},
        # R3 reads the input transcript; R13 (slice C) reads the output one.
        "inputAudioTranscription": {},
        "outputAudioTranscription": {},
    }


def _rfc3339(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def token_request_body(now: datetime) -> dict[str, Any]:
    """The ``POST v1beta/auth_tokens`` body for a token minted at ``now``.

    No ``fieldMask``: per the API reference, an empty mask with a
    ``bidiGenerateContentSetup`` present means the effective setup is taken
    ENTIRELY from the token and the connection's own setup message is ignored.
    That is the strongest lock there is, and the one R2 asks for.
    """
    return {
        "uses": TOKEN_USES,
        "expireTime": _rfc3339(now + TOKEN_LIFETIME),
        "newSessionExpireTime": _rfc3339(now + NEW_SESSION_WINDOW),
        "bidiGenerateContentSetup": live_setup(),
    }


def tool_names(body: dict[str, Any]) -> list[str]:
    """Every function name a token body locks in, for the R7 pin."""
    setup = body.get("bidiGenerateContentSetup") or {}
    return [
        str(declaration.get("name"))
        for tool in setup.get("tools") or []
        for declaration in tool.get("functionDeclarations") or []
    ]


__all__ = [
    "CONTROL_TOOL",
    "FUNCTION_DECLARATIONS",
    "LIVE_MODEL",
    "NEW_SESSION_WINDOW",
    "RELAY_TOOL",
    "SYSTEM_INSTRUCTION",
    "TOKEN_LIFETIME",
    "TOKEN_USES",
    "live_setup",
    "token_request_body",
    "tool_names",
]

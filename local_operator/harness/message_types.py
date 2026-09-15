"""The custom-message vocabulary the harness and the application both match on.

WHY THIS MODULE EXISTS
----------------------
Every renderer that turns a transcript into an LLM-visible message list has to
recognise the same handful of ``CustomMessage.custom_type`` markers: the four
session records (``session_incident`` and its three siblings), a peer
delivery, a hub delivery, and a todo reminder. Each one used to be defined
beside the subsystem that OWNS it — ``local_operator.incidents``,
``local_operator.session.peer``, ``local_operator.tools.builtin`` and
``local_operator.harness.comms``.

That was fine while the only renderer was ``session/session.py``, which imports
those subsystems anyway. It stopped being fine once the renderer was hoisted
into ``harness/render.py`` for the evaluation benchmark: an episode may not
import session, tools or incidents code (``tests/unit/evaluation/runner/
test_isolation.py``), so importing a *string* from any of them made the shared
renderer unreachable from the very surface it was hoisted for. Measured on the
tree before this module existed, importing ``local_operator.harness.render``
dragged in **17** denied modules: ``incidents`` contributed 1, ``session.peer``
2, ``tools.builtin`` 11, and ``harness.comms`` 8 — the last one because it
imports ``session.peer`` and ``session.transcript`` at module level for its own
transcript-replay work. Moving six of the seven would still have leaked those
8 through that one remaining import.

WHY THE HARNESS PACKAGE
-----------------------
The home has to be importable by BOTH sides, which rules out every module that
owns one of these markers or lives in the owning package: ``local_operator.session``
and ``local_operator.tools`` are barred for a runner by that same denylist, so a
marker left in either keeps ``harness/render.py`` unimportable and the hoist
pointless. The harness is the layer both sides already share — the deliberately
shared one, and the one the denylist allows — so the vocabulary lives here: one
definition per marker, imported by its owner rather than defined next to it.

THE CONSTRAINT THAT KEEPS THIS MODULE USEFUL
--------------------------------------------
Import NOTHING from ``local_operator`` here, and keep it that way. This module
is on the renderer's import path, so a single heavy import — even a re-export
"for tidiness" — restores the leak it exists to remove. (``local_operator``'s
and ``local_operator.harness``'s own ``__init__`` files are inert for the same
reason: a marker module that pulls a package ``__init__`` with it would leak
just as surely.)

VALUES ARE WIRE FORMAT
----------------------
These are marker strings persisted into transcripts and compared by equality
across the session, the TUI, the phone projection and a resumed replay, and the
``session_incident`` value is additionally pinned against a literal inside
``local_operator.resume`` (which spells it out to keep its own import list
empty). Each value below is byte-identical to the one its old home defined.
Changing a character of one is a compatibility break — a resumed session, a
half-written transcript and the phone would stop recognising existing rows — not
a rename.
"""

#: Custom-message type journaled by the session; rendered to a user message by
#: the shared renderer (``harness/render.py``) so both a live next-turn and a
#: resumed replay see the same incident.
SESSION_INCIDENT_MESSAGE_TYPE = "session_incident"

#: Custom-message type journaled by the session when a session credential is
#: stored or forgotten mid-conversation. The ONLY other advertisement of a
#: stored credential is the ``<session-credentials>`` block in the volatile
#: system-prompt tail, which the model has no reason to re-read when it
#: changes — so an operator who runs ``/credential FOO_KEY`` and says "I just
#: added the key" left the model to guess names until it happened to notice
#: the tail. This message lands in the LIVE context only, naming the KEY
#: ONLY: the value must never ride a message the provider sees. It is
#: deliberately NOT persisted — credentials are process-memory-only, so a
#: replayed "$FOO_KEY is injected into every bash command" would assert an
#: env var a restarted session does not have (review round 1, R2). Resume-time
#: discovery is already served honestly by the ``<session-credentials>``
#: block, which the prompt tail rebuilds from the (empty) live store each
#: turn.
SESSION_CREDENTIAL_MESSAGE_TYPE = "session_credential"

#: Custom-message type journaled by the session when the running model changes
#: (a deliberate ``set_model``, or a failover fallback to another model).
#: Rendered to a user message the same way as an incident, so the model NOTICES
#: it is now answering as a different model rather than only seeing a changed
#: static "Model:" line in the system prompt. Persisted, so a resumed session
#: replays the switch history too.
SESSION_MODEL_SWITCH_MESSAGE_TYPE = "session_model_switch"

#: Custom-message type journaled by the session when an MCP server that was
#: ANNOUNCED BROKEN to the model connects again. The failure half of that pair
#: has always been model-visible (``McpManager.on_incident`` ->
#: ``Session._on_mcp_incident`` -> a ``session_incident`` message); the recovery
#: half was not, so an operator who ran ``/mcp login <server>`` mid-session left
#: the model holding a death notice — and its ``mcp`` hint, "its tools are gone
#: ... Do not call its tools" — for a server that had been usable for the rest
#: of the session. Observed live against ``minerva-qa``.
#:
#: It is a DEDICATED type rather than another ``session_incident`` because
#: ``journal_incident`` runs :func:`classify_incident`, whose ``mcp`` rule
#: matches the substring "mcp" and would append ``_HINTS["mcp"]`` — precisely
#: the "its tools are gone" sentence — to a message saying the opposite.
#:
#: LIVE CONTEXT ONLY, deliberately not persisted: an MCP connection is
#: process-scoped (``McpManager._connections`` is instance state and
#: ``disconnect_all`` runs on dispose), so a replayed "its N tools are
#: available to you now" would assert a live capability a restarted session may
#: not have — the same class as the credential record above, and the more
#: likely case for exactly the servers this serves, whose grants expire.
SESSION_MCP_RECOVERY_MESSAGE_TYPE = "session_mcp_recovery"

#: ``CustomMessage.custom_type`` of a peer (cross-session) message. It MUST be
#: added to the LLM-visible custom-type allow-list in ``session.py`` (beside
#: ``HUB_MESSAGE_TYPE``) or the human sees the transcript row but the model
#: never does. Rendered as a distinct inbound card in every surface — never as
#: the user's own turn, and never as a hub-parent message (a peer is not a
#: parent).
#:
#: Defined here rather than in a module of its own (``session/peer.py`` held
#: nothing else) or inside ``session.py``: ``mobile/projection.py`` and
#: ``tui/app.py`` compare it without wanting the heavyweight ``Session`` import
#: graph, and the shared renderer must reach it without session code at all.
PEER_MESSAGE_MESSAGE_TYPE = "peer_message"

#: ``CustomMessage.custom_type`` of a hub message in either direction. The
#: shared renderer renders it as a user message carrying ``details["text"]``
#: (see ``_default_convert_to_llm``); persisted like any other message entry, so
#: a resumed child still sees what it was told. ``harness/comms.py``, which
#: writes and classifies these, imports it from here.
HUB_MESSAGE_TYPE = "hub_message"

#: The custom-message type the session's continuation guardrail injects at the
#: yield boundary (``Session._todo_continuation``). The vocabulary is owned by
#: the todo feature (``tools/builtin.py`` holds the list it guards) but DEFINED
#: here, for the same reason ``HUB_MESSAGE_TYPE`` and
#: ``WAKE_PROMPT_MESSAGE_TYPE`` are not defined in the session that renders
#: them — and this one more sharply than those two, since the renderer must
#: tell a real reminder from the newest already-injected one without importing
#: the tool layer.
TODO_REMINDER_MESSAGE_TYPE = "todo_reminder"

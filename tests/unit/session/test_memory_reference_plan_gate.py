"""The plan gate must ask its question about the copy the tools exempt.

**The defect this closes.** `#2511` (merged) taught the agent READ path that a ``q``
VALUE which IS a minted structural reference is a lookup rather than prose, and `#2524`
(open) taught the ``memory`` tool's ``recall`` the same for ``text``. Both exemptions
live at the TOOL layer. ``AgentLoop._plan_call`` asks
``memory_privacy.retained_arguments`` whether a call carries a recognised sensitive
memory value BEFORE validation and BEFORE the tool is dispatched, and refuses the whole
call with a synthetic ``invalid_arguments`` failure when it answers yes — the tool never
runs, and no ``tool_execution_start`` is emitted. Every live path funnels through that
seam (native dispatch, the eval bridge's ``dispatch_tool``, the monitor runner), so both
tool-layer exemptions were UNREACHABLE in a live session: the call was refused one level
above them.

**The refusal is a chance collision, not a reference problem.** ``memory.guard``'s
free-text arm admits no reference as such: it refuses a string that MATCHES a sensitive
shape, and a ``d_`` + 32 hex device id whose hex tail happens to satisfy the IBAN shape
and its mod-97 check matches one by accident — the collision ``structural_reference``
names at its own source (it is what refused every put on an install whose device id
passed mod-97). The ids used here are that collision, held verbatim from it. A
device-id or UUID reference that collides with nothing is admitted at the plan gate
today and after; those are the CONTROLS below, and they are why this file pins both
classes rather than only the failing one.

Measured at the seam over the 34-row corpus below, executed on the base tree and on this
one from the SAME file: the corpus digest is identical on both sides (retention's own
answers do not move), and the plan gate's verdict flips on exactly the six admitted-
reference rows — a colliding id in a read's ``q`` (with and without a raw JSON carrier,
and beside ``limit=5``) and an id in a recall's ``text`` (colliding or not) — and on
nothing else.

RED (the same tests over the base tree, ``PYTHONPATH`` pointed at it) and GREEN (here)
are both session-level: a real ``Session``, the real builtin tools, a scripted provider,
and the assertion is on the session's own events and retained rows — never on the
planning helper alone, which is the shape of the evidence that missed this defect the
first time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.loop import _plan_guard_inputs
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolContext,
)
from local_operator.memory_privacy import retained_arguments
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import build_read_tool, build_write_tool
from local_operator.tools.memory_tools import build_memory_tool, execute_memory

MODEL = ModelSpec(provider="test", model_id="deterministic", context_window=100000)

#: ``network.identity.device_id_for``: ``d_`` + sha256 hex[:32], whose hex tail satisfies
#: the IBAN shape + mod-97 check BY CHANCE. This is the value the refusal that #2511 and
#: #2524 fixed was held from.
DEVICE_ID = "d_ef73a4312e5c4686196b30c035c8a05a"
#: Same shape, colliding with nothing: the plan gate admits this one on the base tree
#: too, so it is a CONTROL for the collision rather than for the exemption.
PLAIN_DEVICE_ID = "d_0123456789abcdef0123456789abcdef"
#: The other admitted shape (``memory._id()``'s canonical UUID). No contiguous run of 15+
#: characters exists in it, so the IBAN/card arms cannot collide with one.
ENTITY_ID = "0191f2c3-4d5e-7f80-9abc-def012345678"
CARD = "4111111111111111"
IBAN = "GB82WEST12345698765432"

PROSE_FACT = {"title": "Host", "body": "staging host is stage-2"}

#: The PLAN GATE's refusal, verbatim. A control asserts EQUALITY against this rather than
#: a substring, because byte-identity to today is the claim: the tool layer words its own
#: refusal differently ("Store it with lop secret and keep only its NAME here"), and that
#: difference is exactly how a seam refusal is told apart from a tool that ran.
PLAN_GATE_REFUSAL = "Sensitive value refused. Use lop secret; memory may hold its NAME only."


class _ScriptedStream:
    """One tool call, then text: the provider half of a real session turn."""

    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        turn = len(self.requests)
        name, arguments = self.name, self.arguments

        async def gen():
            if turn == 1:
                yield StreamToolCallDelta(index=0, id="c1", name=name, argument_delta=arguments)
                yield StreamEndEvent(stop_reason="toolUse")
            else:
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")

        return gen()


class _Run:
    """What one session turn left behind, read from the session's OWN surfaces."""

    def __init__(self, session: Session, events: list[Any]) -> None:
        self.session = session
        self.events = events

    @property
    def starts(self) -> list[Any]:
        return [event for event in self.events if event.type == "tool_execution_start"]

    @property
    def end(self) -> Any:
        """The ``tool_execution_end`` for the call, which a DISPATCHED call always has.

        A call the plan gate refused has neither a start nor an end event: the loop
        parks a synthetic result row and emits nothing, which is the zero-start
        measurement this file is about. Read that case through :attr:`row` instead.
        """
        ends = [
            event
            for event in self.events
            if event.type == "tool_execution_end" and event.tool_call_id == "c1"
        ]
        assert len(ends) == 1, [event.type for event in self.events]
        return ends[0]

    @property
    def row(self) -> dict[str, Any]:
        """The call's durable tool row: the surface BOTH outcomes leave behind."""
        rows = [
            entry.payload
            for entry in self.entries
            if entry.payload.get("role") == "tool" and entry.payload.get("tool_call_id") == "c1"
        ]
        assert len(rows) == 1, [entry.payload.get("role") for entry in self.entries]
        return rows[0]

    @property
    def result_text(self) -> str:
        """The call's answer, from the live event when it ran and the row otherwise.

        The transcript stores CONTENT BLOCKS as plain dicts (they are serialized for the
        journal), while a live event carries the same blocks as objects — both spellings
        of ``text`` are read here so the two surfaces are one string to assert on.
        """
        parts = self.end.result.content if self.starts else self.row.get("content", ())
        pieces = []
        for part in parts:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if isinstance(text, str):
                pieces.append(text)
        return "".join(pieces)

    @property
    def synthetic(self) -> bool:
        """Whether the loop INVENTED this result instead of a tool producing it.

        ``__synthetic`` is set by ``AgentLoop._synthetic_result`` alone, so it is the
        marker that separates "the plan gate refused this" from "the tool answered".
        (A tool's own result carries ``details`` that need not exist at all.)
        """
        details = (self.row.get("provider_payload") or {}).get("details") or {}
        return bool(details.get("__synthetic"))

    @property
    def entries(self) -> list[Any]:
        return list(self.session.transcript.entries())

    def serialized_surfaces(self) -> str:
        """Every session-level surface a value could reach, as one string."""
        return json.dumps(
            {
                "events": [event.model_dump(mode="json") for event in self.events],
                "transcript": [entry.payload for entry in self.entries],
            },
            default=str,
        )


async def _run_one_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, args: dict[str, Any]
) -> _Run:
    """One call through a real Session, real tools and an isolated memory store.

    The store is seeded through the real ``write`` tool first, so the runs below answer
    from a populated bank rather than from "nothing is there" — the prose read in
    :func:`test_the_ordinary_memory_paths_are_untouched` is what proves that wiring.

    Each (name, args) pair gets its OWN store and transcript: several of the tests below
    drive more than one call in a row, and a shared root would let a second session read
    the first one's rows (two rows for one call id) and answer from two seeded facts.
    """
    key = hashlib.sha256(f"{name}|{json.dumps(args, sort_keys=True, default=str)}".encode())
    cfg = tmp_path / key.hexdigest()[:12] / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))
    context = ToolContext(
        session_id="plan-gate",
        memory_config_dir=str(cfg),
        session_dir=str(cfg / "sessions" / "plan-gate"),
    )
    seeded = await execute_memory(
        "seed",
        "write",
        {"path": "memory://global", "content": json.dumps(PROSE_FACT)},
        context,
    )
    assert not seeded.is_error, seeded.content

    session = Session(
        model=MODEL,
        stream_fn=_ScriptedStream(name, json.dumps(args)),
        tools=[build_memory_tool(), build_read_tool(), build_write_tool()],
        transcript=Transcript(cfg / "sessions" / "plan-gate"),
        system_blocks_provider=lambda: ["stable"],
        session_id="plan-gate",
        yolo=True,
        cwd=str(tmp_path),
    )
    events: list[Any] = []
    session.subscribe(events.append)
    await session.prompt("go")
    try:
        return _Run(session, events)
    finally:
        await session.dispose()


# -- the two exempt positions, end to end ------------------------------------


@pytest.mark.asyncio
async def test_a_colliding_reference_query_reaches_the_executor(tmp_path, monkeypatch):
    """``read ?q=<colliding id>``: zero starts before, one after.

    RED, measured over the base tree with this same file::

        tool_execution_start: 0
        result: Sensitive value refused. Use lop secret; memory may hold its NAME only.
        details: {"__fault": "invalid_arguments", "__synthetic": true}

    The expected result is ``[]``, and that is a real answer rather than an empty
    assertion: the store's FTS index holds the id's tokens only if a fact containing it
    was ever written, and a fact cannot even be STORED carrying this value — the same
    guard refuses it on the write path (``test_the_exempt_position_is_the_only_one``
    pins the write case). ``#2524``'s tool-level corpus asserts the same ``[]``.
    """
    run = await _run_one_call(
        tmp_path, monkeypatch, "read", {"path": f"memory://global?q={DEVICE_ID}"}
    )
    assert len(run.starts) == 1, run.result_text
    assert not run.end.result.is_error, run.result_text
    assert not run.synthetic, run.result_text
    assert json.loads(run.result_text) == []


@pytest.mark.asyncio
async def test_a_colliding_reference_recall_reaches_the_executor(tmp_path, monkeypatch):
    """``memory(op=recall, text=<colliding id>)``: zero starts before, one after.

    RED, measured over the base tree with this same file::

        tool_execution_start: 0
        result: Sensitive value refused. Use lop secret; memory may hold its NAME only.
        details: {"__fault": "invalid_arguments", "__synthetic": true}

    ``#2524``'s tool change is the other half of THIS path (the tool guards its own
    argument dict before it validates), so what the plan gate owes here is the dispatch
    itself: the call must reach the tool instead of a synthetic refusal. This assertion
    is therefore on the seam — one start, no synthetic row — and it holds with and
    without ``#2524`` stacked; the run that carries the store's own answer for the
    recall is recorded in the pull request over the stacked tree.
    """
    run = await _run_one_call(tmp_path, monkeypatch, "memory", {"op": "recall", "text": DEVICE_ID})
    assert len(run.starts) == 1, run.result_text
    assert not run.synthetic, run.result_text


# -- session-level negative controls -----------------------------------------


@pytest.mark.asyncio
async def test_a_secret_in_the_same_position_is_still_refused(tmp_path, monkeypatch):
    """The mask blanks a reference and nothing else, so a VALUE keeps its refusal."""
    for name, args in (
        ("read", {"path": f"memory://global?q={CARD}"}),
        ("read", {"path": f"memory://global?q={IBAN}"}),
        ("read", {"path": "memory://global?q=123-45-6789"}),
        ("memory", {"op": "recall", "text": CARD}),
        ("memory", {"op": "recall", "text": IBAN}),
        ("memory", {"op": "recall", "text": f"{DEVICE_ID} extra"}),
    ):
        run = await _run_one_call(tmp_path, monkeypatch, name, args)
        assert len(run.starts) == 0, (name, args, run.result_text)
        assert run.synthetic and "Sensitive value refused" in run.result_text, (
            name,
            args,
            run.result_text,
        )
        assert run.row["tool_name"] == name


@pytest.mark.asyncio
async def test_a_secret_in_a_sibling_parameter_is_still_refused(tmp_path, monkeypatch):
    """The exemption is two POSITIONS, never the call: a sibling still refuses it."""
    run = await _run_one_call(
        tmp_path,
        monkeypatch,
        "read",
        {"path": f"memory://global?q={DEVICE_ID}", "raw": IBAN},
    )
    assert len(run.starts) == 0, run.result_text
    assert run.synthetic and "Sensitive value refused" in run.result_text


@pytest.mark.asyncio
async def test_a_masked_call_that_is_still_sensitive_is_still_refused(tmp_path, monkeypatch):
    """Masking one position must not launder another: ``tokens=`` is refused on the base too.

    The path-like-token arm (`memory.py`) refuses a whitespace-free path carrying
    ``tokens=`` for ANY query, reference or prose — so this call is refused with the
    masked value removed exactly as it is with the value present, which is what
    "byte-identically to today" has to mean for a call the exemption cannot rescue.
    (The tool layer's own version of this refusal says "Sensitive path refused"; the seam
    answers first and says the same thing it says for any other sensitive value.)
    """
    for query in (DEVICE_ID, "stage2"):
        run = await _run_one_call(
            tmp_path,
            monkeypatch,
            "read",
            {"path": f"memory://global?q={query}&tokens=50"},
        )
        # Byte-identical to today: the SAME plan-gate text, still zero starts. The masked
        # copy removed the admitted value and the path-like-token arm refused the rest.
        assert len(run.starts) == 0, (query, run.result_text)
        assert run.synthetic and run.result_text == PLAN_GATE_REFUSAL, run.result_text


@pytest.mark.asyncio
async def test_the_exempt_position_is_the_only_one(tmp_path, monkeypatch):
    """Other openings, and a malformed read path, keep today's plan-gate refusal."""
    run = await _run_one_call(
        tmp_path,
        monkeypatch,
        "write",
        {"path": f"memory://global?q={DEVICE_ID}", "content": "{}"},
    )
    assert len(run.starts) == 0, run.result_text
    assert run.synthetic and "Sensitive value refused" in run.result_text

    # A trailing "&" makes ``parse_url`` refuse the path with strict parsing, so the q
    # value is never classified as a reference and the guard keeps the refusal it has.
    run = await _run_one_call(
        tmp_path, monkeypatch, "read", {"path": f"memory://global?q={DEVICE_ID}&"}
    )
    assert len(run.starts) == 0, run.result_text
    assert "Sensitive value refused" in run.result_text


@pytest.mark.asyncio
async def test_the_ordinary_memory_paths_are_untouched(tmp_path, monkeypatch):
    """The controls that prove the runs above are answering from a live store.

    A prose query returns the seeded fact, a lookup by a NON-colliding reference (both
    admitted shapes) is answered by the store, and the same prose query is what the
    base tree answers too — the seam moved nothing here.
    """
    prose = await _run_one_call(
        tmp_path, monkeypatch, "read", {"path": "memory://global?q=staging host"}
    )
    assert len(prose.starts) == 1 and not prose.end.result.is_error, prose.result_text
    hits = json.loads(prose.result_text)
    assert [hit["body"] for hit in hits] == [PROSE_FACT["body"]]

    for reference in (PLAIN_DEVICE_ID, ENTITY_ID):
        run = await _run_one_call(
            tmp_path, monkeypatch, "read", {"path": f"memory://global?q={reference}"}
        )
        assert len(run.starts) == 1 and not run.end.result.is_error, (reference, run.result_text)
        assert json.loads(run.result_text) == []

    recall = await _run_one_call(
        tmp_path, monkeypatch, "memory", {"op": "recall", "text": "staging host"}
    )
    assert len(recall.starts) == 1 and not recall.end.result.is_error, recall.result_text


# -- condition 4: no surface newly shows an exempt value ---------------------


@pytest.mark.asyncio
async def test_no_session_surface_shows_the_exempt_value(tmp_path, monkeypatch):
    """Nothing the session records newly carries an exempt value.

    The retained copies are computed by ``retained_arguments`` from the caller's
    ORIGINAL arguments, so the exempt call is redacted on history/events exactly as it
    was when it never ran: the invocation is the redacted row, and the store's answer
    for an id lookup is ``[]``. The seeded fact's own text is the only body in play.
    """
    run = await _run_one_call(
        tmp_path, monkeypatch, "read", {"path": f"memory://global?q={DEVICE_ID}"}
    )
    surfaces = run.serialized_surfaces()
    assert DEVICE_ID not in surfaces, surfaces
    assert "memory_arguments_redacted" in surfaces, surfaces


# -- condition 3: retention is byte-identical --------------------------------


#: The corpus, expanded for this slice with reference-shaped arguments in every shape
#: `#2524`'s corpus thinking names: id-shaped and secret-shaped, the read/recall/write/
#: edit tool names, with and without a raw JSON carrier, plus non-memory controls.
#: ``admits`` marks the ONLY rows this slice is allowed to change — the plan gate's
#: verdict must flip on them, and their RETENTION verdict must stay ``True``.
#:
#: The two positions are refused today for DIFFERENT reasons, which is why the colliding
#: and non-colliding ids both appear: a read's ``path`` is guarded with the ``path`` value
#: as an ordinary string, so only a chance shape collision (``DEVICE_ID``, IBAN mod-97)
#: refuses it, while a recall's ``text`` is one of the free-text fields
#: ``guard_encoded`` inspects with ``opaque=True``, so the long-hex rule refuses EVERY
#: ``d_`` + 32 hex id there — ``DEVICE_ID`` and ``PLAIN_DEVICE_ID`` alike.
_RETENTION_CORPUS: list[tuple[str, Any, str | None, bool]] = [
    # the two exempt positions, admitted and refused
    ("read", {"path": f"memory://global?q={DEVICE_ID}"}, None, True),
    (
        "read",
        {"path": f"memory://global?q={DEVICE_ID}"},
        json.dumps({"path": f"memory://global?q={DEVICE_ID}"}),
        True,
    ),
    ("read", {"path": f"memory://global?q={DEVICE_ID}&limit=5"}, None, True),
    ("read", {"path": f"memory://global?q={ENTITY_ID}"}, None, False),
    ("read", {"path": f"memory://global?q={PLAIN_DEVICE_ID}"}, None, False),
    ("memory", {"op": "recall", "text": DEVICE_ID}, None, True),
    (
        "memory",
        {"op": "recall", "text": DEVICE_ID},
        json.dumps({"op": "recall", "text": DEVICE_ID}),
        True,
    ),
    ("memory", {"op": "recall", "text": PLAIN_DEVICE_ID}, None, True),
    ("memory", {"op": "recall", "text": ENTITY_ID, "i": "Looking up the id"}, None, False),
    ("memory", {"op": "recall", "text": f"{DEVICE_ID} extra"}, None, False),
    ("memory", {"op": "store", "text": DEVICE_ID}, None, False),
    (
        "memory",
        {"op": "store", "text": DEVICE_ID},
        json.dumps({"op": "store", "text": DEVICE_ID}),
        False,
    ),
    # secret-shaped values, the same positions and the fields around them
    ("read", {"path": f"memory://global?q={CARD}"}, None, False),
    ("read", {"path": f"memory://global?q={IBAN}"}, None, False),
    ("read", {"path": "memory://global?q=123-45-6789"}, None, False),
    ("read", {"path": f"memory://global?q={DEVICE_ID}", "raw": IBAN}, None, False),
    ("read", {"path": f"memory://global?q={DEVICE_ID}&tokens=50"}, None, False),
    ("memory", {"op": "recall", "text": CARD}, json.dumps({"op": "recall", "text": CARD}), False),
    ("write", {"path": f"memory://global?q={DEVICE_ID}", "content": "{}"}, None, False),
    ("write", {"path": "memory://global?q=staging host", "content": "{}"}, None, False),
    (
        "write",
        {"path": "memory://session", "content": json.dumps({"title": "T", "body": f"IBAN {IBAN}"})},
        None,
        False,
    ),
    ("edit", {"path": "memory://session", "old_text": DEVICE_ID, "new_text": "x"}, None, False),
    (
        "edit",
        {
            "path": "memory://session/x",
            "edits": [{"old_text": "a", "new_text": DEVICE_ID}],
            "replace_all": True,
        },
        None,
        False,
    ),
    # malformed and unparseable carriers, and non-memory controls
    ("read", {"path": f"memory://global?q={DEVICE_ID}&"}, None, False),
    (
        "read",
        {"path": f"memory://global?q={DEVICE_ID}"},
        '{"path": "memory://global?q=' + DEVICE_ID,
        False,
    ),
    ("read", {"path": f"memory://global?q={DEVICE_ID}"}, "not json at all", False),
    ("memory", {"op": "recall", "text": DEVICE_ID, "unknown": "x"}, None, False),
    ("memory", {"op": "recall"}, None, False),
    ("memory", {"op": "recall", "text": 7}, None, False),
    ("memory", {"op": "invalid", "text": DEVICE_ID}, None, False),
    ("read", {"path": "notes.txt"}, None, False),
    ("write", {"path": "notes.txt", "content": CARD}, None, False),
    ("bash", {"command": "echo hi", "text": DEVICE_ID}, None, False),
    ("grep", {"pattern": DEVICE_ID, "path": "."}, None, False),
]

#: SHA-256 over ``retained_arguments``' outputs for every corpus row, computed by
#: executing the corpus against the base tree and against this one. They agree — this is
#: that digest, and the corpus runs on both sides from the same file, so a change to what
#: retention RETAINS (rather than to which copy it is asked about) moves this pin.
_RETENTION_DIGEST = "2d8b7e26cd782a98b44f9b391956bbad02ff73d0676025b68728d3d2bf75f25d"


def _retention_rows() -> list[list[Any]]:
    rows = []
    for name, args, raw, _admits in _RETENTION_CORPUS:
        kept, kept_raw, masked = retained_arguments(name, args, raw)
        rows.append([name, json.dumps(kept, sort_keys=True, default=str), kept_raw, masked])
    return rows


def test_retention_outputs_are_byte_identical_over_ref_shaped_arguments() -> None:
    """Retention's own answers, unchanged: the same corpus digest as the base tree."""
    payload = json.dumps(_retention_rows(), sort_keys=True, default=str)
    assert hashlib.sha256(payload.encode()).hexdigest() == _RETENTION_DIGEST


def test_the_plan_gate_changes_its_verdict_for_admitted_references_only() -> None:
    """The blast radius, row by row: two positions, and the rest byte-identical.

    For every corpus row: the verdict ``retained_arguments`` produces today (same code,
    the caller's own arguments) must still be ``True`` where a value is retained — the
    exemption must never be a route to retaining a value on history or events — and the
    verdict the plan gate now reaches must differ from it only for an ``admits`` row.
    """
    for name, args, raw, admits in _RETENTION_CORPUS:
        retained = retained_arguments(name, args, raw)[2]
        guarded = retained_arguments(name, *_plan_guard_inputs(name, args, raw))[2]
        if admits:
            assert retained is True, (name, args, "the exempt value must stay retained")
            assert guarded is False, (name, args)
        else:
            assert guarded is retained, (name, args, "unchanged rows must not move")

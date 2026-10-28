"""The three control seams, and the production inbound screen driven through them.

WHAT THIS MODULE IS. Two things, in this order of importance:

1. :class:`ScreenLeg` — the REAL inbound screen. It calls the production
   ``render_prompt`` for the template bytes, the production ``ScreenModelRunner``
   for the dispatch (isolation contract, token cap, no tools), and the production
   ``parse_verdict`` for the grammar. Nothing about screening is reimplemented
   here: a second implementation would only ever measure itself, and the whole
   point of the exercise is to measure the one that ships. What this module adds is
   accounting the eval needs and production does not: the rendered prompt's digest,
   the per-call token count, wall time, and a machine-readable failure code.

2. :class:`Controls` — the seams for the three controls the eval exists to
   qualify (design review r1, M7): the outbound gate (A), the work boundary (B) and
   the scope boundary (C).

WHY THE CONTROLS ARE INTERFACES, NOT IMPLEMENTATIONS. On the branch this harness
was built against, none of the three exists yet: there is no outbound gate, no
bridge-time tool profile, and no org policy. The temptation is to write "just
enough" of each so the harness can show something; that is exactly the mistake the
design rejects, because the harness would then be qualifying code that is not the
product. So each control is a Protocol with a documented ABSENT default, the report
records ``present: false`` for it, and the delta that proves the control works is a
comparison between an absent-control run and a post-fix run of the same dataset.
When C2/C3/S1 land, they pass a real implementation into :class:`Controls` and the
report records what was actually exercised.

HEURISTICS: NOT HERE, AND NOT IN THE GATE. ``escalation.py`` in the design allows
deterministic SHAPES to escalate a decision (force the default model, or force an
ask) but never to decide it. The harness honours the same rule: it may record that
a reply contained a credential SHAPE, and that record can never become the
outcome - only a canary or the gate's own verdict can.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

__all__ = [
    "ControlState",
    "Controls",
    "InboundOutcome",
    "NullOutboundGate",
    "OpenProfile",
    "OutboundGate",
    "OutboundVerdict",
    "ScreenLeg",
    "ToolProfile",
    "Usage",
]

#: The screening version a TEXT-ONLY bridge message is rendered under. Production
#: uses the served status value (v2 today) and forces v3 only when images follow;
#: the harness has no images, so v2 is the faithful default. v3 is still selectable
#: for a future image case, and then the renderer itself insists on a listing line.
DEFAULT_SCREEN_VERSION = "bridge-screen-v2"


@dataclass(frozen=True)
class Usage:
    """Token accounting for one call, from the provider's own usage event."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class InboundOutcome:
    """One inbound screening decision, as the report needs to see it."""

    disposition: str  # deliver | hold | bounce
    categories: tuple[str, ...]
    reason: str
    verdict: str
    template_version: str
    template_sha256: str
    prompt_sha256: str
    model: str = ""
    model_identity: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    #: A closed token, never provider prose: what made this screen unusable when it
    #: was. ``""`` on a parsed verdict.
    failure: str = ""

    @property
    def allowed(self) -> bool:
        return self.disposition == "deliver"

    @property
    def refused(self) -> bool:
        return self.disposition == "bounce"

    @property
    def invalid(self) -> bool:
        """Whether this screen FAILED rather than deciding.

        THE B1 DISTINCTION (review R1). The shipped pipeline maps every local failure
        to a HOLD, which is right for the product - fail closed, quarantine, wake
        nobody. For the eval it is catastrophic: a crashed screen would score as a
        deliberate hold and every refuse case would "pass" without a single message
        being delivered. So the eval reads ``failure`` and calls it invalid, never held.
        ``hold`` with an empty ``failure`` is the model's own verdict and stays a hold.
        """

        return bool(str(self.failure or "").strip())


class ScreenLeg:
    """The production inbound screen, driven from outside a session.

    WHY IT BUILDS ITS OWN STREAM FUNCTION. Production's ``make_screening_runner``
    wraps the dispatch in a budget/breaker guard bound to a real owner and message
    row, because a live connector must charge every physical send. The eval is not
    that surface: it has no owner, no account and no hourly bucket, and inventing
    one would make the eval's numbers depend on a budget table. So the harness
    builds the same ``ScreenModelRunner`` with the same isolation contract and a
    plain authenticated stream, and the difference is stated here rather than
    hidden: what the eval measures is the JUDGEMENT, not the charging.

    The runner itself is production's, so every property that shapes a verdict (one
    user message, no system blocks, no history, ``tools=[]``, ``tool_choice="none"``,
    the 256-token cap, the 45 s deadline, identity checking against the selection)
    is the shipped one.
    """

    def __init__(
        self,
        *,
        model_route: tuple[str, str],
        config_dir: Path,
        settings: Mapping[str, Any] | None = None,
        screen_version: str = DEFAULT_SCREEN_VERSION,
    ) -> None:
        self.model_route = model_route
        self.config_dir = Path(config_dir)
        self.settings = settings
        self.screen_version = screen_version
        self._runner: Any | None = None
        self._selection: Any | None = None
        #: The counting wrapper is held here, not on the runner: the runner is
        #: production's and must not grow eval-only attributes.
        self._stream: Any | None = None

    # -- construction ------------------------------------------------------

    def _build(self) -> Any:
        """Build the runner once, on first use.

        Imported in-function on purpose: ``dataset.py``, ``canary.py`` and
        ``fixture.py`` are standard-library-only so ``--dry-run`` validates the
        dataset with no engine on the import graph. This is the first module that
        reaches it, and only when a call is actually about to happen.
        """

        from local_operator.model.configure import build_model_spec, create_stream_fn
        from local_operator.providers.auth_store import AuthStore
        from local_operator.team_bridge.model_boundary import (
            DefaultSelection,
            ScreenModelRunner,
        )

        provider, model_id = self.model_route
        spec = build_model_spec(provider, model_id)
        resolved = str(getattr(spec, "model_id", "") or model_id)
        selection = DefaultSelection(provider=provider, model_id=resolved, spec=spec)
        auth = AuthStore(config_dir=self.config_dir)
        stream = create_stream_fn(
            auth,
            settings=self.settings,
            session_id="tb-moderation-eval-screen",
        )
        self._selection = selection
        self._stream = _CountingStream(stream)
        self._runner = ScreenModelRunner(self._stream, selection)
        return self._runner

    async def screen(
        self,
        *,
        body: str,
        sender: str,
        organization: str,
        msg_type: str = "question",
    ) -> InboundOutcome:
        """Render, dispatch, parse - the production pipeline's first two thirds.

        The third third (receipt persistence, budget, quarantine) is the
        connector's, and the eval deliberately does not simulate it: a HOLD here
        means the model's own verdict was hold, not that a store refused.
        """

        from local_operator.team_bridge.screening import parse_verdict, render_prompt

        started = time.monotonic()
        try:
            rendered = render_prompt(
                body_text=body,
                sender=sender,
                organization=organization,
                msg_type=msg_type,
                version=self.screen_version,
            )
        except ValueError as error:
            # A renderer refusal is a real finding: an unrenderable message is one
            # the connector would hold. Recorded as a failure, never skipped.
            return InboundOutcome(
                disposition="hold",
                categories=(),
                reason="",
                verdict="",
                template_version=self.screen_version,
                template_sha256="",
                prompt_sha256="",
                latency_s=time.monotonic() - started,
                failure=f"render_refused:{type(error).__name__}",
            )

        prompt = rendered.private_text()
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        try:
            runner = self._runner or self._build()
        except Exception as error:  # noqa: BLE001 - a build failure is a hold, not a crash
            return InboundOutcome(
                disposition="hold",
                categories=(),
                reason="",
                verdict="",
                template_version=self.screen_version,
                template_sha256=rendered.template_sha256,
                prompt_sha256=prompt_digest,
                latency_s=time.monotonic() - started,
                failure=f"runner_build_failed:{type(error).__name__}",
            )

        assert self._selection is not None
        try:
            reply, identity = await runner.acall(prompt, default_model=self._selection.default_model)
        except Exception as error:  # noqa: BLE001 - every provider fault is a HOLD
            return InboundOutcome(
                disposition="hold",
                categories=(),
                reason="",
                verdict="",
                template_version=self.screen_version,
                template_sha256=rendered.template_sha256,
                prompt_sha256=prompt_digest,
                model=self._selection.default_model,
                latency_s=time.monotonic() - started,
                failure=f"provider:{type(error).__name__}",
            )

        elapsed = time.monotonic() - started
        # Usage comes from the counting wrapper, never from the production runner:
        # the runner is the shipped object and the eval does not add fields to it.
        usage = getattr(self._stream, "last_usage", Usage()) or Usage()
        if not reply:
            # The runner returns "" for every identity or parse refusal; production
            # calls those HOLDs (model_identity_unknown / parse_failed). Keep that
            # meaning rather than inventing a fourth disposition.
            return InboundOutcome(
                disposition="hold",
                categories=(),
                reason="",
                verdict="",
                template_version=self.screen_version,
                template_sha256=rendered.template_sha256,
                prompt_sha256=prompt_digest,
                model=self._selection.default_model,
                model_identity=identity,
                usage=usage,
                latency_s=elapsed,
                failure="empty_reply",
            )

        parsed = parse_verdict(reply)
        if parsed is None:
            return InboundOutcome(
                disposition="hold",
                categories=(),
                reason="",
                verdict="",
                template_version=self.screen_version,
                template_sha256=rendered.template_sha256,
                prompt_sha256=prompt_digest,
                model=self._selection.default_model,
                model_identity=identity,
                usage=usage,
                latency_s=elapsed,
                failure="parse_failed",
            )
        disposition, categories, reason, verdict = parsed
        if disposition == "bounce" and not categories:
            # parse_verdict returns a bounce with no known category when the reason
            # named none; production holds those. Same rule here.
            return InboundOutcome(
                disposition="hold",
                categories=(),
                reason=reason,
                verdict=verdict,
                template_version=self.screen_version,
                template_sha256=rendered.template_sha256,
                prompt_sha256=prompt_digest,
                model=self._selection.default_model,
                model_identity=identity,
                usage=usage,
                latency_s=elapsed,
                failure="bounce_without_category",
            )
        return InboundOutcome(
            disposition=disposition,
            categories=categories,
            reason=reason,
            verdict=verdict,
            template_version=self.screen_version,
            template_sha256=rendered.template_sha256,
            prompt_sha256=prompt_digest,
            model=self._selection.default_model,
            model_identity=identity,
            usage=usage,
            latency_s=elapsed,
        )


class _CountingStream:
    """Pass-through wrapper that remembers the last call's usage and model.

    A wrapper rather than a second client: the production runner must receive the
    stream function it would have received anyway, or the eval stops measuring the
    shipped dispatch. This only observes.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_usage = Usage()

    def __call__(self, request: Any, api_key: Any) -> Any:
        return self._stream(request, api_key)

    async def _stream(self, request: Any, api_key: Any) -> Any:
        usage = Usage()
        async for event in self._inner(request, api_key):
            if str(getattr(event, "type", "")) == "usage":
                reported = getattr(event, "usage", None)
                usage = Usage(
                    input_tokens=int(getattr(reported, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(reported, "output_tokens", 0) or 0),
                )
            yield event
        self.last_usage = usage

    async def close(self) -> None:
        closer = getattr(self._inner, "close", None)
        if closer is not None:
            result = closer()
            if hasattr(result, "__await__"):
                await result


# ---------------------------------------------------------------------------
# Control seams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutboundVerdict:
    """What an outbound gate decided. Mirrors design §3.2's grammar."""

    verdict: str  # send | block | ask_owner
    category: str = ""
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.verdict != "send"


class OutboundGate(Protocol):
    """Design §3.2's G2. Implemented in C3, NOT here."""

    async def judge(
        self,
        *,
        reply: str,
        inbound: Sequence[str],
        asker: Mapping[str, Any],
        session_id: str,
    ) -> OutboundVerdict: ...


class NullOutboundGate:
    """The absent control: every reply is sent, and the report says so.

    This is not a stub standing in for a gate - it is the MEASUREMENT of having
    none. Its presence in a control run is what the baseline documents, and its
    replacement by a real :class:`OutboundGate` is what a later delta run proves.
    """

    present = False
    name = "A"
    detail = (
        "no outbound gate on main: team_bridge.tool send/reply/channel_post go "
        "straight to the outbox and the wire"
    )

    async def judge(
        self,
        *,
        reply: str,
        inbound: Sequence[str],
        asker: Mapping[str, Any],
        session_id: str,
    ) -> OutboundVerdict:
        return OutboundVerdict(verdict="send", category="", reason="no gate installed")


class ToolProfile(Protocol):
    """Design §4.2's bridge-turn capability profile. Implemented in C2, NOT here."""

    def allows(self, tool_name: str, arguments: Mapping[str, Any]) -> bool: ...


    def allows(self, tool_name: str, arguments: Mapping[str, Any]) -> bool: ...


@dataclass(frozen=True)
class OpenProfile:
    """The absent control: today's reach, which is not narrowed at all.

    ``present = False`` is the honest record. The profile's ABSENCE is measured,
    not assumed: every tool call the agent makes is recorded with the path-shaped
    arguments it used, and the report says which of them resolved outside every
    declared work root and whether anything refused them. A run under this class is
    therefore a measurement of the gap, and a later run under the real profile is
    the delta that proves the fix.
    """

    present: bool = False
    name: str = "B"
    detail: str = (
        "no bridge-turn capability profile on main: read/glob/grep are read-tier and "
        "never prompt, and bash is read-tier when write-confined"
    )

    def allows(self, tool_name: str, arguments: Mapping[str, Any]) -> bool:
        return True


@dataclass(frozen=True)
class ControlState:
    """One control's state in THIS run, as data the report can print."""

    name: str
    present: bool
    detail: str
    #: Which implementation answered, so a report names the code that was
    #: exercised rather than a boolean nobody can check.
    source: str = ""

    def as_record(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "present": self.present,
            "detail": self.detail,
            "source": self.source,
        }


@dataclass
class Controls:
    """The three controls as one object the runner passes around.

    ``disabled`` exists for the delta proof the design review demanded: a run that
    turns a control OFF must change the result, or the control is not doing
    anything. On the pre-fix branch every control is already off, so the flag is
    inert - which the report states rather than leaving a reader to infer.
    """

    outbound: OutboundGate = field(default_factory=NullOutboundGate)
    profile: Any = field(default_factory=OpenProfile)
    #: Control C's implementation (S1/C1). ``None`` means no org policy exists, so
    #: the in-scope/out-of-scope pair is measured only through the screener's own
    #: verdicts.
    scope: Any = None
    disabled: frozenset[str] = frozenset()

    def is_present(self, name: str) -> bool:
        if name in self.disabled:
            return False
        if name == "A":
            return bool(getattr(self.outbound, "present", False))
        if name == "B":
            return bool(getattr(self.profile, "present", False))
        if name == "C":
            return self.scope is not None and bool(getattr(self.scope, "present", False))
        return False

    def outbound_gate(self) -> OutboundGate | None:
        """The gate to run, or ``None`` when control A is absent or disabled."""

        return self.outbound if self.is_present("A") else None

    def states(self) -> tuple[ControlState, ...]:
        return (
            ControlState(
                name="A",
                present=self.is_present("A"),
                detail=str(getattr(self.outbound, "detail", "")),
                source=type(self.outbound).__name__,
            ),
            ControlState(
                name="B",
                present=self.is_present("B"),
                detail=str(getattr(self.profile, "detail", "")),
                source=type(self.profile).__name__,
            ),
            ControlState(
                name="C",
                present=self.is_present("C"),
                detail=(
                    "no org policy input: the screener has out_of_scope_work but no "
                    "in-scope entity list (design R4)"
                    if self.scope is None
                    else str(getattr(self.scope, "detail", ""))
                ),
                source="None" if self.scope is None else type(self.scope).__name__,
            ),
        )

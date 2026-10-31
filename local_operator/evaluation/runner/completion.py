"""The runner's one-shot completion challenge, and the rule that scopes it.

WHY THIS EXISTS, and why it is not a claim-grounding check. The paid campaign
of 2026-09-25 scored 0 of 13 binary while the model's own finishes described
the screen it was looking at. Reading each finish against the frame artifact it
was bound to (``~/worktrees/osworld/INFRA.md`` §"CORRECTED (2026-09-25 PM)")
showed 3 of 3 claims were TRUE about the observation: the model looked, formed
a true claim, and then judged that screen to satisfy the task. Two mechanisms
that follow from the older, wrong diagnosis are unreachable or empty on this
corpus and are deliberately NOT built here:

* refusing a finish with no observation after the last mutation -- for 16 of 16
  ``status="done"`` finishes the finish's observation IS the newest one, bound
  structurally by ``PendingObservationToken``, ``ActionBatch.validate_for`` and
  ``parse_decision``'s validation pair, so the rule could never fire;
* comparing the model's notes against its claim -- the ``public_observations``
  agree with the ``reason`` in substance, so there is no inconsistency to catch.

What is left is the missing COMPARISON, so the gate asks for exactly that, once:
the claim is quoted back as a claim, the task is restated as it was stated, the
current observation's frames are re-attached, and the model is asked whether
THIS observation shows each thing the task requires -- answering either with the
same finish, unchanged, or with an action batch that closes the gap.

TWO PROPERTIES ARE THE WHOLE SAFETY ARGUMENT, and both are structural:

* the challenge count is BOUNDED and separate from ``max_decision_retries``
  (the precedent is ``MAX_EMPTY_TRUNCATION_RETRIES`` in ``episode.py``: spending
  one allowance must not eat another). At the default budget of 1 the SECOND
  ``done`` declaration is always accepted, so the gate can never drive an
  episode into ``_ModelFailure`` or an unscored seal. Worst case is one wasted
  cycle plus a warning-shaped ``error`` event.
* the gate lives in the RUNNER, not in the model client. A client that cannot
  re-present the end state cannot enforce the gate at all, which the driver
  reports loudly rather than degrading in silence; the client owns only the
  words, and returns them so the runner can publish exactly what was shown.

Nothing here names a benchmark, an application or a task family: the gate fires
on the protocol's own terminal action, and its content is the model's own words
plus the task text it was handed at reset. A task with nothing left to observe
-- terminal-only, or already correct -- is explicitly not blocked, because
re-declaring the same finish is a permitted reply.

THE SECOND RULE: A PREMATURE TERMINAL (2026-10-30). The ``done``-only scope
above misses a different failure, seen in two sealed OSWorld records: a
``failed``/``infeasible`` finish declared before the model had executed a
single action. In ``osworld-21/runs/t24w1-t004-*`` call #1 was
``finish(status="failed", reason="Need to inspect slides first")`` and the
model's own closing text called it "ended the episode prematurely by mistake"
(scored 0 after 1 step); in ``runs/s20a-t001-20261028-071043`` call #1 was
``[wait, finish(infeasible)]`` (refused at admission there, as a finish must be
alone) and the same run then completed the task. A non-done terminal with NO
executed action is not a conclusion drawn from the environment -- nothing has
been tried -- so it gets the same one-shot treatment, with three differences
that keep it honest:

* it is keyed ONLY on the protocol's terminal action and the episode's count of
  executed (non-terminal) batches: no task, application or benchmark name, and
  the ``reason`` text is never read;
* its budget is a SEPARATE, fixed count of ONE (:data:`PREMATURE_TERMINAL_CHALLENGES`).
  Sharing ``completion_challenges`` would let a premature challenge silently
  disarm the ``done`` gate for the rest of the episode, and a knob would buy
  nothing: once the model has re-declared, the episode ends, so the condition
  cannot recur;
* its wording is NEUTRAL (``provider_client.build_completion_challenge``): it
  restates the task, says nothing has been executed yet, and offers the same
  finish, unchanged, as a first-class reply. It never says the task is
  feasible, because some are deliberately not -- a genuinely infeasible task
  still ends ``infeasible``, one turn later, and is scored on that.

Default ON is safe for the same structural reason as the ``done`` gate: the
second declaration is always accepted, so the worst case is one extra model
cycle; the episode can never be driven into ``_ModelFailure`` or an unscored
seal by it. ``EpisodeConfig.premature_terminal_gate`` (driver flag
``--no-premature-terminal-gate``) is the control arm.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from local_operator.evaluation.protocol import ActionBatch, FinishAction
from local_operator.evaluation.runner.model import CompletionChallenger, EpisodeTurn

#: Which rule a challenge was raised under. The runner maps it to the ``error``
#: event's ``diagnostic_code`` (:data:`CHALLENGE_DIAGNOSTIC_CODES`) and the
#: session arm records it as the ``trigger``, so a sealed record says which
#: rule fired.
ChallengeKind = Literal["completion", "premature-terminal"]

#: The premature-terminal gate's fixed budget (see the module docstring for why
#: it is separate from ``completion_challenges`` and not configurable).
PREMATURE_TERMINAL_CHALLENGES = 1

#: The runner's ``error`` event code per rule. ``completion-challenged`` is the
#: historical code and stays byte-identical so existing audits keep counting.
CHALLENGE_DIAGNOSTIC_CODES: dict[str, str] = {
    "completion": "completion-challenged",
    "premature-terminal": "premature-terminal-challenged",
}


def challenge_kind(
    claim: FinishAction,
    *,
    executed_steps: int,
    completion_open: bool,
    premature_open: bool,
) -> ChallengeKind | None:
    """The ONE rule both arms apply to a terminal claim.

    ``completion_open``/``premature_open`` are "this gate is enabled and its
    budget is not spent" -- the counters stay with each arm (the runner's
    :class:`CompletionGate`, the session ``ActionBridge``), the decision lives
    here so the two channels cannot drift. ``executed_steps`` is the count of
    non-terminal batches the episode has executed (an ``ask_user`` exchange
    counts: the model acted and got an answer).
    """

    if claim.status == "done":
        return "completion" if completion_open else None
    if executed_steps == 0 and premature_open:
        return "premature-terminal"
    return None


def finish_claim(batch: ActionBatch) -> FinishAction:
    """The batch's finish action.

    A terminal action is the ONLY action in its batch (``ActionBatch``'s
    ``_bind_and_isolate_actions``), so there is exactly one to find and no
    question about which claim was declared. Raising rather than returning
    ``None`` keeps the caller free of a branch that could never be taken for a
    batch the caller has already classified as a finish.
    """

    for action in batch.actions:
        if isinstance(action, FinishAction):
            return action
    raise ValueError("batch carries no finish action")


class CompletionGate:
    """One episode's terminal-claim gates: whether to challenge, and how often.

    The counters are per EPISODE and live here rather than on the model client,
    which is rebuilt per episode anyway: the bound is a property of the
    episode's protocol, not of the client that happens to answer.
    """

    def __init__(
        self,
        *,
        client: Any,
        enabled: bool,
        challenges: int,
        premature: bool = True,
    ) -> None:
        self._client = client
        self._enabled = enabled
        self._budget = challenges
        self._premature = premature
        self._fired = 0
        self._premature_fired = 0

    @property
    def fired(self) -> int:
        """How many ``done`` challenges this episode has made."""

        return self._fired

    @property
    def premature_fired(self) -> int:
        """How many premature-terminal challenges this episode has made."""

        return self._premature_fired

    def should_challenge(self, claim: FinishAction, *, executed_steps: int) -> ChallengeKind | None:
        """Which rule, if any, asks the model to re-check this terminal claim.

        The conditions, each load-bearing:

        * the rule's gate is enabled (``EpisodeConfig.completion_gate`` /
          ``premature_terminal_gate``), so a control arm is a config flip;
        * the rule's budget is not spent, which is what bounds the exchange;
        * the claim matches the rule (:func:`challenge_kind`): ``done`` for the
          completion gate; ``failed``/``infeasible`` with no executed action for
          the premature gate. A non-done finish AFTER an action is the model
          reporting that it could NOT finish, which is never challenged;
        * the client can actually re-present the end state -- it satisfies
          :class:`~local_operator.evaluation.runner.model.CompletionChallenger`.
          A client that cannot must not be handed a gate whose only effect would
          be a recorded challenge the model never saw.
        """

        kind = challenge_kind(
            claim,
            executed_steps=executed_steps,
            completion_open=self._enabled and self._fired < self._budget,
            premature_open=self._premature
            and self._premature_fired < PREMATURE_TERMINAL_CHALLENGES,
        )
        if kind is None or not isinstance(self._client, CompletionChallenger):
            return None
        return kind

    async def challenge(
        self, batch: ActionBatch, turns: Sequence[EpisodeTurn], kind: ChallengeKind
    ) -> str:
        """Append the challenge and return the text the model was shown.

        The task instruction is the RESET observation's text (``turns[0]``),
        which is the task as the adapter published it -- including the task
        files whose literal is interpolated at runtime, which is the whole point
        of reading the observation rather than a task id. The WORDS for each
        rule come from the client (``build_completion_challenge`` keys them on
        the claim's status), so the protocol method is unchanged.

        The counter advances only after the client has accepted the turn: a
        client that raises did not show the model a challenge, and charging it
        here would spend the allowance on a turn that never happened.
        """

        observation = turns[-1].observation
        instruction = turns[0].observation.text or ""
        client = self._client
        if not isinstance(client, CompletionChallenger):
            # ``should_challenge`` is the only caller and it refuses exactly this
            # case, so reaching here is a caller bug -- and it is raised rather
            # than swallowed because the alternative is a challenge counted as
            # fired that the model never saw.
            raise TypeError("model client cannot re-present the end state")
        text = await client.challenge_completion(
            observation,
            tuple(turns),
            batch=batch,
            instruction=instruction,
        )
        if kind == "premature-terminal":
            self._premature_fired += 1
        else:
            self._fired += 1
        return text

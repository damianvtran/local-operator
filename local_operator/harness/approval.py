"""The tool-approval gate: its type, and the ONE place its arity is resolved.

A host installs a gate to answer "may this tool run?" for write- and
exec-tier calls. The gate has two accepted shapes and both must keep working:

- ``(tool_name, description)`` — the original, and what every host in the tree
  wrote against (the CLI's stdin prompt, the TUI's approval card, the FastAPI
  facade, and a long tail of test fakes);
- ``(tool_name, description, job_id)`` — the same question plus the provenance
  a host needs to tell a foreground ask apart from a background one.

Widening the callback in place rather than versioning it is what keeps every
existing host assignable and type-clean: :data:`ApprovalGate` is a union of the
two shapes, so a two-argument handler satisfies it unchanged.

Arity is resolved HERE, once, and never at a call site. There are two call
sites — the loop's tier gate and the builtin self-gate — they must agree or the
same host is asked two different ways depending on which tool is running, and
the failure mode of getting it wrong is a ``TypeError`` that both call sites
catch and turn into a silent denial. Resolution is by signature inspection and
NOT by calling with three arguments and retrying on ``TypeError``: a
``TypeError`` raised from inside the host's own body is indistinguishable from
an arity mismatch, and the retry would then invoke a handler that had already
mounted a prompt or written a log line.

The rule is BY NAME — a parameter literally called ``job_id`` — and not by
counting parameters, because counting is wrong in both directions and both
wrong answers are silent:

- ``(tool_name, description, *, job_id=None)`` is the natural way to add
  provenance to an existing handler without breaking its callers, and it has
  two POSITIONAL parameters. Counting calls it with two and the host's
  ``job_id`` stays ``None`` forever, so a host trying to scope a denial to a
  background job simply never can.
- ``(tool_name, description, timeout=30)`` is a pre-existing handler whose
  third parameter means something else. Counting hands it a job id as its
  timeout.
- ``async def wrapper(*args)`` forwarding to a two-argument gate is the shape
  this codebase actually writes. Counting ``*args`` as wide calls the wrapper
  with three, the INNER gate raises ``TypeError``, and both call sites turn
  that into exactly the invisible denial this module exists to prevent.

So a differently-named third parameter, ``*args``, and an unreadable signature
all degrade to the two-argument shape — the one that always works. A host that
wants provenance names it ``job_id`` and gets it.

This module ALSO owns the gate's authority model, because the two predicates
that decide who may move a running gate have to live together or they drift:
:func:`loosening_is_authorised` answers "may this *settings write* loosen the
gate?" (issue #1282, the config-file path) and :func:`transition_authority`
answers "is this *control-plane request* one that removes the gate?" (issue
#1310, the socket path). Both exist for one invariant — the constrained subject
must not be able to mint the authority that removes its own approval
requirement — and the operator capability the second one demands is minted,
handed over and tracked here too, so there is one place to read the whole rule.
See ``docs/design/approval-authority.md``.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import logging
import os
import secrets
import socket
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Union, cast

logger = logging.getLogger(__name__)

#: Transcript row written when an approval gate expires with nobody attached.
#:
#: Lives here, beside the gate concept itself, because three layers need to
#: agree on it and none of them may import the others: the runtime WRITES it
#: (``session/runtime/serving.py``), the session RENDERS it for the model
#: (``session/session.py``), and the TUI renders it for the user. A copy in
#: any one of them would be a fourth place for the string to drift.
#:
#: The distinction it preserves is that an expiry is not a decision: without
#: the row, the next turn reads a plain denial and re-plans around a choice
#: nobody made.
GATE_TIMEOUT_CUSTOM_TYPE = "gate_timed_out_unattended"

#: The two accepted host gate shapes. Declared as a union rather than a
#: Protocol with an optional parameter because Python cannot express "callable
#: of two OR three arguments" in one signature, and a Protocol declaring the
#: third would make every existing two-argument host a type error.
ApprovalGate = Union[
    Callable[[str, str], Awaitable[bool]],
    Callable[[str, str, "str | None"], Awaitable[bool]],
]

#: How (and whether) a gate takes the job id. ``keyword`` covers the ordinary
#: third parameter as well as the keyword-only one: passing by keyword is
#: unambiguous for either, and it is the only spelling that works for the
#: keyword-only case.
_JobIdStyle = Literal["none", "positional", "keyword"]


def _job_id_style(gate: ApprovalGate) -> _JobIdStyle:
    """Find this gate's ``job_id`` parameter, or report that it has none.

    Positional-only (``..., job_id, /``) has to be passed positionally; every
    other binding takes the keyword. A gate whose signature cannot be read at
    all — a C callable, some ``functools.partial`` shapes — is the
    two-argument form, because guessing wide there raises inside the gate.
    """
    try:
        parameters = inspect.signature(gate).parameters.values()
    except (TypeError, ValueError):
        return "none"
    for parameter in parameters:
        if parameter.name != "job_id":
            continue
        if parameter.kind is parameter.POSITIONAL_ONLY:
            return "positional"
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
            return "keyword"
    return "none"


async def ask_approval(
    gate: ApprovalGate,
    tool_name: str,
    description: str,
    job_id: str | None = None,
) -> bool:
    """Put one approval question to the host. ``True`` means proceed."""
    style = _job_id_style(gate)
    if style == "positional":
        wide = cast("Callable[[str, str, str | None], Awaitable[bool]]", gate)
        return bool(await wide(tool_name, description, job_id))
    if style == "keyword":
        by_keyword = cast("Callable[..., Awaitable[bool]]", gate)
        return bool(await by_keyword(tool_name, description, job_id=job_id))
    narrow = cast("Callable[[str, str], Awaitable[bool]]", gate)
    return bool(await narrow(tool_name, description))


def loosening_is_authorised(*, source: str, gate_is_here: bool) -> bool:
    """Whether a write may LOOSEN a live approval gate (``ask`` -> ``auto``).

    The gate's own boundary, and the whole of it: a loosening is an operator
    action only when it is a write this process made through the operator's own
    settings facade (``settings_io`` -> ``config_watch.notify_local``, which
    delivers ``source="local"``) *in the process that holds the gate*.
    Everything else is unattributed from here and may only tighten:

    * a model tool's own file write (the party being gated is not the authority
      that may lower its own gate -- the reason this predicate exists);
    * an editor, a second pane, or any other process's edit;
    * the settings API or ``lop config edit`` in ANOTHER process, which is a
      genuine operator action but happened where this gate is not, so this
      process cannot tell it apart from the first case;
    * the embedded TUI's ``/settings`` page while an attached runtime owns the
      gate (``gate_is_here=False``): the file write is attributed in the app's
      own process, but the engine consults the runtime's flag.

    ``source`` is ``ConfigChange.source`` and ``gate_is_here`` is whether the
    caller is the process whose flag the engine reads -- ``True`` for the
    runtime/daemon host, ``not self._gate_is_owned_elsewhere()`` for the TUI.
    Spelled ``source == "local"`` rather than "not disk" so widening what
    counts as ``local`` in ``config_watch`` widens who may loosen the gate;
    that is the direction the literal deliberately makes visible at the
    call site rather than absorbing here.

    Tightening is NOT this predicate's business: ``auto`` -> ``ask`` and every
    other hardening path stay live and unconditional. Nor is ``--yolo``, which
    is an explicit pin on the run rather than a transition of the gate.
    """
    return source == "local" and gate_is_here


#: The ONE sentence both hosts print when a write tried to loosen a live gate
#: without being authorised to (see :func:`loosening_is_authorised`).
#:
#: Lives here for the same reason :data:`GATE_TIMEOUT_CUSTOM_TYPE` does: two
#: layers emit it and neither may import the other — the runtime
#: (``session/runtime/serving.py``) as a ``NoticeEvent``, the embedded TUI
#: (``tui/app.py``) into its own transcript — and a second copy is a second
#: chance for one surface to describe the rule differently from the other. The
#: wording is also load-bearing rather than decorative: it names the RULE ("a
#: write from outside this session") rather than the author, because the
#: emitting process cannot know who wrote the file, and in the attached-pane
#: case the person reading it is the one who just clicked the row (design round
#: 1, D3). ``/approvals auto`` is the route that does loosen the gate.
LOOSENING_REFUSED_NOTICE = (
    "keeping tool approvals: ask — config.yml now says auto, but a write from outside "
    "this session cannot loosen it; type /approvals auto in the terminal or app window "
    "that started this session"
)


#: The sibling sentence, and the reason it is a constant too (agent review round
#: 2, n2's family): the same event — the file says ``auto``, this session keeps
#: its typed ``ask`` — is emitted by the runtime and by the embedded pane, and
#: round 1 left the two copies inline in each host. Two copies of one sentence is
#: the drift U5 found for the receipts, one surface over.
LOOSENING_KEPT_BY_ASK_NOTICE = (
    "keeping tool approvals: ask — set with /approvals in this session; config.yml "
    "now says auto, /approvals auto adopts it"
)


# ---------------------------------------------------------------------------
# The control plane's operator capability (issue #1310)
# ---------------------------------------------------------------------------
#
# ``loosening_is_authorised`` closes the CONFIG-FILE path: a settings write is
# attributed to the operator or it may only tighten. It does not touch the
# CONTROL-PLANE path, and that path is a different defect class: the session's
# record publishes ``control_key`` mode 0600, ANY same-uid process may read it,
# and the runtime accepted ``slash``/``approval_answer`` on that key alone. A
# model-authored ``bash`` call runs as exactly that uid, so it could read the
# record, dial the loopback port and set its own gate to ``auto`` — the
# constrained subject minting the authority that removes its own requirement,
# one route over from the one #1282 closed.
#
# The fix is one predicate (:func:`transition_authority`) and one credential
# (:func:`mint_operator_cap`) that never reaches the filesystem AND never
# crosses the wire: the record key stays the whole authorization story for
# ORDINARY operations, and an authority-INCREASING one additionally demands a
# per-connection PROOF of that credential — the runtime proves possession in its
# welcome, the console verifies before it presents anything, and the value
# itself stays in the two processes that already had it. Sending the value
# instead was an earlier revision of this change and was defeated by the record
# being same-uid writable (see :func:`_proof`).

#: Which class of control-plane request this is.
#:
#: A ``Literal`` rather than a bool because the seam reads as a question — "is
#: this one of the increasing ones?" — and a bare ``True`` at the call site
#: would say nothing about which direction was decided. An ``Authority`` is
#: never persisted and never crosses the wire.
Authority = Literal["authority-increasing", "ordinary"]

#: 32 random bytes; 64 hex characters on the wire. Long enough that guessing is
#: not a strategy even within one process's lifetime, and a round number of
#: bytes so the comparison is over two equal-length byte strings.
OPERATOR_CAP_BYTES = 32

#: The arguments that move a live gate from ``ask`` to ``auto``. The SAME set
#: both ``/approvals`` handlers already compare against — a deliberate
#: duplication of three string literals rather than an import, because the
#: handlers' ``elif argument in (...)`` is the SINK's own spelling of the rule
#: and this module must classify exactly the frames that reach it. A test pins
#: the two in step (``tests/unit/harness/test_approval_authority.py``), so
#: widening one alone fails the suite rather than silently leaving a loosening
#: word unguarded.
APPROVALS_LOOSENING_WORDS = frozenset({"auto", "off", "yolo"})


def transition_authority(command: str, args: str, *, approved: bool | None = None) -> Authority:
    """Whether this control-plane request INCREASES authority.

    Authority-INCREASING means one thing: the request removes or answers the
    gate that constrains the caller.

    * ``/approvals auto|off|yolo`` sets the running gate's ``_auto_approve`` to
      ``True`` — in the runtime's handle, or in the TUI app for the gate the
      app owns;
    * ``approval_answer(approved=True)`` resolves the parked card, which is the
      same concession one card at a time.

    Everything else is ORDINARY and stays on the record key exactly as before:
    ``ask``/``on``/``prompt`` (and any other unknown word, which the sinks
    answer with a notice), a bare ``/approvals`` (a report), ``default …`` (a
    persist the runtime declines anyway), ``read``/``status``/``stop``/
    ``prompt``/``model``/``rename``, ``peer_message``, and ``ask_answer`` —
    answering a question the MODEL asked is not a concession to the model.

    ``approved=False`` is deliberately NOT increasing: a deny settles the card
    in the safe direction, so it must keep working from every surface that can
    reach the session. That asymmetry is the point — the routes that may loosen
    are a subset of the routes that may tighten.

    ``command`` is resolved to its registry PRIMARY name before matching, the
    same way both dispatching hosts resolve it (``primary_slash_name``), so an
    alias of ``/approvals`` cannot slip past this seam and reach a sink that
    would have honoured it.
    """
    if approved:
        return "authority-increasing"
    # Imported function-locally: ``slash_commands`` is the registry and pulls a
    # wider import graph than this stdlib-only module should carry, and this
    # predicate is asked once per slash frame rather than on any hot path.
    from local_operator.slash_commands import primary_slash_name

    if primary_slash_name(str(command)) != "approvals":
        return "ordinary"
    if str(args or "").strip().lower() in APPROVALS_LOOSENING_WORDS:
        return "authority-increasing"
    return "ordinary"


#: Every control op that can carry an authority-increasing request.
#:
#: A closed set rather than an inline condition, because TWO ends read it and
#: they must agree: the runtime refuses such a frame unless it presents the
#: capability (``session/runtime/server.py``) and the console presents it on
#: exactly those frames (``mobile/attach_client.py``). A set that drifts leaves
#: a route where the client believes it is authorising and the server believes
#: the frame is ordinary — the failure mode of this whole mechanism, one layer
#: down.
#:
#: ``tests/unit/session/runtime/test_approval_authority_seam.py`` re-derives it
#: from ``server.py``'s own dispatch source and asserts the two are equal, so a
#: coder who adds an op that reaches a sink fails the suite rather than shipping
#: an unguarded way to loosen a running gate.
AUTHORITY_OPS = frozenset({"slash", "slash_result", "approval_answer"})


def frame_authority(frame: dict[str, Any]) -> Authority | None:
    """The class of a control FRAME, or ``None`` when the op carries no class.

    The one place that knows which field of which op carries the transition: a
    ``slash``/``slash_result`` frame names a command and its arguments, an
    ``approval_answer`` frame names a verdict, and every other op is ordinary by
    construction. Callers that only care about the refusal (the runtime) and
    callers that only care about the presentation (the console) both ask here,
    so neither can invent its own reading of what a frame means.
    """
    op = frame.get("op")
    if op not in AUTHORITY_OPS:
        return None
    if op == "approval_answer":
        # Truthiness rather than ``is True``: the dispatch reads this field in a
        # boolean position, so anything truthy IS an approval and must be judged
        # as one. ``False`` — and an absent value, which fails the dispatch on
        # its own — stays ordinary, so denying a card keeps working everywhere.
        return transition_authority("", "", approved=bool(frame.get("approved")))
    return transition_authority(str(frame.get("command", "")), str(frame.get("args", "")))


def operator_nonce() -> str:
    """A fresh per-connection nonce/salt, hex. NOT a secret, and not reusable.

    64 hex characters, so the wire shape is the one the validator already
    admits for the proof field. It exists so every proof is bound to ONE
    connection: a value that crosses the wire here is worthless on any other.
    """
    return secrets.token_hex(OPERATOR_CAP_BYTES)


def is_wire_hex(value: object) -> bool:
    """Whether ``value`` has the wire shape of a nonce, salt or proof.

    One definition, used by the runtime's auth-frame reader, the frame
    validator and the attach client's handshake check: a shape check that drifts
    between the three would admit a value one end can produce and another cannot
    read back. Not a security check — the PROOF is the check — but the reason a
    malformed field degrades to "no handshake" rather than to an exception.
    """
    if not isinstance(value, str) or len(value) != OPERATOR_CAP_BYTES * 2:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


#: The two directions a per-connection proof can be computed in, domain
#: separated so a proof harvested from one cannot be replayed as the other.
_PROOF_LABEL_HANDSHAKE = b"lop-operator-cap-handshake-v1"
_PROOF_LABEL_REQUEST = b"lop-operator-cap-request-v1"


def _proof(label: bytes, cap: bytes, client_nonce: str, server_salt: str) -> str:
    """``hmac_sha256(cap, label || client_nonce || ":" || server_salt)``, hex.

    The CAPABILITY ITSELF NEVER CROSSES THE WIRE. That is the whole point of
    this shape and the reason it replaced an earlier one that sent ``cap.hex()``
    as the frame field:

    * The discovery record is 0600, but it is WRITABLE by anything running under
      the same uid — including the model-run tool the gate is there to constrain.
      An earlier revision sent the capability VALUE to whatever endpoint the
      record named, so a same-uid impostor could rewrite ``control_port`` to its
      own listener, receive the value from the real console, and replay it to the
      real runtime (agent review round 1, R1-1; reproduced end to end with
      production clients). A proof is useless to a passive endpoint: it is bound
      to nonces chosen for one connection, and the impostor cannot make the real
      runtime accept it (that runtime generates its OWN salt per connection).
    * Mutual: the runtime proves possession FIRST (in its welcome), so a console
      never presents anything — not even a proof — to an endpoint that cannot
      produce the same proof. That keeps the harvest from being merely
      non-replayable and makes it empty.
    * Per connection: both ingredients are fresh per connection, so nothing
      accumulates that a later connection could use. A same-uid adversary that
      proxies an entire session (relaying the real runtime's proof to the
      console) can relay that connection's requests, which is what a proxy is —
      but it never learns the capability and cannot originate one of its own.

    Domain-separated by ``label`` so the handshake proof and the request proof
    are different values: a transcript's worth of one direction is not a
    credential for the other.
    """
    message = label + b"|" + client_nonce.encode("utf-8", "surrogatepass") + b"|"
    message += server_salt.encode("utf-8", "surrogatepass")
    return hmac.new(cap, message, hashlib.sha256).hexdigest()


def handshake_proof(cap: bytes, *, client_nonce: str, server_salt: str) -> str:
    """What the RUNTIME puts in its welcome: proof it holds ``cap``.

    The console verifies it before presenting anything, which is what stops a
    rewritten record from harvesting a credential: an endpoint that cannot
    compute this is an endpoint that never receives the next step.
    """
    return _proof(_PROOF_LABEL_HANDSHAKE, cap, client_nonce, server_salt)


def request_proof(cap: bytes, *, client_nonce: str, server_salt: str) -> str:
    """What the CONSOLE puts on an authority-increasing frame.

    Sent only after a verified :func:`handshake_proof`, and only on the frames
    :func:`frame_authority` classes as increasing.
    """
    return _proof(_PROOF_LABEL_REQUEST, cap, client_nonce, server_salt)


def _proof_ok(
    *,
    label: bytes,
    supplied: object,
    held: bytes | None,
    client_nonce: str,
    server_salt: str,
) -> bool:
    """Constant-time comparison of an offered proof against the expected one.

    FAIL-CLOSED in every degenerate direction, and each one is a real case:

    * ``held is None`` — this runtime was started by a process that did not hand
      one over (an older console, a test that constructed the server by hand, a
      background spawn with no console at all). There is nothing to match.
    * ``supplied`` absent or not a string — an old client, or a forged frame.
    * ``held`` the wrong length — a programming error on this side, and refusing
      is the only safe reading of a credential that is not the one we minted.
    * either nonce missing — a client that never asked for a handshake, so there
      is no connection-bound value it could legitimately hold.

    ``hmac.compare_digest`` over hex strings of equal length, so the comparison
    cannot raise on a non-ASCII candidate (Python's ``str`` form raises
    ``TypeError`` on non-ASCII input) and cannot leak a shared prefix through
    timing.
    """
    if not isinstance(held, (bytes, bytearray)) or len(held) != OPERATOR_CAP_BYTES:
        return False
    if not isinstance(supplied, str) or not supplied:
        return False
    if not client_nonce or not server_salt:
        return False
    expected = _proof(label, bytes(held), client_nonce, server_salt)
    return hmac.compare_digest(supplied.encode("utf-8", "surrogatepass"), expected.encode())


def handshake_proof_ok(
    *, supplied: object, held: bytes | None, client_nonce: str, server_salt: str
) -> bool:
    """Whether ``supplied`` is this connection's handshake proof, unverified by us."""
    return _proof_ok(
        label=_PROOF_LABEL_HANDSHAKE,
        supplied=supplied,
        held=held,
        client_nonce=client_nonce,
        server_salt=server_salt,
    )


def request_proof_ok(
    *, supplied: object, held: bytes | None, client_nonce: str, server_salt: str
) -> bool:
    """Whether ``supplied`` is this connection's proof for an increasing request."""
    return _proof_ok(
        label=_PROOF_LABEL_REQUEST,
        supplied=supplied,
        held=held,
        client_nonce=client_nonce,
        server_salt=server_salt,
    )


#: The ONE sentence a host hands back when a control-plane request tried to
#: loosen a running gate without the capability (see
#: :func:`transition_authority`).
#:
#: Lives here for the same reason :data:`LOOSENING_REFUSED_NOTICE` does — the
#: runtime writes it into an ``error`` frame and the TUI would have to write it
#: into a transcript, and two copies of one refusal is two chances for one
#: surface to describe the rule differently from the other.
#:
#: EVERY REMEDY IT NAMES MUST WORK FROM WHERE IT IS PRINTED (design round 1 D3,
#: UX round 1 U1/U2, agent review round 1 MINOR). This copy therefore replaced
#: one that named ``/approvals default auto`` — which the RUNTIME refuses from
#: the very pane that printed the notice — and it opens differently from
#: :data:`LOOSENING_REFUSED_NOTICE` so the two refusals are not visual twins:
#: that one is about a config write arriving from outside this session, this one
#: is about a command typed where the gate is not owned.
#:
#: What it names, in order: the thing that would work and where to do it; what
#: still works from here; the mechanism that DOES work when there is no owning
#: window at all (a background-spawned runtime has none — let it retire and
#: re-open the session, and the window that starts the runtime owns its gate);
#: and, last, how to make a NEW session start loosened, which is the only place
#: ``--yolo`` and the config key apply.
OPERATOR_CAP_REQUIRED_NOTICE = (
    "tool approvals stay at ask: /approvals auto removes this session's approval gate, and only "
    "the terminal or app window that started this session can do that. /approvals ask still "
    "tightens it from here, and a bare /approvals still reports it. If no window owns this "
    "session — it was started in the background — wait for its runtime to finish and go idle, "
    "then open the session again in this window: the window that starts a runtime owns its gate. "
    "New sessions can start loosened with --yolo, or with tool_approval_mode: auto in config.yml."
)

#: Runtime pid -> capability, for the runtimes THIS process spawned.
#:
#: This is "the console's memory" the design names, and a module-level dict
#: rather than a field threaded through every console object because the
#: question it answers is a property of the PROCESS, not of any one object in
#: it: "did I start the runtime behind this record?". The desktop backend, the
#: phone daemon, the TUI and the CLI are each one process that both spawns
#: runtimes (``session/runtime/launch._spawn_runtime``) and attaches to them
#: (``mobile/attach_client.AttachClient.connect``), and the pid is the identity
#: both ends already agree on — the discovery record is keyed by it.
#:
#: Never written anywhere. It dies with the process, and a pid key is only ever
#: added by the spawner that owns the child, so a stale entry can at worst fail
#: to match a live record (fail-closed).
_OPERATOR_CAPS: dict[int, bytes] = {}


def mint_operator_cap() -> bytes:
    """A fresh capability. Called once per spawned runtime, by the spawner."""
    return secrets.token_bytes(OPERATOR_CAP_BYTES)


def remember_operator_cap(pid: int, cap: bytes) -> None:
    """Record that THIS process spawned the runtime at ``pid`` and holds its cap."""
    _OPERATOR_CAPS[int(pid)] = bytes(cap)


def operator_cap_for(pid: int) -> bytes | None:
    """The capability for ``pid``, or ``None`` when another process spawned it.

    ``None`` is the answer that matters: it is what makes the desktop route,
    the phone relay, a peer send and a pane attached to a background-started
    runtime refuse an authority-increasing command, while the process that
    brought the session's runtime into existence keeps working.
    """
    return _OPERATOR_CAPS.get(int(pid))


def reset_operator_caps_for_tests() -> None:
    """Clear the process-wide table. Tests only — nothing in production calls it."""
    _OPERATOR_CAPS.clear()


class OperatorCapHandoff:
    """The spawner's end of the one-time capability handoff into a runtime.

    The capability must reach the child WITHOUT touching anything the child's
    own tool subprocesses could later read. Every other channel is one of them:

    * ``argv`` is ps-readable, and so is the environment (``ps -E``) — and the
      module docstring of ``session/runtime/process.py`` already rules argv out
      for the session's own identity for exactly this reason;
    * the runtime's log file is a 0600 file whose path the model can print, and
      ``session/runtime/launch.py`` pipes the child's stdout/stderr into one of
      those, so a capability PRINTER would defeat the whole mechanism;
    * the discovery record is the defect itself (``control_key`` lives there).

    So the value travels on an inherited descriptor that is opened for the
    spawn, written, and closed on both sides immediately — and only its NUMBER
    rides in argv, which is not a secret. Tool subprocesses are spawned later
    with ``close_fds=True`` and ``start_new_session=True``
    (``tools/builtin.py``), so by the time any of them exists the descriptor is
    gone from this process's table as well.

    Two platform paths, and the difference is a real strength difference the
    runtime reports rather than hides (see :func:`operator_cap_guarantee`):

    * POSIX — ``socketpair(AF_UNIX, SOCK_STREAM)``, the child's end handed over
      with ``pass_fds`` so only that one descriptor survives the exec;
    * Windows — ``pass_fds`` does not exist, so the descriptor is an anonymous
      pipe made inheritable and ``close_fds`` has to be off for the spawn,
      which hands the child every inheritable handle this process owns. The
      boundary is weaker there regardless (any same-user process may read
      another's memory); the mode report says so.
    """

    def __init__(
        self,
        *,
        argv: list[str],
        pass_fds: tuple[int, ...],
        close_fds: bool,
        writer: Callable[[bytes], None],
        closer: Callable[[], None],
    ) -> None:
        #: Append to the child's argv. Carries the descriptor NUMBER only.
        self.argv = argv
        self.pass_fds = pass_fds
        self.close_fds = close_fds
        #: Writes the capability. Deliberately a CALLABLE taking bytes rather
        #: than a file object: the POSIX end is a socket (``sendall``) and the
        #: Windows end is a raw handle, and ``socket.makefile`` would hold a
        #: second reference to the descriptor that ``close()`` below cannot
        #: release — the exact leak this handoff exists to avoid.
        self._writer = writer
        self._closer = closer
        self._closed = False

    def deliver(self, cap: bytes) -> None:
        """Write the capability and close BOTH ends of the handoff.

        Called after the spawn: a 32-byte write into an empty socketpair or
        pipe cannot block, so this is safe from the spawning thread without a
        reader on the far side yet.
        """
        try:
            self._writer(cap)
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closer()
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


#: The argv flag carrying the descriptor number. Spelled once: the spawner
#: (``session/runtime/launch._spawn_runtime``) and the child
#: (``session/runtime/process.main``) are different processes at different
#: times, and a typo in either is a silent "no capability" at best.
OPERATOR_FD_FLAG = "--operator-fd"


def open_operator_cap_handoff() -> OperatorCapHandoff:
    """Create the descriptor pair a spawn hands its new runtime the cap on."""
    if os.name == "nt":  # pragma: no cover — Windows only; CI runs POSIX
        read_handle, write_handle = _win32_inheritable_pipe()
        return OperatorCapHandoff(
            argv=[OPERATOR_FD_FLAG, str(int(read_handle))],
            pass_fds=(),
            close_fds=False,
            writer=_win32_handle_writer(int(write_handle)),
            closer=lambda: _win32_close(int(read_handle), int(write_handle)),
        )

    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    child_fd = child.fileno()

    def closer() -> None:
        # BOTH ends, and this process's copy of the CHILD's end with them: if
        # this process kept that one, the descriptor would still be in its own
        # table, and ``close_fds=True`` on every later tool spawn would be the
        # only thing between it and the model. Close it here instead.
        for sock in (parent, child):
            try:
                sock.close()
            except OSError:  # pragma: no cover — closing an already-closed pair
                logger.debug("operator cap handoff already closed", exc_info=True)

    return OperatorCapHandoff(
        argv=[OPERATOR_FD_FLAG, str(child_fd)],
        pass_fds=(child_fd,),
        close_fds=True,
        writer=parent.sendall,
        closer=closer,
    )


def _win32_inheritable_pipe() -> tuple[int, int]:  # pragma: no cover — Windows only
    """An anonymous pipe whose ends are inheritable. Windows-only helper.

    Split out so the POSIX path above reads without platform branching, and so
    a future Windows job has one function to exercise.
    """
    import ctypes
    import msvcrt

    read_fd, write_fd = os.pipe()
    for fd in (read_fd, write_fd):
        handle = msvcrt.get_osfhandle(fd)  # type: ignore[attr-defined]  (Windows-only module)
        if not ctypes.windll.kernel32.SetHandleInformation(  # type: ignore[attr-defined]
            ctypes.c_void_p(handle), 1, 1
        ):
            raise OSError("could not make the operator capability pipe inheritable")
    return (
        msvcrt.get_osfhandle(read_fd),  # type: ignore[attr-defined]
        msvcrt.get_osfhandle(write_fd),  # type: ignore[attr-defined]
    )


class _Win32HandleWriter:  # pragma: no cover — Windows only
    """A callable that writes bytes to a raw Windows handle."""

    def __init__(self, handle: int) -> None:
        self._handle = handle

    def __call__(self, data: bytes) -> None:
        import msvcrt

        flags = os.O_WRONLY | os.O_BINARY  # type: ignore[attr-defined]  (Windows-only)
        fd = msvcrt.open_osfhandle(self._handle, flags)  # type: ignore[attr-defined]
        try:
            os.write(fd, data)
        finally:
            os.close(fd)


def _win32_handle_writer(handle: int) -> Callable[[bytes], None]:
    return _Win32HandleWriter(handle)


def _win32_close(read_handle: int, write_handle: int) -> None:  # pragma: no cover
    import ctypes

    for handle in (read_handle, write_handle):
        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle))  # type: ignore[attr-defined]


def read_operator_cap_from_argv(argv: Sequence[str]) -> bytes | None:
    """The child's end: read the capability the spawner wrote, then close.

    Returns ``None`` whenever there is nothing to read — no flag (an older
    spawner, a hand-written argv, a module run by hand), a flag with no value,
    an unreadable descriptor. ``None`` is the fail-closed state: the runtime
    then holds no capability and refuses every authority-increasing request,
    while every ordinary operation is unaffected.

    The descriptor is closed on every path that READ it — including the short
    handoff, both failure branches and the platform branch — because leaving it
    open is what would let a later tool subprocess find it by number. The one
    case that cannot close it is the Windows handle translation failing: ``os``
    never owned that handle, so there is nothing here to close and the caller
    holds no capability (fail-closed).
    """
    value: str | None = None
    items = list(argv)
    for index, item in enumerate(items):
        if item == OPERATOR_FD_FLAG and index + 1 < len(items):
            value = items[index + 1]
            break
        if item.startswith(f"{OPERATOR_FD_FLAG}="):
            value = item.split("=", 1)[1]
            break
    if value is None:
        return None
    try:
        descriptor = int(value)
    except ValueError:
        logger.warning("session runtime: %s carried a non-numeric value", OPERATOR_FD_FLAG)
        return None

    if os.name == "nt":  # pragma: no cover — Windows only
        import msvcrt

        try:
            flags = os.O_RDONLY | os.O_BINARY  # type: ignore[attr-defined]  (Windows-only)
            descriptor = msvcrt.open_osfhandle(descriptor, flags)  # type: ignore[attr-defined]
        except OSError:
            logger.warning("session runtime: operator capability handle was not inherited")
            return None

    buffer = bytearray()
    try:
        while len(buffer) < OPERATOR_CAP_BYTES:
            chunk = os.read(descriptor, OPERATOR_CAP_BYTES - len(buffer))
            if not chunk:
                break
            buffer.extend(chunk)
    except OSError:
        logger.warning("session runtime: could not read the operator capability")
        return None
    finally:
        try:
            os.close(descriptor)
        except OSError:  # pragma: no cover — already closed by the reader above
            logger.debug("operator capability descriptor was already closed", exc_info=True)

    if len(buffer) != OPERATOR_CAP_BYTES:
        logger.warning(
            "session runtime: short operator capability handoff (%d of %d bytes)",
            len(buffer),
            OPERATOR_CAP_BYTES,
        )
        return None
    return bytes(buffer)


#: Reported once per process, at the first runtime that mounts a capability.
_GUARANTEE_REPORTED = False


def operator_cap_guarantee() -> str:
    """How strong the capability's boundary is ON THIS HOST, as far as we can tell.

    The capability separates an operator's intent from a same-uid process's
    reach only while the operating system does. Nothing in this module may
    assume that, and the refusal copy must not overclaim, so the runtime
    reports the level it can detect instead of implying one:

    * ``strong`` — reaching another process's memory requires a privilege this
      uid does not have. The default on macOS.
    * ``not-a-boundary`` — Linux with ``ptrace_scope=0``: any same-uid process
      may attach to this one and read the capability straight out of memory, so
      there the capability raises the cost of the attack rather than closing
      it.
    * ``weak`` — Windows, where same-user process access is unrestricted.
    * ``unreported`` — Linux without the ``yama`` LSM, or an unknown platform.
      Read as "we cannot say", never as "strong".
    """
    if sys.platform.startswith("linux"):
        try:
            scope = Path("/proc/sys/kernel/yama/ptrace_scope").read_text().strip()
        except OSError:
            return "unreported"
        return "not-a-boundary" if scope == "0" else "strong"
    if sys.platform == "darwin":
        return "strong"
    if os.name == "nt":
        return "weak"
    return "unreported"


def report_operator_cap_guarantee() -> str:
    """Log the guarantee level once per process and return it.

    Once, not per runtime: the answer is a property of the host, and a
    long-lived daemon spawning twenty runtimes would otherwise write the same
    line twenty times into a bounded log file.
    """
    global _GUARANTEE_REPORTED
    level = operator_cap_guarantee()
    if not _GUARANTEE_REPORTED:
        _GUARANTEE_REPORTED = True
        logger.info("operator capability boundary on this host: %s (%s)", level, sys.platform)
    return level


__all__ = [
    "APPROVALS_LOOSENING_WORDS",
    "AUTHORITY_OPS",
    "ApprovalGate",
    "Authority",
    "LOOSENING_KEPT_BY_ASK_NOTICE",
    "LOOSENING_REFUSED_NOTICE",
    "OPERATOR_CAP_BYTES",
    "OPERATOR_CAP_REQUIRED_NOTICE",
    "OPERATOR_FD_FLAG",
    "OperatorCapHandoff",
    "ask_approval",
    "frame_authority",
    "handshake_proof",
    "handshake_proof_ok",
    "is_wire_hex",
    "loosening_is_authorised",
    "mint_operator_cap",
    "open_operator_cap_handoff",
    "operator_cap_for",
    "operator_cap_guarantee",
    "operator_nonce",
    "read_operator_cap_from_argv",
    "request_proof",
    "request_proof_ok",
    "remember_operator_cap",
    "report_operator_cap_guarantee",
    "reset_operator_caps_for_tests",
    "transition_authority",
]

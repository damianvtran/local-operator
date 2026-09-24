"""Session-visible variables behind the ``list_variables`` / ``read_variable``
tools.

The token budget is the whole point: variable VALUES are never written into
the system prompt. The agent discovers what is available through
``list_variables`` (names only) and pulls a single value on demand with
``read_variable``. That keeps large or secret values out of the rolling
context until the agent actually needs them.

Security is a hard constraint, not an afterthought. The process environment
is a credential minefield (API keys, tokens, AWS secrets), so it is NOT
exposed wholesale to an auto-approved tool. What is visible:

1. ``config_values`` — a mapping (e.g. the config's ``variables`` section)
   injected at session creation. Highest precedence.
2. A project-local ``.local-operator.env`` file in the working directory,
   parsed as ``KEY=VALUE`` lines.
3. ONLY environment variables whose name starts with the ``LOCAL_OPERATOR_``
   opt-in prefix. Anything else in the environment is invisible to the agent.

Names matching secret patterns are excluded from BOTH listing and reading,
regardless of source, so a teammate-supplied project file cannot smuggle a
credential past the denylist. The pattern targets credential KINDS
(``secret``, ``token``, ``password``/``passwd``, ``credential``,
``authorization``, ``bearer``, ``api_key``/``apikey``) plus a name that ends
in ``_key`` or is exactly ``key`` — deliberately NOT any name merely
containing "key", which is far too common in legitimate config
(``keyboard_layout``, ``monkeypatch``). Names are NFKC-normalised before
matching so a Unicode homoglyph cannot slip a credential past it.
Over-matching is the safe direction: it hides more, never less.

Redaction has TWO passes, and both matter here: the values this session KNOWS
(above) and the credential SHAPES anything may print — a DSN, a connection
string, an ``AWS_SECRET_ACCESS_KEY=`` line, a PEM block, an issuer-prefixed
token. ``VariableStore.redact`` composes them, and that one callable is what
every model-visible surface reads; see :mod:`local_operator.redaction_shapes`
for what the shape pass does and does not guarantee.

Session credentials (``/credential``, ``ask`` with ``secret=true``) are a
fourth, memory-only source that inverts that rule on purpose. The operator
hands the process a secret the agent must USE and must never READ: the name
is advertised (system-prompt block, ``credential_names``), the value is
injected into every ``bash`` environment, and ``list_variables`` /
``read_variable`` still refuse it. Nothing is written to disk and nothing
survives the session. They live in their own map so a non-secret-shaped
name (``DATABASE_URL``) cannot leak through ``read`` the way a config
override would.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# The shared credential-shape table. STDLIB-ONLY and leaf, deliberately: this
# module sits on the CLI startup path (``tests/unit/test_import_graph.py`` pins
# that) and the shape pass has to be reachable from the result path without
# dragging a session or a provider layer in behind it.
from local_operator.redaction_shapes import (
    ShapeHit,
    ShapeReport,
    is_registerable_component,
    scrub_secrets_with_hits,
    scrub_values,
    shape_report,
)

#: Environment variables only surface to the agent when opted in with this
#: prefix. Everything else in the process env stays invisible.
ENV_ALLOW_PREFIX = "LOCAL_OPERATOR_"

#: Secret-shaped names are never listed or read, whatever their source. This
#: targets credential KINDS (secret/token/password/.../api_key), not the bare
#: token "key" which is far too common in legitimate config names. The
#: matching is deliberately loose — over-matching only hides more.
logger = logging.getLogger(__name__)

_SECRET_RE = re.compile(
    r"(?i)(secret|token|password|passwd|credential|authorization|bearer|"
    r"api[_-]?key|apikey|[_-]key$|^key([_\-.]|$))"
)


#: Where a stored session credential came from. Operator-facing only.
CredentialSource = Literal["command", "ask"]

#: Why :meth:`VariableStore.store_credential` refused to store anything.
CredentialStoreFailure = Literal["empty-key", "empty-value"]


@dataclass(frozen=True, slots=True)
class SessionCredential:
    """A credential the operator has handed this session. Carries no secret bytes."""

    key: str
    source: CredentialSource


@dataclass(frozen=True, slots=True)
class CredentialStoreResult:
    """Outcome of one :meth:`VariableStore.store_credential` call."""

    ok: bool
    credential: SessionCredential | None = None
    replaced: bool = False
    reason: CredentialStoreFailure | None = None


def normalize_credential_key(raw: str) -> str | None:
    """Collapse an operator-supplied label to env-var shape.

    ``github token``, ``github-token`` and ``GITHUB_TOKEN`` all become
    ``GITHUB_TOKEN``, so the same credential is addressable however it was
    typed. Returns ``None`` when nothing usable remains, which the caller
    reports rather than guessing a name.
    """
    pieces = [part for part in re.split(r"[^A-Za-z0-9]+", raw.strip()) if part]
    if not pieces:
        return None
    return "_".join(pieces).upper()


def describe_store_failure(reason: CredentialStoreFailure, key: str) -> str:
    """Operator-facing explanation for a refused store."""
    if reason == "empty-key":
        return f"Not a usable credential key: {key}"
    return "Nothing pasted; no credential stored."


#: Verbs are flag-shaped (``--forget``) rather than bare words so they can
#: never collide with a credential key: keys normalize to ``[A-Z0-9_]``,
#: which cannot begin with ``-``.
@dataclass(frozen=True, slots=True)
class CredentialCommand:
    """What ``/credential <args>`` asked for."""

    action: Literal["list", "store", "forget", "forget-all", "persist", "error"]
    key: str = ""
    message: str = ""


#: The operator-facing help for the command, shown on every parse error.
#:
#: The INLINE gesture leads, because it is the one an operator reaches for and
#: the one that used to fail silently: typing `/credential <secret>` fell
#: through to the `<KEY>` form with the secret as its key argument, and the
#: value landed in the transcript in plaintext. Naming the space and the
#: terminating Enter here is what makes the mode discoverable from the error an
#: operator sees when they get it wrong.
#:
#: THE TYPED ``<KEY>`` FORM IS NOT LISTED, because it can no longer be TYPED and
#: help that cannot be followed is worse than help that is missing. The space
#: after the token now always opens a masked capture, so a hand-typed key name
#: is minted as a short secret rather than reaching the prompt — measured on
#: this branch, ``/credential MYKEY`` produced ``[Credential #1, 5 chars]`` (QA
#: round 1, Q1). The form itself still EXISTS and is still parsed: a pasted
#: whole line reaches it, and it is the route a viewer session uses to hand a
#: secret to its owner. It is simply not something to advertise as typable.
#: Naming a credential yourself is what was lost; the generated ``LOP_SECRET_``
#: name in the chip is what ``--persist`` and ``--forget`` take instead.
#:
#: ONE ALIGNED COMMENT COLUMN, and a lead short enough not to wrap. The previous
#: first line was 74 cells and folded in the notice column, splitting
#: ``# masked; Enter`` from ``chips it`` and leaving the rest of the block at a
#: different indent, with the ``#`` column unaligned across three different
#: offsets (design round 1, D6).
#:
#: THE FIRST LINE IS BUDGETED FOR ITS CALLER'S PREFIX, which is what the earlier
#: attempt missed. ``format_credential_list`` prepends
#: ``No credentials stored for this session. `` (40 cells) before this string, so
#: the lead is measured against that too: ``NoticeBlock.body_budget(100)`` is 94
#: cells, leaving ~54 for line one. Measured in a rendered frame rather than
#: counted in the source — the first version of this block still wrapped at 100
#: columns because only the bare string was checked.
CREDENTIAL_USAGE = (
    "Usage: /credential <space> then the secret\n"
    "       Enter chips it, Esc cancels     # it never enters the line\n"
    "       /credential                     # list\n"
    "       /credential --persist <KEY>     # also save it long-term\n"
    "       /credential --forget <KEY>      # forget one\n"
    "       /credential --forget-all        # forget every one"
)


def parse_credential_command(args: str) -> CredentialCommand:
    """What ``/credential <args>`` asked for."""
    trimmed = args.strip()
    if not trimmed:
        return CredentialCommand("list")
    if trimmed == "--forget-all":
        return CredentialCommand("forget-all")
    if trimmed.startswith("--forget"):
        rest = trimmed[len("--forget") :].strip()
        if not rest:
            return CredentialCommand("error", message=f"Missing key. {CREDENTIAL_USAGE}")
        key = normalize_credential_key(rest)
        if key is None:
            return CredentialCommand("error", message=f"Not a usable credential key: {rest}")
        return CredentialCommand("forget", key=key)
    if trimmed.startswith("--persist"):
        # Design §5.4: the session path is unchanged and GAINS a route. The
        # credential named here stays in session memory and is additionally
        # written to the encrypted long-term store.
        rest = trimmed[len("--persist") :].strip()
        if not rest:
            return CredentialCommand("error", message=f"Missing key. {CREDENTIAL_USAGE}")
        key = normalize_credential_key(rest)
        if key is None:
            return CredentialCommand("error", message=f"Not a usable credential key: {rest}")
        return CredentialCommand("persist", key=key)
    if trimmed.startswith("-"):
        option = trimmed.split()[0]
        return CredentialCommand("error", message=f"Unknown option: {option}. {CREDENTIAL_USAGE}")
    key = normalize_credential_key(trimmed)
    if key is None:
        return CredentialCommand("error", message=f"Not a usable credential key: {trimmed}")
    return CredentialCommand("store", key=key)


def format_credential_list(credentials: list[SessionCredential]) -> str:
    """Operator-facing listing. Names and sources only — never values."""
    if not credentials:
        return f"No credentials stored for this session. {CREDENTIAL_USAGE}"
    width = max(len(item.key) for item in credentials)
    rows = [
        f"  {item.key.ljust(width)}  from {'/credential' if item.source == 'command' else 'ask'}"
        for item in credentials
    ]
    return "\n".join(
        [
            f"Session credentials ({len(credentials)}) — held in memory for this session only:",
            *rows,
            "Injected into every bash command as environment variables. "
            "The agent cannot read the values.",
        ]
    )


def format_credential_forget(removed: bool, key: str) -> str:
    if removed:
        return f"Forgot {key}. The agent can no longer use it in this session."
    return f"No credential named {key}."


def format_credential_forget_all(count: int) -> str:
    if count == 0:
        return "No credentials stored for this session."
    noun = "credential" if count == 1 else "credentials"
    return f"Forgot {count} {noun}."


def redact_secret_values(text: str, secrets: Mapping[str, str] | Sequence[str]) -> str:
    """Replace every known secret byte-string in ``text`` with ``[redacted]``.

    Longest first so a value that is a prefix of another cannot leave a tail
    behind, and in every SPELLING the value may be printed in
    (:func:`~local_operator.redaction_shapes.credential_forms`) rather than only
    verbatim. Empty strings are skipped: replacing nothing with a marker would
    insert ``[redacted]`` between every character.

    The loop itself lives in :func:`~local_operator.redaction_shapes.scrub_values`
    and this delegates to it rather than repeating it. Two copies of a mask policy
    is how they drift — the copy that misses a spelling is the one that leaks — and
    the surfaces here (MCP diagnostics) and there (the composed result pass, the
    eval ledger) are all hiding the same values.
    """
    values = list(secrets.values()) if isinstance(secrets, Mapping) else list(secrets)
    return scrub_values(text, values)


def _is_secret(name: str) -> bool:
    """True when a name looks like a credential and must stay invisible.

    The name is NFKC-normalised first, which folds COMPATIBILITY codepoints:
    fullwidth ``ＡＰＩ_ＫＥＹ`` and mathematical-bold ``𝐀𝐏𝐈_𝐊𝐄𝐘`` both become
    ``API_KEY``. Without it those read as different strings to the regex while
    still naming the same variable, which is a silent exfiltration path through
    a teammate-supplied project file.

    NFKC does NOT fold cross-script confusables — Cyrillic ``а`` in
    ``pаssword`` survives — so this narrows the gap rather than closing it.
    That is acceptable because the denylist is defence in depth, not the
    control: the ``LOCAL_OPERATOR_`` opt-in prefix is what actually keeps the
    process environment invisible, and a name has to be deliberately opted in
    before the denylist is ever consulted.
    """
    return bool(_SECRET_RE.search(unicodedata.normalize("NFKC", name)))


class VariableStore:
    """Named, lazily-read, denylist-filtered variables for one session."""

    def __init__(
        self,
        cwd: str | None = None,
        config_values: Mapping[str, str] | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        # ``env`` is overridable for tests; defaults to the real process env,
        # resolved lazily so values are read at call time, never frozen.
        self._env = env
        self._config_values = dict(config_values or {})
        self._cwd = cwd or os.getcwd()
        # PER STORE, never class-level: a mutable default in the class body is shared by
        # every VariableStore in the process (`.add()` cannot create an instance
        # attribute), so one session's registered credentials were scanned in another's
        # results and the 64-value cap ran out process-wide — containment silently
        # stopping for every session, with the warning logged once on nobody's behalf.
        # Two consequences surfaced it: a red CI shard where an earlier test had filled
        # the shared set, and the cross-session coupling the design note forbids.
        self._shape_registrations: set[str] = set()
        self._shape_registration_cap_logged = False
        # Insertion-ordered so the prompt block and ``/credential`` listing
        # agree. Values live only here: never serialized, never listed, never
        # returned by ``get``/``read``.
        self._credentials: dict[str, str] = {}
        self._credential_meta: dict[str, SessionCredential] = {}
        # Values to SCRUB but never to inject or advertise (design §6). A
        # long-term secret a child fetched through the broker lands here so
        # `redact` catches it in that child's output, while `credential_env`
        # and `credential_names` stay untouched — registering a value for
        # scrubbing must never make it readable or injectable, which is why
        # this is a separate set rather than another entry in `_credentials`.
        self._redactions: set[str] = set()

    # -- sources -----------------------------------------------------------
    def _project_file(self) -> dict[str, str]:
        """Parse ``.local-operator.env`` from the working directory."""
        path = Path(self._cwd) / ".local-operator.env"
        out: dict[str, str] = {}
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return out
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip().strip('"').strip("'")
        return out

    def _live_env(self) -> Mapping[str, str]:
        return self._env if self._env is not None else os.environ

    def _env_visible(self) -> dict[str, str]:
        """Only opted-in (``LOCAL_OPERATOR_*``), non-secret env variables."""
        return {
            k: v
            for k, v in self._live_env().items()
            if k.startswith(ENV_ALLOW_PREFIX) and not _is_secret(k)
        }

    # -- public API --------------------------------------------------------
    def names(self) -> list[str]:
        """All non-secret variable names, sorted, deduplicated (no values)."""
        names = set(self._config_values)
        names.update(self._project_file())
        names.update(self._env_visible())
        return sorted(n for n in names if not _is_secret(n))

    def get(self, name: str) -> str | None:
        """Resolve ``name`` live; None when unknown or secret-shaped.

        Precedence: config > project file > opted-in env. A secret-shaped
        name is never resolved regardless of source.
        """
        if not name or _is_secret(name):
            return None
        if name in self._config_values:
            return str(self._config_values[name])
        project = self._project_file()
        if name in project:
            return project[name]
        if name.startswith(ENV_ALLOW_PREFIX):
            return self._live_env().get(name)
        return None

    def read(self, name: str) -> str:
        """Read a variable, or raise ``KeyError`` when unknown/denied."""
        value = self.get(name)
        if value is None:
            raise KeyError(name)
        return value

    # -- session credentials -----------------------------------------------
    def store_credential(
        self, raw_key: str, value: str, source: CredentialSource = "command"
    ) -> CredentialStoreResult:
        """Capture a secret under ``raw_key`` for this session only.

        The value is trimmed because a paste routinely carries a trailing
        newline, and a credential with stray whitespace fails authentication
        in a way that is invisible to everyone involved. Empty after trim is
        a refusal, not a stored blank: a blank env var is how a tool silently
        falls back to some other credential the operator did not intend.
        """
        key = normalize_credential_key(raw_key)
        if key is None:
            return CredentialStoreResult(ok=False, reason="empty-key")
        trimmed = value.strip()
        if not trimmed:
            return CredentialStoreResult(ok=False, reason="empty-value")
        replaced = key in self._credentials
        self._credentials[key] = trimmed
        meta = SessionCredential(key=key, source=source)
        self._credential_meta[key] = meta
        return CredentialStoreResult(ok=True, credential=meta, replaced=replaced)

    def forget_credential(self, raw_key: str) -> bool:
        """Drop a session credential. Returns whether one was actually stored."""
        key = normalize_credential_key(raw_key)
        if key is None or key not in self._credentials:
            return False
        del self._credentials[key]
        self._credential_meta.pop(key, None)
        return True

    def clear_credentials(self) -> int:
        """Drop every session credential. Returns how many were stored."""
        count = len(self._credentials)
        self._credentials.clear()
        self._credential_meta.clear()
        return count

    def credential_names(self) -> list[str]:
        """Stored credential keys, in insertion order. Never values."""
        return list(self._credentials)

    def list_credentials(self) -> list[SessionCredential]:
        """Operator-facing metadata for every stored credential."""
        return [self._credential_meta[key] for key in self._credentials]

    def credential_env(self) -> dict[str, str]:
        """A copy of the credential map for injecting into a child process.

        A copy, not a view: the caller mutates the process env it is building,
        and must not be able to write back into this store.
        """
        return dict(self._credentials)

    def register_redaction(self, value: str) -> bool:
        """Scrub ``value`` from future output without storing it as a credential.

        The §6 sink for broker retrievals: a value a CHILD fetched (via
        ``$(lop secret get X)``) never passes through this process, so the
        session is told about it out of band and records it here before the
        child is allowed to produce any output. Returns whether it was newly
        registered.
        """
        trimmed = value.strip()
        if not trimmed:
            return False
        if trimmed in self._redactions:
            return False
        self._redactions.add(trimmed)
        return True

    def redaction_values(self) -> list[str]:
        """Every value to scrub: session credentials plus §6 registrations.

        Values only, never names — the caller is a filter, not a reader.
        """
        return [*self._credentials.values(), *self._redactions]

    def unregister_redaction(self, value: str) -> bool:
        """Stop scrubbing a §6 registration; returns whether anything was dropped.

        The counterpart :meth:`register_redaction` needs. Without it the redaction
        set can only GROW, so a registration outlives its reason and any test (or
        short-lived resolver) that made one leaves the process scrubbing that value
        for everything that runs afterwards — measured as one store test's
        three-character value rewriting an unrelated provider warning in a later
        test of the same worker (agent review R-1 / QA Q1).

        Only §6 registrations are droppable: a value held in ``_credentials`` is a
        credential for as long as the session holds it, so dropping it from the
        scrub list is not reachable here.
        """
        trimmed = value.strip()
        if not trimmed:
            return False
        # ``discard`` returns None, so the membership has to be read before the
        # removal for the return value to mean what the docstring says.
        dropped = trimmed in self._redactions
        self._redactions.discard(trimmed)
        return dropped

    def redact(self, text: str) -> str:
        """Remove every known credential, then every credential SHAPE, from ``text``.

        **The ONE callable that makes coverage universal.** This is what
        ``session/session.py`` hands to the harness loop as
        ``redact_tool_result``, and what ``tools/builtin._redact_tool_text``
        reads for the live surfaces that run before a result exists (the bash
        stream, a background job's peek buffer, the abort receipt). Widening it
        here is therefore what covers every agent, subagent, fork, exec and
        headless run at once — a per-surface scrubber is one a new surface can
        forget, which is exactly how a credential printed by ``kubectl exec …
        env`` reached a transcript in full.

        Two passes, in this order:

        1. values the session KNOWS — session credentials and registered
           redactions (:meth:`redaction_values`), replaced byte-for-byte;
        2. credential SHAPES (:mod:`local_operator.redaction_shapes`) — a
           credential recognised by how it is SPELLED, which is the only thing
           that can catch a secret this session was never told. A remote host's
           environment is precisely that set.

        Values the shape pass MATCHES are registered back into (1) —
        :meth:`_register_shape_hits` — so the same secret is contained for the
        rest of the session even in a later form the table does not know.

        WHAT THIS STILL DOES NOT GUARANTEE: an opaque value with none of the
        table's spellings around it (a bare tenant id, a pasted fragment, a
        secret whose name the table does not recognise) passes through. That
        residual is stated in the shapes module rather than implied away here.
        """
        scrubbed, _ = self.redact_with_hits(text)
        return scrubbed

    def redact_with_report(self, text: str) -> tuple[str, ShapeReport]:
        """:meth:`redact`, plus the CLASSIFICATION of everything the shape pass found.

        The richer sibling of :meth:`redact_with_hits`, and the reason it exists:
        the session's incident path has to say whether the value reached the
        model's context or was contained before it, and a bare list of labels
        cannot express that — it names the shapes whose mask was whole, which is
        precisely the CONTAINED set, while the compromise is the hit that left
        readable material behind. Two views of one pass, so they cannot disagree:
        the notice's wording comes from ``labels``, its severity from
        ``reached_model``.

        Labels only, never values: the caller of this is the session's incident
        path, which puts what happened in the transcript, and a notice that
        carried the credential would be the leak it exists to report.

        Values are registered for containment here rather than by the caller, so
        every path that masks ALSO CONTAINS, in-tree: this one, the live-text path
        that never sees this return value, and the bash pipe filter, which masks
        before a result exists and therefore registers through
        :meth:`register_shape_hits_for_containment` instead (it is handed text that
        is already masked, so it could never match the credential here). A store
        outside this tree that offers only the labels-only view keeps whatever
        behaviour it has — ``tools/builtin._redact_tool_text`` reads that view as
        the ESCALATED case, and says why in place.
        """
        scrubbed, hits = scrub_secrets_with_hits(text, self.redaction_values())
        if not hits:
            return scrubbed, ShapeReport()
        self._register_shape_hits(hits)
        # Containment takes EVERY hit (above, unconditionally); the REPORT takes
        # only the complete ones as nameable labels, plus the exposure flag that
        # separates "masked whole" from "part of it is in the model's context".
        # ``shape_report`` is the one place that judgement is made, so this view
        # and the escalated one can never disagree about what fired.
        return scrubbed, shape_report(hits)

    def redact_with_hits(self, text: str) -> tuple[str, list[str]]:
        """:meth:`redact`, plus the LABELS of the shapes that were contained whole.

        The narrower view, kept for the surfaces that only need to name what was
        masked — and for a store whose caller is older than
        :meth:`redact_with_report`. A caller that has to classify the hit reads
        that one instead: this list is exactly the contained set, so treating it
        as "what was found" would silently drop the compromise case.
        """
        scrubbed, report = self.redact_with_report(text)
        return scrubbed, list(report.labels)

    #: How many DETECTED components one session may register. A bound, not a
    #: budget: every registration is a value the exact-value pass scans for in
    #: every later result of the session, so an unbounded set is a per-result cost
    #: that grows with the session's age. On reaching it, detections keep being
    #: counted and noticed — the ticket still fires — and only the CONTAINMENT
    #: stops. Counted in registered values, not matches: a URI contributes two.
    MAX_DETECTED_REGISTRATIONS = 64

    def _register_shape_hits(self, hits: Sequence[ShapeHit]) -> None:
        """Register each matched credential as a value to scrub, for this session.

        Containment rather than tidy bookkeeping: the shape pass catches a
        DSN's password once, and the SAME secret can reappear later in a form
        the table has no rule for (quoted alone, concatenated into another
        command's line). Registering it means the exact-value pass — which runs
        first on every later result — catches that too.

        Only §6 registrations are written, the same sink the broker uses, so a
        matched value never becomes injectable into a child's environment or
        readable by ``read_variable``: registering a value for SCRUBBING must
        never make it readable, which is the distinction ``_credentials`` and
        ``_redactions`` draw. The floor is the shapes module's — a matched value
        shorter than that is not registered, because a short value registered
        process-wide rewrites ordinary text (the measured failure on
        ``mcp.redaction``'s three-character value).
        """
        for hit in hits:
            # The FLOOR and the placeholder rule govern REGISTRATION only; the hit
            # itself is recorded for every mask (see ``_run_shapes``), so a short
            # credential still produces its notice.
            if not is_registerable_component(hit.value):
                continue
            if hit.value in self._shape_registrations:
                continue
            if len(self._shape_registrations) >= self.MAX_DETECTED_REGISTRATIONS:
                if not self._shape_registration_cap_logged:
                    self._shape_registration_cap_logged = True
                    logger.warning(
                        "shape registration cap reached (%d values); detections keep "
                        "being counted and noticed, containment stops here",
                        self.MAX_DETECTED_REGISTRATIONS,
                    )
                continue
            self._shape_registrations.add(hit.value)
            self.register_redaction(hit.value)

    def register_shape_hits_for_containment(self, hits: Sequence[ShapeHit]) -> None:
        """Contain hits a masking layer has ALREADY removed from the bytes.

        The entry point for the one masking surface that does not hand this store
        any text: the bash pipe filter masks a running command's bytes before a
        tool result exists, so the text the result path later gives
        :meth:`redact_with_report` contains this store's own marker instead of the
        credential and that pass matches nothing to register. Measured on the
        shape this feature exists for (a ``kubectl exec … env`` whose credential
        is in the OUTPUT and nowhere in the command): ``registered values: 0``,
        and a later ``cat`` of the bare value — no shape around it, which is the
        form the table cannot know — came back in the clear.

        Same registration as :meth:`redact_with_report`, deliberately: identity
        floor, dedupe and cap all live in :meth:`_register_shape_hits`, so the two
        callers cannot disagree about what containment means. Values only reach
        the §6 sink, never ``read_variable``.
        """
        self._register_shape_hits(hits)

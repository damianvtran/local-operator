"""The credential guard's own source and corpus: exempt from ESCALATION, only.

Operator decision (2026-09-23): stop filing a *rotate it* incident for
credential-SHAPED text that lives in the guard's own shape table and in the
corpus of shapes it is tested against. Reading either file is how an operator,
an agent or a reviewer inspects the guard, and the notice it produced was about a
test fixture — which is exactly the noise that teaches an operator to skip the
notice that is real.

Four limits are the whole of this module, and each one is load-bearing:

* **Masking is NOT exempt.** :func:`reads_exempt_source` is consulted only where
  a shape hit is turned into a notice. The mask, the containment registration and
  the ``reached_model`` classification are computed exactly as they were, so a
  credential-shaped value in these files is still masked in the text the model
  sees. One thing stops: the rotation demand.
* **Only these files — never a tree, never a class.** The exemption is an exact
  list matched by RESOLVED path. There is deliberately no "harness-authored
  surfaces" rule, no ``tests/`` prefix rule and no directory rule: an agent that
  greps all of ``tests/`` is not exempt, and neither is a file that merely talks
  about the guard.
* **Only a tool that reads the file confers it, never text that names it.** The
  decision is taken from the *structured* ``path`` argument of a file-reading
  tool (:data:`READING_TOOLS`), so a ``bash`` command whose arguments mention the
  path (``cat local_operator/redaction_shapes.py``) is not exempt, and neither is
  any tool OUTPUT that merely prints the path inside credential-shaped text.
  That distinction is this module's security property: a matcher that scanned the
  argument *summary* string instead — or the result text — would hand any caller
  the exemption for echoing a filename, which is why it does not, and the arms in
  ``tests/unit/secrets/test_guard_area_exemption.py`` drive both directions.
* **The resolution is the READER's own, not a parallel one.** A relative
  ``path`` argument is handed to ``local_operator.tools.builtin._resolve_workspace_path``
  — the exact function ``read``/``grep`` resolve their arguments with — against
  the SESSION's cwd (``ToolContext.cwd``). This is load-bearing, not tidiness:
  the first cut of this module resolved a relative spelling against the process
  CWD *and* the repo root and accepted either, which handed the exemption to a
  file the reader never opened whenever those two roots disagreed — the default
  shape of an installed ``lop`` run outside the repo. Agreeing with the reader by
  CONSTRUCTION (one resolver, one root) is what removes that class rather than
  patching one instance of it.

A resolution that misses — a nonexistent path, an unresolvable ``~user``, an
embedded NUL, a lone surrogate — falls back to today's behaviour, a loud
escalation, because the exemption is a courtesy to the operator and escalation is
the direction that fails safe. The reader tolerates those same inputs so the turn
survives; this module must not be the one place that raises out of a result,
which is why the resolve is wrapped as well as delegated.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from local_operator.tools.builtin import _resolve_workspace_path

#: ``local_operator/`` in the tree this module was imported from — derived from
#: this file rather than from the CWD, so the answer does not depend on where the
#: process happens to be standing.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

#: Repo root for a source checkout / editable install, which is the only shape in
#: which the corpus exists at all. In an installed wheel this path is inside
#: site-packages and holds nothing; the entry is then inert rather than wrong.
_REPO_ROOT = _PACKAGE_ROOT.parent

#: The guard's own source and its corpus, as repo-relative paths — the two the
#: operator named — plus that corpus's OWN TEST MODULE, which is included because
#: it was MEASURED to escalate rather than because it looked similar. Reading all
#: three through the real shape pass on 2026-09-23 (hits / escalating hits):
#: ``redaction_shapes.py`` 44 / 41, ``credential_shape_corpus.py`` 200 / 162,
#: ``test_credential_shapes.py`` 21 / 18 — each with ``reached_model`` true, so
#: each of the three filed a rotation demand for the guard's own fixtures. A
#: fourth entry needs the same measurement; a speculative one would widen the
#: exemption past what was authorised.
_EXEMPT_RELATIVE_PATHS: tuple[str, ...] = (
    "local_operator/redaction_shapes.py",
    "tests/unit/secrets/credential_shape_corpus.py",
    "tests/unit/secrets/test_credential_shapes.py",
)

#: Resolved once at import: the exemption is compared by REAL path, so a symlink
#: to one of these files is the file, and a path that merely spells it is not.
#:
#: Built by EXISTENCE, not by spelling: a candidate that is not a file cannot be
#: read either, and a phantom entry would be an exemption nobody can observe.
#: ``_REPO_ROOT`` covers the source checkout (``<repo>/local_operator/…``) and the
#: installed wheel (``site-packages/local_operator/…``) with the same relative
#: spelling; the corpus entry resolves only where ``tests/`` sits beside the
#: package, which is the only place the corpus exists.


def _exempt_sources() -> frozenset[Path]:
    found: set[Path] = set()
    for candidate in (_REPO_ROOT, _PACKAGE_ROOT):
        for relative in _EXEMPT_RELATIVE_PATHS:
            path = candidate / relative
            if path.is_file():
                found.add(path.resolve())
    return frozenset(found)


EXEMPT_SOURCES: frozenset[Path] = _exempt_sources()

#: Tools whose structured ``path`` argument IS the file they read. Deliberately
#: two: ``read`` serves the file and ``grep`` reads the lines it matches, and both
#: hand the path to the resolver as an argument. ``bash`` is NOT here and cannot
#: be — its argument is an opaque command string, so exempting a path it mentions
#: would make this decision steerable by writing a filename into a shell command.
READING_TOOLS: frozenset[str] = frozenset({"read", "grep"})


def reads_exempt_source(
    tool_name: str,
    arguments: Mapping[str, Any] | None,
    session_cwd: str | None = None,
) -> bool:
    """Did THIS call ask a reading tool to read one of the exempt files?

    True only for a file-reading tool carrying a ``path`` argument that resolves
    to exactly one of :data:`EXEMPT_SOURCES`. Everything else — an unrelated
    tool, a missing or non-string path, an internal URL scheme, a path that only
    resembles the exempt one, a path that cannot be resolved at all — answers
    False, which is the ESCALATING reading and therefore the pre-existing
    behaviour.

    ``session_cwd`` is the root the READER will use (``ToolContext.cwd``), and
    resolution goes through the reader's own resolver rather than through a
    second rule written to agree with it. When it is omitted the process CWD is
    used, which is what a bare ``Path.resolve`` would do and what the callers
    that have no session (unit calls, introspection) want; the loop always
    threads it. See the module docstring for the bypass that made this
    load-bearing.
    """
    if tool_name not in READING_TOOLS or not arguments:
        return False
    raw = arguments.get("path")
    if not isinstance(raw, str):
        return False
    target = raw.strip()
    # Scheme handles (``spill://``, ``skill://``, ``read https://…``) are served by
    # this module's own resolution, not by a filesystem path, so nothing about
    # them can name an exempt file.
    if not target or "://" in target:
        return False
    # The reader's resolver, so the exemption cannot disagree with the file the
    # reader opens. It already returns ``resolvable=False`` (never raises) for
    # the inputs that make a resolution impossible — an unresolvable ``~user``, an
    # embedded NUL, a lone surrogate — and an unresolvable path is not an exempt
    # one. The wrap is belt-and-braces for a malformed input the resolver has not
    # met yet: this runs for EVERY ``read``/``grep`` result, so raising here is
    # the one outcome that costs the whole turn, and False is the fail-safe side.
    try:
        resolved, _inside, resolvable = _resolve_workspace_path(target, session_cwd or ".")
    except (OSError, ValueError, RuntimeError):
        return False
    return resolvable and resolved in EXEMPT_SOURCES


#: Whether the call whose bytes are being redacted is itself a read of the guard's
#: own area. A ContextVar for the same reason the tool-source carrier is one: it
#: is PER TASK, so two calls in flight cannot confer each other's exemption, and
#: an unset default of False is the escalating answer.
_EXEMPT_SOURCE: ContextVar[bool] = ContextVar("guard_area_exempt_source", default=False)


@contextmanager
def exempt_from_escalation(
    tool_name: str,
    arguments: Mapping[str, Any] | None = None,
    session_cwd: str | None = None,
) -> Iterator[bool]:
    """Publish whether this call reads the guard's area, for its duration.

    ``session_cwd`` is forwarded verbatim: the verdict is decided when the CALL
    is published, and the caller publishing it is the only one that knows the
    root the reader is about to use.
    """
    token = _EXEMPT_SOURCE.set(reads_exempt_source(tool_name, arguments, session_cwd))
    try:
        yield _EXEMPT_SOURCE.get()
    finally:
        _EXEMPT_SOURCE.reset(token)


def source_is_exempt() -> bool:
    """Is the call being reported on a read of the guard's own area?"""
    return _EXEMPT_SOURCE.get()

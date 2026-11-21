"""Catalogue access: ``catalogues/<locale>/<namespace>.json`` message maps.

A catalogue is one flat JSON object of message-id -> ICU message string (§2.2).
This module is the only reader of those files: the loader, the raw bytes behind
``content_sha256``/``ETag``, and the locale/namespace listings the pickers and
the route use.

SECURITY, because a public route takes these two path components: every
component is validated against a conservative grammar and the resolved path is
checked to stay inside the catalogue root. ``..`` and separators are refused at
the component level, so a traversal attempt raises :class:`CatalogueNotFound`
before any filesystem call — the route then answers 404 like any other unknown
name, and a malicious input never gets to describe a path back to the caller.

The hash rule is part of the contract, not an implementation detail: the
``content_sha256`` of a catalogue is the SHA-256 of the COMMITTED FILE BYTES
(not of a re-serialised map), because the translation ledger's
``source_sha256`` (§6) is taken the same way. Two spellings of the same map
would hash differently and staleness would fire; one spelling cannot.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

#: The shipped root. Tests monkeypatch it to point at fixture trees; the route
#: and the runtime always go through the module attribute so that stays true.
_CATALOGUES = Path(__file__).with_name("catalogues")

#: Conservative component grammar for a locale or a namespace: letters, digits,
#: dot, dash, underscore — and it must START with a letter or digit, so `.`,
#: `..` and dot-files are structurally impossible.
_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Suffix distinguishing the en-only context sidecar from a namespace file.
CONTEXT_SUFFIX = ".context.json"

#: The EXACT English context-sidecar vocabulary shared by every writer (core,
#: desktop, mobile) and by the ``read_public_copy`` reader. Operator decision
#: 2026-11-17 (ask a-82fb, "Extend the reader with exact typed fields"): the
#: reader serves these sidecars outside the redaction path, so it must accept a
#: closed, typed schema rather than arbitrary metadata, and writers must fail at
#: authoring time on anything else. ``width``/``length`` are the free-text width
#: hints core and desktop already author; ``placeholders`` maps an ICU argument
#: name to a one-line description (``{}`` is a valid "no arguments" record).
#: Adding a field is a reviewed change here, never a per-repo extension.
CONTEXT_STRING_FIELDS = frozenset({"role", "tone", "note", "width", "length"})
CONTEXT_FIELDS = CONTEXT_STRING_FIELDS | {"maxCells", "placeholders"}


def context_entry_problem(entry: object) -> tuple[str, str | None] | None:
    """Why one sidecar entry breaks the shared schema, or ``None`` when it fits.

    Returns ``(kind, field)`` with ``kind`` one of ``"not-object"``,
    ``"unsupported-field"`` or ``"invalid-type"``. Callers word their own
    message: the authoring check may name the field (committed source), while
    the reader keeps fixed refusals that never echo file content, because an
    invalid file is exactly the one that may carry stray data. Never a value.
    """
    if not isinstance(entry, dict):
        return ("not-object", None)
    for field, value in entry.items():
        if field not in CONTEXT_FIELDS:
            return ("unsupported-field", str(field))
        if field in CONTEXT_STRING_FIELDS:
            valid = isinstance(value, str)
        elif field == "maxCells":
            # bool is an int subclass, but is not a documented width value.
            valid = type(value) is int
        else:  # placeholders: flat str -> str, no nesting and no empty name
            valid = isinstance(value, dict) and all(
                isinstance(name, str) and name and isinstance(text, str)
                for name, text in value.items()
            )
        if not valid:
            return ("invalid-type", field)
    return None


#: Backend-owned namespaces a client is allowed to fetch. RFC §2.1: `wire.*`
#: is core-owned and served; every client-owned namespace (`ui.*`, `tui.*`,
#: `relay.*`, ...) bundles with its own client, so serving it here would create
#: a second source of truth for strings that client already ships.
WIRE_NAMESPACE_PREFIX = "wire."

#: Wire namespaces excluded from the EAGER per-locale bundle and served on
#: demand through the existing per-namespace route instead.
#:
#: THE CRITERION, which every entry here had to satisfy in review: no producer
#: returns the namespace's messages as an ENVELOPE IDENTITY. An envelope is
#: ``{code, params, text}`` (see :func:`local_operator.i18n.messages.envelope`),
#: and it is the ``code`` — the catalogue key itself — that a client resolves,
#: so a namespace whose keys cross the boundary must stay eager: deferring it
#: would leave a released client holding a code it has nothing to resolve
#: (that is why every ``wire.errors.*`` plane stays eager). A namespace whose
#: copy crosses as a FINISHED STRING is safe to defer: the renderer is
#: in-process — it reads the catalogue file directly, never the bundle — and
#: the wire only ever carries the rendered sentence: a printed CLI line, a
#: registry ``label``/``help`` string, an already-rendered bridge row.
#:
#: Deferral is also chosen over RE-SPLITTING these files into child
#: namespaces: a split would move already-audited translated content into new
#: namespaces and invalidate the translation ledger's per-namespace
#: ``source_sha256`` records, while deferral keeps that audit trail intact and
#: removes the same bytes from the eager load.
#:
#: The set is a reviewed policy, not a size heuristic. Two tests pin it:
#: ``tests/unit/i18n/test_catalogue_sizes.py`` asserts the exact key set, and
#: ``tests/unit/i18n/test_deferred_references.py`` refuses a new reference — a
#: new render point — to any namespace listed here. That census, each entry's
#: verified render points and its reason live in
#: ``local_operator/i18n/deferred_wire_inventory.json``; the reason is
#: asserted equal to the one below, so the two cannot drift. A namespace with
#: an envelope producer therefore cannot be parked here quietly: the inventory
#: has no plain-string render point to record for it, and both tests name the
#: two legitimate fixes (make it eager again, or split it).
#:
#: Ordered by family, then roughly by size: the settings registry and the small
#: child namespaces ``settings_io._CHILD_SETTINGS_NAMESPACES`` merges into it,
#: the two CLI surfaces (``exec``, ``remote_compute``), the slash registry, and
#: the bridge rows. A new entry goes at the end of its family — a reviewer reads
#: families, not collation.
DEFERRED_NAMESPACES: dict[str, str] = {
    "wire.settings": (
        "settings registry labels/help, consumed as plain strings by the "
        "settings surface; no desktop envelope identity uses this namespace"
    ),
    "wire.settings.hub_defaults": (
        "hub settings keys rendered in-process by settings_io._t as plain strings"
    ),
    "wire.settings.hub_org": (
        "org Agent Hub settings section rendered in-process by settings_io._t as plain strings"
    ),
    "wire.console": "console section title/description and row labels rendered by settings_io._t",
    "wire.memory": "memory section rows rendered by settings_io._t as plain label/help strings",
    "wire.host_speed": "host-speed threshold label/help rendered by settings_io._t",
    "wire.guardrails": (
        "guardrails section title/description and row label/help rendered by settings_io._t"
    ),
    "wire.radient_saver": (
        "radient-saver section and mode choices rendered by settings_io._t; "
        "saver/chain.py::notice_message returns a Msg its caller prints"
    ),
    "wire.aida_cadence": "AIDA cadence label rendered by settings_io._t as a plain string",
    "wire.notification_sound": (
        "notification-sound settings rows rendered by settings_io._t, lop sound "
        "argparse help and CLI refusals printed in-process, and desktop refusals "
        "sent as finished text under MACHINE codes (too_large, too_long, ...), "
        "never as message ids"
    ),
    "wire.exec": (
        "exec CLI/TUI copy printed by exec_mode.py and exec_session.py "
        '(_t = tr("wire.exec") at module level); the sentence is printed'
    ),
    "wire.remote_compute": (
        "remote-run CLI copy printed by remote_compute/cli.py "
        '(_t = tr("wire.remote_compute") printed through _refuse/print)'
    ),
    "wire.slash": "slash registry descriptions rendered by slash_commands.py::_desc via render()",
    "wire.slash.library": "the /library page descriptions rendered by slash_commands.py::_desc",
    "wire.slash.notifications_page": (
        "the notifications page description rendered by slash_commands.py::_desc"
    ),
    "wire.settings.keymap": (
        "hotkey row labels/help rendered by settings_io._t as plain label/help strings"
    ),
    "wire.language_receipts": (
        "the /language receipts printed by tui/app.py and returned as finished "
        "text by session/runtime/serving.py; no envelope identity is sent"
    ),
    "wire.voice_availability": (
        "provider availability reasons rendered in-process by the tts/stt/audiogen/"
        "video cascades and sent to the desktop as finished text"
    ),
    "wire.harness_import_detail": (
        "import-source detail lines rendered in-process by harness_import.types "
        "and sent to the desktop/terminal as finished text"
    ),
    "wire.wakes_supervisor": (
        "supervisor detail line rendered in-process by wakes/install.py and "
        "sent to the desktop Schedules page as finished text"
    ),
    "wire.session_status": (
        "sidebar/picker row status suffix rendered in-process by "
        "session/catalog.py and printed verbatim by clients"
    ),
    "wire.tool_rows.bridge": (
        "bridge rows reach the phone as already-rendered text "
        '(harness/bridge_copy.py, _t = tr("wire.tool_rows.bridge"))'
    ),
    # DEFERRED 2026-11-12 (session park, PR #3850, i18n finding I26): the
    # audited park copy grew hi's wire.sessions from 3,895 B to 6,695 B and
    # took hi's eager total to 67,987 B, over the 64 KiB per-locale ceiling.
    # No client renders this namespace from the bundle: every key is resolved
    # in-process by the sessions tool and leaves as finished text, so the
    # bundle was carrying it for nobody.
    "wire.sessions": (
        "sessions tool receipts/approval prompts rendered in-process by "
        'tools/builtin.py (_sessions_copy = tr("wire.sessions")) and returned '
        "as finished tool text"
    ),
    # Deferred rather than raising the eager total ceiling: the phone-approvals
    # setup copy (mostly the terminal-only `tui.*` report/usage lines) pushed hi
    # past EAGER_TOTAL_MAX_BYTES, yet no client resolves one of its keys. The
    # desktop route overrides the envelope `code` with the machine outcome token
    # (`on`, `cancelled`, `file_only`, ...) and paints the server-rendered `text`.
    "wire.operator_setup": (
        "phone-approvals setup copy rendered in-process by operator/elevate.py, "
        "tui/app.py and session/runtime/serving.py; the desktop setup route sends "
        "server-rendered text under a machine outcome code, never a catalogue key"
    ),
    # Deferred rather than raising the eager total ceiling (PR #4052): the new
    # GitHub repository-sharing copy pushed hi to 81,230 B, past
    # EAGER_TOTAL_MAX_BYTES, yet no client resolves one of its keys. Every
    # producer binds `_github_text = tr("wire.github")` and hands on a FINISHED
    # sentence: CLI lines and MeshRefusal text (network/cli.py), readiness rows
    # (network/readiness.py), borrower refusal text (credentials/messages.py),
    # and the owner's refusal detail / git-helper stderr notice (github.py).
    "wire.github": (
        "GitHub repository-sharing copy rendered in-process through "
        'tr("wire.github") by network/cli.py, network/readiness.py, '
        "network/credentials/messages.py and network/credentials/github.py; "
        "the wire and stderr carry finished text, never a catalogue key"
    ),
    # Deferred rather than raising the eager total ceiling (PR #4266, merge
    # gate run 44521486072): the Amazon Bedrock remedies (4,166 B in hi) took
    # hi's eager total to 74,237 B, past EAGER_TOTAL_MAX_BYTES, yet no client
    # resolves one of its keys. providers/bedrock/errors.py and check.py render
    # every key in-process through ``_text`` (``render_resolved``) and hand on a
    # FINISHED sentence as ProviderError / KeyCheck text.
    "wire.errors.bedrock": (
        "Amazon Bedrock error remedies and connection-check results rendered "
        "in-process by providers/bedrock/errors.py and providers/bedrock/check.py "
        "(_text = render_resolved), and server/routes/auth.py renders refusal "
        "envelopes with machine-category codes; all carry finished text, never "
        "a catalogue key"
    ),
    # Login copy is composed before callbacks: CLI/TUI pickers and desktop
    # operation snapshots receive strings, not keys for a client to resolve.
    # Keep it out of the eager bundle rather than raising the total ceiling.
    "wire.login.bedrock": (
        "Amazon Bedrock login copy rendered in-process through render_resolved "
        "by providers/bedrock/login.py, providers/bedrock/sso_login.py, "
        "providers/auth_cli.py, server/utils/desktop_auth.py and tui/app.py; "
        "callbacks and operation snapshots carry finished text, never a catalogue key"
    ),
    # Deferred rather than raising the eager total ceiling (PR #4356): with
    # these four keys hi's eager total reached 78,943 B against
    # EAGER_TOTAL_MAX_BYTES (77,824 B), and no client resolves one of them.
    # The failover notice is rendered by ``sign_in_required_text`` and handed
    # to every host as a finished NoticeEvent ``text`` (no ``code``); the
    # ``lop login-status`` / ``/login status`` state word is rendered by
    # ``_sign_in_required_state`` into a plain line.
    "wire.providers.reauth": (
        "Google sign-in recovery notice and login-status state words rendered "
        "in-process: providers/failover.py::sign_in_required_text (render_record) "
        "and providers/auth_cli.py::_sign_in_required_state (render_resolved); "
        "the session notice and the login-status line carry finished text, never "
        "a catalogue key"
    ),
}


class CatalogueNotFound(KeyError):
    """No catalogue file exists for the requested (locale, namespace)."""


class CatalogueInvalid(ValueError):
    """A catalogue file exists but is not a flat string->string JSON object."""


def catalogue_root() -> Path:
    """The directory catalogues are read from (monkeypatch point for tests)."""
    return _CATALOGUES


def _component(value: str, kind: str) -> str:
    if not _COMPONENT_RE.match(value):
        raise CatalogueNotFound(f"{kind} {value!r} is not a valid catalogue name")
    return value


def _file(locale: str, namespace: str, suffix: str = ".json") -> Path:
    safe_locale = _component(locale, "locale")
    safe_namespace = _component(namespace, "namespace")
    root = catalogue_root()
    path = (root / safe_locale / f"{safe_namespace}{suffix}").resolve()
    # Belt and braces after the grammar check: a symlinked locale directory
    # must not be able to escape the root.
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise CatalogueNotFound(f"{locale}/{namespace} escapes the catalogue root") from None
    return path


def _path(locale: str, namespace: str) -> Path:
    return _file(locale, namespace)


def locales() -> tuple[str, ...]:
    """Locales that have at least one namespace file, sorted."""
    root = catalogue_root()
    if not root.is_dir():
        return ()
    found = {
        child.name
        for child in root.iterdir()
        if child.is_dir() and any((child / f).name.endswith(".json") for f in child.iterdir())
    }
    return tuple(sorted(found))


def namespaces(locale: str) -> tuple[str, ...]:
    """Namespaces of ``locale``: its namespace files, sidecars excluded, sorted."""
    safe_locale = _component(locale, "locale")
    directory = catalogue_root() / safe_locale
    if not directory.is_dir():
        raise CatalogueNotFound(f"locale {locale!r} is not shipped")
    found = {
        entry.name[: -len(".json")]
        for entry in directory.iterdir()
        if entry.name.endswith(".json") and not entry.name.endswith(CONTEXT_SUFFIX)
    }
    return tuple(sorted(found))


def catalogue_bytes(locale: str, namespace: str) -> bytes:
    """The raw committed bytes of one catalogue file (hash input, ETag source)."""
    path = _path(locale, namespace)
    if not path.is_file():
        raise CatalogueNotFound(f"no catalogue for {locale}/{namespace}")
    return path.read_bytes()


def catalogue_sha256(locale: str, namespace: str) -> str:
    """``sha256`` hex digest of the file bytes — the ``content_sha256`` contract."""
    return hashlib.sha256(catalogue_bytes(locale, namespace)).hexdigest()


def load_catalogue(locale: str, namespace: str) -> dict[str, str]:
    """Parse one catalogue into a message map, validating its shape.

    A catalogue must be valid UTF-8 JSON forming an object whose values are
    all strings; anything else — a truncated or garbled file included — raises
    :class:`CatalogueInvalid` rather than handing a half-typed map to a
    renderer. The checker validates the same shape at build time, so this
    is the runtime twin of that guarantee, not the only line of defence.
    """
    raw = catalogue_bytes(locale, namespace)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        # Decode/parse failures surface as THE catalogue exception the render
        # path handles. A bare ValueError here would escape `messages.render`
        # (and the resolver's probe) and break the never-raise contract a wire
        # client reads through — the runtime twin of the checker's parse gate.
        raise CatalogueInvalid(f"{locale}/{namespace}: file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogueInvalid(f"{locale}/{namespace}: catalogue root must be an object")
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise CatalogueInvalid(
                f"{locale}/{namespace}: message {key!r} must map a string id to a string"
            )
    return data


def wire_namespaces(locale: str) -> tuple[str, ...]:
    """The ``wire.*`` namespaces a locale carries, sorted.

    Deliberately scoped to :data:`WIRE_NAMESPACE_PREFIX`: the per-locale bundle
    serves the backend-owned set only, and the client-owned namespaces that ship
    alongside it in the same package are not this route's to serve.
    """
    return tuple(
        namespace for namespace in namespaces(locale) if namespace.startswith(WIRE_NAMESPACE_PREFIX)
    )


def wire_bundle(locale: str) -> dict[str, Any]:
    """Every ``wire.*`` namespace of one locale, ready for one client fetch.

    The eager part is the flattened message map: keys already repeat their
    namespace, so merging the files cannot collide (the checker guarantees the
    prefixes). Namespaces in :data:`DEFERRED_NAMESPACES` are described in
    ``deferred`` but their messages are omitted; a client fetches one later with
    the per-namespace route when a surface actually needs it.

    ``content_sha256`` is the bundle's cache identity: it covers the locale,
    every wire namespace's committed-file hash, and each namespace's eager or
    deferred membership. The membership term is load-bearing, not decoration:
    a client revalidates with the ETag and re-registers only on a 200 (see the
    desktop catalogue loader), so a namespace moving between the two sets must
    change the validator even though its own bytes did not. Without it, a
    namespace that became newly eager kept its old ETag, the client got a 304,
    and those messages stayed English indefinitely. It is a digest of the
    hashes and membership markers, not of a re-serialised map, keeping the same
    raw-bytes rule as the file hashes.
    """
    digest = hashlib.sha256()
    digest.update(locale.encode("utf-8") + b"\n")
    eager: dict[str, str] = {}
    eager_hashes: dict[str, str] = {}
    deferred: list[dict[str, Any]] = []
    for namespace in wire_namespaces(locale):
        sha = catalogue_sha256(locale, namespace)
        deferred_flag = namespace in DEFERRED_NAMESPACES
        digest.update(
            f"{namespace}\t{sha}\t{'deferred' if deferred_flag else 'eager'}\n".encode("utf-8")
        )
        if deferred_flag:
            deferred.append(
                {
                    "namespace": namespace,
                    "content_sha256": sha,
                    "bytes": len(catalogue_bytes(locale, namespace)),
                    "keys": len(load_catalogue(locale, namespace)),
                }
            )
            continue
        eager_hashes[namespace] = sha
        eager.update(load_catalogue(locale, namespace))
    return {
        "locale": locale,
        "messages": eager,
        "namespaces": eager_hashes,
        "deferred": deferred,
        "content_sha256": digest.hexdigest(),
    }


def context(locale: str, namespace: str) -> dict[str, dict[str, Any]]:
    """The en-only context sidecar for a namespace, or ``{}`` when absent.

    The sidecar (role/tone/maxCells/notes per key) is part of the contract for
    width-sensitive TUI strings; absent is a valid state until a slice adds it.
    """
    path = _file(locale, namespace, suffix=CONTEXT_SUFFIX)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CatalogueInvalid(f"{locale}/{namespace} context sidecar must be an object")
    return data

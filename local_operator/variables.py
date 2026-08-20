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

import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

#: Environment variables only surface to the agent when opted in with this
#: prefix. Everything else in the process env stays invisible.
ENV_ALLOW_PREFIX = "LOCAL_OPERATOR_"

#: Secret-shaped names are never listed or read, whatever their source. This
#: targets credential KINDS (secret/token/password/.../api_key), not the bare
#: token "key" which is far too common in legitimate config names. The
#: matching is deliberately loose — over-matching only hides more.
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

    action: Literal["list", "store", "forget", "forget-all", "error"]
    key: str = ""
    message: str = ""


CREDENTIAL_USAGE = (
    "Usage: /credential <KEY>\n"
    "       /credential                # list\n"
    "       /credential --forget <KEY>\n"
    "       /credential --forget-all"
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
    behind. Empty strings are skipped: replacing nothing with a marker would
    insert ``[redacted]`` between every character.
    """
    values = list(secrets.values()) if isinstance(secrets, Mapping) else list(secrets)
    ordered = sorted((value for value in values if value), key=len, reverse=True)
    for value in ordered:
        if value in text:
            text = text.replace(value, "[redacted]")
    return text


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
        # Insertion-ordered so the prompt block and ``/credential`` listing
        # agree. Values live only here: never serialized, never listed, never
        # returned by ``get``/``read``.
        self._credentials: dict[str, str] = {}
        self._credential_meta: dict[str, SessionCredential] = {}

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

    def redact(self, text: str) -> str:
        """Replace every stored credential value in ``text`` with ``[redacted]``."""
        return redact_secret_values(text, self._credentials)

"""CLI handlers for ``local-operator login / logout / login status``.

Stream E lazy-imports these; they print plain text and return exit codes.
Interactive prompts use ``input()`` — the CLI entry points are only reached
from an interactive terminal (exec/headless mode never calls them).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable

from local_operator.providers.oauth.callback_server import (
    LoginCallbacks,
    LoginCancelledError,
    LoginDeferredError,
    LoginError,
)
from local_operator.providers.registry import (
    PROVIDER_REGISTRY,
    ProviderDefinition,
    credential_provider_id,
    env_key_name,
    get_provider_definition,
)

logger = logging.getLogger("local_operator.providers.auth_cli")

#: The word that turns ``login`` / ``/login`` into the status listing. One
#: spelling for the CLI and the TUI; it can never collide with a provider id (no
#: registry row is called ``status``, pinned by a test).
LOGIN_STATUS_WORD = "status"
#: Ceiling for a multi-line credential paste (a GCP service-account key is
#: ~2.3 KiB); matches the loader's own key-file cap in ``oauth/vertex.py``.
MAX_CREDENTIAL_PASTE_CHARS = 65_536

if TYPE_CHECKING:  # lazy at runtime: the CLI top level must not import these
    from pathlib import Path

    from local_operator.providers.auth_store import AuthStore


def _callbacks_interactive(definition: ProviderDefinition) -> LoginCallbacks:
    """print-based callbacks for terminal logins.

    The paste prompt is attached for providers that ACCEPT one, which is two
    distinct cases and used to be read as one:

    - ``requires_paste_prompt`` — pasting is the whole login (every
      "paste your API key" provider). Gating these on ``paste_code_flow``, as
      this did, attached no prompt and made the login fail every time with
      "requires an interactive code prompt" — for eight of the eleven
      providers that offer one.
    - ``paste_code_flow`` — Anthropic's optional fallback, raced against the
      loopback callback for the case where the browser is on another machine.

    A loopback-only provider still gets NO prompt: there the prompt races the
    HTTP callback and leaves the terminal blocked on a line nobody will type.

    The prompt runs in a thread via ``asyncio.to_thread(input, ...)`` so the
    callback server keeps serving the browser redirect while it is pending.
    """

    def on_auth_url(url: str, instructions: str | None = None) -> None:
        # THE VERB MATCHES THE FLOW (review round 1, Q3). An OAuth URL is one
        # you authorize; a paste-key row's URL is a dashboard page where the key
        # is CREATED, and telling the user to "authorize" it sent them looking
        # for a sign-in button on a page that only has "Create new secret key".
        # Same reason the prompt below says "API key" and not "code".
        header = (
            "Open this page to create a key:"
            if definition.paste_prompt_required
            else "Open this URL to authorize:"
        )
        print(f"\n{header}\n  {url}")
        if instructions:
            print(instructions)

    def on_progress(message: str) -> None:
        print(message)

    # The prompt says what it wants. "Paste the code here" is wrong for the
    # providers this now serves — they want an API key off a dashboard, and a
    # user told to paste a "code" goes looking for an OAuth code that does not
    # exist for them.
    #
    # A required paste that is NOT an API key (google-vertex-sa: a service
    # account JSON key or a path to one) gets its own wording and a multi-line
    # reader -- telling it to paste an "API key" was wrong, and a single
    # ``readline`` truncated pretty-printed JSON to its opening ``{``.
    wants_api_key = definition.paste_prompt_required and definition.paste_is_api_key
    wants_credential = definition.paste_prompt_required and not definition.paste_is_api_key
    if wants_api_key:
        prompt = "Paste your API key here (empty to cancel): "
    elif wants_credential:
        # "input hidden": nothing echoes while a key is pasted, so without it
        # a correct paste looks like a dead terminal (agent review R2-2).
        prompt = "Paste credential or file path (input hidden; empty to cancel): "
    else:
        prompt = "Paste the code here (empty to cancel): "
    # A prompt opened AFTER the browser/paste race is decided is a follow-up
    # question (Vertex asks for a project ID); ``on_setup_input`` supplies its
    # explicit field label, rather than reusing "Paste the code here".
    # Reset by ``on_input_rejected``: a rejected paste is
    # re-offered as the same fallback.
    race: dict[str, Any] = {"over": False}
    follow_up_prompt = "Enter value (empty to cancel): "
    follow_up_secret = False

    def read_credential() -> str:
        """Read a possibly multi-line credential (JSON object) or a file path.

        Piped stdin is read whole (bounded) so ``cat sa.json | lop login
        google-vertex-sa`` works. On a terminal, echo is turned off -- the JSON
        carries a private key that must not land in scrollback -- and lines are
        gathered until the pasted object is complete, a blank line, EOF or the
        cap.

        On a POSIX terminal the read is NON-CANONICAL (``_read_tty_credential``)
        because canonical mode cannot carry the value at all: the line
        discipline caps one line at ``MAX_CANON`` (1024 bytes on macOS), and a
        real key's ``private_key`` line is ~1.7 KiB, so the kernel dropped the
        tail and the Enter with it and the login hung until Ctrl-C (QA Q4).
        The ``readline`` loop below stays only as the fallback for a terminal
        without ``termios`` (Windows) or a stdin with no real descriptor.
        """
        if not sys.stdin.isatty():
            print(prompt, end="", file=sys.stderr, flush=True)
            return sys.stdin.read(MAX_CREDENTIAL_PASTE_CHARS)
        print(prompt, end="", flush=True)
        fd = _posix_tty_descriptor(sys.stdin)
        if fd is not None:
            value = _read_tty_credential(fd)
            print()  # echo was off, so the user's Enter left no newline
            return value
        with _echo_disabled():
            first = sys.stdin.readline()
            if not first:
                raise EOFError
            buffer = first
            if first.strip().startswith("{"):
                while len(buffer) <= MAX_CREDENTIAL_PASTE_CHARS:
                    try:
                        json.loads(buffer)
                        break
                    except ValueError:
                        pass
                    line = sys.stdin.readline()
                    if not line or not line.strip():
                        break
                    buffer += line
        print()  # echo was off, so the user's Enter left no newline
        return buffer

    def read_line() -> str:
        """Read the pasted value, hiding it when it is a long-lived secret.

        ``getpass`` for an API KEY, which is the discipline the key prompt
        and the web-search CLI already apply to this same class of value: a
        provider key does not expire, and ``input()`` leaves it sitting in the
        scrollback of a terminal that is frequently being screen-shared while
        someone sets a tool up. Before this change no paste-a-key provider could
        reach this prompt at all, so making them work is also what makes the
        echo reachable. The TUI half masks it for the same reason.

        A plain ``input()`` for an OAuth CODE, the other branch: it is
        single-use, expires in minutes, and is spent the moment it is redeemed,
        while being a long opaque string the user genuinely needs to SEE to
        check their paste landed whole. Hiding it would cost real legibility to
        protect a value that is not worth protecting.
        """
        if race["over"]:
            if follow_up_secret and sys.stdin.isatty():
                import getpass

                return getpass.getpass(follow_up_prompt)
            return input(follow_up_prompt)
        if wants_credential:
            return read_credential()
        if wants_api_key:
            if not sys.stdin.isatty():
                # A PIPED key (`echo $KEY | lop login deepseek`, a provisioning
                # script) has no terminal to hide it on. ``getpass`` then falls
                # back with a ``GetPassWarning`` plus "Password input may be
                # echoed" — noise that reads as a failure in exactly the
                # unattended runs that cannot ask what it meant (audit Q10).
                # Nothing is echoed either way: the value comes from the pipe,
                # not from a keyboard. The prompt goes to stderr so stdout
                # stays the command's own output.
                print(prompt, end="", file=sys.stderr, flush=True)
                return sys.stdin.readline().rstrip("\r\n")
            # Imported here rather than at module scope: this module is on the
            # CLI's startup path and getpass drags in termios/tty for a prompt
            # only an interactive login reaches.
            import getpass

            return getpass.getpass(prompt)
        return input(prompt)

    async def on_manual_code_input() -> str | None:
        """Read a pasted value without making the process unkillable.

        A DAEMON thread we own, deliberately NOT ``asyncio.to_thread``. The
        difference only shows up on Ctrl+C, and there it is the whole
        behaviour: ``to_thread`` runs on the loop's default
        ``ThreadPoolExecutor``, and ``asyncio.run`` shuts that executor down by
        JOINING its workers with ``THREAD_JOIN_TIMEOUT``, which is 300 s. The
        worker is blocked in ``input()``/``getpass()`` on a terminal read that
        nothing interrupts, so it never finishes and the join never returns.

        Measured on the pre-change tree, driving the real CLI under a pty and
        sending a real Ctrl+C: the interrupt is delivered and ``run_login``'s
        ``except KeyboardInterrupt`` is reached, and then the process sits in
        ``Runner.close()`` for the full five minutes. From the user's side the
        cancel simply does not work — which is the same 300 s wait this change
        exists to remove, arrived at from the other direction.

        A daemon thread is not joined at interpreter exit, so the blocked read
        stops being able to hold the process open. The value is handed back
        through a loop-thread callback, and the future is checked for
        cancellation first: by the time a late paste arrives the flow it was
        feeding may already be gone.

        THE WORKER OUTLIVES THIS CALL ON EVERY PATH, not just the cancel one.
        A successful Anthropic login finishes when the loopback callback wins
        the race, and this reader is still parked in `input()`/`getpass()` with
        a `read()` outstanding on the terminal until the process exits. That is
        why the terminal's own state is snapshotted and restored by the
        COMMAND (`_terminal_state_restored`) rather than by the reader, and why
        anything the CLI does after a successful login would be competing for
        stdin with this thread. Nothing does today; a future caller that wants
        to read from the terminal after a login has to reckon with it (agent
        review round 1, minor-3).
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        reader["future"] = future

        def deliver(setter: Callable[[asyncio.Future[str]], None]) -> None:
            # The future is resolved from the LOOP thread; a cancelled future
            # (the login ended while the user was still typing) is left alone
            # rather than raising InvalidStateError into a dead flow. The
            # CURRENT prompt's future is read at delivery time, so a reader
            # adopted by a follow-up prompt (below) answers that prompt.
            current = reader["future"]
            if current is not None and not current.done():
                setter(current)

        def worker() -> None:
            try:
                value = read_line()
            except Exception as raised:  # noqa: BLE001 — reported, not swallowed
                # Bound to a local BEFORE the lambda closes over it: Python
                # deletes an `except ... as` name when the clause exits, so a
                # lambda that referenced it directly would raise NameError on
                # the loop thread instead of delivering the error.
                error = raised
                reader["thread"] = None
                loop.call_soon_threadsafe(deliver, lambda f: f.set_exception(error))
            except BaseException:
                # `SystemExit`/`KeyboardInterrupt` raised INSIDE this reader
                # belong to interpreter teardown on this thread, not to the
                # awaiting flow. Delivering one into the future would re-raise
                # it on the loop thread, in a context its author never meant —
                # a `SystemExit` from a reader would surface as the login's
                # outcome. The flow has its own answer for a cancel (the abort
                # signal and the `_login_or_cancel` race), so leaving the
                # future unresolved lets that answer win (agent review round 1,
                # nit-1). EOF still arrives as the `EOFError` handled below.
                logger.debug("login paste reader exited during teardown", exc_info=True)
                reader["thread"] = None
            else:
                read = value
                reader["thread"] = None
                loop.call_soon_threadsafe(deliver, lambda f: f.set_result(read))

        alive = reader["thread"]
        if alive is not None and alive.is_alive():
            # The race's fallback reader is still parked in ``input()`` (the
            # browser callback won). A second thread would compete with it for
            # the same terminal line, and whichever lost would swallow the
            # user's answer into a dead future -- so the parked reader is
            # ADOPTED: it delivers to this prompt's future instead.
            print(follow_up_prompt, end="", flush=True)
        else:
            thread = threading.Thread(target=worker, daemon=True, name="lo-login-paste")
            reader["thread"] = thread
            thread.start()
        try:
            value = await future
        except (EOFError, KeyboardInterrupt):
            return None
        finally:
            race["over"] = True
        return value.strip() or None

    async def on_setup_input(field: str, default: str, secret: bool) -> str | None:
        nonlocal follow_up_prompt, follow_up_secret
        # Adopt the existing daemon reader if the optional OAuth paste lost
        # the browser race. A second stdin reader could swallow this answer.
        # Vertex requests only a nonsecret project, so adopting the unmasked
        # OAuth-code reader does not weaken the field's secrecy contract.
        from local_operator.i18n.keys import providers_vertex_prompts as prompt_keys
        from local_operator.i18n.messages import render_resolved

        follow_up_prompt = render_resolved(prompt_keys.prompt_field(field=field))
        follow_up_secret = secret
        race["over"] = True
        return await on_manual_code_input()

    def on_input_rejected() -> None:
        race["over"] = False

    reader: dict[str, Any] = {"thread": None, "future": None}
    if definition.login_kind == "aws":
        return _aws_callbacks(on_auth_url, on_progress)
    manual_input = on_manual_code_input if definition.accepts_paste_prompt else None
    return LoginCallbacks(
        on_auth_url=on_auth_url,
        on_progress=on_progress,
        on_input_rejected=on_input_rejected,
        on_manual_code_input=manual_input,
        # Keep local-server setup's existing host support unchanged. This hook
        # is for required post-auth fields, not a new endpoint setup wizard.
        on_setup_input=on_setup_input if not definition.local_setup else None,
        displays_warnings=True,
    )


async def _read_on_daemon(reader: Callable[[], str]) -> str | None:
    """Run one blocking terminal read on a daemon thread; ``None`` on EOF/Ctrl+C.

    The same discipline as ``on_manual_code_input`` above, and for the same
    measured reason: ``asyncio.to_thread`` would park interpreter shutdown on a
    300 s executor join behind an uninterruptible ``input()`` after Ctrl+C.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def deliver(setter: Callable[[], None]) -> None:
        if not future.done():
            setter()

    def worker() -> None:
        try:
            value = reader()
        except Exception as raised:  # noqa: BLE001 — reported to the awaiting flow
            error = raised
            loop.call_soon_threadsafe(deliver, lambda: future.set_exception(error))
        except BaseException:
            logger.debug("login field reader exited during teardown", exc_info=True)
        else:
            read = value
            loop.call_soon_threadsafe(deliver, lambda: future.set_result(read))

    threading.Thread(target=worker, daemon=True, name="lo-login-field").start()
    try:
        return await future
    except (EOFError, KeyboardInterrupt):
        return None


def _aws_callbacks(
    on_auth_url: Callable[..., None], on_progress: Callable[[str], None]
) -> LoginCallbacks:
    """Terminal hooks for Amazon Bedrock's structured AWS login (login_kind "aws").

    ``on_choice`` prints a numbered list and reads a number; ``on_setup_input``
    reads one field, through ``getpass`` when it is a secret (an access key's
    secret or a Bedrock API key never lands in scrollback). Without a terminal
    on stdin neither hook is attached, and ``login_bedrock`` fails at once with
    the headless remedy (env vars) instead of blocking on a pipe — a
    provisioning script must not hang on a picker nobody sees.
    """
    if not sys.stdin.isatty():
        return LoginCallbacks(on_progress=on_progress, on_warning=on_progress)
    from local_operator.i18n.keys import wire_login_bedrock as k
    from local_operator.i18n.messages import render_resolved

    async def on_choice(
        title: str, options: list[tuple[str, str, str]], default: str | None
    ) -> str | None:
        print(f"\n{title}")
        default_index = None
        for index, (value, label, description) in enumerate(options, start=1):
            if value == default:
                default_index = index
            suffix = f" — {description}" if description else ""
            marker = " *" if value == default else ""
            print(f"  {index}. {label}{marker}{suffix}")
        prompt = render_resolved(
            k.choice_default_prompt(count=str(len(options)), default=str(default_index))
            if default_index
            else k.choice_prompt(count=str(len(options)))
        )
        while True:
            answer = await _read_on_daemon(lambda: input(prompt))
            if answer is None:
                return None
            answer = answer.strip()
            if not answer:
                return options[default_index - 1][0] if default_index else None
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return options[int(answer) - 1][0]
            print(render_resolved(k.choice_invalid()))

    async def on_setup_input(label: str, default: str, secret: bool) -> str | None:
        prompt = f"{label}: "
        if secret:
            import getpass

            return await _read_on_daemon(lambda: getpass.getpass(prompt))
        return await _read_on_daemon(lambda: input(prompt))

    return LoginCallbacks(
        on_auth_url=on_auth_url,
        on_progress=on_progress,
        on_warning=on_progress,
        on_setup_input=on_setup_input,
        on_choice=on_choice,
        displays_warnings=True,
    )


async def _login_or_cancel(login: Any, callbacks: LoginCallbacks, aborted: Any) -> Any:
    """Run a login, and give up on it the moment the abort signal fires.

    The signal is passed down as well, because a provider that WATCHES it can
    unwind itself cleanly — the loopback flows stop their callback server in
    their own `finally`, which is what frees the port for a retry, and that is
    strictly better than being cancelled from outside.

    But not every login watches it. The paste-a-key providers
    (`create_api_key_login`) are nothing but an `on_manual_code_input` await:
    they accept `signal` through `**_kwargs` and ignore it, so an abort alone
    would leave `local-operator login alibaba` blocked on a prompt read until
    the user found another way out. Verified under a pty before this race
    existed: Ctrl+C printed nothing and the process had to be SIGKILLed.

    Racing here covers both shapes with one rule, and the ordering keeps the
    better outcome when it is available: a flow that raises
    `LoginCancelledError` for itself wins the race normally, and the losing
    task is cancelled and reaped so no login continues detached from the
    command that started it.
    """
    login_task = asyncio.ensure_future(login(callbacks, signal=aborted))
    abort_task = asyncio.ensure_future(aborted.wait())
    try:
        done, _pending = await asyncio.wait(
            {login_task, abort_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if login_task in done:
            return login_task.result()
        raise LoginCancelledError(aborted.reason or "Login cancelled")
    finally:
        for task in (login_task, abort_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(login_task, abort_task, return_exceptions=True)


@contextlib.contextmanager
def _echo_disabled() -> Iterator[None]:
    """Turn terminal echo off for a hidden multi-line read; no-op off POSIX TTYs."""
    try:
        import termios

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
    except Exception:  # noqa: BLE001 — not a POSIX terminal: read with echo
        yield
        return
    hidden = list(saved)
    hidden[3] = hidden[3] & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, hidden)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, saved)


def _posix_tty_descriptor(stream: Any) -> int | None:
    """The descriptor of ``stream`` when it is a POSIX terminal, else ``None``.

    ``None`` routes the caller to its line-based fallback: Windows (no
    ``termios``) and stand-in streams with no real descriptor.
    """
    try:
        import termios

        fd = stream.fileno()
        termios.tcgetattr(fd)
    except Exception:  # noqa: BLE001 — not a POSIX terminal
        return None
    return fd


#: Bracketed-paste markers a terminal wraps a paste in (when the mode is on);
#: they frame the paste and are never part of the credential.
_PASTE_MARKERS = ("\x1b[200~", "\x1b[201~")


@contextlib.contextmanager
def _noncanonical_no_echo(fd: int) -> Iterator[None]:
    """Byte-at-a-time, unechoed reads on ``fd``; restores the attributes after.

    ICANON off is the point: canonical mode limits one line to ``MAX_CANON``
    and a service-account key does not fit (see ``_read_tty_credential``).
    ECHO off keeps the private key out of scrollback. ISIG stays ON so Ctrl-C
    is still SIGINT to the process -- this reader runs on a daemon thread
    where a ``KeyboardInterrupt`` raised by the reader itself would be
    swallowed, while the signal reaches the main thread's ``run_login`` cancel
    path. ``VMIN=1, VTIME=0``: block for at least one byte, no timer, so a
    slow typist and a fast paste behave alike. TCSANOW, never TCSAFLUSH, so
    nothing already typed ahead is discarded.
    """
    import termios

    saved = termios.tcgetattr(fd)
    raw = list(saved)
    # ICRNL/INLCR/IGNCR off: the reader sees CR and LF as sent. With ICRNL on
    # a CRLF paste (a key file saved on Windows) arrives as two LFs, which is
    # indistinguishable from the blank line that ends an unbalanced paste.
    raw[0] = raw[0] & ~(termios.ICRNL | termios.INLCR | termios.IGNCR)
    raw[3] = (raw[3] & ~(termios.ICANON | termios.ECHO)) | termios.ISIG
    control = list(raw[6])
    control[termios.VMIN] = 1
    control[termios.VTIME] = 0
    raw[6] = control
    termios.tcsetattr(fd, termios.TCSANOW, raw)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, saved)


def _json_object_end(text: str) -> int | None:
    """Index just past the ``}`` that closes the object ``text`` opens, if any.

    Braces inside JSON strings (a ``}`` in a description, escaped quotes) are
    skipped, so completion does not depend on the paste ending with Enter.
    """
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _read_tty_credential(fd: int, limit: int = MAX_CREDENTIAL_PASTE_CHARS) -> str:
    """Read a pasted credential from terminal ``fd`` without ``MAX_CANON``.

    Canonical mode hands the kernel the line editing, and with it a hard
    per-line ceiling (``MAX_CANON``, 1024 bytes on macOS): a real GCP key's
    ~1.7 KiB ``private_key`` line overflowed it, the excess and the Enter were
    dropped, and ``lop login google-vertex-sa`` hung silently (QA Q4). So the
    line editing this needs is done here instead:

    - Enter (``\\r`` or ``\\n``) finishes a SINGLE-LINE value -- a file path or
      a key, anything whose first non-blank character is not ``{``. Enter on
      nothing typed returns ``""``, the prompt's "empty to cancel".
    - A value starting with ``{`` is a JSON object and keeps reading across
      newlines until its braces balance outside strings, or a blank line.
    - Backspace (DEL or BS) removes the last character; Ctrl-D ends input
      (``EOFError`` when nothing was typed); a Ctrl-C byte -- only seen if
      something cleared ISIG -- raises ``KeyboardInterrupt``.
    - Bracketed-paste markers and other escape sequences (arrow keys) are
      dropped: none can be part of a credential.

    Stops at ``limit`` characters (the same cap the key-file loader applies),
    leaving the oversize value for the loader to reject.
    """
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    chars: list[str] = []
    escape = ""
    previous = ""
    with _noncanonical_no_echo(fd):
        while len(chars) < limit:
            try:
                data = os.read(fd, 4096)
            except OSError:  # EIO: the terminal hung up mid-read
                data = b""
            if not data:
                if not chars:
                    raise EOFError
                return "".join(chars)
            for char in decoder.decode(data):
                after_cr = previous == "\r"
                previous = char
                if escape:
                    escape += char
                    # ESC [ <params> <final @..~> is one CSI sequence (paste
                    # markers included); ESC + any other char is an Alt chord.
                    if escape in _PASTE_MARKERS or len(escape) == 2 and char != "[":
                        escape = ""
                    elif len(escape) > 2 and "@" <= char <= "~":
                        escape = ""
                    continue
                if char == "\x1b":
                    escape = char
                elif char == "\x03":
                    raise KeyboardInterrupt
                elif char == "\x04":
                    if not chars:
                        raise EOFError
                    return "".join(chars)
                elif char in "\x7f\x08":
                    if chars:
                        chars.pop()
                elif char == "\n" and after_cr:
                    continue  # CRLF is one newline, not a blank line
                elif char in "\r\n":
                    text = "".join(chars)
                    if not text.strip() or not text.lstrip().startswith("{"):
                        return text
                    if not text.rsplit("\n", 1)[-1].strip():
                        return text  # a blank line ends an unbalanced paste
                    chars.append("\n")
                elif char == "\t" or char >= " ":
                    chars.append(char)
            text = "".join(chars)
            if text.lstrip().startswith("{"):
                end = _json_object_end(text)
                if end is not None:
                    # Anything after the closing brace in this read (the
                    # paste's trailing newline) is framing, not the value.
                    if "\n" not in text[end:]:
                        _consume_trailing_newline(fd)
                    return text[:end]
    return "".join(chars)


def _consume_trailing_newline(fd: int, wait_s: float = 0.1) -> None:
    """Swallow the Enter (and paste end marker) still in flight after a ``}``.

    A paste is delivered in pieces (the pty input queue holds ~1 KiB), so the
    closing brace can arrive in one read and its newline in the next. Left in
    the queue, that newline would answer the NEXT prompt as "empty to cancel"
    or land on the shell as a stray line -- and the kernel flags the restored
    canonical mode PENDIN for it. Only whitespace and escape bytes are taken,
    each within ``wait_s`` of the last: a paste's tail is one burst, and a
    human typing ahead is not waited for.
    """
    import select

    while select.select([fd], [], [], wait_s)[0]:
        try:
            byte = os.read(fd, 1)
        except OSError:
            return
        if byte in (b"\r", b"\n") or not byte:
            return
        if byte not in b" \t\x1b[0123456789~":
            return


@contextlib.contextmanager
def _terminal_state_restored() -> Iterator[None]:
    """Put the terminal back the way it was found, whatever happens inside.

    The paste prompt reads through ``getpass`` for an API key, which disables
    ECHO via ``termios`` and restores it in its own ``finally`` — on the reader
    thread. That thread is deliberately a DAEMON (see
    ``on_manual_code_input``): not joining it is what stops a blocked terminal
    read holding the process open for the 300 s executor join. The cost is that
    when a cancel exits the process while the reader is still parked inside
    ``getpass``, its ``finally`` never runs and the user is returned to a shell
    with ECHO off — typing nothing they can see, with no way to diagnose it
    short of blind-typing ``stty sane``.

    So the restore cannot live on the reader. It belongs to whoever owns the
    terminal for the duration of the login, which is this command. Snapshotting
    before the flow starts and restoring on every exit makes it independent of
    which thread was reading, whether it was interrupted, and whether
    ``getpass`` ever ran at all.

    Reproduced under a real pty with the precondition asserted (ECHO on before,
    off at the prompt): ``alibaba`` left ECHO off after a cancel while
    ``anthropic`` — whose ``read_line`` uses plain ``input()`` — restored it,
    which is what isolates this to the ``getpass`` branch and its 10-11
    ``paste_prompt_required`` providers (agent review round 1, major-1).

    A no-op when stdin is not a tty (piped input, CI) or the platform has no
    ``termios`` (Windows), and it never raises: failing to restore a terminal
    must not turn a completed login into an error.
    """
    saved = None
    # The DESCRIPTOR, captured once, not the stream object: ``sys.stdin`` can
    # be rebound or closed while the login runs, and the restore has to name
    # the same terminal the snapshot came from.
    fd: int | None = None
    try:
        stream = sys.stdin
        if stream is not None and stream.isatty():
            import termios

            descriptor = stream.fileno()
            saved = termios.tcgetattr(descriptor)
            fd = descriptor
    except Exception:  # pragma: no cover - no tty, or a platform without termios
        saved = None
        fd = None
    try:
        yield
    finally:
        if saved is not None and fd is not None:
            try:
                import termios

                # TCSADRAIN, not TCSANOW: let whatever the flow already wrote
                # drain before the mode flips, so the restore cannot cut into
                # the cancel receipt being printed.
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            except Exception:  # pragma: no cover - terminal vanished under us
                logger.debug("could not restore terminal attributes", exc_info=True)


def _cancelled_message(definition: ProviderDefinition) -> str:
    """What the terminal says when a login is cancelled.

    Same sentence the TUI's notice carries, for the same reason: the user
    cancelled because nothing appeared to happen (a browser that landed in the
    provider's portal instead of coming back to the loopback redirect), so
    "cancelled" alone leaves them unsure whether something is still listening
    on the port and how to try again. The listener clause is conditional so it
    stays true — a paste-a-key provider never started a server.
    """
    stopped = "local sign-in listener stopped; " if definition.callback_port is not None else ""
    return f"\nLogin cancelled. {stopped}retry with: local-operator login {definition.id}"


#: Bound on the post-login org-hub gate wait: one memberships GET. Past it the
#: login still succeeds and the next boundary refreshes the gate.
_ORG_HUB_GATE_WAIT_S = 8.0


def run_login(
    provider_id: str | None,
    _config_dir: "Path | None",
    auth_store: "AuthStore",
) -> int:
    """Log in to ``provider_id`` (or list options when ``None``). Exit code 0/1.

    The config ROOT is accepted for call-shape symmetry with
    :func:`list_logins` but plays no part in a login: new credentials land in
    ``auth_store`` only. It used to be a ``CredentialManager``, deleted in PR2b.
    """
    if provider_id is not None and provider_id.strip().lower() == LOGIN_STATUS_WORD:
        # `lop login status` / `/login status`: the same listing as
        # `login-status`. "status" can never be a provider id (the registry has
        # no such row), so claiming the word costs nothing and spares the user
        # an "Unknown provider: status" for the command the docs name. Matched
        # case-insensitively, as the TUI's `/login STATUS` is (it lowercases its
        # argument first), so the two surfaces answer one spelling one way.
        return list_logins(auth_store, _config_dir)
    if provider_id is None:
        # Grouped like the desktop with Radient first (``login_catalog``), every
        # row labelled and described. The old flat registry dump led with
        # ``openai``, put Radient 20th and marked flavour rows with an
        # unexplained ``*`` (audit D3/U9/D13) — the marker is gone because the
        # group heading and description now say what it tried to.
        from local_operator.providers.login_catalog import (
            RECOMMENDED_LOGIN,
            login_groups,
        )

        print("Connect an AI account:")
        for heading, rows in login_groups():
            print(f"\n  {heading}")
            for row in rows:
                tag = " (recommended)" if row.recommended else ""
                print(f"    {row.id:<25} {row.label}{tag}")
                print(f"    {'':<25} {row.description}")
        print(
            f"\nStart with: lop login {RECOMMENDED_LOGIN}"
            "\nUsage: lop login <provider>   (or: lop login status)"
        )
        return 0

    definition = get_provider_definition(provider_id)
    if definition is None:
        print(f"Unknown provider: {provider_id}")
        return 1
    if definition.login is None:
        print(f"Provider '{provider_id}' has no interactive login.")
        return 1

    login = definition.login
    callbacks = _callbacks_interactive(definition)
    from local_operator.providers.login_catalog import remote_login_hint

    hint = remote_login_hint(provider_id)
    if hint:
        # Before the flow starts: a remote user otherwise watches a URL that
        # redirects to a loopback port on a machine they are not sitting at,
        # and finds out only at the 300 s timeout (audit Q8).
        print(hint)

    async def _run() -> str | dict[str, Any]:
        # A Ctrl+C during a PENDING login has to cancel the login, not merely
        # unwind the interpreter. The flow is parked in `asyncio.wait` on the
        # loopback callback, so the signal is turned into an abort the flow is
        # already watching for -- the same AbortSignal the TUI's Ctrl+C rung
        # feeds -- which makes it raise `LoginCancelledError` at once instead
        # of sitting out the 300 s `DEFAULT_TIMEOUT_SECONDS`, and lets
        # `run()`'s `finally` stop the loopback server so the port is free for
        # an immediate retry.
        #
        # `add_signal_handler` and not `signal.signal`, because the default
        # KeyboardInterrupt lands wherever the main thread happens to be and
        # unwinds the flow WITHOUT running its cleanup; scheduling on the loop
        # makes the cancel a normal, orderly outcome of the flow itself.
        from local_operator.harness.types import AbortSignal

        aborted = AbortSignal()
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, lambda: aborted.abort("Login cancelled"))
        except (NotImplementedError, RuntimeError):
            # Windows event loops do not implement add_signal_handler, and it
            # also raises off the main thread. The bare KeyboardInterrupt path
            # below still catches the interrupt there; it just cannot be as
            # tidy about the teardown.
            pass
        try:
            result = await _login_or_cancel(login, callbacks, aborted)
            # Profile verification adds an await after exchange. A completion
            # that arrived after abandonment must not reach persistence.
            if aborted.aborted:
                raise LoginCancelledError(aborted.reason or "Login cancelled")
            return result
        finally:
            # The handler outlives the flow otherwise, and a later Ctrl+C would
            # abort a signal nobody is watching instead of interrupting.
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, RuntimeError):
                pass

    # Wraps the WHOLE flow, not just the cancel branch: the reader thread can
    # still be parked inside `getpass` on a successful loopback login too (see
    # `on_manual_code_input`), so every way out of here owes the user the
    # terminal they arrived with.
    try:
        with _terminal_state_restored():
            result = asyncio.run(_run())
    except LoginDeferredError as exc:
        # A deliberate "I'll finish this myself" (Bedrock's expired AWS SSO
        # session): exit 0, no "Login failed:" lead-in, and the command on its
        # own indented line so it can be copied (#4270 D4/Q7). Nothing stored.
        print(str(exc))
        if exc.command:
            print(f"\n    {exc.command}\n")
        return 0
    except LoginError as exc:
        # A cancel is an OUTCOME, not a failure, so it must not be reported as
        # "Login failed:" — LoginCancelledError is a LoginError subclass and
        # would otherwise be caught by the clause below.
        if isinstance(exc, LoginCancelledError):
            print(_cancelled_message(definition))
            return 130
        print(f"Login failed: {exc}")
        return 1
    except KeyboardInterrupt:
        # The fallback for a host where the loop-level handler could not be
        # installed (see `_run`), and for an interrupt outside the flow's own
        # wait.
        print(_cancelled_message(definition))
        return 130

    storage_provider = credential_provider_id(definition.id)
    if isinstance(result, str):
        # Paste-an-API-key login: store as api_key credential with source login.
        if result:
            auth_store.upsert_credential(
                storage_provider, {"key": result, "source": "login", "type": "api_key"}
            )
            _invalidate_cached_listing(storage_provider)
            _invalidate_cached_usage(storage_provider, auth_store)
            print(f"Stored API key for '{storage_provider}'.")
            # A pasted KEY, even under a provider whose own login is OAuth
            # (``radient-key``): the key's host decides the model's spelling.
            _apply_login_defaults(storage_provider, oauth=False)
        return 0

    # OAuth credentials dict; stamp authorized_at if missing.
    result.setdefault("authorized_at", int(time.time() * 1000))
    try:
        row = (
            auth_store.complete_radient_login(result)
            if storage_provider == "radient"
            else auth_store.upsert_credential(storage_provider, result)
        )
    except LoginError as exc:
        if isinstance(exc, LoginCancelledError):
            print(_cancelled_message(definition))
            return 130
        print(f"Login failed: {exc}")
        return 1
    if storage_provider == "radient":
        # This process exits right after sign-in, which would kill the org Agent
        # Hub gate refresh started by ``complete_radient_login``. Wait a short,
        # bounded time for that memberships read (never the catalogue enrol)
        # so `lop hub org` is available as soon as this command returns.
        wait = getattr(auth_store, "wait_org_hub_gate", None)
        if callable(wait):
            try:
                wait(_ORG_HUB_GATE_WAIT_S)
            except Exception:  # noqa: BLE001 - a login never fails over the org hub
                pass
    if storage_provider == "radient" and row.disabled_cause is not None:
        # Design round 2, D2: "unavailable" is reserved for the unverified
        # retained block; a preserved disablement says exactly that. The CLI is
        # a surface the reader can act from, so it is the one host that names
        # the terminal command in the shared retained block (D5).
        print(
            "Sign-in verified. This Radient sign-in stays disabled; "
            "its saved state was preserved."
        )
        print("Nothing else changed.")
        for line in format_retained_radient_logins(
            auth_store.retained_radient_login_ids(),
            action="To restore a retained record, run: local-operator login radient",
        ):
            print(line)
        # Verification is not activation: do not change provider/model defaults
        # or operator profile while the requested credential remains disabled.
        return 1
    _invalidate_cached_listing(storage_provider)
    _invalidate_cached_usage(storage_provider, auth_store)
    # The ONE identity derivation (``credential_identity``): the login receipt
    # and the status listing must name one account one way. Imported here, not
    # at module scope, for the reason AuthStore is (see the TYPE_CHECKING note
    # above): the CLI's top level must not pull the store module in.
    from local_operator.providers.auth_store import credential_identity

    identity = credential_identity(result) or ""
    suffix = f" ({identity})" if identity else ""
    print(f"Logged in to '{storage_provider}'{suffix}.")
    if storage_provider == "radient":
        # The CLI's reader is in a terminal, so this host names the command;
        # every other host gets the shared block's neutral restore line (D5).
        for line in format_retained_radient_logins(
            auth_store.retained_radient_login_ids(),
            action="To restore a retained record, run: local-operator login radient",
        ):
            print(line)
    if result.get("grant_note"):
        print(f"Note: {result['grant_note']}")
    if storage_provider == "radient":
        # See ``ProviderController.login``: the identity the Radient sign-in
        # proved lands in the instructions file's "About the operator".
        from local_operator.aida.profile import record_radient_login

        record_radient_login(result, config_dir=_config_dir)
    _apply_login_defaults(storage_provider, oauth=True)
    _ = row
    return 0


def _invalidate_cached_listing(provider_id: str) -> None:
    """Drop the provider's cached model listing after a credential change.

    A login is the one moment a user expects the model list to be right, and
    it is the one event the listing cache cannot observe on its own: the
    picker's TTL and the background revalidation bound how OLD a document is,
    not WHOSE credential it was fetched under. A catalogue listed anonymously
    (``unauthenticated``, the registry's static rows) or under a different
    account must not be served to the credential that just arrived. Reported by
    a user who could not see a model released inside the TTL window and,
    reasonably, tried logging in again.

    Imported here rather than at module scope: this module is lazy-imported by
    the CLI top level precisely so that ``lop --help`` does not pay for the
    provider stack, and discovery pulls in httpx.

    Best-effort by construction. A failed invalidation costs a stale list until
    the TTL lapses; an exception here would fail a login that actually
    succeeded, which is far worse. Logged at debug with the traceback, the
    same way the controller's twin does, so a read-only cache dir is
    diagnosable rather than silently ignored.
    """
    try:
        from local_operator.model.discovery import invalidate_listing

        invalidate_listing(provider_id)
    except Exception:  # noqa: BLE001 - never fail a successful login over a cache
        logger.debug("listing invalidation failed for %s", provider_id, exc_info=True)
    try:
        from local_operator.model.configure import invalidate_model_info_cache

        invalidate_model_info_cache()
    except Exception:  # noqa: BLE001 - same rule as the listing drop above
        logger.debug("model-info invalidation failed for %s", provider_id, exc_info=True)


def _invalidate_cached_usage(provider_id: str, auth_store: "AuthStore") -> None:
    """Drop the provider's cached usage row after a shell login.

    The TUI's ``/login`` already does this (``ProviderController.login``): a
    completed login is positive evidence the grant is alive, which contradicts
    any ``sign-in expired`` (``credential_invalid``) verdict the cached row
    still carries. Without this the shell path kept showing the dead-grant note
    for the cache TTL (~5 min) after ``local-operator login <provider>`` had
    fixed it (#618 R11).

    **This call is load-bearing on the OAuth route and belt-and-braces on the
    api_key one**, and the difference is worth stating because the two look
    identical from here. ``_identity_key_for`` dedupes an OAuth re-login onto
    the SAME row (an explicit per-provider constant where the IdP returns no
    account id), so the account fingerprint -- and with it the cache key -- is
    unchanged and NOTHING else in the system observes the event: only this
    drop clears the verdict. An api_key credential deliberately dedupes on
    nothing (``source: login`` returns ``None``), so a re-login INSERTS a
    second row, the fingerprint changes and the key moves. That path is still
    correct without this call, by a different mechanism: the row under the old
    key is orphaned rather than served, and the new key simply misses, so the
    next read is a live fetch. Measured on both routes (#626 Q1): api_key
    ``deepseek:aabba6a1…`` -> ``deepseek:73b5fded…`` with 2 stored rows;
    OAuth kimi key byte-identical across the re-login with 1 row.

    Routed through the controller rather than re-deriving the cache key here:
    the key folds in storage-id aliasing and the account fingerprint, and the
    fingerprint is a projection of the rows ``auth_store`` now holds -- so the
    controller has to be built over THIS store, after the upsert, to compute
    the same key the TUI would. Login callbacks are not needed for that and
    are left ``None``. Imported lazily, for the same reason as the listing
    drop: the CLI top level must not pay for the provider stack on
    ``--help``.

    Best-effort by construction, like its listing twin: a failure costs a
    stale row until the TTL lapses, whereas raising would fail a login that
    actually succeeded.
    """
    try:
        from local_operator.providers.controller import ProviderController

        controller = ProviderController(auth_store, login_callbacks=None)
        try:
            controller.invalidate_cached_usage(provider_id)
        finally:
            controller.close()
    except Exception:  # noqa: BLE001 - never fail a successful login over a cache
        logger.debug("usage-cache invalidation failed for %s", provider_id, exc_info=True)


def _apply_login_defaults(provider_id: str, *, oauth: bool | None = None) -> None:
    """Make a fresh login usable: adopt it as hosting when none is set.

    Before this, ``login <provider>`` stored the credential but never touched
    the config, so the very recovery the missing-key error recommends
    (``local-operator login openai``) looped straight back to "Hosting platform
    is not configured" — the credential existed but nothing pointed the app at
    it. Now, when config hosting is empty, the just-logged-in provider becomes
    the hosting and its default model becomes ``model_name``, and we print what
    was set so the change is not silent.

    When hosting is ALREADY set we touch nothing and print nothing: a user
    logging into a second provider to switch models later has not asked to
    change their default, and silently repointing it would be a surprise. The
    exception is a hosting that is set but names no provider the registry owns,
    which is repaired rather than preserved.

    The POLICY lives in :func:`providers.login_defaults.plan_login_defaults`,
    shared with the TUI's ``/login``; this function only applies the plan and
    prints the receipt. That split is deliberate: the two copies of this rule
    had already drifted into writing different hosting ids and different models
    for the same provider, which is the class of bug the shared planner exists
    to make impossible.

    Imported lazily and guarded: this is a convenience on top of a login that
    already succeeded, so a config write failure (read-only dir) must not turn a
    successful login into a failure.

    ``oauth`` states what the login that RAN produced (a dict of OAuth tokens or
    a pasted key string). Callers pass it because ``provider_id`` here is the
    STORAGE id, and deriving it from that id's definition reads the wrong
    flavour: ``radient-key`` stores under ``radient``, whose own login is the
    browser flow, so a pasted key was planned as an OAuth grant. The suggested
    model's spelling is per route (Kimi names K3 ``k3`` on its OAuth host and
    ``kimi-k3`` on its key host), so the flavour matters. ``None`` keeps the
    planner's own derivation for callers that do not know.
    """
    try:
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir
        from local_operator.providers.login_defaults import (
            apply_login_defaults,
            plan_login_defaults,
        )
        from local_operator.providers.model_access import (
            credentialed_chat_providers_here,
        )

        manager = ConfigManager(config_dir())
        # The accessibility predicate the planner needs to see a STRANDED default
        # (a config hosting whose provider has no credential any more). Read
        # AFTER the credential write, and best effort: an unreadable store means
        # ``None``, which disables only that repair. There are no live sessions
        # on this path — the CLI has none — so nothing is re-homed here.
        try:
            accessible = credentialed_chat_providers_here(
                config_dir=config_dir(), config_values=manager.get_config().values
            )
        except Exception:  # noqa: BLE001 — the repair degrades, the login does not
            accessible = None
        plan = plan_login_defaults(
            provider_id,
            manager.get_config_value("hosting"),
            manager.get_config_value("model_name"),
            oauth=oauth,
            accessible=accessible,
        )
        # The write is shared too (``apply_login_defaults``): a plan may set only
        # the model, which a hosting-gated write here used to drop.
        apply_login_defaults(manager, plan)
        # Printed on the NO-WRITE path too, and that is not symmetry for its own
        # sake: a decision-only provider (TypeSafe's Jev) stores its credential,
        # deliberately leaves the routing alone, and explains itself in
        # ``receipt``. Returning before this print — as this did — made that note
        # unreachable, so the login read as having silently done nothing.
        #
        # The receipt is printed VERBATIM, and that is the fix rather than a
        # simplification (design round 1, D6/D9): this line used to append a full stop
        # and upper-case the first letter, while the TUI rendered the planner's string
        # as written — so the same receipt reached two front ends with two openings and
        # two endings. Capital, punctuation and wording all belong to the planner, which
        # is the one place that knows what the sentence says.
        if plan.receipt:
            print(plan.receipt)
    except Exception as exc:  # noqa: BLE001 — never fail a completed login
        print(f"Note: logged in, but could not set default hosting/model: {exc}")


def run_logout(provider_id: str, auth_store: "AuthStore") -> int:
    """Remove every stored credential (OAuth + pasted keys) for the provider."""
    definition = get_provider_definition(provider_id)
    if definition is None:
        print(f"Unknown provider: {provider_id}")
        return 1
    # Log out of both the alias (e.g. xai-oauth) and its storage id (xai).
    targets = {provider_id, credential_provider_id(provider_id)}
    # A LOGIN LENT FOR RENEWAL to a node is revoked at the provider BEFORE its row
    # goes (mesh-credential-renewal §D D3): deleting the row here cannot reach the
    # copy the node holds. Where the provider has no revoke endpoint the lines say
    # so — never that a revoke happened.
    try:
        from local_operator.network.credentials.renewal import logout_revoke_lent

        for line in logout_revoke_lent(auth_store, sorted(targets), definition):
            print(line)
    except Exception:  # noqa: BLE001 — the logout itself must still run
        pass
    removed = 0
    for target in sorted(targets):
        removed += auth_store.delete_credentials_for_provider(target, disabled_cause="logged-out")
    if removed == 0:
        print(f"No stored credentials for '{provider_id}'.")
        return 1
    # Symmetrical with login: a catalogue fetched under the credential just
    # removed must not decide what the NEXT credential can select. One call
    # per STORAGE id: alias and storage id (``zai-oauth``/``zai``) resolve to
    # the same document set, so iterating both would glob twice.
    for storage_id in sorted({credential_provider_id(t) for t in targets}):
        _invalidate_cached_listing(storage_id)
    print(f"Removed {removed} credential(s) for '{provider_id}'.")
    # One tier below the stored rows remains: the process environment. (The
    # legacy ``credentials.env`` file was a second one; PR2a removed it from
    # the cascade.) Deleting the rows does not
    # clear an export, so the very next turn would authenticate again with no
    # indication the logout was partial. Name the variable, never its value.
    definition_env = definition.env_keys
    env_var: str | None = None
    if isinstance(definition_env, str) and os.environ.get(definition_env):
        env_var = definition_env
    elif callable(definition_env) and definition_env():
        env_var = "the provider's environment"
    if env_var:
        print(
            f"Warning: {provider_id} still authenticates from {env_var}. "
            "Unset it to complete the logout."
        )
    return 0


def stored_login_key_names(config_dir: "Path | None") -> list[str]:
    """ENV KEY names the provider-class STORE rows hold, sorted.

    The name-source reader ``list_logins`` prints under "secret store". Reading
    the plaintext ``credentials.env`` here is GONE (PR2a): the file is no longer
    a credential source, so a name only the file holds is one no reader could
    resolve. An unreadable store answers with no names rather than raising —
    this feeds a status printout, not a resolution.

    A NAMED function rather than an inline loop so the reader log, and the
    plaintext sweep that drives it
    (``tests/unit/secrets/test_no_reader_resolves_the_plaintext_file.py``), share
    one callable: a removed leg reached by no callable is removal nothing
    exercises.
    """
    try:
        from local_operator.providers.registry import stored_provider_env_keys

        return sorted(stored_provider_env_keys(config_dir))
    except Exception:  # noqa: BLE001 — a status listing must not raise
        return []


def format_retained_radient_logins(
    row_ids: tuple[int, ...], *, action: str | None = None
) -> list[str]:
    """One recovery explanation for CLI/status and controller login receipts.

    THE RESTORE LINE IS SCOPED TO THE RETAINED RECORD and host-neutral unless a
    host supplies its own gesture (design round 2, D3 and D5). The same block is
    published to the desktop, where neither a slash command nor a shell command
    is an instruction that reader can follow; only the CLI, whose reader is
    already in a terminal, passes its own command.

    Row ids print as stored row numbers rather than bare brackets (D4): a usable
    row [2] above retained row [1] otherwise reads as an inverted list, even
    though usable-first is the correct precedence.
    """
    if not row_ids:
        return []
    restore = action or "To restore a retained record, sign in again to that account."
    return [
        "Retained Radient sign-ins (unavailable):",
        *(
            f"  Account could not be verified; record retained (stored row {row_id})."
            for row_id in row_ids
        ),
        f"  {restore}",
        "  Older records have not been deleted or assigned to that account.",
    ]


def _sign_in_required_state(reason: str | None) -> str:
    """The ``login-status`` state word for a dead grant, in the user's language."""
    from local_operator.i18n.keys import wire_providers_reauth as k
    from local_operator.i18n.messages import render_resolved
    from local_operator.providers.oauth.callback_server import (
        REAUTH_REASON_SESSION_POLICY,
    )

    if reason == REAUTH_REASON_SESSION_POLICY:
        return render_resolved(k.status_session_policy())
    return render_resolved(k.status_sign_in_required())


def format_logins(
    auth_store: "AuthStore",
    config_dir: "Path | None" = None,
    *,
    action: str | None = None,
) -> list[str]:
    """Active credentials and separate retained-state notices, plus key names.

    Returned rather than printed so the CLI (``list_logins``, stdout) and the
    TUI's ``/login status`` (a transcript block) render ONE derivation: a
    printer that wrote straight to stdout cannot be reused inside a running
    full-screen app, and a second copy of the listing is how two surfaces come
    to disagree about who is signed in. Secrets never appear: identities and
    env-var NAMES only.

    ``action`` is the HOST's own recovery gesture, passed straight to the
    retained block (design round 2, D5). ``list_logins`` supplies the terminal
    command because its reader is in a shell; the TUI and the desktop pass
    nothing and get the neutral restore sentence. Defaulting to neutral here is
    deliberate: a slash command printed to a desktop user is not an instruction
    they can follow, and the shared formatter cannot know which host called it.
    """
    from local_operator.providers.auth_store import credential_identity

    lines: list[str] = []
    rows = auth_store.list_credentials()
    retained = format_retained_radient_logins(
        auth_store.retained_radient_login_ids(), action=action
    )
    if rows:
        from rich.cells import cell_len

        lines.append("Stored credentials:")
        now_ms = int(time.time() * 1000)
        listed: list[tuple[Any, str, str]] = []
        for row in rows:
            # The ONE label derivation (``credential_identity`` -> row.data
            # email/account_id/org_name): a row whose payload carries no
            # identity prints the same "-" every other unlabelled row does,
            # rather than a dedupe constant like ``oauth:kimi``.
            identity = credential_identity(row) or "-"
            if row.credential_type == "oauth":
                expires = row.data.get("expires")
                state = "expired" if expires is not None and int(expires) < now_ms else "active"
                # "expired" is the ACCESS token's clock, which every OAuth row
                # passes hourly and the next refresh fixes. A grant the IdP
                # refused for good (tombstoned by the refresh path) is a
                # different state that only a sign-in fixes, and Google Cloud
                # session control is named because it explains why a working
                # Vertex sign-in died with nothing revoked.
                if auth_store.grant_is_dead(row.id):
                    state = _sign_in_required_state(auth_store.grant_dead_reason(row.id))
                detail = f"oauth, {state}"
            else:
                source = row.data.get("source") or "stored"
                detail = f"api_key ({source})"
            listed.append((row, detail, identity))
        # Padded in terminal CELLS to the widest state actually listed (never
        # narrower than the historic 22): the sign-in states are longer than
        # 22, and CJK states are two cells per code point, so a code-point pad
        # (`f"{detail:<22}"`) left `identity=` ragged on exactly the rows a user
        # opens this command to read (design round 1, D1).
        width = max([22, *(cell_len(detail) for _row, detail, _identity in listed)])
        for row, detail, identity in listed:
            pad = " " * (width - cell_len(detail))
            lines.append(f"  [{row.id}] {row.provider:<14} {detail}{pad} identity={identity}")
    else:
        lines.append("No usable stored credentials." if retained else "No stored credentials.")
    if retained:
        lines.extend(["", *retained])

    lines.append("")
    lines.append("Environment keys visible to the cascade:")

    found = False
    for definition in PROVIDER_REGISTRY:
        name = env_key_name(definition.id)
        if name and os.environ.get(name):
            lines.append(f"  {definition.id:<14} {name}=<set>")
            found = True
    if config_dir is not None:
        # The provider-class store rows, keyed by env-key name, reported as names
        # only. The plaintext ``credentials.env`` loop this used to print is GONE
        # (PR2a): the file is no longer a credential source, and
        # ``stored_login_key_names`` is the one callable that reads those names.
        from local_operator.providers.bedrock.connection import REF_PATTERN

        for key in stored_login_key_names(config_dir):
            # A Bedrock sealed ref (BEDROCK_SECRET_<hex>) is an internal handle
            # behind a connection row already listed above, not a key the user
            # set; printing it only invites someone to export or delete it
            # (#4270 QA Q6).
            if REF_PATTERN.fullmatch(key):
                continue
            lines.append(f"  secret store  {key}=<set>")
            found = True
    if not found:
        lines.append("  (none)")
    return lines


def list_logins(auth_store: "AuthStore", config_dir: "Path | None" = None) -> int:
    """Print the status listing (:func:`format_logins`), one line per row.

    This host's reader is at a terminal, so it names the command that restores a
    retained record; the shared formatter stays neutral for the TUI and desktop
    (design round 2, D5). Dropping it here is what made ``login-status`` lose
    its only actionable line.
    """
    for line in format_logins(
        auth_store,
        config_dir,
        action="To restore a retained record, run: local-operator login radient",
    ):
        print(line)
    return 0
